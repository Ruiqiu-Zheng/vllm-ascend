import gc

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

# enable internal format
torch_npu.npu.config.allow_internal_format = True
# enable vllm-ascend custom ops
enable_custom_op()


def gmm_swiglu_quant(
    x: torch.Tensor, weight: torch.Tensor, perChannelScale: torch.Tensor, perTokenScale: torch.Tensor, m: int
):
    """
    Perform quantized GMM (Grouped Matrix Multiplication) operation with SwiGLU activation function.

    Parameters:
        x (torch.Tensor): Input tensor with shape (m, k).
        weight (torch.Tensor): Weight tensor with shape (k, n).
        perChannelScale (torch.Tensor): Per-channel scaling factor with shape (n,).
        perTokenScale (torch.Tensor): Per-token scaling factor with shape (m,).
        m (int): Number of tokens (rows of x).

    Returns:
        quantOutput (torch.Tensor): Quantized output tensor with shape (m, k // 2).
        quantScaleOutput (torch.Tensor): Quantization scaling factor with shape (m,).
    """
    # Perform matrix multiplication with int32 precision
    c_temp1 = torch.matmul(x.to(torch.int32), weight.to(torch.int32))
    c_temp1 = c_temp1.to(torch.float32)  # Convert back to float32 for scaling

    # Apply per-channel and per-token scaling
    c_temp2 = torch.mul(c_temp1, perChannelScale)
    c_temp3 = torch.mul(c_temp2, perTokenScale.reshape(m, 1))

    # Split the result into two parts to apply SwiGLU activation function
    c_temp4, gate = c_temp3.chunk(2, dim=-1)
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU activation
    c_temp6 = c_temp5 * gate  # Element-wise multiplication with gating values

    # Quantize the output
    max = torch.max(torch.abs(c_temp6), -1).values  # Find maximum absolute value to calculate scaling factor
    quantScaleOutput = 127 / max  # Calculate quantization scaling factor
    quantOutput = torch.round(c_temp6 * quantScaleOutput.reshape(m, 1)).to(torch.int8)  # Quantize to int8
    quantScaleOutput = 1 / quantScaleOutput  # Inverse quantization scaling factor for subsequent dequantization

    return quantOutput, quantScaleOutput


def process_groups(
    x: torch.Tensor,
    weight: torch.Tensor,
    perChannelScale: torch.Tensor,
    perTokenScale: torch.Tensor,
    groupList: torch.Tensor,
):
    """
    Process input data by groups and call GMM_Swiglu_quant function for quantized computation.

    Parameters:
        x (torch.Tensor): Input tensor with shape (M, K).
        weight (torch.Tensor): List of weight tensors, each with shape (E, K, N).
        perChannelScale (torch.Tensor): List of per-channel scaling factors, each with shape (E, N).
        perTokenScale (torch.Tensor): Per-token scaling factor with shape (M,).
        groupList (list): List defining the number of tokens in each group.

    Returns:
        quantOutput (torch.Tensor): Quantized output tensor with shape (M, N // 2).
        quantScaleOutput (torch.Tensor): Quantization scaling factor with shape (M,).
    """
    M, N = x.shape[0], weight.shape[2]  # Get the shape of the input tensor
    quantOutput = torch.zeros(M, N // 2).to(torch.int8)  # Initialize quantized output tensor
    quantScaleOutput = torch.zeros(M).to(torch.float32)  # Initialize quantization scaling factor tensor

    start_idx = 0  # Starting index
    preV = 0  # Number of tokens in the previous group
    groupList = groupList.tolist()
    # Iterate through groupList to process data by groups
    for i, v in enumerate(groupList):
        currV = v
        tempV = currV - preV  # Calculate number of tokens in the current group
        preV = currV  # Update number of tokens in the previous group
        if tempV > 0:
            # Call GMM_Swiglu_quant to process the current group
            quantOutput[start_idx : start_idx + tempV], quantScaleOutput[start_idx : start_idx + tempV] = (
                gmm_swiglu_quant(
                    x[start_idx : start_idx + tempV],
                    weight[i],
                    perChannelScale[i],
                    perTokenScale[start_idx : start_idx + tempV],
                    tempV,
                )
            )

        start_idx += tempV  # Update starting index to process the next group
    return quantOutput, quantScaleOutput


@torch.inference_mode()
def test_gmm_swiglu_quant_weight_nz_tensor_list():
    M, K, E, N = 8192, 7168, 4, 4096

    # x (M, K) - int8
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8)

    # weight (E, N, K) - int8
    weight = torch.randint(-128, 127, size=(E, K, N), dtype=torch.int8)

    # weight_scale (E, N) - float32
    weight_scale = torch.rand(E, N) * 0.9 + 0.1  # uniform(0.1, 1.0)
    weight_scale = weight_scale.to(torch.float32)

    weight_nz_npu = []
    weight_scale_npu = []
    for i in range(E):
        weight_nz_npu.append(torch_npu.npu_format_cast(weight[i].npu(), 29))
        weight_scale_npu.append(weight_scale[i].npu())

    # x_scale (M,) - float32
    x_scale = torch.rand(M) * 0.9 + 0.1  # uniform(0.1, 1.0)
    x_scale = x_scale.to(torch.float32)

    group_list = torch.tensor([2048, 4096, 6144, 8192], dtype=torch.int64)

    output_cpu, output_scale_cpu = process_groups(x, weight, weight_scale, x_scale, group_list)
    output_npu, output_scale_npu, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
        x.npu(), weight_nz_npu, weight_scale_npu, x_scale.npu(), group_list.npu()
    )
    output_npu_valid = output_npu[: group_list[-1], :]
    output_scale_npu_valid = output_scale_npu[: group_list[-1]]

    torch.testing.assert_close(output_npu_valid.cpu(), output_cpu, atol=1, rtol=2**-13)
    torch.testing.assert_close(output_scale_npu_valid.cpu(), output_scale_cpu, atol=1e-9, rtol=1e-6)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_gmm_swiglu_quant_weight_nz_tensor_list_k1_split_workspace():
    # Current non-prefill tiling rule:
    #   m_limit = floor(256 MiB / (2 workspaces * sizeof(int32) * N))
    #   tiling key 1 iff M > 2 * m_limit.
    # N=4096 matches the existing tensor-list regression width, while K=128
    # matches this kernel's BASIC_K tile and avoids adding an unrelated
    # minimum-K / maximum-N edge to the synchronization regression.
    m, k, expert_num, n = 16385, 128, 2, 4096
    user_workspace_limit = 256 * 1024 * 1024
    m_limit = (user_workspace_limit // 2 // 4) // n
    assert m_limit == 8192
    assert m == 2 * m_limit + 1

    torch.manual_seed(27)
    x = torch.randint(-32, 32, (m, k), dtype=torch.int8)
    weight = torch.randint(-8, 8, (expert_num, k, n), dtype=torch.int8)
    weight_scale = torch.rand(expert_num, n, dtype=torch.float32) * 0.009 + 0.001
    x_scale = torch.rand(m, dtype=torch.float32) * 0.009 + 0.001
    # Expert 0 is deliberately empty; all rows map to expert 1.
    group_list = torch.tensor([0, m], dtype=torch.int64)

    weight_nz_npu = [torch_npu.npu_format_cast(item.npu(), 29) for item in weight]
    weight_scale_npu = [item.npu() for item in weight_scale]
    x_npu = x.npu()
    x_scale_npu = x_scale.npu()
    group_list_npu = group_list.npu()

    try:
        output_1, output_scale_1, output_offset_1 = (
            torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x_npu, weight_nz_npu, weight_scale_npu, x_scale_npu, group_list_npu
            )
        )
        torch.npu.synchronize()
        output_1_cpu = output_1.cpu()
        output_scale_1_cpu = output_scale_1.cpu()

        output_2, output_scale_2, output_offset_2 = (
            torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x_npu, weight_nz_npu, weight_scale_npu, x_scale_npu, group_list_npu
            )
        )
        torch.npu.synchronize()
        output_2_cpu = output_2.cpu()
        output_scale_2_cpu = output_scale_2.cpu()

        # Full-buffer repeatability catches stale event/scalar state without
        # materializing a full CPU golden matrix.
        assert torch.equal(output_1_cpu, output_2_cpu)
        assert torch.equal(output_scale_1_cpu, output_scale_2_cpu)

        # Check every workspace split boundary and the final one-row tail.
        checked_rows = torch.tensor([0, m_limit - 1, m_limit, 2 * m_limit - 1, 2 * m_limit])
        output_cpu, output_scale_cpu = gmm_swiglu_quant(
            x[checked_rows],
            weight[1],
            weight_scale[1],
            x_scale[checked_rows],
            checked_rows.numel(),
        )
        torch.testing.assert_close(output_1_cpu[checked_rows], output_cpu, atol=1, rtol=2**-13)
        torch.testing.assert_close(output_scale_1_cpu[checked_rows], output_scale_cpu, atol=1e-9, rtol=1e-6)

        assert output_1.shape == (m, n // 2)
        assert output_1.dtype == torch.int8
        assert output_1.is_contiguous()
        assert output_scale_1.shape == (m,)
        assert output_scale_1.dtype == torch.float32
        assert output_scale_1.is_contiguous()
        # output_offset is currently an ABI placeholder allocated by the
        # torch adapter; the ACLNN path warns that its values are unsupported.
        # Validate only shape/dtype, not uninitialized contents.
        assert output_offset_1.shape == (m,)
        assert output_offset_1.dtype == torch.float32
        assert output_offset_2.shape == (m,)
        assert output_offset_2.dtype == torch.float32
    finally:
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
