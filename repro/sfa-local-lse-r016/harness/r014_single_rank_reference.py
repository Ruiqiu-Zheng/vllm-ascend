#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/cache/triton-single")

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
import vllm_ascend.ops  # noqa: E402,F401
from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402
from vllm_ascend.ops.triton.sfa_cp import fused_sfa_dcp_lse_combine, pack_sfa_dcp_output_lse  # noqa: E402

try:
    from vllm_ascend.ops.triton.sfa_cp import pack_sfa_dcp_output_max_sum  # noqa: E402
except ImportError:
    # The archive-bound base intentionally predates this candidate-only helper.
    pack_sfa_dcp_output_max_sum = None


TASK_ROOT = Path("/workspace")
DCP_SIZE = 2
NUM_TOKENS = 4
NUM_HEADS = 8
HEAD_DIM = 512
ROPE_DIM = 64
SEQ_LEN = 128
SPARSE_COUNT = 2048
SEEDS = {torch.bfloat16: 20260913, torch.float16: 20260914}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(order="C")
    if value.dtype == torch.float16:
        return value.numpy().astype("<f2", copy=False).tobytes(order="C")
    if value.dtype == torch.float32:
        return value.numpy().astype("<f4", copy=False).tobytes(order="C")
    if value.dtype == torch.int32:
        return value.numpy().astype("<i4", copy=False).tobytes(order="C")
    raise TypeError(value.dtype)


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


@dataclass
class ProducerFixture:
    dtype: torch.dtype
    q: torch.Tensor
    q_rope: torch.Tensor
    k: torch.Tensor
    k_rope: torch.Tensor
    indices: torch.Tensor
    block_table: torch.Tensor
    actual_q: torch.Tensor
    actual_k: torch.Tensor


def make_producer_fixture(dtype: torch.dtype) -> ProducerFixture:
    generator = torch.Generator(device="cpu").manual_seed(SEEDS[dtype])
    q = torch.randn((NUM_TOKENS, NUM_HEADS, HEAD_DIM), generator=generator, dtype=torch.float32).to(dtype)
    q_rope = torch.randn((NUM_TOKENS, NUM_HEADS, ROPE_DIM), generator=generator, dtype=torch.float32).to(dtype)
    k = torch.randn((1, SEQ_LEN, 1, HEAD_DIM), generator=generator, dtype=torch.float32).to(dtype)
    k_rope = torch.randn((1, SEQ_LEN, 1, ROPE_DIM), generator=generator, dtype=torch.float32).to(dtype)
    indices = torch.full((NUM_TOKENS, 1, SPARSE_COUNT), -1, dtype=torch.int32)
    indices[:, 0, :SEQ_LEN] = torch.arange(SEQ_LEN, dtype=torch.int32)
    return ProducerFixture(
        dtype=dtype,
        q=q.npu(),
        q_rope=q_rope.npu(),
        k=k.npu(),
        k_rope=k_rope.npu(),
        indices=indices.npu(),
        block_table=torch.tensor([[0]], dtype=torch.int32, device="npu"),
        actual_q=torch.tensor([NUM_TOKENS], dtype=torch.int32, device="npu"),
        actual_k=torch.tensor([SEQ_LEN], dtype=torch.int32, device="npu"),
    )


def run_producer(fixture: ProducerFixture) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    scale = 1.0 / math.sqrt(HEAD_DIM + ROPE_DIM)
    result = DeviceOperator.execute_sparse_flash_attention_process(
        SimpleNamespace(scale=scale),
        fixture.q,
        fixture.q_rope,
        (fixture.k, fixture.k_rope),
        fixture.indices,
        SimpleNamespace(block_table=fixture.block_table),
        fixture.actual_q,
        fixture.actual_k,
        sparse_mode=0,
        return_lse=True,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise AssertionError(f"producer tuple arity/type mismatch: {type(result)!r} {result!r}")
    output, maximum, summation = result
    torch.npu.synchronize()
    assert output.shape == (NUM_TOKENS, NUM_HEADS, HEAD_DIM)
    assert maximum.shape == (1, NUM_TOKENS, NUM_HEADS)
    assert summation.shape == (1, NUM_TOKENS, NUM_HEADS)
    assert output.dtype == fixture.dtype
    assert maximum.dtype == torch.float32 and summation.dtype == torch.float32

    q_cpu = torch.cat((fixture.q.cpu(), fixture.q_rope.cpu()), dim=-1).float()
    k_cpu = torch.cat((fixture.k.cpu(), fixture.k_rope.cpu()), dim=-1).float()[0, :, 0]
    value_cpu = fixture.k.cpu().float()[0, :, 0]
    scores = torch.einsum("thd,sd->ths", q_cpu, k_cpu) * scale
    reference_lse = torch.logsumexp(scores, dim=-1)
    reference_output = torch.einsum("ths,sd->thd", torch.softmax(scores, dim=-1), value_cpu).to(fixture.dtype)
    actual_lse = (maximum.float() + torch.log(summation.float())).permute(1, 0, 2).reshape(NUM_TOKENS, NUM_HEADS)
    tolerance = 2e-2 if fixture.dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(output.cpu(), reference_output, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_lse.cpu(), reference_lse, rtol=1e-4, atol=1e-4)
    return output, maximum, summation, {
        "tuple_arity": 3,
        "output_shape": list(output.shape),
        "max_shape": list(maximum.shape),
        "sum_shape": list(summation.shape),
        "output_dtype": str(output.dtype),
        "max_dtype": str(maximum.dtype),
        "sum_dtype": str(summation.dtype),
        "output_sha256": tensor_digest(output),
        "max_sha256": tensor_digest(maximum),
        "sum_sha256": tensor_digest(summation),
        "output_max_abs_fp32_oracle_error": float((output.cpu().float() - reference_output.float()).abs().max()),
        "lse_max_abs_fp32_oracle_error": float((actual_lse.cpu() - reference_lse).abs().max()),
        "finite_max": bool(torch.isfinite(maximum).all()),
        "finite_positive_sum": bool((torch.isfinite(summation) & (summation > 0)).all()),
    }


def independent_merge(outputs: torch.Tensor, maxima: torch.Tensor, sums: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(maxima) & torch.isfinite(sums) & (sums > 0)
    lses = torch.where(valid, maxima.float() + torch.log(torch.where(valid, sums.float(), 1.0)), -torch.inf)
    max_lse = lses.max(dim=0).values
    any_valid = valid.any(dim=0)
    safe_max = torch.where(any_valid, max_lse, torch.zeros_like(max_lse))
    weights = torch.where(valid, torch.exp(lses - safe_max.unsqueeze(0)), torch.zeros_like(lses))
    denominator = weights.sum(dim=0)
    safe_outputs = torch.where(valid.unsqueeze(-1), outputs.float(), torch.zeros_like(outputs.float()))
    numerator = (safe_outputs * weights.unsqueeze(-1)).sum(dim=0)
    return torch.where(
        any_valid.unsqueeze(-1),
        numerator / torch.where(any_valid, denominator, torch.ones_like(denominator)).unsqueeze(-1),
        torch.zeros_like(numerator),
    ).to(outputs.dtype)


def run_pack_cell(
    dtype: torch.dtype,
    scatter_dim: int,
    output: torch.Tensor,
    maximum: torch.Tensor,
    summation: torch.Tensor,
) -> dict[str, object]:
    lse = (maximum.float() + torch.log(summation.float())).permute(1, 0, 2).reshape(
        NUM_TOKENS, NUM_HEADS, 1
    )
    candidate = pack_sfa_dcp_output_max_sum(output, maximum, summation, DCP_SIZE, scatter_dim)
    materialized = pack_sfa_dcp_output_lse(output, lse, DCP_SIZE, scatter_dim)
    torch.npu.synchronize()
    torch.testing.assert_close(candidate, materialized, rtol=0, atol=0)

    local_size = output.shape[scatter_dim] // DCP_SIZE
    routing = []
    for destination in range(DCP_SIZE):
        if scatter_dim == 0:
            expected = output[destination * local_size : (destination + 1) * local_size]
            packed = candidate[destination, ..., :HEAD_DIM]
        else:
            expected = output[:, destination * local_size : (destination + 1) * local_size].permute(1, 0, 2)
            packed = candidate[destination, ..., :HEAD_DIM]
        assert torch.equal(packed, expected)
        routing.append(
            {
                "destination": destination,
                "expected_sha256": tensor_digest(expected),
                "packed_sha256": tensor_digest(packed),
                "exact": True,
            }
        )

    source_outputs = torch.stack((output, output * 0.5 + 0.25))
    source_max = torch.stack((maximum, maximum + 0.25))
    source_sum = torch.stack((summation, summation * 0.75 + 0.125))
    source_outputs[:, 0] = float("nan")
    source_max[:, :, 0, :] = float("-inf")
    source_sum[:, :, 0, :] = 0.0
    source_outputs[0, 1, 0] = float("nan")
    source_max[0, 0, 1, 0] = float("nan")
    source_outputs[0, 1, 1] = float("nan")
    source_sum[0, 0, 1, 1] = float("nan")
    source_outputs[1, 1, 2] = float("nan")
    source_sum[1, 0, 1, 2] = float("inf")
    source_outputs[0, 2, 3] = float("nan")
    source_sum[0, 0, 2, 3] = 0.0
    source_outputs[1, 2, 4] = float("nan")
    source_sum[1, 0, 2, 4] = -1.0

    sends = [
        pack_sfa_dcp_output_max_sum(source_outputs[source], source_max[source], source_sum[source], 2, scatter_dim)
        for source in range(2)
    ]
    recv = torch.stack([sends[source][0] for source in range(2)])
    actual = fused_sfa_dcp_lse_combine(recv, HEAD_DIM, scatter_dim)
    if scatter_dim == 0:
        expected = independent_merge(
            source_outputs[:, :local_size], source_max[:, 0, :local_size], source_sum[:, 0, :local_size]
        )
    else:
        expected = independent_merge(
            source_outputs[:, :, :local_size], source_max[:, 0, :, :local_size], source_sum[:, 0, :, :local_size]
        )
    torch.npu.synchronize()
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert torch.count_nonzero(actual[0]).item() == 0
    assert torch.isfinite(actual[1:]).all()
    return {
        "dtype": str(dtype),
        "scatter_dim": scatter_dim,
        "status": "PASS",
        "packed_shape": list(candidate.shape),
        "candidate_pack_sha256": tensor_digest(candidate),
        "materialized_lse_pack_sha256": tensor_digest(materialized),
        "packed_exact_match": True,
        "routing": routing,
        "invalid_statistics": {
            "nonfinite_max": "PASS_ZERO_CONTRIBUTION",
            "nonfinite_sum_nan": "PASS_ZERO_CONTRIBUTION",
            "nonfinite_sum_inf": "PASS_ZERO_CONTRIBUTION",
            "zero_sum": "PASS_ZERO_CONTRIBUTION",
            "negative_sum": "PASS_ZERO_CONTRIBUTION",
            "all_invalid": "PASS_EXACT_ZERO",
            "adjacent_non_contamination": True,
        },
        "combine": {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "actual_sha256": tensor_digest(actual),
            "reference_sha256": tensor_digest(expected),
            "max_abs_fp32_oracle_error": float((actual.cpu().float() - expected.cpu().float()).abs().max()),
            "tolerance": tolerance,
        },
    }


def triton_artifacts() -> list[dict[str, object]]:
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    records = []
    if cache.exists():
        for path in sorted(cache.rglob("*")):
            if path.is_file():
                records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return records


def main() -> None:
    torch.npu.set_device(0)
    result: dict[str, object] = {
        "schema": "sfa-r014-single-rank-v1",
        "status": "IN_PROGRESS",
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": 0,
        "seeds": {"bfloat16": 20260913, "float16": 20260914},
        "producer_positive_controls": [],
        "cells": [],
    }
    for dtype in (torch.bfloat16, torch.float16):
        fixture = make_producer_fixture(dtype)
        output, maximum, summation, producer = run_producer(fixture)
        result["producer_positive_controls"].append({"dtype": str(dtype), "status": "PASS", **producer})
        for scatter_dim in (0, 1):
            result["cells"].append(run_pack_cell(dtype, scatter_dim, output, maximum, summation))
    result["triton_artifacts"] = triton_artifacts()
    result["status"] = "PASS_SINGLE_RANK_PRODUCER_AND_FOUR_PACK_CELLS"
    output_path = TASK_ROOT / "output" / "single-rank.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "cells": [
                    {"dtype": cell["dtype"], "scatter_dim": cell["scatter_dim"], "status": cell["status"]}
                    for cell in result["cells"]
                ],
                "result_sha256": sha256(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
