# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "vllm_ascend" / "attention" / "context_parallel" / "sfa_dcp_indexer.py"
_SFA_CP_PATH = _REPO_ROOT / "vllm_ascend" / "attention" / "context_parallel" / "sfa_cp.py"
_SPEC = importlib.util.spec_from_file_location("sfa_dcp_indexer_host", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DCP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DCP)

K = _DCP.DCP_SHARDED_INDEXER_TOPK
P = _DCP.DCP_SHARDED_INDEXER_WORLD_SIZE
MAX_VISIBLE = _DCP.DCP_SHARDED_INDEXER_MAX_VISIBLE_LENGTH


def _strictly_increasing_bf16_scores(count: int) -> torch.Tensor:
    # Positive BF16 bit patterns below 0x7f80 are finite and strictly
    # increasing. Raw bits avoid accidental FP32-to-BF16 collisions.
    score_bits = torch.arange(1, count + 1, dtype=torch.int32).to(torch.int16)
    scores = score_bits.view(torch.bfloat16)
    assert torch.unique(scores).numel() == count
    assert torch.all(torch.isfinite(scores))
    return scores


def _build_publications(
    visible_length: int,
    score_by_global_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gathered_indices = []
    gathered_scores = []
    visible = torch.tensor([[visible_length]], dtype=torch.int64)
    for rank in range(P):
        producer_rank = torch.tensor(rank, dtype=torch.int64)
        local_visible = int(_DCP.dcp_local_visible_counts(visible, producer_rank)[0, 0])
        local_indices = torch.full((1, 1, K), -1, dtype=torch.int32)
        local_scores = torch.full((1, 1, K), float("-inf"), dtype=torch.bfloat16)
        if local_visible:
            all_local_indices = torch.arange(local_visible, dtype=torch.int64)
            all_global_indices = torch.tensor(
                [_DCP.dcp_local_to_global_index(int(index), rank) for index in all_local_indices],
                dtype=torch.int64,
            )
            all_local_scores = score_by_global_index[all_global_indices]
            local_order = torch.argsort(all_local_scores, descending=True, stable=True)[:K]
            candidate_count = local_order.numel()
            local_indices[0, 0, :candidate_count] = all_local_indices[local_order].to(torch.int32)
            local_scores[0, 0, :candidate_count] = all_local_scores[local_order]
        local_indices, local_scores = _DCP.publish_dcp_local_candidates(
            local_indices,
            local_scores,
            torch.tensor([[local_visible]], dtype=torch.int32),
            visible,
            producer_rank=producer_rank,
        )
        gathered_indices.append(local_indices)
        gathered_scores.append(local_scores)
    return torch.stack(gathered_indices), torch.stack(gathered_scores)


def _assert_bf16_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16))


def test_tensor_assert_skips_host_fallback_during_npu_graph_capture() -> None:
    condition = type("FakeCondition", (), {"device": type("FakeDevice", (), {"type": "npu"})()})()
    fake_npu = type("FakeNPU", (), {"is_current_stream_capturing": staticmethod(lambda: True)})()

    with (
        patch.object(torch, "npu", fake_npu, create=True),
        patch.object(torch, "_assert_async", side_effect=AssertionError("unexpected host fallback")) as assert_mock,
    ):
        _DCP._assert_tensor(condition, "capture-safe")

    assert_mock.assert_not_called()


def test_tensor_assert_retains_eager_npu_validation() -> None:
    condition = type("FakeCondition", (), {"device": type("FakeDevice", (), {"type": "npu"})()})()
    fake_npu = type("FakeNPU", (), {"is_current_stream_capturing": staticmethod(lambda: False)})()

    with (
        patch.object(torch, "npu", fake_npu, create=True),
        patch.object(torch, "_assert_async") as assert_mock,
    ):
        _DCP._assert_tensor(condition, "eager-check")

    assert_mock.assert_called_once_with(condition, "eager-check")


def test_replay_length_host_validation_accepts_boundaries() -> None:
    _DCP.validate_dcp_sharded_indexer_replay_lengths(torch.tensor([0, 1, MAX_VISIBLE], dtype=torch.int32))


@pytest.mark.parametrize(
    ("visible_lengths", "message"),
    [
        (torch.tensor([-1], dtype=torch.int32), "must be in"),
        (torch.tensor([MAX_VISIBLE + 1], dtype=torch.int64), "must be in"),
        (torch.tensor([[1]], dtype=torch.int32), "one-dimensional"),
        (torch.tensor([1.0], dtype=torch.float32), "must be integer"),
    ],
)
def test_replay_length_host_validation_rejects_invalid_contract(
    visible_lengths: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _DCP.validate_dcp_sharded_indexer_replay_lengths(visible_lengths)


def test_fixed_row_indexer_inputs_preserve_active_rows_and_sanitize_empty_owners() -> None:
    seq_lens = torch.tensor([2, 0, 1], dtype=torch.int32)
    block_table = torch.tensor([[3, 4], [5, 6], [7, 8]], dtype=torch.int32)

    active_rows, safe_key_lens, safe_block_table = _DCP.prepare_dcp_fixed_row_indexer_inputs(
        seq_lens,
        block_table,
    )

    torch.testing.assert_close(active_rows, torch.tensor([True, False, True]))
    torch.testing.assert_close(safe_key_lens, torch.tensor([2, 1, 1], dtype=torch.int32))
    torch.testing.assert_close(
        safe_block_table,
        torch.tensor([[3, 4], [0, 0], [7, 8]], dtype=torch.int32),
    )


def test_fixed_row_mask_allows_zero_count_publication_without_weakening_active_rows() -> None:
    local_indices = torch.full((3, 1, K), -1, dtype=torch.int32)
    local_scores = torch.full((3, 1, K), float("-inf"), dtype=torch.bfloat16)
    local_indices[0, 0, :2] = torch.tensor([0, 1], dtype=torch.int32)
    local_scores[0, 0, :2] = torch.tensor([3.0, 2.0], dtype=torch.bfloat16)
    local_indices[1, 0, 0] = 17
    local_scores[1, 0, 0] = float("nan")
    local_indices[2, 0, 0] = 0
    local_scores[2, 0, 0] = 4.0

    masked_indices, masked_scores = _DCP.mask_dcp_inactive_local_candidates(
        local_indices,
        local_scores,
        torch.tensor([True, False, True]),
    )
    published_indices, published_scores = _DCP.publish_dcp_local_candidates(
        masked_indices,
        masked_scores,
        torch.tensor([[2], [0], [1]], dtype=torch.int32),
        torch.tensor([[2], [0], [1]], dtype=torch.int32),
        producer_rank=torch.tensor(0, dtype=torch.int64),
    )

    torch.testing.assert_close(published_indices[0, 0, :2], torch.tensor([0, 1], dtype=torch.int32))
    _assert_bf16_bits_equal(published_scores[0, 0, :2], local_scores[0, 0, :2])
    assert torch.all(published_indices[1] == -1)
    assert torch.all(torch.isneginf(published_scores[1].float()))
    assert published_indices[2, 0, 0] == 0
    assert published_scores[2, 0, 0] == torch.tensor(4.0, dtype=torch.bfloat16)


def test_extracted_sfa_cp_method_executes_fixed_rows_through_exchange_and_merge() -> None:
    tree = ast.parse(_SFA_CP_PATH.read_text())
    impl_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFADCPImpl")
    method = next(
        node
        for node in impl_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "indexer_select_post_process"
    )
    method_source = ast.get_source_segment(_SFA_CP_PATH.read_text(), method)
    assert method_source is not None

    class StubMetadata:
        pass

    class StubBase:
        @staticmethod
        def _has_prefill(metadata) -> bool:
            return metadata.num_prefills > 0

        def indexer_select_post_process(self, *args, **kwargs):
            self.native_args = args
            self.native_kwargs = kwargs
            return self.native_result

    namespace = {
        "torch": torch,
        "M": object,
        "AscendSFADCPMetadata": StubMetadata,
        "DCP_SHARDED_INDEXER_TOPK": K,
        "prepare_dcp_fixed_row_indexer_inputs": _DCP.prepare_dcp_fixed_row_indexer_inputs,
        "mask_dcp_inactive_local_candidates": _DCP.mask_dcp_inactive_local_candidates,
        "publish_dcp_local_candidates": _DCP.publish_dcp_local_candidates,
        "merge_dcp_global_topk": _DCP.merge_dcp_global_topk,
        "StubBase": StubBase,
    }
    exec(
        compile(
            "from __future__ import annotations\nclass ExtractedImpl(StubBase):\n"
            + textwrap.indent(method_source, "    "),
            str(_SFA_CP_PATH),
            "exec",
        ),
        namespace,
    )
    extracted_impl = namespace["ExtractedImpl"]

    num_rows = 3
    local_indices = torch.full((num_rows, 1, K), -1, dtype=torch.int32)
    local_scores = torch.full((num_rows, 1, K), float("-inf"), dtype=torch.bfloat16)
    local_indices[0, 0, :2] = torch.tensor([0, 1], dtype=torch.int32)
    local_scores[0, 0, :2] = torch.tensor([3.0, 2.0], dtype=torch.bfloat16)
    local_indices[1, 0, 0] = 17
    local_scores[1, 0, 0] = float("nan")
    local_indices[2, 0, 0] = 0
    local_scores[2, 0, 0] = 4.0

    def gather_rank_zero(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        sentinel = -1 if tensor.dtype == torch.int32 else float("-inf")
        publications = torch.full((P, *tensor.shape), sentinel, dtype=tensor.dtype)
        publications[0].copy_(tensor)
        return publications.flatten(0, 1)

    impl = extracted_impl()
    impl.enable_sfa_dcp_sharded_indexer = True
    impl.dcp_size = P
    impl._dcp_producer_rank = torch.tensor(0, dtype=torch.int64)
    impl.dcp_group = SimpleNamespace(all_gather=gather_rank_zero)
    impl.native_result = (local_indices, local_scores)
    metadata = StubMetadata()
    metadata.num_prefills = 0
    metadata.block_table = torch.tensor([[3, 4], [5, 6], [7, 8]], dtype=torch.int32)
    metadata.dcp_context = SimpleNamespace(
        seq_lens=torch.tensor([2, 0, 1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([2, 0, 1], dtype=torch.int32),
        global_seq_lens=torch.tensor([2, 0, 1], dtype=torch.int32),
    )
    query_lens = torch.tensor([1, 2, 3], dtype=torch.int32)

    indices = impl.indexer_select_post_process(
        torch.zeros(num_rows, 4, dtype=torch.bfloat16),
        torch.zeros(num_rows),
        (),
        metadata,
        torch.zeros(num_rows),
        torch.zeros(num_rows),
        query_lens,
        metadata.dcp_context.seq_lens,
    )

    assert impl.native_args[0].shape[0] == num_rows
    torch.testing.assert_close(impl.native_args[6], query_lens)
    torch.testing.assert_close(impl.native_args[7], torch.tensor([2, 1, 1], dtype=torch.int32))
    torch.testing.assert_close(
        impl.native_kwargs["block_table"],
        torch.tensor([[3, 4], [0, 0], [7, 8]], dtype=torch.int32),
    )
    assert "selected_rows" not in impl.native_kwargs
    assert impl.native_kwargs["return_selected_scores"] is True
    assert impl.native_kwargs["sparse_mode"] == 0
    torch.testing.assert_close(indices[0, 0, :2], torch.tensor([0, 1], dtype=torch.int32))
    assert torch.all(indices[0, 0, 2:] == -1)
    assert torch.all(indices[1] == -1)
    assert indices[2, 0, 0] == 0
    assert torch.all(indices[2, 0, 1:] == -1)


def test_sfa_cp_sharded_indexer_native_call_has_fixed_row_geometry() -> None:
    tree = ast.parse(_SFA_CP_PATH.read_text())
    impl = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFADCPImpl")
    method = next(
        node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name == "indexer_select_post_process"
    )
    source = ast.get_source_segment(_SFA_CP_PATH.read_text(), method)
    assert source is not None

    torch_calls = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    }
    assert "nonzero" not in torch_calls
    assert "index_select" not in torch_calls
    assert "positive_rows" not in source
    assert "selected_rows=" not in source
    assert "prepare_dcp_fixed_row_indexer_inputs" in source
    assert "mask_dcp_inactive_local_candidates" in source
    assert "safe_key_lens" in source
    assert "safe_block_table" in source


def test_sfa_cp_uses_validated_cpu_lengths_as_replay_authority() -> None:
    tree = ast.parse(_SFA_CP_PATH.read_text())
    builder = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFADCPMetadataBuilder"
    )
    build = next(
        node for node in builder.body if isinstance(node, ast.FunctionDef) and node.name == "_build_with_metadata_view"
    )
    source = ast.get_source_segment(_SFA_CP_PATH.read_text(), build)
    assert source is not None

    host_bind = source.index('global_seq_lens_cpu = torch.as_tensor(metadata.seq_lens_cpu, device="cpu")')
    validation = source.index("validate_dcp_sharded_indexer_replay_lengths(global_seq_lens_cpu)")
    int32_conversion = source.index("global_seq_lens_cpu = global_seq_lens_cpu.to(dtype=torch.int32)")
    host_derivation = source.index("local_seq_lens_cpu = self._get_dcp_local_seq_lens(global_seq_lens_cpu)")
    device_derivation = source.index("dcp_local_seq_lens = self._get_dcp_local_seq_lens(metadata.seq_lens)")
    persistent_copy = source.index("self.dcp_local_seq_lens_buf[:num_reqs].copy_(local_seq_lens_src")
    assert host_bind < validation < int32_conversion < host_derivation < device_derivation < persistent_copy
    assert "local_seq_lens_src = local_seq_lens_cpu.to(" not in source


def test_visible_count_reuses_rank_tensor_without_factory() -> None:
    visible = torch.tensor([[MAX_VISIBLE]], dtype=torch.int64)
    producer_rank = torch.tensor(11, dtype=torch.int64)

    with patch.object(torch, "as_tensor", side_effect=AssertionError("unexpected rank tensor factory")):
        count = _DCP.dcp_local_visible_counts(visible, producer_rank)

    assert count.item() == 16_384


def test_sfa_cp_reuses_device_static_rank_in_captured_method() -> None:
    tree = ast.parse(_SFA_CP_PATH.read_text())
    impl = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFADCPImpl")
    init = next(node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    select = next(
        node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name == "indexer_select_post_process"
    )

    cached_rank_assignments = []
    for node in ast.walk(init):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "_dcp_producer_rank"
        ):
            cached_rank_assignments.append(node.value)

    assert len(cached_rank_assignments) == 1
    cached_rank = cached_rank_assignments[0]
    assert isinstance(cached_rank, ast.Call)
    assert isinstance(cached_rank.func, ast.Attribute)
    assert isinstance(cached_rank.func.value, ast.Name)
    assert (cached_rank.func.value.id, cached_rank.func.attr) == ("torch", "tensor")
    assert len(cached_rank.args) == 1
    assert ast.unparse(cached_rank.args[0]) == "self.dcp_rank"
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in cached_rank.keywords} == {
        "dtype": "torch.int64",
        "device": "device",
    }

    captured_factories = [
        ast.unparse(node.func)
        for node in ast.walk(select)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr in {"tensor", "as_tensor"}
    ]
    assert captured_factories == []

    select_source = ast.get_source_segment(_SFA_CP_PATH.read_text(), select)
    assert select_source is not None
    assert "producer_rank = self._dcp_producer_rank" in select_source
    assert "dcp_context.global_seq_lens.view(num_rows, 1),\n            producer_rank," in select_source


def test_publication_rejects_duplicate_local_index() -> None:
    local_indices = torch.full((1, 1, K), -1, dtype=torch.int32)
    local_indices[0, 0, :2] = 0
    local_scores = torch.full((1, 1, K), float("-inf"), dtype=torch.bfloat16)
    local_scores[0, 0, :2] = 1.0

    with pytest.raises(ValueError, match="exactly once"):
        _DCP.publish_dcp_local_candidates(
            local_indices,
            local_scores,
            torch.tensor([[2]], dtype=torch.int32),
            torch.tensor([[2]], dtype=torch.int32),
            producer_rank=torch.tensor(0, dtype=torch.int64),
        )


def test_reference_oracle_rejects_duplicate_global_publication() -> None:
    visible_length = 2050
    scores = _strictly_increasing_bf16_scores(visible_length)
    gathered_indices, gathered_scores = _build_publications(visible_length, scores)
    gathered_indices[0, 0, 0, 1] = gathered_indices[0, 0, 0, 0]

    # The exact reference keeps this global completeness oracle for host verification.
    with pytest.raises(ValueError, match="duplicate global index"):
        _DCP.merge_dcp_global_topk(
            gathered_indices,
            gathered_scores,
            torch.tensor([[visible_length]], dtype=torch.int32),
        )


def test_candidate_count_clamps_rank_local_visible_length_at_k() -> None:
    local_visible = torch.tensor([[0, K - 1, K, K + 1, 2304]], dtype=torch.int32)

    candidate_counts = _DCP.dcp_local_candidate_counts(local_visible)

    torch.testing.assert_close(
        candidate_counts,
        torch.tensor([[0, K - 1, K, K, K]], dtype=torch.int64),
    )


def test_publication_accepts_exact_k_from_local_visible_above_k_and_maps_high_indices() -> None:
    visible_length = 36_864
    producer_rank = torch.tensor(0, dtype=torch.int64)
    visible = torch.tensor([[visible_length]], dtype=torch.int32)
    local_visible = _DCP.dcp_local_visible_counts(visible, producer_rank).to(torch.int32)
    assert local_visible.item() == 2304

    local_indices = torch.arange(256, 2304, dtype=torch.int32).view(1, 1, K)
    score_bits = torch.arange(0x4500, 0x4500 - K, -1, dtype=torch.int32).to(torch.uint16)
    local_scores = score_bits.view(torch.bfloat16).view(1, 1, K)

    published_indices, published_scores = _DCP.publish_dcp_local_candidates(
        local_indices,
        local_scores,
        local_visible,
        visible,
        producer_rank,
    )

    expected = torch.tensor(
        [_DCP.dcp_local_to_global_index(index, 0) for index in range(256, 2304)],
        dtype=torch.int32,
    ).view(1, 1, K)
    torch.testing.assert_close(published_indices, expected)
    _assert_bf16_bits_equal(published_scores, local_scores)
    assert published_indices.max().item() > K


def test_publication_rejects_duplicate_candidate_when_local_visible_above_k() -> None:
    visible_length = 36_864
    producer_rank = torch.tensor(0, dtype=torch.int64)
    visible = torch.tensor([[visible_length]], dtype=torch.int32)
    local_visible = _DCP.dcp_local_visible_counts(visible, producer_rank).to(torch.int32)
    local_indices = torch.arange(K, dtype=torch.int32).view(1, 1, K)
    local_indices[0, 0, 1] = 0
    local_scores = torch.ones((1, 1, K), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="must be unique"):
        _DCP.publish_dcp_local_candidates(
            local_indices,
            local_scores,
            local_visible,
            visible,
            producer_rank,
        )


def test_publication_rejects_unstable_equal_score_local_order() -> None:
    local_indices = torch.full((1, 1, K), -1, dtype=torch.int32)
    local_scores = torch.full((1, 1, K), float("-inf"), dtype=torch.bfloat16)
    local_indices[0, 0, :2] = torch.tensor([1, 0], dtype=torch.int32)
    local_scores[0, 0, :2] = 1.0

    with pytest.raises(ValueError, match="local index ascending"):
        _DCP.publish_dcp_local_candidates(
            local_indices,
            local_scores,
            torch.tensor([[2]], dtype=torch.int32),
            torch.tensor([[2]], dtype=torch.int32),
            producer_rank=torch.tensor(0, dtype=torch.int64),
        )


def test_long_context_global_tie_merge_is_exact_when_every_rank_local_visible_exceeds_k() -> None:
    visible_length = 36_864
    scores = torch.ones(visible_length, dtype=torch.bfloat16)
    gathered_indices, gathered_scores = _build_publications(visible_length, scores)

    merged_indices, merged_scores = _DCP.merge_dcp_global_topk(
        gathered_indices,
        gathered_scores,
        torch.tensor([[visible_length]], dtype=torch.int32),
    )

    torch.testing.assert_close(merged_indices[0, 0], torch.arange(K, dtype=torch.int32))
    _assert_bf16_bits_equal(merged_scores[0, 0], torch.ones(K, dtype=torch.bfloat16))
    local_visible = torch.stack(
        [
            _DCP.dcp_local_visible_counts(
                torch.tensor([[visible_length]], dtype=torch.int64),
                torch.tensor(rank, dtype=torch.int64),
            )
            for rank in range(P)
        ]
    )
    assert torch.all(local_visible == 2304)
