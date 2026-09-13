import inspect
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm_ascend.attention.attention_v1 as attn_module
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    split_decodes_and_prefills,
)


class _CapturedMetadata(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs


class _MaskBuilder:
    def __init__(self):
        self.mask = object()

    def get_attention_mask(self, causal, model_config):
        return self.mask


class _SubCounter(TorchDispatchMode):
    def __init__(self):
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        schema = getattr(func, "_schema", None)
        if getattr(schema, "name", None) == "aten::sub":
            self.count += 1
        return func(*args, **(kwargs or {}))


def _builder(cls=AscendAttentionMetadataBuilder):
    builder = cls.__new__(cls)
    builder.pcp_enabled = False
    builder.decode_threshold = 1
    builder.attn_mask_builder = _MaskBuilder()
    builder.model_config = SimpleNamespace(runner_type="test-runner")
    builder.device = "cpu"
    builder.kv_cache_spec = None
    builder.speculative_config = None
    builder.metadata_cls = _CapturedMetadata
    return builder


def _common_metadata(
    lengths,
    *,
    qstart_dtype=torch.int64,
    seq_lens=None,
    block_table_rows=None,
    slot_mapping_dtype=torch.int64,
    strided_qstart=False,
):
    query_start_values = [0, *torch.tensor(lengths, dtype=qstart_dtype).cumsum(0).tolist()]
    query_start = torch.tensor(query_start_values, dtype=qstart_dtype)
    if strided_qstart:
        backing = torch.empty(len(query_start_values) * 2, dtype=qstart_dtype)
        backing[::2] = query_start
        query_start = backing[::2]
    num_reqs = len(lengths)
    num_actual_tokens = int(query_start[-1].item())
    if seq_lens is None:
        seq_lens = [max(length, 1) + 4 for length in lengths]
    if block_table_rows is None:
        block_table_rows = num_reqs
    return AscendCommonAttentionMetadata(
        query_start_loc=query_start.clone(),
        query_start_loc_cpu=query_start.clone(),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        _seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max(lengths, default=0),
        block_table_tensor=torch.arange(block_table_rows * 2, dtype=torch.int64).reshape(
            block_table_rows,
            2,
        ),
        slot_mapping=torch.arange(num_actual_tokens, dtype=slot_mapping_dtype),
        causal=True,
        actual_seq_lengths_q=list(lengths),
        positions=torch.arange(num_actual_tokens, dtype=torch.int64),
        attn_state=AscendAttentionState.ChunkedPrefill,
        max_seq_len=max(seq_lens, default=0),
        context_parallel_metadata=None,
    )


def _expected_phase_counts(builder, common_attn_metadata):
    return split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=builder.decode_threshold,
        treat_short_extends_as_decodes=not builder.pcp_enabled,
    )


def _build(builder, common_attn_metadata):
    with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
        return builder.build(0, common_attn_metadata)


def _subtraction_count(call):
    count, _ = _subtractions_during(call)
    return count


def _subtractions_during(call):
    counter = _SubCounter()
    with counter:
        result = call()
    return counter.count, result


def _phase_tuple(metadata):
    return (
        metadata.num_decodes,
        metadata.num_prefills,
        metadata.num_decode_tokens,
        metadata.num_actual_tokens - metadata.num_decode_tokens,
    )


def test_base_original_noop_elides_hook_call_and_target_subtraction():
    builder = _builder()
    common_attn_metadata = _common_metadata([1, 1, 3, 7], qstart_dtype=torch.int32, strided_qstart=True)
    expected_counts = _expected_phase_counts(builder, common_attn_metadata)
    splitter_subtractions = _subtraction_count(lambda: _expected_phase_counts(builder, common_attn_metadata))
    calls = 0

    def profiler(frame, event, arg):
        nonlocal calls
        if event == "call" and frame.f_code is attn_module._ORIGINAL_BUILD_BACKEND_METADATA.__code__:
            calls += 1
        return profiler

    old_profiler = sys.getprofile()
    sys.setprofile(profiler)
    try:
        build_subtractions, metadata = _subtractions_during(lambda: _build(builder, common_attn_metadata))
    finally:
        sys.setprofile(old_profiler)

    assert calls == 0
    assert not common_attn_metadata.query_start_loc_cpu.is_contiguous()
    assert build_subtractions == splitter_subtractions
    assert _phase_tuple(metadata) == expected_counts


def test_inherited_original_noop_is_elided_for_empty_and_single_request_cases():
    class InheritedBuilder(AscendAttentionMetadataBuilder):
        pass

    for lengths in ([], [1], [5]):
        builder = _builder(InheritedBuilder)
        common_attn_metadata = _common_metadata(lengths)
        metadata = _build(builder, common_attn_metadata)
        expected_counts = _expected_phase_counts(builder, common_attn_metadata)
        assert _phase_tuple(metadata) == expected_counts


def test_subclass_override_receives_original_keywords_once():
    calls = []

    class ConsumingBuilder(AscendAttentionMetadataBuilder):
        def _build_backend_metadata(
            self,
            common_attn_metadata,
            *,
            block_table,
            query_lens,
            seq_lens,
            num_decodes,
            num_prefills,
        ):
            calls.append(
                {
                    "common": common_attn_metadata,
                    "block_table": block_table,
                    "query_lens": query_lens,
                    "seq_lens": seq_lens,
                    "num_decodes": num_decodes,
                    "num_prefills": num_prefills,
                }
            )
            return {"backend_payload": query_lens.clone()}

    builder = _builder(ConsumingBuilder)
    common_attn_metadata = _common_metadata([1, 1, 3, 7], seq_lens=[5, 6, 10, 12])
    splitter_subtractions = _subtraction_count(lambda: _expected_phase_counts(builder, common_attn_metadata))
    build_subtractions, metadata = _subtractions_during(lambda: _build(builder, common_attn_metadata))

    assert len(calls) == 1
    payload = calls[-1]
    assert build_subtractions == splitter_subtractions + 1
    assert payload["common"] is common_attn_metadata
    assert torch.equal(payload["query_lens"], torch.tensor([1, 1, 3, 7], dtype=torch.int64))
    assert payload["query_lens"].dtype == torch.int64
    assert payload["query_lens"].device.type == "cpu"
    assert payload["block_table"] is common_attn_metadata.block_table_tensor
    assert payload["seq_lens"] is metadata.seq_lens
    assert torch.equal(payload["seq_lens"], common_attn_metadata._seq_lens_cpu)
    assert payload["num_decodes"] == metadata.num_decodes == 2
    assert payload["num_prefills"] == metadata.num_prefills == 2
    assert torch.equal(metadata.backend_payload, torch.tensor([1, 1, 3, 7]))


def test_instance_methodtype_callback_and_plain_callable_fall_back_once():
    common_attn_metadata = _common_metadata([3], qstart_dtype=torch.int32)

    for replacement in ("method", "callable"):
        builder = _builder()
        calls = []

        def callback(
            self,
            common_attn_metadata,
            _calls=calls,
            *,
            block_table,
            query_lens,
            seq_lens,
            num_decodes,
            num_prefills,
        ):
            _calls.append(query_lens)
            return {"backend_payload": query_lens}

        if replacement == "method":
            builder._build_backend_metadata = MethodType(callback, builder)
        else:
            builder._build_backend_metadata = type(
                "CallableBackendMetadata",
                (),
                {
                    "__call__": (
                        lambda self, common_attn_metadata, _calls=calls, **kwargs: _calls.append(kwargs["query_lens"])
                        or {}
                    )
                },
            )()

        metadata = _build(builder, common_attn_metadata)

        assert len(calls) == 1
        assert torch.equal(calls[0], torch.tensor([3], dtype=torch.int32))
        if replacement == "method":
            assert metadata.backend_payload is calls[0]


def test_class_replacement_after_builder_use_is_not_mistaken_for_original():
    builder = _builder()
    common_attn_metadata = _common_metadata([1, 4])

    _build(builder, common_attn_metadata)
    calls = []

    def replacement(
        self,
        common_attn_metadata,
        *,
        block_table,
        query_lens,
        seq_lens,
        num_decodes,
        num_prefills,
    ):
        calls.append((common_attn_metadata, query_lens))
        return {"backend_payload": query_lens}

    original = AscendAttentionMetadataBuilder._build_backend_metadata
    AscendAttentionMetadataBuilder._build_backend_metadata = replacement
    try:
        metadata = _build(builder, common_attn_metadata)
    finally:
        AscendAttentionMetadataBuilder._build_backend_metadata = original

    assert len(calls) == 1
    assert calls[0][0] is common_attn_metadata
    assert torch.equal(calls[0][1], torch.tensor([1, 4]))
    assert metadata.backend_payload is calls[0][1]


def test_callable_with_spoofed_func_attribute_falls_back_once():
    builder = _builder()
    common_attn_metadata = _common_metadata([2], qstart_dtype=torch.int32)
    calls = []

    class SpoofedCallable:
        __func__ = attn_module._ORIGINAL_BUILD_BACKEND_METADATA

        def __call__(self, common_attn_metadata, **kwargs):
            calls.append(kwargs["query_lens"])
            return {"backend_payload": kwargs["query_lens"]}

    builder._build_backend_metadata = SpoofedCallable()
    metadata = _build(builder, common_attn_metadata)

    assert len(calls) == 1
    assert torch.equal(calls[0], torch.tensor([2], dtype=torch.int32))
    assert metadata.backend_payload is calls[0]


def test_padding_forwards_prepared_tensors_and_keeps_inputs_unchanged():
    calls = []

    class PaddingConsumerBuilder(AscendAttentionMetadataBuilder):
        def _build_backend_metadata(
            self,
            common_attn_metadata,
            *,
            block_table,
            query_lens,
            seq_lens,
            num_decodes,
            num_prefills,
        ):
            calls.append((block_table, query_lens, seq_lens))
            return {}

    builder = _builder(PaddingConsumerBuilder)
    common_attn_metadata = _common_metadata([1, 1, 3], seq_lens=[8, 9], block_table_rows=2)
    original_qstart = common_attn_metadata.query_start_loc_cpu.clone()
    original_seq_lens = common_attn_metadata._seq_lens_cpu.clone()
    original_block_table = common_attn_metadata.block_table_tensor.clone()
    original_slot_mapping = common_attn_metadata.slot_mapping.clone()

    metadata = _build(builder, common_attn_metadata)

    block_table, query_lens, seq_lens = calls[0]
    assert torch.equal(query_lens, torch.tensor([1, 1, 3]))
    assert block_table.shape == (3, 2)
    assert seq_lens.shape == (3,)
    assert block_table is metadata.block_tables
    assert seq_lens is metadata.seq_lens
    assert torch.equal(seq_lens, torch.tensor([8, 9, 1]))
    assert torch.equal(block_table[-1], torch.tensor([0, 0]))
    assert _phase_tuple(metadata) == (2, 1, 2, 3)
    assert torch.equal(common_attn_metadata.query_start_loc_cpu, original_qstart)
    assert torch.equal(common_attn_metadata._seq_lens_cpu, original_seq_lens)
    assert torch.equal(common_attn_metadata.block_table_tensor, original_block_table)
    assert torch.equal(common_attn_metadata.slot_mapping, original_slot_mapping)


def test_original_noop_branch_uses_literal_backend_metadata_dict():
    source = inspect.getsource(AscendAttentionMetadataBuilder.build)

    assert "backend_metadata = {}" in source
    assert "is _ORIGINAL_BUILD_BACKEND_METADATA" in source
