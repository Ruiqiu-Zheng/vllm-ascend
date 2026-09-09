# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
    compose_indexer_cache_metadata,
)

_KERNEL_BLOCK_SIZE = 128


def _make_builder(pcp_size: int = 1, dcp_size: int = 1) -> AscendSFAIndexerMetadataBuilder:
    kv_cache_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=160,
        dtype=torch.uint8,
    )
    layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
    vllm_config = MagicMock()
    vllm_config.parallel_config.prefill_context_parallel_size = pcp_size
    vllm_config.parallel_config.decode_context_parallel_size = dcp_size
    with patch(
        "vllm_ascend.attention.indexer.select_common_block_size",
        return_value=_KERNEL_BLOCK_SIZE,
    ):
        return AscendSFAIndexerMetadataBuilder(
            kv_cache_spec,
            layer_names,
            vllm_config,
            torch.device("cpu"),
        )


def _make_common_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        num_input_tokens=4,
        slot_mapping=torch.tensor([1, 2, 3, 4, 5]),
        positions=torch.tensor([0, 1, 0, 1, 9]),
        query_start_loc=torch.tensor([0, 2, 4]),
        seq_lens=torch.tensor([5, 6, 7]),
        block_table_tensor=torch.arange(6).view(3, 2),
        group_len=MagicMock(name="group_len"),
        group_key_idx=MagicMock(name="group_key_idx"),
        group_key_cache_idx=MagicMock(name="group_key_cache_idx"),
    )


def test_sfa_indexer_backend_contract():
    assert AscendSFAIndexerBackend.accept_output_buffer
    assert AscendSFAIndexerBackend.get_name() == "ASCEND_SFA_INDEXER"
    assert AscendSFAIndexerBackend.get_builder_cls() is AscendSFAIndexerMetadataBuilder
    assert AscendSFAIndexerBackend.get_kv_cache_shape(8, 128, 1, 160) == (
        8,
        128,
        1,
        160,
    )
    assert AscendSFAIndexerBackend.get_supported_kernel_block_sizes() == [128]


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_builds_kernel_metadata(mock_cos_sin, mock_get_ascend_config):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    cos = torch.zeros(5, 1, 1, 8)
    sin = torch.zeros(5, 1, 1, 8)
    mock_cos_sin.return_value = (cos, sin)

    builder = _make_builder()
    assert builder.reorder_batch_threshold is None
    assert builder.get_cudagraph_support(MagicMock(), MagicMock()) is AttentionCGSupport.UNIFORM_BATCH

    common = _make_common_metadata()
    metadata = builder.build(0, common)

    assert isinstance(metadata, AscendSFAIndexerMetadata)
    assert metadata.num_actual_tokens == 4
    assert torch.equal(metadata.slot_mapping, common.slot_mapping[:4])
    assert torch.equal(metadata.seq_lens, common.seq_lens[:2])
    assert torch.equal(metadata.cum_query_lens, common.query_start_loc[1:3])
    assert torch.equal(metadata.block_table, common.block_table_tensor[:2])
    assert metadata.block_size == _KERNEL_BLOCK_SIZE
    assert metadata.group_len is common.group_len
    assert metadata.group_key_idx is common.group_key_idx
    assert metadata.group_key_cache_idx is common.group_key_cache_idx

    positions = mock_cos_sin.call_args.args[0]
    assert torch.equal(positions, common.positions[:4])
    assert mock_cos_sin.call_args.kwargs["use_cache"] is True
    assert torch.equal(metadata.cos, cos[:4])
    assert torch.equal(metadata.sin, sin[:4])


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_emits_full_slot_mapping_under_pcp(mock_cos_sin, mock_get_ascend_config):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder(pcp_size=2)
    common = _make_common_metadata()
    metadata = builder.build(0, common)

    # Under PCP the commit writes the gathered prefill region too, so the
    # builder emits the full slot mapping instead of the input-token slice.
    assert metadata.slot_mapping is common.slot_mapping


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.torch.ops._C_ascend.store_kv_block_metadata", create=True)
def test_sfa_indexer_metadata_builder_primes_reshape_optim(
    mock_store_kv_block_metadata,
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = True
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder()
    common = _make_common_metadata()
    metadata = builder.build(0, common)

    mock_store_kv_block_metadata.assert_called_once_with(
        metadata.slot_mapping,
        common.group_len,
        common.group_key_idx,
        common.group_key_cache_idx,
        _KERNEL_BLOCK_SIZE,
    )


def _make_indexer_metadata() -> AscendSFAIndexerMetadata:
    return AscendSFAIndexerMetadata(
        num_actual_tokens=2,
        slot_mapping=torch.tensor([1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        cum_query_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[3, 4]], dtype=torch.int32),
        sin=torch.zeros(2, 1, 1, 8),
        cos=torch.zeros(2, 1, 1, 8),
        block_size=_KERNEL_BLOCK_SIZE,
        group_len=torch.tensor([2], dtype=torch.int32),
        group_key_idx=torch.tensor([0, 1], dtype=torch.int32),
        group_key_cache_idx=torch.tensor([1, 2], dtype=torch.int32),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([2], dtype=torch.int32),
    )


def _make_replicated_sfa_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        dcp_context=object(),
        dsa_cp_context=None,
        block_size=_KERNEL_BLOCK_SIZE,
        block_table=torch.tensor([[6, 7, 8, 9]], dtype=torch.int32),
        slot_mapping=torch.tensor([768, 769], dtype=torch.int32),
    )


def test_compose_indexer_cache_metadata_uses_replicated_dcp_addresses():
    metadata = _make_indexer_metadata()
    sfa_metadata = _make_replicated_sfa_metadata()
    cache = (torch.empty(16, _KERNEL_BLOCK_SIZE, 1, 128),)

    composed = compose_indexer_cache_metadata(metadata, sfa_metadata, cache)

    assert composed is not metadata
    assert composed.block_table is sfa_metadata.block_table
    assert composed.slot_mapping is sfa_metadata.slot_mapping
    assert composed.actual_seq_lengths_query is metadata.actual_seq_lengths_query
    assert composed.actual_seq_lengths_key is metadata.actual_seq_lengths_key
    assert composed.group_len is None
    assert composed.group_key_idx is None
    assert composed.group_key_cache_idx is None
    assert metadata.group_len is not None
    assert metadata.group_key_idx is not None
    assert metadata.group_key_cache_idx is not None


def test_compose_indexer_cache_metadata_is_identity_without_dcp():
    metadata = _make_indexer_metadata()
    sfa_metadata = _make_replicated_sfa_metadata()
    sfa_metadata.dcp_context = None

    assert compose_indexer_cache_metadata(metadata, sfa_metadata, ()) is metadata


def test_compose_indexer_cache_metadata_validates_dsa_padded_slots():
    metadata = _make_indexer_metadata()
    sfa_metadata = _make_replicated_sfa_metadata()
    sfa_metadata.dsa_cp_context = SimpleNamespace(num_tokens_pad=4)
    sfa_metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
    cache = (torch.empty(16, _KERNEL_BLOCK_SIZE, 1, 128),)

    composed = compose_indexer_cache_metadata(metadata, sfa_metadata, cache)
    assert composed.slot_mapping is sfa_metadata.slot_mapping

    sfa_metadata.slot_mapping = torch.arange(3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="TP-padded gathered keys"):
        compose_indexer_cache_metadata(metadata, sfa_metadata, cache)


def test_compose_indexer_cache_metadata_validates_block_geometry():
    metadata = _make_indexer_metadata()
    sfa_metadata = _make_replicated_sfa_metadata()

    sfa_metadata.block_size = 256
    with pytest.raises(RuntimeError, match="matching kernel block sizes"):
        compose_indexer_cache_metadata(metadata, sfa_metadata, (torch.empty(16, 256, 1, 128),))

    sfa_metadata.block_size = _KERNEL_BLOCK_SIZE
    with pytest.raises(RuntimeError, match="matching physical and kernel block sizes"):
        compose_indexer_cache_metadata(metadata, sfa_metadata, (torch.empty(16, 256, 1, 128),))


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.torch.ops._C_ascend.store_kv_block_metadata", create=True)
def test_sfa_indexer_metadata_builder_disables_c8_reshape_metadata_under_dcp(
    mock_store_kv_block_metadata,
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = True
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))
    builder = _make_builder(dcp_size=2)

    metadata = builder.build(0, _make_common_metadata())

    mock_store_kv_block_metadata.assert_not_called()
    assert metadata.group_len is None
    assert metadata.group_key_idx is None
    assert metadata.group_key_cache_idx is None


def test_sfa_indexer_backend_disables_c8_reshape_write_under_dcp():
    backend = SimpleNamespace(enable_sparse_li_c8=True, _dcp_active=True)
    with patch("vllm_ascend.attention.indexer.get_ascend_config") as mock_get_ascend_config:
        mock_get_ascend_config.return_value.c8_reshape_optim_enabled = True
        assert not AscendSFAIndexerBackend._use_c8_reshape_optim(backend)
