# SPDX-License-Identifier: Apache-2.0
"""Exact candidate publication for the bound SFA DCP sharded indexer.

This module deliberately implements only the GLM-5.2 DCP16 contract retained
by R392/R399/R401.  The runtime entry points import torch lazily so the scalar
index mapping remains available to dependency-free source/static gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


DCP_SHARDED_INDEXER_WORLD_SIZE = 16
DCP_SHARDED_INDEXER_INTERLEAVE_SIZE = 128
DCP_SHARDED_INDEXER_TOPK = 2048
DCP_SHARDED_INDEXER_MAX_VISIBLE_LENGTH = 262_144


def validate_dcp_sharded_indexer_replay_lengths(visible_lengths: torch.Tensor) -> None:
    """Validate replay-variable DCP visible lengths on their CPU authority."""
    import torch

    if visible_lengths.device.type != "cpu":
        raise ValueError(
            "DCP sharded-indexer replay lengths must be validated on CPU before graph replay, "
            f"got {visible_lengths.device}."
        )
    if visible_lengths.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"DCP sharded-indexer replay lengths must be integer, got {visible_lengths.dtype}.")
    if visible_lengths.ndim != 1:
        raise ValueError(
            "DCP sharded-indexer replay lengths must be a one-dimensional batch vector, "
            f"got shape {tuple(visible_lengths.shape)}."
        )
    if visible_lengths.numel() == 0:
        return

    minimum = int(torch.min(visible_lengths))
    maximum = int(torch.max(visible_lengths))
    if minimum < 0 or maximum > DCP_SHARDED_INDEXER_MAX_VISIBLE_LENGTH:
        raise ValueError(
            "DCP sharded-indexer replay visible lengths must be in "
            f"[0, {DCP_SHARDED_INDEXER_MAX_VISIBLE_LENGTH}], got min={minimum}, max={maximum}."
        )


def dcp_local_to_global_index(
    local_index: int,
    producer_rank: int,
    *,
    world_size: int = DCP_SHARDED_INDEXER_WORLD_SIZE,
    interleave_size: int = DCP_SHARDED_INDEXER_INTERLEAVE_SIZE,
) -> int:
    """Map an owner-local logical index to its global logical index."""
    if local_index < 0:
        raise ValueError(f"local_index must be nonnegative, got {local_index}.")
    if not 0 <= producer_rank < world_size:
        raise ValueError(f"producer_rank must be in [0, {world_size}), got {producer_rank}.")
    if world_size <= 0 or interleave_size <= 0:
        raise ValueError("world_size and interleave_size must be positive.")
    local_block, offset = divmod(local_index, interleave_size)
    return (local_block * world_size + producer_rank) * interleave_size + offset


def _assert_tensor(condition: torch.Tensor, message: str) -> None:
    import torch

    if condition.device.type == "cpu":
        if not bool(condition):
            raise ValueError(message)
        return
    if condition.device.type == "npu" and torch.npu.is_current_stream_capturing():
        # ``aten::_assert_async`` falls back to a synchronized host copy on
        # this runtime and is therefore illegal while native ACL capture is
        # active.  Eager execution retains the assertion below; graph outputs
        # are validated after replay by the caller's eager correctness gate.
        return
    torch._assert_async(condition, message)


def _validate_bound_shape(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim < 2 or tensor.shape[-1] != DCP_SHARDED_INDEXER_TOPK:
        raise ValueError(
            f"{name} must have at least two dimensions and K={DCP_SHARDED_INDEXER_TOPK} slots, "
            f"got {tuple(tensor.shape)}."
        )


def dcp_local_visible_counts(
    visible_lengths: torch.Tensor,
    producer_rank: torch.Tensor,
) -> torch.Tensor:
    """Return exact B=128 interleaved owner-visible counts for global prefixes."""
    import torch

    world_size = DCP_SHARDED_INDEXER_WORLD_SIZE
    interleave_size = DCP_SHARDED_INDEXER_INTERLEAVE_SIZE
    cycle_size = world_size * interleave_size
    visible_lengths_i64 = visible_lengths.to(torch.int64)
    if producer_rank.dtype != torch.int64:
        raise ValueError(f"producer_rank must be int64, got {producer_rank.dtype}.")
    if producer_rank.device != visible_lengths.device:
        raise ValueError(
            "producer_rank and visible_lengths must share a device, "
            f"got {producer_rank.device} and {visible_lengths.device}."
        )
    full_cycles = torch.div(visible_lengths_i64, cycle_size, rounding_mode="floor")
    remainder = visible_lengths_i64 - full_cycles * cycle_size
    rank_remainder = torch.clamp(
        remainder - producer_rank * interleave_size,
        min=0,
        max=interleave_size,
    )
    return full_cycles * interleave_size + rank_remainder


def dcp_local_candidate_counts(local_visible_counts: torch.Tensor) -> torch.Tensor:
    """Return the fixed-K publication count for each rank-local visible prefix."""
    import torch

    if local_visible_counts.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"local_visible_counts must be integer, got {local_visible_counts.dtype}.")
    return torch.clamp_max(local_visible_counts.to(torch.int64), DCP_SHARDED_INDEXER_TOPK)


def prepare_dcp_fixed_row_indexer_inputs(
    local_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-row PA inputs while making zero-owner rows device-safe."""
    import torch

    if local_seq_lens.ndim != 1:
        raise ValueError(f"local_seq_lens must be one-dimensional, got {tuple(local_seq_lens.shape)}.")
    if local_seq_lens.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"local_seq_lens must be integer, got {local_seq_lens.dtype}.")
    if block_table.ndim != 2 or block_table.shape[0] != local_seq_lens.shape[0]:
        raise ValueError(
            "block_table must be a two-dimensional fixed-row matrix matching local_seq_lens, "
            f"got {tuple(block_table.shape)} and {tuple(local_seq_lens.shape)}."
        )
    if block_table.dtype != torch.int32:
        raise ValueError(f"block_table must be int32, got {block_table.dtype}.")
    if block_table.device != local_seq_lens.device:
        raise ValueError(
            f"block_table and local_seq_lens must share a device, got {block_table.device} and {local_seq_lens.device}."
        )

    active_rows = local_seq_lens > 0
    safe_key_lens = torch.clamp_min(local_seq_lens, 1)
    safe_block_table = torch.where(active_rows.view(-1, 1), block_table, 0)
    return active_rows, safe_key_lens, safe_block_table


def mask_dcp_inactive_local_candidates(
    local_indices: torch.Tensor,
    local_scores: torch.Tensor,
    active_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore canonical sentinels for fixed-row native calls with empty owners."""
    import torch

    _validate_bound_shape(local_indices, "local_indices")
    _validate_bound_shape(local_scores, "local_scores")
    if local_indices.shape != local_scores.shape:
        raise ValueError(
            f"local score/index shapes must match, got {tuple(local_scores.shape)} and {tuple(local_indices.shape)}."
        )
    if active_rows.ndim != 1 or active_rows.shape[0] != local_indices.shape[0]:
        raise ValueError(
            "active_rows must be a one-dimensional mask matching the candidate row count, "
            f"got {tuple(active_rows.shape)} and {tuple(local_indices.shape)}."
        )
    if active_rows.dtype != torch.bool:
        raise ValueError(f"active_rows must be bool, got {active_rows.dtype}.")
    if active_rows.device != local_indices.device or local_scores.device != local_indices.device:
        raise ValueError("active_rows, local_indices, and local_scores must share a device.")

    active_candidate_rows = active_rows.view(
        active_rows.shape[0],
        *([1] * (local_indices.ndim - 1)),
    )
    masked_indices = torch.where(active_candidate_rows, local_indices, -1)
    masked_scores = torch.where(active_candidate_rows, local_scores, float("-inf"))
    return masked_indices, masked_scores


def publish_dcp_local_candidates(
    local_indices: torch.Tensor,
    local_scores: torch.Tensor,
    local_visible_counts: torch.Tensor,
    visible_lengths: torch.Tensor,
    producer_rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish one rank's exact local TopK in global logical coordinates.

    ``local_visible_counts`` is the authoritative number of visible tokens on
    this DCP rank. The native publication contains ``min(local_visible, K)``
    aligned score/index candidates followed by canonical ``(-1, -inf)``
    sentinels. For short rows every visible local index must be present. For
    long rows the native candidates must be a unique, in-range, exact-stable
    local TopK ordered by score descending and local index ascending on ties.
    """
    import torch

    _validate_bound_shape(local_indices, "local_indices")
    _validate_bound_shape(local_scores, "local_scores")
    if local_indices.shape != local_scores.shape:
        raise ValueError(
            f"local score/index shapes must match, got {tuple(local_scores.shape)} and {tuple(local_indices.shape)}."
        )
    if local_indices.dtype != torch.int32:
        raise ValueError(f"local_indices must be int32, got {local_indices.dtype}.")
    if local_scores.dtype != torch.bfloat16:
        raise ValueError(f"local_scores must be bfloat16, got {local_scores.dtype}.")
    expected_prefix_shape = local_indices.shape[:-1]
    if local_visible_counts.shape != expected_prefix_shape or visible_lengths.shape != expected_prefix_shape:
        raise ValueError(
            "local_visible_counts and visible_lengths must match the score/index prefix shape, "
            f"got {tuple(local_visible_counts.shape)}, {tuple(visible_lengths.shape)}, and {expected_prefix_shape}."
        )
    if producer_rank.numel() != 1:
        raise ValueError(f"producer_rank must be scalar, got shape {tuple(producer_rank.shape)}.")
    if producer_rank.dtype != torch.int64:
        raise ValueError(f"producer_rank must be int64, got {producer_rank.dtype}.")
    if producer_rank.device != local_indices.device:
        raise ValueError(
            "producer_rank and local candidates must share a device, "
            f"got {producer_rank.device} and {local_indices.device}."
        )
    if local_visible_counts.device != local_indices.device or visible_lengths.device != local_indices.device:
        raise ValueError("local_visible_counts, visible_lengths, and local candidates must share a device.")

    local_visible_i64 = local_visible_counts.to(torch.int64)
    visible_lengths_i64 = visible_lengths.to(torch.int64)
    _assert_tensor(
        torch.all((visible_lengths_i64 >= 0) & (visible_lengths_i64 <= DCP_SHARDED_INDEXER_MAX_VISIBLE_LENGTH)),
        f"visible_lengths must be in [0, {DCP_SHARDED_INDEXER_MAX_VISIBLE_LENGTH}].",
    )
    _assert_tensor(
        torch.all(local_visible_i64 >= 0),
        "Rank-local visible counts must be nonnegative.",
    )
    expected_visible = dcp_local_visible_counts(visible_lengths_i64, producer_rank)
    _assert_tensor(
        torch.all(local_visible_i64 == expected_visible),
        "Rank-local visible count does not match the DCP layout.",
    )
    candidate_counts = dcp_local_candidate_counts(local_visible_i64)

    slot_ids = torch.arange(
        DCP_SHARDED_INDEXER_TOPK,
        dtype=torch.int64,
        device=local_indices.device,
    )
    valid_mask = slot_ids < candidate_counts.unsqueeze(-1)
    local_indices_i64 = local_indices.to(torch.int64)
    valid_local_range = (local_indices_i64 >= 0) & (local_indices_i64 < local_visible_i64.unsqueeze(-1))
    _assert_tensor(
        torch.all(torch.where(valid_mask, valid_local_range, True)),
        "A valid native candidate index is outside its owner-local visible range.",
    )
    _assert_tensor(
        torch.all(torch.where(valid_mask, True, local_indices_i64 == -1)),
        "Every native suffix index must be -1.",
    )
    _assert_tensor(
        torch.all(torch.where(valid_mask, ~torch.isnan(local_scores.to(torch.float32)), True)),
        "NaN is not a valid DCP candidate score.",
    )
    _assert_tensor(
        torch.all(torch.where(valid_mask, True, torch.isneginf(local_scores.to(torch.float32)))),
        "Every native suffix score must be canonical negative infinity.",
    )

    invalid_sort_key = torch.full_like(local_indices_i64, torch.iinfo(torch.int64).max)
    sorted_local = torch.sort(torch.where(valid_mask, local_indices_i64, invalid_sort_key), dim=-1).values
    expected_short_local = torch.where(valid_mask, slot_ids, invalid_sort_key)
    short_rows = local_visible_i64 <= DCP_SHARDED_INDEXER_TOPK
    _assert_tensor(
        torch.all(torch.where(short_rows.unsqueeze(-1), sorted_local == expected_short_local, True)),
        "Short native candidate rows must contain every owner-local visible index exactly once.",
    )
    pair_is_valid = slot_ids[:-1] + 1 < candidate_counts.unsqueeze(-1)
    duplicate_pairs = sorted_local[..., :-1] == sorted_local[..., 1:]
    _assert_tensor(
        torch.all(torch.where(pair_is_valid, ~duplicate_pairs, True)),
        "Native local TopK candidate indices must be unique.",
    )

    left_scores = local_scores[..., :-1]
    right_scores = local_scores[..., 1:]
    left_indices = local_indices_i64[..., :-1]
    right_indices = local_indices_i64[..., 1:]
    scores_nonincreasing = left_scores >= right_scores
    equal_score_indices_ascending = (left_scores != right_scores) | (left_indices < right_indices)
    _assert_tensor(
        torch.all(torch.where(pair_is_valid, scores_nonincreasing & equal_score_indices_ascending, True)),
        "Native local TopK must be ordered by score descending and local index ascending on equal scores.",
    )

    safe_local = torch.where(valid_mask, local_indices_i64, 0)
    local_blocks = torch.div(
        safe_local,
        DCP_SHARDED_INDEXER_INTERLEAVE_SIZE,
        rounding_mode="floor",
    )
    local_offsets = safe_local - local_blocks * DCP_SHARDED_INDEXER_INTERLEAVE_SIZE
    global_indices_i64 = (
        local_blocks * DCP_SHARDED_INDEXER_WORLD_SIZE + producer_rank
    ) * DCP_SHARDED_INDEXER_INTERLEAVE_SIZE + local_offsets
    _assert_tensor(
        torch.all(torch.where(valid_mask, global_indices_i64 < visible_lengths_i64.unsqueeze(-1), True)),
        "A mapped DCP candidate is outside its visible global prefix.",
    )
    _assert_tensor(
        torch.all(torch.where(valid_mask, global_indices_i64 <= torch.iinfo(torch.int32).max, True)),
        "A mapped DCP candidate does not fit int32.",
    )

    published_indices = torch.where(valid_mask, global_indices_i64, -1).to(torch.int32)
    published_scores = torch.where(
        valid_mask,
        local_scores,
        torch.full_like(local_scores, float("-inf")),
    )
    return published_indices, published_scores


def merge_dcp_global_topk(
    gathered_global_indices: torch.Tensor,
    gathered_scores: torch.Tensor,
    visible_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge exact per-rank local TopKs by score-desc/global-index-asc."""
    import torch

    _validate_bound_shape(gathered_global_indices, "gathered_global_indices")
    _validate_bound_shape(gathered_scores, "gathered_scores")
    if gathered_global_indices.shape != gathered_scores.shape:
        raise ValueError("Gathered score/index shapes must match.")
    if gathered_global_indices.shape[0] != DCP_SHARDED_INDEXER_WORLD_SIZE:
        raise ValueError(f"The bound merge requires 16 producers, got {gathered_global_indices.shape[0]}.")
    if gathered_global_indices.dtype != torch.int32:
        raise ValueError(f"Gathered indices must be int32, got {gathered_global_indices.dtype}.")
    if gathered_scores.dtype != torch.bfloat16:
        raise ValueError(f"Gathered scores must be bfloat16, got {gathered_scores.dtype}.")
    if visible_lengths.shape != gathered_global_indices.shape[1:-1]:
        raise ValueError(
            f"visible_lengths must have shape {gathered_global_indices.shape[1:-1]}, "
            f"got {tuple(visible_lengths.shape)}."
        )
    if (
        visible_lengths.device != gathered_global_indices.device
        or gathered_scores.device != gathered_global_indices.device
    ):
        raise ValueError("visible_lengths, gathered indices, and gathered scores must share a device.")

    visible_lengths_i64 = visible_lengths.to(torch.int64)
    _assert_tensor(
        torch.all((visible_lengths_i64 >= 0) & (visible_lengths_i64 <= DCP_SHARDED_INDEXER_MAX_VISIBLE_LENGTH)),
        f"visible_lengths must be in [0, {DCP_SHARDED_INDEXER_MAX_VISIBLE_LENGTH}].",
    )
    rank_shape = (DCP_SHARDED_INDEXER_WORLD_SIZE,) + (1,) * visible_lengths.ndim
    ranks = torch.arange(
        DCP_SHARDED_INDEXER_WORLD_SIZE,
        dtype=torch.int64,
        device=gathered_global_indices.device,
    ).view(rank_shape)
    local_visible_counts = dcp_local_visible_counts(visible_lengths_i64.unsqueeze(0), ranks)
    candidate_counts = dcp_local_candidate_counts(local_visible_counts)
    slot_ids = torch.arange(
        DCP_SHARDED_INDEXER_TOPK,
        dtype=torch.int64,
        device=gathered_global_indices.device,
    )
    valid_mask = slot_ids < candidate_counts.unsqueeze(-1)
    indices_i64 = gathered_global_indices.to(torch.int64)
    _assert_tensor(
        torch.all(torch.where(valid_mask, indices_i64 >= 0, indices_i64 == -1)),
        "Gathered index validity disagrees with the authoritative candidate count.",
    )
    _assert_tensor(
        torch.all(torch.where(valid_mask, True, torch.isneginf(gathered_scores.to(torch.float32)))),
        "Every invalid gathered score must be canonical negative infinity.",
    )
    _assert_tensor(
        torch.all(torch.where(valid_mask, ~torch.isnan(gathered_scores.to(torch.float32)), True)),
        "NaN is not a valid DCP candidate score.",
    )
    owners = torch.remainder(
        torch.div(torch.clamp(indices_i64, min=0), DCP_SHARDED_INDEXER_INTERLEAVE_SIZE, rounding_mode="floor"),
        DCP_SHARDED_INDEXER_WORLD_SIZE,
    )
    _assert_tensor(
        torch.all(torch.where(valid_mask, owners == ranks.unsqueeze(-1), True)),
        "A gathered candidate does not belong to its producer rank.",
    )
    _assert_tensor(
        torch.all(torch.where(valid_mask, indices_i64 < visible_lengths_i64.unsqueeze(0).unsqueeze(-1), True)),
        "A gathered candidate is outside its visible global prefix.",
    )

    pair_is_valid = slot_ids[:-1] + 1 < candidate_counts.unsqueeze(-1)
    left_scores = gathered_scores[..., :-1]
    right_scores = gathered_scores[..., 1:]
    left_indices = indices_i64[..., :-1]
    right_indices = indices_i64[..., 1:]
    scores_nonincreasing = left_scores >= right_scores
    equal_score_indices_ascending = (left_scores != right_scores) | (left_indices < right_indices)
    _assert_tensor(
        torch.all(torch.where(pair_is_valid, scores_nonincreasing & equal_score_indices_ascending, True)),
        "Each gathered local TopK must be ordered by score descending and global index ascending on ties.",
    )

    flat_indices = gathered_global_indices.movedim(0, -2).flatten(start_dim=-2).to(torch.int64)
    flat_scores = gathered_scores.movedim(0, -2).flatten(start_dim=-2)
    flat_valid = valid_mask.movedim(0, -2).flatten(start_dim=-2)
    invalid_index = torch.iinfo(torch.int64).max
    candidate_keys = torch.where(flat_valid, flat_indices, invalid_index)
    sorted_candidates = torch.sort(candidate_keys, dim=-1).values
    duplicate_candidates = (sorted_candidates[..., :-1] == sorted_candidates[..., 1:]) & (
        sorted_candidates[..., 1:] != invalid_index
    )
    _assert_tensor(
        torch.all(~duplicate_candidates),
        "DCP candidate publications contain a duplicate global index.",
    )

    all_local_complete = torch.all(local_visible_counts <= DCP_SHARDED_INDEXER_TOPK, dim=0)
    all_slot_ids = torch.arange(
        DCP_SHARDED_INDEXER_WORLD_SIZE * DCP_SHARDED_INDEXER_TOPK,
        dtype=torch.int64,
        device=gathered_global_indices.device,
    )
    expected_complete = torch.where(
        all_slot_ids < visible_lengths_i64.unsqueeze(-1),
        all_slot_ids,
        invalid_index,
    )
    _assert_tensor(
        torch.all(torch.where(all_local_complete.unsqueeze(-1), sorted_candidates == expected_complete, True)),
        "Short-row DCP publications omit a visible global index.",
    )
    required_output_count = torch.clamp_max(visible_lengths_i64, DCP_SHARDED_INDEXER_TOPK)
    published_candidate_count = torch.sum(candidate_counts, dim=0)
    _assert_tensor(
        torch.all(published_candidate_count >= required_output_count),
        "DCP publications do not contain enough candidates for the requested global TopK.",
    )

    # First order by the secondary key, then use a stable score sort. This
    # preserves global-index ascending order for every numerically equal BF16
    # score, including signed zero and real negative infinity.
    index_order = torch.argsort(candidate_keys, dim=-1, stable=True)
    indices_by_index = torch.gather(flat_indices, -1, index_order)
    scores_by_index = torch.gather(flat_scores, -1, index_order)
    score_order = torch.argsort(scores_by_index, dim=-1, descending=True, stable=True)
    merged_indices = torch.gather(indices_by_index, -1, score_order)[..., :DCP_SHARDED_INDEXER_TOPK]
    merged_scores = torch.gather(scores_by_index, -1, score_order)[..., :DCP_SHARDED_INDEXER_TOPK]

    output_slots = slot_ids < required_output_count.unsqueeze(-1)
    output_indices = torch.where(output_slots, merged_indices, -1).to(torch.int32)
    output_scores = torch.where(
        output_slots,
        merged_scores,
        torch.full_like(merged_scores, float("-inf")),
    )
    return output_indices, output_scores
