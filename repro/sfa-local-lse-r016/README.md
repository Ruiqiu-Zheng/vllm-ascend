# Local SFA LSE R016 performance reproducibility packet

This auxiliary packet makes the performance evidence for the Local SFA LSE
packing optimization inspectable without adding benchmark-only files to the
production PR.

## What changed

Baseline:

```text
native FP32 softmax_max / softmax_sum
-> materialize LSE = max + log(sum)
-> permute/reshape
-> DCP pack
-> HCCL All2All
-> stable combine
```

Candidate:

```text
native FP32 softmax_max / softmax_sum
-> max/sum-aware DCP pack (forms LSE while writing packed layout)
-> same HCCL All2All
-> unchanged stable combine
```

## Fixed protocol

The protocol was preregistered before measurement:

- protocol seed: `20260916`
- BF16 producer seed: `20260913`
- FP16 correctness seed: `20260914`
- warmup: `10`
- samples per arm/block: `5`
- paired blocks: `8`
- base/candidate pair order: fixed in `preregistration.json`
- pre/post base-only null-control order: fixed in `preregistration.json`
- no post-hoc performance samples

The exact binary fixture is `inputs/fixture.bin`:

```text
SHA256 63785fa8f4592b7f8f9e06d345a20cf29a4ac8d19adc1e89dc8f90e60eb7db19
```

`run_two_rank_perf.py` asserts this hash before loading it.

## Measurement and decision rule

Three intervals are measured:

1. local composition: precomputed output/max/sum -> LSE/layout/pack
2. world-size-one: SparseFlashAttention producer -> pack
3. same-host two-rank: producer -> custom op -> HCCL All2All -> stable combine

Each timed sample is synchronized with `torch.npu.synchronize()`. Two-rank
samples additionally use HCCL barriers and take the maximum rank elapsed time.

Positive rule, fixed before measurement:

```text
at least 75% of paired blocks favor candidate
AND
paired median improvement > max(1%, 2 * relative_noise_floor)
```

The noise floor is the maximum of:

- median absolute relative base-A/base-B null-pair difference;
- absolute pre/post baseline median drift.

## Recorded results

| Interval | Baseline median | Candidate median | Paired median effect | Noise floor | Classification |
|---|---:|---:|---:|---:|---|
| local composition | 310.342 us | 242.750 us | 21.655% faster | 4.237% | positive, 8/8 pairs |
| world-size-one producer -> pack | 435.846 us | 375.199 us | 12.945% faster | 5.847% | positive, 8/8 pairs |
| same-host two-rank producer -> HCCL -> combine | 5.043 ms | 5.068 ms | 0.385% paired effect in favor of candidate | 4.410% | neutral, 4/8 pairs |

For the two-rank lane, the arm medians and paired median point in slightly
different directions, but both differences are far below the 8.819% decision
threshold. It is therefore classified as **neutral**, not as a speedup or a
regression.

These are local/laboratory interval measurements, not end-to-end model or
production performance claims.

## Recompute the reported statistics

No NPU is required to audit the statistical reduction:

```bash
python3 reduce_results.py
```

The script reads the sanitized raw per-block JSON files and asserts the recorded
medians, paired effects, null-control noise floors, and decision thresholds.

## Rerun the benchmark

See `SOURCE_LINEAGE.md`. Reconstruct the candidate by applying:

```bash
git checkout e8f47fc11c81f3eb3efeb9b40400db8b0fa6eef3
gzip -dc /path/to/inputs/SFA-LSE-R013-C01-BA07.patch.gz | git apply
```

Use separate baseline and candidate Python environments built from the two source
trees, with the same public vLLM reference commit and an Ascend runtime matching
the recorded lineage. Then export:

```bash
export BASE_PYTHON=/path/to/base/venv/bin/python
export CANDIDATE_PYTHON=/path/to/candidate/venv/bin/python
export BASE_SOURCE=/path/to/base/vllm-ascend
export CANDIDATE_SOURCE=/path/to/candidate/vllm-ascend
export PYTHONPATH_EXTRA=/path/to/vllm-ba07
bash harness/run_protocol_public.sh
```

The wrapper preserves the original fixed A/B/null order, seeds, shapes, frozen
iteration counts and synchronization protocol, without depending on the internal
experiment control plane.

Exact absolute latency is not expected to reproduce bit-for-bit across different
hardware/runtime states. The reproducibility target is the paired effect relative
to the freshly observed null-control noise floor under matched lineage and the
same protocol.

## Contents

- `preregistration.json`: fixed protocol and decision rules
- `inputs/fixture.bin`: exact correctness fixture
- `inputs/SFA-LSE-R013-C01-BA07.patch.gz`: gzip-compressed exact candidate reconstruction patch
- `harness/run_single_perf.py`: single-rank local/world-size-one runner
- `harness/run_two_rank_perf.py`: same-host two-rank HCCL runner
- `harness/r014_single_rank_reference.py`: independent correctness reference
- `harness/run_protocol_public.sh`: public orchestration wrapper
- `results/measurements/`: sanitized raw measurements
- `results/calibration/`: calibration/frozen-iteration records
- `results/sentinels/`: sanitized correctness sentinels
- `reduce_results.py`: independent statistical reducer
- `SOURCE_LINEAGE.md`: source/runtime lineage
- `SHA256SUMS`: packet integrity manifest
