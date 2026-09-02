#!/usr/bin/env bash
set -Eeuo pipefail

# Public wrapper for the original R016 preregistered ordering. It intentionally
# has no dependency on the internal control plane or Docker labels.
# Required environment variables:
#   BASE_PYTHON, CANDIDATE_PYTHON
#   BASE_SOURCE, CANDIDATE_SOURCE
# Optional:
#   PYTHONPATH_EXTRA (for the matching public vLLM reference checkout)

: "${BASE_PYTHON:?set BASE_PYTHON}"
: "${CANDIDATE_PYTHON:?set CANDIDATE_PYTHON}"
: "${BASE_SOURCE:?set BASE_SOURCE}"
: "${CANDIDATE_SOURCE:?set CANDIDATE_SOURCE}"
PYTHONPATH_EXTRA=${PYTHONPATH_EXTRA:-}

mkdir -p output/measurements/{primary,world1,two-rank} output/calibration output/sentinels

run_single() {
  local arm=$1 logical=$2 path=$3 mode=$4 output=$5 iterations=$6 block=$7
  local python source
  if [[ "$arm" == base ]]; then python=$BASE_PYTHON; source=$BASE_SOURCE; else python=$CANDIDATE_PYTHON; source=$CANDIDATE_SOURCE; fi
  PYTHONPATH="$PYTHONPATH_EXTRA${PYTHONPATH_EXTRA:+:}${PYTHONPATH:-}" \
    "$python" harness/run_single_perf.py \
      --arm "$arm" --logical-arm "$logical" --path "$path" --mode "$mode" \
      --output "$output" --source-root "$source" --fixture inputs/fixture.bin \
      --iterations "$iterations" --warmup 10 --samples 5 --target-ms 250 \
      --max-iterations 65536 --block "$block"
}

run_two() {
  local arm=$1 logical=$2 mode=$3 output=$4 iterations=$5 block=$6
  local python source
  if [[ "$arm" == base ]]; then python=$BASE_PYTHON; source=$BASE_SOURCE; else python=$CANDIDATE_PYTHON; source=$CANDIDATE_SOURCE; fi
  PYTHONPATH="$PYTHONPATH_EXTRA${PYTHONPATH_EXTRA:+:}${PYTHONPATH:-}" \
    "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
      harness/run_two_rank_perf.py \
      --arm "$arm" --logical-arm "$logical" --mode "$mode" --output "$output" \
      --source-root "$source" --fixture inputs/fixture.bin --iterations "$iterations" \
      --warmup 10 --samples 5 --target-ms 250 --max-iterations 65536 --block "$block"
}

main_order=(base candidate candidate base candidate base base candidate base candidate candidate base candidate base base candidate)
null_order=(base_a base_b base_b base_a base_a base_b base_b base_a)

# Frozen calibration from the recorded protocol.
primary_iterations=1062
world1_iterations=822
two_rank_iterations=92

for path in primary world1; do
  if [[ "$path" == primary ]]; then iterations=$primary_iterations; else iterations=$world1_iterations; fi
  for i in {0..7}; do
    logical=
    run_single base "$logical" "$path" measure "output/measurements/$path/null-pre-$i-$logical.json" "$iterations" "$((i/2))"
  done
  for i in {0..15}; do
    arm=
    run_single "$arm" "$arm" "$path" measure "output/measurements/$path/main-$i-$arm.json" "$iterations" "$((i/2))"
  done
  for i in {0..7}; do
    logical=
    run_single base "$logical" "$path" measure "output/measurements/$path/null-post-$i-$logical.json" "$iterations" "$((i/2))"
  done
done

for i in {0..7}; do
  logical=
  run_two base "$logical" measure "output/measurements/two-rank/null-pre-$i-$logical.json" "$two_rank_iterations" "$((i/2))"
done
for i in {0..15}; do
  arm=
  run_two "$arm" "$arm" measure "output/measurements/two-rank/main-$i-$arm.json" "$two_rank_iterations" "$((i/2))"
done
for i in {0..7}; do
  logical=
  run_two base "$logical" measure "output/measurements/two-rank/null-post-$i-$logical.json" "$two_rank_iterations" "$((i/2))"
done
