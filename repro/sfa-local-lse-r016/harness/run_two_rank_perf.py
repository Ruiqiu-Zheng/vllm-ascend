#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable
import weakref


parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=("base", "candidate"), required=True)
parser.add_argument("--mode", choices=("sentinel", "calibrate", "measure"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--iterations", type=int, default=1)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--samples", type=int, default=5)
parser.add_argument("--target-ms", type=float, default=250.0)
parser.add_argument("--max-iterations", type=int, default=65536)
parser.add_argument("--block", type=int, default=-1)
parser.add_argument("--logical-arm", default="")
args = parser.parse_args()

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
os.environ.setdefault("TRITON_CACHE_DIR", f"/workspace/run/cache/{args.arm}/two-rank/rank{RANK}")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: E402,F401
from vllm.distributed.parallel_state import _groups  # noqa: E402
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl  # noqa: E402
from vllm_ascend.utils import enable_custom_op  # noqa: E402

enable_custom_op()
import vllm_ascend.ops  # noqa: E402,F401

sys.path.insert(0, "/workspace/harness")
from r014_single_rank_reference import independent_merge, make_producer_fixture, run_producer, sha256, tensor_digest  # noqa: E402


GROUP_NAME = "sfa-r016-hccl:0"
DCP_SIZE = 2
NUM_TOKENS = 4
NUM_HEADS = 8
HEAD_DIM = 512
FIXTURE_SHA256 = "63785fa8f4592b7f8f9e06d345a20cf29a4ac8d19adc1e89dc8f90e60eb7db19"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class HarnessGroup:
    def __init__(self, device_group: dist.ProcessGroup):
        self.device_group = device_group
        self.world_size = dist.get_world_size(device_group)
        self.rank = dist.get_rank(device_group)
        self.unique_name = GROUP_NAME


def make_impl(group: HarnessGroup) -> AscendSFADCPImpl:
    impl = object.__new__(AscendSFADCPImpl)
    impl.dcp_group = group
    impl.dcp_size = DCP_SIZE
    impl.dcp_rank = RANK
    return impl


def dsa_context(scatter_dim: int, num_tokens: int) -> SimpleNamespace | None:
    if scatter_dim == 1:
        return None
    local = num_tokens // DCP_SIZE
    return SimpleNamespace(num_tokens_pad=num_tokens, local_start=RANK * local, local_end_with_pad=(RANK + 1) * local)


def merge(impl: AscendSFADCPImpl, output: torch.Tensor, maximum: torch.Tensor, summation: torch.Tensor, scatter_dim: int) -> torch.Tensor:
    context = dsa_context(scatter_dim, output.shape[0])
    if args.arm == "base":
        lse = (maximum + torch.log(summation)).permute(1, 0, 2).reshape(output.shape[0], output.shape[1], 1)
        return impl._merge_dcp_outputs(output, lse, context)
    return impl._merge_dcp_outputs_max_sum(output, maximum, summation, context)


def raw_tensor(raw: bytes, dtype: str, shape: list[int]) -> torch.Tensor:
    buffer = bytearray(raw)
    if dtype == "bfloat16": value = torch.frombuffer(buffer, dtype=torch.uint16).clone().view(torch.bfloat16)
    elif dtype == "float16": value = torch.frombuffer(buffer, dtype=torch.float16).clone()
    elif dtype == "float32": value = torch.frombuffer(buffer, dtype=torch.float32).clone()
    elif dtype == "int32": value = torch.frombuffer(buffer, dtype=torch.int32).clone()
    else: raise ValueError(dtype)
    return value.reshape(shape)


def load_exact_fixture() -> dict[str, torch.Tensor]:
    assert sha256(args.fixture) == FIXTURE_SHA256
    records: dict[str, torch.Tensor] = {}
    with args.fixture.open("rb") as stream:
        assert stream.read(len(b"SFA-R013-FIXTURE-V1\0")) == b"SFA-R013-FIXTURE-V1\0"
        while True:
            length = stream.read(4)
            if not length: break
            metadata = json.loads(stream.read(struct.unpack("<I", length)[0]))
            raw = stream.read(struct.unpack("<Q", stream.read(8))[0])
            assert hashlib.sha256(raw).hexdigest() == metadata["raw_sha256"]
            records[metadata["name"]] = raw_tensor(raw, metadata["dtype"], metadata["shape"])
    return records


def exact_fixture_sentinel(impl: AscendSFADCPImpl) -> list[dict[str, Any]]:
    records = load_exact_fixture()
    cells = []
    for dtype_name in ("bfloat16", "float16"):
        all_outputs = records[f"{dtype_name}:outputs"].to("npu")
        all_maxima = records[f"{dtype_name}:maxima"].to("npu")
        all_sums = records[f"{dtype_name}:sums"].to("npu")
        output = all_outputs[RANK]
        maximum = all_maxima[RANK].unsqueeze(0)
        summation = all_sums[RANK].unsqueeze(0)
        for scatter_dim in (0, 1):
            actual = merge(impl, output, maximum, summation, scatter_dim)
            torch.npu.synchronize()
            local = all_outputs.shape[1 + scatter_dim] // DCP_SIZE
            if scatter_dim == 0:
                expected = independent_merge(all_outputs[:, RANK * local:(RANK + 1) * local], all_maxima[:, RANK * local:(RANK + 1) * local], all_sums[:, RANK * local:(RANK + 1) * local])
            else:
                expected = independent_merge(all_outputs[:, :, RANK * local:(RANK + 1) * local], all_maxima[:, :, RANK * local:(RANK + 1) * local], all_sums[:, :, RANK * local:(RANK + 1) * local])
            tolerance = 2e-2 if dtype_name == "bfloat16" else 1e-2
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            cells.append({"dtype": dtype_name, "scatter_dim": scatter_dim, "status": "PASS_R014_INDEPENDENT_MERGE", "actual_sha256": tensor_digest(actual), "reference_sha256": tensor_digest(expected), "max_abs_fp32_error": float((actual.cpu().float() - expected.cpu().float()).abs().max()), "tolerance": tolerance})
    return cells


def producer_sentinel(impl: AscendSFADCPImpl) -> list[dict[str, Any]]:
    cells = []
    for dtype in (torch.bfloat16, torch.float16):
        fixture = make_producer_fixture(dtype)
        delta = RANK * 0.03125
        fixture.q.add_(delta); fixture.q_rope.add_(delta); fixture.k.add_(delta * 0.5); fixture.k_rope.add_(delta * 0.5)
        output, maximum, summation, producer = run_producer(fixture)
        gathered_output = [torch.empty_like(output) for _ in range(WORLD_SIZE)]
        gathered_maximum = [torch.empty_like(maximum) for _ in range(WORLD_SIZE)]
        gathered_sum = [torch.empty_like(summation) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered_output, output); dist.all_gather(gathered_maximum, maximum); dist.all_gather(gathered_sum, summation)
        all_outputs = torch.stack(gathered_output)
        all_maxima = torch.stack(gathered_maximum)[:, 0]
        all_sums = torch.stack(gathered_sum)[:, 0]
        for scatter_dim in (0, 1):
            actual = merge(impl, output, maximum, summation, scatter_dim)
            torch.npu.synchronize()
            local = output.shape[scatter_dim] // DCP_SIZE
            if scatter_dim == 0:
                expected = independent_merge(all_outputs[:, RANK * local:(RANK + 1) * local], all_maxima[:, RANK * local:(RANK + 1) * local], all_sums[:, RANK * local:(RANK + 1) * local])
            else:
                expected = independent_merge(all_outputs[:, :, RANK * local:(RANK + 1) * local], all_maxima[:, :, RANK * local:(RANK + 1) * local], all_sums[:, :, RANK * local:(RANK + 1) * local])
            tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            cells.append({"dtype": str(dtype), "scatter_dim": scatter_dim, "status": "PASS_R014_PRODUCER_AND_MERGE_ORACLE", "producer": producer, "actual_sha256": tensor_digest(actual), "reference_sha256": tensor_digest(expected), "max_abs_fp32_error": float((actual.cpu().float() - expected.cpu().float()).abs().max()), "tolerance": tolerance})
    return cells


def timed_sample(function: Callable[[], torch.Tensor], iterations: int) -> dict[str, Any]:
    dist.barrier(); torch.npu.synchronize()
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter_ns()
    last = None
    for _ in range(iterations): last = function()
    torch.npu.synchronize()
    local_elapsed = time.perf_counter_ns() - begin
    elapsed_tensor = torch.tensor([local_elapsed], dtype=torch.int64, device="npu")
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    dist.barrier()
    assert last is not None
    return {"started_at": started, "local_elapsed_ns": local_elapsed, "critical_path_elapsed_ns": int(elapsed_tensor.item()), "iterations": iterations, "per_iteration_ns": int(elapsed_tensor.item()) / iterations, "last_output_sha256": tensor_digest(last)}


def imported_identity() -> dict[str, Any]:
    module_paths = {
        "vllm_ascend": Path(sys.modules["vllm_ascend"].__file__).resolve(),
        "context": Path(sys.modules["vllm_ascend.attention.context_parallel.sfa_cp"].__file__).resolve(),
        "device_op": Path(sys.modules["vllm_ascend.device.device_op"].__file__).resolve(),
    }
    source_paths = {
        "pack": args.source_root / "vllm_ascend/ops/triton/sfa_cp.py",
        "context": args.source_root / "vllm_ascend/attention/context_parallel/sfa_cp.py",
        "producer": args.source_root / "vllm_ascend/device/device_op.py",
    }
    return {
        "source": {name: {"path": str(path), "sha256": sha256(path)} for name, path in source_paths.items()},
        "imported_modules": {name: {"path": str(path), "sha256": sha256(path)} for name, path in module_paths.items()},
        "runtime": {"python": sys.version, "torch": torch.__version__, "torch_npu": torch_npu.__version__, "cann": os.environ.get("ASCEND_HOME_PATH"), "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "triton_cache": os.environ["TRITON_CACHE_DIR"]},
    }


def main() -> None:
    assert WORLD_SIZE == 2
    torch.npu.set_device(LOCAL_RANK)
    dist.init_process_group(backend="hccl")
    group = HarnessGroup(dist.group.WORLD)
    _groups[GROUP_NAME] = weakref.ref(group)
    try:
        impl = make_impl(group)
        fixture = make_producer_fixture(torch.bfloat16)
        delta = RANK * 0.03125
        fixture.q.add_(delta); fixture.q_rope.add_(delta); fixture.k.add_(delta * 0.5); fixture.k_rope.add_(delta * 0.5)

        def full_path() -> torch.Tensor:
            output, maximum, summation, _ = run_producer(fixture)
            return merge(impl, output, maximum, summation, 0)

        for _ in range(args.warmup): full_path()
        dist.barrier(); torch.npu.synchronize()
        result: dict[str, Any] = {
            "schema": "sfa-r016-two-rank-arm-v1", "arm": args.arm, "logical_arm": args.logical_arm or args.arm,
            "path": "two_rank", "mode": args.mode, "block": args.block, "rank": RANK, "local_rank": LOCAL_RANK,
            "recorded_at": datetime.now(timezone.utc).isoformat(), "warmup_iterations": args.warmup,
            "fixture_file": {"path": str(args.fixture), "bytes": args.fixture.stat().st_size, "sha256": sha256(args.fixture)},
            "group": {"name": GROUP_NAME, "world_size": group.world_size, "rank": group.rank, "backend": dist.get_backend()},
            **imported_identity(),
        }
        if args.mode == "sentinel":
            result.update({"status": "PASS_SENTINEL", "producer_cells": producer_sentinel(impl), "exact_fixture_cells": exact_fixture_sentinel(impl), "samples": []})
        elif args.mode == "calibrate":
            steps = []
            iterations = max(1, args.iterations)
            while True:
                sample = timed_sample(full_path, iterations); steps.append(sample)
                if sample["critical_path_elapsed_ns"] >= args.target_ms * 1_000_000 or iterations >= args.max_iterations: break
                scale = max(2, math.ceil(args.target_ms * 1_000_000 / max(sample["critical_path_elapsed_ns"], 1)))
                iterations = min(args.max_iterations, iterations * scale)
            result.update({"status": "PASS_CALIBRATION", "target_ms": args.target_ms, "selected_iterations": iterations, "calibration_steps": steps, "samples": []})
        else:
            samples = [timed_sample(full_path, args.iterations) for _ in range(args.samples)]
            per_iter = [item["per_iteration_ns"] for item in samples]
            result.update({"status": "PASS_MEASUREMENT_BLOCK", "iterations": args.iterations, "samples": samples, "block_median_ns": statistics.median(per_iter), "block_min_ns": min(per_iter), "block_max_ns": max(per_iter)})
        rank_path = args.output.with_name(args.output.stem + f"-rank{RANK}" + args.output.suffix)
        dump(rank_path, result)
        dist.barrier()
        if RANK == 0:
            ranks = []
            for rank in range(WORLD_SIZE):
                path = args.output.with_name(args.output.stem + f"-rank{rank}" + args.output.suffix)
                ranks.append({"rank": rank, "path": str(path), "sha256": sha256(path), "result": json.loads(path.read_text())})
            aggregate = {"schema": "sfa-r016-two-rank-aggregate-v1", "status": result["status"], "arm": args.arm, "logical_arm": args.logical_arm or args.arm, "mode": args.mode, "block": args.block, "ranks": ranks, "selected_iterations": result.get("selected_iterations"), "block_median_ns": result.get("block_median_ns"), "samples": result.get("samples", [])}
            dump(args.output, aggregate)
            print(json.dumps({"status": aggregate["status"], "arm": args.arm, "block": args.block, "output": str(args.output), "sha256": sha256(args.output), "selected_iterations": aggregate.get("selected_iterations"), "block_median_ns": aggregate.get("block_median_ns")}, sort_keys=True))
        dist.barrier()
    finally:
        _groups.pop(GROUP_NAME, None)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
