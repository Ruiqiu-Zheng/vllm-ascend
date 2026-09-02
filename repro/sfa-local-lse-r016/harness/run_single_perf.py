#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=("base", "candidate"), required=True)
parser.add_argument("--path", choices=("primary", "world1"), required=True)
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

os.environ.setdefault("TRITON_CACHE_DIR", f"/workspace/run/cache/{args.arm}")

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
from vllm_ascend.utils import enable_custom_op  # noqa: E402

enable_custom_op()
import vllm_ascend.ops  # noqa: E402,F401
from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402
from vllm_ascend.ops.triton import sfa_cp as sfa_ops  # noqa: E402


DCP_SIZE = 2
NUM_TOKENS = 4
NUM_HEADS = 8
HEAD_DIM = 512
ROPE_DIM = 64
SEQ_LEN = 128
SPARSE_COUNT = 2048
DTYPE = torch.bfloat16
SEED = 20260913


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
    q: torch.Tensor
    q_rope: torch.Tensor
    k: torch.Tensor
    k_rope: torch.Tensor
    indices: torch.Tensor
    block_table: torch.Tensor
    actual_q: torch.Tensor
    actual_k: torch.Tensor


def make_fixture() -> ProducerFixture:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn((NUM_TOKENS, NUM_HEADS, HEAD_DIM), generator=generator, dtype=torch.float32).to(DTYPE)
    q_rope = torch.randn((NUM_TOKENS, NUM_HEADS, ROPE_DIM), generator=generator, dtype=torch.float32).to(DTYPE)
    k = torch.randn((1, SEQ_LEN, 1, HEAD_DIM), generator=generator, dtype=torch.float32).to(DTYPE)
    k_rope = torch.randn((1, SEQ_LEN, 1, ROPE_DIM), generator=generator, dtype=torch.float32).to(DTYPE)
    indices = torch.full((NUM_TOKENS, 1, SPARSE_COUNT), -1, dtype=torch.int32)
    indices[:, 0, :SEQ_LEN] = torch.arange(SEQ_LEN, dtype=torch.int32)
    return ProducerFixture(
        q=q.npu(), q_rope=q_rope.npu(), k=k.npu(), k_rope=k_rope.npu(), indices=indices.npu(),
        block_table=torch.tensor([[0]], dtype=torch.int32, device="npu"),
        actual_q=torch.tensor([NUM_TOKENS], dtype=torch.int32, device="npu"),
        actual_k=torch.tensor([SEQ_LEN], dtype=torch.int32, device="npu"),
    )


def run_producer(fixture: ProducerFixture) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = DeviceOperator.execute_sparse_flash_attention_process(
        SimpleNamespace(scale=1.0 / math.sqrt(HEAD_DIM + ROPE_DIM)),
        fixture.q, fixture.q_rope, (fixture.k, fixture.k_rope), fixture.indices,
        SimpleNamespace(block_table=fixture.block_table), fixture.actual_q, fixture.actual_k,
        sparse_mode=0, return_lse=True,
    )
    assert isinstance(result, tuple) and len(result) == 3
    output, maximum, summation = result
    assert output.shape == (NUM_TOKENS, NUM_HEADS, HEAD_DIM)
    assert maximum.shape == summation.shape == (1, NUM_TOKENS, NUM_HEADS)
    return output, maximum, summation


def pack(arm: str, output: torch.Tensor, maximum: torch.Tensor, summation: torch.Tensor) -> torch.Tensor:
    if arm == "base":
        lse = (maximum + torch.log(summation)).permute(1, 2, 0).reshape(NUM_TOKENS, NUM_HEADS, 1)
        return sfa_ops.pack_sfa_dcp_output_lse(output, lse, DCP_SIZE, 0)
    return sfa_ops.pack_sfa_dcp_output_max_sum(output, maximum, summation, DCP_SIZE, 0)


def oracle(fixture: ProducerFixture, output: torch.Tensor, maximum: torch.Tensor, summation: torch.Tensor) -> dict[str, Any]:
    scale = 1.0 / math.sqrt(HEAD_DIM + ROPE_DIM)
    q = torch.cat((fixture.q.cpu(), fixture.q_rope.cpu()), dim=-1).float()
    k = torch.cat((fixture.k.cpu(), fixture.k_rope.cpu()), dim=-1).float()[0, :, 0]
    value = fixture.k.cpu().float()[0, :, 0]
    scores = torch.einsum("thd,sd->ths", q, k) * scale
    reference_lse = torch.logsumexp(scores, dim=-1)
    reference_output = torch.einsum("ths,sd->thd", torch.softmax(scores, dim=-1), value).to(DTYPE)
    actual_lse = (maximum.float() + torch.log(summation.float())).permute(1, 0, 2).reshape(NUM_TOKENS, NUM_HEADS)
    torch.testing.assert_close(output.cpu(), reference_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse.cpu(), reference_lse, rtol=1e-4, atol=1e-4)
    return {
        "status": "PASS_R014_INDEPENDENT_FP32_ORACLE",
        "output_max_abs_fp32_error": float((output.cpu().float() - reference_output.float()).abs().max()),
        "lse_max_abs_fp32_error": float((actual_lse.cpu() - reference_lse).abs().max()),
        "reference_lse_sha256": tensor_digest(reference_lse),
    }


def fixture_identity(fixture: ProducerFixture) -> dict[str, Any]:
    return {
        "seed": SEED,
        "dtype": str(DTYPE),
        "shape": {"tokens": NUM_TOKENS, "heads": NUM_HEADS, "head_dim": HEAD_DIM, "rope_dim": ROPE_DIM, "seq_len": SEQ_LEN, "sparse_count": SPARSE_COUNT},
        "tensor_sha256": {name: tensor_digest(getattr(fixture, name)) for name in ("q", "q_rope", "k", "k_rope", "indices", "block_table", "actual_q", "actual_k")},
    }


def mapped_native_objects() -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        item = fields[-1]
        if item.startswith("/") and (item.endswith(".so") or ".so." in item):
            path = Path(item)
            if path.exists() and any(key in item.lower() for key in ("vllm", "cust_op", "ascend", "triton")):
                paths.add(path)
    return [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(paths)]


def schema() -> dict[str, Any]:
    name = "vllm::sfa_dcp_a2a_fused" if args.arm == "base" else "vllm::sfa_dcp_a2a_fused_max_sum"
    handle = torch._C._dispatch_find_schema_or_throw(name, "")
    return {
        "name": name, "schema": str(handle.schema()),
        "PrivateUse1": torch._C._dispatch_has_kernel_for_dispatch_key(name, "PrivateUse1"),
        "Meta": torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta"),
    }


def source_and_runtime() -> dict[str, Any]:
    imported = {
        "vllm_ascend": Path(sys.modules["vllm_ascend"].__file__).resolve(),
        "sfa_ops": Path(sfa_ops.__file__).resolve(),
        "device_op": Path(sys.modules["vllm_ascend.device.device_op"].__file__).resolve(),
    }
    source = {
        "pack": args.source_root / "vllm_ascend/ops/triton/sfa_cp.py",
        "context": args.source_root / "vllm_ascend/attention/context_parallel/sfa_cp.py",
        "producer": args.source_root / "vllm_ascend/device/device_op.py",
    }
    return {
        "source": {k: {"path": str(v), "sha256": sha256(v)} for k, v in source.items()},
        "imported_modules": {k: {"path": str(v), "sha256": sha256(v)} for k, v in imported.items()},
        "schema": schema(),
        "runtime": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "cann": os.environ.get("ASCEND_HOME_PATH"),
            "triton_cache": os.environ["TRITON_CACHE_DIR"], "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        },
        "loaded_native_objects": mapped_native_objects(),
    }


def timed_sample(function: Callable[[], torch.Tensor], iterations: int) -> dict[str, Any]:
    torch.npu.synchronize()
    started_wall = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter_ns()
    last = None
    for _ in range(iterations):
        last = function()
    torch.npu.synchronize()
    elapsed = time.perf_counter_ns() - start
    assert last is not None
    return {"started_at": started_wall, "elapsed_ns": elapsed, "iterations": iterations, "per_iteration_ns": elapsed / iterations, "last_output_sha256": tensor_digest(last)}


def main() -> None:
    torch.npu.set_device(0)
    fixture = make_fixture()
    output, maximum, summation = run_producer(fixture)
    torch.npu.synchronize()

    def primary() -> torch.Tensor:
        return pack(args.arm, output, maximum, summation)

    def world1() -> torch.Tensor:
        current_output, current_maximum, current_sum = run_producer(fixture)
        return pack(args.arm, current_output, current_maximum, current_sum)

    function = primary if args.path == "primary" else world1
    for _ in range(args.warmup):
        function()
    torch.npu.synchronize()

    packed = pack(args.arm, output, maximum, summation)
    torch.npu.synchronize()
    correctness = {
        "oracle": oracle(fixture, output, maximum, summation),
        "packed_sha256": tensor_digest(packed),
        "packed_shape": list(packed.shape),
        "packed_dtype": str(packed.dtype),
        "producer_output_sha256": tensor_digest(output),
        "maximum_sha256": tensor_digest(maximum),
        "summation_sha256": tensor_digest(summation),
        "finite_positive_statistics": bool(torch.isfinite(maximum).all() and (torch.isfinite(summation) & (summation > 0)).all()),
    }
    result: dict[str, Any] = {
        "schema": "sfa-r016-single-arm-v1", "arm": args.arm, "logical_arm": args.logical_arm or args.arm,
        "path": args.path, "mode": args.mode, "block": args.block,
        "recorded_at": datetime.now(timezone.utc).isoformat(), "warmup_iterations": args.warmup,
        "fixture_file": {"path": str(args.fixture), "bytes": args.fixture.stat().st_size, "sha256": sha256(args.fixture)},
        "producer_fixture": fixture_identity(fixture), "correctness": correctness, **source_and_runtime(),
    }
    if args.mode == "sentinel":
        result["status"] = "PASS_SENTINEL"
        result["samples"] = []
    elif args.mode == "calibrate":
        steps = []
        iterations = max(1, args.iterations)
        while True:
            sample = timed_sample(function, iterations)
            steps.append(sample)
            if sample["elapsed_ns"] >= args.target_ms * 1_000_000 or iterations >= args.max_iterations:
                break
            scale = max(2, math.ceil(args.target_ms * 1_000_000 / max(sample["elapsed_ns"], 1)))
            iterations = min(args.max_iterations, iterations * scale)
        result.update({"status": "PASS_CALIBRATION", "target_ms": args.target_ms, "selected_iterations": iterations, "calibration_steps": steps, "samples": []})
    else:
        assert args.iterations > 0
        samples = [timed_sample(function, args.iterations) for _ in range(args.samples)]
        per_iter = [item["per_iteration_ns"] for item in samples]
        result.update({
            "status": "PASS_MEASUREMENT_BLOCK", "iterations": args.iterations, "samples": samples,
            "block_median_ns": statistics.median(per_iter), "block_min_ns": min(per_iter), "block_max_ns": max(per_iter),
        })
    dump(args.output, result)
    print(json.dumps({"status": result["status"], "arm": args.arm, "path": args.path, "block": args.block, "output": str(args.output), "sha256": sha256(args.output), "selected_iterations": result.get("selected_iterations"), "block_median_ns": result.get("block_median_ns")}, sort_keys=True))


if __name__ == "__main__":
    main()
