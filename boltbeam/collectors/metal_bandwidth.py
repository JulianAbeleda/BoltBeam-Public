"""Read-only GPU memory speed on this Mac, through BoltBeam's own Metal bridge.

Decode reads the weights once per token and writes almost nothing, so the speed limit needs the READ rate.
A copy probe (read plus write) under-measures it: on the M4 it gave 90.4 GB/s while llama.cpp decode
already read 95 GB/s. This kernel reads 1 GiB once per dispatch (far past the system cache) and writes
one float per simdgroup. It sweeps a few launch shapes and keeps the best of five per shape, the same
"best of five" rule the targets registry records for every measured bandwidth.

measure_matrix_tflops is the compute side of a chip profile: fp16 matrix multiply-accumulate (simdgroup 8x8, fp32
accumulate) on values held in registers, so no memory is read. It is a lower bound on the matrix unit, and it only
bounds prefill: decode is memory-bound.
"""
from __future__ import annotations

import json
import struct
import sys

from boltbeam.runtime.metal_device import Metal

_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void readsum(device const float4* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                    uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint tg [[threadgroup_position_in_grid]],
                    uint sgs [[simdgroups_per_threadgroup]]) {
  float4 acc = 0;
  for (uint i = gid; i < n4; i += gsz) acc += a[i];
  float s = simd_sum(acc.x + acc.y + acc.z + acc.w);
  if (lane == 0) out[tg * sgs + sg] = s;
}
"""
SHAPES = ((256, 1024), (256, 4096), (256, 16384), (1024, 1024), (1024, 4096), (1024, 16384))


def measure_read_gbs(nbytes:int = 1 << 30, reps:int = 5) -> dict:
  metal = Metal()
  try:
    pso = metal.compile(_SRC, ["readsum"])["readsum"]["pso"]
    src = metal.buffer(length=nbytes)
    rows = []
    for threads, groups in SHAPES:
      out = metal.buffer(length=groups * threads // 32 * 4)
      metal.run(pso, [src, out], [struct.pack("I", nbytes // 16)], groups, threads)  # warm-up, not counted
      us = [metal.run(pso, [src, out], [struct.pack("I", nbytes // 16)], groups, threads) for _ in range(reps)]
      rows.append({"threads": threads, "groups": groups, "best_us": min(us), "gbs": nbytes / (min(us) * 1e-6) / 1e9})
    best = max(rows, key=lambda r: r["gbs"])
    return {"schema": "boltbeam.metal_read_bandwidth.v1", "device": metal.name, "bytes": nbytes, "reps": reps,
            "method": "BoltBeam metal_bandwidth: read-only float4 sum over 1 GiB, best of five per launch shape",
            "read_gbs": round(best["gbs"], 1), "best_shape": {"threads": best["threads"], "groups": best["groups"]},
            "shapes": rows}
  finally:
    metal.close()


_MMA_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void mma(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint sgs [[simdgroups_per_threadgroup]]) {
  simdgroup_half8x8 a(half(0.001)), b(half(0.002));
  simdgroup_float8x8 c0(0.0f), c1(0.0f), c2(0.0f), c3(0.0f), c4(0.0f), c5(0.0f), c6(0.0f), c7(0.0f);
  for (uint i = 0; i < iters; i++) {
    simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1);
    simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);
    simdgroup_multiply_accumulate(c4, a, b, c4); simdgroup_multiply_accumulate(c5, a, b, c5);
    simdgroup_multiply_accumulate(c6, a, b, c6); simdgroup_multiply_accumulate(c7, a, b, c7);
  }
  device float* o = out + (tg * sgs + sg) * 64;
  simdgroup_store(c0, o, 8); simdgroup_store(c1, o, 8); simdgroup_store(c2, o, 8); simdgroup_store(c3, o, 8);
  simdgroup_store(c4, o, 8); simdgroup_store(c5, o, 8); simdgroup_store(c6, o, 8); simdgroup_store(c7, o, 8);
}
"""
MMA_ACCUMULATORS = 8
MMA_FLOP = 2 * 8 * 8 * 8  # one 8x8x8 multiply-accumulate
MMA_SHAPES = ((256, 64), (256, 256), (1024, 64), (1024, 256))  # threads, threadgroups
MATRIX_NOTE = "lower bound, prefill only; decode is memory-bound"


def measure_matrix_tflops(iters:int = 4096, reps:int = 5) -> dict:
  """fp16 matrix multiply-accumulate rate, best of `reps` per launch shape, in TFLOP/s."""
  metal = Metal()
  try:
    pso = metal.compile(_MMA_SRC, ["mma"])["mma"]["pso"]
    rows = []
    for threads, groups in MMA_SHAPES:
      out = metal.buffer(length=groups * threads // 32 * 64 * 4)
      metal.run(pso, [out], [struct.pack("I", iters)], groups, threads)  # warm-up, not counted
      us = min(metal.run(pso, [out], [struct.pack("I", iters)], groups, threads) for _ in range(reps))
      flop = groups * threads // 32 * iters * MMA_ACCUMULATORS * MMA_FLOP
      rows.append({"threads": threads, "groups": groups, "best_us": us, "tflops": flop / (us * 1e-6) / 1e12})
    best = max(rows, key=lambda r: r["tflops"])
    return {"schema": "boltbeam.metal_matrix_peak.v1", "device": metal.name, "dtype": "fp16",
            "method": f"BoltBeam metal_bandwidth: simdgroup 8x8 fp16 multiply-accumulate into fp32, {MMA_ACCUMULATORS} "
                      f"accumulators in registers, {iters} steps, best of {reps} per launch shape",
            "tflops": round(best["tflops"], 3), "note": MATRIX_NOTE, "shapes": rows,
            "working_set_bytes": metal.working_set_bytes}
  finally:
    metal.close()


def run_cli(args) -> int:
  out = measure_read_gbs()
  sys.stdout.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
  return 0
