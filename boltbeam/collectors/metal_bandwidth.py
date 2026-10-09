"""Read-only GPU memory speed on this Mac, through BoltBeam's own Metal bridge.

Decode reads the weights once per token and writes almost nothing, so the speed limit needs the READ rate.
A copy probe (read plus write) under-measures it: on the M4 it gave 90.4 GB/s while llama.cpp decode
already read 95 GB/s. This kernel reads 1 GiB once per dispatch (far past the system cache) and writes
one float per simdgroup. It sweeps a few launch shapes and keeps the best of five per shape, the same
"best of five" rule the targets registry records for every measured bandwidth.
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


def run_cli(args) -> int:
  out = measure_read_gbs()
  sys.stdout.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
  return 0
