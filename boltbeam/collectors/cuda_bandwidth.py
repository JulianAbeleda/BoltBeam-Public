"""Read-only GPU memory speed of one NVIDIA GPU, with a small native CUDA kernel through BoltBeam's CUDA bridge.

    python -m boltbeam.collectors.cuda_bandwidth [--device N] [--gib 1] [--reps 10]

Decode reads the weights once per token, so the speed limit needs the READ rate. tinygrad's sum reaches only about
22% of a 5090's bandwidth (382 GB/s against about 1,700), so it is no measure of bandwidth. This kernel reads a
buffer of at least 1 GiB with 16-byte vector loads in a grid-stride loop and writes one value per block. The kernel
is compiled at run time by the bridge (runtime/cuda_device.py: nvcc -cubin -arch=native, cached) and each launch is
timed between CUDA events there; the launches run through the one loop (collectors/kernel_timer.py samples) and
this probe keeps the best of N per launch shape, the rule metal_bandwidth.py and the targets registry use for every
measured bandwidth.

measure_matrix_tflops is the compute side of a chip profile: fp16 WMMA 16x16x16 multiply-accumulate into fp32 on
fragments held in registers, so no memory is read. It is a lower bound on the tensor cores (WMMA, not the newest
instructions), and it only bounds prefill: decode is memory-bound. The driver's device facts (SM count, shared
memory per SM, L2, clocks, bus width) come from the bridge's facts().
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from typing import Any

from boltbeam.collectors.kernel_timer import samples
from boltbeam.runtime.cuda_device import Cuda, find_nvcc  # noqa: F401  (find_nvcc: the one place nvcc is looked up)

CUDA_SRC = r"""
extern "C" __global__ void readsum(const int4* __restrict__ a, unsigned long long n4, float* out) {
  int acc = 0;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n4;
       i += (unsigned long long)gridDim.x * blockDim.x) {
    int4 v = a[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  __shared__ int s[1024];
  s[threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    int r = 0;
    for (int k = 0; k < blockDim.x; k++) r ^= s[k];
    out[blockIdx.x] = (float)r;  // one write per block, so nothing is optimised away
  }
}
"""
MATRIX_SRC = r"""
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
#define ACC 8

extern "C" __global__ void mmaloop(float* out, int iters) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[ACC];
  wmma::fill_fragment(a, __float2half(0.001f));
  wmma::fill_fragment(b, __float2half(0.002f));
#pragma unroll
  for (int k = 0; k < ACC; k++) wmma::fill_fragment(c[k], 0.0f);
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int k = 0; k < ACC; k++) wmma::mma_sync(c[k], a, b, c[k]);
  }
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < ACC; k++)
    for (int t = 0; t < c[k].num_elements; t++) s += c[k].x[t];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // every result is kept, so nothing is optimised away
}
"""
READ_SHAPES = ((256, 4), (256, 8), (256, 16), (512, 4), (512, 8), (1024, 2), (1024, 4))  # threads, blocks per SM
MATRIX_SHAPES = ((128, 4), (256, 2), (256, 4), (512, 2))
MATRIX_ACC = 8
MATRIX_NOTE = "lower bound, prefill only; decode is memory-bound"
MATRIX_METHOD = ("BoltBeam cuda_bandwidth: native CUDA WMMA 16x16x16 fp16 multiply-accumulate into fp32, 8 accumulators "
                 "in registers, {iters} steps, best of {reps} per launch shape")
METHOD = "BoltBeam cuda_bandwidth: native CUDA read-only kernel, 16-byte loads over {gib} GiB, best of {reps} per launch shape"
SUSTAIN_S = 5.0  # about the length of a decode capture
FACT_KEYS = ("sm_count", "shared_mem_per_sm_bytes", "l2_cache_bytes", "clock_khz", "memory_clock_khz", "memory_bus_bits",
             "total_global_mem_bytes")


def plausibility(cold:float, shapes:list[float], sustained:float | None) -> dict[str, Any]:
  """The chip's read ceiling and its one plausibility band. The ceiling is roofline_ceiling.resolve_achieved_ceiling's
  pick between the cold burst and the sustained series (sustained when the clocks ramp or throttle, the larger when
  flat). The band is the larger of the probe's spread over its launch shapes, the cold to sustained difference, and
  1%: a measured kernel within it of the floor is at the limit, further below is refused."""
  from boltbeam.roofline.roofline_ceiling import resolve_achieved_ceiling
  res = resolve_achieved_ceiling(cold_samples=[cold], sustained_samples=[sustained] if sustained else None)
  spread = (max(shapes) - min(shapes)) / max(shapes) if shapes else 0.0
  drift = abs(sustained - cold) / sustained if sustained else 0.0
  return {"read_gbs": res.achieved_gbs, "regime": res.regime, "cold_gbs": cold, "sustained_gbs": sustained,
          "spread": round(spread, 4), "drift": round(drift, 4), "band": round(max(spread, drift, 0.01), 4)}


def measure_read_gbs(device:int = 0, *, gib:int = 1, reps:int = 10, nvcc:str | None = None, bridge=None,
                     sustain_s:float = SUSTAIN_S) -> dict[str, Any]:
  """Read bandwidth of one GPU: best of `reps` per launch shape, then the best shape back to back for sustain_s."""
  cuda = bridge or Cuda(device, nvcc=nvcc)
  try:
    pso = cuda.pipeline(cuda.library(CUDA_SRC), "readsum")["pso"]
    sms = cuda.facts()["sm_count"] or 1
    nbytes = (gib << 30) // 16 * 16
    a, out = cuda.buffer(length=nbytes), cuda.buffer(length=1024 * 64 * sms * 4)
    n4 = struct.pack("<Q", nbytes // 16)
    rows = []
    for threads, per_sm in READ_SHAPES:
      blocks = per_sm * sms
      launch = lambda: cuda.dispatch(pso, [a, n4, out], (blocks, 1, 1), (threads, 1, 1))  # noqa: E731
      best = min(samples(launch, warmups=1, count=reps))  # the one loop; a bandwidth keeps its best launch
      rows.append({"threads": threads, "blocks": blocks, "best_ms": best / 1000.0, "gbs": nbytes / (best * 1e-6) / 1e9})
    best = max(rows, key=lambda r: r["gbs"])
    # sustained: the best shape back to back for sustain_s seconds, as a decode reads (the clocks settle under load)
    series, done = [], 0.0
    while not series or done < sustain_s * 1e6:  # at least one launch, then until the seconds are up
      us = cuda.dispatch(pso, [a, n4, out], (best["blocks"], 1, 1), (best["threads"], 1, 1))
      series.append(nbytes / (us * 1e-6) / 1e9)
      done += us
    result = {"device": device, "name": cuda.name, "bytes": nbytes, "reps": reps, "shapes": rows,
              "best_shape": {"threads": best["threads"], "blocks": best["blocks"]},
              "sustained": {"seconds": done / 1e6, "launches": len(series), "best_gbs": round(max(series), 1),
                            "mean_gbs": round(sum(series) / len(series), 1), "min_gbs": round(min(series), 1)}}
  finally:
    if bridge is None:
      cuda.close()
  result.update(plausibility(round(best["gbs"], 1), [r["gbs"] for r in rows], result["sustained"]["best_gbs"]))
  result.update(schema="boltbeam.cuda_read_bandwidth.v1", method=METHOD.format(gib=gib, reps=reps) +
                f"; then the best shape back to back for {sustain_s:.0f} s (sustained)")
  return result


def measure_matrix_tflops(device:int = 0, *, iters:int = 8192, reps:int = 5, nvcc:str | None = None, bridge=None) -> dict[str, Any]:
  """fp16 tensor-core rate in TFLOP/s (a lower bound) and the driver's device facts, through the bridge."""
  cuda = bridge or Cuda(device, nvcc=nvcc)
  try:
    pso = cuda.pipeline(cuda.library(MATRIX_SRC), "mmaloop")["pso"]
    facts = cuda.facts()
    sms = facts["sm_count"] or 1
    out = cuda.buffer(length=512 * 8 * sms * 4)
    rows = []
    for threads, per_sm in MATRIX_SHAPES:
      blocks = per_sm * sms
      launch = lambda: cuda.dispatch(pso, [out, struct.pack("<i", iters)], (blocks, 1, 1), (threads, 1, 1))  # noqa: E731
      best = min(samples(launch, warmups=1, count=reps))
      flop = blocks * (threads // 32) * iters * MATRIX_ACC * 2.0 * 16 * 16 * 16
      rows.append({"threads": threads, "blocks": blocks, "best_ms": best / 1000.0, "tflops": flop / (best * 1e-6) / 1e12})
    result = {"device": device, "name": cuda.name, **{k: facts.get(k) for k in FACT_KEYS}, "iters": iters, "reps": reps,
              "shapes": rows, "tflops": round(max(r["tflops"] for r in rows), 2)}
  finally:
    if bridge is None:
      cuda.close()
  result.update(schema="boltbeam.cuda_matrix_peak.v1", dtype="fp16", note=MATRIX_NOTE,
                method=MATRIX_METHOD.format(iters=iters, reps=reps))
  return result


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog="python -m boltbeam.collectors.cuda_bandwidth")
  p.add_argument("--device", type=int, default=0)
  p.add_argument("--gib", type=int, default=1)
  p.add_argument("--reps", type=int, default=10)
  a = p.parse_args(argv)
  sys.stdout.write(json.dumps(measure_read_gbs(a.device, gib=a.gib, reps=a.reps), indent=1, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
