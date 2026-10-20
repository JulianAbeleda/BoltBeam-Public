"""Read-only GPU memory speed of one NVIDIA or AMD GPU, with a small native kernel through BoltBeam's CUDA or HIP
bridge. The read kernels are plain CUDA C, which hipcc compiles unchanged; the bridge decides the toolchain, and the
result names the API it ran on (cuda_read_bandwidth or hip_read_bandwidth), never one for the other.

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
// one block per contiguous chunk and a grid that covers the buffer once, no grid-stride loop: the shape of a
// matrix-vector kernel that gives each block its own rows. W is the load width (int: 4 bytes, int4: 16 bytes).
#define CHUNK(NAME, W, FOLD) extern "C" __global__ void NAME(const W* __restrict__ a, unsigned long long n, \
    unsigned int per_block, float* out) { \
  int acc = 0; \
  unsigned long long i0 = (unsigned long long)blockIdx.x * per_block; \
  for (unsigned int j = threadIdx.x; j < per_block && i0 + j < n; j += blockDim.x) { W v = a[i0 + j]; acc ^= FOLD; } \
  if (acc == 0x7fffffff) out[blockIdx.x] = (float)acc;  /* a write the compiler cannot rule out, so every load stays */ \
}
CHUNK(readchunk4, int, v)
CHUNK(readchunk16, int4, v.x ^ v.y ^ v.z ^ v.w)
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
# threads, bytes per block, load bytes: one block per chunk, the grid covers the buffer (a launch shape, not a model rule)
CHUNK_SHAPES = ((128, 1024, 16), (128, 4096, 4), (128, 4096, 16), (256, 4096, 4), (256, 16384, 16), (512, 65536, 16))
MATRIX_SHAPES = ((128, 4), (256, 2), (256, 4), (512, 2))
MATRIX_ACC = 8
MATRIX_NOTE = "lower bound, prefill only; decode is memory-bound"
MATRIX_METHOD = ("BoltBeam cuda_bandwidth: native CUDA WMMA 16x16x16 fp16 multiply-accumulate into fp32, 8 accumulators "
                 "in registers, {iters} steps, best of {reps} per launch shape")
METHOD = ("BoltBeam cuda_bandwidth: native {api} read-only kernels over {gib} GiB, a grid-stride loop of 16-byte loads and one "
          "block per contiguous chunk, best of {reps} per launch shape, the launch's fixed cost (the best "
          "shape over half the buffer) removed")
SUSTAIN_S = 5.0  # about the length of a decode capture
FACT_KEYS = ("sm_count", "shared_mem_per_sm_bytes", "l2_cache_bytes", "clock_khz", "memory_clock_khz", "memory_bus_bits",
             "total_global_mem_bytes")


def plausibility(cold:float, shapes:list[float], sustained:float | None) -> dict[str, Any]:
  """The chip's read ceiling from one probe run. The ceiling is roofline_ceiling.resolve_achieved_ceiling's pick between
  the cold burst and the sustained series (sustained when the clocks ramp or throttle, the larger when flat).
  shape_spread is how much the launch shape matters and drift the cold to sustained difference: data to read, not the
  figure's uncertainty, which only repeated runs measure (repeated)."""
  from boltbeam.roofline.roofline_ceiling import resolve_achieved_ceiling
  res = resolve_achieved_ceiling(cold_samples=[cold], sustained_samples=[sustained] if sustained else None)
  drift = abs(sustained - cold) / sustained if sustained else 0.0
  return {"read_gbs": res.achieved_gbs, "regime": res.regime, "cold_gbs": cold, "sustained_gbs": sustained,
          "shape_spread": spread(shapes), "drift": round(drift, 4)}


def spread(xs:list[float]) -> float | None:
  """(max - min) / max, or None for no values."""
  return round((max(xs) - min(xs)) / max(xs), 4) if xs else None


def repeated(runs:list[dict[str, Any]], value:str, shape_value:str) -> dict[str, Any]:
  """Whole probe runs as one figure: the best run. Its uncertainty is the spread of the runs' figures, each taken at
  its run's best launch shape, (max - min) / max with no floor; one run has none. shape_spread, the spread over every
  run's launch shapes, is how much the shape matters: data only, no guard reads it. A failed path stays as it is."""
  if not all(r.get(value) for r in runs):
    return runs[0]
  xs = [r[value] for r in runs]
  return {**max(runs, key=lambda r: r[value]), "runs": len(runs), "run_values": xs,
          "shape_spread": spread([s[shape_value] for r in runs for s in r.get("shapes", [])]),
          **({"uncertainty": spread(xs)} if len(xs) > 1 else {})}


def parallel_units(facts:dict[str, Any]) -> int:
  """How many units a grid is spread over, as the bridge's driver counts them: CUDA's SMs, HIP's multiprocessors
  (work-group processors on RDNA). It sizes launch grids only; it is not a compute-unit count."""
  return facts.get("sm_count") or facts.get("multiprocessor_count") or 1


def measure_read_gbs(device:int = 0, *, gib:int = 1, reps:int = 10, nvcc:str | None = None, bridge=None,
                     sustain_s:float = SUSTAIN_S) -> dict[str, Any]:
  """Read bandwidth of one GPU: best of `reps` per launch shape, then the best shape back to back for sustain_s."""
  cuda = bridge or Cuda(device, nvcc=nvcc)
  try:
    lib = cuda.library(CUDA_SRC)
    pso = {k: cuda.pipeline(lib, k)["pso"] for k in ("readsum", "readchunk4", "readchunk16")}
    sms = parallel_units(cuda.facts())
    nbytes = (gib << 30) // 16 * 16
    a, out = cuda.buffer(length=nbytes), cuda.buffer(length=max(1024 * 64 * sms, nbytes // min(c for _, c, _ in CHUNK_SHAPES)) * 4)
    sized = lambda size: ([({"threads": t, "blocks": per_sm * sms}, "readsum", [a, struct.pack("<Q", size // 16), out])  # noqa: E731
                           for t, per_sm in READ_SHAPES] +  # every launch shape over the buffer's first `size` bytes
                          [({"threads": t, "blocks": -(-size // c), "chunk_bytes": c, "load_bytes": w}, f"readchunk{w}",
                            [a, struct.pack("<Q", size // w), struct.pack("<I", c // w), out]) for t, c, w in CHUNK_SHAPES])
    shapes = sized(nbytes)
    def go(shape):
      meta, name, args = shape
      return cuda.dispatch(pso[name], args, (meta["blocks"], 1, 1), (meta["threads"], 1, 1))
    rows = []
    for shape in shapes:
      best = min(samples(lambda: go(shape), warmups=1, count=reps))  # the one loop; a bandwidth keeps its best launch
      rows.append({**shape[0], "best_ms": best / 1000.0, "gbs": nbytes / (best * 1e-6) / 1e9})
    i = max(range(len(rows)), key=lambda k: rows[k]["gbs"])
    best = rows[i]
    # a launch timed between events carries a fixed cost no read pays (about 16 us on an R9700, 1% of a 1 GiB read):
    # the best shape over half the buffer gives it, 2 * half - whole, the least of three so its noise can only lower
    # the rate; a cost that is not under half the whole launch is no fixed cost (a read's time grows with its bytes)
    timed, half = (lambda s: min(samples(lambda: go(s), warmups=1, count=reps))), sized(nbytes // 2)[i]  # noqa: E731
    whole = timed(shapes[i])
    fixed_us = min(2 * timed(half) - whole for _ in range(3))
    fixed_us = fixed_us if 0 < fixed_us < whole / 2 else 0.0
    for r in rows:
      r["gbs"] = nbytes / ((r["best_ms"] * 1000.0 - fixed_us) * 1e-6) / 1e9
    # sustained: the best shape back to back for sustain_s seconds, as a decode reads (the clocks settle under load)
    series, done = [], 0.0
    while not series or done < sustain_s * 1e6:  # at least one launch, then until the seconds are up
      us = go(shapes[i])
      series.append(nbytes / ((us - fixed_us) * 1e-6) / 1e9)
      done += us
    result = {"device": device, "name": cuda.name, "bytes": nbytes, "reps": reps, "shapes": rows,
              "launch_fixed_us": round(fixed_us, 2), "best_shape": shapes[i][0],
              "sustained": {"seconds": done / 1e6, "launches": len(series), "best_gbs": round(max(series), 1),
                            "mean_gbs": round(sum(series) / len(series), 1), "min_gbs": round(min(series), 1)}}
  finally:
    if bridge is None:
      cuda.close()
  result.update(plausibility(round(best["gbs"], 1), [r["gbs"] for r in rows], result["sustained"]["best_gbs"]))
  api = getattr(cuda, "API", "CUDA")
  result.update(schema=f"boltbeam.{api.lower()}_read_bandwidth.v1", method=METHOD.format(api=api, gib=gib, reps=reps) +
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
  p.add_argument("--runs", type=int, default=1, help="whole probe runs; the uncertainty is their spread at the best shape")
  a = p.parse_args(argv)
  got = repeated([measure_read_gbs(a.device, gib=a.gib, reps=a.reps) for _ in range(a.runs)], "read_gbs", "gbs")
  sys.stdout.write(json.dumps(got, indent=1, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
