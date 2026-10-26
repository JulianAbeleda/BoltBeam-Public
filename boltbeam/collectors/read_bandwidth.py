"""Read-only GPU memory speed of one NVIDIA or AMD GPU, with a small native kernel through BoltBeam's CUDA or HIP
bridge. The read kernels are plain CUDA C, which hipcc compiles unchanged; the bridge decides the toolchain, and the
result names the API it ran on (cuda_read_bandwidth or hip_read_bandwidth), never one for the other.

    python -m boltbeam.collectors.read_bandwidth --backend CUDA|AMD [--device N] [--gib 1] [--reps 10]

Decode reads the weights once per token, so the speed limit needs the READ rate. tinygrad's sum reaches only about
22% of a 5090's bandwidth (382 GB/s against about 1,700), so it is no measure of bandwidth. This kernel reads a
buffer of at least 1 GiB with 16-byte vector loads in a grid-stride loop and writes one value per block. The kernel
is compiled at run time by the bridge (nvcc or hipcc, cached) and each launch is
timed between the runtime's events there; the launches run through the one loop (collectors/kernel_timer.py samples) and
this probe keeps the best of N per launch shape, the rule metal_bandwidth.py and the targets registry use for every
measured bandwidth.

The bridge is the caller's, opened on the card the scan found (workflow/chips._probes); each vendor's matrix probe
is its own (the vendor row's compute module). The driver's device facts come from the bridge's facts().
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from typing import Any

from boltbeam.collectors.kernel_timer import bridge_for, samples

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
READ_SHAPES = ((256, 4), (256, 8), (256, 16), (512, 4), (512, 8), (1024, 2), (1024, 4))  # threads, blocks per SM
# threads, bytes per block, load bytes: one block per chunk, the grid covers the buffer (a launch shape, not a model rule)
CHUNK_SHAPES = ((128, 1024, 16), (128, 4096, 4), (128, 4096, 16), (256, 4096, 4), (256, 16384, 16), (512, 65536, 16))
METHOD = ("BoltBeam read_bandwidth: native {api} read-only kernels over {gib} GiB, a grid-stride loop of 16-byte loads and one "
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


def repeated_paths(runs:list[dict[str, Any]]) -> dict[str, Any]:
  """Whole matrix-probe runs as one result: each path's best run, its uncertainty the runs' spread (repeated)."""
  if not runs[0].get("paths"):
    return runs[0]
  return {**runs[0], "runs": len(runs), "paths": [repeated([r["paths"][i] for r in runs], "tops", "tops")
                                                  for i in range(len(runs[0]["paths"]))]}


def parallel_units(facts:dict[str, Any]) -> int:
  """How many units a grid is spread over, as the bridge's driver counts them: CUDA's SMs, HIP's multiprocessors
  (work-group processors on RDNA). It sizes launch grids only; it is not a compute-unit count."""
  return facts.get("sm_count") or facts.get("multiprocessor_count") or 1


def measure_read_gbs(device:int, *, bridge, gib:int = 1, reps:int = 10, sustain_s:float = SUSTAIN_S) -> dict[str, Any]:
  """Read bandwidth of one GPU through its bridge: best of `reps` per launch shape, then the best shape back to back
  for sustain_s."""
  lib = bridge.library(CUDA_SRC)
  pso = {k: bridge.pipeline(lib, k)["pso"] for k in ("readsum", "readchunk4", "readchunk16")}
  sms = parallel_units(bridge.facts())
  nbytes = (gib << 30) // 16 * 16
  a, out = bridge.buffer(length=nbytes), bridge.buffer(length=max(1024 * 64 * sms, nbytes // min(c for _, c, _ in CHUNK_SHAPES)) * 4)
  sized = lambda size: ([({"threads": t, "blocks": per_sm * sms}, "readsum", [a, struct.pack("<Q", size // 16), out])  # noqa: E731
                         for t, per_sm in READ_SHAPES] +  # every launch shape over the buffer's first `size` bytes
                        [({"threads": t, "blocks": -(-size // c), "chunk_bytes": c, "load_bytes": w}, f"readchunk{w}",
                          [a, struct.pack("<Q", size // w), struct.pack("<I", c // w), out]) for t, c, w in CHUNK_SHAPES])
  shapes = sized(nbytes)
  def go(shape):
    meta, name, args = shape
    return bridge.dispatch(pso[name], args, (meta["blocks"], 1, 1), (meta["threads"], 1, 1))
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
  result = {"device": device, "name": bridge.name, "bytes": nbytes, "reps": reps, "shapes": rows,
            "launch_fixed_us": round(fixed_us, 2), "best_shape": shapes[i][0],
            "sustained": {"seconds": done / 1e6, "launches": len(series), "best_gbs": round(max(series), 1),
                          "mean_gbs": round(sum(series) / len(series), 1), "min_gbs": round(min(series), 1)}}
  result.update(plausibility(round(best["gbs"], 1), [r["gbs"] for r in rows], result["sustained"]["best_gbs"]))
  api = bridge.API
  result.update(schema=f"boltbeam.{api.lower()}_read_bandwidth.v1", method=METHOD.format(api=api, gib=gib, reps=reps) +
                f"; then the best shape back to back for {sustain_s:.0f} s (sustained)")
  return result


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog="python -m boltbeam.collectors.read_bandwidth")
  p.add_argument("--backend", required=True, help="the target backend whose bridge reads (CUDA, AMD)")
  p.add_argument("--device", type=int, default=0)
  p.add_argument("--gib", type=int, default=1)
  p.add_argument("--reps", type=int, default=10)
  p.add_argument("--runs", type=int, default=1, help="whole probe runs; the uncertainty is their spread at the best shape")
  a = p.parse_args(argv)
  bridge = bridge_for(a.backend, device=a.device)
  try:
    got = repeated([measure_read_gbs(a.device, bridge=bridge, gib=a.gib, reps=a.reps) for _ in range(a.runs)], "read_gbs", "gbs")
  finally:
    bridge.close()
  sys.stdout.write(json.dumps(got, indent=1, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
