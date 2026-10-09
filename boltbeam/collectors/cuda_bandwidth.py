"""Read-only GPU memory speed of one NVIDIA GPU, with a small native CUDA kernel.

    python -m boltbeam.collectors.cuda_bandwidth [--device N] [--gib 1] [--reps 10]

Decode reads the weights once per token, so the speed limit needs the READ rate. tinygrad's sum reaches only about
22% of a 5090's bandwidth (382 GB/s against about 1,700), so it is no measure of bandwidth. This kernel reads a
buffer of at least 1 GiB with 16-byte vector loads in a grid-stride loop and writes one value per block. The program
is compiled at run time with nvcc (from PATH or /usr/local/cuda/bin) for the GPU it runs on (-arch=native), timed
with CUDA events, and keeps the best of N runs per launch shape, the rule metal_bandwidth.py and the targets
registry use for every measured bandwidth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any

CUDA_SRC = r"""
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void readsum(const int4* __restrict__ a, size_t n4, float* out) {
  int acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
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

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("{\"error\": \"%s\"}\n", cudaGetErrorString(e)); return 1; } } while (0)

int main(int argc, char** argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  size_t bytes = argc > 2 ? strtoull(argv[2], 0, 10) : (1ull << 30);
  int reps = argc > 3 ? atoi(argv[3]) : 10;
  CK(cudaSetDevice(dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  size_t n4 = bytes / 16;
  int4* a; float* out;
  CK(cudaMalloc(&a, n4 * 16));
  CK(cudaMemset(a, 1, n4 * 16));
  int shapes[][2] = {{256, 4}, {256, 8}, {256, 16}, {512, 4}, {512, 8}, {1024, 2}, {1024, 4}};  // threads, blocks per SM
  CK(cudaMalloc(&out, 1024 * 64 * p.multiProcessorCount * sizeof(float)));
  cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  double best = 0; int bt = 0, bb = 0;
  printf("{\"device\": %d, \"name\": \"%s\", \"bytes\": %zu, \"reps\": %d, \"shapes\": [", dev, p.name, n4 * 16, reps);
  for (int s = 0; s < 7; s++) {
    int threads = shapes[s][0], blocks = shapes[s][1] * p.multiProcessorCount;
    readsum<<<blocks, threads>>>(a, n4, out);  // warm-up, not counted
    CK(cudaDeviceSynchronize());
    float ms_best = 1e30f;
    for (int r = 0; r < reps; r++) {
      CK(cudaEventRecord(t0));
      readsum<<<blocks, threads>>>(a, n4, out);
      CK(cudaEventRecord(t1));
      CK(cudaEventSynchronize(t1));
      float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
      if (ms < ms_best) ms_best = ms;
    }
    double gbs = (double)(n4 * 16) / (ms_best * 1e-3) / 1e9;
    printf("%s{\"threads\": %d, \"blocks\": %d, \"best_ms\": %.4f, \"gbs\": %.1f}", s ? ", " : "", threads, blocks, ms_best, gbs);
    if (gbs > best) { best = gbs; bt = threads; bb = blocks; }
  }
  printf("], \"read_gbs\": %.1f, \"best_shape\": {\"threads\": %d, \"blocks\": %d}}\n", best, bt, bb);
  return 0;
}
"""
METHOD = "BoltBeam cuda_bandwidth: native CUDA read-only kernel, 16-byte loads over {gib} GiB, best of {reps} per launch shape"


def find_nvcc(env:dict[str, str] | None = None) -> str | None:
  env = os.environ if env is None else env
  if env.get("BOLTBEAM_NVCC"):
    return env["BOLTBEAM_NVCC"]
  return shutil.which("nvcc") or next((p for p in ("/usr/local/cuda/bin/nvcc",) if pathlib.Path(p).is_file()), None)


def build(nvcc:str, cache:pathlib.Path | None = None) -> pathlib.Path:
  """Compile the probe once per source and nvcc; the binary is cached."""
  cache = cache or pathlib.Path(tempfile.gettempdir()) / "boltbeam-cuda-bandwidth"
  cache.mkdir(parents=True, exist_ok=True)
  key = hashlib.sha256((CUDA_SRC + nvcc).encode()).hexdigest()[:12]
  exe = cache / f"readbw-{key}"
  if not exe.exists():
    src = cache / f"readbw-{key}.cu"
    src.write_text(CUDA_SRC)
    proc = subprocess.run([nvcc, "-O3", "-arch=native", "-o", str(exe), str(src)], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
      raise RuntimeError(f"nvcc failed: {(proc.stderr or proc.stdout).strip()[-300:]}")
  return exe


def parse(stdout:str) -> dict[str, Any]:
  line = next((l for l in reversed(stdout.splitlines()) if l.startswith("{")), None)
  if line is None:
    raise RuntimeError(f"the CUDA probe printed no result: {stdout.strip()[-200:]}")
  out = json.loads(line)
  if "error" in out:
    raise RuntimeError(f"CUDA: {out['error']}")
  return out


def measure_read_gbs(device:int = 0, *, gib:int = 1, reps:int = 10, nvcc:str | None = None,
                     run=subprocess.run) -> dict[str, Any]:
  nvcc = nvcc or find_nvcc()
  if not nvcc:
    raise RuntimeError("nvcc is not installed (set BOLTBEAM_NVCC or put /usr/local/cuda/bin on PATH)")
  exe = build(nvcc)
  proc = run([str(exe), str(device), str(gib << 30), str(reps)], capture_output=True, text=True, timeout=600)
  if proc.returncode != 0:
    raise RuntimeError(f"the CUDA probe exited {proc.returncode}: {(proc.stdout + proc.stderr).strip()[-300:]}")
  out = parse(proc.stdout)
  out.update(schema="boltbeam.cuda_read_bandwidth.v1", method=METHOD.format(gib=gib, reps=reps))
  return out


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
