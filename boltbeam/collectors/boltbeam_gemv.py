"""BoltBeam's own quantized GEMV kernels, the reference kernels of the building-block probe, as KernelSpecs for the
one kernel timer (collectors/kernel_timer.py). One source per backend, the same algorithm: one simdgroup (warp) per
output row, ROWS_PER_GROUP rows per threadgroup (block), a Q4_K lane owning one 32-bit load per block, Q6_K bytes.

These are not the engine's kernels. They say what a plain, correct kernel reads on this GPU for the model's real
tensor bytes, so a probe row is "BoltBeam's own kernel (reference)". The engine's kernels are timed by the same loop
through collectors/engine_kernels.py. The GGUF block layouts and the pure-Python reference live in
collectors/metal_native.py and are read from there, never restated.
"""
from __future__ import annotations

import random
import struct
from typing import Any

from boltbeam.collectors.kernel_timer import Check, KernelSpec

ADAPTER = "boltbeam"
ROWS_PER_GROUP = 4  # simdgroups (warps) per threadgroup (block); one simdgroup owns one output row
KERNELS = {"Q4_K": "gemv_q4_k", "Q6_K": "gemv_q6_k"}
# the kernel's source-level facts: what BoltBeam wrote, not what the compiler emitted
KERNEL_SOURCE = {"loads": "Q4_K: one 32-bit load per lane per block; Q6_K: bytes", "rows_per_simdgroup": 1, "reduction": "simd_sum"}

MSL = r"""
#include <metal_stdlib>
using namespace metal;

static inline void scale_min_k4(uint j, device const uchar* q, thread float& sc, thread float& mn) {
  if (j < 4) { sc = float(q[j] & 63); mn = float(q[j + 4] & 63); }
  else { sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); mn = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

kernel void gemv_q4_k(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  uint row = tg * nsg + sg;
  if (row >= rows) return;
  uint nb = cols / 256;
  device const uchar* rp = w + ulong(row) * nb * 144;
  float acc = 0.0f;
  for (uint b = 0; b < nb; b++) {
    device const uchar* blk = rp + b * 144;
    float d = float(*(device const half*)(blk));
    float dmin = float(*(device const half*)(blk + 2));
    // lane owns qs bytes 4*lane..4*lane+3: one 32-bit load, one scale pair per block
    uint j = lane / 8, i0 = (lane % 8) * 4;
    float s0, m0, s1, m1;
    scale_min_k4(2 * j, blk + 4, s0, m0);
    scale_min_k4(2 * j + 1, blk + 4, s1, m1);
    uint q = *(device const uint*)(blk + 16 + 4 * lane);
    float4 lo = *(device const float4*)(x + b * 256 + 64 * j + i0);
    float4 hi = *(device const float4*)(x + b * 256 + 64 * j + 32 + i0);
    float4 ql = float4(q & 0xF, (q >> 8) & 0xF, (q >> 16) & 0xF, (q >> 24) & 0xF);
    float4 qh = float4((q >> 4) & 0xF, (q >> 12) & 0xF, (q >> 20) & 0xF, (q >> 28) & 0xF);
    acc += d * s0 * dot(ql, lo) - dmin * m0 * (lo.x + lo.y + lo.z + lo.w);
    acc += d * s1 * dot(qh, hi) - dmin * m1 * (hi.x + hi.y + hi.z + hi.w);
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}

kernel void gemv_q6_k(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  uint row = tg * nsg + sg;
  if (row >= rows) return;
  uint nb = cols / 256;
  device const uchar* rp = w + ulong(row) * nb * 210;
  float acc = 0.0f;
  uint is = lane / 16;
  for (uint b = 0; b < nb; b++) {
    device const uchar* blk = rp + b * 210;
    float d = float(*(device const half*)(blk + 208));
    device const float* xb = x + b * 256;
    for (uint n = 0; n < 2; n++) {
      device const uchar* ql = blk + 64 * n;
      device const uchar* qh = blk + 128 + 32 * n;
      device const char* sc = (device const char*)(blk + 192 + 8 * n);
      uchar h = qh[lane];
      int q1 = int((ql[lane] & 0xF) | ((h & 3) << 4)) - 32;
      int q2 = int((ql[lane + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32;
      int q3 = int((ql[lane] >> 4) | (((h >> 4) & 3) << 4)) - 32;
      int q4 = int((ql[lane + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
      device const float* xh = xb + 128 * n;
      acc += d * float(sc[is]) * float(q1) * xh[lane];
      acc += d * float(sc[is + 2]) * float(q2) * xh[lane + 32];
      acc += d * float(sc[is + 4]) * float(q3) * xh[lane + 64];
      acc += d * float(sc[is + 6]) * float(q4) * xh[lane + 96];
    }
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}
"""

# the same two kernels in CUDA: a warp per row, ROWS_PER_GROUP warps per block, the row reduced with shuffles
CUDA = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ void scale_min_k4(unsigned j, const unsigned char* q, float& sc, float& mn) {
  if (j < 4) { sc = (float)(q[j] & 63); mn = (float)(q[j + 4] & 63); }
  else { sc = (float)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); mn = (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

extern "C" __global__ void gemv_q4_k(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                                     unsigned rows, unsigned cols) {
  unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  unsigned row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= rows) return;
  unsigned nb = cols / 256;
  const unsigned char* rp = w + (size_t)row * nb * 144;
  float acc = 0.0f;
  for (unsigned b = 0; b < nb; b++) {
    const unsigned char* blk = rp + b * 144;
    float d = __half2float(*(const __half*)(blk));
    float dmin = __half2float(*(const __half*)(blk + 2));
    unsigned j = lane / 8, i0 = (lane % 8) * 4;
    float s0, m0, s1, m1;
    scale_min_k4(2 * j, blk + 4, s0, m0);
    scale_min_k4(2 * j + 1, blk + 4, s1, m1);
    unsigned q = *(const unsigned*)(blk + 16 + 4 * lane);
    float4 lo = *(const float4*)(x + b * 256 + 64 * j + i0);
    float4 hi = *(const float4*)(x + b * 256 + 64 * j + 32 + i0);
    float dl = (float)(q & 0xF) * lo.x + (float)((q >> 8) & 0xF) * lo.y + (float)((q >> 16) & 0xF) * lo.z + (float)((q >> 24) & 0xF) * lo.w;
    float dh = (float)((q >> 4) & 0xF) * hi.x + (float)((q >> 12) & 0xF) * hi.y + (float)((q >> 20) & 0xF) * hi.z + (float)((q >> 28) & 0xF) * hi.w;
    acc += d * s0 * dl - dmin * m0 * (lo.x + lo.y + lo.z + lo.w);
    acc += d * s1 * dh - dmin * m1 * (hi.x + hi.y + hi.z + hi.w);
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

extern "C" __global__ void gemv_q6_k(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                                     unsigned rows, unsigned cols) {
  unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  unsigned row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= rows) return;
  unsigned nb = cols / 256;
  const unsigned char* rp = w + (size_t)row * nb * 210;
  float acc = 0.0f;
  unsigned is = lane / 16;
  for (unsigned b = 0; b < nb; b++) {
    const unsigned char* blk = rp + b * 210;
    float d = __half2float(*(const __half*)(blk + 208));
    const float* xb = x + b * 256;
    for (unsigned n = 0; n < 2; n++) {
      const unsigned char* ql = blk + 64 * n;
      const unsigned char* qh = blk + 128 + 32 * n;
      const signed char* sc = (const signed char*)(blk + 192 + 8 * n);
      unsigned char h = qh[lane];
      int q1 = (int)((ql[lane] & 0xF) | ((h & 3) << 4)) - 32;
      int q2 = (int)((ql[lane + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32;
      int q3 = (int)((ql[lane] >> 4) | (((h >> 4) & 3) << 4)) - 32;
      int q4 = (int)((ql[lane + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
      const float* xh = xb + 128 * n;
      acc += d * (float)sc[is] * (float)q1 * xh[lane];
      acc += d * (float)sc[is + 2] * (float)q2 * xh[lane + 32];
      acc += d * (float)sc[is + 4] * (float)q3 * xh[lane + 64];
      acc += d * (float)sc[is + 6] * (float)q4 * xh[lane + 96];
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}
"""
SOURCES = {"Metal": MSL, "CUDA": CUDA}


def vector(seed:str, cols:int) -> list[float]:
  """The probe's input vector: uniform in [-1, 1), the same for the same probe id on every machine."""
  rng = random.Random(seed)
  return [rng.uniform(-1.0, 1.0) for _ in range(cols)]


def spec(backend:str, quant:str, rows:int, cols:int, weights:bytes, x:list[float], *, label:str | None = None) -> KernelSpec:
  """BoltBeam's GEMV for one weight of rows x cols in this quant, on these real bytes, as a KernelSpec. bytes_read
  counts the weights and the activation vector and the output, as the probe always has."""
  from boltbeam.collectors import metal_native as native
  if backend not in SOURCES:
    raise RuntimeError(f"BoltBeam's GEMV has no {backend} source")
  if quant not in KERNELS:
    raise RuntimeError(f"BoltBeam has no {backend} kernel for {quant}")
  name = KERNELS[quant]
  groups = (rows + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP
  check = Check(indices=native.check_rows(rows), reference=lambda r: native.reference_row(weights, quant, r, cols, x),
                rel_tol=native.TOLERANCE)
  return KernelSpec(label=label or f"{ADAPTER} {name}", adapter=ADAPTER, source=SOURCES[backend], kernel=name,
                    args=[weights, struct.pack(f"<{cols}f", *x), rows * 4, ("value", struct.pack("<I", rows)), ("value", struct.pack("<I", cols))],
                    grid=(groups, 1, 1), block=(ROWS_PER_GROUP * 32, 1, 1), bytes_read=len(weights) + cols * 4 + rows * 4,
                    check=check,
                    record={"threadgroup_threads": ROWS_PER_GROUP * 32, "threadgroups": groups, "source": KERNEL_SOURCE})


def facts(result:dict[str, Any], spec_:KernelSpec) -> dict[str, Any]:
  """The kernel facts a probe row records about this adapter's kernel."""
  return {"name": spec_.kernel, "threadgroup_threads": spec_.block[0], "threadgroups": spec_.grid[0],
          "thread_execution_width": result["pipeline"]["thread_execution_width"],
          "max_threads_per_threadgroup": result["pipeline"]["max_threads_per_threadgroup"], "source": KERNEL_SOURCE}
