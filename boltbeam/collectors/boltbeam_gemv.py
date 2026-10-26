"""BoltBeam's own quantized GEMV kernels, the reference kernels of the building-block probe, as KernelSpecs for the
one kernel timer (collectors/kernel_timer.py). One source per kernel dialect (MSL; CUDA, which hipcc also takes), the
same algorithm: one simdgroup (warp) per output row, ROWS_PER_GROUP rows per threadgroup (block), a Q4_K lane owning
one 32-bit load per block, Q6_K bytes.

These are not the engine's kernels. They say what a plain, correct kernel reads on this GPU for the model's real
tensor bytes, so a probe row is "BoltBeam's own kernel (reference)". The engine's kernels are timed by the same loop
through collectors/engine_kernels.py. The GGUF block layouts and the pure-Python reference live in
collectors/gemv_reference.py and are read from there, never restated.

The building-block probe (collect_probe_evidence) runs these kernels on any GPU whose backend has a source here,
through that backend's bridge: probe_request.json -> probe_evidence.json.
"""
from __future__ import annotations

import pathlib
import random
import struct
from typing import Any, Callable

from boltbeam.collectors import gemv_reference as reference
from boltbeam.collectors import kernel_timer
from boltbeam.collectors.kernel_timer import Check, KernelSpec
from boltbeam.target import vendors

ADAPTER = "boltbeam"
COLLECTOR_ID = "boltbeam-gemv"
PROVIDER_ID = "boltbeam/gemv"
WORKING_SET_SHARE = 0.8  # never ask for more than this share of the GPU's working set (the bridge's working_set_bytes)
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

# the same two kernels in CUDA: a warp per row, ROWS_PER_GROUP warps per block, the row reduced with shuffles. hipcc
# compiles this source too (the HIP bridge, kernel_dialect "hip"); the shim at the top is all that differs: HIP has
# no cuda_fp16.h, and its __shfl_xor_sync static_asserts a 64-bit mask, so the AMD side takes hip_fp16.h and the
# maskless __shfl_xor. Its width 32 keeps a row's reduction inside the row's 32 lanes on a wave32 GPU (RDNA, hipcc's
# default there) and a wave64 one (CDNA, or RDNA under -mwavefrontsize64) alike: the "warp" here is 32 lanes, not the
# hardware's wave, and the row index (threadIdx.x >> 5) is the same on both.
CUDA = r"""
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_fp16.h>
#define BB_SHFL_XOR(v, o) __shfl_xor(v, o, 32)
#else
#include <cuda_fp16.h>
#define BB_SHFL_XOR(v, o) __shfl_xor_sync(0xffffffffu, v, o)
#endif

__device__ __forceinline__ void scale_min_k4(unsigned j, const unsigned char* q, float& sc, float& mn) {
  if (j < 4) { sc = (float)(q[j] & 63); mn = (float)(q[j + 4] & 63); }
  else { sc = (float)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); mn = (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += BB_SHFL_XOR(v, o);
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
# the source per dialect (target/vendors.py kernel_dialect), as kernel_timer.FLUSH_SOURCES: hipcc takes the CUDA one
DIALECT_SOURCES = {"msl": MSL, "cuda": CUDA, "hip": CUDA}
# the source per backend with a kernel bridge: a vendor row whose dialect has a source here gets the probe
SOURCES = {v.backend: DIALECT_SOURCES[v.kernel_dialect] for v in vendors.VENDORS.values() if v.kernel_dialect in DIALECT_SOURCES}


def vector(seed:str, cols:int) -> list[float]:
  """The probe's input vector: uniform in [-1, 1), the same for the same probe id on every machine."""
  rng = random.Random(seed)
  return [rng.uniform(-1.0, 1.0) for _ in range(cols)]


def synthetic(quant:str, rows:int, cols:int, seed:str = "synthetic") -> bytes:
  """Valid GGUF blocks for a bridge check without a model: random quants, finite fp16 scales. Raw random bytes are
  not a weight: about 1 in 32 random fp16 values is inf or NaN, so most rows of a random Q4_K tensor dot to NaN."""
  if quant not in KERNELS or cols % 256:
    raise ValueError(f"no synthetic {quant} blocks for {cols} columns")
  rng = random.Random(seed)
  blocks = []
  for _ in range(rows * cols // 256):
    if quant == "Q4_K":  # d, dmin, 12 scale bytes, 128 quant bytes
      blocks.append(struct.pack("<ee", rng.uniform(0.001, 0.05), rng.uniform(0.0, 0.02)) + rng.randbytes(140))
    else:  # Q6_K: 128 ql, 64 qh, 16 int8 scales, d
      blocks.append(rng.randbytes(192) + struct.pack("<16b", *(rng.randrange(-64, 64) for _ in range(16)))
                    + struct.pack("<e", rng.uniform(0.001, 0.05)))
  return b"".join(blocks)


def spec(backend:str, quant:str, rows:int, cols:int, weights:bytes, x:list[float], *, label:str | None = None) -> KernelSpec:
  """BoltBeam's GEMV for one weight of rows x cols in this quant, on these real bytes, as a KernelSpec. bytes_read
  counts the weights and the activation vector and the output, as the probe always has."""
  if backend not in SOURCES:
    raise RuntimeError(f"BoltBeam's GEMV has no {backend} source")
  if quant not in KERNELS:
    raise RuntimeError(f"BoltBeam has no {backend} kernel for {quant}")
  name = KERNELS[quant]
  groups = (rows + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP
  check = Check(indices=reference.check_rows(rows), reference=lambda r: reference.reference_row(weights, quant, r, cols, x),
                rel_tol=reference.TOLERANCE)
  return KernelSpec(label=label or f"{ADAPTER} {name}", adapter=ADAPTER, source=SOURCES[backend], kernel=name,
                    args=[weights, struct.pack(f"<{cols}f", *x), rows * 4, ("value", struct.pack("<I", rows)), ("value", struct.pack("<I", cols))],
                    grid=(groups, 1, 1), block=(ROWS_PER_GROUP * 32, 1, 1), bytes_read=len(weights) + cols * 4 + rows * 4,
                    check=check, warm=(1,),
                    record={"threadgroup_threads": ROWS_PER_GROUP * 32, "threadgroups": groups, "source": KERNEL_SOURCE})


def facts(result:dict[str, Any], spec_:KernelSpec) -> dict[str, Any]:
  """The kernel facts a probe row records about this adapter's kernel."""
  return {"name": spec_.kernel, "threadgroup_threads": spec_.block[0], "threadgroups": spec_.grid[0],
          "thread_execution_width": result["pipeline"]["thread_execution_width"],
          "max_threads_per_threadgroup": result["pipeline"]["max_threads_per_threadgroup"], "source": KERNEL_SOURCE}


# --- the building-block probe: these kernels on the model's real bytes ---------------------------------------

def _absent(probe_quant:str, backend:str) -> dict[str, str]:
  """What this backend's probe cannot report, by the field the full-probe schema asks for (target/vendors.py
  probe_absent). Said in the row's `absent` so the next-step ladder never asks this GPU for it (report/html.py
  next_step)."""
  row = vendors.BY_BACKEND.get(backend)
  words = (row and row.probe_absent) or {k: f"{backend} is not read by BoltBeam's probe" for k in vendors.PROBE_FIELDS}
  why = words["counters"]
  return {
    "latency_concurrency.memory_latency_ns": why, "latency_concurrency.active_waves": why,
    "latency_concurrency.in_flight_groups": why, "latency_concurrency.occupancy_pct": why,
    **{k: v for k, v in words.items() if k != "counters"},
  }


def measure_probe(bridge, flusher, probe:dict[str, Any], model:pathlib.Path, data_start:int, tensors:list,
                  floor_us:float, target_gbs:float | None, *, backend:str,
                  libraries:dict[str, int] | None = None) -> dict[str, Any]:
  """One probe row: BoltBeam's GEMV (adapter 0) on the role's real tensor bytes, timed by the one kernel timer."""
  rows, cols = (int(v) for v in probe["shape"])
  quant = probe["quant"]
  row = {"probe_id": probe["probe_id"], "kind": "quant_gemv", "role": probe["role"], "quant": quant,
         "shape": [rows, cols]}
  if quant not in KERNELS or cols % reference.BLOCK_ELEMS[quant]:
    row["status"] = "not_measured"
    row["reason"] = f"BoltBeam has no {backend} kernel for {quant} with {cols} columns"
    return row
  name, offset = reference.tensor_for(quant, rows, cols, tensors, probe.get("tensor_name"), probe.get("n_expert"))
  size = rows * (cols // reference.BLOCK_ELEMS[quant]) * reference.BLOCK_BYTES[quant]
  if size + kernel_timer.FLUSH[backend]["bytes"] > bridge.working_set_bytes * WORKING_SET_SHARE:
    row["status"] = "not_measured"
    row["reason"] = f"{size / 2**30:.1f} GiB of weights does not fit the GPU's working set"
    return row
  with open(model, "rb") as f:
    f.seek(data_start + offset)
    weights = f.read(size)
  x = vector(f"{probe['probe_id']}", cols)
  kspec = spec(backend, quant, rows, cols, weights, x)
  got = kernel_timer.time_spec(bridge, kspec, flusher, libraries=libraries, dram_gbs=target_gbs)
  row["correctness"] = got["correctness"]
  if not got["samples"]:
    row["status"] = "correctness_failed"
    return row
  med = got["median_us"]
  metadata = rows * (cols // 256) * reference.METADATA_BYTES[quant]
  activation = cols * 4 + rows * 4
  row.update({
    "status": "measured", "tensor": name,
    "timing": {"candidate_us": med, "spread_pct": got["spread_pct"], "min_us": got["min_us"], "samples": len(got["samples"]),
               **got["timing"], "dispatch_floor_us": floor_us},
    "bytes": {"physical_weight_bytes": size - metadata, "metadata_bytes": metadata, "activation_bytes": activation},
    "throughput": {"achieved_gbs": got["gbs"], "target_gbs": target_gbs},
    "latency_concurrency": {},
    "isa": {},
    "resources": {"lds_bytes": got["pipeline"]["static_threadgroup_memory_bytes"]},
    "kernel": facts(got, kspec),
    "timed_by": kspec.label,
    "absent": _absent(quant, backend),
  })
  return row


def collect_probe_evidence(run:pathlib.Path, say:Callable[[str], None] = lambda _: None) -> dict[str, Any]:
  """probe_request.json -> probe_evidence.json on this machine's GPU, through its backend's bridge: BoltBeam's own
  GEMV per role on the model's real bytes. The peak every row is measured against is the run's one read bandwidth.
  CannotMeasure, with the reason, where this machine has no such GPU or no kernel for the backend."""
  from boltbeam.collectors.llama_bench_decode import CannotMeasure, model_file
  from boltbeam.profile.gguf import read_gguf_layout
  from boltbeam.target.targets import get_target
  from boltbeam.vocab import SCHEMA_PROBE_EVIDENCE
  from boltbeam.workflow.common import load_manifest, read_json
  from boltbeam.workflow.screen import run_bandwidth
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  backend, model = target.backend, model_file(manifest.get("model_path") or "")
  if backend not in SOURCES:
    raise CannotMeasure(f"BoltBeam has no {backend} GEMV kernel", f"measure on a machine with a {' or '.join(SOURCES)} GPU")
  request = read_json(run / "probe_request.json")
  _, tensors, data_start = read_gguf_layout(model)
  try:
    bridge = kernel_timer.bridge_for(backend)
  except RuntimeError as exc:  # MetalUnavailable, CudaUnavailable: this machine has no such GPU or no compiler
    raise CannotMeasure(str(exc), f"measure on a machine with a {backend} GPU") from exc
  try:
    flusher = kernel_timer.Flusher(bridge, backend)
    floor_us = flusher.floor_us()
    target_gbs, target_source = run_bandwidth(run, target)
    rows = []
    libraries:dict[str, int] = {}
    for i, probe in enumerate(request.get("probes", []), start=1):
      say(f"probe {i} of {len(request['probes'])}: {probe['role']} {probe['quant']} {probe['shape'][0]}x{probe['shape'][1]}")
      rows.append(measure_probe(bridge, flusher, probe, model, data_start, tensors, floor_us, target_gbs,
                                backend=backend, libraries=libraries))
  finally:
    bridge.close()
  return {
    "schema": SCHEMA_PROBE_EVIDENCE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
    "workload": manifest["workload"], "provider_id": PROVIDER_ID, "collector_id": COLLECTOR_ID,
    "device": bridge.name, "dispatch_floor_us": floor_us, "probes": rows, "peak_source": target_source,
    "measured": ["timing", "throughput.achieved_gbs", "resources.lds_bytes", "correctness"],
    "derived": ["bytes (from the GGUF block layout)", f"throughput.target_gbs ({target_source})"],
    "notes": [f"Kernels are BoltBeam's own {backend} GEMVs on the model's real tensor bytes (collectors/boltbeam_gemv.py), "
              "not the engine's kernels.",
              "Each timed sample follows a flush of a buffer larger than the last-level cache.",
              "The timed interval includes dispatch_floor_us of fixed GPU start cost; it is not subtracted."],
  }
