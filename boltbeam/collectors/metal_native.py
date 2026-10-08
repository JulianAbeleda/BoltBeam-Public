"""BoltBeam's own Metal measurement: probe evidence and a timing trace with no Xcode, no Instruments, no tinygrad.

    python -m boltbeam.collectors.metal_native --run RUN [--probe-out P] [--timing-out T] [--only probe|timing]

It answers the two requests `analyze` writes into a run folder:

  probe_request.json  -> probe_evidence.json   (boltbeam.probe_evidence.v1)
    Each probe is one quantized GEMV (role, quant, shape). The weights are the model's real tensor bytes. The
    kernel is BoltBeam's MSL source, compiled at run time. Before any timing, sampled output rows are checked
    against a pure-Python dequantize-and-dot reference. Every timed sample runs after a flush kernel that
    rewrites a buffer larger than the system level cache, so the weights come from DRAM, not cache.

  trace_request.json  -> timing_trace.json     (boltbeam.timing_trace.v1)
    whole_step rows: llama-bench decode at each requested depth (`-p 0 -n N -d CTX`). This is the measured
    tokens per second of a real decode on this GPU.
    candidate rows: BoltBeam's flushed GEMV time per role times that role's count per token. These are measured
    BoltBeam kernels, not the kernels llama.cpp ran, so they are not kernel rows: they cannot split llama.cpp's
    step. llama.cpp's per-kernel time stays absent (Metal shows per-dispatch time only to Instruments).

What Metal on Apple GPUs does not expose without Instruments stays absent: registers, scratch, occupancy,
memory latency, in-flight groups, the compiled ISA. Those probes classify as inconclusive (missing evidence).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import statistics
import struct
import subprocess
import sys
from typing import Any, Callable

from boltbeam.core.canonical import pretty_json
from boltbeam.profile.gguf import read_gguf_layout
from boltbeam.target.targets import get_target
from boltbeam.vocab import SCHEMA_PROBE_EVIDENCE, SCHEMA_TIMING_TRACE
from boltbeam.workflow.common import load_manifest, read_json, run_dir

COLLECTOR_ID = "metal-native"
PROVIDER_ID = "boltbeam/metal-native"
GGML_TYPES = {"Q4_K": 12, "Q6_K": 14}
BLOCK_BYTES = {"Q4_K": 144, "Q6_K": 210}  # 256 weights per block
METADATA_BYTES = {"Q4_K": 16, "Q6_K": 18}  # Q4_K: d, dmin, 12 scale bytes. Q6_K: 16 int8 scales, d.
FLUSH_BYTES = 64 << 20  # larger than the M-series system level cache
ROWS_PER_GROUP = 4  # simdgroups per threadgroup; one simdgroup owns one output row
SAMPLES, WARMUPS = 20, 10  # warmups let the GPU clock ramp up before the first timed sample
CHECK_ROWS = 9
TOLERANCE = 1e-3  # max |gpu - cpu| over max |cpu|: float32 sums in a different order
GEN_TOKENS, BENCH_REPS = 32, 3
WORKING_SET_SHARE = 0.8  # never ask for more than this share of Metal's recommended working set

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

kernel void flush(device float* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] = b[i] * 0.5f + 1.0f; }

kernel void empty(device float* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0xFFFFFFFF) b[0] = 0.0f; }
"""
KERNELS = {"Q4_K": "gemv_q4_k", "Q6_K": "gemv_q6_k"}
# the kernel's source-level facts: what BoltBeam wrote, not what the compiler emitted
KERNEL_SOURCE = {"loads": "Q4_K: one 32-bit load per lane per block; Q6_K: bytes", "rows_per_simdgroup": 1, "reduction": "simd_sum"}


class CannotMeasure(RuntimeError):
  """Measuring is impossible here. `reason` says why; `command` says what to run instead."""

  def __init__(self, reason:str, command:str) -> None:
    super().__init__(reason)
    self.reason, self.command = reason, command


# --- the CPU reference: pure Python, the GGUF block layouts written out once more ---------------------------

def _half(raw:bytes, at:int) -> float:
  return struct.unpack_from("<e", raw, at)[0]


def _scale_min_k4(j:int, q:bytes) -> tuple[int, int]:
  if j < 4:
    return q[j] & 63, q[j + 4] & 63
  return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dequant_q4_k_block(blk:bytes) -> list[float]:
  d, dmin = _half(blk, 0), _half(blk, 2)
  scales, qs = blk[4:16], blk[16:144]
  out = [0.0] * 256
  for j in range(4):
    s0, m0 = _scale_min_k4(2 * j, scales)
    s1, m1 = _scale_min_k4(2 * j + 1, scales)
    for l in range(32):
      q = qs[32 * j + l]
      out[64 * j + l] = d * s0 * (q & 0xF) - dmin * m0
      out[64 * j + 32 + l] = d * s1 * (q >> 4) - dmin * m1
  return out


def dequant_q6_k_block(blk:bytes) -> list[float]:
  d = _half(blk, 208)
  out = [0.0] * 256
  for n in range(2):
    ql, qh = blk[64 * n:64 * n + 64], blk[128 + 32 * n:160 + 32 * n]
    sc = struct.unpack_from("<8b", blk, 192 + 8 * n)
    for l in range(32):
      i, h = l // 16, qh[l]
      out[128 * n + l] = d * sc[i] * (((ql[l] & 0xF) | ((h & 3) << 4)) - 32)
      out[128 * n + l + 32] = d * sc[i + 2] * (((ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32)
      out[128 * n + l + 64] = d * sc[i + 4] * (((ql[l] >> 4) | (((h >> 4) & 3) << 4)) - 32)
      out[128 * n + l + 96] = d * sc[i + 6] * (((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32)
  return out


DEQUANT: dict[str, Callable[[bytes], list[float]]] = {"Q4_K": dequant_q4_k_block, "Q6_K": dequant_q6_k_block}


def reference_row(weights:bytes, quant:str, row:int, cols:int, x:list[float]) -> float:
  size, nb = BLOCK_BYTES[quant], cols // 256
  base = row * nb * size
  acc = 0.0
  for b in range(nb):
    vals = DEQUANT[quant](weights[base + b * size: base + (b + 1) * size])
    xb = x[b * 256:(b + 1) * 256]
    acc += sum(v * xv for v, xv in zip(vals, xb))
  return acc


def check_rows(rows:int) -> list[int]:
  return sorted({min(rows - 1, (rows - 1) * i // (CHECK_ROWS - 1)) for i in range(CHECK_ROWS)})


# --- where measuring is possible ---------------------------------------------------------------------------

def this_machine_target() -> str | None:
  from boltbeam.workflow.autoscan import _hardware_profile
  return _hardware_profile()["gpu"].get("target_id")


def preflight(target_id:str, model:str | pathlib.Path, *, run:str | pathlib.Path = "RUN", need_bench:bool = True,
              llama_bench:str = "llama-bench", here:Callable[[], str | None] = this_machine_target) -> dict[str, Any]:
  """What the measurement will use, or CannotMeasure with the reason and the command to run instead."""
  target = get_target(target_id)
  if target.backend != "Metal":
    raise CannotMeasure(f"{target_id} is a {target.backend} chip; BoltBeam measures by itself on Metal only",
                        f"on a machine with {target_id}: boltbeam collect-hw-trace --provider llama --run {run}")
  if sys.platform != "darwin":
    raise CannotMeasure(f"this machine is {sys.platform}, not a Mac", f"on a Mac with {target_id}: boltbeam metal-measure --run {run}")
  mine = here()
  if mine != target_id and target_id != "apple_metal":
    raise CannotMeasure(f"this Mac is {mine or 'an unknown chip'}, not {target_id}",
                        f"pick {mine} in step 2, or on a Mac with {target_id}: boltbeam metal-measure --run {run}")
  path = pathlib.Path(model).expanduser()
  if not path.is_file():
    raise CannotMeasure(f"the model file is gone: {path}", "measure again with the model file in place")
  bench = shutil.which(llama_bench) or (llama_bench if pathlib.Path(llama_bench).is_file() else None)
  if need_bench and bench is None:
    raise CannotMeasure("llama-bench is not installed, so there is no whole-step decode time",
                        "brew install llama.cpp, then measure again")
  return {"target": target, "model": path, "llama_bench": bench}


def _run_facts(run:pathlib.Path, **kw:Any) -> dict[str, Any]:
  manifest = load_manifest(run)
  facts = preflight(manifest.get("target_id"), manifest.get("model_path") or "", run=run, **kw)
  facts["manifest"] = manifest
  return facts


# --- probe evidence ----------------------------------------------------------------------------------------

def _tensor_for(probe:dict[str, Any], tensors:list[tuple[str, tuple[int, ...], int, int]]) -> tuple[str, int]:
  rows, cols = (int(v) for v in probe["shape"])
  want = GGML_TYPES.get(probe["quant"])
  matches = [(name, off) for name, dims, typ, off in tensors if typ == want and tuple(dims) == (cols, rows)]
  if not matches:
    raise ValueError(f"no {probe['quant']} tensor of shape {rows}x{cols} in the model")
  named = [m for m in matches if m[0] == probe.get("tensor_name")]
  return (named or matches)[0]


def _percentile(values:list[float], q:float) -> float:
  ordered = sorted(values)
  return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _absent(probe_quant:str) -> dict[str, str]:
  why = "Metal reports no per-dispatch hardware counters without Instruments"
  return {
    "latency_concurrency.memory_latency_ns": why, "latency_concurrency.active_waves": why,
    "latency_concurrency.in_flight_groups": why, "latency_concurrency.occupancy_pct": why,
    "isa.instruction_histogram": "Metal does not export the compiled AGX ISA",
    "isa.vector_load_bits": "the compiled load width is not visible; the source loads are in kernel.source",
    "resources.registers": "MTLComputePipelineState does not report registers",
    "resources.scratch_bytes": "MTLComputePipelineState does not report scratch",
  }


def measure_probe(metal:Any, kernels:dict[str, Any], probe:dict[str, Any], model:pathlib.Path, data_start:int,
                  tensors:list, flush_buf:int, flush_groups:int, floor_us:float, target_gbs:float | None) -> dict[str, Any]:
  rows, cols = (int(v) for v in probe["shape"])
  quant = probe["quant"]
  row = {"probe_id": probe["probe_id"], "kind": "quant_gemv", "role": probe["role"], "quant": quant,
         "shape": [rows, cols]}
  if quant not in KERNELS or cols % 256:
    row["status"] = "not_measured"
    row["reason"] = f"BoltBeam has no Metal kernel for {quant} with {cols} columns"
    return row
  name, offset = _tensor_for(probe, tensors)
  size = rows * (cols // 256) * BLOCK_BYTES[quant]
  if size + FLUSH_BYTES > metal.working_set_bytes * WORKING_SET_SHARE:
    row["status"] = "not_measured"
    row["reason"] = f"{size / 2**30:.1f} GiB of weights does not fit the Metal working set"
    return row
  with open(model, "rb") as f:
    f.seek(data_start + offset)
    weights = f.read(size)
  rng = random.Random(f"{probe['probe_id']}")
  x = [rng.uniform(-1.0, 1.0) for _ in range(cols)]
  w_buf, x_buf, y_buf = metal.buffer(weights), metal.buffer(struct.pack(f"<{cols}f", *x)), metal.buffer(length=rows * 4)
  try:
    k = kernels[KERNELS[quant]]
    groups, threads = (rows + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP, ROWS_PER_GROUP * 32
    consts = [struct.pack("<I", rows), struct.pack("<I", cols)]
    launch = lambda: metal.run(k["pso"], [w_buf, x_buf, y_buf], consts, groups, threads)  # noqa: E731
    launch()
    gpu = struct.unpack(f"<{rows}f", metal.read(y_buf, rows * 4))
    checked = check_rows(rows)
    ref = {r: reference_row(weights, quant, r, cols, x) for r in checked}
    scale = max(abs(v) for v in ref.values()) or 1.0
    err = max(abs(gpu[r] - ref[r]) for r in checked) / scale
    row["correctness"] = {"checked_rows": checked, "reference": "pure-Python dequantize and dot", "max_rel_err": err,
                          "tolerance": TOLERANCE, "passed": err <= TOLERANCE}
    if err > TOLERANCE:
      row["status"] = "correctness_failed"
      return row
    for _ in range(WARMUPS):
      launch()
    samples = []
    for _ in range(SAMPLES):
      metal.run(kernels["flush"]["pso"], [flush_buf], [], flush_groups, 256)
      samples.append(launch())
  finally:
    for buf in (w_buf, x_buf, y_buf):
      metal.release(buf)
  med = statistics.median(samples)
  metadata = rows * (cols // 256) * METADATA_BYTES[quant]
  activation = cols * 4 + rows * 4
  total = size + activation
  row.update({
    "status": "measured", "tensor": name,
    "timing": {"candidate_us": med, "spread_pct": 100.0 * (_percentile(samples, 0.9) - _percentile(samples, 0.1)) / med,
               "min_us": min(samples), "samples": len(samples), "warmups": WARMUPS, "cache": "flushed",
               "flush_bytes": FLUSH_BYTES, "clock": "command buffer GPUStartTime/GPUEndTime",
               "dispatch_floor_us": floor_us},
    "bytes": {"physical_weight_bytes": size - metadata, "metadata_bytes": metadata, "activation_bytes": activation},
    "throughput": {"achieved_gbs": total / (med * 1e3), "target_gbs": target_gbs},
    "latency_concurrency": {},
    "isa": {},
    "resources": {"lds_bytes": k["static_threadgroup_memory_bytes"]},
    "kernel": {"name": KERNELS[quant], "threadgroup_threads": threads, "threadgroups": groups,
               "thread_execution_width": k["thread_execution_width"],
               "max_threads_per_threadgroup": k["max_threads_per_threadgroup"], "source": KERNEL_SOURCE},
    "absent": _absent(quant),
  })
  return row


def collect_probe_evidence(run:pathlib.Path, say:Callable[[str], None] = lambda _: None) -> dict[str, Any]:
  from boltbeam.runtime.metal_device import Metal, MetalUnavailable
  facts = _run_facts(run, need_bench=False)
  request = read_json(run / "probe_request.json")
  _, tensors, data_start = read_gguf_layout(facts["model"])
  try:
    metal = Metal()
  except MetalUnavailable as exc:
    raise CannotMeasure(str(exc), "measure on a Mac with a Metal GPU") from exc
  try:
    kernels = metal.compile(MSL, [*KERNELS.values(), "flush", "empty"])
    flush_buf = metal.buffer(length=FLUSH_BYTES)
    flush_groups = FLUSH_BYTES // 4 // 256
    empty_buf = metal.buffer(length=16)
    floor_us = statistics.median(metal.run(kernels["empty"]["pso"], [empty_buf], [], 1, 32) for _ in range(SAMPLES))
    target_gbs = facts["target"].memory_bandwidth_gbs
    rows = []
    for i, probe in enumerate(request.get("probes", []), start=1):
      say(f"probe {i} of {len(request['probes'])}: {probe['role']} {probe['quant']} {probe['shape'][0]}x{probe['shape'][1]}")
      rows.append(measure_probe(metal, kernels, probe, facts["model"], data_start, tensors, flush_buf, flush_groups,
                                floor_us, target_gbs))
  finally:
    metal.close()
  manifest = facts["manifest"]
  return {
    "schema": SCHEMA_PROBE_EVIDENCE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
    "workload": manifest["workload"], "provider_id": PROVIDER_ID, "collector_id": COLLECTOR_ID,
    "device": metal.name, "dispatch_floor_us": floor_us, "probes": rows,
    "measured": ["timing", "throughput.achieved_gbs", "resources.lds_bytes", "correctness"],
    "derived": ["bytes (from the GGUF block layout)", "throughput.target_gbs (from the target registry)"],
    "notes": ["Kernels are BoltBeam's MSL GEMVs on the model's real tensor bytes, not llama.cpp's kernels.",
              "Each timed sample follows a flush of a buffer larger than the system level cache.",
              "The command-buffer interval includes dispatch_floor_us of fixed GPU start cost; it is not subtracted."],
  }


# --- timing trace ------------------------------------------------------------------------------------------

def bench_decode(llama_bench:str, model:pathlib.Path, depth:int) -> dict[str, Any]:
  cmd = [llama_bench, "-m", str(model), "-p", "0", "-n", str(GEN_TOKENS), "-d", str(depth), "-ngl", "99",
         "-r", str(BENCH_REPS), "-o", "json"]
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
  if proc.returncode != 0:
    raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
  rows = json.loads(proc.stdout[proc.stdout.find("["):])
  row = next(r for r in rows if int(r.get("n_gen", 0)) > 0)
  return {"tok_s": float(row["avg_ts"]), "stddev_tok_s": float(row.get("stddev_ts", 0.0)), "command": cmd,
          "build": row.get("build_commit"), "backends": row.get("backends"), "gpu": row.get("gpu_info")}


def build_timing_trace(manifest:dict[str, Any], request:dict[str, Any], evidence:dict[str, Any],
                       bench:dict[int, dict[str, Any]], peak_gbs:float | None) -> dict[str, Any]:
  """The trace from the measured pieces. Pure: the tests feed it numbers."""
  by_key = {(p["role"], p["quant"], tuple(p["shape"])): p for p in evidence.get("probes", []) if p.get("status") == "measured"}
  rows: list[dict[str, Any]] = []
  for ctx, b in sorted(bench.items()):
    rows.append({"scope": "whole_step", "context": ctx, "wall_us": 1e6 / b["tok_s"], "tok_s": b["tok_s"],
                 "stddev_tok_s": b["stddev_tok_s"], "source": "llama-bench"})
    for role in request.get("priority_roles", []):
      probe = by_key.get((role["role"], role["quant"], tuple(role["shape"])))
      if probe is None:
        continue
      count = int(role.get("count") or 1)
      bytes_one = sum(probe["bytes"].values())
      us = probe["timing"]["candidate_us"] * count
      # a candidate row, not a kernel row: BoltBeam's kernel is a different program from the llama.cpp step above,
      # so it cannot be a share of that step. baseline_us stays absent: llama.cpp's own kernel time is not visible.
      rows.append({"scope": "candidate", "context": ctx, "candidate_id": f"boltbeam_{probe['kernel']['name']}",
                   "kernel": f"boltbeam_{probe['kernel']['name']}_{role['shape'][0]}x{role['shape'][1]}",
                   "role": role["role"], "quant": role["quant"], "shape": list(role["shape"]), "count": count,
                   "candidate_us": us, "spread_pct": probe["timing"]["spread_pct"], "phys_bytes": bytes_one * count,
                   "gbs": bytes_one * count / (us * 1e3), "token_match": None, "route_bound": None,
                   "source": "boltbeam metal-native probe, flushed, times count per token"})
  trace = {
    "schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
    "workload": manifest["workload"], "provider_id": PROVIDER_ID, "collector_id": COLLECTOR_ID,
    "timing_source": "llama-bench whole step; boltbeam metal-native GEMVs as role candidates",
    "contexts": sorted(bench), "rows": rows,
    "measured": ["whole_step.tok_s (llama-bench decode at each depth)", "candidate.candidate_us (BoltBeam GEMV, flushed, times count)"],
    "absent": ["kernel rows: llama.cpp's per-kernel time needs per-dispatch timing, which Metal gives only to Instruments",
               "attention, norm and RoPE: BoltBeam has no Metal kernel for them yet",
               "candidate baseline_us: there is no llama.cpp kernel time to compare against"],
    "notes": ["Candidate rows are BoltBeam's GEMVs timed alone on the real weights, one row per role per context.",
              "The GEMV does not depend on context, so the same measurement appears at every context."],
  }
  if peak_gbs:
    trace["peak_gbs"] = peak_gbs
  return trace


def collect_timing_trace(run:pathlib.Path, evidence:dict[str, Any], *, llama_bench:str = "llama-bench",
                         say:Callable[[str], None] = lambda _: None) -> dict[str, Any]:
  facts = _run_facts(run, llama_bench=llama_bench)
  request = read_json(run / "trace_request.json")
  if request.get("workload") != "decode":
    raise CannotMeasure("metal-native times decode only; this run is prefill", "plan the run with workload decode")
  bench = {}
  for ctx in request.get("contexts") or [0]:
    say(f"decode at depth {ctx}: llama-bench")
    bench[int(ctx)] = bench_decode(facts["llama_bench"], facts["model"], int(ctx))
  trace = build_timing_trace(facts["manifest"], request, evidence, bench, facts["target"].memory_bandwidth_gbs)
  trace["aux_sources"] = {"llama_bench": {str(k): v for k, v in bench.items()}}
  return trace


# --- the command line --------------------------------------------------------------------------------------

def _write(obj:dict[str, Any], path:pathlib.Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(pretty_json(obj))


def measure(run:pathlib.Path, *, only:str | None = None, probe_out:pathlib.Path | None = None,
            timing_out:pathlib.Path | None = None, llama_bench:str = "llama-bench",
            say:Callable[[str], None] = lambda _: None) -> dict[str, pathlib.Path]:
  probe_out = probe_out or run / "probe_evidence.json"
  timing_out = timing_out or run / "timing_trace.json"
  written = {}
  if only in (None, "probe"):
    _write(collect_probe_evidence(run, say), probe_out)
    written["probe"] = probe_out
  if only in (None, "timing"):
    evidence = read_json(probe_out) if probe_out.exists() else {"probes": []}
    _write(collect_timing_trace(run, evidence, llama_bench=llama_bench, say=say), timing_out)
    written["timing"] = timing_out
  return written


def add_arguments(p:argparse.ArgumentParser) -> None:
  p.add_argument("--run", required=True, help="run folder holding probe_request.json and trace_request.json")
  p.add_argument("--only", choices=["probe", "timing"], default=None)
  p.add_argument("--probe-out", default=None, help="default: <run>/probe_evidence.json")
  p.add_argument("--timing-out", default=None, help="default: <run>/timing_trace.json")
  p.add_argument("--llama-bench", default="llama-bench")


def run_cli(args:argparse.Namespace) -> int:
  run = run_dir(args.run)
  try:
    written = measure(run, only=args.only, llama_bench=args.llama_bench,
                      probe_out=pathlib.Path(args.probe_out).expanduser() if args.probe_out else None,
                      timing_out=pathlib.Path(args.timing_out).expanduser() if args.timing_out else None,
                      say=lambda line: print(line, flush=True))
  except CannotMeasure as exc:
    sys.stderr.write(f"metal-measure: cannot measure here: {exc.reason}\n  instead: {exc.command}\n")
    return 2
  except (FileNotFoundError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
    sys.stderr.write(f"metal-measure: {exc}\n")
    return 1
  for kind, path in written.items():
    print(f"{kind}: {path}")
  return 0


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog="python -m boltbeam.collectors.metal_native", description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  add_arguments(p)
  return run_cli(p.parse_args(argv))


if __name__ == "__main__":
  raise SystemExit(main())
