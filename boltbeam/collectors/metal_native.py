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
import struct
import sys
from typing import Any, Callable

from boltbeam.collectors import llama_bench_decode
from boltbeam.collectors.llama_bench_decode import CannotMeasure, bench_decode  # one refusal type for every collector
from boltbeam.core.canonical import pretty_json
from boltbeam.profile.gguf import read_gguf_layout
from boltbeam.target.targets import get_target
from boltbeam.vocab import SCHEMA_PROBE_EVIDENCE, SCHEMA_TIMING_TRACE
from boltbeam.workflow.common import load_manifest, read_json, run_dir

COLLECTOR_ID = "metal-native"
PROVIDER_ID = "boltbeam/metal-native"
GGML_TYPES = {"Q4_K": 12, "Q6_K": 14, "Q8_0": 8, "Q5_0": 6, "Q4_0": 2, "Q5_K": 13}
# the block layouts the pure-Python reference can dequantize (DEQUANT): bytes per block and weights per block, the
# same figures as data/quants.json's rows (block_bytes, block_elems) for these quants
BLOCK_BYTES = {"Q4_K": 144, "Q6_K": 210, "Q8_0": 34, "Q5_0": 22, "Q4_0": 18, "Q5_K": 176}
BLOCK_ELEMS = {"Q4_K": 256, "Q6_K": 256, "Q8_0": 32, "Q5_0": 32, "Q4_0": 32, "Q5_K": 256}
METADATA_BYTES = {"Q4_K": 16, "Q6_K": 18, "Q8_0": 2, "Q5_0": 6, "Q4_0": 2, "Q5_K": 48}  # Q4_K: d, dmin, 12 scale bytes. Q6_K: 16 int8 scales, d.
# Q8_0, Q4_0: d. Q5_0: d and the 4 bytes of fifth bits. Q5_K: d, dmin, 12 scale bytes, 32 bytes of fifth bits. The rest of a block is its codes (code_bytes).
WORKING_SET_SHARE = 0.8  # never ask for more than this share of Metal's recommended working set
CHECK_ROWS = 9
TOLERANCE = 1e-3  # max |gpu - cpu| over max |cpu|: float32 sums in a different order
# the kernels are adapter 0 of the one kernel timer (collectors/boltbeam_gemv.py); the loop is kernel_timer's
from boltbeam.collectors import boltbeam_gemv as gemv  # noqa: E402
from boltbeam.collectors import kernel_timer  # noqa: E402
MSL, KERNELS, ROWS_PER_GROUP, KERNEL_SOURCE = gemv.MSL, gemv.KERNELS, gemv.ROWS_PER_GROUP, gemv.KERNEL_SOURCE
SAMPLES, WARMUPS = kernel_timer.SAMPLES, kernel_timer.WARMUPS
FLUSH_BYTES = kernel_timer.FLUSH["Metal"]["bytes"]


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


def dequant_q8_0_block(blk:bytes) -> list[float]:
  d = _half(blk, 0)  # block_q8_0: half d, 32 int8 (ggml-common.h)
  return [d * q for q in struct.unpack_from("<32b", blk, 2)]


def dequant_q4_0_block(blk:bytes) -> list[float]:
  d = _half(blk, 0)  # block_q4_0: half d, 16 qs bytes; byte j holds element j (low nibble) and j+16 (high)
  qs = blk[2:18]
  return [d * ((qs[j] & 0xF) - 8) for j in range(16)] + [d * ((qs[j] >> 4) - 8) for j in range(16)]


def dequant_q5_0_block(blk:bytes) -> list[float]:
  d = _half(blk, 0)  # block_q5_0: half d, uint32 qh (bit j = fifth bit of element j), 16 qs bytes as Q4_0
  qh = struct.unpack_from("<I", blk, 2)[0]
  qs = blk[6:22]
  lo = [d * (((qs[j] & 0xF) | ((qh >> j) & 1) << 4) - 16) for j in range(16)]
  hi = [d * (((qs[j] >> 4) | ((qh >> (j + 16)) & 1) << 4) - 16) for j in range(16)]
  return lo + hi


def dequant_q5_k_block(blk:bytes) -> list[float]:
  d, dmin = _half(blk, 0), _half(blk, 2)  # block_q5_K: d, dmin, scales[12], qh[32], qs[128] (ggml-common.h)
  sc12, qh, qs = blk[4:16], blk[16:48], blk[48:176]
  out = [0.0] * 256
  for i in range(4):          # four 32-byte rows of qs, two groups of 32 each
    for hi in (0, 1):
      j = 2 * i + hi
      sc, m = _scale_min_k4(j, sc12)
      for l in range(32):
        q = ((qs[32 * i + l] >> (4 * hi)) & 0xF) | (((qh[l] >> j) & 1) << 4)
        out[64 * i + 32 * hi + l] = d * sc * q - dmin * m
  return out


def code_bytes(quant:str) -> int:
  """The bytes of a block that hold its codes, after its scales and extra bits."""
  return BLOCK_BYTES[quant] - METADATA_BYTES[quant]


DEQUANT: dict[str, Callable[[bytes], list[float]]] = {"Q4_K": dequant_q4_k_block, "Q6_K": dequant_q6_k_block, "Q8_0": dequant_q8_0_block,
                                                      "Q5_0": dequant_q5_0_block, "Q4_0": dequant_q4_0_block, "Q5_K": dequant_q5_k_block}


def reference_row(weights:bytes, quant:str, row:int, cols:int, x:list[float]) -> float:
  size, elems = BLOCK_BYTES[quant], BLOCK_ELEMS[quant]
  nb = cols // elems
  base = row * nb * size
  acc = 0.0
  for b in range(nb):
    vals = DEQUANT[quant](weights[base + b * size: base + (b + 1) * size])
    xb = x[b * elems:(b + 1) * elems]
    acc += sum(v * xv for v, xv in zip(vals, xb))
  return acc


def check_rows(rows:int) -> list[int]:
  return sorted({min(rows - 1, (rows - 1) * i // (CHECK_ROWS - 1)) for i in range(CHECK_ROWS)})


# --- where measuring is possible ---------------------------------------------------------------------------

def this_machine_target() -> str | None:
  from boltbeam.workflow.autoscan import this_machine_target as here
  return here()


def preflight(target_id:str, model:str | pathlib.Path, *, run:str | pathlib.Path = "RUN", need_bench:bool = True,
              llama_bench:str = "llama-bench", here:Callable[[], str | None] = this_machine_target) -> dict[str, Any]:
  """What the measurement will use, or CannotMeasure with the reason and the command to run instead."""
  target = get_target(target_id)
  if target.backend == "CUDA" and not need_bench:  # the probe alone: adapter 0 runs through the CUDA bridge
    path = pathlib.Path(model).expanduser()
    if not path.is_file():
      raise CannotMeasure(f"the model file is gone: {path}", "measure again with the model file in place")
    return {"target": target, "model": path, "llama_bench": None}
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
  bench = llama_bench_decode.find(llama_bench)
  if need_bench and bench is None:
    raise CannotMeasure("llama-bench is not installed, so there is no whole-step decode time",
                        f"brew install llama.cpp (or export {llama_bench_decode.ENV}=/path/to/llama-bench), "
                        "then measure again")
  return {"target": target, "model": path, "llama_bench": bench}


def _run_facts(run:pathlib.Path, **kw:Any) -> dict[str, Any]:
  manifest = load_manifest(run)
  facts = preflight(manifest.get("target_id"), manifest.get("model_path") or "", run=run, **kw)
  facts["manifest"] = manifest
  return facts


# --- probe evidence ----------------------------------------------------------------------------------------

def tensor_for(quant:str, rows:int, cols:int, tensors:list[tuple[str, tuple[int, ...], int, int]],
               tensor_name:str | None = None) -> tuple[str, int]:
  """The model tensor of this quant and shape (the named one when it is among them): its name and data offset.
  Shared by every collector that binds real weight bytes (the probes here, collectors/engine_kernels.py)."""
  want = GGML_TYPES.get(quant)
  matches = [(name, off) for name, dims, typ, off in tensors if typ == want and tuple(dims) == (cols, rows)]
  if not matches:
    raise ValueError(f"no {quant} tensor of shape {rows}x{cols} in the model")
  named = [m for m in matches if m[0] == tensor_name]
  return (named or matches)[0]


def _tensor_for(probe:dict[str, Any], tensors:list[tuple[str, tuple[int, ...], int, int]]) -> tuple[str, int]:
  rows, cols = (int(v) for v in probe["shape"])
  return tensor_for(probe["quant"], rows, cols, tensors, probe.get("tensor_name"))


def _percentile(values:list[float], q:float) -> float:
  ordered = sorted(values)
  return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


# What each backend's probe cannot report, by the field the full-probe schema asks for. Said in the row's `absent`
# so the next-step ladder never asks this GPU for it (report/html.py next_step).
_ABSENT = {
  "Metal": {
    "counters": "Metal reports no per-dispatch hardware counters without Instruments",
    "isa.instruction_histogram": "Metal does not export the compiled AGX ISA",
    "isa.vector_load_bits": "the compiled load width is not visible; the source loads are in kernel.source",
    "resources.registers": "MTLComputePipelineState does not report registers",
    "resources.scratch_bytes": "MTLComputePipelineState does not report scratch",
  },
  "CUDA": {
    "counters": "the CUDA bridge reads no per-dispatch hardware counters (that is Nsight Compute's)",
    "isa.instruction_histogram": "the cubin's SASS is not disassembled by BoltBeam",
    "isa.vector_load_bits": "the compiled load width is not visible; the source loads are in kernel.source",
    "resources.registers": "the CUDA bridge does not read cuFuncGetAttribute(NUM_REGS)",
    "resources.scratch_bytes": "the CUDA bridge does not read cuFuncGetAttribute(LOCAL_SIZE_BYTES)",
  },
}


def _absent(probe_quant:str, backend:str = "Metal") -> dict[str, str]:
  words = _ABSENT.get(backend) or {k: f"{backend} is not read by BoltBeam's probe" for k in _ABSENT["Metal"]}
  why = words["counters"]
  return {
    "latency_concurrency.memory_latency_ns": why, "latency_concurrency.active_waves": why,
    "latency_concurrency.in_flight_groups": why, "latency_concurrency.occupancy_pct": why,
    **{k: v for k, v in words.items() if k != "counters"},
  }


def measure_probe(bridge, flusher, probe:dict[str, Any], model:pathlib.Path, data_start:int, tensors:list,
                  floor_us:float, target_gbs:float | None, *, backend:str = "Metal",
                  libraries:dict[str, int] | None = None) -> dict[str, Any]:
  """One probe row: BoltBeam's GEMV (adapter 0) on the role's real tensor bytes, timed by the one kernel timer."""
  rows, cols = (int(v) for v in probe["shape"])
  quant = probe["quant"]
  row = {"probe_id": probe["probe_id"], "kind": "quant_gemv", "role": probe["role"], "quant": quant,
         "shape": [rows, cols]}
  if quant not in KERNELS or cols % BLOCK_ELEMS[quant]:
    row["status"] = "not_measured"
    row["reason"] = f"BoltBeam has no {backend} kernel for {quant} with {cols} columns"
    return row
  name, offset = _tensor_for(probe, tensors)
  size = rows * (cols // BLOCK_ELEMS[quant]) * BLOCK_BYTES[quant]
  if size + FLUSH_BYTES > bridge.working_set_bytes * WORKING_SET_SHARE:
    row["status"] = "not_measured"
    row["reason"] = f"{size / 2**30:.1f} GiB of weights does not fit the GPU's working set"
    return row
  with open(model, "rb") as f:
    f.seek(data_start + offset)
    weights = f.read(size)
  x = gemv.vector(f"{probe['probe_id']}", cols)
  spec = gemv.spec(backend, quant, rows, cols, weights, x)
  got = kernel_timer.time_spec(bridge, spec, flusher, libraries=libraries)
  row["correctness"] = got["correctness"]
  if not got["samples"]:
    row["status"] = "correctness_failed"
    return row
  med = got["median_us"]
  metadata = rows * (cols // 256) * METADATA_BYTES[quant]
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
    "kernel": gemv.facts(got, spec),
    "timed_by": spec.label,
    "absent": _absent(quant, backend),
  })
  return row


def collect_probe_evidence(run:pathlib.Path, say:Callable[[str], None] = lambda _: None) -> dict[str, Any]:
  """probe_request.json -> probe_evidence.json on this machine's GPU (Metal or CUDA): BoltBeam's own GEMV per role
  on the model's real bytes. The peak every row is measured against is the run's one read bandwidth."""
  from boltbeam.workflow.screen import run_bandwidth
  facts = _run_facts(run, need_bench=False)
  request = read_json(run / "probe_request.json")
  _, tensors, data_start = read_gguf_layout(facts["model"])
  backend = facts["target"].backend
  try:
    bridge = kernel_timer.bridge_for(backend)
  except RuntimeError as exc:  # MetalUnavailable, CudaUnavailable: this machine has no such GPU or no compiler
    raise CannotMeasure(str(exc), f"measure on a machine with a {backend} GPU") from exc
  try:
    flusher = kernel_timer.Flusher(bridge, backend)
    floor_us = flusher.floor_us()
    target_gbs, target_source = run_bandwidth(run, facts["target"])
    rows = []
    libraries:dict[str, int] = {}
    for i, probe in enumerate(request.get("probes", []), start=1):
      say(f"probe {i} of {len(request['probes'])}: {probe['role']} {probe['quant']} {probe['shape'][0]}x{probe['shape'][1]}")
      rows.append(measure_probe(bridge, flusher, probe, facts["model"], data_start, tensors, floor_us, target_gbs,
                                backend=backend, libraries=libraries))
  finally:
    bridge.close()
  manifest = facts["manifest"]
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


# --- timing trace ------------------------------------------------------------------------------------------

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
                         say:Callable[[str], None] = lambda _: None, batches=(1,)) -> dict[str, Any]:
  from boltbeam.workflow import progress
  facts = _run_facts(run, llama_bench=llama_bench)
  request = read_json(run / "trace_request.json")
  if request.get("workload") != "decode":
    raise CannotMeasure("metal-native times decode only; this run is prefill", "plan the run with workload decode")
  bench = {}
  contexts = request.get("contexts") or [0]
  parts = len(contexts) * (2 if any(int(b) > 1 for b in batches) else 1)
  progress.report(0, parts)
  for ctx in contexts:
    say(f"decode at depth {ctx}: llama-bench")
    bench[int(ctx)] = bench_decode(facts["llama_bench"], facts["model"], int(ctx), cwd=run)
    progress.report(len(bench), parts)
  from boltbeam.workflow.screen import run_bandwidth
  peak_gbs, peak_source = run_bandwidth(run, facts["target"])
  trace = build_timing_trace(facts["manifest"], request, evidence, bench, peak_gbs)
  trace["peak_source"] = peak_source
  trace["aux_sources"] = {"llama_bench": {str(k): v for k, v in bench.items()}}
  trace["batches"] = llama_bench_decode.batch_points(bench, facts["model"], batches, say=say)
  return trace


# --- the command line --------------------------------------------------------------------------------------

def _write(obj:dict[str, Any], path:pathlib.Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(pretty_json(obj))


def measure(run:pathlib.Path, *, only:str | None = None, probe_out:pathlib.Path | None = None,
            timing_out:pathlib.Path | None = None, llama_bench:str = "llama-bench",
            say:Callable[[str], None] = lambda _: None, batches=(1,)) -> dict[str, pathlib.Path]:
  probe_out = probe_out or run / "probe_evidence.json"
  timing_out = timing_out or run / "timing_trace.json"
  written = {}
  if only in (None, "probe"):
    _write(collect_probe_evidence(run, say), probe_out)
    written["probe"] = probe_out
  if only in (None, "timing"):
    evidence = read_json(probe_out) if probe_out.exists() else {"probes": []}
    _write(collect_timing_trace(run, evidence, llama_bench=llama_bench, say=say, batches=batches), timing_out)
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
