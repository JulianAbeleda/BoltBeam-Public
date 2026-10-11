"""Prefill, measured faithfully: the untraced time to read a prompt, what the engine batched, and where that time goes.

    pipeline MODEL --run RUN --target T --provider P --workload prefill --ctxs 2048 --analyze --measure auto --no-search

Step 4 (`measure`) answers trace_request.json with one whole_step row per prompt length: the prefill's wall time as
the engine runs it, untraced, at the engine's own chunk (llama.cpp's ubatch, the fork's prefill chunk), with the SM
clock, power and temperature logged through the run (gpu_telemetry) and a throttle flagged. What the engine batched
is recorded from the engine itself: llama-bench's own n_ubatch and n_batch, the fork's forward calls and their token
widths. An engine whose chunk is one token is "not batched prefill", said on the row.

Step 5 (`role_time`) captures one prefill per length in the model (nsys for an engine CUDA sees; the fork's own
GPU timestamps for the fork, which CUDA's tools cannot see), reads every kernel's matrix path from its code
(instruction_path), and pairs kernels with roles by launch counts and operations (prefill_attribution). The result
is prefill_trace.json; the where table is built from it (workflow/prefill.py).

ENGINES holds what differs per engine: how to time one prefill, how to capture one, where its kernels' code is.
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import subprocess
import time
from typing import Any, Callable

from boltbeam.collectors import gpu_telemetry

TRACE = "prefill_trace.json"
RAW = "kernel_compare/prefill_capture"
REPS = 3
NOT_BATCHED = "not batched prefill: the engine reads the prompt one token at a time"
DRIVER = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "tinygrad_prefill_profile.py"


def chunk_record(widths:list[Any], how:str, outputs:int | None = None) -> dict[str, Any]:
  """What the engine batched: the token width of each chunk its matrices run over, whether that is batching, and
  how many calls produce an output (the work for a call's last token, the output head, runs once per call: the fork
  calls once per chunk, llama.cpp once per n_batch tokens and splits that into ubatch chunks)."""
  ints = [w for w in widths if isinstance(w, int)]
  batched = bool(ints) and max(ints) > 1
  return {"widths": widths, "calls": len(widths), "outputs": outputs if outputs is not None else len(widths),
          "size": max(ints) if ints else None, "batched": batched,
          "words": (f"{len(widths)} forward call(s) of up to {max(ints)} tokens ({how})" if batched else
                    f"{NOT_BATCHED} ({how})") if ints else f"widths not readable ({how})"}


def split_chunks(length:int, size:int) -> list[int]:
  return [min(size, length - i) for i in range(0, length, size)]


# --- llama.cpp -------------------------------------------------------------------------------------------------

def _bench(run:pathlib.Path) -> str:
  from boltbeam.collectors.llama_bench_decode import ENV, CannotMeasure, find
  bench = find()
  if bench is None:
    raise CannotMeasure(f"llama-bench was not found in ${ENV} or on PATH", f"export {ENV}=/path/to/llama-bench")
  return bench


def llama_argv(bench:str, model:str, length:int, reps:int, *, warmup:bool = True, depth:int = 0, gen:int = 0) -> list[str]:
  """One prompt of `length` tokens, no generation, the engine's own ubatch and batch (its defaults, recorded back).
  depth: the prompt goes after that many tokens already in the cache (llama-bench -d, which fills them untimed);
  gen: also that many generated tokens at the depth, its own row."""
  return [bench, "-m", str(pathlib.Path(model).absolute()), "-p", str(length), "-n", str(gen), "-ngl", "99", "-r", str(reps),
          *(["-d", str(depth)] if depth else []), "-o", "json", *([] if warmup else ["--no-warmup"])]


def llama_row(text:str, field:str = "n_prompt") -> dict[str, Any]:
  """llama-bench's JSON row for the prompt (or, with field n_gen, for the generation), from its output (a capture's
  log has the tool's own lines after it)."""
  start = text.find("[")
  while start >= 0:
    try:
      rows = json.JSONDecoder().raw_decode(text[start:])[0]
      row = next((r for r in rows if isinstance(r, dict) and int(r.get(field, 0)) > 0), None)
      if row:
        return row
    except (ValueError, TypeError):
      pass
    start = text.find("[", start + 1)
  raise RuntimeError(f"llama-bench printed no row with {field}")


def llama_truth(run:pathlib.Path, model:str, length:int, *, device:int = 0) -> dict[str, Any]:
  bench = _bench(run)
  cmd = llama_argv(bench, model, length, REPS)
  with gpu_telemetry.Telemetry(device) as tele:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=run)
  if proc.returncode != 0:
    raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
  row = llama_row(proc.stdout)
  samples = [float(x) / 1e6 for x in row.get("samples_ns") or [row["avg_ns"]]]
  ub, b = int(row.get("n_ubatch") or 0), int(row.get("n_batch") or 0)
  chunk = chunk_record(split_chunks(length, ub) if ub else [], f"llama-bench n_ubatch {ub}, n_batch {b}",
                       outputs=-(-length // b) if b else None)
  return {"ms": statistics.median(samples), "samples_ms": samples, "chunk": chunk, "telemetry": tele.summary(),
          "telemetry_samples": gpu_telemetry.compact(tele.samples),
          "source": f"llama-bench -p {length} -n 0, {len(samples)} repetitions after its warm-up, untraced",
          "engine": {"build": row.get("build_commit"), "flash_attn": row.get("flash_attn"), "n_ubatch": ub, "n_batch": b,
                     "type_k": row.get("type_k"), "type_v": row.get("type_v"), "command": cmd,
                     "n_ctx": int(row.get("n_prompt") or length) + int(row.get("n_gen") or 0),
                     "n_ctx_source": "llama-bench sizes its context to n_prompt + n_gen + n_depth",
                     "prompt": PROMPT_WORDS["llama.cpp"]}}


LLAMA_TILE_ROWS = 128  # mmq_y: the weight rows of one mul_mat_q output tile on a CUDA matrix-instruction build
LLAMA_K_ITER = 256  # MMQ_ITER_K: the K elements one stream-k step covers


def llama_stream_k(tmpl:str) -> dict[str, int] | None:
  """llama.cpp's stream-k layout for a mul_mat_q fixup template: its second template argument is the tile's token
  columns (mmq_x); None for any other fixup (flash attention's has its own layout, not modelled)."""
  import re
  m = re.match(r"mul_mat_q_stream_k_fixup<[^,]+,(\d+),", tmpl.replace(" ", ""))
  return {"tile_m": int(m.group(1)), "tile_n": LLAMA_TILE_ROWS, "k_iter": LLAMA_K_ITER} if m else None


def llama_formats(type_k:str | None) -> dict[str, Any]:
  """What llama.cpp's prefill kernels read and write: mul_mat_q reads activations quantized to q8_1 and writes fp32,
  flash attention reads fp32 Q and the cache in its type_k and writes fp32, stream-k partials are fp32."""
  return {"activations": {"in": "Q8_1", "out": "F32"}, "attention_activations": {"in": "F32", "out": "F32"},
          "kv": str(type_k or "f16").upper(), "partials": "F32", "stream_k": llama_stream_k,
          "words": "mul_mat_q: q8_1 in, fp32 out; flash attention: fp32 Q and output, the cache in type_k; fp32 partials"}


PROMPT_WORDS = {"llama.cpp": "llama-bench's own prompt: token ids drawn by std::rand() % n_vocab (never seeded: one fixed "
                            "sequence; each repetition draws new ids), the KV cache cleared before every repetition, so "
                            "nothing is cached between prompts"}
MAX_CAPTURE_LAUNCHES = 1_000_000  # a capture estimated to hold more launches than this is not taken: its time stands alone


def llama_capture(run:pathlib.Path, model:str, length:int, target, depth:int = 0) -> dict[str, Any]:
  """One prefill captured after the warm-up; at a depth, the depth filled then the prompt twice (llama-bench keeps
  the filled cache for its second repetition), the last one measured (prefill_attribution.REPEAT)."""
  from boltbeam.collectors import vendor_capture
  from boltbeam.collectors.prefill_attribution import REPEAT
  bench = _bench(run)
  raw = run / RAW / (f"{length}-at-{depth}" if depth else str(length))
  launches = vendor_capture.capture(target.backend, llama_argv(bench, model, length, 2 if depth else 1, depth=depth), raw)
  row = llama_row(vendor_capture.program_output(raw))
  ub, b = int(row.get("n_ubatch") or 0), int(row.get("n_batch") or 0)
  return {"launches": launches, "chunk": split_chunks(length, ub), "outputs": -(-length // b) if b else 1,
          "warmup": REPEAT if depth else True, "formats": llama_formats(row.get("type_k")),
          "method": vendor_capture.plan(target.backend)["method"],
          "paths": library_paths(pathlib.Path(bench).resolve().parent), "captured_ms": float(row["avg_ns"]) / 1e6}


def library_paths(folder:pathlib.Path) -> dict[str, Any]:
  """The matrix paths of every kernel in the engine's own libraries: each shared library beside its binary that
  holds device code (a library without any is skipped)."""
  from boltbeam.collectors.instruction_path import library_table
  kernels: dict[str, dict[str, int]] = {}
  read = []
  for lib in sorted(folder.glob("*.so*")):
    if lib.is_symlink():
      continue
    try:
      table = library_table(lib)
    except (RuntimeError, OSError):
      continue
    if table["kernels"]:
      kernels.update(table["kernels"])
      read.append({"library": str(lib), "sha256": table["sha256"], "kernels": len(table["kernels"])})
  return {"kernels": kernels, "libraries": read, "how": "PTX embedded in the engine's libraries (cuobjdump --dump-ptx)"}


# --- the tinygrad fork -------------------------------------------------------------------------------------------

def _fork(root:pathlib.Path | None):
  from boltbeam.search import role_compare
  root = pathlib.Path(root) if root else role_compare.default_fork_root()
  return root, role_compare.fork_python(root)


EAGER_SAMPLES = 1  # an eager prefill schedules its graph again: minutes at long prompts, so one is timed after the warm one


def _fork_run(root, python, target, model, length, mode, out:pathlib.Path | None = None) -> dict[str, Any]:
  """The driver, replaying the captured graph; when the captured graphs do not fit in GPU memory, again in the
  fork's default eager mode, and the result says so (replay False, why)."""
  from boltbeam.collectors.tinygrad_role_time import DecodeFailed
  try:
    return _fork_once(root, python, target, model, length, mode, out, eager=False)
  except DecodeFailed as exc:
    if "MemoryError" not in str(exc) and "Allocation" not in str(exc):
      raise
    got = _fork_once(root, python, target, model, length, mode, out, eager=True)
    got["replay_failed"] = str(exc)[:300]
    return got


def _fork_once(root, python, target, model, length, mode, out, *, eager:bool) -> dict[str, Any]:
  argv = ["--model", model, "--length", str(length), "--mode", mode]
  if out is not None:
    argv += ["--out", str(out)]
  if eager:
    argv += ["--eager", "--samples", str(EAGER_SAMPLES)]
  return fork_exec(root, python, target, DRIVER, argv, mode, "prefill")


def fork_exec(root, python, target, driver:pathlib.Path, argv:list[str], mode:str, what:str) -> dict[str, Any]:
  """A fork driver run by the fork's python in the fork's folder: untraced (truth) or with every launch timed
  (capture); its last JSON line."""
  from boltbeam.collectors.tinygrad_role_time import DecodeFailed, device_for
  env = {**os.environ, "PYTHONPATH": ".", "DEV": device_for(target), "VIZ": "0",
         **({"JIT": "1", "PROFILE": "0"} if mode == "truth" else {"JIT": "2", "PROFILE": "1"})}
  proc = subprocess.run([str(python), str(driver), *argv], cwd=root, capture_output=True, text=True, timeout=7200, env=env)
  lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise DecodeFailed(f"tinygrad {what} ({mode}) failed", proc.stderr or proc.stdout, root)
  return json.loads(lines[-1])


def tinygrad_truth(run:pathlib.Path, model:str, length:int, *, target, root=None, device:int = 0) -> dict[str, Any]:
  root, python = _fork(root)
  with gpu_telemetry.Telemetry(device) as tele:
    got = _fork_run(root, python, target, model, length, "truth")
  return {"ms": got["wall_s"] * 1e3, "samples_ms": [s * 1e3 for s in got["samples_s"]],
          "chunk": chunk_record(got["chunks"], "the fork's forward calls during the prompt"), "telemetry": tele.summary(),
          "telemetry_samples": gpu_telemetry.compact(tele.samples),
          "source": (f"the fork's generate() from its first forward call to the first token, {len(got['samples_s'])} "
                     + ("prefill(s) replayed from the captured graph, untraced" if got.get("replay", True) else
                        "eager prefill(s), the fork's default (its graph scheduled again: the replay ran out of GPU memory), "
                        "untraced")),
          "replay": got.get("replay", True), "replay_failed": got.get("replay_failed"),
          "engine": {"eager_first_prompt_ms": got["warm_s"] * 1e3,
                     "capture_ms": got["capture_s"] * 1e3 if got.get("capture_s") is not None else None,
                     "workload_reuse_default": got.get("workload_reuse_default"),
                     "eager_words": ("the fork's default for an independent prompt is eager: its graph is scheduled "
                                     "again on every prompt (workload reuse not admitted); eager_first_prompt_ms is that, "
                                     "with compiles")}}


def tinygrad_capture(run:pathlib.Path, model:str, length:int, target, *, root=None) -> dict[str, Any]:
  root, python = _fork(root)
  raw = run / RAW / str(length)
  raw.mkdir(parents=True, exist_ok=True)
  got = _fork_run(root, python, target, model, length, "capture", raw / "launches.json")
  return fork_capture(raw, got)


def fork_capture(raw:pathlib.Path, got:dict[str, Any]) -> dict[str, Any]:
  """A fork driver's capture (launches.json in raw, its JSON line got) as the attribution reads it."""
  from boltbeam.collectors.instruction_path import paths_in_text, template
  (raw / "driver.json").write_text(json.dumps(got))
  data = json.loads((raw / "launches.json").read_text())
  # a source with no kernel body in it (a precompiled binary's stub) does not say the program's path: left out, so
  # its ceiling is refused, named, rather than read as "no matrix instruction"
  kernels = {template(name): paths_in_text(src) for name, src in (data.get("sources") or {}).items() if is_kernel_code(src)}
  from boltbeam.artifacts.tinygrad_profile_events import _census_program_info
  meta = {template(name): info for (name, binary), info in _census_program_info(data.get("census")).items()
          if binary is None and info.get("role") not in (None, "mixed") and info.get("shape")}
  widths = [w for w in data["chunks"] if isinstance(w, int)]
  return {"launches": data["launches"], "chunk": widths, "outputs": len(widths), "warmup": False,
          "method": "tinygrad-profile-events", "captured_ms": got["wall_s"] * 1e3, "program_roles": meta, "own_timing": True,
          "paths": {"kernels": kernels, "libraries": [], "how": "each program's own source, as the fork's JIT captured it"}}


def is_kernel_code(src:str) -> bool:
  """Whether a program's source text holds its kernel: a CUDA C kernel or a PTX entry."""
  return "__global__" in src or ".entry" in src


ENGINES: dict[str, dict[str, Callable[..., dict[str, Any]]]] = {
  "llama.cpp": {"truth": lambda run, model, length, target, root: llama_truth(run, model, length),
                "capture": lambda run, model, length, target, root: llama_capture(run, model, length, target)},
  "tinygrad": {"truth": lambda run, model, length, target, root: tinygrad_truth(run, model, length, target=target, root=root),
               "capture": lambda run, model, length, target, root: tinygrad_capture(run, model, length, target, root=root)},
}


def _engine(provider:str) -> dict[str, Callable[..., dict[str, Any]]]:
  if provider not in ENGINES:
    raise RuntimeError(f"prefill is measured for {', '.join(ENGINES)}; {provider} has no prefill adapter")
  return ENGINES[provider]


# --- step 4 and step 5 -----------------------------------------------------------------------------------------

def measure(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None) -> pathlib.Path:
  """trace_request.json answered with one untraced prefill per requested prompt length."""
  from boltbeam.core.canonical import pretty_json
  from boltbeam.target.targets import get_target
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  from boltbeam.workflow.common import load_manifest, read_json
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  rows = []
  for length in read_json(run / "trace_request.json").get("contexts") or []:
    say(f"prefill of {length} tokens: {provider}")
    began = time.monotonic()
    try:
      got = _engine(provider)["truth"](run, str(manifest["model_path"]), int(length), target, root)
    except RuntimeError as exc:  # a length the engine could not read (out of memory at the cap): said, the rest go on
      say(f"prefill of {length} tokens failed: {str(exc)[-200:]}")
      rows.append({"scope": "whole_step", "workload": "prefill", "context": int(length), "prompt_tokens": int(length),
                   "failed": str(exc)[-600:], "seconds": round(time.monotonic() - began, 1)})
      continue
    tele = got["telemetry"]
    rows.append({"scope": "whole_step", "workload": "prefill", "context": int(length), "prompt_tokens": int(length),
                 "wall_us": got["ms"] * 1e3, "prefill_ms": got["ms"], "tok_s": int(length) / got["ms"] * 1e3,
                 "samples_ms": got["samples_ms"], "chunk": got["chunk"], "batched": got["chunk"]["batched"],
                 **({"replay": got["replay"], "replay_failed": got.get("replay_failed")} if "replay" in got else {}),
                 "telemetry": tele, "throttled": tele.get("throttled"), "telemetry_samples": got.get("telemetry_samples"),
                 "source": got["source"], "engine": got["engine"],
                 "seconds": round(time.monotonic() - began, 1)})
  if not any(r.get("prefill_ms") for r in rows):
    raise RuntimeError("no prompt length was read: " + "; ".join(f"{r['context']}: {r['failed'][-160:]}" for r in rows))
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
           "workload": "prefill", "provider": provider, "provider_id": provider, "collector_id": "prefill",
           "timing_source": "untraced prefill wall time", "contexts": [r["context"] for r in rows], "rows": rows,
           "clock_policy": gpu_telemetry.clock_policy(0),
           "measured": ["whole_step.prefill_ms (the prompt read untraced at the engine's own chunk)"],
           **({"sweep": read_json(run / "prefill_sweep.json")} if (run / "prefill_sweep.json").exists() else {})}
  out = run / "timing_trace.json"
  out.write_text(pretty_json(trace))
  return out


def last_level_cache(target) -> dict[str, float] | None:
  """The chip's last-level cache: its size and measured read bandwidth, from the target's facts; None if either is
  not recorded."""
  facts = (target.capabilities or {}).get("fact_sources") or {}
  size = next((v.get("l2_cache_bytes") for v in facts.values() if isinstance(v, dict) and v.get("l2_cache_bytes")), None)
  gbs = (facts.get("l2_bandwidth_gbs") or {}).get("value_gbs")
  return {"bytes": float(size), "gbs": float(gbs)} if size and gbs else None


def prefill_roles(profile:dict[str, Any], ceil_roles:list[dict[str, Any]]) -> list[dict[str, Any]]:
  """The profile's weight roles with their shape, launches per chunk and weight bytes per launch (the limit's own
  bytes over the role's count, so a role's bytes are the decode table's)."""
  from boltbeam.role_key import role_key
  by = {role_key(r): r for r in ceil_roles}
  out = []
  for r in profile.get("roles") or []:
    c = by.get(role_key(r))
    if c and r.get("count") and r.get("rows") and r.get("cols"):
      out.append({"role": r["role"], "quant": r["quant"], "n": int(r["rows"]), "k": int(r["cols"]), "count": int(r["count"]),
                  "weight_bytes": float(c["bytes_moved"]) / int(r["count"])})
  return out


def role_time(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None, depth:int = 0,
              lengths:list[int] | None = None, truth_ms:dict[int, float] | None = None,
              capture:Callable[..., dict[str, Any]] | None = None, trace:str = TRACE) -> dict[str, Any]:
  """One captured prefill per prompt length, attributed: prefill_trace.json. A session's turn is the same at a depth:
  its lengths, its untraced turn times, its own capture and trace file (collectors/session.py)."""
  from boltbeam.collectors.prefill_attribution import attribute_prefill
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest, read_json
  from boltbeam.workflow.screen import _measured_vs_ceiling, _optional, run_bandwidth
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  profile = _optional(run, "model_profile.json")
  ceil = _measured_vs_ceiling({**manifest, "workload": "decode"}, profile, run)
  bw, bw_source = run_bandwidth(run, target)
  roles = prefill_roles(profile, ceil.get("_roles") or [])
  # the engine's own timing slows its run (one command buffer per kernel): its idle is profiling, not the engine's,
  # so its gaps are the untraced prefill less the busy time, as the decode tie-out treats its own timing
  truth_rows = [r for r in read_json(run / "timing_trace.json").get("rows") or [] if r.get("prefill_ms")]
  if truth_ms is None:
    truth_ms = {int(r["context"]): r["prefill_ms"] for r in truth_rows}
  truth_chunk = {int(r["context"]): r.get("chunk") or {} for r in truth_rows}
  out = {"schema": "boltbeam.prefill_trace.v1", "provider": provider, "model_id": manifest["model_id"],
         "target_id": target.target_id, "bandwidth_gbs": bw, "bandwidth_source": bw_source,
         "peaks": dict(target.matrix_tflops or {}), "peaks_source": ((target.capabilities or {}).get("fact_sources") or {}).get("matrix_tflops"),
         "lengths": {}, **({"depth": depth} if depth else {})}
  per_chunk = None  # launches per engine chunk, from the last capture: the next capture's size is estimated from it
  for length in lengths or read_json(run / "trace_request.json").get("contexts") or []:
    if int(length) not in truth_ms:
      out["lengths"][str(length)] = {"not_captured": "the untraced prefill failed at this length (timing_trace.json)"}
      continue
    chunks = len(split_chunks(int(length), (truth_chunk.get(int(length)) or {}).get("size") or int(length)))
    if per_chunk and 2 * per_chunk * chunks > MAX_CAPTURE_LAUNCHES:
      say(f"prefill of {length} tokens: not captured (about {2 * per_chunk * chunks:,.0f} launches)")
      out["lengths"][str(length)] = {"not_captured": f"about {2 * per_chunk * chunks:,.0f} launches with the warm-up, over "
                                                     f"{MAX_CAPTURE_LAUNCHES:,}: the untraced time, MFU and clock stand alone"}
      continue
    say(f"prefill of {length} tokens in the model{f' after {depth} in the cache' if depth else ''}: {provider}")
    with gpu_telemetry.Telemetry(0) as tele:
      cap = (capture or _engine(provider)["capture"])(run, str(manifest["model_path"]), int(length), target, root)
    attr = attribute_prefill(cap["launches"], [dict(r) for r in roles], chunk_widths=cap["chunk"], outputs=cap["outputs"],
                             layers=int(profile.get("layer_count") or 0), kernel_paths=cap["paths"]["kernels"],
                             peaks=dict(target.matrix_tflops or {}), bandwidth_gbs=bw, profile=profile, length=int(length),
                             program_roles=cap.get("program_roles"), warmup=cap["warmup"],
                             untraced_ms=(truth_ms.get(int(length)) if cap.get("own_timing") else None),
                             formats=cap.get("formats"), cache=last_level_cache(target), depth=depth)
    per_chunk = len(cap["launches"]) / 2 / max(len(cap["chunk"]), 1) if cap["warmup"] else len(cap["launches"]) / max(len(cap["chunk"]), 1)
    out["lengths"][str(length)] = {"formats": (cap.get("formats") or {}).get("words") or "fp16 activations in and out (the adapter names none)",
                                   "method": cap["method"], "captured_ms": cap.get("captured_ms"), "telemetry": tele.summary(),
                                   "paths_how": cap["paths"]["how"], "libraries": cap["paths"]["libraries"], **attr}
  (run / trace).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
  return out
