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


def llama_argv(bench:str, model:str, length:int, reps:int, *, warmup:bool = True) -> list[str]:
  """One prompt of `length` tokens, no generation, the engine's own ubatch and batch (its defaults, recorded back)."""
  return [bench, "-m", str(pathlib.Path(model).absolute()), "-p", str(length), "-n", "0", "-ngl", "99", "-r", str(reps),
          "-o", "json", *([] if warmup else ["--no-warmup"])]


def llama_row(text:str) -> dict[str, Any]:
  """llama-bench's JSON row for the prompt, from its output (a capture's log has the tool's own lines after it)."""
  start = text.find("[")
  while start >= 0:
    try:
      rows = json.JSONDecoder().raw_decode(text[start:])[0]
      row = next((r for r in rows if isinstance(r, dict) and int(r.get("n_prompt", 0)) > 0), None)
      if row:
        return row
    except (ValueError, TypeError):
      pass
    start = text.find("[", start + 1)
  raise RuntimeError("llama-bench printed no prompt row")


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
                     "type_k": row.get("type_k"), "command": cmd}}


def llama_capture(run:pathlib.Path, model:str, length:int, target) -> dict[str, Any]:
  from boltbeam.collectors import vendor_capture
  bench = _bench(run)
  raw = run / RAW / str(length)
  launches = vendor_capture.capture(target.backend, llama_argv(bench, model, length, 1), raw)
  row = llama_row(vendor_capture.program_output(raw))
  ub, b = int(row.get("n_ubatch") or 0), int(row.get("n_batch") or 0)
  return {"launches": launches, "chunk": split_chunks(length, ub), "outputs": -(-length // b) if b else 1, "warmup": True,
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
  from boltbeam.collectors.tinygrad_role_time import DecodeFailed, device_for
  argv = [str(python), str(DRIVER), "--model", model, "--length", str(length), "--mode", mode]
  if out is not None:
    argv += ["--out", str(out)]
  if eager:
    argv += ["--eager", "--samples", str(EAGER_SAMPLES)]
  env = {**os.environ, "PYTHONPATH": ".", "DEV": device_for(target), "VIZ": "0",
         **({"JIT": "1", "PROFILE": "0"} if mode == "truth" else {"JIT": "2", "PROFILE": "1"})}
  proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=7200, env=env)
  lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise DecodeFailed(f"tinygrad prefill ({mode}) failed", proc.stderr or proc.stdout, root)
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
  from boltbeam.collectors.instruction_path import paths_in_text, template
  root, python = _fork(root)
  raw = run / RAW / str(length)
  raw.mkdir(parents=True, exist_ok=True)
  got = _fork_run(root, python, target, model, length, "capture", raw / "launches.json")
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
    got = _engine(provider)["truth"](run, str(manifest["model_path"]), int(length), target, root)
    tele = got["telemetry"]
    rows.append({"scope": "whole_step", "workload": "prefill", "context": int(length), "prompt_tokens": int(length),
                 "wall_us": got["ms"] * 1e3, "prefill_ms": got["ms"], "tok_s": int(length) / got["ms"] * 1e3,
                 "samples_ms": got["samples_ms"], "chunk": got["chunk"], "batched": got["chunk"]["batched"],
                 **({"replay": got["replay"], "replay_failed": got.get("replay_failed")} if "replay" in got else {}),
                 "telemetry": tele, "throttled": tele.get("throttled"), "telemetry_samples": got.get("telemetry_samples"),
                 "source": got["source"], "engine": got["engine"],
                 "seconds": round(time.monotonic() - began, 1)})
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
           "workload": "prefill", "provider": provider, "provider_id": provider, "collector_id": "prefill",
           "timing_source": "untraced prefill wall time", "contexts": [r["context"] for r in rows], "rows": rows,
           "clock_policy": gpu_telemetry.clock_policy(0),
           "measured": ["whole_step.prefill_ms (the prompt read untraced at the engine's own chunk)"]}
  out = run / "timing_trace.json"
  out.write_text(pretty_json(trace))
  return out


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


def role_time(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None) -> dict[str, Any]:
  """One captured prefill per prompt length, attributed: prefill_trace.json."""
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
  truth_ms = {int(r["context"]): r["prefill_ms"] for r in read_json(run / "timing_trace.json").get("rows") or []
              if r.get("prefill_ms")}
  out = {"schema": "boltbeam.prefill_trace.v1", "provider": provider, "model_id": manifest["model_id"],
         "target_id": target.target_id, "bandwidth_gbs": bw, "bandwidth_source": bw_source,
         "peaks": dict(target.matrix_tflops or {}), "peaks_source": ((target.capabilities or {}).get("fact_sources") or {}).get("matrix_tflops"),
         "lengths": {}}
  for length in read_json(run / "trace_request.json").get("contexts") or []:
    say(f"prefill of {length} tokens in the model: {provider}")
    with gpu_telemetry.Telemetry(0) as tele:
      cap = _engine(provider)["capture"](run, str(manifest["model_path"]), int(length), target, root)
    attr = attribute_prefill(cap["launches"], [dict(r) for r in roles], chunk_widths=cap["chunk"], outputs=cap["outputs"],
                             layers=int(profile.get("layer_count") or 0), kernel_paths=cap["paths"]["kernels"],
                             peaks=dict(target.matrix_tflops or {}), bandwidth_gbs=bw, profile=profile, length=int(length),
                             program_roles=cap.get("program_roles"), warmup=cap["warmup"],
                             untraced_ms=(truth_ms.get(int(length)) if cap.get("own_timing") else None))
    out["lengths"][str(length)] = {"method": cap["method"], "captured_ms": cap.get("captured_ms"), "telemetry": tele.summary(),
                                   "paths_how": cap["paths"]["how"], "libraries": cap["paths"]["libraries"], **attr}
  (run / TRACE).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
  return out
