"""Prefill, measured faithfully: the untraced time to read a prompt, what the engine batched, and where that time goes.

    pipeline MODEL --run RUN --target T --provider P --workload prefill --ctxs 2048 --analyze --measure auto --no-search

Step 4 (`measure`) answers trace_request.json with one whole_step row per prompt length: the prefill's wall time as
the engine runs it, untraced, at the engine's own chunk (llama.cpp's ubatch, the fork's prefill chunk), with the SM
clock, power and temperature logged through the run (gpu_telemetry) and a throttle flagged. What the engine batched
is recorded from the engine itself: llama-bench's own n_ubatch and n_batch, the fork's forward calls and their token
widths. An engine whose chunk is one token is "not batched prefill", said on the row.

ENGINES holds what differs per engine: how to time one prefill.
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


# --- the tinygrad fork -------------------------------------------------------------------------------------------

def _fork(root:pathlib.Path | None):
  from boltbeam.search import role_compare
  root = pathlib.Path(root) if root else role_compare.default_fork_root()
  return root, role_compare.fork_python(root)


def _fork_run(root, python, target, model, length, mode, out:pathlib.Path | None = None) -> dict[str, Any]:
  from boltbeam.collectors.tinygrad_role_time import DecodeFailed, device_for
  argv = [str(python), str(DRIVER), "--model", model, "--length", str(length), "--mode", mode]
  if out is not None:
    argv += ["--out", str(out)]
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
                     "prefills replayed from the captured graph, untraced"),
          "engine": {"eager_first_prompt_ms": got["warm_s"] * 1e3, "capture_ms": got["capture_s"] * 1e3,
                     "workload_reuse_default": got.get("workload_reuse_default"),
                     "eager_words": ("the fork's default for an independent prompt is eager: its graph is scheduled "
                                     "again on every prompt (workload reuse not admitted); eager_first_prompt_ms is that, "
                                     "with compiles")}}


ENGINES: dict[str, dict[str, Callable[..., dict[str, Any]]]] = {
  "llama.cpp": {"truth": lambda run, model, length, target, root: llama_truth(run, model, length)},
  "tinygrad": {"truth": lambda run, model, length, target, root: tinygrad_truth(run, model, length, target=target, root=root)},
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


