"""Whole-step decode timing with llama-bench, on any GPU llama.cpp runs on (collector `llama-bench-decode`).

    pipeline MODEL --run RUN --target nvidia_sm120 --measure auto

It answers trace_request.json with a boltbeam.timing_trace.v1 holding one whole_step row per requested depth:
llama-bench decode (`-p 0 -n N -d CTX`), the measured tokens per second of a real decode on this GPU. The Metal
collector (metal_native.py) takes the same rows from here.

It writes no probe evidence and no kernel rows. BoltBeam's building-block probe kernels are Metal (MSL) only, and
no fork path writes boltbeam.probe_evidence.v1, so on other GPUs the probe request stays open and says why.

llama-bench is found in this order: the --llama-bench path given, $BOLTBEAM_LLAMA_BENCH, then PATH.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from typing import Any, Callable

from boltbeam.core.canonical import pretty_json
from boltbeam.target.targets import get_target
from boltbeam.vocab import SCHEMA_TIMING_TRACE
from boltbeam.workflow.common import load_manifest, read_json

COLLECTOR_ID = "llama-bench-decode"
PROVIDER_ID = "llama.cpp/llama-bench-decode"
PROVIDER = "llama.cpp"  # the provider adapter's name (collectors/providers.py)
TRACE = "llama_timing_trace.json"  # per-role time of llama.cpp's own kernels, from a vendor capture
RAW = "kernel_compare/llama_capture"
ROLE_TOKENS = 64  # decode tokens in the captured window
ENV = "BOLTBEAM_LLAMA_BENCH"
DEFAULT = "llama-bench"
GEN_TOKENS, BENCH_REPS = 32, 3
PROBE_ABSENT = ("no building-block probe for this chip: BoltBeam's probe kernels are Metal only, "
                "and no fork path writes probe evidence")


class CannotMeasure(RuntimeError):
  """Measuring is not possible here. `reason` says why; `command` is what to run instead."""

  def __init__(self, reason:str, command:str) -> None:
    super().__init__(reason)
    self.reason, self.command = reason, command


def find(llama_bench:str = DEFAULT, env:dict[str, str] | None = None) -> str | None:
  """The llama-bench to run: an explicit path, then $BOLTBEAM_LLAMA_BENCH, then PATH. None when there is none."""
  env = os.environ if env is None else env
  if llama_bench != DEFAULT:
    return llama_bench if pathlib.Path(llama_bench).expanduser().is_file() else shutil.which(llama_bench)
  if env.get(ENV):
    path = pathlib.Path(env[ENV]).expanduser()
    return str(path) if path.is_file() else None
  return shutil.which(DEFAULT)


def available(env:dict[str, str] | None = None) -> str | None:
  """Why llama.cpp cannot measure here, or None when it can."""
  if find(DEFAULT, env) is None:
    return f"llama-bench was not found in ${ENV} or on PATH"
  return None


def decode_argv(llama_bench:str, model:pathlib.Path | str, depth:int, tokens:int, extra:list[str] | None = None) -> list[str]:
  """One decode of `tokens` tokens after a `depth`-token context: no warmup, one repetition, so a capture of
  this command holds exactly that work."""
  return [llama_bench, "-m", str(model), "-p", "0", "-n", str(tokens), "-d", str(depth), "-ngl", "99", "-r", "1",
          "--no-warmup", "-o", "json", *(extra or [])]


def role_time(run:pathlib.Path, *, target, model:str, model_id:str, context:int = 128, tokens:int = ROLE_TOKENS,
              llama_bench:str = DEFAULT, layout:str = "one", gpus:int = 1) -> dict[str, Any]:
  """llama.cpp's kernels per role, captured from outside by the GPU vendor's tool (collectors/vendor_capture.py)
  and attributed by bytes and count (collectors/attribution.py). Two captures at the same depth, one with
  `tokens` decode tokens and one with 1, are subtracted per kernel: what is left is exactly tokens - 1 decode
  tokens, with the context fill, the setup and the first token removed."""
  from boltbeam.collectors import attribution, vendor_capture
  from boltbeam.workflow import layout as lay
  if layout == "row":
    raise attribution.CaptureRefused("split by rows runs every weight kernel on every GPU at once; per-role "
                                     f"attribution across devices is not built, so this run has the whole step only ({lay.LIMITED})")
  bench = find(llama_bench)
  if bench is None:
    raise CannotMeasure(available() or "llama-bench is missing", f"export {ENV}=/path/to/llama-bench")
  extra, env = lay.engine_args(layout, PROVIDER) if gpus > 1 else ([], {})
  raw = run / RAW
  long = vendor_capture.capture(target.backend, decode_argv(bench, model, context, tokens, extra), raw / "long", env=env)
  short = vendor_capture.capture(target.backend, decode_argv(bench, model, context, 1, extra), raw / "short", env=env)
  window, overlap = attribution.shared_window(long, short)
  method = vendor_capture.plan(target.backend)["method"]
  tok_s = bench_tok_s(vendor_capture.program_output(raw / "long"))
  return attribution.provider_trace(run, window, provider=PROVIDER, method=method, model_id=model_id,
                                    target=target, context=context, tokens=tokens - 1, out=TRACE,
                                    token_ms=1000.0 / tok_s if tok_s else None, long_kernels=len(long),
                                    overlap=overlap)


def bench_tok_s(text:str) -> float | None:
  """The decode tokens per second llama-bench printed in its JSON (`avg_ts` of the n_gen row), or None."""
  start = text.find("[")
  while start >= 0:
    try:
      rows = json.JSONDecoder().raw_decode(text[start:])[0]
      row = next((r for r in rows if isinstance(r, dict) and int(r.get("n_gen", 0)) > 0), None)
      if row:
        return float(row["avg_ts"])
    except (ValueError, TypeError, KeyError):
      pass
    start = text.find("[", start + 1)
  return None


def bench_decode(llama_bench:str, model:pathlib.Path, depth:int, layout_args:list[str] | None = None,
                 env:dict[str, str] | None = None) -> dict[str, Any]:
  cmd = [llama_bench, "-m", str(model), "-p", "0", "-n", str(GEN_TOKENS), "-d", str(depth), "-ngl", "99",
         "-r", str(BENCH_REPS), "-o", "json", *(layout_args or [])]
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env={**os.environ, **(env or {})})
  if proc.returncode != 0:
    raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
  rows = json.loads(proc.stdout[proc.stdout.find("["):])
  row = next(r for r in rows if int(r.get("n_gen", 0)) > 0)
  return {"tok_s": float(row["avg_ts"]), "stddev_tok_s": float(row.get("stddev_ts", 0.0)), "command": cmd,
          "build": row.get("build_commit"), "backends": row.get("backends"), "gpu": row.get("gpu_info")}


def _here() -> str | None:
  from boltbeam.workflow.autoscan import this_machine_target
  return this_machine_target()


def preflight(target_id:str, model:str | pathlib.Path, *, run:str | pathlib.Path = "RUN", llama_bench:str = DEFAULT,
              here:Callable[[], str | None] | None = None, env:dict[str, str] | None = None) -> dict[str, Any]:
  """What the measurement will use, or CannotMeasure with the reason and the command to run instead."""
  target = get_target(target_id)
  mine = (here or _here)()
  if mine != target_id:
    raise CannotMeasure(f"this machine is {mine or 'no registered GPU'}, not {target_id}",
                        f"on the machine with {target_id}: pipeline MODEL --run {run} --target {target_id} --measure auto")
  path = pathlib.Path(model).expanduser()
  if not path.is_file():
    raise CannotMeasure(f"the model file is gone: {path}", "measure again with the model file in place")
  bench = find(llama_bench, env)
  if bench is None:
    raise CannotMeasure(f"llama-bench was not found in ${ENV} or on PATH, so there is no whole-step decode time",
                        f"export {ENV}=/path/to/llama.cpp/build/bin/llama-bench, then measure again")
  return {"target": target, "model": path, "llama_bench": bench}


def build_timing_trace(manifest:dict[str, Any], bench:dict[int, dict[str, Any]], peak_gbs:float | None) -> dict[str, Any]:
  """The trace from the measured decodes. Pure: the tests feed it numbers."""
  rows = [{"scope": "whole_step", "context": ctx, "wall_us": 1e6 / b["tok_s"], "tok_s": b["tok_s"],
           "stddev_tok_s": b["stddev_tok_s"], "source": "llama-bench"} for ctx, b in sorted(bench.items())]
  trace = {
    "schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
    "workload": manifest["workload"], "provider_id": PROVIDER_ID, "collector_id": COLLECTOR_ID,
    "timing_source": "llama-bench whole step", "contexts": sorted(bench), "rows": rows,
    "measured": ["whole_step.tok_s (llama-bench decode at each depth)"],
    "absent": ["kernel rows: llama.cpp's per-kernel time needs the vendor profiler (collect-hw-trace)",
               "candidate rows: " + PROBE_ABSENT],
    "notes": ["Whole-step tokens per second is llama-bench decode at the requested depth; nothing else stands in for it."],
  }
  if peak_gbs:
    trace["peak_gbs"] = peak_gbs
  return trace


def measure(run:pathlib.Path, *, timing_out:pathlib.Path | None = None, llama_bench:str = DEFAULT,
            say:Callable[[str], None] = lambda _: None,
            layout:str = "one", gpus:int = 1) -> pathlib.Path:
  manifest = load_manifest(run)
  facts = preflight(manifest.get("target_id"), manifest.get("model_path") or "", run=run, llama_bench=llama_bench)
  from boltbeam.workflow import layout as lay
  layout_args, layout_env = lay.engine_args(layout, PROVIDER) if gpus > 1 else ([], {})
  request = read_json(run / "trace_request.json")
  if request.get("workload") != "decode":
    raise CannotMeasure("llama-bench-decode times decode only; this run is prefill", "plan the run with workload decode")
  bench = {}
  for ctx in request.get("contexts") or [0]:
    say(f"decode at depth {ctx}: llama-bench")
    bench[int(ctx)] = bench_decode(facts["llama_bench"], facts["model"], int(ctx), layout_args, layout_env)
  trace = build_timing_trace(manifest, bench, facts["target"].memory_bandwidth_gbs)
  trace["aux_sources"] = {"llama_bench": {str(k): v for k, v in bench.items()}}
  out = timing_out or run / "timing_trace.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(pretty_json(trace))
  return out
