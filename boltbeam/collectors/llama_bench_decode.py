"""Whole-step decode timing with llama-bench, on any GPU llama.cpp runs on (collector `llama-bench-decode`).

    pipeline MODEL --run RUN --target nvidia_sm120 --measure auto

It answers trace_request.json with a boltbeam.timing_trace.v1 holding one whole_step row per requested depth:
llama-bench decode (`-p 0 -n N -d CTX`), the measured tokens per second of a real decode on this GPU. The Metal
collector (metal_native.py) takes the same rows from here.

It writes no probe evidence and no kernel rows. BoltBeam's building-block probe kernels are Metal (MSL) only, and
no fork path writes boltbeam.probe_evidence.v1, so on other GPUs the probe request stays open and says why.

llama-bench is found in this order: the --llama-bench path given, $BOLTBEAM_LLAMA_BENCH, then PATH.

Batch. llama-bench decodes one stream. A batch of B streams is timed with llama-batched-bench from the same
llama.cpp ($BOLTBEAM_LLAMA_BATCHED_BENCH, else next to llama-bench, else PATH): `-npp C -ntg N -npl B`, each of B
streams with its own C-token prompt, then N decode steps of B tokens. Its t_tg over N is one step.
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
BATCHED_ENV = "BOLTBEAM_LLAMA_BATCHED_BENCH"
BATCHED = "llama-batched-bench"
KV_ELEMENT = (2, "llama-bench's default KV cache type, f16")
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


def find_batched(env:dict[str, str] | None = None) -> str | None:
  """llama-batched-bench: $BOLTBEAM_LLAMA_BATCHED_BENCH, else next to the llama-bench in use, else PATH."""
  env = os.environ if env is None else env
  if env.get(BATCHED_ENV):
    path = pathlib.Path(env[BATCHED_ENV]).expanduser()
    return str(path) if path.is_file() else None
  bench = find(DEFAULT, env)
  if bench and (pathlib.Path(bench).parent / BATCHED).is_file():
    return str(pathlib.Path(bench).parent / BATCHED)
  return shutil.which(BATCHED)


def batched_argv(binary:str, model:pathlib.Path | str, context:int, tokens:int, batches:list[int],
                 extra:list[str] | None = None) -> list[str]:
  """B streams, each with its own context-token prompt, then `tokens` decode steps; the KV cache holds them all."""
  n_kv = max(batches) * (context + tokens) + 16
  return [binary, "-m", str(model), "-c", str(n_kv), "-b", "2048", "-ub", "512", "-npp", str(context),
          "-ntg", str(tokens), "-npl", ",".join(map(str, batches)), "-ngl", "99", "--output-format", "jsonl",
          *(extra or [])]


def batched_points(text:str, tokens:int) -> list[dict[str, Any]]:
  """llama-batched-bench's JSONL rows as points: step ms = t_tg / tg; tokens/s total = its speed_tg."""
  out = []
  for line in text.splitlines():
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      r = json.loads(line)
    except json.JSONDecodeError:
      continue
    if "t_tg" not in r or not r.get("tg"):
      continue
    step_ms = float(r["t_tg"]) * 1000.0 / int(r["tg"])
    b = int(r["pl"])
    out.append({"context": int(r["pp"]), "batch": b, "step_ms": step_ms, "tok_s_stream": 1000.0 / step_ms,
                "tok_s_total": b * 1000.0 / step_ms, "tokens": int(r["tg"]), "source": "llama-batched-bench",
                "speed_tg": float(r.get("speed_tg") or 0.0)})
  return out


def bench_batched(binary:str, model:pathlib.Path, context:int, batches:list[int], layout_args:list[str] | None = None,
                  env:dict[str, str] | None = None, tokens:int = GEN_TOKENS) -> list[dict[str, Any]]:
  cmd = batched_argv(binary, model, context, tokens, batches, layout_args)
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env={**os.environ, **(env or {})})
  if proc.returncode != 0:
    raise RuntimeError(f"llama-batched-bench exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
  got = batched_points(proc.stdout, tokens)
  if not got:
    raise RuntimeError(f"llama-batched-bench printed no JSONL rows: {proc.stdout.strip()[-300:]}")
  return got


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
              llama_bench:str = DEFAULT, layout:str = "one", gpus:int = 1, batch:int = 1) -> dict[str, Any]:
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
  if batch > 1:  # B streams: llama-batched-bench, the step time from its own t_tg
    batched = find_batched()
    if batched is None:
      raise CannotMeasure("llama-batched-bench was not found, so a batch cannot be timed",
                          f"export {BATCHED_ENV}=/path/to/llama-batched-bench")
    argv_for = lambda n: batched_argv(batched, model, context, n, [batch], extra)  # noqa: E731
  else:
    argv_for = lambda n: decode_argv(bench, model, context, n, extra)  # noqa: E731
  long = vendor_capture.capture(target.backend, argv_for(tokens), raw / "long", env=env)
  short = vendor_capture.capture(target.backend, argv_for(1), raw / "short", env=env)
  window, overlap = attribution.shared_window(long, short)
  method = vendor_capture.plan(target.backend)["method"]
  printed = vendor_capture.program_output(raw / "long")
  if batch > 1:
    pts = batched_points(printed, tokens)
    token_ms = pts[-1]["step_ms"] if pts else None
  else:
    tok_s = bench_tok_s(printed)
    token_ms = 1000.0 / tok_s if tok_s else None
  return attribution.provider_trace(run, window, provider=PROVIDER, method=method, model_id=model_id,
                                    target=target, context=context, tokens=tokens - 1, out=TRACE,
                                    token_ms=token_ms, long_kernels=len(long), overlap=overlap, batch=batch)


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


def batch_points(bench:dict[int, dict[str, Any]], model:pathlib.Path, batches, *, say:Callable[[str], None] = lambda _: None,
                 layout_args:list[str] | None = None, env:dict[str, str] | None = None) -> list[dict[str, Any]]:
  """Every timed point: batch 1 from the llama-bench decodes, then each batch > 1 per depth from
  llama-batched-bench. Reports progress as the second half of the stage (the first half is the decodes)."""
  from boltbeam.workflow import progress
  points = [{"context": ctx, "batch": 1, "step_ms": 1000.0 / b["tok_s"], "tok_s_stream": b["tok_s"],
             "tok_s_total": b["tok_s"], "tokens": GEN_TOKENS, "source": "llama-bench"} for ctx, b in sorted(bench.items())]
  wide = sorted({int(b) for b in batches if int(b) > 1})
  if not wide:
    return points
  batched = find_batched()
  if batched is None:
    raise CannotMeasure("llama-batched-bench was not found, so a batch cannot be timed",
                        f"export {BATCHED_ENV}=/path/to/llama-batched-bench")
  for i, ctx in enumerate(sorted(bench)):
    say(f"decode at depth {ctx}, batches {wide}: llama-batched-bench")
    points += bench_batched(batched, model, ctx, wide, layout_args, env)
    progress.report(len(bench) + i + 1, 2 * len(bench))
  return points


def measure(run:pathlib.Path, *, timing_out:pathlib.Path | None = None, llama_bench:str = DEFAULT,
            say:Callable[[str], None] = lambda _: None,
            layout:str = "one", gpus:int = 1, batches:list[int] | tuple[int, ...] = (1,)) -> pathlib.Path:
  manifest = load_manifest(run)
  facts = preflight(manifest.get("target_id"), manifest.get("model_path") or "", run=run, llama_bench=llama_bench)
  from boltbeam.workflow import layout as lay
  layout_args, layout_env = lay.engine_args(layout, PROVIDER) if gpus > 1 else ([], {})
  request = read_json(run / "trace_request.json")
  if request.get("workload") != "decode":
    raise CannotMeasure("llama-bench-decode times decode only; this run is prefill", "plan the run with workload decode")
  from boltbeam.workflow import progress
  bench = {}
  contexts = request.get("contexts") or [0]
  parts = len(contexts) * (2 if any(int(b) > 1 for b in batches) else 1)
  progress.report(0, parts)
  for ctx in contexts:
    say(f"decode at depth {ctx}: llama-bench")
    bench[int(ctx)] = bench_decode(facts["llama_bench"], facts["model"], int(ctx), layout_args, layout_env)
    progress.report(len(bench), parts)
  from boltbeam.workflow.screen import run_bandwidth
  trace = build_timing_trace(manifest, bench, run_bandwidth(run, facts["target"])[0])
  trace["aux_sources"] = {"llama_bench": {str(k): v for k, v in bench.items()}}
  trace["provider"] = PROVIDER
  trace["batches"] = batch_points(bench, facts["model"], batches, say=say, layout_args=layout_args, env=layout_env)
  out = timing_out or run / "timing_trace.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(pretty_json(trace))
  return out
