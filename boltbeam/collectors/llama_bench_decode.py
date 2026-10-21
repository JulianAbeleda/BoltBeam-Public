"""Whole-step decode timing with llama-bench, on any GPU llama.cpp runs on (collector `llama-bench-decode`).

    pipeline MODEL --run RUN --target nvidia_sm120 --measure auto

It answers trace_request.json with a boltbeam.timing_trace.v1 holding one whole_step row per requested depth:
fresh llama-bench long/short decode pairs (`-p 0 -n N -d CTX`), estimating tokens 2..N from elapsed differences. The Metal
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
import math
import os
import pathlib
import shutil
import subprocess
import statistics
import tempfile
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
PROBE_ABSENT = ("no building-block probe in this run: this provider's path does not run BoltBeam's probe kernels, "
                "and no fork path writes probe evidence")


class CannotMeasure(RuntimeError):
  """Measuring is not possible here. `reason` says why; `command` is what to run instead."""

  def __init__(self, reason:str, command:str) -> None:
    super().__init__(reason)
    self.reason, self.command = reason, command


def find(llama_bench:str = DEFAULT, env:dict[str, str] | None = None) -> str | None:
  """The llama-bench to run: an explicit path, then $BOLTBEAM_LLAMA_BENCH, then PATH, then what the engine scan
  saved (collectors/engine_scan.py). None when there is none."""
  from boltbeam.collectors import engine_scan
  env = os.environ if env is None else env
  if llama_bench != DEFAULT:
    return llama_bench if pathlib.Path(llama_bench).expanduser().is_file() else shutil.which(llama_bench)
  if env.get(ENV):
    path = pathlib.Path(env[ENV]).expanduser()
    return str(path) if path.is_file() else None
  return shutil.which(DEFAULT) or engine_scan.saved(ENV)


def find_batched(env:dict[str, str] | None = None) -> str | None:
  """llama-batched-bench: $BOLTBEAM_LLAMA_BATCHED_BENCH, else next to the llama-bench in use, else PATH."""
  env = os.environ if env is None else env
  if env.get(BATCHED_ENV):
    path = pathlib.Path(env[BATCHED_ENV]).expanduser()
    return str(path) if path.is_file() else None
  bench = find(DEFAULT, env)
  if bench and (pathlib.Path(bench).parent / BATCHED).is_file():
    return str(pathlib.Path(bench).parent / BATCHED)
  from boltbeam.collectors import engine_scan
  return shutil.which(BATCHED) or engine_scan.saved(BATCHED_ENV)


def batched_argv(binary:str, model:pathlib.Path | str, context:int, tokens:int, batches:list[int],
                 extra:list[str] | None = None) -> list[str]:
  """B streams, each with its own context-token prompt, then `tokens` decode steps; the KV cache holds them all."""
  n_kv = max(batches) * (context + tokens) + 16
  return [binary, "-m", str(pathlib.Path(model).absolute()), "-c", str(n_kv), "-npp", str(context),
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
                  env:dict[str, str] | None = None, tokens:int = GEN_TOKENS, cwd:pathlib.Path | None = None) -> list[dict[str, Any]]:
  cmd = batched_argv(binary, model, context, tokens, batches, layout_args)
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env={**os.environ, **(env or {})}, cwd=cwd)
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
  this command holds exactly that work. The model path is made absolute: a capture runs the program in its own
  capture folder (vendor_capture.capture)."""
  return [llama_bench, "-m", str(pathlib.Path(model).absolute()), "-p", "0", "-n", str(tokens), "-d", str(depth), "-ngl", "99", "-r", "1",
          "--no-warmup", "-o", "json", *(extra or [])]


def role_time(run:pathlib.Path, *, target, model:str, model_id:str, context:int, tokens:int = ROLE_TOKENS,
              llama_bench:str = DEFAULT, layout:str = "one", gpus:int = 1, batch:int = 1,
              flags:list[str] | None = None, raw:pathlib.Path | None = None, out:str = TRACE) -> dict[str, Any]:
  """llama.cpp's kernels per role, captured from outside by the GPU vendor's tool (collectors/vendor_capture.py)
  and attributed by bytes and count (collectors/attribution.py). Two captures at the same depth, one with
  `tokens` decode tokens and one with 1, are subtracted per kernel: what is left is exactly tokens - 1 decode
  tokens, with the context fill, the setup and the first token removed. flags: llama-bench flags a caller matches to
  another configuration (a replay's cache types); raw and out: the capture folder and trace file, when a run holds
  more than one decode capture."""
  from boltbeam.collectors import attribution, vendor_capture
  from boltbeam.workflow import layout as lay
  if batch == 1:
    decode_window(context, tokens)
  if layout == "row":
    raise attribution.CaptureRefused("split by rows runs every weight kernel on every GPU at once; per-role "
                                     f"attribution across devices is not built, so this run has the whole step only ({lay.LIMITED})")
  bench = find(llama_bench)
  if bench is None:
    raise CannotMeasure(available() or "llama-bench is missing", f"export {ENV}=/path/to/llama-bench")
  extra, env = lay.engine_args(layout, PROVIDER) if gpus > 1 else ([], {})
  from boltbeam.collectors.capture_settings import llama_flags
  extra = [*extra, *llama_flags(run, context, capture=True), *(flags or [])]
  raw = raw or run / RAW
  if batch > 1:  # B streams: llama-batched-bench, the step time from its own t_tg
    batched = find_batched()
    if batched is None:
      raise CannotMeasure("llama-batched-bench was not found, so a batch cannot be timed",
                          f"export {BATCHED_ENV}=/path/to/llama-batched-bench")
    argv_for = lambda n: batched_argv(batched, model, context, n, [batch], extra)  # noqa: E731
  else:
    argv_for = lambda n: decode_argv(bench, model, context, n, extra)  # noqa: E731
  from boltbeam.workflow.common import run_gpu
  long = vendor_capture.capture(target.backend, argv_for(tokens), raw / "long", env=env, device=run_gpu(run))
  short = vendor_capture.capture(target.backend, argv_for(1), raw / "short", env=env, device=run_gpu(run))
  window, overlap = attribution.shared_window(long, short)
  method = vendor_capture.plan(target.backend)["method"]
  printed = vendor_capture.program_output(raw / "long")
  engine = None
  if batch > 1:
    pts = batched_points(printed, tokens)
    token_ms = pts[-1]["step_ms"] if pts else None
  else:
    pair = decode_pair(bench_row(printed), bench_row(vendor_capture.program_output(raw / "short")), context, tokens)
    token_ms = pair["step_ms"]
    engine = {**pair["engine"], "timing_window": decode_window(context, tokens)}
  return attribution.provider_trace(run, window, provider=PROVIDER, method=method, model_id=model_id,
                                    target=target, context=context, tokens=tokens - 1, out=out,
                                    token_ms=token_ms, long_kernels=len(long), overlap=overlap, batch=batch, engine=engine)


def bench_row(text:str) -> dict[str, Any]:
  """The engine's decode result, allowing diagnostic text before or after the JSON."""
  start = text.find("[")
  while start >= 0:
    try:
      rows = json.JSONDecoder().raw_decode(text[start:])[0]
      row = next((r for r in rows if isinstance(r, dict) and int(r.get("n_gen", 0)) > 0), None)
      if row:
        return row
    except (ValueError, TypeError):
      pass
    start = text.find("[", start + 1)
  raise ValueError("llama-bench printed no decode JSON row")


def decode_window(depth:int, tokens:int) -> dict[str, Any]:
  """A differential wall estimate and the exact token range used for kernel attribution."""
  if type(tokens) is not int or tokens < 2 or type(depth) is not int or depth < 0:
    raise ValueError("decode difference needs at least two tokens at a nonnegative depth")
  return {"method": "long_minus_short", "long_tokens": tokens, "short_tokens": 1,
          "context_start": depth + 2, "context_end": depth + tokens,
          "repetitions": 1, "warmup": False, "process": "fresh_per_sample"}


def decode_pair(long:dict, short:dict, depth:int, tokens:int) -> dict[str, Any]:
  """Subtract synchronized engine elapsed samples, never rates or kernel busy time.
  The first token is in a separate process: this estimates tokens 2..N and retains its subtraction noise."""
  decode_window(depth, tokens)
  fields = ("build_commit", "build_number", "model_filename", "n_batch", "n_ubatch", "type_k", "type_v", "flash_attn")
  if any(long.get(k) != short.get(k) for k in fields):
    raise ValueError("decode long/short engine settings differ")
  times = []
  for row, count in ((long, tokens), (short, 1)):
    if any(type(row.get(k)) is not int for k in ("n_gen", "n_depth", "n_prompt")) \
        or row.get("n_gen") != count or row.get("n_depth") != depth or row.get("n_prompt") != 0:
      raise ValueError("decode long/short token window differs from the requested work")
    samples = row.get("samples_ns")
    if not isinstance(samples, list) or len(samples) != 1 or type(samples[0]) not in (int, float) \
        or not math.isfinite(samples[0]) or samples[0] <= 0:
      raise ValueError("decode difference needs one positive elapsed sample per fresh process")
    times.append(samples[0])
  if times[0] <= times[1]:
    raise ValueError("decode long-minus-short elapsed time is not positive")
  return {"long_ns": times[0], "short_ns": times[1], "step_ms": (times[0] - times[1]) / (tokens - 1) / 1e6,
          "engine": {k: long[k] for k in ("n_ubatch", "n_batch", "type_k", "type_v", "flash_attn") if k in long}}


def bench_tok_s(text:str) -> float | None:
  """The decode tokens per second llama-bench printed in its JSON (`avg_ts` of the n_gen row), or None."""
  try:
    return float(bench_row(text)["avg_ts"])
  except (ValueError, TypeError, KeyError):
    return None


def bench_decode(llama_bench:str, model:pathlib.Path, depth:int, layout_args:list[str] | None = None,
                 env:dict[str, str] | None = None, cwd:pathlib.Path | None = None) -> dict[str, Any]:
  """Independent fresh long/short pairs with the same lifecycle as the vendor capture.
  Each pair estimates tokens 2..N; raw commands, stdout and stderr survive even a refused pair in `cwd`."""
  raw = pathlib.Path(tempfile.mkdtemp(prefix=f"decode-{depth}-", dir=cwd))
  pairs = []
  for repeat in range(BENCH_REPS):
    rows = []
    for label, count in (("long", GEN_TOKENS), ("short", 1)):
      cmd = decode_argv(llama_bench, model, depth, count, layout_args)
      stem = raw / f"{repeat + 1}-{label}"
      stem.with_suffix(".command.json").write_text(pretty_json(cmd))
      proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env={**os.environ, **(env or {})}, cwd=cwd)
      stem.with_suffix(".stdout").write_text(proc.stdout)
      stem.with_suffix(".stderr").write_text(proc.stderr)
      if proc.returncode != 0:
        raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
      rows.append(bench_row(proc.stdout))
    pairs.append(decode_pair(*rows, depth, GEN_TOKENS))
  row = rows[0]
  times = [p["step_ms"] for p in pairs]
  return {"tok_s": 1000.0 / statistics.mean(times), "stddev_tok_s": statistics.stdev([1000.0 / t for t in times]) if len(times) > 1 else 0.0,
          "command": decode_argv(llama_bench, model, depth, GEN_TOKENS, layout_args), "raw": str(raw), "samples": pairs,
          "build": row.get("build_commit"), "build_number": row.get("build_number"), "build_installed": _installed(llama_bench), "backends": row.get("backends"), "gpu": row.get("gpu_info"),
          "decode_tokens": GEN_TOKENS - 1, "measurement_scope": "differential_decode_elapsed",
          "engine": {**pairs[0]["engine"], "timing_window": decode_window(depth, GEN_TOKENS)}}


def _installed(llama_bench:str) -> str | None:
  """The day the llama-bench binary was put on this machine (its file's date): when this build arrived here."""
  import datetime
  try:
    return datetime.date.fromtimestamp(pathlib.Path(llama_bench).resolve().stat().st_mtime).isoformat()
  except OSError:
    return None


def _here() -> str | None:
  from boltbeam.workflow.autoscan import this_machine_target
  return this_machine_target()


def model_file(model:str | pathlib.Path) -> pathlib.Path:
  """The GGUF file llama-bench and the building-block probe read, or CannotMeasure saying why not: a model folder
  (safetensors, for another engine) is not one, and is said so, not called gone."""
  path = pathlib.Path(model).expanduser()
  if path.is_dir():
    raise CannotMeasure(f"{path} is a model folder; this step reads one GGUF file", "measure with the model's GGUF file")
  if not path.is_file():
    raise CannotMeasure(f"the model file is gone: {path}", "measure again with the model file in place")
  return path


def preflight(target_id:str, model:str | pathlib.Path, *, run:str | pathlib.Path = "RUN", llama_bench:str = DEFAULT,
              here:Callable[[], str | None] | None = None, env:dict[str, str] | None = None) -> dict[str, Any]:
  """What the measurement will use, or CannotMeasure with the reason and the command to run instead."""
  target = get_target(target_id)
  mine = (here or _here)()
  if mine != target_id:
    raise CannotMeasure(f"this machine is {mine or 'no registered GPU'}, not {target_id}",
                        f"on the machine with {target_id}: pipeline MODEL --run {run} --target {target_id} --measure auto")
  path = model_file(model)
  bench = find(llama_bench, env)
  if bench is None:
    raise CannotMeasure(f"llama-bench was not found in ${ENV} or on PATH, so there is no whole-step decode time",
                        f"export {ENV}=/path/to/llama.cpp/build/bin/llama-bench, then measure again")
  return {"target": target, "model": path, "llama_bench": bench}


# what bench_decode measured about the engine build, carried onto every whole-step row (providers.pin_check reads it)
BUILD_FIELDS = ("build", "build_number", "build_installed")


def build_facts(b:dict[str, Any]) -> dict[str, Any]:
  return {k: b[k] for k in BUILD_FIELDS if b.get(k) is not None}


def build_timing_trace(manifest:dict[str, Any], bench:dict[int, dict[str, Any]], peak_gbs:float | None) -> dict[str, Any]:
  """The trace from the measured decodes. Pure: the tests feed it numbers."""
  rows = [{"scope": "whole_step", "context": ctx, "wall_us": 1e6 / b["tok_s"], "tok_s": b["tok_s"],
           "stddev_tok_s": b["stddev_tok_s"], "source": "llama-bench", "engine": b.get("engine", {}), "decode_tokens": b.get("decode_tokens"),
           **({"measurement_scope": b["measurement_scope"]} if b.get("measurement_scope") else {}),
           **build_facts(b)}
          for ctx, b in sorted(bench.items())]
  trace = {
    "schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
    "workload": manifest["workload"], "provider_id": PROVIDER_ID, "collector_id": COLLECTOR_ID,
    "timing_source": "llama-bench whole step", "contexts": sorted(bench), "rows": rows,
    "measured": ["whole_step.tok_s (llama-bench long-minus-short elapsed estimate at each depth)"],
    "absent": ["kernel rows: llama.cpp's per-kernel time needs the vendor profiler (collect-hw-trace)",
               "candidate rows: " + PROBE_ABSENT],
    "notes": ["Fresh long/short pairs estimate tokens 2..N at each requested depth; the subtracted first token is from a separate process, so its run-to-run noise remains."],
  }
  if peak_gbs:
    trace["peak_gbs"] = peak_gbs
  return trace


def batch_points(bench:dict[int, dict[str, Any]], model:pathlib.Path, batches, *, say:Callable[[str], None] = lambda _: None,
                 layout_args:list[str] | None = None, env:dict[str, str] | None = None,
                 cwd:pathlib.Path | None = None) -> list[dict[str, Any]]:
  """Every timed point: batch 1 from the llama-bench decodes, then each batch > 1 per depth from
  llama-batched-bench. Reports progress as the second half of the stage (the first half is the decodes)."""
  from boltbeam.workflow import progress
  points = [{"context": ctx, "batch": 1, "step_ms": 1000.0 / b["tok_s"], "tok_s_stream": b["tok_s"],
             "tok_s_total": b["tok_s"], "tokens": b.get("decode_tokens", GEN_TOKENS), "source": "llama-bench"} for ctx, b in sorted(bench.items())]
  wide = sorted({int(b) for b in batches if int(b) > 1})
  if not wide:
    return points
  batched = find_batched()
  if batched is None:
    raise CannotMeasure("llama-batched-bench was not found, so a batch cannot be timed",
                        f"export {BATCHED_ENV}=/path/to/llama-batched-bench")
  for i, ctx in enumerate(sorted(bench)):
    say(f"decode at depth {ctx}, batches {wide}: llama-batched-bench")
    points += bench_batched(batched, model, ctx, wide, layout_args, env, cwd=cwd)
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
  from boltbeam.collectors.capture_settings import contexts as requested_contexts, llama_flags
  contexts = requested_contexts(run)
  parts = len(contexts) * (2 if any(int(b) > 1 for b in batches) else 1)
  progress.report(0, parts)
  for ctx in contexts:
    say(f"decode at depth {ctx}: llama-bench")
    bench[int(ctx)] = bench_decode(facts["llama_bench"], facts["model"], int(ctx), [*layout_args, *llama_flags(run, int(ctx))], layout_env, cwd=run)
    progress.report(len(bench), parts)
  from boltbeam.workflow.screen import run_bandwidth
  trace = build_timing_trace(manifest, bench, run_bandwidth(run, facts["target"])[0])
  trace["aux_sources"] = {"llama_bench": {str(k): v for k, v in bench.items()}}
  trace["provider"] = PROVIDER
  trace["batches"] = batch_points(bench, facts["model"], batches, say=say, layout_args=[*layout_args, *llama_flags(run, max(contexts))], env=layout_env, cwd=run)
  out = timing_out or run / "timing_trace.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(pretty_json(trace))
  return out
