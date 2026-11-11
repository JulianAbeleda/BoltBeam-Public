"""llama.cpp's decode through llama-server, for the settings llama-bench cannot run: the MoE cache (--moe-cache-mib,
--moe-cache-cpu-share), --cpu-moe, a draft or MTP head (--spec-type draft-mtp -md HEAD). Which settings those are is
the build's fork row (data/engine_families.json flags, capture_settings.missing_flags): a request whose flags this
build's llama-bench lacks and its llama-server has is run here, by the same provider (llama.cpp) and into the same
trace files, so the tie-out and the reports read it as any llama.cpp run.

The server is $BOLTBEAM_LLAMA_SERVER, else the llama-server beside the run's llama-bench (one build), else llama-server
on PATH. Any python3 runs boltbeam/runtime/llama_server_decode_drive.py, which starts it on its own port with the run's
flags and stops that one process when it ends.

Whole step: the server's own timings, predicted_n over predicted_ms, at each requested context, one stream.
Per role: the driver under the vendor capture (nsys follows the server it starts), N tokens and 1, subtracted and
attributed by bytes and count (engine_common.captured_role_time); the capture's memory copies are its copies.json
(vendor_capture.read_nsys_copies). A draft head's kernels are no weight role of the model: they stay unattributed.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
from typing import Any, Callable

from boltbeam.collectors import engine_common as ec

PROVIDER = "llama.cpp"
PROVIDER_ID = "llama.cpp/llama-server-decode"
COLLECTOR_ID = "llama-server-decode"
TOOL = "llama-server"
ENV = "BOLTBEAM_LLAMA_SERVER"
DRIVER = ec.RUNTIME / "llama_server_decode_drive.py"
TOKENS, REPS = 64, 3
ROLE_TOKENS = 32
PORT = 0  # the driver takes a free port: never a fixed one a user's server or a stale one may hold
FLUSH_MS = 100  # the server is stopped by signal: nsys flushes on this timer so its kernels are kept
HEADROOM = 64  # context cells past the prompt and the generated tokens


def find(bench:str | None = None, env:dict[str, str] | None = None) -> str | None:
  env = os.environ if env is None else env
  if env.get(ENV):
    return env[ENV] if pathlib.Path(env[ENV]).expanduser().is_file() else None
  beside = pathlib.Path(bench).expanduser().resolve().parent / TOOL if bench else None
  return str(beside) if beside and beside.is_file() else shutil.which(TOOL, path=env.get("PATH"))


def needs_server(run:pathlib.Path, bench:str) -> bool:
  """True when the run asks for a setting whose flag this build's llama-bench lacks and its llama-server has."""
  from boltbeam.collectors import engine_families
  from boltbeam.collectors.capture_settings import LLAMA_FLAGS, requested
  if not (pathlib.Path(run) / "trace_request.json").is_file():
    return False  # no request: nothing asked of either program
  row, have = engine_families.tool_flags(bench, "llama-bench")
  server = ((row or {}).get("flags") or {}).get(TOOL) or ()
  return have is not None and any(LLAMA_FLAGS[k] not in have and LLAMA_FLAGS[k] in server for k in requested(run) if k in LLAMA_FLAGS)


def argv(server:str, model:str, contexts:list[int], flags:list[str], *, tokens:int = TOKENS, reps:int = REPS,
         once:int | None = None, log:str = "llama-server.log") -> list[str]:
  return [sys.executable, str(DRIVER), "--server", server, "--model", str(pathlib.Path(model).absolute()), "--port", str(PORT),
          "--ctx-size", str(max(contexts) + max(tokens, once or 0) + HEADROOM), "--contexts", ",".join(map(str, contexts)),
          "--tokens", str(tokens), "--reps", str(reps), "--flags", json.dumps(flags), "--log", log, *(["--once", str(once)] if once else [])]


def _flags(run:pathlib.Path, context:int, server:str, *, capture:bool = False) -> list[str]:
  from boltbeam.collectors.capture_settings import all_layers, llama_flags
  flags = llama_flags(run, context, capture=capture, binary=server, tool=TOOL)
  return [*all_layers(flags), *flags]  # every layer on the GPU unless the run places them, as llama-bench is told


def measure(run:pathlib.Path, *, bench:str, model:pathlib.Path, target, timing_out:pathlib.Path | None = None,
            say:Callable[[str], None] = lambda _: None) -> pathlib.Path:
  """Step 4 through llama-server: timing_trace.json with one whole-step row per requested context."""
  from boltbeam.artifacts.run_io import load_manifest
  from boltbeam.collectors import engine_args
  from boltbeam.collectors.capture_settings import contexts as requested_contexts
  from boltbeam.collectors.llama_bench_decode import CannotMeasure, fork_fact
  from boltbeam.core.canonical import pretty_json
  from boltbeam.roofline.limit import run_bandwidth
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  server = find(bench)
  if server is None:
    raise CannotMeasure(f"the requested settings need llama-server, which was not found beside {bench}, in ${ENV} or on PATH",
                        f"export {ENV}=/path/to/llama-server")
  contexts = requested_contexts(run)
  if 0 in contexts:
    raise CannotMeasure("llama-server decodes after a prompt, and context 0 has none", "ask contexts of 1 token or more")
  flags = _flags(run, max(contexts), server)
  say(f"decode at depths {', '.join(map(str, contexts))}: llama-server")
  log = run / "kernel_compare" / "llama_server_decode.log"
  rows = ec.run_driver(argv(server, str(model), contexts, flags, log=str(log.with_suffix(".serve.log"))), log=log, cwd=run)
  serve_log = log.with_suffix(".serve.log")
  engine_args.record(run, [server, "-m", str(model), *flags][1:], engine=PROVIDER,
                     log=serve_log.read_text(errors="replace") if serve_log.is_file() else None)
  fork = fork_fact(server)  # the server's own build: beside llama-bench, or the $BOLTBEAM_LLAMA_SERVER one
  manifest = load_manifest(run)
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
           "workload": manifest["workload"], "provider": PROVIDER, "provider_id": PROVIDER_ID, "collector_id": COLLECTOR_ID,
           "timing_source": "llama-server timings (predicted_n / predicted_ms)", "contexts": contexts,
           "rows": [{"scope": "whole_step", "context": r["context"], "wall_us": 1e3 * r["step_ms"], "tok_s": r["tok_s_stream"],
                     "source": "llama-server", "decode_tokens": r["tokens"], "prompt_tokens": r.get("prompt_tokens"),
                     "engine": {**(r.get("engine") or {}), **fork}} for r in rows],
           "measured": ["whole_step.tok_s (llama-server's predicted_n over predicted_ms, median of the repetitions)"],
           "absent": ["kernel rows: per role is the capture of the same server (role_time)"]}
  if (peak := run_bandwidth(run, target)[0]):
    trace["peak_gbs"] = peak
  out = timing_out or run / "timing_trace.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(pretty_json(trace))
  return out


def role_time(run:pathlib.Path, *, target, model:str, model_id:str, context:int, bench:str,
              tokens:int = ROLE_TOKENS) -> dict[str, Any]:
  """The server's kernels per role at one context, one stream (llama_bench_decode.TRACE, as llama-bench's)."""
  from boltbeam.collectors.llama_bench_decode import TRACE, CannotMeasure
  server = find(bench)
  if server is None:
    raise CannotMeasure(f"the requested settings need llama-server, which was not found beside {bench}, in ${ENV} or on PATH",
                        f"export {ENV}=/path/to/llama-server")
  flags = _flags(run, context, server, capture=True)
  raw = run / "kernel_compare" / "llama_server_capture"
  raw.mkdir(parents=True, exist_ok=True)
  return ec.captured_role_time(run, provider=PROVIDER, trace=TRACE, target=target, model_id=model_id, context=context,
                               tokens=tokens, flush_ms=FLUSH_MS,
                               argv_for=lambda n: argv(server, model, [context], flags, tokens=n, reps=1, once=n,
                                                       log=str(raw / f"llama-server-{n}.log")))
