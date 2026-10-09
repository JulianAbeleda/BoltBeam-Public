"""Ollama as a provider: whole-step decode and per-role time of a GGUF in a private `ollama serve`.

The Ollama program is $BOLTBEAM_OLLAMA, else `ollama` on PATH. Its models folder is $OLLAMA_MODELS, else
Ollama's own default (~/.ollama/models); importing a GGUF copies it there once. Any python3 runs
boltbeam/runtime/ollama_decode_drive.py, which starts the server on its own port, imports the run's GGUF with a
Modelfile and stops the server when it ends.

Whole step: Ollama's own counters, eval_count over eval_duration, at a fixed prompt length and batch (the driver's
docstring says how). The weights are the run's GGUF, the same file llama.cpp reads, so the roofline is the same.

Per role: the driver under nsys, which follows `ollama serve` and the runner it starts, N tokens and 1,
subtracted and attributed by bytes and count (engine_common.captured_role_time).
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

from boltbeam.collectors import engine_common as ec

PROVIDER = "ollama"
PROVIDER_ID = "ollama/api-eval-counters"
TRACE = "ollama_timing_trace.json"
FORMATS = ("gguf",)
BACKENDS = ("CUDA", "Metal", "AMD")
ENV = "BOLTBEAM_OLLAMA"
DRIVER = ec.RUNTIME / "ollama_decode_drive.py"
TOKENS, REPS = 64, 3
ROLE_TOKENS = 32
PORT = 11534  # never Ollama's default 11434: a user's own server may hold that
KV_ELEMENT = (2, "Ollama's default KV cache type, f16")
FLUSH_MS = 100  # Ollama stops its runner by signal: nsys flushes on this timer so the runner's kernels are kept


def program(env:dict[str, str] | None = None) -> str | None:
  return ec.find_program(ENV, ("ollama",), env)


def models_folder() -> str:
  return os.environ.get("OLLAMA_MODELS") or str(pathlib.Path("~/.ollama/models").expanduser())


def available(target, env:dict[str, str] | None = None) -> str | None:
  if target.backend not in BACKENDS:
    return f"BoltBeam knows no Ollama build for {target.backend}"
  if program(env) is None:
    return f"ollama was not found in ${ENV} or on PATH"
  return None


def argv(model:str, contexts:list[int], batches:list[int], *, tokens:int = TOKENS, reps:int = REPS,
         once:int | None = None, log:str = "ollama-serve.log") -> list[str]:
  exe = program()
  if exe is None:
    raise RuntimeError(f"ollama was not found in ${ENV} or on PATH")
  out = [sys.executable, str(DRIVER), "--ollama", exe, "--gguf", model, "--models", models_folder(),
         "--port", str(PORT), "--contexts", ",".join(map(str, contexts)), "--batches", ",".join(map(str, batches)),
         "--tokens", str(tokens), "--reps", str(reps), "--log", log]
  return out + (["--once", str(once)] if once else [])


def whole_step(model:str, contexts:list[int], batches:list[int], *, log:pathlib.Path | None = None,
               tokens:int = TOKENS) -> list[dict[str, Any]]:
  if log:
    log.parent.mkdir(parents=True, exist_ok=True)
  serve_log = str(log.with_suffix(".serve.log")) if log else "ollama-serve.log"
  rows = ec.run_driver(argv(model, contexts, batches, tokens=tokens, log=serve_log), log=log)
  return [ec.point(r["context"], r["batch"], r["step_ms"], tokens=r["tokens"], prompt_tokens=r.get("prompt_tokens"),
                   tok_s_total_measured=r["tok_s_total"], source="Ollama API eval_count / eval_duration")
          for r in rows]


def role_time(run:pathlib.Path, *, target, model:str, model_id:str, context:int = 128, batch:int = 1,
              tokens:int = ROLE_TOKENS) -> dict[str, Any]:
  raw = run / "kernel_compare"
  raw.mkdir(parents=True, exist_ok=True)
  return ec.captured_role_time(run, provider=PROVIDER, trace=TRACE, target=target, model_id=model_id,
                               context=context, tokens=tokens, batch=batch, flush_ms=FLUSH_MS,
                               argv_for=lambda n: argv(model, [context], [batch], tokens=n, reps=1, once=n,
                                                       log=str(raw / f"ollama-serve-{n}.log")))
