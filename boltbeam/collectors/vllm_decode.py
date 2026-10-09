"""vLLM as a provider: whole-step decode and per-role time of a Hugging Face model (safetensors) in vLLM.

vLLM is not imported here. Its own python runs boltbeam/runtime/vllm_decode_drive.py, found as
$BOLTBEAM_VLLM_PYTHON, else the python next to `vllm` on PATH (the venv vLLM is installed in).

Whole step: the offline LLM API at a fixed context and batch, N tokens minus 1 token over N - 1 (the driver's
docstring says how). This is chosen over `vllm bench latency`: that reports whole requests, prefill included, so
its per-token time depends on the prompt; the difference of two calls with the same prefill is the decode alone.

Per role: the driver under nsys (--cuda-graph-trace=node, so the kernels inside vLLM's CUDA graphs are seen),
twice, N and 1 tokens, subtracted and attributed by bytes and count (engine_common.captured_role_time). The
capture runs the engine in the driver's own process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so nsys traces one
process; the timed run keeps vLLM's default.
"""
from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.collectors import engine_common as ec

PROVIDER = "vllm"
PROVIDER_ID = "vllm/offline-llm-decode"
TRACE = "vllm_timing_trace.json"
FORMATS = ("safetensors",)  # vLLM reads a Hugging Face folder; BoltBeam does not run vLLM on a GGUF
BACKENDS = ("CUDA",)
ENV = "BOLTBEAM_VLLM_PYTHON"
DRIVER = ec.RUNTIME / "vllm_decode_drive.py"
TOKENS, REPS = 64, 3
ROLE_TOKENS = 32
KV_ELEMENT = (2, "vLLM's default KV cache dtype (auto: the model's bf16)")
# FlashInfer's sampler is a run-time kernel build that needs curand.h; a system CUDA without it (the 5090 box's
# /usr/local/cuda 13.2) fails the engine at start, and FlashInfer's own headers refuse pip's newer CUDA. vLLM's own
# sampler is used instead: sampling is not weight reading, so the decode step's memory work is unchanged.
RUN_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
CAPTURE_ENV = {**RUN_ENV, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}


def python(env:dict[str, str] | None = None) -> str | None:
  return ec.python_with(ENV, "vllm", env)


def available(target, env:dict[str, str] | None = None) -> str | None:
  """Why vLLM cannot measure this target here, or None when it can."""
  if target.backend not in BACKENDS:
    return f"BoltBeam runs vLLM on NVIDIA GPUs only; {target.target_id} is {target.backend}"
  py = python(env)
  if py is None:
    return f"vLLM was not found: set ${ENV} to the python of the venv vLLM is installed in"
  if not ec.has_module(py, "vllm"):
    return f"{py} cannot import vllm"
  return None


def argv(model:str, contexts:list[int], batches:list[int], *, tokens:int = TOKENS, reps:int = REPS,
         once:int | None = None) -> list[str]:
  py = python()
  if py is None:
    raise RuntimeError(f"vLLM was not found: set ${ENV}")
  out = [py, str(DRIVER), "--model", model, "--contexts", ",".join(map(str, contexts)),
         "--batches", ",".join(map(str, batches)), "--tokens", str(tokens), "--reps", str(reps)]
  return out + (["--once", str(once)] if once else [])


def whole_step(model:str, contexts:list[int], batches:list[int], *, log:pathlib.Path | None = None,
               tokens:int = TOKENS) -> list[dict[str, Any]]:
  rows = ec.run_driver(argv(model, contexts, batches, tokens=tokens), log=log, env={**ec.venv_env(python()), **RUN_ENV})
  return [ec.point(r["context"], r["batch"], r["step_ms"], tokens=r["tokens"], source="vLLM offline LLM API, "
                   "N tokens minus 1 token, CUDA graphs on, vLLM's own sampler") for r in rows]


def role_time(run:pathlib.Path, *, target, model:str, model_id:str, context:int = 128, batch:int = 1,
              tokens:int = ROLE_TOKENS) -> dict[str, Any]:
  return ec.captured_role_time(run, provider=PROVIDER, trace=TRACE, target=target, model_id=model_id,
                               context=context, tokens=tokens, batch=batch, env={**ec.venv_env(python()), **CAPTURE_ENV},
                               argv_for=lambda n: argv(model, [context], [batch], tokens=n, reps=1, once=n))
