"""vLLM as a provider: whole-step decode and per-role time of a Hugging Face model (safetensors) in vLLM.

vLLM is not imported here. Its own python runs boltbeam/runtime/vllm_decode_drive.py, found as
$BOLTBEAM_VLLM_PYTHON, else the python next to `vllm` on PATH (the venv vLLM is installed in).

Whole step: the offline LLM API at a fixed context and batch, N tokens minus 1 token over N - 1 (the driver's
docstring says how). This is chosen over `vllm bench latency`: that reports whole requests, prefill included, so
its per-token time depends on the prompt; the difference of two calls with the same prefill is the decode alone.

Per role: production CUDA-graph timing is joined to compiled graphless labels by exact kernel sequence.
Only single-stream captures are supported; other batches refuse rather than subtract independent programs.
The join reads Nsight Systems' trace on NVIDIA, and rocprofv3's database written as the same tables on AMD
(collectors/rocpd_trace.py), with each kernel's matrix path read from the code object it ran (collectors/rocm_isa.py).
The capture runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0); whole-step timing keeps vLLM's default.
"""
from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.collectors import engine_common as ec

PROVIDER = "vllm"
PROVIDER_ID = "vllm/offline-llm-decode"
TRACE = "vllm_timing_trace.json"
FORMATS = ("safetensors", "awq", "gptq")  # vLLM reads a Hugging Face folder; BoltBeam does not run vLLM on a GGUF
BACKENDS = ("CUDA", "AMD")  # vLLM's CUDA and ROCm builds; the target's backend says which this machine has
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
    return f"BoltBeam runs vLLM on {' or '.join(BACKENDS)} GPUs; {target.target_id} is {target.backend}"
  py = python(env)
  if py is None:
    return f"vLLM was not found: set ${ENV} to the python of the venv vLLM is installed in"
  if not ec.has_module(py, "vllm"):
    return f"{py} cannot import vllm"
  return None


def argv(model:str, contexts:list[int], batches:list[int], *, tokens:int = TOKENS, reps:int = REPS,
         once:int | None = None, context_limit:int | None = None) -> list[str]:
  py = python()
  if py is None:
    raise RuntimeError(f"vLLM was not found: set ${ENV}")
  out = [py, str(DRIVER), "--model", model, "--contexts", ",".join(map(str, contexts)),
         "--batches", ",".join(map(str, batches)), "--tokens", str(tokens), "--reps", str(reps)]
  if context_limit is not None:
    out += ["--model-context-limit", str(context_limit)]
  return out + (["--once", str(once)] if once else [])


def decode_window_vllm(context:int, tokens:int, batch:int, engine:dict[str, Any]) -> dict[str, Any]:
  """The window both arms of a vLLM decode timing record: host seconds, N-token call minus 1-token call (compare
  llama_bench_decode.decode_window), from the engine fields either arm records. Contexts count the generated
  tokens' positions as llama_bench_decode does (tokens 2..N); one sample is one long and one short call, and the number of samples is recorded beside it."""
  return {"method": "long_minus_short", "long_tokens": tokens, "short_tokens": 1, "context_start": context + 2, "context_end": context + tokens,
          "call_order": engine["call_order"], "warmup": engine["warm"], "batch": batch, "graph_mode": "eager" if engine["enforce_eager"] else "graphs",
          "vllm_version": engine.get("build") or engine["vllm"], "seed": engine["seed"]}


def whole_step(model:str, contexts:list[int], batches:list[int], *, log:pathlib.Path | None = None,
               tokens:int = TOKENS, settings:list[str] | None = None, graphs:str) -> list[dict[str, Any]]:
  """vLLM's decode at each context and batch; `graphs` names the backend's graph replay vLLM runs the step with
  (target/vendors.py terms)."""
  rows = ec.run_driver([*argv(model, contexts, batches, tokens=tokens), *(settings or [])], log=log, env={**ec.venv_env(python()), **RUN_ENV})
  return [ec.point(r["context"], r["batch"], r["step_ms"], tokens=r["tokens"], engine={**r["engine"], "timing_window": decode_window_vllm(r["context"], r["tokens"], r["batch"], r["engine"])},
                   step_samples_ms=r.get('step_samples_ms'),repeat_band_ms=r.get('repeat_band_ms'),source="vLLM offline LLM API, "
                   f"N tokens minus 1 token, {graphs} on, vLLM's own sampler") for r in rows]


def role_time(run:pathlib.Path, *, target, model:str, model_id:str, context:int, batch:int = 1,
              tokens:int = ROLE_TOKENS) -> dict[str, Any]:
  from boltbeam.collectors.vllm_semantic_decode import collect
  return collect(run,target=target,model=model,context=context,tokens=tokens,batch=batch)
