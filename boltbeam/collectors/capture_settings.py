"""Capture and untraced measurement answer the same persisted request."""
from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.workflow.common import read_json

# The engine settings the vLLM adapter can honor: the one list every vLLM run and capture checks a request against.
MTP_CONTEXT = "mtp_context.json"  # a speculative run's natural context ids, the same for every process of the run
VLLM_SETTINGS = ("chunk_size", "attention_backend", "tensor_parallel_size", "gemm_backends", "mtp", "mtp_prompt")
LLAMA_FLAGS = {"chunk_size": "-ub", "batch_size": "-b", "cache_type_k": "-ctk",
               "cache_type_v": "-ctv", "flash_attention": "-fa"}
LLAMA_FIELDS = {"n_ubatch": "chunk_size", "n_batch": "batch_size", "type_k": "cache_type_k",
                "type_v": "cache_type_v", "flash_attn": "flash_attention"}


def contexts(run:pathlib.Path) -> list[int]:
  values = read_json(run / "trace_request.json").get("contexts")
  if not isinstance(values, list) or not values or any(type(c) is not int or c < 0 for c in values):
    raise ValueError("trace_request.json needs explicit nonnegative contexts; capture settings cannot be guessed")
  return values


def selected_context(run:pathlib.Path, context:int | None = None) -> int:
  requested = contexts(run)
  chosen = max(requested) if context is None else context
  if chosen not in requested:
    raise ValueError(f"capture context {chosen} is absent from the requested contexts {requested}")
  return chosen


def requested(run:pathlib.Path) -> dict[str, Any]:
  settings = read_json(run / "trace_request.json").get("engine_settings") or {}
  if not isinstance(settings, dict):
    raise ValueError("engine_settings must be an object")
  return settings


def setting_words(settings:dict[str, Any]) -> list[str]:
  """The requested engine settings as the reader sees them, one phrase each, in every output (summary, report, TUI).
  A matrix-backend request changes which kernels run, so it says the run is not stock vLLM."""
  words = {"chunk_size": lambda v: f"chunk size {v}",
           "attention_backend": lambda v: f"attention backend {v}",
           "tensor_parallel_size": lambda v: f"split across {v} GPUs by rows (tensor parallel)",
           "gemm_backends": lambda v: (f"matrix backends {str(v).upper()}: the compiler's own GEMM kernels in place of "
                                       f"cuBLAS; not stock vLLM"),
           "mtp": lambda v: f"MTP k={v}: speculative decoding, {v} draft token{'s' if v != 1 else ''} per step",
           "mtp_prompt": lambda v: f"MTP context: natural text from prompt {v}"}
  return [words[k](v) if k in words else f"{k} {v}" for k, v in settings.items() if v is not None]


def mtp_request(settings:dict[str, Any]) -> dict[str, Any] | None:
  """The requested speculation: vLLM's speculative_config for MTP with k draft tokens, and the natural prompt the
  context is built from (collectors/vllm_mtp.PROMPTS, the prompt set's first prompt unless one is named). None when
  the run requests none. A prompt without k, or a k that is not a positive whole number, is refused."""
  k, prompt = settings.get("mtp"), settings.get("mtp_prompt")
  if k is None:
    if prompt is not None:
      raise ValueError("mtp_prompt names the natural text a speculative decode runs on; it needs mtp (draft tokens per step)")
    return None
  if type(k) is not int or k < 1:
    raise ValueError("mtp is the draft tokens per step, a whole number of 1 or more")
  from boltbeam.collectors.vllm_mtp import PROMPTS, prompt_set
  ids = prompt_set()["ids"]
  if prompt is not None and prompt not in ids:
    raise ValueError(f"mtp_prompt {prompt!r} is not in {PROMPTS.name} ({', '.join(ids)})")
  return {"config": {"method": "mtp", "num_speculative_tokens": k}, "prompt": prompt or ids[0], "prompt_set": str(PROMPTS)}


def observed(run:pathlib.Path, context:int) -> dict[str, Any]:
  path = run / "timing_trace.json"
  trace = read_json(path) if path.exists() else {}
  return next((r.get("engine") or {} for r in trace.get("rows", []) if r.get("context") == context), {})


def llama_flags(run:pathlib.Path, context:int, *, capture:bool = False) -> list[str]:
  settings = requested(run)
  if capture:
    engine = observed(run, context)
    for field, name in LLAMA_FIELDS.items():
      if engine.get(field) is not None:
        actual = engine[field]
        if name in settings and str(settings[name]).lower() != str(actual).lower():
          raise ValueError(f"untraced {name} {actual!r} differs from request {settings[name]!r}")
        settings[name] = actual
  unknown = set(settings) - LLAMA_FLAGS.keys()
  if unknown:
    raise ValueError(f"llama.cpp does not support requested settings {sorted(unknown)}")
  for key in ("chunk_size", "batch_size"):
    if key in settings and (type(settings[key]) is not int or settings[key] <= 0):
      raise ValueError(f"{key} must be a positive integer")
  return [value for key, flag in LLAMA_FLAGS.items() if key in settings for value in (flag, str(settings[key]))]


def require_supported(run:pathlib.Path, supported:tuple[str, ...] = ()) -> dict[str, Any]:
  settings = requested(run)
  if unknown := set(settings) - set(supported):
    raise ValueError(f"this adapter cannot honor requested settings {sorted(unknown)}")
  return settings


def context_limit(run:pathlib.Path, context:int, field:str) -> int:
  value = observed(run, context).get(field)
  if type(value) is not int or value <= context:
    raise ValueError(f"the untraced run did not record {field}; a capture cannot choose another allocation")
  return value


def vllm_flags(run:pathlib.Path, context:int, *, capture:bool = False) -> list[str]:
  """The driver flags that hold the requested engine settings: the prefill chunk, the attention backend and the
  tensor-parallel split (collectors/multi_gpu.py). A
  capture takes what the untraced run recorded, so the two cannot differ; a recorded value that differs from the
  request is an error, not a choice."""
  settings = require_supported(run, VLLM_SETTINGS)
  size, backend, split = settings.get("chunk_size"), settings.get("attention_backend"), settings.get("tensor_parallel_size")
  gemm = settings.get("gemm_backends")
  if capture:
    engine = observed(run, context)
    if engine.get("chunked_prefill"):
      actual = engine.get("max_num_batched_tokens")
      if size is not None and size != actual:
        raise ValueError("untraced chunk size differs from the request")
      size = actual
    if engine.get("attention_backend"):
      actual = engine["attention_backend"]
      if backend is not None and str(backend).upper() != str(actual).upper():
        raise ValueError(f"untraced attention backend {actual!r} differs from the request {backend!r}")
      backend = actual if backend is not None else None
    if "gemm_backends" in engine and engine["gemm_backends"] != gemm:
      raise ValueError(f"untraced matrix backends {engine['gemm_backends']!r} differ from the request {gemm!r}")
    if (actual := engine.get("tensor_parallel_size")) is not None and actual != (split or 1):
      raise ValueError(f"untraced tensor parallel size {actual} differs from the request {split or 1}")
    if engine and engine.get("speculative_config") != ((mtp_request(settings) or {}).get("config")):
      raise ValueError(f"untraced speculation {engine.get('speculative_config')!r} differs from the request "
                       f"{(mtp_request(settings) or {}).get('config')!r}")
  if split is not None and (type(split) is not int or split < 1):
    raise ValueError("tensor_parallel_size must be a positive integer")
  if size is not None and (type(size) is not int or size <= 0):
    raise ValueError("chunk_size must be a positive integer")
  if gemm is not None and (type(gemm) is not str or not set(gemm.upper().split(",")) <= {"ATEN", "TRITON", "CUTLASS", "CPP"}):
    raise ValueError("gemm_backends is a comma list of Inductor's matrix backends: ATEN, TRITON, CUTLASS, CPP")
  if backend is not None and (type(backend) is not str or not backend.strip()):
    raise ValueError("attention_backend must name one of vLLM's attention backends")
  flags = [*(["--max-num-batched-tokens", str(size)] if size is not None else []),
           *(["--attention-backend", backend] if backend is not None else []),
           *(["--tensor-parallel-size", str(split)] if split and split > 1 else []),
           *(["--gemm-backends", gemm.upper()] if gemm is not None else [])]
  if (spec := mtp_request(settings)) is not None:
    import json
    # one natural context per run, read by every process that decodes it (vllm_decode_drive.natural_context)
    flags += ["--speculative-config", json.dumps(spec["config"]), "--mtp-prompt", spec["prompt"], "--prompt-set",
              spec["prompt_set"], "--mtp-context", str(run / MTP_CONTEXT)]
  if capture and observed(run, context).get("max_model_len"):
    flags += ["--model-context-limit", str(context_limit(run, context, "max_model_len"))]
  return flags


def decode_tokens(run:pathlib.Path, context:int) -> int:
  trace = read_json(run / "timing_trace.json")
  row = next((r for r in trace.get("rows", []) if r.get("scope") == "whole_step" and r.get("context") == context), {})
  tokens = row.get("decode_tokens")
  window = (row.get("engine") or {}).get("timing_window") or {}
  if window.get("method") == "long_minus_short":
    from boltbeam.workflow.tie_out import window_context
    window_context(row)
    return window["long_tokens"]
  if type(tokens) is not int or tokens < 2:
    raise ValueError("untraced run must record at least two decode tokens before taking a matching capture")
  return tokens
