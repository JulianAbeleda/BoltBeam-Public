"""Capture and untraced measurement answer the same persisted request."""
from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.workflow.common import read_json

# The engine settings the vLLM adapter can honor: the one list every vLLM run and capture checks a request against.
VLLM_SETTINGS = ("chunk_size", "attention_backend", "tensor_parallel_size")
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
    if (actual := engine.get("tensor_parallel_size")) is not None and actual != (split or 1):
      raise ValueError(f"untraced tensor parallel size {actual} differs from the request {split or 1}")
  if split is not None and (type(split) is not int or split < 1):
    raise ValueError("tensor_parallel_size must be a positive integer")
  if size is not None and (type(size) is not int or size <= 0):
    raise ValueError("chunk_size must be a positive integer")
  if backend is not None and (type(backend) is not str or not backend.strip()):
    raise ValueError("attention_backend must name one of vLLM's attention backends")
  flags = [*(["--max-num-batched-tokens", str(size)] if size is not None else []),
           *(["--attention-backend", backend] if backend is not None else []),
           *(["--tensor-parallel-size", str(split)] if split and split > 1 else [])]
  if capture and observed(run, context).get("max_model_len"):
    flags += ["--model-context-limit", str(context_limit(run, context, "max_model_len"))]
  return flags


def decode_tokens(run:pathlib.Path, context:int) -> int:
  trace = read_json(run / "timing_trace.json")
  row = next((r for r in trace.get("rows", []) if r.get("scope") == "whole_step" and r.get("context") == context), {})
  tokens = row.get("decode_tokens")
  if type(tokens) is not int or tokens < 2:
    raise ValueError("untraced run must record at least two decode tokens before taking a matching capture")
  return tokens
