"""The work a prefill does, per role and for attention, and the ceiling of each kernel that does it.

A prefill of L prompt tokens runs every weight matrix over a chunk of tokens at a time (llama.cpp's ubatch, the
fork's prefill chunk): one launch of an N x K weight over M tokens is a GEMM of 2 x M x N x K operations that reads
the weight once and M x (K + N) activations. Attention is causal: the token at position p attends p + 1 keys, so a
prompt is L x (L + 1) / 2 query-key pairs per head, each 2 x head_dim operations for QK and 2 x head_dim for PV.

Only work the engine did is counted: the launches the capture saw, at the tokens each ran over. A role the engine
prunes (the last layer's FFN runs for the last token only) is fewer launches, not a smaller per-call figure, and the
output head over one token is a GEMV of 2 x N x K.

A kernel's ceiling is max(operations / the measured peak of the path its instructions use, bytes / the measured
bandwidth): the roofline taken per kernel, with the path's own peak (collectors/cuda_compute.py) instead of one
number for the chip. MFU is operations over (time x that peak).
"""
from __future__ import annotations

from typing import Any

ACT_BYTES = 2  # an activation element in and out of a GEMM, fp16: the bytes side of a GEMM ceiling at prefill sizes


def gemm(m:float, n:int, k:int, weight_bytes:float) -> dict[str, float]:
  """One launch: operations and bytes (the weight once, activations in and out at ACT_BYTES)."""
  return {"flops": 2.0 * m * n * k, "bytes": float(weight_bytes) + ACT_BYTES * m * (n + k)}


def attention(profile:dict[str, Any], length:int, chunks:list[int] | None = None,
              kv_element_bytes:int = 2) -> dict[str, Any] | None:
  """The prompt's attention work over all layers: operations for the causal pairs, and bytes as each chunk reads
  the K and V written so far plus its own Q and output. None when the profile does not say the attention shape."""
  att = (profile.get("metadata") or {}).get("attention") or {}
  layers, heads, kv_heads, dim = profile.get("layer_count"), att.get("head_count"), att.get("head_count_kv"), att.get("head_dim")
  attention_layers = att.get("layer_count") or layers  # a hybrid model attends in some layers only
  if not (attention_layers and heads and kv_heads and dim):
    return None
  pairs = length * (length + 1) / 2.0
  flops = 4.0 * dim * heads * pairs * attention_layers
  done, read = 0, 0.0
  for w in chunks or [length]:
    done += w
    read += done * 2 * kv_heads * dim * kv_element_bytes + 2 * w * heads * dim * ACT_BYTES
  return {"flops": flops, "bytes": read * attention_layers, "pairs_per_head": pairs, "layers": attention_layers,
          "words": f"causal: {length} x {length + 1} / 2 query-key pairs per head, 4 x {dim} operations each, "
                   f"{heads} heads, {attention_layers} layers"}


def ceiling(flops:float, nbytes:float, peak_tflops:float | None, bandwidth_gbs:float | None) -> dict[str, Any]:
  """ms at the ceiling and which side binds; compute_ms None when no peak is known for the kernel's path."""
  compute = flops / (peak_tflops * 1e12) * 1e3 if peak_tflops and flops else None
  memory = nbytes / (bandwidth_gbs * 1e9) * 1e3 if bandwidth_gbs and nbytes else None
  if compute is None and memory is None:
    return {"ms": None, "compute_ms": None, "memory_ms": None, "bound": None}
  if compute is None:
    return {"ms": memory, "compute_ms": None, "memory_ms": memory, "bound": "memory"}
  bound = "compute" if memory is None or compute >= memory else "memory"
  return {"ms": max(compute, memory or 0.0), "compute_ms": compute, "memory_ms": memory, "bound": bound}


def mfu(flops:float, ms:float, peak_tflops:float | None) -> float | None:
  """Operations done over what the path's peak could do in the same time, %."""
  if not flops or not ms or not peak_tflops:
    return None
  return 100.0 * flops / (ms * 1e-3 * peak_tflops * 1e12)
