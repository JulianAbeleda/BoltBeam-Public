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
number for the chip. A kernel whose instructions are on several paths (flash attention: fp16 with fp32 and with fp16
accumulate) has its operations split over its paths by its instruction sites, each part at its own path's peak
(`compute_ms`). MFU is operations over (time x the peak those parts make together).

Bytes come from the formats the engine reads and writes (the engine adapter names them, data/quants.json sizes
them): a GEMM reads its weight once, M x K activations in the engine's activation-in format and writes M x N in its
activation-out format. A stream-k fixup (`stream_k_fixup`) moves the partial tiles its GEMM left (stream_k_partials).
"""
from __future__ import annotations

from typing import Any

from boltbeam.quantization.quant import load_quant_registry

DEFAULT_ACTIVATIONS = {"in": "F16", "out": "F16"}  # an engine adapter that names no formats: fp16 in and out, said


def elem_bytes(fmt:str) -> float:
  """Bytes per element of a format in data/quants.json (Q8_1 1.125, F32 4, F16 2); an absent format is refused."""
  q = load_quant_registry(activations=True).get(str(fmt).upper())
  if q is None:
    raise KeyError(f"format {fmt} is not in data/quants.json: its bytes per element are not known")
  return q.bytes_per_elem


def gemm(m:float, n:int, k:int, weight_bytes:float, activations:dict[str, str] | None = None) -> dict[str, float]:
  """One launch: operations and bytes (the weight once, M x K in and M x N out in the engine's formats)."""
  a = activations or DEFAULT_ACTIVATIONS
  return {"flops": 2.0 * m * n * k, "bytes": float(weight_bytes) + m * k * elem_bytes(a["in"]) + m * n * elem_bytes(a["out"])}


def recurrent_state_bytes(profile:dict[str, Any], element_bytes:int = 4) -> float | None:
  """The fixed state a recurrent block holds whatever the context (Mamba2, Gated DeltaNet), over its layers: the
  convolution's last conv_kernel - 1 inputs of inner + 2 x groups x state channels, and the state matrix of state x
  inner, at element_bytes (llama.cpp keeps both in f32). None when the model has no recurrent layers."""
  r = (profile.get("metadata") or {}).get("recurrent") or {}
  if not (r.get("layer_count") and r.get("state_size") and r.get("inner_size")):
    return None
  conv = (int(r.get("conv_kernel") or 1) - 1) * (r["inner_size"] + 2 * int(r.get("group_count") or 0) * r["state_size"])
  return float(r["layer_count"] * (conv + r["state_size"] * r["inner_size"]) * element_bytes)


def attention(profile:dict[str, Any], length:int, chunks:list[int] | None = None, kv_format:str = "F16",
              activations:dict[str, str] | None = None, depth:int = 0) -> dict[str, Any] | None:
  """The prompt's attention work over all layers: operations for the causal pairs, and bytes as each chunk reads
  the K and V written so far plus its own Q and output. None when the profile does not say the attention shape.
  depth: tokens already in the cache before these `length` (a turn appended to a session), so the token at offset i
  attends depth + i + 1 keys."""
  att = (profile.get("metadata") or {}).get("attention") or {}
  heads, dim = att.get("head_count"), att.get("head_dim")
  attention_layers = attention_layer_count(profile)  # a hybrid model attends in some layers only
  per_token = kv_bytes_per_token(profile, kv_format)
  if not (attention_layers and heads and per_token):
    return None
  pairs = length * depth + length * (length + 1) / 2.0
  flops = 4.0 * dim * heads * pairs * attention_layers
  a = activations or DEFAULT_ACTIVATIONS
  done, read = depth, 0.0
  for w in chunks or [length]:
    done += w
    read += done * per_token / attention_layers + w * heads * dim * (elem_bytes(a["in"]) + elem_bytes(a["out"]))
  at = f" after {depth} tokens in the cache" if depth else ""
  return {"flops": flops, "bytes": read * attention_layers, "pairs_per_head": pairs, "layers": attention_layers,
          "words": (f"causal{at}: {length} x {depth} + {length} x {length + 1} / 2" if depth else
                    f"causal: {length} x {length + 1} / 2") + f" query-key pairs per head, 4 x {dim} operations each, "
                   f"{heads} heads, {attention_layers} layers"}


SWEEP_FIRST = 512  # the first prompt length of a sweep
SWEEP_STEP = 4  # each next length is this many times the last, up to the binding cap, which is the last point


def attention_layer_count(profile:dict[str, Any]) -> int | None:
  """The profile's attended layers, shared by KV limits and flash comparisons."""
  return ((profile.get("metadata") or {}).get("attention") or {}).get("layer_count") or profile.get("layer_count")


def kv_bytes_per_token(profile:dict[str, Any], kv_format:str = "F16", element_bytes:float | None = None) -> float | None:
  """The KV cache one token adds: K and V, kv heads x head dim, in the layers that attend, in the cache's format (or
  at element_bytes, when the caller knows only the element's size). The one rule for KV bytes."""
  att = (profile.get("metadata") or {}).get("attention") or {}
  layers = attention_layer_count(profile)
  if not (layers and att.get("head_count_kv") and att.get("head_dim")):
    return None
  return 2.0 * att["head_count_kv"] * att["head_dim"] * layers * (element_bytes or elem_bytes(kv_format))


def sweep(profile:dict[str, Any], *, memory_bytes:float | None, weight_bytes:float | None,
          kv_format:str = "F16") -> dict[str, Any]:
  """The prompt lengths a prefill is measured at: SWEEP_FIRST, then SWEEP_STEP times each, and the binding cap last.
  The model's cap is the context it declares (profile metadata context); the memory's cap is what the GPU's memory
  holds after the weights, at the KV bytes one token adds (no reserve for the engine's buffers: the run is the check).
  The lower binds, cut to a multiple of SWEEP_FIRST."""
  ctx = (profile.get("metadata") or {}).get("context") or {}
  per = kv_bytes_per_token(profile, kv_format)
  model_cap = ctx.get("length")
  memory_cap = int((memory_bytes - weight_bytes) // per) if memory_bytes and weight_bytes and per else None
  caps = {k: v for k, v in (("model", model_cap), ("memory", memory_cap)) if v}
  if not caps:
    raise ValueError("no cap for the sweep: the model declares no context length and the memory left for the KV is not known")
  binding = min(caps, key=lambda k: caps[k])
  cap = caps[binding] if binding == "model" else caps[binding] // SWEEP_FIRST * SWEEP_FIRST
  lengths, x = [], SWEEP_FIRST
  while x < cap:
    lengths.append(x)
    x *= SWEEP_STEP
  lengths.append(cap)
  return {"lengths": lengths, "model_cap": model_cap, "model_cap_source": ctx.get("source"), "memory_cap": memory_cap,
          "binding": binding, "cap": cap, "kv_bytes_per_token": per, "kv_format": kv_format, "weight_bytes": weight_bytes,
          "memory_bytes": memory_bytes,
          "words": f"{SWEEP_FIRST} x {SWEEP_STEP} per step to the {binding} cap {cap:,}"
                   + (f" (model {model_cap:,} from {ctx.get('source')}" if model_cap else " (the model declares none")
                   + (f"; memory {memory_cap:,}: {(memory_bytes or 0) / 1e9:.1f} GB less {(weight_bytes or 0) / 1e9:.1f} GB of "
                      f"weights at {per:,.0f} bytes of {kv_format} KV per token)" if memory_cap else ")")}


def path_shares(paths:dict[str, int]) -> dict[str, float]:
  """A kernel's operations over its matrix paths, by its instruction sites on each (one path: all of them)."""
  total = sum(paths.values())
  return {p: n / total for p, n in paths.items()} if total else {}


def compute_ms(flops:float, paths:dict[str, int], peaks:dict[str, float]) -> float | None:
  """Each path's share of the operations at that path's own peak, summed; None if a path has no measured peak."""
  shares = path_shares(paths)
  if not shares or not flops or any(p not in peaks for p in shares):
    return None
  return sum(flops * f / (peaks[p] * 1e12) for p, f in shares.items()) * 1e3


def effective_peak(paths:dict[str, int], peaks:dict[str, float]) -> float | None:
  """The peak the paths make together for this kernel's mix: operations over compute_ms."""
  ms = compute_ms(1e12, paths, peaks)
  return 1e12 / (ms * 1e-3) / 1e12 if ms else None


def stream_k_partials(m:int, n:int, k:int, *, blocks:int, tile_m:int, tile_n:int, k_block:int, k_iter:int) -> dict[str, int]:
  """What a stream-k GEMM leaves its fixup: `blocks` blocks share tiles x k-blocks of work in equal contiguous spans
  (rounded down to k_iter); a block that starts inside a tile and ends past it combines the partial tiles of the blocks
  before it in that tile. Returns {partials read, tiles fixed}. A grid of one block per tile leaves nothing."""
  per = max(k // k_block, 1)
  tiles = -(-m // tile_m) * -(-n // tile_n)
  work = tiles * per
  if blocks <= 0 or tiles % blocks == 0:
    return {"partials": 0, "tiles": 0}
  def start(b:int) -> int:
    x = b * work // blocks
    return x - (x % per) % k_iter
  reads = fixed = 0
  for b in range(blocks):
    k0, k1 = start(b), start(b + 1)
    if k0 == k1 or k0 % per == 0 or (k0 // per == k1 // per and k1 % per):
      continue
    bi, stop, any_ = b - 1, k0, False
    while bi >= 0:
      kb = start(bi)
      if kb == stop:
        bi, stop = bi - 1, kb
        continue
      any_, reads = True, reads + 1
      if kb % per == 0 or kb // per < k0 // per:
        break
      bi, stop = bi - 1, kb
    fixed += any_
  return {"partials": reads, "tiles": fixed}


def ceiling(flops:float, nbytes:float, peak_tflops:float | None, bandwidth_gbs:float | None,
            compute:float | None = None) -> dict[str, Any]:
  """ms at the ceiling and which side binds; compute_ms None when no peak is known for the kernel's path. A kernel
  on several paths passes its own compute ms (compute_ms)."""
  if compute is None:
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
