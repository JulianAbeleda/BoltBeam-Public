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

from typing import Any, Callable

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


def recurrent(profile:dict[str, Any], length:int, chunks:list[int] | None = None,
              element_bytes:int = 4) -> dict[str, Any] | None:
  """A Gated DeltaNet block's prompt work over its layers, plus its short causal
  convolution), from the profile's recurrent shape. None when the model has no recurrent layers.

  The scan keeps an S x S state per head (S = state_size, H = inner_size / S heads) and per token per state column
  does kv = S.k (2S), the update S = g S + k delta (3S) and the output S.q (2S): 7 x S^2 x H operations per token
  per layer. Its bytes: q, k and v in, g and beta in, the output out, per token, and the whole state read once and
  written once per launch (one launch per engine chunk). The convolution (width conv_kernel over inner + 2 x groups x
  state channels) adds 2 x width operations and an f32 read and write per channel per token. The scan issues no
  matrix instruction, so its compute side is the fp32 vector rate; the context depth does not change it (the state
  is fixed-size)."""
  r = (profile.get("metadata") or {}).get("recurrent") or {}
  if r.get('kind') == 'mamba2':
    raise ValueError('Mamba2 requires resolved state evidence and SSD stage work; the DeltaNet formula does not apply')
  layers, state, inner = r.get("layer_count"), r.get("state_size"), r.get("inner_size")
  if not (layers and state and inner):
    return None
  heads = inner // state
  groups = int(r.get("group_count") or 0)
  width = int(r.get("conv_kernel") or 0)
  qk = (groups * state if groups else inner)
  launches = len(chunks) if chunks else 1
  scan_flops = 7.0 * state * state * heads * length
  scan_bytes = launches * 2.0 * heads * state * state * element_bytes + length * (2 * qk + 2 * inner + 2 * heads) * element_bytes
  channels = inner + 2 * groups * state
  conv_flops = 2.0 * width * channels * length if width else 0.0
  conv_bytes = 2.0 * channels * length * element_bytes if width else 0.0
  return {"flops": layers * (scan_flops + conv_flops), "bytes": layers * (scan_bytes + conv_bytes), "layers": layers,
          "path": FP32_PATH,
          "words": f"recurrent scan: 7 x {state}^2 x {heads} heads per token, state {heads} x {state}^2 read and written "
                   f"once per launch ({launches}), {layers} layers" + (f"; convolution width {width} over {channels} "
                   f"channels" if width else "")}


# The arithmetic an engine's recurrent kernels execute, when its adapter names the algorithm they run. The ceiling
# stays the work the model needs (recurrent(): the same on every engine); this is a separate, labelled figure.
#
# chunked_wy: the delta rule in chunks of C tokens with the WY / UT transform (Yang et al., "Parallelizing Linear
# Transformers with the Delta Rule over Sequence Length", 2024), run chunk-parallel: the prompt is cut into spans,
# each span builds its own transfer matrix and local state, a scan fixes the state at each span's start, then the
# output pass recomputes every chunk from the fixed state. Per chunk, per head, state S (each product 2 m n k):
#   five C x C x S products: K K^T (the UT system), X = T K, Q K^T, the new values, the intra-chunk output;
#   seven C x S x S products: the span's transfer and local state (two each), and in the output pass the output
#   from the state, the state against K, and the state update;
#   the inverse of the C x C triangular system, by doubling from 8 x 8 diagonal blocks (inverted without matrix
#   instructions): at size n, C / 2n blocks of two n x n x n products, each n padded to the matrix tile's 16 rows.
# Per span, per head: its first chunk starts from the identity and zero, so it skips two C x S x S products, and
# the fixup scan composes one S x S x S product. A span of None is the whole prompt. Tokens round up to whole chunks.
MMA_ROWS = 16  # the matrix instruction tile's rows (16 x 8 x 16); a smaller product pads up to them
INVERSE_BASE = 8  # the triangular inverse's diagonal blocks, inverted without matrix instructions


def chunked_wy_ops(state:int, chunk:int, length:int, span:int | None = None) -> float:
  """Executed operations per head of the chunk-parallel chunked WY delta rule over `length` tokens."""
  s, c = state, chunk
  blocks, spans = -(-length // c), (-(-length // span) if span else 1)
  inverse, n = 0, INVERSE_BASE
  while 2 * n <= c:
    inverse += (c // (2 * n)) * 2 * 2 * max(n, MMA_ROWS) * n * n
    n *= 2
  return float(blocks * (5 * 2 * c * c * s + 7 * 2 * c * s * s + inverse) - spans * 2 * 2 * c * s * s + spans * 2 * s ** 3)


RECURRENT_FORMS: dict[str, tuple[str, Any]] = {
  "chunked_wy": ("the chunk-parallel chunked WY form of the delta rule", chunked_wy_ops),
}


def recurrent_executed(profile:dict[str, Any], length:int, form:dict[str, Any] | None) -> dict[str, Any] | None:
  """The recurrent scan's executed operations under the engine's named algorithm (form: {"name", "chunk", optional
  "span"}), over its layers, with its words; None when no form is named, the name is not in RECURRENT_FORMS, or
  there is no recurrent block."""
  r = (profile.get("metadata") or {}).get("recurrent") or {}
  layers, state, inner = r.get("layer_count"), r.get("state_size"), r.get("inner_size")
  if not (form and form.get("name") in RECURRENT_FORMS and layers and state and inner and form.get("chunk")):
    return None
  words, per = RECURRENT_FORMS[form["name"]]
  chunk, span = int(form["chunk"]), (int(form["span"]) if form.get("span") else None)
  heads = inner // state
  spans = f"spans of {span}" if span else "one span"
  return {"flops": layers * heads * per(state, chunk, length, span),
          "words": f"executed by {words}, chunk {chunk}, {spans}: per chunk per head five {chunk} x {chunk} x {state} "
                   f"and seven {chunk} x {state} x {state} products, the triangular inverse padded to the "
                   f"{MMA_ROWS}-row matrix tile, and per span one {state}^3 fixup less two skipped products; "
                   f"{heads} heads, {layers} layers (the ceiling counts the model's 7 x {state}^2 per token)"}


FP32_PATH = "fp32_fma"  # the measured vector path (cuda_compute.PATHS) a kernel with no matrix instruction runs on


def peaks_at(peaks:dict[str, float], source:dict[str, Any] | None, sm_mhz:float | None) -> tuple[dict[str, float], str]:
  """Each measured path's rate at the SM clock the run actually held: its measured rate x observed / measured clock
  (the probe stores both, per path). Paths without a measured clock, and every path when no clock was observed,
  keep their measured rate; the words say which. Bandwidth is not scaled (the memory clock does not follow the SM
  clock under a power cap)."""
  paths = (source or {}).get("paths") or {}
  if not sm_mhz:
    return dict(peaks), "peaks at the clock they were measured at (no SM clock was observed for this run)"
  out, scaled, measured = {}, [], set()
  for name, tops in peaks.items():
    at = (paths.get(name) or {}).get("sm_mhz")
    if at:
      out[name] = tops * sm_mhz / at
      scaled.append(name)
      measured.add(round(at))
    else:
      out[name] = tops
  if not scaled:
    return out, "peaks at the clock they were measured at (the peak source records no clock)"
  at_words = ", ".join(f"{m:,}" for m in sorted(measured))
  return out, f"peaks scaled to the {sm_mhz:,.0f} MHz SM clock observed (measured at {at_words} MHz)"


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


# The matrix paths (cuda_compute.PATHS) that hold a weight format's values exactly and accumulate in fp32 or int32.
# Integer weights of at most 8 bits (block quants, INTn) are exact as int8, fp16 or bf16 operands, their scales applied
# outside the product; a float weight only on its own type. fp16 with fp16 accumulation, tf32 and fp8 round.
FLOAT_WEIGHT_PATHS = {"F16": ("fp16",), "BF16": ("bf16",), "F32": (FP32_PATH,),
                      "FP8_E4M3": ("fp8", "fp16", "bf16"), "NVFP4_G16": ("nvfp4", "fp16", "bf16")}
INTEGER_WEIGHT_PATHS = ("int8", "fp16", "bf16")
CHIP_BEST_WORDS = ("ms at chip best is the same bytes at the bandwidth, with the operations at the fastest path this "
                   "chip measured that holds the row's weights exactly (fp32 or int32 accumulate), or the engine's own "
                   "path when that is faster")


def weight_paths(fmt:str | None) -> tuple[str, ...]:
  """The paths that hold a weight (or KV) format's values exactly; () for a format BoltBeam does not know."""
  if not fmt:
    return ()
  name = str(fmt).upper()
  q = load_quant_registry(activations=True).get(name)
  if q is not None:
    return FLOAT_WEIGHT_PATHS.get(name, ()) if q.dequant_family == "float_passthrough" else INTEGER_WEIGHT_PATHS
  bits = name[3:].split("_")[0] if name.startswith("INT") else ""
  return INTEGER_WEIGHT_PATHS if bits.isdigit() and int(bits) <= 8 else ()


def chip_best(fmt:str | None, peaks:dict[str, float], own_peak:float | None = None) -> dict[str, Any] | None:
  """THE chip-best path for a row's data: the fastest measured path that holds `fmt` exactly, or the row's own path
  (own_peak) when that is faster, so the chip-best ceiling is never above the engine-path one. None when neither has
  a measured peak. {"path", "peak_tflops"}."""
  got = [(peaks[p], 1, p) for p in weight_paths(fmt) if peaks.get(p)]  # a named path wins a tie
  if own_peak:
    got.append((own_peak, 0, "the engine's own path"))
  if not got:
    return None
  peak, _, path = max(got)
  return {"path": path, "peak_tflops": peak}


def best_ms(flops:float, nbytes:float, fmt:str | None, peaks:dict[str, float], bandwidth_gbs:float | None,
            engine_ms:float | None, own_peak:float | None) -> tuple[float | None, str | None]:
  """A row's chip-best ceiling and the path it names (None: the engine's own). It needs no engine path: a row whose
  kernel's path is unknown still has one when its format has a measured path. Never above the engine's ceiling."""
  b = chip_best(fmt, peaks, own_peak) if flops else None
  if b is None or b["path"] == "the engine's own path":
    return engine_ms, None
  at = ceiling(flops, nbytes, b["peak_tflops"], bandwidth_gbs)["ms"]
  return (min(at, engine_ms) if engine_ms is not None else at), b["path"]


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


def turn_ceiling(profile:dict[str, Any], roles:list[dict[str, Any]], length:int, depth:int, *, ubatch:int, batch:int,
                 formats:dict[str, Any], weight_path:Callable[[str], str | None], attention_paths:dict[str, int],
                 peaks:dict[str, float], bandwidth_gbs:float) -> dict[str, Any]:
  """The ceiling of `length` tokens appended after `depth` in the cache, from the profile alone (no capture): per
  weight role, per chunk, a GEMM at its path's peak or the bandwidth (a role once per model runs once per output call
  over one token: the output head); attention at the depth with its operations split over its paths by instruction
  sites; the recurrent block at the fp32 vector rate. Kernels the model counts no work for add nothing. roles:
  prefill.prefill_roles; formats: the engine's (activations, attention_activations, kv); weight_path: a weight quant's
  matrix path in this engine; ubatch, batch: the engine's chunk and the tokens per output call. Each row and the total
  also carry best_ms, the same work at the chip-best path for the row's data (chip_best)."""
  if length <= 0:
    return {"ms": 0.0, "best_ms": 0.0, "rows": [], "unmodelled": []}

  def best(row:dict[str, Any], fmt:str | None, own:float | None) -> dict[str, Any]:
    return {**row, "best_ms": best_ms(row["flops"], row["bytes"], fmt, peaks, bandwidth_gbs, row["ms"], own)[0]}
  chunks = [min(ubatch, length - i) for i in range(0, length, ubatch)]
  outputs = -(-length // batch)
  acts = formats.get("activations")
  rows, unmodelled = [], []
  for r in roles:
    path = weight_path(r["quant"])
    once = r["count"] == 1
    role_acts = (formats.get("activations_by_quant") or {}).get(r["quant"], acts)
    work_rows = [gemm(1, r["n"], r["k"], r["weight_bytes"], role_acts)] * outputs if once else \
                [gemm(w, r["n"], r["k"], r["weight_bytes"], role_acts) for w in chunks for _ in range(r["count"])]
    flops, nbytes = sum(x["flops"] for x in work_rows), sum(x["bytes"] for x in work_rows)
    peak = peaks.get(path) if path and not once else None
    if path and not once and peak is None:
      unmodelled.append(f"{r['role']} {r['quant']}: no measured peak for {path}")
    row = {"name": f"{r['role']} {r['quant']}", "flops": flops, "bytes": nbytes, **ceiling(flops, nbytes, peak, bandwidth_gbs)}
    rows.append(best(row, None if once else r["quant"], peak))
  att = attention(profile, length, chunks, formats.get("kv") or "F16", formats.get("attention_activations") or acts, depth=depth)
  if att:
    comp = compute_ms(att["flops"], attention_paths, peaks) if attention_paths else None
    if comp is None:
      unmodelled.append("attention: its instruction paths were not read, so only its bytes bound it")
    row = {"name": "attention", "flops": att["flops"], "bytes": att["bytes"], **ceiling(att["flops"], att["bytes"], None, bandwidth_gbs, comp)}
    rows.append(best(row, formats.get("kv") or "F16", effective_peak(attention_paths, peaks) if attention_paths else None))
  rec = recurrent(profile, length, chunks)
  if rec:
    row = {"name": "recurrent", "flops": rec["flops"], "bytes": rec["bytes"],
           **ceiling(rec["flops"], rec["bytes"], peaks.get(rec["path"]), bandwidth_gbs)}
    rows.append({**row, "best_ms": row["ms"]})  # its state's format is not a weight's: its own path
  return {"ms": sum(r["ms"] or 0.0 for r in rows), "best_ms": sum(r["best_ms"] or 0.0 for r in rows), "rows": rows, "unmodelled": unmodelled, "chunks": len(chunks),
          "outputs": outputs}


def mfu(flops:float, ms:float, peak_tflops:float | None) -> float | None:
  """Operations done over what the path's peak could do in the same time, %."""
  if not flops or not ms or not peak_tflops:
    return None
  return 100.0 * flops / (ms * 1e-3 * peak_tflops * 1e12)
