"""Recommended setup: the engine settings for this model on these GPUs, derived from the checkpoint, the profile and
this machine's measured facts, before any run (docs/recommended-setup.md).

One row per layout the engine can run (one GPU, one replica per GPU, split by rows, split by layer). Every figure
is {value, formula, inputs, kind}: kind is "derived" (arithmetic on the inputs), "measured" (a number read from an
engine start or run on this machine) or "upper bound" (derived without the engine's measured overhead). Nothing
here runs the engine: `overhead_from_log` reads one start's own lines (the Check) and `facts_with_overhead` caches
the result in the machine facts, where the next call finds it.

  weights per GPU   checkpoint tensor bytes (safetensors headers: dtype and shape, not names), less the vision tower
                    when the engine drops it; a row split halves all but the one-dimensional tensors (norms)
  max context       (usable - weights - engine overhead - state x seqs) / KV bytes per token, capped at the model's
                    own context; usable = utilization x the smallest used GPU's memory
  max sequences     the largest B with weights + overhead + B x (state + KV at the stated context) in usable
  batch-1 limit     (weight bytes per token + KV at the stated context) / measured read bandwidth (layout.limit)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import re
import shlex
import shutil
import statistics
import signal
import struct
import subprocess
import tempfile
import time
from typing import Any, NamedTuple

from boltbeam.collectors import engine_common as ec, vllm_mtp
from boltbeam.core.refused import Refused
from boltbeam.profile import loaders
from boltbeam.roofline import prefill_work as pw
from boltbeam.target import gpu as tgpu, vendors
from boltbeam.workflow import layout as lay, offload as off

MTP_TIE = 0.05  # two draft counts within this share are one speed; derivation (run-to-run noise of one k) below, set from data
MTP_BELOW = 0.5  # a stated context under this share of the shallowest measured depth is outside the measurement too
TIE = 0.05  # limits within this share of the best are one prediction (docs/multi-gpu-recommendation-checks.md)
UTILIZATION = 0.90  # vLLM --gpu-memory-utilization, Arkey's value
CONTEXT = 4096  # the context the sequence count and the speed are stated at
# flags the user's engine host owns: a copied setup must not set them (arkey_v3 internal/runtime/vllm.go)
RESERVED = ("--model", "--host", "--port", "--served-model-name", "--max-model-len", "-m")
# measured on the 5090, ~/vllm-bench/layout27b/README.md: engine efficiency against the limit, per engine and chip
EFFICIENCY = {("vllm", "nvidia_sm120"): (0.79, "vLLM Marlin on the 5090, batch 1, ~/vllm-bench/layout27b"),
              ("llama.cpp", "nvidia_sm120"): (0.775, "llama.cpp on the 5090, batch 1, ~/vllm-bench/layout27b"),
              # a layout's own efficiency: (engine, chip, layout)
              ("vllm", "amd_radeon_ai_pro_r9700_32g", "row"): (0.77, "vLLM TP=2 on 2 x R9700, 56.1 tok/s of a 73.0 limit, TRITON_ATTN, "
                                                               "GPU_MAX_HW_QUEUES=1, the 27B AWQ, 2026-10-27")}
# measured on a model of this shape (hidden, layers, ffn) on the 5090 with vLLM: the 27B AWQ
MEASURED = {("vllm", "nvidia_sm120", 5120, 64, 17408): {
  "throughput": (1863, "tok/s at batch 32 on one 5090, the 27B sweep, ~/vllm-bench/layout27b/sweep.json"),
  "mtp": ("1.6x at k=3 on text, accept length 2.75", "~/vllm-bench/layout27b/README.md")}}
MTP_NOTE = "1.6x at k=3 on text (accept length 2.75), measured on the 27B on the 5090 only"
# MTP, measured: mean accept length per draft count k on text (greedy), per model shape (hidden, layers, ffn)
MTP_ACCEPT = {(5120, 64, 17408): ({1: 1.86, 2: 2.47, 3: 2.88}, "the 27B AWQ, 4 text prompts x 256 tokens, greedy, vLLM 0.31, "
                                  "2 x R9700, 2026-10-27 (k=3 read 2.75 on the 5090)")}
# MTP's per-step cost besides the reads (the verify's extra tokens, rejection), per engine and chip: the measured step
# less step(1 token) + k x draft read, at this chip's engine efficiency (mtp_speed)
MTP_FIXED = {("vllm", "nvidia_sm120"): (0.55, "5090, the 27B AWQ k=3 at 144 tok/s (0.09 ms) and its 4-bit-head variant at 188 (1.02 ms)"),
             ("vllm", "amd_radeon_ai_pro_r9700_32g"): (2.9, "2 x R9700 TP=2, the 27B AWQ k=1 and k=2: 23.95 and 27.78 ms steps "
                                                       "(2.68, 3.08 ms), 2026-10-27")}
# MTP measured, where the formula is not enough: tok/s per k, per (engine, chip, hidden, layers, ffn, layout, lm_head GB),
# short context; the head's size tells a 4-bit-head variant of the same model apart
MTP_MEASURED = {("vllm", "amd_radeon_ai_pro_r9700_32g", 5120, 64, 17408, "row", 2.5): (
  {1: 77.9, 2: 88.9, 3: 80.8}, "TP=2, TRITON_ATTN, GPU_MAX_HW_QUEUES=1, 4 text prompts x 256 tokens, 2026-10-27; k=3 runs 15% under the "
  "formula (~4 ms more per step, also with a split-KV verify: 86.0 against 92.3 at k=2)"),
  ("vllm", "amd_radeon_ai_pro_r9700_32g", 5120, 64, 17408, "row", 0.7): (
  {2: 115.5, 3: 105.5}, "the 4-bit-head variant, same run setup with a split-KV verify, 2026-10-27; k=3 17% under the formula")}
RUN_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "CUDA_HOME": "/usr/local/cuda"}  # collectors/vllm_decode.RUN_ENV, the 5090 box's CUDA
OVERHEAD = "engine_overhead"  # in the machine facts: {engine: {checkpoint path: {cache state: {bytes, source, ...}}}}
COLD_COMPILE_S = 5.0  # vLLM torch.compile at or above this is a cold compile cache (measured 25.4 s cold, 0.6 s warm, 27B AWQ on the 5090)


def fig(value:Any, formula:str, inputs:list[str], kind:str = "derived") -> dict[str, Any]:
  return {"value": value, "formula": formula, "inputs": inputs, "kind": kind}


# --- the checkpoint ----------------------------------------------------------------------------------------------

def tensor_bytes(source:str | pathlib.Path) -> dict[str, Any]:
  """The checkpoint's bytes by what they are, from the safetensors headers (data_offsets, shape): total, the
  vision tower's, the draft head's, and the one-dimensional tensors' (a row split keeps these whole on every GPU).
  A GGUF reads its tensor table (gguf_tensor_bytes); one it cannot read counts its file bytes."""
  path = pathlib.Path(source).expanduser()
  out = {"total": 0, "vision": 0, "mtp": 0, "mtp_layers": 0, "head": 0, "one_dim": 0, "host": 0, "files": 0, "source": "safetensors headers"}
  for f in (f for f in loaders._safetensors_files(path) if f.suffix == ".safetensors"):
    out["files"] += 1
    for name, t in loaders._read_safetensors_header(f).items():
      if name == "__metadata__":
        continue
      n = t["data_offsets"][1] - t["data_offsets"][0]
      out["total"] += n
      out["vision"] += n if any(k in name for k in loaders._NOT_IN_TEXT_PREFILL[1:]) else 0
      out["mtp"] += n if "mtp" in name else 0
      out["head"] += n if "lm_head" in name else 0
      out["one_dim"] += n if len(t.get("shape") or []) == 1 else 0
  out["mtp_layers"] = 1 if out["mtp"] else 0  # a safetensors draft head is one layer
  if not out["files"]:
    files = _gguf_files(path)
    split = len(files) > 1 or bool(re.search(r"-\d+-of-\d+\.gguf$", path.name))
    try:
      kv = None
      for f in files:
        kv = gguf_tensor_bytes(f, out, kv)
      out.update(files=len(files), source="GGUF tensor tables")
    except (ValueError, EOFError, OSError, struct.error) as exc:
      if split:  # a split model with a shard unread is a fit that silently misses that shard's bytes
        raise Refused(f"{path.name} is a split GGUF and a shard could not be read ({exc}); the fit needs every shard") from exc
      out = {**out, "total": 0}
    if not out["total"]:
      out.update(total=sum(f.stat().st_size for f in files), vision=0, mtp=0, mtp_layers=0, head=0, one_dim=0, host=0, files=len(files), source="file sizes")
  return out


SHARD = re.compile(r"(.+)-(\d+)-of-(\d+)\.gguf")
HOST_ONLY = ("token_embd", "per_layer_token_embd")  # tensor roles llama.cpp keeps in host RAM and reads one row of per token, in whichever shard they sit


def _gguf_files(path:pathlib.Path) -> list[pathlib.Path]:
  """The model's GGUF files, a split model's every shard (<stem>-00001-of-0000N.gguf ...) after its first. Refused: a shard
  that is missing, or a later shard named alone (the first holds the metadata the others lack)."""
  out: list[pathlib.Path] = []
  for f in [path] if path.is_file() else sorted(path.glob("*.gguf")):
    m = SHARD.fullmatch(f.name)
    if m and int(m.group(2)) > 1:
      if path.is_file():
        raise Refused(f"{f.name} is shard {int(m.group(2))} of a split GGUF; name its first shard, {m.group(1)}-{1:0{len(m.group(2))}d}-of-{m.group(3)}.gguf")
      continue  # reached through its first shard
    shards = [f.with_name(f"{m.group(1)}-{i:0{len(m.group(2))}d}-of-{m.group(3)}.gguf") for i in range(1, int(m.group(3)) + 1)] if m else [f]
    if gone := [s.name for s in shards if not s.is_file()]:
      raise Refused(f"{f.name} is split in {len(shards)} files and {', '.join(gone)} is missing; the fit needs every shard")
    out += shards
  return out


def gguf_tensor_bytes(f:pathlib.Path, out:dict[str, Any], first_kv:dict[str, Any] | None = None) -> dict[str, Any]:
  """Adds one GGUF's tensor bytes to out, each the gap to the next tensor's offset (the last runs to the file end), and
  returns its metadata. llama.cpp keeps the input embeddings (HOST_ONLY roles: token_embd, per_layer_token_embd) on the host
  and does not load the draft blocks (the last nextn_predict_layers): on the 27B Q4_K_M the GPU holds 14,674 MiB of the
  15,691 (682 embedding, 335 draft); a projector file (mmproj) is the vision tower. first_kv: a split model's first shard
  metadata, which the later shards (tensors only) lack."""
  from boltbeam.profile import gguf
  kv, ts, start = gguf.read_gguf_layout(f)
  meta = first_kv or kv
  arch = meta.get("general.architecture")
  draft = int(meta.get(f"{arch}.nextn_predict_layers") or 0)
  first_draft = int(meta.get(f"{arch}.block_count") or 0) - draft if draft else None
  vision = kv.get("general.type") == "mmproj"
  offs = sorted((o, name, dims) for name, dims, _, o in ts)
  for (o, name, dims), nxt in zip(offs, [o for o, *_ in offs[1:]] + [f.stat().st_size - start]):
    n = nxt - o
    out["total"] += n
    out["vision"] += n if vision else 0
    out["host"] += n if name.split(".")[0] in HOST_ONLY else 0
    out["head"] += n if name == "output.weight" else 0
    is_draft = first_draft is not None and name.startswith("blk.") and int(name.split(".")[1]) >= first_draft
    out["mtp"] += n if is_draft else 0
    out["one_dim"] += n if len(dims) == 1 else 0
  out["mtp_layers"] = draft
  return meta


def weights_resident(tb:dict[str, Any], *, text_only:bool, mtp:bool = False) -> int:
  """What the engine holds on the GPU: all tensors, less the vision tower when it is dropped, less the draft head
  unless it is used, less what it keeps on the host (a GGUF's input embedding, tensor_bytes)."""
  return tb["total"] - (tb["vision"] if text_only else 0) - (0 if mtp else tb["mtp"]) - tb.get("host", 0)


# --- the engine's overhead: measured, not derived ----------------------------------------------------------------

def parse_vllm(log:str) -> dict[str, Any]:
  """What one vLLM start says: 'Available KV cache memory: X GiB', 'GPU KV cache size: N tokens' and the served
  context (the auto-fit max_model_len, else 'Using max model len'), 'torch.compile took X s' (compile_s) and 'X GiB for
  peak activation' (activation_gib). The compile time says the cache state: a cold torch.compile cache makes vLLM
  profile a larger peak activation, so a first start serves less context than the starts after it."""
  kv = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", log)
  tokens = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log)
  served = re.search(r"Auto-fit max_model_len: reduced from \d+ to (\d+)", log) or re.search(r"Using max model len (\d+)", log)
  if not (kv and tokens):
    raise ValueError("the engine log has no 'Available KV cache memory' and 'GPU KV cache size' lines")
  comp = re.search(r"torch\.compile took\s*([\d.]+)\s*s", log)
  act = re.search(r"([\d.]+)\s*GiB for peak activation", log)
  return {"available_kv_bytes": float(kv.group(1)) * 2**30, "cache_tokens": int(tokens.group(1).replace(",", "")),
          "served_context": int(served.group(1)) if served else None,
          "compile_s": float(comp.group(1)) if comp else None, "activation_gib": float(act.group(1)) if act else None,
          "state": None if not comp else ("cold" if float(comp.group(1)) >= COLD_COMPILE_S else "warm")}


def parse_llama(log:str) -> dict[str, Any]:
  """llama.cpp's server/cli log: 'llama_context: n_ctx = N' (the context it fitted) and 'llama_kv_cache: size = X MiB
  (N cells ...)' (the KV it allocated), printed only at -lv 4 (b11571: the default prints neither, and the ready line
  is 'llama_server: listening on', older builds 'server is listening on'). The memory breakdown it prints on exit,
  after the ready line ('| - CUDA0 (...) | total = free + (self = model + context + compute) + unaccounted |', MiB),
  says what it left free and its context buffers (KV and recurrent state): llama.cpp holds only the context asked
  for, so the free memory is room the overhead must not absorb."""
  ctx = re.search(r"n_ctx\s*=\s*(\d+)", log)
  kv = re.search(r"KV (?:buffer|cache)[^\n]*?size\s*=\s*([\d.]+)\s*MiB(?:[^\n]*?\(\s*(\d+)\s*cells)?", log) or \
    re.search(r"llama_kv_cache[^\n]*?size\s*=\s*([\d.]+)\s*MiB(?:[^\n]*?\(\s*(\d+)\s*cells)?", log)
  if not (ctx and kv):
    raise ValueError("the engine log has no 'n_ctx' and KV cache 'size = ... MiB' lines")
  rows = re.findall(r"\|\s*-\s*(?!Host)[^|\n]+\|\s*\d+\s*=\s*(\d+)\s*\+\s*\(\s*\d+\s*=\s*\d+\s*\+\s*(\d+)\s*\+",
                    log[log.rfind("listening on"):] if "listening on" in log else "")
  return {"available_kv_bytes": float(kv.group(1)) * 2**20, "cache_tokens": int(kv.group(2) or ctx.group(1)),
          "served_context": int(ctx.group(1)),
          **({"free_bytes": float(rows[-1][0]) * 2**20, "context_bytes": float(rows[-1][1]) * 2**20} if rows else {})}


def overhead_from_log(log:str, *, memory_bytes:float, utilization:float, weights:float, parse=parse_vllm,
                      scratch:float = 0.0) -> dict[str, Any]:
  """The engine's own figures (parse) and the overhead they imply: the memory the engine took besides the
  checkpoint's tensor bytes and the KV cache, usable (utilization x memory) - tensor bytes - the KV the engine
  holds. It absorbs repacked weights, activations and graphs, so it is a property of this engine on this machine
  and checkpoint, read, not derived. An engine that says what it left free (llama.cpp's exit breakdown) has that and
  its context buffers taken off instead, else the overhead would hold the free memory. scratch: the engine's
  per-token buffer besides the KV (Engine.scratch), taken off at the cells it holds, so the overhead is the fixed part."""
  got = parse(log)
  held = got["context_bytes"] + got["free_bytes"] if "free_bytes" in got else got["available_kv_bytes"]
  return {**got, "bytes": utilization * memory_bytes - weights - held - scratch * got["cache_tokens"], "utilization": utilization,
          "weights": weights}


def _by_path(entry:dict[str, Any] | None) -> dict[str, dict[str, Any]]:
  """An engine's cached overheads as {checkpoint path: {state: entry}}; a flat entry (an older cache) is one path, one state."""
  if not entry:
    return {}
  return {"*": {"any": entry}} if "bytes" in entry else dict(entry)


def facts_with_overhead(facts:dict[str, Any], engine:str, got:dict[str, Any], source:str, path:str | None = None) -> dict[str, Any]:
  """The machine facts with this engine's measured overhead cached (the Check's result), per checkpoint path and per
  cache state (cold or warm where the engine says; else 'any'): a start never overwrites another state's figure."""
  cur, key = _by_path((facts.get(OVERHEAD) or {}).get(engine)), str(path or "*")
  cur[key] = {**cur.get(key, {}), got.get("state") or "any": {**got, "source": source}}
  return {**facts, OVERHEAD: {**(facts.get(OVERHEAD) or {}), engine: cur}}


MTP_CACHE = "engine_mtp"  # in the machine facts: {engine: {checkpoint path: a collectors/vllm_mtp summary}}


def facts_with_mtp(facts:dict[str, Any], engine:str, path:str, record:dict[str, Any]) -> dict[str, Any]:
  """The machine facts with an in-repo MTP measurement (collectors/vllm_mtp) cached for this checkpoint, as
  facts_with_overhead caches a Check: the per-k summary and where it came from, not its rows."""
  entry = {"depths": record["depths"], "prompts": record["prompts"], "tokens": record["tokens"],
           "long_tokens": record["long_tokens"], "measured_at": record["measured_at"],
           "vllm": sorted({e.get("vllm") for e in record.get("engines", []) if e.get("vllm")})}
  return {**facts, MTP_CACHE: {**(facts.get(MTP_CACHE) or {}), engine: {**((facts.get(MTP_CACHE) or {}).get(engine) or {}),
                                                                         str(path): entry}}}


def mtp_of(facts:dict[str, Any] | None, engine:str, path:str | None) -> dict[str, Any] | None:
  """This checkpoint's cached MTP measurement, or None: another checkpoint's acceptance is not this one's."""
  return (((facts or {}).get(MTP_CACHE) or {}).get(engine) or {}).get(str(path)) if path else None


def best_k(speeds:dict[int, dict[str, Any]]) -> int | None:
  """The draft count to recommend, from the smallest up: a larger k replaces the pick only when it beats it by more
  than MTP_TIE. Measured figures carry their prompts (per_prompt), and then the win is the median of the per-prompt
  ratios: every k ran the same prompts, and their spread is the text's (a story drafts worse than a derivation at
  every k), not noise, so it cancels in the pair. Spread across prompts as the margin picked k=1 on the 27B AWQ,
  whose k=2 beat k=1 on all eight prompts (RTX 5090, 2026-11-08).
  Measured at several depths (by_depth), the pair is judged at every depth, so the pick does not flip with the
  prompt's length: the larger k replaces the pick when it is nowhere slower by more than MTP_TIE and somewhere faster by
  more than MTP_TIE (the 27B AWQ's k=3 against k=2: 1.003 at 73 tokens, 1.129 at 2,121, so k=3 at any context); a tie at
  every depth keeps the smaller k (less verify work, a narrower spread). Only a real trade-off, faster at one depth
  and slower at another by more than MTP_TIE, is decided at the depth nearest the stated context (the figure's own)."""
  ks = sorted(k for k, f in speeds.items() if f.get("value"))
  if not ks:
    return None
  pick = ks[0]
  for k in ks[1:]:
    wins = depth_wins(speeds[k], speeds[pick])
    if wins and min(wins) >= 1 - MTP_TIE:
      replace = max(wins) > 1 + MTP_TIE
    else:
      replace = win(speeds[k], speeds[pick]) > 1 + MTP_TIE
    pick = k if replace else pick
  return pick


def depth_wins(new:dict[str, Any], old:dict[str, Any]) -> list[float]:
  """win at each depth both were measured at (by_depth: per_prompt per depth); [] with fewer than two such depths."""
  a, b = new.get("by_depth") or {}, old.get("by_depth") or {}
  common = [d for d in a if d in b and a[d] and set(a[d]) == set(b[d])]
  return [win({"per_prompt": a[d]}, {"per_prompt": b[d]}) for d in common] if len(common) > 1 else []


def win(new:dict[str, Any], old:dict[str, Any]) -> float:
  """How much faster `new` is than `old`: the median per-prompt ratio when both were measured on the same prompts,
  else the ratio of their values."""
  a, b = new.get("per_prompt") or {}, old.get("per_prompt") or {}
  if a and set(a) == set(b):
    return statistics.median(a[p] / b[p] for p in a)
  return new["value"] / old["value"]


def layout_key(path:str | None, layout:str, gpus:int) -> str:
  """The cache key of a start's overhead: the checkpoint path on one GPU; a split's carries the layout and GPU count,
  since its overhead is per GPU and smaller (activations split too: cold, 3.82 GiB on one R9700, 2.54 per GPU at TP=2)."""
  return str(path or "*") if layout == "one" else f"{path or '*'}#{layout}x{gpus}"


def cached(states:dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """The guaranteed overhead (guaranteed) and the warm one when that is a second figure."""
  measured = guaranteed(states)[0]
  return measured, states.get("warm") if measured is not states.get("warm") else None


def overhead_of(facts:dict[str, Any] | None, engine:str, path:str | None = None) -> dict[str, dict[str, Any]]:
  """The cached overheads {state: entry} for this checkpoint path; with none for it, the other checkpoint with the
  most states (flagged other_checkpoint: overhead is mostly the engine's, but it is not this checkpoint's own)."""
  by = _by_path(((facts or {}).get(OVERHEAD) or {}).get(engine))
  if path is not None and by.get(str(path)):
    return by[str(path)]
  other = max(by.values(), key=len, default={})
  return {k: {**v, "other_checkpoint": True} for k, v in other.items()}


def guaranteed(states:dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
  """The overhead the context is promised at: the cold-cache one (a first start), else the only state cached, and
  what to tell the user about the states that are missing."""
  if "cold" in states:
    return states["cold"], None if "warm" in states else "Contexts are for a first start (cold compile cache), the only one measured; later starts serve more."
  if "warm" in states:
    return states["warm"], "Contexts are for a warm compile cache, the only one measured: a first start serves less; run Check with the cache cleared for that figure."
  one = states.get("any")
  return one, "The overhead is another checkpoint's: run Check on this one." if one and one.get("other_checkpoint") else None


# --- one layout ----------------------------------------------------------------------------------------------------

def _split(layout:str, g:int, tb:dict[str, Any], resident:int) -> tuple[float, float, str]:
  """Weights per GPU, the share of KV and state each GPU keeps, and the formula."""
  if layout == "row":
    return (resident - tb["one_dim"]) / g + tb["one_dim"], 1.0 / g, "(weights - one-dimensional tensors) / GPUs + one-dimensional tensors; KV heads and state split"
  if layout == "layer":
    return resident / g, 1.0 / g, "weights / GPUs; each GPU holds its layers' KV and state"
  return float(resident), 1.0, "all weights, all KV"


def context_figures(profile:dict[str, Any], *, weights:float, share:float, usable:float, overhead:float | None,
                    seqs:int, kvs:dict[str, float], extra:dict[str, float] | None = None) -> dict[str, Any]:
  """Max context per KV dtype on one GPU of a layout. With no measured overhead the number is an upper bound."""
  cap = ((profile.get("metadata") or {}).get("context") or {}).get("length")
  state = (pw.recurrent_state_bytes(profile) or 0.0) * share
  out = {}
  for label, elem in kvs.items():
    per = pw.kv_bytes_per_token(profile, element_bytes=elem)
    room = usable - weights - (overhead or 0.0) - state * seqs
    if not per:  # no full-attention layers: the model's own maximum is the only limit
      out[label] = fig(cap, "the model's context length; it has no per-token KV", [f"context {cap}"], "derived")
      continue
    per = per * share + (extra or {}).get(label, 0.0)  # the engine's per-token scratch besides the KV (Engine.scratch)
    tokens = int(room // per) if room > 0 else 0
    capped = min(tokens, cap) if cap else tokens
    out[label] = fig(capped, "(usable - weights - overhead - state x seqs) / KV bytes per token, capped at the model's context",
                     [f"usable {usable / 2**30:.2f} GiB", f"weights {weights / 2**30:.2f} GiB",
                      f"overhead {overhead / 2**30:.2f} GiB (measured)" if overhead is not None else "overhead not measured here",
                      f"state {state / 2**20:.0f} MiB x {seqs} seqs", f"{per:,.0f} bytes {label} KV per token" +
                      (f" (with {extra[label]:,.0f} of the engine's scratch)" if (extra or {}).get(label) else ""),
                      f"model cap {cap}"], "derived" if overhead is not None else "upper bound")
    out[label]["uncapped"] = tokens
  return out


def max_seqs(profile:dict[str, Any], *, weights:float, share:float, usable:float, overhead:float | None,
             context:int, elem:int, extra:float = 0.0) -> dict[str, Any]:
  """The largest B where weights + overhead + B x (state + KV at the context) fits one GPU of the layout."""
  per = ((pw.kv_bytes_per_token(profile, element_bytes=elem) or 0.0) * share + extra) * context + (pw.recurrent_state_bytes(profile) or 0.0) * share
  room = usable - weights - (overhead or 0.0)
  return fig(int(room // per) if per and room > 0 else 0, "(usable - weights - overhead) / (state + KV at the context)",
             [f"context {context}", f"{per / 2**20:.1f} MiB per sequence"], "derived" if overhead is not None else "upper bound")


# --- the settings ----------------------------------------------------------------------------------------------------

def vllm_settings(profile:dict[str, Any], tb:dict[str, Any], *, kv:str, seqs:int, layout:str, gpus:int, text_only:bool,
             full_context:int | None, attention:str | None = None, mtp:tuple[int, dict[str, Any]] | None = None, mtp_held=None, **_) -> list[dict[str, Any]]:
  """Each setting with what it changes and whether to use it here (docs/recommended-setup.md). A flag is None for a
  setting that is advice only."""
  cap = ((profile.get("metadata") or {}).get("context") or {}).get("length")
  state = pw.recurrent_state_bytes(profile)
  short = bool(full_context is not None and cap and full_context < cap)
  head = next((r for r in profile.get("roles") or [] if r.get("role") == "lm_head"), None)
  rows = [
    ("--kv-cache-dtype fp8", kv == "fp8", "KV bytes per token / 2: about 2x context, faster decode at long context; quality not measured",
     f"full-precision KV context {full_context} is short of the model's {cap}" if short else "full-precision KV reaches the model's context"),
    ("--language-model-only", text_only and tb["vision"] > 0, f"drops the vision weights ({tb['vision'] / 1e9:.2f} GB)", "text-only use"),
    ("--mamba-ssm-cache-dtype bfloat16", bool(state) and seqs > 1, f"linear-attention state / 2 ({(state or 0) / 1e6:.0f} to {(state or 0) / 2e6:.0f} MB per sequence)", "batch > 1"),
    (f"--max-num-seqs {seqs}", True, "the sequences the engine batches; capacity is in the layout row", "serving several users" if seqs > 1 else "one user"),
    ("--no-enable-prefix-caching", False, "'align' mode costs cache tokens on a hybrid model", "no shared prompts (not known here)"),
    (mtp_flag(mtp[0]) if mtp else "--speculative-config mtp", bool(mtp), f"MTP k={mtp[0]}: {mtp[1]['value']:.0f} tok/s, {mtp[1]['formula']}" if mtp else
     mtp_held[0] if mtp_held else f"MTP: {MTP_NOTE}", "batch 1, the largest k that beats the next smaller one by more than the tie, prompt by prompt, and is slower at no measured depth by more than it; and beats one-token decode" + (f". {mtp[1]['envelope']}" if mtp[1].get("envelope") else "") if mtp else
     mtp_held[1] if mtp_held else "batch 1 and the checkpoint has an MTP head; accept lengths or the fixed cost are not measured for this model and chip"),
    (None, False, "4-bit lm_head checkpoint: -1.9 GB read per token and per MTP draft on the 27B; +1.8 GiB for KV. 2 x R9700 TP=2: "
     "56 to 61 tok/s, MTP k=2 92 to 116 (with a split-KV verify), 98.0% top-1 agreement, perplexity 1.228 to 1.242; vLLM 0.31's "
     "Quark loader keeps lm_head bf16 (ParallelLMHead is not a LinearBase), so it needs that loader change",
     f"the head is {head['quant']}" if head and "INT4" not in str(head.get("quant")) and "Q4" not in str(head.get("quant")) else "the head is already 4-bit"),
  ]
  if attention:
    rows.append((f"--attention-backend {attention}", True, "vLLM's own pick decodes slower here, slower as the context grows", "this GPU's vendor row"))
  if layout == "row":
    rows.append((f"--tensor-parallel-size {gpus}", True, "each weight split across the GPUs, read on all at once", "chosen layout"))
  return [{"flag": f, "recommended": bool(r), "changes": c, "when": w} for f, r, c, w in rows]


def vllm_flags(profile:dict[str, Any], *, kv:str, seqs:int, layout:str, gpus:int, text_only:bool, tb:dict[str, Any],
               attention:str | None = None, mtp:tuple[int, dict[str, Any]] | None = None, **_) -> list[str]:
  """The engine command line's options (vLLM) for the setting: the fixed ones, then each recommended setting."""
  out = ["--max-model-len", "auto", "--max-num-seqs", str(seqs), "--max-num-batched-tokens", "2048",
         "--gpu-memory-utilization", f"{UTILIZATION:.2f}", "--dtype", "bfloat16"]
  if text_only and tb["vision"] > 0:
    out.append("--language-model-only")
  if attention:
    out += ["--attention-backend", attention]
  if kv == "fp8":
    out += ["--kv-cache-dtype", "fp8"]
  if pw.recurrent_state_bytes(profile) and seqs > 1:
    out += ["--mamba-ssm-cache-dtype", "bfloat16"]
  if mtp:
    out += mtp_flag(mtp[0]).split(" ", 1)
  if layout == "row":
    out += ["--tensor-parallel-size", str(gpus)]
  return out


def mtp_flag(k:int) -> str:
  return "--speculative-config " + json.dumps({"method": "mtp", "num_speculative_tokens": k}, separators=(",", ":"))


def arkey_args(argv:list[str]) -> list[str]:
  """argv as Arkey's vllm_args: the host owns the reserved options (each with its value), so they are left out."""
  out, skip = [], False
  for a in argv:
    if skip:
      skip = False
    elif a.split("=", 1)[0] in RESERVED:
      skip = "=" not in a
    else:
      out.append(a)
  return out


def llama_settings(profile:dict[str, Any], tb:dict[str, Any], *, kv:str, seqs:int, layout:str, gpus:int, full_context:int | None,
                   machine_gpus:int = 1, **_) -> list[dict[str, Any]]:
  """llama.cpp's settings: the KV format, flash attention (needed for a quantized V), full offload, the split."""
  rows = [("-ctk q8_0 -ctv q8_0", kv == "q8_0", "KV bytes per token / 1.88 (8-bit blocks of 32 + a scale): longer context; quality not measured",
           f"f16 KV context {full_context} is short of the model's" if kv == "q8_0" else "f16 KV reaches the model's context"),
          ("-fa on", True, "flash attention: required for a quantized V cache, less attention memory", "always"),
          ("-ngl 99", True, "every layer on the GPU", "the model fits"), (f"-np {seqs}", True, "parallel sequences (the context is shared)", "serving")]
  if layout in ("row", "layer"):
    rows.append((" ".join(tgpu.engine_args(layout, "llama.cpp", _llama_bench())[0]), True, "how the GPUs share the weights", "chosen layout"))
  elif machine_gpus > 1:
    rows.append((" ".join(tgpu.engine_args("one", "llama.cpp")[0]), True, "one GPU: llama.cpp splits by layer over every GPU unless told", "chosen layout"))
  return [{"flag": f, "recommended": bool(r), "changes": c, "when": w} for f, r, c, w in rows]


def llama_flags(profile:dict[str, Any], *, kv:str, seqs:int, layout:str, gpus:int, context:int, machine_gpus:int = 1,
                placement:dict[str, Any] | None = None, **_) -> list[str]:
  """llama.cpp's options: -c is the context to hold (b11571's -fit fits one only when -c is not given), then the
  recommended settings. placement: where the layers and experts sit and the expert cache, as engine settings
  (n_gpu_layers, n_cpu_moe, override_tensor, cpu_moe, moe_cache_mib, moe_cache_cpu_share, chunk_size), checked and
  turned into flags by the one table captures use (collectors/capture_settings.render_llama); all layers on the GPU
  unless it places them."""
  from boltbeam.collectors.capture_settings import all_layers, check_llama, render_llama
  placed = render_llama(placement) if placement else []
  if placement:
    check_llama(placement)
  out = ["-c", str(context // 256 * 256), "-np", str(seqs), *all_layers(placed), "-fa", "on"]  # it rounds -c up to 256 cells
  if kv == "q8_0":
    out += ["-ctk", "q8_0", "-ctv", "q8_0"]
  return out + (tgpu.engine_args(layout, "llama.cpp", _llama_bench())[0] if layout in ("row", "layer") else
                tgpu.engine_args("one", "llama.cpp")[0] if machine_gpus > 1 else []) + placed


def _llama_bench() -> str | None:
  from boltbeam.collectors.llama_bench_decode import find
  return find()


def llama_scratch(profile:dict[str, Any], kv:str, target) -> float:
  """Where ggml's flash attention on a quantized KV dequantizes one attention layer's K and V to f16 across the whole
  context (the vendor row's ggml_kv_dequant), that buffer is per token, not overhead: CUDA, 1,007 MiB more compute
  buffer at 257,792 q8_0 cells than f16 (b11571, the 27B on the 5090, equal fixed overhead within 17 MiB); Metal reads
  the quantized KV in place (the 8B on the M4: q8_0 and f16 fixed overhead both 300 MiB)."""
  layers = pw.attention_layer_count(profile)
  dequant = vendors.for_backend(target.backend).ggml_kv_dequant
  return (pw.kv_bytes_per_token(profile, element_bytes=2) or 0.0) / layers if dequant and kv != "f16" and layers else 0.0


class Engine(NamedTuple):
  """What differs per engine: its KV formats (label: bytes per element, full precision first), the memory share it
  may use, its flags and settings, how to start it once (binary, ready line) and how to read its fit figures.
  start is None where the engine prints none (Check says so)."""
  kv: dict[str, float]
  utilization: float
  flags: Any
  settings: Any
  command: Any
  binary: Any
  argv: Any
  ready: str | None
  parse: Any
  note: str = ""
  scratch: Any = None  # (profile, KV label, target) -> bytes one token adds besides its KV, else none
  reserve: float = 0.0  # memory the engine keeps free by its own rule, off the usable memory


def _llama_binary() -> str | None:
  return os.environ.get("BOLTBEAM_LLAMA_SERVER") or shutil.which("llama-server")


def _vllm_python() -> str | None:
  from boltbeam.collectors import vllm_decode
  return vllm_decode.python()


def split_kv_verify(py:str | None) -> bool:  # tools/patches/split-kv-verify.patch in this python's vLLM: no MTP context limit
  return any("MAX_3D_QUERY_LEN" in f.read_text(errors="ignore") for f in pathlib.Path(py).parent.parent.glob(
    "lib/python3*/site-packages/vllm/v1/attention/ops/triton_unified_attention.py")) if py else False


ENGINES = {
  "vllm": Engine({"bf16": 2, "fp8": 1}, UTILIZATION, vllm_flags, vllm_settings, lambda m, a: " ".join(["vllm", "serve", m, *a]),
                 _vllm_python, lambda b, m, a, port: [b, "-m", "vllm.entrypoints.openai.api_server", "--model", m, "--host", "127.0.0.1",
                                                      "--port", str(port), "--served-model-name", "recommend-check", *a],
                 "GPU KV cache size", parse_vllm),
  "llama.cpp": Engine({"f16": 2, "q8_0": 34 / 32}, 1.0, llama_flags, llama_settings, lambda m, a: " ".join(["llama-server", "-m", m, *a]),
                      _llama_binary, lambda b, m, a, port: [b, "-m", m, "--host", "127.0.0.1", "--port", str(port), "-lv", "4", *a],
                      "listening on", parse_llama, "Check hook: llama-server's n_ctx and KV size lines", llama_scratch,
                      1024 * 2**20),  # llama.cpp's own -fit target: 1024 MiB free (common_params_fit, -fitt)
  "tinygrad": Engine({"f16": 2}, 1.0, None, None, None, lambda: None, None, None, None,
                     "one GPU only; tinygrad prints no KV fit figure, so Check has nothing to read"),
}


# --- the recommendation among layouts ----------------------------------------------------------------------------

def pick(rows:list[dict[str, Any]], *, seqs:int, fits_one:bool) -> dict[str, Any]:
  """Which layout to use, by the rules in docs/recommended-setup.md and docs/multi-gpu-recommendation-checks.md:
  several users and the model fits one GPU -> one replica per GPU; the model does not fit one GPU -> split by rows,
  else by layer; batch-1 latency -> the highest limit, but only when every offered layout is scored; limits within
  TIE are one prediction and go to the layout using fewer GPUs (a layer split never wins a tie on speed); an unscored
  layout its weight reads cap at or below the best scored limit is set aside; a tie at one GPU count, or any other
  unscored layout, gives no recommendation and says why."""
  avail = {r["id"]: r for r in rows if r["available"]}
  if not fits_one:
    for k in ("row", "layer"):
      if k in avail:
        return {"id": k, "why": "the model does not fit one GPU"}
    return {"id": None, "why": "the model does not fit one GPU and no offered layout splits it"}
  if "replica" in avail and seqs > 1:
    return {"id": "replica", "why": "the model fits one GPU and several users are served"}
  scored = [r for k, r in avail.items() if k != "replica"]
  top = max((r["tok_s"] for r in scored if r.get("tok_s")), default=None)
  beaten = [r for r in scored if not r.get("tok_s") and top and (r.get("tok_s_at_most") or top + 1) <= top]
  scored = [r for r in scored if r not in beaten]
  aside = "".join(f"; {r['label']} reads its weights at {r['tok_s_at_most']:.0f} tok/s at most" for r in beaten)
  if len(scored) < 2:
    return {"id": "one", "why": f"one GPU{aside}"}
  missing = [r for r in scored if not r.get("tok_s")]
  if missing:
    r = missing[0]
    return {"id": None, "why": f"{r['label']}: {r.get('limit_reason') or 'no limit'}; remeasure this machine's facts to compare"}
  best = max(r["tok_s"] for r in scored)
  tied = [r for r in scored if r["tok_s"] >= best * (1 - TIE)]
  if len(tied) == 1:
    return {"id": tied[0]["id"], "why": f"highest limit, {tied[0]['tok_s']:.0f} tok/s"}
  fewest = min(r["gpus"] for r in tied)
  winners = [r for r in tied if r["gpus"] == fewest]
  if len(winners) == 1:
    return {"id": winners[0]["id"], "why": "tie, uses fewer GPUs"}
  return {"id": None, "why": "within 5%: measure both"}


# --- the page ---------------------------------------------------------------------------------------------------------

def setup(profile:dict[str, Any], facts:dict[str, Any] | None, provider:str, gpus:list[dict[str, Any]], *, target,
          seqs:int = 1, context:int = CONTEXT, text_only:bool = True, host:dict[str, Any] | None = None,
          moe_estimate:dict[str, Any] | None = None, offload:bool = False, draft:str | None = None) -> dict[str, Any]:
  """The recommended setup for this model on these GPUs with this engine: one row per layout, the settings, the
  recommended layout and the command lines. facts: machine facts (read bandwidth, join, cached engine overhead).
  When no layout holds the weights (or offload=True), `offload` is the experts-in-RAM setup with its limit band, or the
  reason there is none (workflow/offload.py); host: the host's RAM and PCIe (target/host.py) and moe_estimate (experts,
  bits_per_weight) stand in for a probe or a header. draft: a draft model's file (an MTP head kept apart): its bytes and its KV
  at the stated context count against the fit, as an MTP head the plan turns on does (draft_cost)."""
  from boltbeam.roofline import limit
  tb = tensor_bytes(profile["source"])
  resident = weights_resident(tb, text_only=text_only)
  ws = [g.get("working_set_bytes") for g in gpus]  # Metal's measured per-process limit replaces the vendor's assumed share
  mem = min(ws) if ws and all(ws) else min((g.get("memory_bytes") or 0 for g in gpus), default=0)
  eng = ENGINES.get(provider) or Engine({"f16": 2}, 1.0, None, None, None, lambda: None, None, None, None)
  row = vendors.for_backend(target.backend)
  share_of_mem = eng.utilization * (1.0 if ws and all(ws) else row.usable_memory)
  usable = share_of_mem * mem - eng.reserve
  extra = {k: eng.scratch(profile, k, target) for k in eng.kv} if eng.scratch else {}
  full, reduced = next(iter(eng.kv)), list(eng.kv)[-1]
  states = overhead_of(facts, provider, profile.get("source"))
  measured, missing = guaranteed(states)
  overhead, warm = measured["bytes"] if measured else None, cached(states)[1]
  by_key = _by_path(((facts or {}).get(OVERHEAD) or {}).get(provider))
  try:
    base = limit.ceiling(profile, target, read=limit.read_bandwidth(facts))["decode"]["bytes_moved"]
  except (Refused, SystemExit):
    base = None
  n = len(gpus)
  rows = lay.layouts(n, provider, target.backend)
  if n > 1 and "one" in {r["id"] for r in rows if r["available"]}:
    rows.insert(1, {"id": "replica", "label": "one replica per GPU", "available": True, "reason": None, "gpus": n,
                    "speed": "one GPU's speed per user, one copy per GPU"})
  cap = ((profile.get("metadata") or {}).get("context") or {}).get("length")
  full_ctx = None
  for r in rows:
    layout = "one" if r["id"] == "replica" else r["id"]
    g = 1 if layout == "one" else r["gpus"]
    w, share, how = _split(layout, g, tb, resident)
    r["weights_per_gpu"] = fig(w, how, [f"{resident / 2**30:.2f} GiB resident", f"{tb['one_dim'] / 2**20:.1f} MiB one-dimensional"] if layout == "row" else [f"{resident / 2**30:.2f} GiB resident"])
    r["fits"] = fig(w <= usable, "resident weights per GPU <= usable VRAM", [f"usable {usable / 2**30:.2f} GiB = {share_of_mem:g} x {mem / 2**30:.2f} GiB" + (" (the vendor's usable share of memory, assumed)" if share_of_mem != eng.utilization else
   " (Metal's recommendedMaxWorkingSetSize, measured)" if ws and all(ws) else "") +
   (f", less the engine's {eng.reserve / 2**20:.0f} MiB reserve" if eng.reserve else "")])
    if not r["available"] or w > usable:
      continue
    own = by_key.get(layout_key(profile.get("source"), layout, g)) if layout != "one" else None  # else one GPU's: more per GPU
    ro, rwarm = cached(own) if own else (measured, warm)
    r["max_context"] = context_figures(profile, weights=w, share=share, usable=usable, overhead=ro and ro["bytes"], seqs=seqs, kvs=eng.kv, extra=extra)
    if rwarm:
      r["max_context_warm"] = context_figures(profile, weights=w, share=share, usable=usable, overhead=rwarm["bytes"], seqs=seqs, kvs=eng.kv, extra=extra)
    r["_w"], r["_share"], r["_o"] = w, share, ro and ro["bytes"]
    if r["id"] == "one":
      full_ctx = r["max_context"][full]["value"]
  kv = reduced if (full_ctx is not None and cap and full_ctx < cap) else full
  for r in rows:
    if "_w" in r:
      r["max_seqs"] = max_seqs(profile, weights=r.pop("_w"), share=r.pop("_share"), usable=usable, overhead=r.pop("_o"),
                               context=context, elem=eng.kv[kv], extra=extra.get(kv, 0.0))
      if r["id"] == "replica":
        r["total_seqs"] = fig(r["max_seqs"]["value"] * n, "max sequences x GPUs", [f"{n} GPUs"], r["max_seqs"]["kind"])
  if base:
    kvb = (pw.kv_bytes_per_token(profile, element_bytes=eng.kv[kv]) or 0.0) * context
    scored = lay.predict([r for r in rows if r["id"] != "replica"], provider, bytes_per_token=base + kvb, facts=facts,
                         hidden_size=profile.get("hidden_size"), layers=profile.get("layer_count"))
    by = {r["id"]: r for r in scored}
    rows = [{**r, **{k: by[r["id"]][k] for k in ("tok_s", "speed", "limit_reason", "tok_s_at_most")}} if r["id"] in by else
            {**r, "tok_s": by["one"].get("tok_s"), "speed": by["one"]["speed"]} for r in rows]
    for r in rows:
      if r.get("tok_s"):
        r["batch1_limit"] = fig(r["tok_s"], "(weight bytes + KV at the context) / measured read bandwidth", [f"{(base + kvb) / 1e9:.2f} GB per token", f"context {context}"])
        chip = (facts or {}).get("gpus", [{}])[0].get("target_id")
        eff = EFFICIENCY.get((provider, chip, r["id"])) or (EFFICIENCY.get((provider, chip)) if r["id"] in ("one", "replica") else None)
        r["batch1_expected"] = fig(r["tok_s"] * eff[0], f"limit x {eff[0]:.0%} engine efficiency", [eff[1]], "measured") if eff else \
          fig(None, "efficiency not measured here for this engine, chip and layout", [])
        if eff and tb["mtp"]:
          r["mtp"] = mtp_speed(profile, tb, provider, chip, r, eff[0], (facts or {}).get("gpus", [{}])[0].get("read_gbs"),
                               measured=mtp_of(facts, provider, profile.get("source")), context=context)
  fits_one = bool(rows and rows[0]["fits"]["value"])
  chosen = pick(rows, seqs=seqs, fits_one=fits_one)
  layout = chosen["id"] if chosen["id"] in ("row", "layer") else "one"
  gcount = next((r["gpus"] for r in rows if r["id"] == layout), 1)
  one = next((r for r in rows if r["id"] == "one" and "max_context" in r), None)
  fit = int(one["max_context"][kv]["value"] or 0) if one else 0
  started = next((r for r in rows if r["id"] == layout), {})
  speeds = {k: f for k, f in (started.get("mtp") or {}).items() if f["value"]}
  best = best_k(speeds)
  past = row.vllm_mtp_context if provider == "vllm" and row.vllm_mtp_context and context > row.vllm_mtp_context[0] and not split_kv_verify(_vllm_python()) else None
  mtp = (best, speeds[best]) if best and seqs == 1 and not past and speeds[best]["value"] > started["batch1_expected"]["value"] * (1 + TIE) else None
  if started.get("mtp") and past:
    started["mtp_note"] = f"not recommended past {past[0]} tokens of context (stated {context}; --context {past[0]} turns it on): {past[1]}"
  held = None if mtp or not best else (f"MTP k={best}: {speeds[best]['value']:.0f} tok/s at short context, {speeds[best]['formula']}",  # measured, held back
    started.get("mtp_note") or ("batch > 1 (MTP pays at batch 1)" if seqs != 1 else "no draft count beats one-token decode by more than the tie"))
  dr = draft_cost(profile, tb, eng.kv[kv], context, mtp=bool(mtp), draft=draft)
  if dr:  # the draft is loaded beside the model and holds KV of its own: the model alone fitting is not the plan fitting
    per_gpu = started["weights_per_gpu"]["value"] + (dr["bytes"] + dr["kv_bytes"]) / max(started.get("gpus") or 1, 1)
    started["draft"] = dr
    if started["fits"]["value"] and per_gpu > usable:
      started["fits"] = fig(False, "resident weights + the draft's weights and KV per GPU <= usable VRAM",
                            [f"{started['weights_per_gpu']['value'] / 2**30:.2f} GiB weights", f"{dr['bytes'] / 2**30:.2f} GiB draft weights ({dr['what']})",
                             f"{dr['kv_bytes'] / 2**30:.2f} GiB draft KV at {context} tokens", f"usable {usable / 2**30:.2f} GiB"])
  opts = dict(machine_gpus=n, attention=row.vllm_attention, mtp=mtp, mtp_held=held)
  argv = eng.flags(profile, kv=kv, seqs=seqs, layout=layout, gpus=gcount, text_only=text_only, tb=tb, context=fit or context, **opts) if eng.flags else None
  moe = None
  if offload or not any(r["fits"]["value"] for r in rows):  # nothing holds the weights: say what offloading the experts gives
    kvb = ((pw.kv_bytes_per_token(profile, element_bytes=eng.kv[kv]) or 0.0) + extra.get(kv, 0.0)) * context
    moe = off.for_setup(profile, provider, usable=usable - (dr["bytes"] if dr else 0), kv_bytes=kvb + (dr["kv_bytes"] if dr else 0), vram_gbs=(limit.read_bandwidth(facts) or {}).get("gbs") or target.memory_bandwidth_gbs,
                        pci_bus_id=(gpus[0].get("pci_bus_id") if gpus else None), context=context, target=target, kv=kv, host=host, estimate=moe_estimate)
    if moe["status"] == "recommended" and argv is not None:
      argv = eng.flags(profile, kv=kv, seqs=seqs, layout="one", gpus=1, text_only=text_only, tb=tb, context=context, placement=moe["settings"], **opts)
      if (pre := moe.get("prefill") or {}).get("status") == "recommended":  # the setup for prompts: its own command
        pre_argv = eng.flags(profile, kv=kv, seqs=seqs, layout="one", gpus=1, text_only=text_only, tb=tb, context=context, placement=pre["settings"], **opts)
        pre["command"] = eng.command(shlex.quote(profile["source"]), [shlex.quote(a) for a in pre_argv])
  env = dict(row.vllm_env) if provider == "vllm" else {}
  key = (provider, (facts or {}).get("gpus", [{}])[0].get("target_id"), profile.get("hidden_size"), profile.get("layer_count"), profile.get("ffn_size"))
  return {"schema": "boltbeam.recommended_setup.v1", "model": profile.get("source"), "provider": provider, "gpu_count": n,
          "weights": {"resident": resident, "total": tb["total"], "vision": tb["vision"], "mtp": tb["mtp"], "source": tb["source"]},
          "overhead": measured, "overhead_states": states, "overhead_note": missing, "context_is_upper_bound": measured is None, "stated_context": context, "seqs": seqs,
          "layouts": rows, "recommended_layout": chosen, "flags_layout": {"id": layout, "gpus": gcount}, "kv_dtype": kv, "kv_formats": list(eng.kv), "engine_note": eng.note,
          "offload": moe, "draft": dr,
          "settings": (eng.settings(profile, tb, kv=kv, seqs=seqs, layout=layout, gpus=gcount, text_only=text_only, full_context=full_ctx, **opts) if eng.settings else [])
                      + (off.settings(moe) if moe and moe["status"] == "recommended" else []),
          "flags": argv, "env": env, "command": " ".join([*(f"{k}={v}" for k, v in env.items()), eng.command(shlex.quote(profile["source"]), [shlex.quote(a) for a in argv])]) if argv else None,
          "arkey_vllm_args": arkey_args(argv) if argv and provider == "vllm" else None, "usable_bytes": usable, "reserve_bytes": eng.reserve, "scratch_per_token": extra,
          "measured_only": MEASURED.get(key) or "not measured here (MTP, throughput, prefill share)",
          # Measure MTP: what it runs here (collectors/vllm_mtp.offer), and this checkpoint's measurement kept in the facts
          "mtp_measure": vllm_mtp.offer(provider, tb["mtp"], facts=bool(facts)), "mtp_measured": mtp_of(facts, provider, profile.get("source"))}


def draft_cost(profile:dict[str, Any], tb:dict[str, Any], element_bytes:float, context:int, *, mtp:bool, draft:str | None) -> dict[str, Any] | None:
  """What a draft costs the card: its weights and its KV at `context`. The MTP head the plan turns on (tb['mtp'], in the
  checkpoint, so not in the resident weights) or a draft file (`draft`, counted whole). The KV is the draft file's own attention
  (its header) else one attended layer of this model's per MTP layer: an MTP layer attends, so it holds KV like any layer.
  None: no draft. Refused: a draft whose KV cannot be sized (an unknown draft is not a free one)."""
  if not (mtp and tb["mtp"]) and not draft:
    return None
  per_layer = (pw.kv_bytes_per_token(profile, element_bytes=element_bytes) or 0.0) / max(pw.attention_layer_count(profile) or 1, 1)
  if not draft:
    return {"bytes": tb["mtp"], "kv_bytes": per_layer * max(tb["mtp_layers"], 1) * context, "what": "the MTP head, in the checkpoint",
            "kv_source": "one attended layer of this model per MTP layer"}
  db = tensor_bytes(draft)
  if db["source"] == "file sizes":
    raise Refused(f"the draft {draft} cannot be read as a model (no tensor table), so its bytes are the file's and its KV cannot be sized")
  try:
    from boltbeam.profile.loaders import profile_from_model
    own = pw.kv_bytes_per_token(profile_from_model(draft).to_json(), element_bytes=element_bytes)
  except (ValueError, OSError, KeyError, Refused):
    own = None
  per = own or per_layer * max(db["mtp_layers"], 1)
  if not per:
    raise Refused(f"the draft {draft} has no attention shape and neither has the model, so its KV cannot be sized")
  return {"bytes": db["total"] - db["host"], "kv_bytes": per * context, "what": f"the draft {pathlib.Path(draft).name}",
          "kv_source": "the draft's own header" if own else "one attended layer of this model per draft layer"}


def mtp_speed(profile:dict[str, Any], tb:dict[str, Any], provider:str, chip:str | None, r:dict[str, Any], eff:float,
              read_gbs:float | None, measured:dict[str, Any] | None = None,
              context:int | None = None) -> dict[int, dict[str, Any]] | None:
  """Batch-1 decode with MTP at each measured draft count k: accept length / (one-token step + fixed + k x draft read).
  A draft step reads the MTP layer and the whole lm_head (on the 5090 a 4-bit head gave +30% with MTP against +11%
  without, inferred from step times), split over the GPUs of a row split, at the layout's engine efficiency. The
  accept lengths and the fixed cost are measured (MTP_ACCEPT, MTP_FIXED); None where either is not. `measured`, this
  checkpoint's in-repo measurement on this machine (mtp_of), stands in on one GPU, which is what it ran on."""
  acc = MTP_ACCEPT.get((profile.get("hidden_size"), profile.get("layer_count"), profile.get("ffn_size")))
  fixed = MTP_FIXED.get((provider, chip))
  own = _measured_mtp(measured, context) if measured and r["id"] == "one" else {}
  if not (acc and fixed and read_gbs and r["batch1_expected"]["value"]):
    return own or None
  g = r["gpus"] if r["id"] == "row" else 1
  seen = MTP_MEASURED.get((provider, chip, profile.get("hidden_size"), profile.get("layer_count"), profile.get("ffn_size"), r["id"],
                           round(tb["head"] / 1e9, 1)))
  draft = (tb["mtp"] + tb["head"]) / g / (read_gbs * 1e9) / eff * 1e3
  step = 1000.0 / r["batch1_expected"]["value"]
  inputs = [f"one-token step {step:.2f} ms (batch-1 expected)", f"draft read {(tb['mtp'] + tb['head']) / 1e9:.2f} GB "
            f"(MTP layer + lm_head){f' / {g} GPUs' if g > 1 else ''} = {draft:.2f} ms", f"fixed {fixed[0]} ms: {fixed[1]}", acc[1]]
  out = {k: fig(a / (step + fixed[0] + k * draft) * 1e3, f"accept length {a} / (step + fixed + {k} x draft read)", inputs)
         for k, a in acc[0].items()}
  for k, v in (seen[0] if seen else {}).items():  # a measured figure stands in for the derived one, which it keeps
    out[k] = {**fig(v, "measured", [seen[1]], "measured"), "derived": out.get(k, {}).get("value")}
  for k, v in own.items():  # this checkpoint, this machine, in the repo: it stands in over both
    out[k] = {**v, "derived": out.get(k, {}).get("derived", out.get(k, {}).get("value"))}
  return out


def mtp_envelope(contexts:dict[str, float], context:int | None) -> str | None:
  """Where MTP was measured, said in words when the stated context lies outside it (beyond the deepest measured depth,
  or under MTP_BELOW of the shallowest): a safeguard's operating envelope is part of what it claims. None inside it or with
  no context stated. The one place this is computed; the figure carries it and the settings row and the setup read it."""
  if not contexts or not context:
    return None
  lo, hi = min(contexts.values()), max(contexts.values())
  if context > hi or context < lo * MTP_BELOW:
    return f"measured at {lo:,.0f} to {hi:,.0f} tokens of context; not measured at the stated {context:,}"
  return None


def _measured_mtp(m:dict[str, Any], context:int | None = None) -> dict[int, dict[str, Any]]:
  """An in-repo measurement's figures per k > 0 at the measured depth nearest the stated context (the short depth
  with none stated): the median over the prompts, each prompt's speed, their spread and the acceptance. Acceptance
  moves with depth (the 27B AWQ's k=3 tied k=2 at 75 tokens and beat it on 7 of 8 prompts at 2,120), so the depth is
  chosen, not averaged."""
  depths = {d: ks for d, ks in m["depths"].items() if ks}
  if not depths:
    return {}
  at = lambda d: next(iter(depths[d].values()))["context"]  # noqa: E731
  depth = min(depths, key=lambda d: abs(at(d) - context)) if context else ("short" if "short" in depths else next(iter(depths)))
  where = (f"{len(m['prompts']['ids'])} natural prompts (set v{m['prompts']['version']}, sha256 {m['prompts']['sha256'][:12]}), "
           f"{m['tokens']} tokens each at a {at(depth):.0f}-token context, vLLM {', '.join(m['vllm']) or 'unknown'}, {m['measured_at'][:10]}")
  contexts = {d: at(d) for d in depths}
  note = mtp_envelope(contexts, context)
  return {int(k): {**fig(v["median_tok_s"], "measured: median over the prompts", [where], "measured"),
                   "per_prompt": v["per_prompt"], "spread": v["spread"], "range": [v["min_tok_s"], v["max_tok_s"]],
                   "accept_length": v.get("accept_length"), "speedup": v["speedup"], "depth": depth, "context": at(depth), "contexts": contexts,
                   "envelope": f"MTP k={int(k)} {note}" if note else None,
                   "by_depth": {d: ks[k]["per_prompt"] for d, ks in depths.items() if k in ks and ks[k].get("per_prompt")}}
          for k, v in depths[depth].items() if int(k) > 0}


def predicted_context(rec:dict[str, Any], state:str | None = None, layout:str = "one") -> dict[str, Any]:
  """The max context per GPU of this layout at the recommended KV dtype, what Check compares with the engine's own
  figure: the guaranteed (first-start) one, or the warm-cache one when state is 'warm' and it is cached."""
  one = next(r for r in rec["layouts"] if r["id"] == layout)
  return (one.get("max_context_warm") if state == "warm" and "max_context_warm" in one else one["max_context"])[rec["kv_dtype"]]


def compare(rec:dict[str, Any], got:dict[str, Any], layout:str = "one") -> dict[str, Any]:
  """Predicted against the engine's served context (else its cache tokens), as a signed share of the measured."""
  p = predicted_context(rec, got.get("state"), layout)
  m = got.get("served_context") or got["cache_tokens"]
  return {"predicted": p["value"], "predicted_kind": p["kind"], "measured": m, "state": got.get("state"), "cache_tokens": got["cache_tokens"],
          "error": (p["value"] - m) / m, "available_kv_gib": got["available_kv_bytes"] / 2**30}


# --- Check: one real engine start with the recommended flags -------------------------------------------------------------

def stop_group(proc:subprocess.Popen, grace_s:float = 30.0) -> None:
  """Stops the server's whole process group. vLLM's per-GPU workers can outlive the server and ignore SIGTERM, holding
  their memory (2x R9700 TP=2, 2026-11-07), so whatever is left of the group after the grace gets SIGKILL."""
  for sig in (signal.SIGTERM, signal.SIGKILL):
    try:
      os.killpg(proc.pid, sig)
    except ProcessLookupError:
      return
    end = time.monotonic() + grace_s
    while time.monotonic() < end:
      try:
        proc.wait(0.5)
        os.killpg(proc.pid, 0)  # the server is gone; is anything of its group left?
      except subprocess.TimeoutExpired:
        continue
      except ProcessLookupError:
        return
      time.sleep(0.5)


def start_engine(cmd:list[str], ready:str, *, env:dict[str, str] | None = None, timeout_s:float = 900.0) -> str:
  """Starts the engine's server with this command and returns its log once it has printed the `ready` text (or it
  exits, or timeout_s passes); the server is then stopped. Nothing is served: the fit lines come before any request."""
  with tempfile.TemporaryFile("w+") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **RUN_ENV, **(env or {})},
                            start_new_session=True)
    end = time.monotonic() + timeout_s
    try:
      while time.monotonic() < end and proc.poll() is None:
        log.seek(0)
        if ready in log.read():
          break
        time.sleep(2.0)
    finally:
      stop_group(proc)
    log.seek(0)
    return log.read()


def check(rec:dict[str, Any], model:str, *, facts_file:pathlib.Path | None, facts:dict[str, Any] | None,
          run=start_engine, port:int = 18080) -> dict[str, Any]:
  """One real start with the recommended flags: the engine's own figures against the page's prediction (made before
  the start, from the overhead cached until now, else an upper bound), the overhead measured and cached in
  facts_file, and the prediction again with that overhead. The weights are one GPU's in the layout the flags start."""
  eng = ENGINES.get(rec["provider"])
  if eng is None or eng.parse is None or not rec.get("flags"):
    raise ValueError(f"{rec['provider']}: " + (eng.note if eng else "no Check for this engine"))
  binary = eng.binary()
  if not binary:
    raise ValueError(f"{rec['provider']} was not found: set its environment variable (BOLTBEAM_VLLM_PYTHON, BOLTBEAM_LLAMA_SERVER)")
  env = {**ec.venv_env(binary), **(rec.get("env") or {})} if rec["provider"] == "vllm" else None
  log = run(eng.argv(binary, model, rec["flags"], port), eng.ready, env=env)
  started = rec.get("flags_layout") or {"id": "one", "gpus": 1}
  row = next(r for r in rec["layouts"] if r["id"] == started["id"])
  try:
    got = overhead_from_log(log, memory_bytes=rec["usable_bytes"] + rec["reserve_bytes"], utilization=1.0, weights=row["weights_per_gpu"]["value"],
                            parse=eng.parse, scratch=rec["scratch_per_token"].get(rec["kv_dtype"], 0.0))
  except ValueError as exc:
    raise ValueError(f"{exc}; the engine said: {' | '.join(log.strip().splitlines()[-4:])[-600:]}") from exc
  source = f"{rec['provider']} start with {' '.join(rec['flags'])}, {_dt.date.today().isoformat()}, checkpoint {pathlib.Path(model).name}"
  key = layout_key(model, started["id"], started["gpus"])
  new = facts_with_overhead(facts or {}, rec["provider"], got, source, path=key)
  if facts_file is not None:
    facts_file.write_text(json.dumps(new, indent=2, sort_keys=True))
  return {"before": compare(rec, got, started["id"]), "overhead": new[OVERHEAD][rec["provider"]][key][got.get("state") or "any"],
          "layout": started["id"], "key": key, "cached_in": str(facts_file) if facts_file else None,
          "engine_said": [l.split("] ", 1)[-1][:400] for l in log.splitlines() if re.search(
            r"Actual usage|Desired GPU memory|Model loading took|Available KV|Auto-fit|n_ctx|KV cache", l)][:8]}
