"""Flash decode in the kernel search: one fusion candidate over a layer's attention nodes.

A flash decode candidate replaces the attention of a decode layer (scores, softmax and the weighted sum of values)
with two launches: the split tile and the combine. The attention shape (query heads, KV heads, head dim) is the
model's own, read from the GGUF; nothing here holds a model shape, a backend or a chip.

  BubbleBeam   proposes the tile dimensions from the chip's facts (search/flash_decode_candidate.py is the schema):
               the lane width from the subgroup size, the stage widths, the query groups the head ratio allows, both
               reduce structures, and the split count the model's decode installs
  FutureSight  applies the schema's legality rules (build_flash_legality) and orders the survivors
               (build_flash_static_priority)
  provider     compiles, checks and times each survivor (search_provider --live-flash); its numbers order only
  BoltBeam     rebuilds both launches from the source the provider returned, runs the tile then the combine on its
               own q and KV cache, checks the output against attention it computes itself, and times both launches
               with the kernel timer, less the floor

The verdict needs two bars: BoltBeam's fastest candidate must beat the geometry the model's decode installs for this
shape, timed the same way (like for like), and the in-model attention time per token (the tie-out's attention row).
"""
from __future__ import annotations

import math, struct
from collections.abc import Mapping
from typing import Any

from boltbeam.search.flash_decode_candidate import FLASH_DECODE_CANDIDATE_SCHEMA_VERSION, FlashDecodeCandidate

GENERATOR = "boltbeam.flash_space"
STAGE_WIDTHS = (1, 2, 4)  # the tile's coalesce widths tried (the emitter admits 1, 2, 4 and 8)
REDUCES = ("staged", "inline")
ATTENTION = "attention"  # the tie-out's kind row for attention kernels (workflow/tie_out.py KIND_WORDS)
COVERS = ("attention_score", "attention_softmax", "attention_pv")
LIMIT_NOTE = "the attention shape is the model's; the kernel's own validate admits it (flash_decode_route_for)"


def attention_shape(profile:Mapping[str, Any]) -> dict[str, int] | None:
  """Query heads, KV heads and head dim from the model profile (read from the GGUF's attention keys), or None."""
  att = (profile.get("metadata") or {}).get("attention") or {}
  hq, hkv, hd = att.get("head_count"), att.get("head_count_kv"), att.get("head_dim")
  if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (hq, hkv, hd)): return None
  return {"Hq": hq, "Hkv": hkv, "Hd": hd}


def max_context(context:int) -> int:
  """The cache length the candidates are built for: the next power of two at or above four times the context."""
  return 1 << max(0, (4 * context - 1).bit_length())


def _payload(tile:dict[str, Any], target:Mapping[str, Any]) -> dict[str, Any]:
  return {"schema_version": FLASH_DECODE_CANDIDATE_SCHEMA_VERSION,
          "descriptor": {"tile": tile, "combine": {"stride": None, "output_fp16": False, "lane_width": None}},
          "target": dict(target),
          "provenance": {"generator_id": GENERATOR, "generator_revision": "1", "schema_revision": FLASH_DECODE_CANDIDATE_SCHEMA_VERSION}}


def propose(shape:Mapping[str, int], context:int, target:Mapping[str, Any], installed:Mapping[str, Any]) -> dict[str, Any]:
  """BubbleBeam: the flash candidates for this shape on this chip, the installed one first. Returns {"candidates",
  "refused", "installed_hash"}: a dimension combination the schema itself refuses is kept with its reason."""
  hq, hkv, hd = shape["Hq"], shape["Hkv"], shape["Hd"]
  subgroup = int(target["subgroup_size"])
  lanes = sorted({w for w in (subgroup, subgroup // 2) if w > 0 and w & (w - 1) == 0}, reverse=True)
  groups = [None] + [g for g in range(2, hq // hkv + 1) if (hq // hkv) % g == 0]
  base = {"Hq": hq, "Hd": hd, "Hkv": hkv, "MAXC": max_context(context), "split_count": int(installed["split_count"]),
          "staging": installed.get("staging") or "KV_BOTH", "quant": False, "rope": False, "token_block": 16,
          "lane_width": lanes[0] if lanes else subgroup, "score_group_width": None, "warps": None,
          "query_group_size": installed.get("query_group_size"), "stage_width": int(installed.get("stage_width") or 1),
          "reduce_structure": "staged", "dot_pair_width": 2}
  rows = [dict(base)]
  for lane in lanes:
    for group in groups:
      for stage in STAGE_WIDTHS:
        for reduce in REDUCES:
          rows.append(dict(base, lane_width=lane, query_group_size=group, stage_width=stage, reduce_structure=reduce))
  out, refused, seen = [], [], set()
  for tile in rows:
    try:
      cand = FlashDecodeCandidate.from_dict(_payload(tile, target))
    except ValueError as exc:
      refused.append({"tile": tile, "reason": f"schema: {exc}"}); continue
    if cand.candidate_hash in seen: continue
    seen.add(cand.candidate_hash)
    out.append(cand)
  return {"candidates": out, "refused": refused, "installed_hash": out[0].candidate_hash if out else None}


def assess(candidates:list[FlashDecodeCandidate], facts:Mapping[str, Any]) -> dict[str, Any]:
  """FutureSight: the schema's legality rules on every candidate, then the static order of the survivors."""
  from boltbeam.search.futuresight import build_flash_legality, build_flash_static_priority, candidate_report
  docs = [c.to_dict() | {"candidate_hash": c.candidate_hash} for c in candidates]
  return candidate_report(docs, (build_flash_legality({}, dict(facts)),), build_flash_static_priority(dict(facts)))


# --- BoltBeam's own run of both launches ---------------------------------------------------------------------

def _vector(seed:str, n:int) -> list[float]:
  from boltbeam.collectors import boltbeam_gemv
  return [v * 0.5 for v in boltbeam_gemv.vector(seed, n)]


def reference(q:list[float], kv:list[float], shape:Mapping[str, int], maxc:int, context:int, head:int) -> list[float]:
  """Attention for one query head over the first `context` positions, computed by BoltBeam: softmax(q.k / sqrt(Hd)).v.
  The cache is (2, 1, Hkv, MAXC, Hd), K then V, the layout the fork's decode writes."""
  hq, hkv, hd = shape["Hq"], shape["Hkv"], shape["Hd"]
  g = head // (hq // hkv)
  qh = q[head * hd:(head + 1) * hd]
  def row(which:int, t:int) -> list[float]:
    base = ((which * hkv + g) * maxc + t) * hd
    return kv[base:base + hd]
  scores = [sum(a * b for a, b in zip(qh, row(0, t))) / math.sqrt(hd) for t in range(context)]
  top = max(scores)
  w = [math.exp(s - top) for s in scores]
  total = sum(w)
  out = [0.0] * hd
  for t in range(context):
    v = row(1, t)
    p = w[t] / total
    for i in range(hd): out[i] += p * v[i]
  return out


def retime(bridge, flusher, floor_us:float | None, kernels:list[Mapping[str, Any]], *, shape:Mapping[str, int], maxc:int,
           context:int, label:str, libraries:dict[str, int] | None = None) -> dict[str, Any]:
  """Both launches the provider described, rebuilt and run by BoltBeam: the tile on BoltBeam's q and KV cache, the
  combine on the tile's own output; the combine's output checked against `reference` on sampled heads; each launch
  timed by the kernel timer. The time is the sum of the two, each less the floor."""
  from boltbeam.collectors import kernel_timer, metal_native as native
  from boltbeam.collectors.kernel_timer import Check, KernelSpec
  from boltbeam.search.provider_check import half_round
  hq, hkv, hd = shape["Hq"], shape["Hkv"], shape["Hd"]
  try:
    by_name = {str(k.get("name")): k for k in kernels}
    if set(by_name) != {"tile", "combine"}: raise ValueError(f"the provider described launches {sorted(by_name)}, not a tile and a combine")
    q = half_round(_vector(f"{label}:q", hq * hd))
    kv = half_round(_vector(f"{label}:kv", 2 * hkv * maxc * hd))
    data = {"value:q": struct.pack(f"<{len(q)}e", *q), "value:kv_cache": struct.pack(f"<{len(kv)}e", *kv)}
    results = {}
    for name in ("tile", "combine"):
      rec = by_name[name]
      args, out_bytes, out_fmt = [], None, "f"
      for buf in rec.get("buffers") or []:
        role, nbytes, dtype = str(buf.get("role")), int(buf.get("nbytes") or 0), str(buf.get("dtype"))
        if (name == "tile" and role == "value:partial") or (name == "combine" and role == "out"):
          out_fmt = {"float": "f", "half": "e"}.get(dtype)
          if out_fmt is None: raise ValueError(f"the {name} writes {dtype}, not float or half")
          if name == "combine" and nbytes != hq * hd * struct.calcsize(out_fmt):
            raise ValueError(f"the combine writes {nbytes} bytes, not {hq}x{hd} values")
          args.append(nbytes); out_bytes = nbytes
        elif role in data:
          if nbytes != len(data[role]): raise ValueError(f"the {name}'s {role} is {nbytes} bytes; BoltBeam's is {len(data[role])}")
          args.append(data[role])
        else:
          raise ValueError(f"the {name} lists a buffer BoltBeam does not expect: {role!r}")
      if out_bytes is None: raise ValueError(f"the {name} writes no output")
      check = None
      if name == "combine":
        heads = sorted({min(hq - 1, (hq - 1) * i // 3) for i in range(4)})
        refs = {h: reference(q, kv, shape, maxc, context, h) for h in heads}
        check = Check(indices=[h * hd + i for h in heads for i in range(hd)], reference=lambda j: refs[j // hd][j % hd],
                      rel_tol=native.TOLERANCE * 2, words="BoltBeam computes the attention itself: softmax(q.k / sqrt(Hd)).v",
                      out_format=out_fmt)
      block = (list(rec.get("local_size") or []) + [1, 1, 1])[:3]
      grid = (list(rec.get("global_size") or []) + [1, 1, 1])[:3]
      spec = KernelSpec(label=f"{label} {name}", adapter="provider kernel, rebuilt by BoltBeam", source=str(rec["source"]),
                        kernel=str(rec["function"]), args=args, grid=tuple(int(v) for v in grid), block=tuple(int(v) for v in block),
                        bytes_read=sum(len(a) for a in args if isinstance(a, (bytes, bytearray))), check=check,
                        record={"function": rec["function"]}, keep_output=name == "tile")
      got = kernel_timer.time_spec(bridge, spec, flusher, libraries=libraries, floor_us=floor_us)
      if name == "tile": data["value:partial"] = got.get("output") or b""
      results[name] = got
  except Exception as exc:  # the provider's description or source: BoltBeam could not rebuild it
    return {"status": "not_rebuilt", "reason": f"{type(exc).__name__}: {exc}"[:400]}
  combine = results["combine"]
  if not combine["samples"]:
    return {"status": "incorrect", "correctness": combine["correctness"], "reason": "BoltBeam's attention disagrees with the kernels' output"}
  less = [kernel_timer.less_floor_us(results[n]["median_us"], floor_us) for n in ("tile", "combine")]
  us = sum(results[n]["median_us"] for n in ("tile", "combine"))
  return {"status": "measured", "us": us, "us_less_floor": sum(less) if None not in less else None,
          "launches": {n: {"us": results[n]["median_us"], "us_less_floor": l, "spread_pct": results[n]["spread_pct"]}
                       for n, l in zip(("tile", "combine"), less)},
          "spread_pct": max(results[n]["spread_pct"] for n in ("tile", "combine")), "correctness": combine["correctness"],
          "timing": {**combine["timing"], "dispatch_floor_us": floor_us}, "timed_by": f"BoltBeam's kernel timer: {label}, tile then combine"}


__all__ = ["ATTENTION", "COVERS", "GENERATOR", "LIMIT_NOTE", "assess", "attention_shape", "max_context", "propose", "reference", "retime"]
