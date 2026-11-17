"""The prefill chunk (tokens per forward pass: an engine's ubatch) derived from the model's shapes, the chip's measured
peaks and bandwidth, and the device's memory, and what one prefill turn costs by where its weight bytes come from.

A chunk reads every dense weight once and every expert its tokens reach once; its compute grows with its tokens. So per
token the bytes fall as 1/c and the compute stays: the chunk worth having is the smallest at or past bytes = compute
(the crossover, within the policy's tie), and it must fit the device beside everything else the run holds. Per chunk c:
  bytes     dense + reach(c) x experts, at the measured read bandwidth; reach is a measured row at c, else a measured
            routing profile's (tokens independent: derived), else the expert-count bound min(1, c x top_k / n) (estimate)
  compute   the weight matrices' operations per token at each format's measured matrix peak (path_of names the path);
            the output head runs for the last token only and attention is not counted, so the crossover is at most this
  memory    the compute buffers beyond a reference chunk: a measured row, else the engine's sizing formula (derived),
            labelled; the device budget is the device's memory less the non-expert rows (held, or min_memory's minimum),
            the measured margin and the buffers; what is left is the room for experts
The output label: a chunk changes rounding, so a value is quality-gated only with gate evidence for that chunk on that
model, "gate failed" with a failed gate, else untested; a chunk that failed its gate, and every larger one (the drift
grows with the chunk), is never chosen. Engine-blind: the classifier, an engine's facts (its buffer formula, whether it
copies cached experts, its copy calls) and its defaults come in from the caller and data rows; nothing here names one."""
from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Callable

from boltbeam import formats
from boltbeam.roofline.prefill_work import chip_best, experts_reached

MiB = 2**20
TAB = json.loads((pathlib.Path(__file__).resolve().parents[1] / "data" / "prefill_chunk.json").read_text())
LABELS = ("measured", "derived", "estimate", "unknown")  # strongest first: a figure is as strong as its weakest input


def weakest(*labels:str) -> str:
  return max(labels, key=lambda x: LABELS.index(x) if x in LABELS else len(LABELS) - 1)


def engine_row(engine:str | None, row:dict[str, Any] | None = None) -> dict[str, Any]:
  """The engine's staging row, with a measured row's own staging (its build's switches) over it."""
  return {**TAB["engines"].get(engine or "", {}), **((row or {}).get("staging") or {})}


def measured_row(model_id:str | None, target_id:str | None, engine:str | None, context:int | None, kv:str | None,
                 draft_id:str | None, features:set[str] | list[str] | None = None) -> dict[str, Any]:
  """The data file's measured row for this setup: model and draft by header id, target, engine, context, KV type, and a
  build that has the row's features (switches compiled in). The most specific row wins; features None (the build is not
  read here) takes the most specific row and says the build was not checked. {} when none."""
  key = (model_id, target_id, engine, context, kv, draft_id)
  hit = [r for r in TAB["rows"] if (r["model_id"], r["target_id"], r["engine"], r["context"], r["kv"], r["draft_id"]) == key
         and (features is None or set(r["features"]) <= set(features))]
  if not hit:
    return {}
  r = max(hit, key=lambda r: len(r["features"]))
  return {**r, "build_checked": features is not None}


def fig(value:Any, label:str, why:str) -> dict[str, Any]:
  return {"value": value, "label": label, "why": why}


def weights(meta:dict[str, Any], tensors:list, *, kind:Callable, nbytes:Callable, peaks:dict[str, float],
            path_of:Callable[[str], str | None] | None = None) -> dict[str, Any]:
  """A header table's ({meta, tensors: [name, dims, type id]}) weight facts, sorted by min_memory's classifier: dense
  bytes (read once per chunk), expert bytes per layer, experts and top_k, and the matrix operations one token does per
  path (rank-2 dense and the experts it selects; the head and the lookup tables apart). kind, nbytes: the classifier and
  the sizer (workflow/min_memory's). path_of: the engine's path for a format, else the chip's best path that holds it."""
  arch = meta.get("general.architecture")
  n, k = meta.get(f"{arch}.expert_count"), meta.get(f"{arch}.expert_used_count")
  dense, head, per_layer, ops, pathless, stacks = 0, 0, {}, {}, set(), 0
  for name, dims, typ in tensors:
    kd, b = kind(name, dims), nbytes(name, dims, typ)
    if kd == "experts":
      layer = int(name.split(".")[1])
      per_layer[layer] = per_layer.get(layer, 0) + b
      stacks += 1
    elif kd in ("dense", "unclassified"):
      dense += b
    elif kd == "head":
      head += b
    if len(dims) < 2 or kd in ("head", "lookup"):
      continue
    fmt = formats.header_type(typ)
    path = (path_of(fmt) if path_of else None) or (chip_best(fmt, peaks) or {}).get("path")
    if path is None or not peaks.get(path):
      pathless.add(fmt)
      continue
    ops[path] = ops.get(path, 0.0) + 2.0 * math.prod(dims) * (k / n if kd == "experts" and n else 1)
  return {"dense_bytes": dense, "head_bytes": head, "expert_bytes": sum(per_layer.values()), "per_layer": [per_layer[i] for i in sorted(per_layer)],
          "layer_ids": sorted(per_layer), "n_expert": n, "top_k": k, "ops_per_token": ops, "pathless": sorted(pathless),
          "stacks_per_layer": stacks / max(len(per_layer), 1)}


def compute_s_per_token(w:dict[str, Any], peaks:dict[str, float]) -> float:
  return sum(o / (peaks[p] * 1e12) for p, o in w["ops_per_token"].items())


def reach(c:int, w:dict[str, Any], *, measured:dict[int, float] | None = None, routing:list | None = None) -> dict[str, Any]:
  """The share of a layer's experts a chunk of c tokens reaches: measured (experts per layer at c), else a routing
  profile's (roofline/prefill_work.experts_reached: tokens independent), else the expert-count bound."""
  n, k = w["n_expert"], w["top_k"]
  if measured and c in measured:
    return fig(measured[c] / n, "measured", f"{measured[c]:g} of {n} experts per layer at {c} tokens")
  if routing:
    r = experts_reached(c, n, k, len(w["per_layer"]), routing)
    return fig(sum(r) / len(r) / n, "derived", "the measured routing profile, tokens taken as independent")
  above = min((v for m, v in (measured or {}).items() if m > c), default=n)  # a smaller chunk reaches no more than a larger one did
  return fig(min(1.0, c * k / n, above / n), "estimate", f"the expert-count bound min(1, c x {k} / {n}), capped by the next larger "
             "measured reach: no routing profile; test: a routing capture")


def buffer(c:int, ref:int, *, measured:dict[int, float] | None = None, sizing:Callable[[int], float] | None = None) -> dict[str, Any]:
  """The compute buffers' bytes beyond those at the reference chunk: a measured row (a load probe), else the engine's
  sizing formula (derived, not measured), else unknown (counted 0, said)."""
  if measured and c in measured and ref in measured:
    return fig(measured[c] - measured[ref], "measured", "load probes at each chunk")
  if sizing:
    return fig(sizing(c) - sizing(ref), "derived", "the engine's compute-buffer sizing formula, not measured; test: a load probe at the chunk")
  return fig(0.0, "unknown", "no measured row and no sizing formula; test: a load probe at the chunk")


def room(device:dict[str, Any], rows:dict[str, dict[str, Any]], margin:dict[str, Any]) -> dict[str, Any]:
  """Device bytes for experts at the reference chunk: device - every non-expert row - the margin. A row or the margin
  that is unknown counts 0 and leaves the room unverified (said, with the test)."""
  held = sum(r["value"] or 0.0 for r in rows.values())
  open_ = [k for k, r in rows.items() if r["label"] == "unknown"] + (["margin"] if margin["label"] == "unknown" else [])
  label = weakest("derived", device["label"], margin["label"], *(r["label"] for r in rows.values()))
  return {"value": device["value"] - held - (margin["value"] or 0.0), "label": label,
          "formula": "device - non-expert rows - margin", "inputs": {"device": device, "rows": rows, "margin": margin},
          "unknown": open_}


def curve(chunks:list[int], w:dict[str, Any], *, peaks:dict[str, float], bandwidth_gbs:float, ref:int, room_bytes:float,
          reach_measured:dict[int, float] | None = None, routing:list | None = None, buffer_measured:dict[int, float] | None = None,
          sizing:Callable[[int], float] | None = None) -> list[dict[str, Any]]:
  """One row per chunk: bytes and compute per chunk (ms) and per token (us), the buffers beyond the reference chunk, the
  room left for experts and whether the chunk fits at all (room >= 0)."""
  comp = compute_s_per_token(w, peaks)
  out = []
  for c in chunks:
    r, b = reach(c, w, measured=reach_measured, routing=routing), buffer(c, ref, measured=buffer_measured, sizing=sizing)
    nbytes = w["dense_bytes"] + r["value"] * w["expert_bytes"]
    bytes_ms, comp_ms = nbytes / (bandwidth_gbs * 1e9) * 1e3, comp * c * 1e3
    left = room_bytes - b["value"]
    out.append({"chunk": c, "reach": r, "bytes_per_chunk": nbytes, "bytes_ms": bytes_ms, "compute_ms": comp_ms,
                "us_per_token": max(bytes_ms, comp_ms) * 1e3 / c, "bound": "bytes" if bytes_ms > comp_ms else "compute",
                "buffer": b, "room_for_experts": left, "fits": left >= 0})
  return out


def choose(rows:list[dict[str, Any]], tie_pct:float, failed:int | None = None) -> dict[str, Any]:
  """The smallest chunk at or past the crossover (bytes <= compute x (1 + tie)) that fits and is below a chunk that failed
  its gate (failed); none left past it: the largest such chunk (bytes still bind); none fits: refused by the caller."""
  fit = [r for r in rows if r["fits"] and (failed is None or r["chunk"] < failed)]
  cross = next((r["chunk"] for r in rows if r["bytes_ms"] <= r["compute_ms"]), None)
  if not fit:
    return {"row": None, "crossover": cross, "why": "no chunk's compute buffers fit beside the non-expert rows, below any failed gate"}
  past = [r for r in fit if r["bytes_ms"] <= r["compute_ms"] * (1 + tie_pct / 100)]
  gate = f" (the gate failed at {failed}, so nothing from {failed} up)" if failed else ""
  if past:
    return {"row": past[0], "crossover": cross, "why": f"the smallest chunk at or past bytes = compute (within {tie_pct:g}%) that fits{gate}"}
  return {"row": fit[-1], "crossover": cross, "why": f"bytes bind at every chunk that fits: the largest that fits{gate}"}


def crossover_tokens(w:dict[str, Any], *, peaks:dict[str, float], bandwidth_gbs:float, share:float) -> float:
  """c* = bytes per chunk / compute per token at a reach share (a chunk past it is compute-bound)."""
  return (w["dense_bytes"] + share * w["expert_bytes"]) / (bandwidth_gbs * 1e9) / compute_s_per_token(w, peaks)


def output_label(chunk:int, gated:list[dict[str, Any]]) -> dict[str, str]:
  """quality-gated with a passed gate for this chunk, "gate failed" with a failed one (rows {value, result, evidence}
  already matched to the model), else untested."""
  hit = next((g for g in gated if int(g["value"]) == chunk), None)
  if hit:
    return {"output": "quality-gated" if hit["result"] == "passed" else "gate failed", "why": hit["evidence"]}
  return {"output": "untested", "why": "a chunk changes the order of prefill sums (rounding); no gate evidence for this chunk on this model"}


def failed_gate(gated:list[dict[str, Any]]) -> int | None:
  return min((int(g["value"]) for g in gated if g["result"] == "failed"), default=None)


def placement(w:dict[str, Any], room_bytes:float) -> dict[str, Any]:
  """Experts on the card for a room: the first layers' experts go to RAM (a count of expert layers) until the rest
  fit; what is left of the room is an expert cache for those layers."""
  per, left = w["per_layer"], room_bytes
  for n in range(len(per) + 1):
    on = sum(per[n:])
    if on <= left:
      return {"layers_in_ram": n, "on_card_bytes": on, "host_bytes": sum(per[:n]), "cache_bytes": max(left - on, 0.0) if n else 0.0}
  return {"layers_in_ram": len(per), "on_card_bytes": 0, "host_bytes": sum(per), "cache_bytes": max(left, 0.0)}


def staging(c:int, w:dict[str, Any], *, layers_in_ram:int, share:float, cache_hit:dict[str, Any], engine:dict[str, Any],
            vram_gbs:float, h2d_gbs:float, d2d_gbs:float | None = None, call_us:float | None = None) -> dict[str, Any]:
  """One chunk's expert bytes by where they come from, and their time: the card's layers are read in place (in the GPU
  rows); of the reached experts of the layers in RAM, cache_hit (share of bytes) are on the card in the engine's cache:
  read in place, or first copied device to device when engine["cache_copy"] (at d2d_gbs measured, else read + write at
  the bandwidth: derived); the rest are uploaded at h2d_gbs. engine["copy_calls"]: host calls per reached expert per
  tensor at call_us each (measured), serial on the host."""
  host = sum(w["per_layer"][:layers_in_ram]) * share
  hit = cache_hit["value"]
  cached, uploaded = host * hit, host * (1 - hit)
  copy = bool(engine.get("cache_copy"))
  d2d = d2d_gbs or vram_gbs / 2
  calls = engine.get("copy_calls", 0) * share * w["n_expert"] * layers_in_ram * w["stacks_per_layer"]
  ms = {"cache_copy": cached / (d2d * 1e9) * 1e3 if copy else 0.0, "upload": uploaded / (h2d_gbs * 1e9) * 1e3,
        "host_calls": calls * (call_us or 0.0) / 1e3}
  return {"bytes": {"host_layers": host, "cache": cached, "upload": uploaded}, "calls": calls, "ms": ms,
          "labels": {"cache_copy": "measured rate" if d2d_gbs else "derived (read + write at the bandwidth)", "cache_hit": cache_hit["label"],
                     "host_calls": "measured per-call cost" if call_us else "unknown (no per-call cost; counted 0)"}}


def turn(tokens:int, c:int, w:dict[str, Any], *, peaks:dict[str, float], bandwidth_gbs:float, share:float, stage:dict[str, Any] | None,
         serial:bool, outside:dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
  """One turn of `tokens` new tokens in chunks of c: the GPU rows at their limit (bytes or compute per chunk), the
  staging per chunk (staging()), serial with the GPU when the engine runs it so (else overlapped: the slower), and the
  turn's work outside the chunks when given (measured rows). The result is a lower bound unless every kernel ran at its
  limit: `missing` names what it leaves out."""
  chunks = math.ceil(tokens / c)
  gpu = max((w["dense_bytes"] + share * w["expert_bytes"]) / (bandwidth_gbs * 1e9), compute_s_per_token(w, peaks) * c) * 1e3
  st = sum((stage or {}).get("ms", {}).values())
  per_chunk = gpu + st if serial else max(gpu, st)
  out_ms = sum(r["value"] for r in (outside or {}).values())
  return {"chunks": chunks, "gpu_ms": gpu * chunks, "staging_ms": st * chunks, "outside_ms": out_ms, "ms": per_chunk * chunks + out_ms,
          "terms": {k: v * chunks for k, v in ((stage or {}).get("ms") or {}).items()}, "serial": serial,
          "missing": ["kernels' distance from their limits (the GPU rows here are at the limit)", "attention and the small kernels"]
                     + ([] if outside else ["the turn's work outside the chunks (checkpoints, draft catch-up, bookkeeping)"])}
