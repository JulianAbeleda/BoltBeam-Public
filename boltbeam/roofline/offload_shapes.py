"""The byte counts an offload limit needs, read from a GGUF header (shapes and quant types, never a model's name):
the dense bytes every token reads, the expert bytes one token selects (top_k of N experts per layer), and the experts'
total. Without a header the caller states an estimate (`override`), and the result says `estimated`."""
from __future__ import annotations

import math
import pathlib
from typing import Any

from boltbeam.core.refused import Refused

HIT_RATES = (0.8, 0.9, 0.95, 0.99)  # the sweep read across when the engine reports no hit rate: stated, never fitted


def moe_split(source:str | pathlib.Path, override:dict[str, Any] | None = None) -> dict[str, Any]:
  """dense_bytes (every non-expert tensor once; the input embedding is one row), expert_bytes (what one token selects),
  expert_total, layers, experts and top_k from the GGUF tensor table. override {experts, bits_per_weight} re-sizes the
  experts for a file not on disk (the same layers, top_k and expert matrices; the result is estimated)."""
  from boltbeam.profile.decode_roles import decode_roofline_inventory_from_gguf
  p = pathlib.Path(source).expanduser()
  if p.is_dir() or p.suffix != ".gguf":
    raise Refused(f"{p.name}: expert sizes are read from a GGUF header, and this is not one; there is no offload limit without them")
  try:
    inv = decode_roofline_inventory_from_gguf(p)
  except (OSError, ValueError) as exc:
    raise Refused(f"{p.name}: the GGUF header could not be read ({exc})") from exc
  rows = inv["tensors"]
  exp = [r for r in rows if r.get("n_expert")]
  if not exp:
    raise Refused(f"{p.name} has no expert tensors (it is not a mixture of experts), so there is no expert cache to size; "
                  "a dense model past GPU memory is not covered")
  dense = sum(r["packed_weight_bytes"] for r in rows if not r.get("n_expert"))
  e_tok, e_total = sum(r["packed_weight_bytes"] for r in exp), sum(r["packed_weight_bytes"] * r["n_expert"] // r["experts_read"] for r in exp)
  n, k = max(r["n_expert"] for r in exp), max(r["experts_read"] for r in exp)
  out = {"estimated": False, "source": f"{p.name} GGUF tensor table", "dense_bytes": dense, "expert_bytes": e_tok, "expert_total": e_total,
         "experts": n, "top_k": k, "layers": len({r["tensor_name"].split(".")[1] for r in exp}),
         "expert_layer_ids": sorted({int(r["tensor_name"].split(".")[1]) for r in exp})}
  if override:
    experts, bpw = override.get("experts", n), override.get("bits_per_weight")
    if type(experts) is not int or experts < k or not isinstance(bpw, (int, float)) or not 0 < bpw <= 32:
      raise Refused(f"the estimate needs experts (a whole number of at least {k}) and bits_per_weight in (0, 32], got {override!r}")
    per_expert = sum(math.prod(r["shape"]) for r in exp)  # elements of one expert across all layers' stacks (shape is one expert's)
    out.update(estimated=True, experts=experts, expert_total=int(per_expert * experts * bpw / 8), expert_bytes=int(per_expert * k * bpw / 8),
               source=f"{out['source']}, experts re-sized to {experts} at {bpw:g} bits per weight (estimate)")
  return out


def n_cpu_moe(sh:dict[str, Any], expert_layers:int) -> int:
  """llama.cpp's --n-cpu-moe N for the first `expert_layers` expert layers in RAM: N counts the model's layers (0..N-1),
  dense-first layers included, so it is the layer after the last of them, not the count."""
  ids = sh.get("expert_layer_ids") or list(range(sh["layers"]))
  return ids[min(expert_layers, len(ids)) - 1] + 1 if expert_layers > 0 else 0


def expert_layers_under(sh:dict[str, Any], n:int) -> int:
  """The expert layers --n-cpu-moe N keeps in RAM: those among the model's layers 0..N-1."""
  ids = sh.get("expert_layer_ids") or list(range(sh["layers"]))
  return sum(1 for i in ids if i < n)


def hit_rates(cache_bytes:float, expert_total:float, measured:float | None = None) -> list[float]:
  """The sweep, from the floor up: the cache's share of the experts (what a uniformly random routing would hit), or
  `measured`, the share a cache of the hottest experts serves under a measured routing profile (routing_profile.hit_rate)."""
  floor = min(cache_bytes / expert_total if measured is None else measured, 1.0)
  return [round(floor, 4)] + [h for h in HIT_RATES if h > floor]


def span(rows:list[dict[str, Any]]) -> dict[str, float]:
  """The overlap limit's lowest and highest tokens a second over a band (it is not monotone in the hit rate)."""
  v = [r["overlap"]["tok_s"] for r in rows]
  return {"low": min(v), "high": max(v)}
