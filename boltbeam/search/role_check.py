"""BoltBeam's own check of the provider's claims: every candidate the provider measured, rebuilt from the source it
returned on the role's own weight bytes, checked against BoltBeam's reference and timed with BoltBeam's kernel timer
(search/provider_check.py does one kernel; Checker holds the bridge, flush and floor for a run). One winner rule:
the fastest own measurement that reproduced, the installed kernel retained inside the band.
"""
from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

from boltbeam.search import provider_check, role_numbers


def role_bytes(model:str, quant:str, rows:int, cols:int, tensor_name:str | None = None) -> tuple[str, bytes]:
  """The role's own weight bytes from the model file (the tensor of this quant and shape)."""
  from boltbeam.collectors import gemv_reference
  from boltbeam.profile.gguf import read_gguf_split, read_tensor
  _, tensors, _ = read_gguf_split(pathlib.Path(model))
  name, _ = gemv_reference.tensor_for(quant, rows, cols, tensors, tensor_name)
  size = rows * (cols // gemv_reference.BLOCK_ELEMS[quant]) * gemv_reference.BLOCK_BYTES[quant]
  return name, read_tensor(model, name, size)


class Checker:
  """BoltBeam's bridge on this machine, its flush and dispatch floor, opened once for a run's roles."""

  def __init__(self, backend:str, band:float | None, bridge=None):
    from boltbeam.collectors import kernel_timer
    self.backend, self.band = backend, band
    self.bridge = bridge or kernel_timer.bridge_for(backend)
    self.flusher = kernel_timer.Flusher(self.bridge, backend)
    self.floor_us = self.flusher.floor_us()
    self.libraries: dict[str, int] = {}

  def facts(self) -> dict[str, Any]:
    return provider_check.bridge_facts(self.bridge)

  def __call__(self, summary:Mapping[str, Any], result:Mapping[str, Any], *, model:str, quant:str, rows:int, cols:int,
               role:str, tensor_name:str | None = None) -> dict[str, Any]:
    """Every candidate the provider measured correct, rebuilt and timed by BoltBeam on the role's own bytes, each
    claim judged; BoltBeam's own winner is the fastest whose claim reproduced."""
    from boltbeam.collectors import boltbeam_gemv
    tensor, weights = role_bytes(model, quant, rows, cols, tensor_name)
    x = boltbeam_gemv.vector(f"{role}:{quant}", cols)
    out: dict[str, Any] = {"tensor": tensor, "floor_us": self.floor_us, "band": self.band, "candidates": []}
    for row in sorted((r for r in result.get("population", []) if r.get("state") == "MEASURED"), key=lambda r: r.get("rank") or 1 << 30):
      entry = role_numbers.entry(row)
      record = role_numbers.kernel_record(row)
      if record is None:
        mine = {"status": "not_rebuilt", "reason": "the provider returned no kernel source and launch for this candidate"}
      else:
        mine = provider_check.retime(self.bridge, self.flusher, self.floor_us, record, quant=quant, rows=rows, cols=cols,
                                     weights=weights, x=x, label=entry["plan"], libraries=self.libraries)
      out["candidates"].append({"candidate_hash": entry["candidate_hash"], "plan": entry["plan"], "provider_rank": row.get("rank"),
                                "boltbeam": mine, **provider_check.judge(entry.get("median_ns"), mine, self.band)})
    return out


  def fusion(self, result:Mapping[str, Any], *, model:str, proposal:Mapping[str, Any]) -> dict[str, Any]:
    """Every fused candidate the provider measured correct, rebuilt by BoltBeam on the fusion's own tensors, checked
    against the covered nodes computed by BoltBeam and timed; then each covered node's kernel alone (the unfused
    side, from the provider's compile of the fastest candidate), checked and timed the same way."""
    from boltbeam.search import fusion_space
    inst = proposal["instance"]
    rows, cols = (int(x) for x in inst["shape"])
    quant, order = str(proposal["quant"]), fusion_space.adjacency(proposal["covers"])["order"]
    weights = {node: role_bytes(model, quant, rows, cols, name)[1] for node, name in inst["tensors"].items()}
    out: dict[str, Any] = {"tensors": dict(inst["tensors"]), "floor_us": self.floor_us, "band": self.band, "candidates": [], "unfused": []}
    measured = sorted((r for r in result.get("population", []) if r.get("state") == "MEASURED"), key=lambda r: r.get("rank") or 1 << 30)
    for row in measured:
      entry = role_numbers.entry(row)
      record = role_numbers.kernel_record(row)
      mine = ({"status": "not_rebuilt", "reason": "the provider returned no kernel source and launch for this candidate"} if record is None
              else provider_check.retime_labeled(self.bridge, self.flusher, self.floor_us, record, nodes=order, quant=quant, rows=rows,
                                                 cols=cols, weights=weights, seed=proposal["name"], label=entry["plan"], libraries=self.libraries))
      out["candidates"].append({"candidate_hash": entry["candidate_hash"], "plan": entry["plan"], "provider_rank": row.get("rank"),
                                "boltbeam": mine, **provider_check.judge(entry.get("median_ns"), mine, self.band)})
    pick = pick_winner(out["candidates"])
    source = next((r for r in measured if pick and r["candidate_hash"] == pick["candidate_hash"]), measured[0] if measured else None)
    compiled = ((((source or {}).get("worker") or {}).get("provider") or {}).get("compile") or {}).get("result") or {}
    out["unfused_model_bound"] = (compiled.get("fusion") or {}).get("model_bound") is True
    for node in (compiled.get("fusion") or {}).get("unfused") or []:
      name = str(node.get("node"))
      if name not in order:
        out["unfused"].append({"node": name, "boltbeam": {"status": "not_rebuilt", "reason": "not a node this fusion covers"}})
        continue
      mine = provider_check.retime_labeled(self.bridge, self.flusher, self.floor_us, node.get("kernel") or {}, nodes=[name], quant=quant,
                                           rows=rows, cols=cols, weights=weights, seed=proposal["name"], label=f"{name} alone",
                                           libraries=self.libraries)
      out["unfused"].append({"node": name, "boltbeam": mine, **provider_check.judge(None, mine, self.band)})
    return out


  def flash(self, kernels:list[Mapping[str, Any]], *, shape:Mapping[str, int], maxc:int, context:int, label:str) -> dict[str, Any]:
    """Both launches of one flash candidate, rebuilt, run, checked and timed by BoltBeam (flash_space.retime)."""
    from boltbeam.search import flash_space
    return flash_space.retime(self.bridge, self.flusher, self.floor_us, kernels, shape=shape, maxc=maxc, context=context,
                              label=label, libraries=self.libraries)


def pick_winner(candidates, installed=None, band=0.0):
  """One winner rule: fastest own measurement must reproduce; retain installed inside the band."""
  timed = [c for c in candidates if (c.get("boltbeam") or {}).get("status") == "measured"]
  best = min(timed, key=lambda c: c["boltbeam"]["us_less_floor"], default=None)
  if (best and best.get("verdict") == provider_check.REPRODUCED and installed and
      installed.get("verdict") == provider_check.REPRODUCED and
      best["boltbeam"]["us_less_floor"] >= installed["boltbeam"]["us_less_floor"] * (1.0 - band)):
    return installed
  return best


def checked(summary:Mapping[str, Any], verify:Mapping[str, Any]) -> dict[str, Any]:
  """Provider and installed rows, plus the single BoltBeam winner rule."""
  rows = {c["candidate_hash"]: c for c in verify.get("candidates") or []}
  out = {k: rows.get((summary.get(k) or {}).get("candidate_hash")) for k in ("winner", "model_kernel")}
  out["boltbeam_winner"] = pick_winner(rows.values(), out["model_kernel"], verify.get("band", 0.0))
  out["reproduced"] = sum(c.get("verdict") == provider_check.REPRODUCED for c in rows.values())
  out["not_reproduced"] = len(rows) - out["reproduced"]
  return out
