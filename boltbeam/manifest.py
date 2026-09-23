"""Candidate manifest loader (audit-brain-build-scope BB4).

Candidates are DATA (boltbeam/data/candidates.json), not evaluator branches. Every consumer — evaluator,
reachability audit, policy guard — loads them through here so adding a candidate is a data edit.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from boltbeam.vocab import RowKind, Workload, SCHEMA_CANDIDATE_MANIFEST, architecture_compatible

_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent / "data" / "candidates.json"


@dataclass(frozen=True)
class Candidate:
  candidate_id: str
  workload: str
  quant: tuple[str, ...] = ()
  roles: tuple[str, ...] = ()
  required_evidence_kinds: tuple[str, ...] = ()
  rollback: dict[str, str] = field(default_factory=dict)
  refuted_axis_tags: tuple[str, ...] = ()
  thresholds: dict[str, Any] = field(default_factory=dict)
  description: str = ""
  # audit A7 templating: a candidate is a route-family template keyed on (route_family, quant, roles,
  # architectures), not a model-era hardcode. `architectures` empty = applies to any architecture class;
  # `origin` records whether it is a generic template, a shipped decision (also in the ledger), or a
  # historical refutation kept for reproducibility.
  route_family: str = ""
  architectures: tuple[str, ...] = ()
  origin: str = "template"
  status: str = ""
  default_on: bool = False
  tier: str = ""
  provenance: str = ""
  default_shape: dict[str, Any] = field(default_factory=dict)
  # Structural per-tensor eligibility for route families selected per weight tensor (not once per attention shape).
  # Keys: cols_div / cols_div_mod  -> (cols // cols_div) % cols_div_mod == 0 ; rows_mod -> rows % rows_mod == 0.
  # A candidate with a shape_rule is emitted per matching profile tensor role (G3-style), keyed on real GEMV dims.
  shape_rule: dict[str, Any] = field(default_factory=dict)
  route_params: dict[str, str] = field(default_factory=dict)
  evidence_refs: tuple[str, ...] = ()
  wd_results: dict[str, Any] = field(default_factory=dict)
  # Capability tokens this candidate/route-class needs but which are absent, so it cannot be reached. Each token
  # is either a fully-blocked emitter knob (a lowering gap -> EMITTER_BLOCKED, walled not refuted) or a target
  # capability recorded false (a missing coordination/hardware primitive -> PRIMITIVE_MISSING). Reachability
  # consumes these to classify the candidate and name the exact blocker, turning a prose closeout into a queryable
  # edge. Empty = not capability-walled. A token that resolves to no wall (capability now present) falls through
  # to REACHABLE — that is the reopen signal.
  blocked_on: tuple[str, ...] = ()
  # Precise, measured, mechanistic reason a lever does NOT translate into a speedup — the answer to "why doesn't
  # this work". Distinct from blocked_on (a capability wall): this records WHY, so the search never re-tries a
  # lever for a reason already understood (e.g. "MAC-Amdahl-tiny", "int->f16 doubles converts"). Free text; empty
  # if the lever translates or has not been diagnosed.
  nontranslation_reason: str = ""

  def __post_init__(self):
    if self.workload not in {w.value for w in Workload}:
      raise ValueError(f"candidate {self.candidate_id!r} has unknown workload {self.workload!r}")
    for k in self.required_evidence_kinds:
      if k not in {r.value for r in RowKind}:
        raise ValueError(f"candidate {self.candidate_id!r} requires unknown evidence kind {k!r}")
    for tok in self.blocked_on:
      if not isinstance(tok, str) or not tok:
        raise ValueError(f"candidate {self.candidate_id!r} has an empty/invalid blocked_on token {tok!r}")

  def applies_to(self, *, workload:str, quant:str | None = None, role:str | None = None,
                 architecture:str | None = None) -> bool:
    if self.workload != workload: return False
    if quant is not None and self.quant and quant not in self.quant: return False
    if role is not None and self.roles and role not in self.roles: return False
    if not architecture_compatible(self.architectures, architecture): return False
    return True


def _from_row(d:dict[str, Any]) -> Candidate:
  return Candidate(candidate_id=d["candidate_id"], workload=d["workload"], quant=tuple(d.get("quant", ())),
                   roles=tuple(d.get("roles", ())), required_evidence_kinds=tuple(d.get("required_evidence_kinds", ())),
                   rollback=dict(d.get("rollback", {})), refuted_axis_tags=tuple(d.get("refuted_axis_tags", ())),
                   thresholds=dict(d.get("thresholds", {})), description=d.get("description", ""),
                   route_family=d.get("route_family", ""), architectures=tuple(d.get("architectures", ())),
                   origin=d.get("origin", "template"), status=d.get("status", ""),
                   default_on=bool(d.get("default_on", False)), tier=d.get("tier", ""),
                   provenance=d.get("provenance", ""), default_shape=dict(d.get("default_shape", {})),
                   shape_rule=dict(d.get("shape_rule", {})), route_params=dict(d.get("route_params", {})),
                   evidence_refs=tuple(d.get("evidence_refs", ())), wd_results=dict(d.get("wd_results", {})),
                   blocked_on=tuple(d.get("blocked_on", ())),
                   nontranslation_reason=d.get("nontranslation_reason", ""))


def load_candidates(path:str | pathlib.Path | None = None) -> list[Candidate]:
  p = pathlib.Path(path) if path else _DEFAULT_PATH
  data = json.loads(p.read_text())
  if data.get("schema") != SCHEMA_CANDIDATE_MANIFEST:
    raise ValueError(f"{p} is not a {SCHEMA_CANDIDATE_MANIFEST} manifest")
  cands = [_from_row(r) for r in data["candidates"]]
  ids = [c.candidate_id for c in cands]
  if len(ids) != len(set(ids)): raise ValueError("duplicate candidate_id in manifest")
  return cands


def candidate_by_id(cid:str, path:str | pathlib.Path | None = None) -> Candidate | None:
  return next((c for c in load_candidates(path) if c.candidate_id == cid), None)
