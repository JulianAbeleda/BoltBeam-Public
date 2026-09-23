"""Typed facts that preserve how a value is known."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from boltbeam.artifacts.base import EvidenceSource, sha256_json
from boltbeam.vocab import SCHEMA_FACT


class TruthStatus(str, Enum):
  MEASURED = "measured"
  DERIVED = "derived"
  MODELED = "modeled"
  ASSUMED = "assumed"
  IMPORTED = "imported"
  UNKNOWN = "unknown"


def _validate_json(value:Any, path:str = "value") -> None:
  if value is None or isinstance(value, (str, bool, int)): return
  if isinstance(value, float):
    if not math.isfinite(value): raise ValueError(f"{path} contains a non-finite number")
    return
  if isinstance(value, (list, tuple)):
    for i, item in enumerate(value): _validate_json(item, f"{path}[{i}]")
    return
  if isinstance(value, Mapping):
    for key, item in value.items():
      if not isinstance(key, str): raise ValueError(f"{path} contains a non-string key")
      _validate_json(item, f"{path}.{key}")
    return
  raise ValueError(f"{path} is not JSON-compatible: {type(value).__name__}")


@dataclass(frozen=True)
class Fact:
  name: str
  value: Any
  status: str
  unit: str | None = None
  sources: tuple[EvidenceSource, ...] = ()
  derivation: str | None = None
  input_fact_ids: tuple[str, ...] = ()
  uncertainty: Mapping[str, Any] = field(default_factory=dict)
  observed_at: str | None = None

  def __post_init__(self):
    if not self.name or "." not in self.name: raise ValueError("fact name must be namespaced")
    if self.status not in {s.value for s in TruthStatus}: raise ValueError(f"unknown truth status {self.status!r}")
    _validate_json(self.value)
    _validate_json(self.uncertainty, "uncertainty")
    if self.status == TruthStatus.UNKNOWN.value:
      if self.value is not None: raise ValueError("unknown facts require value=None")
    elif self.value is None:
      raise ValueError(f"{self.status} facts require a value")
    if self.status == TruthStatus.MEASURED.value:
      if not self.sources: raise ValueError("measured facts require a source")
      if self.derivation or self.input_fact_ids: raise ValueError("measured facts forbid derivation and input facts")
    if self.status in {TruthStatus.DERIVED.value, TruthStatus.MODELED.value}:
      if not self.derivation: raise ValueError(f"{self.status} facts require a derivation")
      if not (self.sources or self.input_fact_ids): raise ValueError(f"{self.status} facts require sources or input facts")

  @property
  def fact_id(self) -> str:
    # Paths and observation times describe storage/history, not semantic provenance.
    sources = [{"producer": s.producer, "tool": s.tool, "fingerprint": s.fingerprint} for s in self.sources]
    return sha256_json({"name": self.name, "value": self.value, "status": self.status, "unit": self.unit,
                        "sources": sources, "derivation": self.derivation,
                        "input_fact_ids": list(self.input_fact_ids), "uncertainty": dict(self.uncertainty)})

  def to_json(self) -> dict[str, Any]:
    out = {"schema": SCHEMA_FACT, "fact_id": self.fact_id, "name": self.name, "value": self.value,
           "status": self.status, "sources": [s.to_json() for s in self.sources]}
    if self.unit is not None: out["unit"] = self.unit
    if self.derivation is not None: out["derivation"] = self.derivation
    if self.input_fact_ids: out["input_fact_ids"] = list(self.input_fact_ids)
    if self.uncertainty: out["uncertainty"] = dict(self.uncertainty)
    if self.observed_at is not None: out["observed_at"] = self.observed_at
    return out

  @staticmethod
  def from_json(data:Mapping[str, Any]) -> "Fact":
    if data.get("schema") != SCHEMA_FACT: raise ValueError(f"expected schema {SCHEMA_FACT}")
    sources = tuple(EvidenceSource(s["producer"], s["tool"], s.get("path", ""), s["fingerprint"],
                                   dict(s.get("raw_summary", {}))) for s in data.get("sources", ()))
    fact = Fact(name=data["name"], value=data.get("value"), status=data["status"], unit=data.get("unit"),
                sources=sources, derivation=data.get("derivation"),
                input_fact_ids=tuple(data.get("input_fact_ids", ())), uncertainty=dict(data.get("uncertainty", {})),
                observed_at=data.get("observed_at"))
    if data.get("fact_id") not in (None, fact.fact_id): raise ValueError(f"fact_id mismatch for {fact.name}")
    return fact
