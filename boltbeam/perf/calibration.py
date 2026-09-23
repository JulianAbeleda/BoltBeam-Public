"""Content-addressed calibration parameters with measured uncertainty."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from boltbeam.artifacts.base import sha256_json
from boltbeam.core.facts import Fact, TruthStatus

SCHEMA = "boltbeam.mmq_calibration.v1"


@dataclass(frozen=True)
class Interval:
  low: float
  median: float
  high: float

  def __post_init__(self):
    if not (0 <= self.low <= self.median <= self.high): raise ValueError("interval requires 0 <= low <= median <= high")

  def to_json(self) -> dict[str, float]: return {"low": self.low, "median": self.median, "high": self.high}

  @staticmethod
  def from_json(value: Mapping[str, Any]) -> "Interval":
    return Interval(float(value["low"]), float(value["median"]), float(value["high"]))


@dataclass(frozen=True)
class CalibrationProfile:
  system_snapshot_id: str
  facts: tuple[Fact, ...]
  protocol_id: str
  model_version: str = "mmq-cycle-v1"

  def __post_init__(self):
    if not self.system_snapshot_id or not self.protocol_id: raise ValueError("system snapshot and protocol are required")
    names = [fact.name for fact in self.facts]
    if len(names) != len(set(names)): raise ValueError("duplicate calibration fact names")

  @property
  def calibration_id(self) -> str: return sha256_json(self.semantic_json())

  def semantic_json(self) -> dict[str, Any]:
    return {"system_snapshot_id": self.system_snapshot_id, "protocol_id": self.protocol_id,
            "model_version": self.model_version, "fact_ids": [f.fact_id for f in sorted(self.facts, key=lambda x: x.name)]}

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA, "calibration_id": self.calibration_id, **self.semantic_json(),
            "facts": [f.to_json() for f in self.facts]}

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "CalibrationProfile":
    if data.get("schema") != SCHEMA: raise ValueError(f"expected {SCHEMA}")
    profile = CalibrationProfile(data["system_snapshot_id"], tuple(Fact.from_json(x) for x in data["facts"]),
                                 data["protocol_id"], data.get("model_version", "mmq-cycle-v1"))
    if data.get("calibration_id") != profile.calibration_id: raise ValueError("calibration_id mismatch")
    return profile

  def fact(self, name: str) -> Fact | None: return next((f for f in self.facts if f.name == name), None)

  def interval(self, name: str) -> Interval | None:
    fact = self.fact(name)
    if fact is None or fact.status == TruthStatus.UNKNOWN.value: return None
    value = float(fact.value)
    uncertainty = fact.uncertainty
    if all(k in uncertainty for k in ("low", "high")):
      return Interval(float(uncertainty["low"]), value, float(uncertainty["high"]))
    return Interval(value, value, value)


__all__ = ["CalibrationProfile", "Interval", "SCHEMA"]
