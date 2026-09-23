"""Content-addressed description of the closed execution system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from boltbeam.artifacts.base import sha256_json
from boltbeam.core.facts import Fact
from boltbeam.vocab import SCHEMA_SYSTEM_SNAPSHOT


@dataclass(frozen=True)
class SystemSnapshot:
  facts: tuple[Fact, ...]
  captured_at: str | None = None
  collector_version: str = "1"

  def __post_init__(self):
    names = [fact.name for fact in self.facts]
    if len(names) != len(set(names)): raise ValueError("duplicate fact names in system snapshot")

  @property
  def snapshot_id(self) -> str:
    return sha256_json({"facts": sorted(f.fact_id for f in self.facts), "collector_version": self.collector_version})

  def get(self, name:str) -> Fact | None:
    return next((fact for fact in self.facts if fact.name == name), None)

  def diff(self, other:"SystemSnapshot") -> dict[str, tuple[Fact | None, Fact | None]]:
    left, right = {f.name: f for f in self.facts}, {f.name: f for f in other.facts}
    return {name: (left.get(name), right.get(name)) for name in sorted(left.keys() | right.keys())
            if left.get(name) is None or right.get(name) is None or left[name].fact_id != right[name].fact_id}

  def compatible_with(self, other:"SystemSnapshot", required_facts:tuple[str, ...] = ()) -> bool:
    if not required_facts: return self.snapshot_id == other.snapshot_id
    return all(self.get(name) is not None and other.get(name) is not None and
               self.get(name).fact_id == other.get(name).fact_id for name in required_facts)

  def to_json(self) -> dict[str, Any]:
    out = {"schema": SCHEMA_SYSTEM_SNAPSHOT, "system_snapshot_id": self.snapshot_id,
           "collector_version": self.collector_version, "facts": [f.to_json() for f in self.facts]}
    if self.captured_at is not None: out["captured_at"] = self.captured_at
    return out

  @staticmethod
  def from_json(data:Mapping[str, Any]) -> "SystemSnapshot":
    if data.get("schema") != SCHEMA_SYSTEM_SNAPSHOT: raise ValueError(f"expected schema {SCHEMA_SYSTEM_SNAPSHOT}")
    snapshot = SystemSnapshot(tuple(Fact.from_json(f) for f in data.get("facts", ())), data.get("captured_at"),
                              str(data.get("collector_version", "1")))
    if data.get("system_snapshot_id") not in (None, snapshot.snapshot_id): raise ValueError("system_snapshot_id mismatch")
    return snapshot
