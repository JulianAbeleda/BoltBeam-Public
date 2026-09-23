"""Durable evidence requirements and pure predicate evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from boltbeam.vocab import SCHEMA_EVIDENCE_REQUIREMENTS


class RequirementLevel(str, Enum):
  REQUIRED = "required"
  RECOMMENDED = "recommended"
  OPTIONAL = "optional"


class RequirementStatus(str, Enum):
  SATISFIED = "satisfied"
  MISSING = "missing"
  INVALID = "invalid"
  MISMATCH = "mismatch"
  UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class EvidenceRequirement:
  requirement_id: str
  predicate: str
  level: str = RequirementLevel.REQUIRED.value
  blocks: tuple[str, ...] = ("comparison", "diagnosis")
  params: Mapping[str, Any] = None

  def __post_init__(self):
    if not self.requirement_id or not self.predicate: raise ValueError("requirement id and predicate are required")
    if self.level not in {v.value for v in RequirementLevel}: raise ValueError(f"unknown requirement level {self.level!r}")
    if any(v not in {"comparison", "diagnosis", "promotion"} for v in self.blocks): raise ValueError("unknown blocked decision")
    object.__setattr__(self, "params", dict(self.params or {}))

  def to_json(self) -> dict[str, Any]:
    return {"requirement_id": self.requirement_id, "predicate": self.predicate, "level": self.level,
            "blocks": list(self.blocks), "params": dict(self.params)}

  @staticmethod
  def from_json(data:Mapping[str, Any]) -> "EvidenceRequirement":
    return EvidenceRequirement(data["requirement_id"], data["predicate"], data.get("level", "required"),
                               tuple(data.get("blocks", ("comparison", "diagnosis"))), data.get("params", {}))


@dataclass(frozen=True)
class RequirementResult:
  requirement_id: str
  status: str
  reason: str = ""

  def __post_init__(self):
    if self.status not in {v.value for v in RequirementStatus}: raise ValueError(f"unknown result status {self.status!r}")

  def to_json(self) -> dict[str, str]:
    return {"requirement_id": self.requirement_id, "status": self.status, "reason": self.reason}


Predicate = Callable[[Mapping[str, Any], Mapping[str, Any]], RequirementResult | tuple[str, str] | str]
_PREDICATES: dict[str, Predicate] = {}


def register_predicate(name:str, predicate:Predicate) -> None:
  if not name or name in _PREDICATES: raise ValueError(f"predicate already registered: {name!r}")
  _PREDICATES[name] = predicate


def evaluate_requirements(requirements:tuple[EvidenceRequirement, ...], evidence:Mapping[str, Any]) -> tuple[RequirementResult, ...]:
  results = []
  for req in requirements:
    fn = _PREDICATES.get(req.predicate)
    if fn is None:
      results.append(RequirementResult(req.requirement_id, RequirementStatus.UNSUPPORTED.value,
                                       f"predicate {req.predicate!r} is not registered"))
      continue
    try: raw = fn(evidence, req.params)
    except (KeyError, TypeError, ValueError) as exc:
      results.append(RequirementResult(req.requirement_id, RequirementStatus.INVALID.value, str(exc))); continue
    if isinstance(raw, RequirementResult): result = raw
    elif isinstance(raw, tuple): result = RequirementResult(req.requirement_id, raw[0], raw[1])
    else: result = RequirementResult(req.requirement_id, raw)
    if result.requirement_id != req.requirement_id:
      result = RequirementResult(req.requirement_id, RequirementStatus.INVALID.value, "predicate returned wrong requirement id")
    results.append(result)
  return tuple(results)


def requirements_to_json(requirements:tuple[EvidenceRequirement, ...]) -> dict[str, Any]:
  return {"schema": SCHEMA_EVIDENCE_REQUIREMENTS, "requirements": [r.to_json() for r in requirements]}


def requirements_from_json(data:Mapping[str, Any]) -> tuple[EvidenceRequirement, ...]:
  if data.get("schema") != SCHEMA_EVIDENCE_REQUIREMENTS: raise ValueError(f"expected schema {SCHEMA_EVIDENCE_REQUIREMENTS}")
  reqs = tuple(EvidenceRequirement.from_json(r) for r in data.get("requirements", ()))
  ids = [r.requirement_id for r in reqs]
  if len(ids) != len(set(ids)): raise ValueError("duplicate evidence requirement ids")
  return reqs
