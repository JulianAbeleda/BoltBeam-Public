"""Immutable, content-addressed experiment definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from boltbeam.artifacts.base import sha256_json
from boltbeam.core.facts import _validate_json
from boltbeam.core.requirements import EvidenceRequirement
from boltbeam.vocab import SCHEMA_EXPERIMENT_MANIFEST


@dataclass(frozen=True)
class ExperimentCommand:
  argv: tuple[str, ...]
  env: Mapping[str, str] = field(default_factory=dict)
  cwd: str | None = None

  def __post_init__(self):
    if not self.argv or any(not isinstance(arg, str) or not arg for arg in self.argv):
      raise ValueError("commands require a non-empty argv array")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.env.items()):
      raise ValueError("command environment must contain strings")

  def semantic_json(self) -> dict[str, Any]:
    # cwd is deliberately excluded: checkout location is not experiment semantics.
    return {"argv": list(self.argv), "env": dict(self.env)}

  def to_json(self) -> dict[str, Any]:
    out = self.semantic_json()
    if self.cwd is not None: out["cwd"] = self.cwd
    return out

  @staticmethod
  def from_json(data:Mapping[str, Any]) -> "ExperimentCommand":
    argv = data.get("argv")
    if isinstance(argv, str): raise ValueError("command must be an argv array, not a shell string")
    return ExperimentCommand(tuple(argv or ()), dict(data.get("env", {})), data.get("cwd"))


@dataclass(frozen=True)
class ExperimentManifest:
  system_snapshot_id: str
  candidate_id: str
  comparator_id: str
  backend: str
  shape: Mapping[str, int]
  commands: tuple[ExperimentCommand, ...]
  requirements: tuple[EvidenceRequirement, ...]
  workload: Mapping[str, Any] = field(default_factory=dict)
  input_artifacts: Mapping[str, str] = field(default_factory=dict)
  warmups: int = 3
  repetitions: int = 10
  run_order: str = "interleaved"
  seed: int = 0
  timeout_s: float | None = None
  policies: Mapping[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    for name in ("system_snapshot_id", "candidate_id", "comparator_id", "backend"):
      if not getattr(self, name): raise ValueError(f"{name} is required")
    if not self.commands: raise ValueError("at least one command is required")
    if self.warmups < 0 or self.repetitions <= 0: raise ValueError("invalid warmup or repetition count")
    if self.timeout_s is not None and self.timeout_s <= 0: raise ValueError("timeout must be positive")
    _validate_json(self.shape, "shape"); _validate_json(self.workload, "workload"); _validate_json(self.policies, "policies")
    if any(not k or not isinstance(v, str) or not v.startswith("sha256:") for k, v in self.input_artifacts.items()):
      raise ValueError("input artifacts require named sha256 fingerprints")
    ids = [r.requirement_id for r in self.requirements]
    if len(ids) != len(set(ids)): raise ValueError("duplicate evidence requirement ids")

  def semantic_json(self) -> dict[str, Any]:
    return {"system_snapshot_id": self.system_snapshot_id, "candidate_id": self.candidate_id,
            "comparator_id": self.comparator_id, "backend": self.backend, "shape": dict(self.shape),
            "commands": [c.semantic_json() for c in self.commands],
            "requirements": [r.to_json() for r in self.requirements], "workload": dict(self.workload),
            "input_artifacts": dict(self.input_artifacts), "warmups": self.warmups,
            "repetitions": self.repetitions, "run_order": self.run_order, "seed": self.seed,
            "timeout_s": self.timeout_s, "policies": dict(self.policies)}

  @property
  def experiment_id(self) -> str:
    return sha256_json(self.semantic_json())

  def to_json(self) -> dict[str, Any]:
    out = self.semantic_json()
    out.update({"schema": SCHEMA_EXPERIMENT_MANIFEST, "experiment_id": self.experiment_id,
                "commands": [c.to_json() for c in self.commands]})
    return out

  @staticmethod
  def from_json(data:Mapping[str, Any]) -> "ExperimentManifest":
    if data.get("schema") != SCHEMA_EXPERIMENT_MANIFEST: raise ValueError(f"expected schema {SCHEMA_EXPERIMENT_MANIFEST}")
    manifest = ExperimentManifest(system_snapshot_id=data["system_snapshot_id"], candidate_id=data["candidate_id"],
      comparator_id=data["comparator_id"], backend=data["backend"], shape=dict(data["shape"]),
      commands=tuple(ExperimentCommand.from_json(c) for c in data["commands"]),
      requirements=tuple(EvidenceRequirement.from_json(r) for r in data.get("requirements", ())),
      workload=dict(data.get("workload", {})), input_artifacts=dict(data.get("input_artifacts", {})),
      warmups=int(data.get("warmups", 3)), repetitions=int(data.get("repetitions", 10)),
      run_order=data.get("run_order", "interleaved"), seed=int(data.get("seed", 0)),
      timeout_s=data.get("timeout_s"), policies=dict(data.get("policies", {})))
    if data.get("experiment_id") not in (None, manifest.experiment_id): raise ValueError("experiment_id mismatch")
    return manifest
