"""Canonical kernel-analysis models for provider-neutral kernel evidence and reports."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from boltbeam.artifacts.base import EvidenceSource, sha256_json
from boltbeam.vocab import (
  EvidenceConfidence, OperandStrategy, SCHEMA_KERNEL_ANALYSIS, SCHEMA_KERNEL_EVIDENCE,
  SCHEMA_KERNEL_RECOMMENDATION, ServingTier,
)
from boltbeam.core.facts import TruthStatus


_HEX = set("0123456789abcdefABCDEF")


_TRUE_FALSE = {"true", "false", "null", "none", "zero"}


@dataclass(frozen=True)
class KernelHash:
  value: str

  def __post_init__(self):
    norm = self._normalize(self.value)
    object.__setattr__(self, "value", norm)

  @staticmethod
  def _normalize(raw: Any) -> str:
    if not isinstance(raw, str):
      raise ValueError("hash value must be a string")
    value = raw.strip().lower()
    if value.startswith("sha256:"):
      value = value[len("sha256:"):]
    if len(value) != 64 or any(ch not in _HEX for ch in value):
      raise ValueError(f"hash {raw!r} is not a valid sha256")
    return "sha256:" + value

  def __str__(self) -> str:
    return self.value


def _validate_truth_status(value: str) -> str:
  if value not in {s.value for s in TruthStatus}:
    raise ValueError(f"unknown truth status {value!r}")
  return value


def _validate_non_negative_number(value: Any, *, label: str, allow_none: bool = False) -> float | int | None:
  if value is None:
    if allow_none:
      return None
    raise ValueError(f"{label} is required")
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{label} must be a number")
  if not math.isfinite(value):
    raise ValueError(f"{label} must be finite")
  if value < 0:
    raise ValueError(f"{label} must be non-negative")
  if isinstance(value, bool):
    raise ValueError(f"{label} must not be bool")
  return value


def _validate_id(value: Any, *, label: str, required: bool = True) -> str | None:
  if value is None:
    if required:
      raise ValueError(f"{label} is required")
    return None
  if isinstance(value, str):
    return value
  raise ValueError(f"{label} must be a string")


def _normalize_sources(sources: Any) -> tuple[EvidenceSource, ...]:
  if not isinstance(sources, (list, tuple)):
    return ()
  out = []
  for raw in sources:
    if not isinstance(raw, Mapping):
      raise ValueError("sources must be source objects")
    source = EvidenceSource(
      raw.get("producer", "unknown"),
      raw.get("tool", "unknown"),
      raw.get("path", ""),
      raw.get("fingerprint", ""),
      dict(raw.get("raw_summary", {})),
    )
    if not source.fingerprint:
      raise ValueError("evidence source is missing fingerprint")
    if not source.producer or not source.tool:
      raise ValueError("evidence source missing producer or tool")
    if not source.fingerprint.startswith("sha256:"):
      raise ValueError("evidence source fingerprint must be sha256")
    out.append(source)
  return tuple(out)


def _json_tuple(raw: Any) -> tuple[str, ...]:
  if raw is None:
    return ()
  if not isinstance(raw, (list, tuple)):
    raise ValueError("expect list/tuple")
  return tuple(_validate_id(v, label="tuple element") for v in raw)


def _normalize_unit(unit: str | None) -> str | None:
  if unit is None:
    return None
  if not isinstance(unit, str) or not unit.strip():
    raise ValueError("unit must be a non-empty string")
  return unit.strip()


@dataclass(frozen=True)
class OperandTransport:
  declared_strategy: str | None = None
  requirements: tuple[str, ...] = ()
  fallback_eligible: bool = False
  register_resident: bool | None = None
  lds_staged: bool | None = None
  cache_streamed: bool | None = None
  mall_resident: bool | None = None   # relies on the large last-level cache (AMD Infinity Cache / MALL)
  reloaded: bool | None = None

  _FIELDS = ("register_resident", "lds_staged", "cache_streamed", "mall_resident", "reloaded")

  def __post_init__(self):
    for field in self._FIELDS:
      value = getattr(self, field)
      if value is not None and not isinstance(value, bool):
        raise ValueError(f"{field} must be bool or None")
    if not isinstance(self.fallback_eligible, bool):
      raise ValueError("fallback_eligible must be bool")
    object.__setattr__(self, "requirements", _json_tuple(self.requirements))
    true_strategy_flags = [f for f in self._FIELDS if f != "mall_resident" and getattr(self, f) is True]
    if len(true_strategy_flags) > 1:
      raise ValueError(f"contradictory operand strategies: {true_strategy_flags}")
    declared = self.declared_strategy
    if declared is None:
      declared = true_strategy_flags[0] if true_strategy_flags else OperandStrategy.UNKNOWN.value
    if declared not in {v.value for v in OperandStrategy}:
      raise ValueError(f"unknown operand strategy {declared!r}")
    if true_strategy_flags and declared != true_strategy_flags[0]:
      raise ValueError(f"declared strategy {declared!r} contradicts legacy flag {true_strategy_flags[0]!r}")
    object.__setattr__(self, "declared_strategy", declared)

  @property
  def strategy_hint(self) -> str:
    return {"register_resident": "register", "lds_staged": "lds", "cache_streamed": "cache",
            "reloaded": "reload", "unknown": "unknown"}[str(self.declared_strategy)]

  def to_json(self) -> dict[str, Any]:
    if self.mall_resident is True:
      raise ValueError("mall_resident is a serving-tier observation, not a transport strategy")
    out: dict[str, Any] = {"declared_strategy": self.declared_strategy}
    if self.requirements: out["requirements"] = list(self.requirements)
    if self.fallback_eligible: out["fallback_eligible"] = True
    return out

  @staticmethod
  def from_json(data: Any) -> "OperandTransport":
    if data is None:
      return OperandTransport()
    if not isinstance(data, Mapping):
      raise ValueError("operand transport must be an object")
    requirements = _json_tuple(data.get("requirements", ()))
    if data.get("mall_resident") is True and "legacy_mall_serving_tier_observation" not in requirements:
      requirements += ("legacy_mall_serving_tier_observation",)
    return OperandTransport(
      declared_strategy=data.get("declared_strategy"), requirements=requirements,
      fallback_eligible=data.get("fallback_eligible", False),
      **{f: data.get(f) for f in OperandTransport._FIELDS if f != "mall_resident"},
    )


@dataclass(frozen=True)
class CandidateModel:
  candidate_id: str
  candidate_hash: str | None = None
  strategy: str | None = None
  operand_transports: tuple[tuple[str, OperandTransport], ...] = field(default_factory=tuple)
  knobs: tuple[str, ...] = field(default_factory=tuple)
  comparator_id: str | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    candidate = _validate_id(self.candidate_id, label="candidate_id")
    if not candidate:
      raise ValueError("candidate_id is required")
    object.__setattr__(self, "candidate_id", candidate)

    if self.candidate_hash is not None:
      object.__setattr__(self, "candidate_hash", KernelHash(self.candidate_hash).value)

    transports = tuple((str(k), v if isinstance(v, OperandTransport) else OperandTransport.from_json(v))
                       for k, v in self.operand_transports)
    if len({k for k, _ in transports}) != len(transports):
      raise ValueError("operand names must be unique")
    object.__setattr__(self, "operand_transports", tuple((k, v) for k, v in transports))

    object.__setattr__(self, "knobs", _json_tuple(self.knobs))

    if self.comparator_id is not None:
      object.__setattr__(self, "comparator_id", _validate_id(self.comparator_id, label="comparator_id", required=False))

    if self.strategy is not None and not isinstance(self.strategy, str):
      raise ValueError("strategy must be string")

    if not isinstance(self.extra, dict):
      raise ValueError("candidate.extra must be an object")

  @property
  def transport_map(self) -> dict[str, dict[str, bool]]:
    # Compatibility view for existing contrast/prediction consumers. Serialization below is canonical.
    return {operand: ({transport.declared_strategy: True} if transport.declared_strategy != "unknown" else {})
            for operand, transport in self.operand_transports}

  @property
  def derived_strategy(self) -> str:
    if self.strategy:
      return self.strategy
    if not self.operand_transports:
      return "unknown"
    return "-".join(f"{op}:{t.strategy_hint}" for op, t in self.operand_transports)

  def to_json(self) -> dict[str, Any]:
    out = {
      "candidate_id": self.candidate_id,
      "comparator_id": self.comparator_id,
      "candidate_hash": self.candidate_hash,
      "strategy": self.derived_strategy,
      "knobs": list(self.knobs),
    }
    if self.extra:
      out["extra"] = dict(self.extra)
    if self.operand_transports:
      out["operand_transports"] = {operand: tr.to_json() for operand, tr in self.operand_transports}
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "CandidateModel":
    if not isinstance(data, Mapping):
      raise ValueError("candidate must be object")
    transports = []
    for operand, raw in dict(data.get("operand_transports", {})).items():
      transports.append((str(operand), OperandTransport.from_json(raw)))
    return CandidateModel(
      candidate_id=data["candidate_id"], candidate_hash=data.get("candidate_hash"),
      strategy=data.get("strategy"), comparator_id=data.get("comparator_id"),
      operand_transports=tuple(transports), knobs=_json_tuple(data.get("knobs", ())),
      extra=dict(data.get("extra", {})),
    )


@dataclass(frozen=True)
class IdentityModel:
  provider: str | None = None
  compiler: str | None = None
  backend: str | None = None
  version: str | None = None
  commit: str | None = None
  candidate_id: str | None = None
  binary_hash: str | None = None
  executed_binary_hash: str | None = None
  input_identity: str | None = None
  output_identity: str | None = None
  reference_identity: str | None = None
  target_id: str | None = None
  device_id: str | None = None
  system_snapshot_id: str | None = None
  experiment_id: str | None = None
  session_id: str | None = None
  clock_state_id: str | None = None
  source_hash: str | None = None
  semantic_schedule_digest: str | None = None
  abi_digest: str | None = None
  operand_role_map_digest: str | None = None
  candidate_digest: str | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    if self.provider is not None and not isinstance(self.provider, str):
      raise ValueError("provider must be str")
    for field in ("compiler", "backend", "version", "commit", "candidate_id", "target_id", "device_id",
                  "system_snapshot_id", "experiment_id", "session_id", "semantic_schedule_digest", "abi_digest",
                  "operand_role_map_digest", "candidate_digest", "clock_state_id"):
      value = getattr(self, field)
      if value is not None and not isinstance(value, str):
        raise ValueError(f"{field} must be string")

    if self.binary_hash is not None:
      object.__setattr__(self, "binary_hash", KernelHash(self.binary_hash).value)
    if self.executed_binary_hash is not None:
      object.__setattr__(self, "executed_binary_hash", KernelHash(self.executed_binary_hash).value)
    if self.source_hash is not None:
      object.__setattr__(self, "source_hash", KernelHash(self.source_hash).value)
    if self.input_identity is not None:
      object.__setattr__(self, "input_identity", _validate_id(self.input_identity, label="input_identity", required=False))
    if self.output_identity is not None:
      object.__setattr__(self, "output_identity", _validate_id(self.output_identity, label="output_identity", required=False))
    if self.reference_identity is not None:
      object.__setattr__(self, "reference_identity", _validate_id(self.reference_identity, label="reference_identity", required=False))

    if self.version is not None and not self.version:
      raise ValueError("version cannot be empty")

    if not isinstance(self.extra, dict):
      raise ValueError("extra must be object")

  def to_json(self) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("provider", "compiler", "backend", "version", "commit", "candidate_id", "binary_hash",
                  "executed_binary_hash", "input_identity", "output_identity", "reference_identity", "target_id",
                  "device_id", "system_snapshot_id", "experiment_id", "session_id", "source_hash",
                  "semantic_schedule_digest", "abi_digest", "operand_role_map_digest", "candidate_digest", "clock_state_id"):
      value = getattr(self, field)
      if value:
        out[field] = value
    if self.extra:
      out["extra"] = dict(self.extra)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "IdentityModel":
    if not isinstance(data, Mapping):
      raise ValueError("identity must be object")
    return IdentityModel(**{k: data.get(k) for k in IdentityModel.__dataclass_fields__ if k in data and k != "extra"},
                         extra=dict(data.get("extra", {})))


@dataclass(frozen=True)
class WorkloadModel:
  operation: str
  role: str
  shape: Mapping[str, int]
  dtypes: Mapping[str, str] = field(default_factory=dict)
  layout: Mapping[str, str] = field(default_factory=dict)
  batch: int | None = None
  head: int | None = None
  context: int | None = None
  seed: str | None = None
  seed_index: int | None = None
  transport: str | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    _validate_id(self.operation, label="operation")
    _validate_id(self.role, label="role")
    object.__setattr__(self, "operation", self.operation)
    object.__setattr__(self, "role", self.role)
    if not isinstance(self.shape, Mapping):
      raise ValueError("shape must be object")
    normalized_shape: dict[str, int] = {}
    for key, value in self.shape.items():
      if not isinstance(key, str) or not key:
        raise ValueError("shape keys must be strings")
      if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("shape values must be ints")
      if value < 0:
        raise ValueError("shape dimensions cannot be negative")
      normalized_shape[key] = int(value)
    object.__setattr__(self, "shape", normalized_shape)
    for name, value in (("batch", self.batch), ("head", self.head), ("context", self.context),
                        ("seed_index", self.seed_index)):
      if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
          raise ValueError(f"{name} must be non-negative int")
    if self.seed is not None and not isinstance(self.seed, str):
      raise ValueError("seed must be string")
    if self.transport is not None and not isinstance(self.transport, str):
      raise ValueError("transport must be string")
    if not isinstance(self.dtypes, Mapping) or not isinstance(self.layout, Mapping):
      raise ValueError("dtypes and layout must be objects")
    if self.extra is not None and not isinstance(self.extra, dict):
      raise ValueError("extra must be object")

  def to_json(self) -> dict[str, Any]:
    out = {"operation": self.operation, "role": self.role, "shape": dict(self.shape)}
    if self.dtypes: out["dtypes"] = dict(self.dtypes)
    if self.layout: out["layout"] = dict(self.layout)
    if self.batch is not None: out["batch"] = self.batch
    if self.head is not None: out["head"] = self.head
    if self.context is not None: out["context"] = self.context
    if self.seed is not None: out["seed"] = self.seed
    if self.seed_index is not None: out["seed_index"] = self.seed_index
    if self.transport is not None: out["transport"] = self.transport
    if self.extra: out["extra"] = dict(self.extra)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "WorkloadModel":
    if not isinstance(data, Mapping):
      raise ValueError("workload must be object")
    return WorkloadModel(
      operation=data["operation"], role=data["role"], shape=dict(data.get("shape", {})),
      dtypes=dict(data.get("dtypes", {})), layout=dict(data.get("layout", {})),
      batch=data.get("batch"), head=data.get("head"), context=data.get("context"),
      seed=data.get("seed"), seed_index=data.get("seed_index"), transport=data.get("transport"),
      extra=dict(data.get("extra", {})),
    )


_STAGE_STATUS = {
  "compile": {"not_run", "pass", "unsupported", "fail", "timeout", "no_result"},
  "execution": {"not_run", "pass", "timeout", "runtime_fault", "health_failure", "no_result"},
  "correctness": {"not_run", "pass", "fail", "incomplete"},
  "timing": {"not_run", "measured", "blocked", "invalid"},
}


_ALL_STAGE_STATUS = {s for values in _STAGE_STATUS.values() for s in values}


@dataclass(frozen=True)
class PipelineStage:
  status: str
  duration_ms: float | None = None
  error_class: str | None = None
  bounded_detail: str | None = None
  producer_status: str | None = None
  sources: tuple[EvidenceSource, ...] = field(default_factory=tuple)

  def __post_init__(self):
    if self.status not in _ALL_STAGE_STATUS:
      raise ValueError(f"unknown stage status {self.status!r}")
    if self.duration_ms is not None:
      _validate_non_negative_number(self.duration_ms, label="duration_ms")
    if self.error_class is not None and not isinstance(self.error_class, str):
      raise ValueError("error_class must be string")
    if self.bounded_detail is not None and not isinstance(self.bounded_detail, str):
      raise ValueError("bounded_detail must be string")
    if self.producer_status is not None and not isinstance(self.producer_status, str):
      raise ValueError("producer_status must be string")
    if not isinstance(self.sources, tuple):
      object.__setattr__(self, "sources", tuple(self.sources))

  def to_json(self) -> dict[str, Any]:
    out = {"status": self.status}
    if self.duration_ms is not None: out["duration_ms"] = self.duration_ms
    if self.error_class is not None: out["error_class"] = self.error_class
    if self.bounded_detail is not None: out["bounded_detail"] = self.bounded_detail
    if self.producer_status is not None: out["producer_status"] = self.producer_status
    if self.sources: out["sources"] = [s.to_json() for s in self.sources]
    return out

  @staticmethod
  def from_json(data: Any, *, stage: str) -> "PipelineStage":
    data = dict(data or {})
    stage_obj = PipelineStage(
      status=data.get("status", "not_run"),
      duration_ms=data.get("duration_ms"),
      error_class=data.get("error_class"),
      bounded_detail=data.get("bounded_detail"),
      producer_status=data.get("producer_status"),
      sources=_normalize_sources(data.get("sources", ())),
    )
    _validate_stage_status(stage, stage_obj.status)
    return stage_obj


def _validate_stage_status(stage: str, status: str) -> None:
  allowed = _STAGE_STATUS.get(stage)
  if allowed is not None and status not in allowed:
    raise ValueError(f"invalid {stage} status {status!r}; expected one of {sorted(allowed)}")


@dataclass(frozen=True)
class KernelStages:
  compile: PipelineStage
  execution: PipelineStage
  correctness: PipelineStage
  timing: PipelineStage

  def __post_init__(self):
    for stage in ("compile", "execution", "correctness", "timing"):
      value = getattr(self, stage)
      if not isinstance(value, PipelineStage):
        raise ValueError(f"{stage} must be a PipelineStage")
      _validate_stage_status(stage, value.status)

  def to_json(self) -> dict[str, Any]:
    return {
      "compile": self.compile.to_json(),
      "execution": self.execution.to_json(),
      "correctness": self.correctness.to_json(),
      "timing": self.timing.to_json(),
    }

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "KernelStages":
    if not isinstance(data, Mapping):
      raise ValueError("stages must be object")
    return KernelStages(
      compile=PipelineStage.from_json(data.get("compile"), stage="compile"),
      execution=PipelineStage.from_json(data.get("execution"), stage="execution"),
      correctness=PipelineStage.from_json(data.get("correctness"), stage="correctness"),
      timing=PipelineStage.from_json(data.get("timing"), stage="timing"),
    )


@dataclass(frozen=True)
class Measurement:
  scope: str
  samples: tuple[float, ...] = ()
  units: str | None = None
  warmups: int = 0
  repetitions: int = 0
  inclusion: str = "all"
  sync: str | None = None
  median_ms: float | None = None
  min_ms: float | None = None
  spread_ms: float | None = None

  def __post_init__(self):
    if not isinstance(self.scope, str) or not self.scope:
      raise ValueError("scope is required")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in self.samples):
      raise ValueError("timing samples must be finite numbers")
    if any(v < 0 for v in self.samples):
      raise ValueError("timing samples must be non-negative")
    if not isinstance(self.warmups, int) or self.warmups < 0:
      raise ValueError("warmups must be non-negative int")
    if not isinstance(self.repetitions, int) or self.repetitions < 0:
      raise ValueError("repetitions must be non-negative int")
    _normalize_unit(self.units)

  @property
  def normalized_samples_ms(self) -> tuple[float, ...]:
    return tuple(float(s) for s in self.samples)

  @property
  def median_sample_ms(self) -> float | None:
    if not self.samples:
      return None
    values = sorted(float(s) for s in self.samples)
    return values[len(values)//2]

  def to_json(self) -> dict[str, Any]:
    out = {"scope": self.scope, "samples": list(self.samples)}
    if self.units is not None: out["units"] = self.units
    if self.warmups: out["warmups"] = self.warmups
    if self.repetitions: out["repetitions"] = self.repetitions
    if self.inclusion != "all": out["inclusion"] = self.inclusion
    if self.sync is not None: out["sync"] = self.sync
    if self.median_ms is not None: out["median_ms"] = self.median_ms
    if self.min_ms is not None: out["min_ms"] = self.min_ms
    if self.spread_ms is not None: out["spread_ms"] = self.spread_ms
    return out

  @staticmethod
  def from_json(data: Any) -> "Measurement":
    if data is None:
      raise ValueError("measurement missing")
    if not isinstance(data, Mapping):
      raise ValueError("measurement must be object")
    samples = tuple(float(x) for x in data.get("samples", ()))
    return Measurement(
      scope=data.get("scope", "kernel"),
      samples=samples,
      units=_normalize_unit(data.get("units")),
      warmups=int(data.get("warmups", 0)),
      repetitions=int(data.get("repetitions", 0)),
      inclusion=data.get("inclusion", "all"),
      sync=data.get("sync"),
      median_ms=data.get("median_ms"),
      min_ms=data.get("min_ms"),
      spread_ms=data.get("spread_ms"),
    )


@dataclass(frozen=True)
class CorrectnessMetrics:
  scope: str
  sample_count: int | None = None
  element_count: int | None = None
  max_error: float | None = None
  mean_error: float | None = None
  relative_error: float | None = None
  finite_output: bool | None = None
  numerical_passed: bool | None = None
  zero_fraction: float | None = None
  nan_fraction: float | None = None
  inf_fraction: float | None = None
  correlation: float | None = None
  tolerance_abs: float | None = None
  tolerance_rel: float | None = None
  seed: str | None = None
  sample_deterministic: bool | None = None
  output_signature: str | None = None
  missing_metrics: tuple[str, ...] = ()
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    if not isinstance(self.scope, str) or not self.scope:
      raise ValueError("correctness scope is required")
    for name, value in ("sample_count", self.sample_count), ("element_count", self.element_count):
      if value is not None:
        if not isinstance(value, int) or value < 0:
          raise ValueError(f"{name} must be non-negative int")
    for name, value in (("zero_fraction", self.zero_fraction), ("nan_fraction", self.nan_fraction),
                        ("inf_fraction", self.inf_fraction), ("max_error", self.max_error),
                        ("mean_error", self.mean_error), ("relative_error", self.relative_error),
                        ("tolerance_abs", self.tolerance_abs), ("tolerance_rel", self.tolerance_rel)):
      if value is not None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
          raise ValueError(f"{name} must be numeric")
        if not math.isfinite(value):
          raise ValueError(f"{name} must be finite")
        if name.endswith("fraction") and not 0 <= value <= 1:
          raise ValueError(f"{name} must be in [0, 1]")
    if self.correlation is not None:
      if isinstance(self.correlation, bool) or not isinstance(self.correlation, (int, float)):
        raise ValueError("correlation must be numeric")
      if not math.isfinite(self.correlation) or not -1 <= self.correlation <= 1:
        raise ValueError("correlation must be finite in [-1, 1]")
    if self.numerical_passed is not None and not isinstance(self.numerical_passed, bool):
      raise ValueError("numerical_passed must be bool")
    if not isinstance(self.missing_metrics, tuple):
      raise ValueError("missing_metrics must be tuple")
    for item in self.missing_metrics:
      if not isinstance(item, str) or not item:
        raise ValueError("missing_metrics values must be strings")
    if self.extra is not None and not isinstance(self.extra, dict):
      raise ValueError("extra must be object")

  @property
  def status(self) -> str:
    if self.scope == "full_gemm":
      required = (self.element_count is not None, self.tolerance_abs is not None, self.tolerance_rel is not None,
                  self.max_error is not None, self.finite_output is not None)
      if all(required) and not self.missing_metrics:
        numerical_ok = self.numerical_passed if self.numerical_passed is not None else self.max_error == 0
        if numerical_ok and self.finite_output and (self.nan_fraction in (0, None)) and (self.inf_fraction in (0, None)):
          return "pass"
    if self.missing_metrics:
      return "incomplete"
    return "fail" if self.max_error is not None and self.max_error > 0 else "pass"

  def to_json(self) -> dict[str, Any]:
    out: dict[str, Any] = {"scope": self.scope, "status": self.status}
    for field in ("sample_count", "element_count", "max_error", "mean_error", "relative_error", "finite_output", "numerical_passed",
                  "zero_fraction", "nan_fraction", "inf_fraction", "correlation", "tolerance_abs", "tolerance_rel",
                  "seed", "sample_deterministic", "output_signature"):
      value = getattr(self, field)
      if value is not None:
        out[field] = value
    if self.missing_metrics:
      out["missing_metrics"] = list(self.missing_metrics)
    if self.extra:
      out["extra"] = dict(self.extra)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "CorrectnessMetrics":
    if not isinstance(data, Mapping):
      raise ValueError("correctness must be object")
    return CorrectnessMetrics(
      scope=data.get("scope", "full_gemm"),
      sample_count=data.get("sample_count"),
      element_count=data.get("element_count"),
      max_error=data.get("max_error"),
      mean_error=data.get("mean_error"),
      relative_error=data.get("relative_error"),
      finite_output=data.get("finite_output"),
      numerical_passed=data.get("numerical_passed"),
      zero_fraction=data.get("zero_fraction"),
      nan_fraction=data.get("nan_fraction"),
      inf_fraction=data.get("inf_fraction"),
      correlation=data.get("correlation"),
      tolerance_abs=data.get("tolerance_abs"),
      tolerance_rel=data.get("tolerance_rel"),
      seed=data.get("seed"),
      sample_deterministic=data.get("sample_deterministic"),
      output_signature=data.get("output_signature"),
      missing_metrics=_json_tuple(data.get("missing_metrics", ())),
      extra=dict(data.get("extra", {})),
    )


def _json_object(value: Any, *, label: str) -> dict[str, Any]:
  if value is None:
    return {}
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be an object")
  return dict(value)


@dataclass(frozen=True)
class InstructionSite:
  """One final-program operation attributed (or explicitly not attributable) to an operand."""
  operation_id: str
  kind: str
  width_bytes: int | None = None
  address: str | None = None
  fetch_group: str | None = None
  cache_policy: str | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    _validate_id(self.operation_id, label="operation_id")
    _validate_id(self.kind, label="instruction kind")
    if self.width_bytes is not None and (isinstance(self.width_bytes, bool) or
                                         not isinstance(self.width_bytes, int) or self.width_bytes < 0):
      raise ValueError("width_bytes must be a non-negative int")
    for name in ("address", "fetch_group", "cache_policy"):
      value = getattr(self, name)
      if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be string")
    if not isinstance(self.extra, dict): raise ValueError("instruction extra must be object")

  def to_json(self) -> dict[str, Any]:
    out = {"operation_id": self.operation_id, "kind": self.kind}
    for name in ("width_bytes", "address", "fetch_group", "cache_policy"):
      if getattr(self, name) is not None: out[name] = getattr(self, name)
    if self.extra: out["extra"] = dict(self.extra)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "InstructionSite":
    if not isinstance(data, Mapping): raise ValueError("instruction site must be object")
    return InstructionSite(data["operation_id"], data["kind"], data.get("width_bytes"), data.get("address"),
                           data.get("fetch_group"), data.get("cache_policy"), dict(data.get("extra", {})))


@dataclass(frozen=True)
class OperandStaticEvidence:
  instruction_sites: tuple[InstructionSite, ...] = ()
  compulsory_fetch_groups: tuple[str, ...] = ()
  expected_semantic_bytes: int | None = None
  reload_factor: float | None = None
  fragment_bytes_per_lane: int | None = None
  fragment_bytes_per_wave: int | None = None
  fragment_bytes_per_workgroup: int | None = None
  operand_lds_bytes: int | None = None
  accumulator_register_bytes: int | None = None
  temporary_register_bytes: int | None = None
  spill_bytes: int | None = None
  barriers: int | None = None
  waits: int | None = None
  loops: int | None = None
  pipeline_stages: int | None = None
  buffering: int | None = None
  attribution_status: str = "unknown"
  complete_control_flow: bool | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    object.__setattr__(self, "instruction_sites", tuple(
      s if isinstance(s, InstructionSite) else InstructionSite.from_json(s) for s in self.instruction_sites))
    object.__setattr__(self, "compulsory_fetch_groups", _json_tuple(self.compulsory_fetch_groups))
    for name in ("expected_semantic_bytes", "fragment_bytes_per_lane", "fragment_bytes_per_wave",
                 "fragment_bytes_per_workgroup", "operand_lds_bytes", "accumulator_register_bytes",
                 "temporary_register_bytes", "spill_bytes", "barriers", "waits", "loops",
                 "pipeline_stages", "buffering"):
      value = getattr(self, name)
      if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a non-negative int")
    if self.reload_factor is not None: _validate_non_negative_number(self.reload_factor, label="reload_factor")
    if self.attribution_status not in {"final_isa", "abi_address_analysis", "isolated_variant", "ambiguous",
                                       "source_binary_disagreement", "unknown"}:
      raise ValueError(f"unknown attribution status {self.attribution_status!r}")
    if self.complete_control_flow is not None and not isinstance(self.complete_control_flow, bool):
      raise ValueError("complete_control_flow must be bool")
    if not isinstance(self.extra, dict): raise ValueError("operand static extra must be object")

  def to_json(self) -> dict[str, Any]:
    out: dict[str, Any] = {"attribution_status": self.attribution_status}
    if self.instruction_sites: out["instruction_sites"] = [s.to_json() for s in self.instruction_sites]
    if self.compulsory_fetch_groups: out["compulsory_fetch_groups"] = list(self.compulsory_fetch_groups)
    for name in ("expected_semantic_bytes", "reload_factor", "fragment_bytes_per_lane",
                 "fragment_bytes_per_wave", "fragment_bytes_per_workgroup", "operand_lds_bytes",
                 "accumulator_register_bytes", "temporary_register_bytes", "spill_bytes", "barriers", "waits",
                 "loops", "pipeline_stages", "buffering", "complete_control_flow"):
      if getattr(self, name) is not None: out[name] = getattr(self, name)
    if self.extra: out["extra"] = dict(self.extra)
    return out

  @staticmethod
  def from_json(data: Any) -> "OperandStaticEvidence":
    data = _json_object(data, label="operand static evidence")
    names = {f for f in OperandStaticEvidence.__dataclass_fields__ if f not in {"instruction_sites", "extra"}}
    return OperandStaticEvidence(instruction_sites=tuple(InstructionSite.from_json(s) for s in data.get("instruction_sites", ())),
                                 extra=dict(data.get("extra", {})), **{n: data.get(n) for n in names if n in data})


@dataclass(frozen=True)
class ServingTierObservation:
  tier: str
  status: str = "unknown"
  bytes: float | int | None = None
  requests: float | int | None = None
  hits: float | int | None = None
  misses: float | int | None = None
  hit_rate: float | None = None
  unit: str | None = None
  scope: str = "per_operand"
  uncertainty: dict[str, Any] = field(default_factory=dict)
  source: str | None = None
  reason: str | None = None

  def __post_init__(self):
    if self.tier not in {v.value for v in ServingTier}: raise ValueError(f"unknown serving tier {self.tier!r}")
    if self.status not in {s.value for s in TruthStatus} | {"unsupported"}:
      raise ValueError(f"unknown serving-tier status {self.status!r}")
    for name in ("bytes", "requests", "hits", "misses"):
      value = getattr(self, name)
      if value is not None: _validate_non_negative_number(value, label=name)
    if self.hit_rate is not None and (not isinstance(self.hit_rate, (int, float)) or isinstance(self.hit_rate, bool)
                                      or not 0 <= self.hit_rate <= 1): raise ValueError("hit_rate must be in [0,1]")
    if self.scope not in {"per_operand", "per_kernel", "controlled_proxy"}: raise ValueError("invalid tier scope")
    if not isinstance(self.uncertainty, dict): raise ValueError("uncertainty must be object")
    if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
      raise ValueError("serving-tier reason must be a non-empty string")
    if self.status == "unsupported" and self.reason is None:
      raise ValueError("unsupported serving-tier observation requires a reason")

  def to_json(self) -> dict[str, Any]:
    out = {"tier": self.tier, "status": self.status, "scope": self.scope}
    for name in ("bytes", "requests", "hits", "misses", "hit_rate", "unit", "source", "reason"):
      if getattr(self, name) is not None: out[name] = getattr(self, name)
    if self.uncertainty: out["uncertainty"] = dict(self.uncertainty)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "ServingTierObservation":
    return ServingTierObservation(**{n: data.get(n) for n in ServingTierObservation.__dataclass_fields__ if n in data},
                                  uncertainty=dict(data.get("uncertainty", {})))


@dataclass(frozen=True)
class OperandDynamicEvidence:
  requested_bytes: float | int | None = None
  transferred_bytes: float | int | None = None
  useful_bytes: float | int | None = None
  requests: float | int | None = None
  transactions: float | int | None = None
  transaction_width_bytes: float | int | None = None
  serving_tiers: tuple[ServingTierObservation, ...] = ()
  samples: tuple[float, ...] = ()
  interval: tuple[float, float] | None = None
  repetitions: int | None = None
  multiplexed: bool | None = None
  scope: str = "per_operand"
  counter_id: str | None = None
  calibration_id: str | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    for name in ("requested_bytes", "transferred_bytes", "useful_bytes", "requests", "transactions",
                 "transaction_width_bytes", "repetitions"):
      value = getattr(self, name)
      if value is not None:
        if name == "repetitions" and (isinstance(value, bool) or not isinstance(value, int)):
          raise ValueError("repetitions must be a non-negative int")
        _validate_non_negative_number(value, label=name)
    object.__setattr__(self, "serving_tiers", tuple(t if isinstance(t, ServingTierObservation)
                                                    else ServingTierObservation.from_json(t) for t in self.serving_tiers))
    if len({t.tier for t in self.serving_tiers}) != len(self.serving_tiers): raise ValueError("duplicate serving tier")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in self.samples):
      raise ValueError("dynamic samples must be finite numbers")
    if self.interval is not None:
      interval = tuple(self.interval); object.__setattr__(self, "interval", interval)
      if len(interval) != 2 or interval[0] > interval[1]: raise ValueError("invalid dynamic interval")
    if self.scope not in {"per_operand", "per_kernel", "controlled_proxy"}: raise ValueError("invalid dynamic scope")
    if not isinstance(self.extra, dict): raise ValueError("operand dynamic extra must be object")

  def to_json(self) -> dict[str, Any]:
    out = {"scope": self.scope}
    for name in ("requested_bytes", "transferred_bytes", "useful_bytes", "requests", "transactions",
                 "transaction_width_bytes", "repetitions", "multiplexed", "counter_id", "calibration_id"):
      if getattr(self, name) is not None: out[name] = getattr(self, name)
    if self.serving_tiers: out["serving_tiers"] = [t.to_json() for t in self.serving_tiers]
    if self.samples: out["samples"] = list(self.samples)
    if self.interval is not None: out["interval"] = list(self.interval)
    if self.extra: out["extra"] = dict(self.extra)
    return out

  @staticmethod
  def from_json(data: Any) -> "OperandDynamicEvidence":
    data = _json_object(data, label="operand dynamic evidence")
    return OperandDynamicEvidence(**{n: data.get(n) for n in OperandDynamicEvidence.__dataclass_fields__
                                     if n in data and n not in {"serving_tiers", "extra"}},
                                  serving_tiers=tuple(ServingTierObservation.from_json(t) for t in data.get("serving_tiers", ())),
                                  extra=dict(data.get("extra", {})))


@dataclass(frozen=True)
class OperandClassification:
  strategy: str = OperandStrategy.UNKNOWN.value
  status: str = TruthStatus.UNKNOWN.value
  confidence: str = EvidenceConfidence.UNKNOWN.value
  supporting_evidence_ids: tuple[str, ...] = ()
  contradicting_evidence_ids: tuple[str, ...] = ()
  missing_discriminators: tuple[str, ...] = ()
  ruleset_version: str = "operand-path-structural.v1"

  def __post_init__(self):
    if self.strategy not in {v.value for v in OperandStrategy}: raise ValueError("invalid classified strategy")
    _validate_truth_status(self.status)
    if self.confidence not in {v.value for v in EvidenceConfidence}: raise ValueError("invalid confidence")
    for name in ("supporting_evidence_ids", "contradicting_evidence_ids", "missing_discriminators"):
      object.__setattr__(self, name, _json_tuple(getattr(self, name)))

  def to_json(self) -> dict[str, Any]:
    out = {"strategy": self.strategy, "status": self.status, "confidence": self.confidence,
           "ruleset_version": self.ruleset_version}
    for name in ("supporting_evidence_ids", "contradicting_evidence_ids", "missing_discriminators"):
      if getattr(self, name): out[name] = list(getattr(self, name))
    return out

  @staticmethod
  def from_json(data: Any) -> "OperandClassification":
    data = _json_object(data, label="operand classification")
    return OperandClassification(**{n: data.get(n) for n in OperandClassification.__dataclass_fields__ if n in data})


@dataclass(frozen=True)
class OperandPathEvidence:
  operand_id: str
  semantic_role: str
  scope: str
  dtype: str | None = None
  logical_shape: tuple[int, ...] = ()
  layout: str | None = None
  declared_strategy: str = OperandStrategy.UNKNOWN.value
  static: OperandStaticEvidence = field(default_factory=OperandStaticEvidence)
  dynamic: OperandDynamicEvidence = field(default_factory=OperandDynamicEvidence)
  classification: OperandClassification = field(default_factory=OperandClassification)
  provenance: tuple[str, ...] = ()
  blockers: tuple[FactBlocker, ...] = ()

  def __post_init__(self):
    _validate_id(self.operand_id, label="operand_id"); _validate_id(self.semantic_role, label="semantic_role")
    _validate_id(self.scope, label="operand scope")
    if self.declared_strategy not in {v.value for v in OperandStrategy}: raise ValueError("invalid declared strategy")
    shape = tuple(self.logical_shape); object.__setattr__(self, "logical_shape", shape)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in shape): raise ValueError("invalid logical_shape")
    object.__setattr__(self, "provenance", _json_tuple(self.provenance))
    if not isinstance(self.static, OperandStaticEvidence): object.__setattr__(self, "static", OperandStaticEvidence.from_json(self.static))
    if not isinstance(self.dynamic, OperandDynamicEvidence): object.__setattr__(self, "dynamic", OperandDynamicEvidence.from_json(self.dynamic))
    if not isinstance(self.classification, OperandClassification): object.__setattr__(self, "classification", OperandClassification.from_json(self.classification))
    object.__setattr__(self, "blockers", tuple(b if isinstance(b, FactBlocker) else FactBlocker.from_json(b) for b in self.blockers))

  def to_json(self) -> dict[str, Any]:
    out = {"operand_id": self.operand_id, "semantic_role": self.semantic_role, "scope": self.scope,
           "declared_strategy": self.declared_strategy, "static": self.static.to_json(),
           "dynamic": self.dynamic.to_json(), "classification": self.classification.to_json()}
    if self.dtype is not None: out["dtype"] = self.dtype
    if self.logical_shape: out["logical_shape"] = list(self.logical_shape)
    if self.layout is not None: out["layout"] = self.layout
    if self.provenance: out["provenance"] = list(self.provenance)
    if self.blockers: out["blockers"] = [b.to_json() for b in self.blockers]
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "OperandPathEvidence":
    return OperandPathEvidence(operand_id=data["operand_id"], semantic_role=data.get("semantic_role", data["operand_id"]),
      scope=data.get("scope", "kernel"), dtype=data.get("dtype"), logical_shape=tuple(data.get("logical_shape", ())),
      layout=data.get("layout"), declared_strategy=data.get("declared_strategy", OperandStrategy.UNKNOWN.value),
      static=OperandStaticEvidence.from_json(data.get("static")), dynamic=OperandDynamicEvidence.from_json(data.get("dynamic")),
      classification=OperandClassification.from_json(data.get("classification")), provenance=_json_tuple(data.get("provenance", ())),
      blockers=tuple(FactBlocker.from_json(b) for b in data.get("blockers", ())))


@dataclass(frozen=True)
class ResourceSummary:
  vgpr: int | None = None
  sgpr: int | None = None
  lds_bytes: int | None = None
  scratch_bytes: int | None = None
  scratch_loads: int | None = None
  scratch_stores: int | None = None
  spilled_vgprs: int | None = None
  spilled_sgprs: int | None = None
  occupancy: float | None = None
  grid: tuple[int, int, int] | None = None
  workgroup_threads: int | None = None
  workgroup: tuple[int, int, int] | None = None
  lds_alloc: int | None = None
  waves: int | None = None
  tile: tuple[int, ...] | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    for name, value in (("vgpr", self.vgpr), ("sgpr", self.sgpr), ("lds_bytes", self.lds_bytes),
                        ("scratch_bytes", self.scratch_bytes), ("scratch_loads", self.scratch_loads),
                        ("scratch_stores", self.scratch_stores), ("spilled_vgprs", self.spilled_vgprs),
                        ("spilled_sgprs", self.spilled_sgprs), ("workgroup_threads", self.workgroup_threads)):
      if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
          raise ValueError(f"{name} must be non-negative int")
    if self.occupancy is not None:
      _validate_non_negative_number(self.occupancy, label="occupancy")
    for tuple_name, value in ("grid", self.grid), ("workgroup", self.workgroup):
      if value is not None:
        if not isinstance(value, tuple):
          value = tuple(value)
          object.__setattr__(self, tuple_name, tuple(value))
        if len(value) != 3:
          raise ValueError(f"{tuple_name} must have 3 dims")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in value):
          raise ValueError(f"{tuple_name} dims must be non-negative ints")
    for name, value in (("lds_alloc", self.lds_alloc), ("waves", self.waves)):
      if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be non-negative int")
    if self.tile is not None:
      tile = tuple(self.tile)
      object.__setattr__(self, "tile", tile)
      if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in tile):
        raise ValueError("tile dims must be non-negative ints")
    if not isinstance(self.extra, dict):
      raise ValueError("extra must be object")

  def to_json(self) -> dict[str, Any]:
    out = {}
    for field in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "scratch_loads", "scratch_stores",
                  "spilled_vgprs", "spilled_sgprs", "occupancy", "grid", "workgroup_threads",
                  "workgroup", "lds_alloc", "waves", "tile"):
      value = getattr(self, field)
      if value is not None:
        out[field] = list(value) if isinstance(value, tuple) else value
    if self.extra:
      out["extra"] = dict(self.extra)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "ResourceSummary":
    if data is None:
      return ResourceSummary()
    if not isinstance(data, Mapping):
      raise ValueError("resources must be object")
    return ResourceSummary(
      vgpr=data.get("vgpr"), sgpr=data.get("sgpr"), lds_bytes=data.get("lds_bytes"),
      scratch_bytes=data.get("scratch_bytes"), occupancy=data.get("occupancy"),
      scratch_loads=data.get("scratch_loads"), scratch_stores=data.get("scratch_stores"),
      spilled_vgprs=data.get("spilled_vgprs"), spilled_sgprs=data.get("spilled_sgprs"),
      grid=tuple(data.get("grid", ())) if data.get("grid") else None,
      workgroup_threads=data.get("workgroup_threads"),
      workgroup=tuple(data.get("workgroup", ())) if data.get("workgroup") else None,
      lds_alloc=data.get("lds_alloc"), waves=data.get("waves"),
      tile=tuple(data.get("tile", ())) if data.get("tile") else None,
      extra=dict(data.get("extra", {})),
    )


_STRUCTURE_COUNTS = (
  "barriers", "waits", "tensor_ops", "branches", "predicates", "loops", "pipeline_stages",
  "instructions", "global_loads", "global_stores", "scalar_loads", "shared_loads", "shared_stores",
  "scratch_loads", "scratch_stores", "exits", "buffering",
)


@dataclass(frozen=True)
class StructureSummary:
  barriers: int | None = None
  waits: int | None = None
  tensor_ops: int | None = None
  branches: int | None = None
  predicates: int | None = None
  loops: int | None = None
  pipeline_stages: int | None = None
  instructions: int | None = None
  global_loads: int | None = None
  global_stores: int | None = None
  scalar_loads: int | None = None
  shared_loads: int | None = None
  shared_stores: int | None = None
  scratch_loads: int | None = None
  scratch_stores: int | None = None
  exits: int | None = None
  buffering: int | None = None
  operand_ownership: tuple[tuple[str, str], ...] = ()
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    for name in _STRUCTURE_COUNTS:
      value = getattr(self, name)
      if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
          raise ValueError(f"{name} must be non-negative int")
    ownership = tuple((str(k), str(v)) for k, v in self.operand_ownership)
    object.__setattr__(self, "operand_ownership", ownership)
    if not isinstance(self.extra, dict):
      raise ValueError("extra must be object")

  def to_json(self) -> dict[str, Any]:
    out = {}
    for name in _STRUCTURE_COUNTS:
      value = getattr(self, name)
      if value is not None:
        out[name] = value
    if self.operand_ownership:
      out["operand_ownership"] = {k: v for k, v in self.operand_ownership}
    if self.extra:
      out["extra"] = dict(self.extra)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "StructureSummary":
    if data is None:
      return StructureSummary()
    if not isinstance(data, Mapping):
      raise ValueError("structure must be object")
    ownership = tuple((str(k), str(v)) for k, v in dict(data.get("operand_ownership", {})).items())
    return StructureSummary(
      **{name: data.get(name) for name in _STRUCTURE_COUNTS},
      operand_ownership=ownership,
      extra=dict(data.get("extra", {})),
    )


@dataclass(frozen=True)
class HealthSummary:
  preflight: bool | None = None
  postflight: bool | None = None
  timeout_contained: bool | None = None
  device_available: bool | None = None
  recovery_attempted: bool | None = None
  recovered: bool | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    for name in ("preflight", "postflight", "timeout_contained", "device_available", "recovery_attempted", "recovered"):
      value = getattr(self, name)
      if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    if not isinstance(self.extra, dict):
      raise ValueError("extra must be object")

  def to_json(self) -> dict[str, Any]:
    out = {}
    for field in ("preflight", "postflight", "timeout_contained", "device_available", "recovery_attempted", "recovered"):
      value = getattr(self, field)
      if value is not None:
        out[field] = value
    if self.extra:
      out["extra"] = dict(self.extra)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "HealthSummary":
    if data is None:
      return HealthSummary()
    if not isinstance(data, Mapping):
      raise ValueError("health must be object")
    return HealthSummary(
      preflight=data.get("preflight"), postflight=data.get("postflight"), timeout_contained=data.get("timeout_contained"),
      device_available=data.get("device_available"), recovery_attempted=data.get("recovery_attempted"),
      recovered=data.get("recovered"), extra=dict(data.get("extra", {})),
    )


@dataclass(frozen=True)
class FactBlocker:
  code: str
  scope: str
  message: str = ""
  fields: tuple[str, ...] = ()

  def __post_init__(self):
    if not self.code or not self.scope:
      raise ValueError("blocker code and scope are required")
    if not isinstance(self.message, str):
      raise ValueError("message must be string")
    if not isinstance(self.fields, tuple):
      raise ValueError("fields must be tuple")

  def to_json(self) -> dict[str, Any]:
    out = {"code": self.code, "scope": self.scope}
    if self.message:
      out["message"] = self.message
    if self.fields:
      out["fields"] = list(self.fields)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "FactBlocker":
    if not isinstance(data, Mapping): raise ValueError("blocker must be object")
    return FactBlocker(data.get("code", "unknown"), data.get("scope", "general"), data.get("message", ""),
                       _json_tuple(data.get("fields", ())))


@dataclass(frozen=True)
class KernelEvidence:
  """Normalized per-candidate/variant evidence consumed by the kernel analysis stack."""

  candidate: CandidateModel
  identity: IdentityModel
  workload: WorkloadModel
  stages: KernelStages
  correctness: CorrectnessMetrics | None = None
  resources: ResourceSummary | None = None
  structure: StructureSummary | None = None
  operand_paths: tuple[tuple[str, OperandPathEvidence], ...] = ()
  timing: Measurement | None = None
  health: HealthSummary | None = None
  provenance: tuple[EvidenceSource, ...] = ()
  blockers: tuple[FactBlocker, ...] = ()

  def __post_init__(self):
    paths = tuple((str(k), v if isinstance(v, OperandPathEvidence) else OperandPathEvidence.from_json(v))
                  for k, v in self.operand_paths)
    if len({k for k, _ in paths}) != len(paths): raise ValueError("operand path keys must be unique")
    if any(k != v.operand_id for k, v in paths): raise ValueError("operand path key must equal operand_id")
    object.__setattr__(self, "operand_paths", paths)

  @property
  def evidence_id(self) -> str:
    payload = self.semantic_json()
    payload["provenance"] = [self._source_id(s) for s in self.provenance]
    payload.pop("evidence_id", None)
    return sha256_json(payload)

  @staticmethod
  def _source_id(source: EvidenceSource) -> dict[str, Any]:
    return {
      "producer": source.producer,
      "tool": source.tool,
      "fingerprint": source.fingerprint,
      "raw_summary": dict(source.raw_summary),
    }

  def to_json(self) -> dict[str, Any]:
    out = {
      "schema": SCHEMA_KERNEL_EVIDENCE,
      "evidence_id": self.evidence_id,
      "candidate": self.candidate.to_json(),
      "identity": self.identity.to_json(),
      "workload": self.workload.to_json(),
      "stages": self.stages.to_json(),
    }
    if self.correctness is not None:
      out["correctness"] = self.correctness.to_json()
    if self.resources is not None:
      out["resources"] = self.resources.to_json()
    if self.structure is not None:
      out["structure"] = self.structure.to_json()
    if self.operand_paths:
      out["operand_paths"] = {k: v.to_json() for k, v in self.operand_paths}
    if self.timing is not None:
      out["timing"] = self.timing.to_json()
    if self.health is not None:
      out["health"] = self.health.to_json()
    if self.provenance:
      out["provenance"] = [s.to_json() for s in self.provenance]
    if self.blockers:
      out["blockers"] = [b.to_json() for b in self.blockers]
    return out

  def semantic_json(self) -> dict[str, Any]:
    out = {
      "candidate": self.candidate.to_json(),
      "identity": self.identity.to_json(),
      "workload": self.workload.to_json(),
      "stages": self.stages.to_json(),
      "correctness": self.correctness.to_json() if self.correctness else {},
      "resources": self.resources.to_json() if self.resources else {},
      "structure": self.structure.to_json() if self.structure else {},
      "timing": self.timing.to_json() if self.timing else {},
      "health": self.health.to_json() if self.health else {},
    }
    if self.operand_paths:
      out["operand_paths"] = {k: v.to_json() for k, v in self.operand_paths}
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "KernelEvidence":
    if not isinstance(data, Mapping):
      raise ValueError("kernel evidence must be object")
    if data.get("schema") != SCHEMA_KERNEL_EVIDENCE:
      raise ValueError(f"expected schema {SCHEMA_KERNEL_EVIDENCE}")

    evidence = KernelEvidence(
      candidate=CandidateModel.from_json(data["candidate"]),
      identity=IdentityModel.from_json(dict(data.get("identity", {}))),
      workload=WorkloadModel.from_json(dict(data.get("workload", {}))),
      stages=KernelStages.from_json(data.get("stages", {})),
      correctness=CorrectnessMetrics.from_json(data["correctness"]) if data.get("correctness") is not None else None,
      resources=ResourceSummary.from_json(data["resources"]) if data.get("resources") is not None else None,
      structure=StructureSummary.from_json(data["structure"]) if data.get("structure") is not None else None,
      operand_paths=tuple((str(k), OperandPathEvidence.from_json(v)) for k, v in
                          dict(data.get("operand_paths", {})).items()),
      timing=Measurement.from_json(data["timing"]) if data.get("timing") is not None else None,
      health=HealthSummary.from_json(data["health"]) if data.get("health") is not None else None,
      provenance=_normalize_sources(data.get("provenance", ())),
      blockers=tuple(
        FactBlocker(b.get("code", "unknown"), b.get("scope", "general"), b.get("message", ""),
                    _json_tuple(b.get("fields", ())))
        for b in data.get("blockers", ()) if isinstance(b, Mapping)
      ),
    )
    if data.get("evidence_id") not in (None, evidence.evidence_id):
      raise ValueError("kernel evidence_id mismatch")
    return evidence

  @staticmethod
  def from_paths(*, payload: Mapping[str, Any]) -> "KernelEvidence":
    return KernelEvidence.from_json(payload)


@dataclass(frozen=True)
class KernelAnalysis:
  question: str
  candidates: tuple[KernelEvidence, ...]
  eligibility: dict[str, Any]
  comparability: dict[str, Any]
  contrasts: tuple[dict[str, Any], ...] = ()
  hypotheses: tuple[dict[str, Any], ...] = ()
  conclusions: tuple[dict[str, Any], ...] = ()
  unknowns: tuple[dict[str, Any], ...] = ()
  recommendation: dict[str, Any] | None = None
  evidence_refs: tuple[str, ...] = ()

  def __post_init__(self):
    if not isinstance(self.question, str) or not self.question.strip():
      raise ValueError("question is required")
    if not isinstance(self.candidates, tuple):
      raise ValueError("candidates must be tuple")
    if not isinstance(self.eligibility, dict) or not isinstance(self.comparability, dict):
      raise ValueError("eligibility and comparability must be objects")
    for field, value in (("contrasts", self.contrasts), ("hypotheses", self.hypotheses),
                        ("conclusions", self.conclusions), ("unknowns", self.unknowns)):
      if not isinstance(value, tuple):
        raise ValueError(f"{field} must be tuple")
    if self.recommendation is not None and not isinstance(self.recommendation, dict):
      raise ValueError("recommendation must be object")
    if not isinstance(self.evidence_refs, tuple):
      raise ValueError("evidence_refs must be tuple")

  @property
  def analysis_id(self) -> str:
    return sha256_json(self.semantic_json())

  def semantic_json(self) -> dict[str, Any]:
    return {
      "question": self.question,
      "candidates": [e.evidence_id for e in self.candidates],
      "eligibility": self.eligibility,
      "comparability": self.comparability,
      "contrasts": list(self.contrasts),
      "hypotheses": list(self.hypotheses),
      "conclusions": list(self.conclusions),
      "unknowns": list(self.unknowns),
      "recommendation": self.recommendation,
      "evidence_refs": list(self.evidence_refs),
    }

  def to_json(self) -> dict[str, Any]:
    out = {
      "schema": SCHEMA_KERNEL_ANALYSIS,
      "analysis_id": self.analysis_id,
      "question": self.question,
      "candidates": [e.to_json() for e in self.candidates],
      "eligibility": self.eligibility,
      "comparability": self.comparability,
      "contrasts": list(self.contrasts),
      "hypotheses": list(self.hypotheses),
      "conclusions": list(self.conclusions),
      "unknowns": list(self.unknowns),
      "evidence_refs": list(self.evidence_refs),
    }
    if self.recommendation is not None:
      out["recommendation"] = self.recommendation
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "KernelAnalysis":
    if data.get("schema") != SCHEMA_KERNEL_ANALYSIS:
      raise ValueError(f"expected schema {SCHEMA_KERNEL_ANALYSIS}")
    candidates = tuple(KernelEvidence.from_json(candidate) for candidate in data.get("candidates", ()))
    analysis = KernelAnalysis(
      question=data.get("question", ""),
      candidates=candidates,
      eligibility=dict(data.get("eligibility", {})),
      comparability=dict(data.get("comparability", {})),
      contrasts=tuple(dict(c) for c in data.get("contrasts", ())),
      hypotheses=tuple(dict(c) for c in data.get("hypotheses", ())),
      conclusions=tuple(dict(c) for c in data.get("conclusions", ())),
      unknowns=tuple(dict(c) for c in data.get("unknowns", ())),
      recommendation=(dict(data["recommendation"]) if isinstance(data.get("recommendation"), Mapping) else None),
      evidence_refs=_json_tuple(data.get("evidence_refs", ())),
    )
    if data.get("analysis_id") not in (None, analysis.analysis_id):
      raise ValueError("analysis_id mismatch")
    return analysis


@dataclass(frozen=True)
class KernelRecommendation:
  hypothesis: str
  candidate: str | None
  comparator: str | None = None
  variable: str | None = None
  invariants: dict[str, Any] = field(default_factory=dict)
  required_artifacts: tuple[str, ...] = ()
  workloads: tuple[Mapping[str, Any], ...] = ()
  seeds: tuple[Mapping[str, Any], ...] = ()
  expected_outcomes: tuple[str, ...] = ()
  stop_condition: str | None = None
  reopen_condition: str | None = None
  hints: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    if not isinstance(self.hypothesis, str) or not self.hypothesis:
      raise ValueError("hypothesis is required")
    if self.candidate is not None and not isinstance(self.candidate, str):
      raise ValueError("candidate must be string")
    if self.comparator is not None and not isinstance(self.comparator, str):
      raise ValueError("comparator must be string")
    if self.variable is not None and not isinstance(self.variable, str):
      raise ValueError("variable must be string")
    for field in ("invariants", "workloads", "seeds", "expected_outcomes", "hints"):
      if not isinstance(getattr(self, field), (dict, tuple)):
        raise ValueError(f"{field} must be object or tuple")

  @property
  def recommendation_id(self) -> str:
    return sha256_json(self._semantic_json())

  def _semantic_json(self) -> dict[str, Any]:
    payload = self._to_json_body()
    payload.pop("recommendation_id", None)
    payload.pop("schema", None)
    return payload

  @property
  def confidence(self) -> str:
    if self.stop_condition == "known" and self.reopen_condition == "expected":
      return "high"
    if self.reopen_condition is not None:
      return "medium"
    return "low"

  def to_json(self) -> dict[str, Any]:
    out = {
      "schema": SCHEMA_KERNEL_RECOMMENDATION,
      "recommendation_id": self.recommendation_id,
    }
    out.update(self._to_json_body())
    return out

  def _to_json_body(self) -> dict[str, Any]:
    out = {
      "hypothesis": self.hypothesis,
      "candidate": self.candidate,
      "comparator": self.comparator,
      "variable": self.variable,
      "confidence": self.confidence,
    }
    if self.invariants:
      out["invariants"] = dict(self.invariants)
    if self.required_artifacts:
      out["required_artifacts"] = list(self.required_artifacts)
    if self.workloads:
      out["workloads"] = [dict(w) for w in self.workloads]
    if self.seeds:
      out["seeds"] = [dict(s) for s in self.seeds]
    if self.expected_outcomes:
      out["expected_outcomes"] = list(self.expected_outcomes)
    if self.stop_condition is not None:
      out["stop_condition"] = self.stop_condition
    if self.reopen_condition is not None:
      out["reopen_condition"] = self.reopen_condition
    if self.hints:
      out["hints"] = dict(self.hints)
    return out

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "KernelRecommendation":
    if data.get("schema") != SCHEMA_KERNEL_RECOMMENDATION:
      raise ValueError(f"expected schema {SCHEMA_KERNEL_RECOMMENDATION}")
    recommendation = KernelRecommendation(
      hypothesis=data.get("hypothesis", ""),
      candidate=data.get("candidate"),
      comparator=data.get("comparator"),
      variable=data.get("variable"),
      invariants=dict(data.get("invariants", {})),
      required_artifacts=_json_tuple(data.get("required_artifacts", ())),
      workloads=tuple(dict(w) for w in data.get("workloads", ())),
      seeds=tuple(dict(s) for s in data.get("seeds", ())),
      expected_outcomes=_json_tuple(data.get("expected_outcomes", ())),
      stop_condition=data.get("stop_condition"),
      reopen_condition=data.get("reopen_condition"),
      hints=dict(data.get("hints", {})),
    )
    if data.get("recommendation_id") not in (None, recommendation.recommendation_id):
      raise ValueError("recommendation_id mismatch")
    return recommendation
