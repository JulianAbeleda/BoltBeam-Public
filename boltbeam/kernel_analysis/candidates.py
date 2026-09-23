"""Provider-neutral operand-placement candidate generation, pruning, and modeled ordering."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from boltbeam.kernel_analysis.lds_cache import PLACEMENT_MATRIX
from boltbeam.kernel_analysis.model import OperandTransport
from boltbeam.kernel_analysis.theoretical_roofline import registered_memory_tiers, roofline_interval_ms


class RejectionCode(str, Enum):
  REGISTER_BUDGET = "register_budget_exceeded"
  VGPR_BUDGET = "vgpr_budget_exceeded"
  SGPR_BUDGET = "sgpr_budget_exceeded"
  LDS_BUDGET = "lds_budget_exceeded"
  OCCUPANCY = "occupancy_below_floor"
  SPILL_RISK = "spill_risk"
  GEOMETRY = "illegal_geometry"
  ALIGNMENT = "vector_alignment_unsupported"
  MATRIX_LAYOUT = "matrix_fragment_layout_unsupported"
  SYNCHRONIZATION = "synchronization_unsupported"
  CAPABILITY = "provider_capability_missing"
  REUSE_WINDOW = "reuse_window_ineligible"


@dataclass(frozen=True)
class RejectionReason:
  code: RejectionCode
  detail: str
  operand_id: str | None = None

  def to_json(self) -> dict[str, str]:
    return {k: v for k, v in {"code": self.code.value, "detail": self.detail,
                              "operand_id": self.operand_id}.items() if v is not None}

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "RejectionReason":
    return RejectionReason(RejectionCode(data["code"]), data["detail"], data.get("operand_id"))


@dataclass(frozen=True)
class OperandEstimate:
  fragment_register_bytes: int = 0
  lds_bytes: int = 0
  compulsory_global_bytes: int = 0
  repeated_global_bytes: int = 0
  lds_traffic_bytes: int = 0
  reuse: float = 1.0
  alignment_bytes: int | None = None
  required_matrix_layout: str | None = None
  spill_risk: bool = False

  def __post_init__(self):
    values = (self.fragment_register_bytes, self.lds_bytes, self.compulsory_global_bytes,
              self.repeated_global_bytes, self.lds_traffic_bytes, self.reuse)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
      raise ValueError("operand resource and traffic estimates must be non-negative")
    if self.alignment_bytes is not None and (isinstance(self.alignment_bytes, bool) or
        not isinstance(self.alignment_bytes, int) or self.alignment_bytes <= 0):
      raise ValueError("alignment_bytes must be a positive int")
    if not isinstance(self.spill_risk, bool): raise ValueError("spill_risk must be bool")


@dataclass(frozen=True)
class CandidateFactors:
  tile: tuple[int, int, int] | None = None
  workgroup: tuple[int, int, int] | None = None
  waves_per_workgroup: int | None = None
  pipeline_stages: int = 1
  vector_width_bytes: int | None = None
  cache_policy: str | None = None
  accumulator_layout: str | None = None
  entangled: tuple[str, ...] = ()

  def __post_init__(self):
    for name, value in (("tile", self.tile), ("workgroup", self.workgroup)):
      if value is not None and (not isinstance(value, tuple) or len(value) != 3 or
                                any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value)):
        raise ValueError(f"{name} must contain three positive dimensions")
    for name in ("waves_per_workgroup", "pipeline_stages", "vector_width_bytes"):
      value = getattr(self, name)
      if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"{name} must be a positive int")
    if any(not isinstance(v, str) or not v for v in self.entangled): raise ValueError("entangled factors must be strings")

  def to_json(self) -> dict[str, Any]:
    return {k: v for k, v in {
      "tile": list(self.tile) if self.tile else None, "workgroup": list(self.workgroup) if self.workgroup else None,
      "waves_per_workgroup": self.waves_per_workgroup, "pipeline_stages": self.pipeline_stages,
      "vector_width_bytes": self.vector_width_bytes, "cache_policy": self.cache_policy,
      "accumulator_layout": self.accumulator_layout, "entangled": list(self.entangled),
    }.items() if v is not None}

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "CandidateFactors":
    return CandidateFactors(
      tile=tuple(data["tile"]) if data.get("tile") else None,
      workgroup=tuple(data["workgroup"]) if data.get("workgroup") else None,
      waves_per_workgroup=data.get("waves_per_workgroup"), pipeline_stages=data.get("pipeline_stages", 1),
      vector_width_bytes=data.get("vector_width_bytes"), cache_policy=data.get("cache_policy"),
      accumulator_layout=data.get("accumulator_layout"), entangled=tuple(data.get("entangled", ())))


@dataclass(frozen=True)
class CandidateSpec:
  candidate_id: str
  operand_transports: tuple[tuple[str, OperandTransport], ...]
  factors: CandidateFactors = field(default_factory=CandidateFactors)
  diagnostic: bool = False
  required_capabilities: tuple[str, ...] = ()

  def __post_init__(self):
    if not isinstance(self.candidate_id, str) or not self.candidate_id: raise ValueError("candidate_id is required")
    if not isinstance(self.operand_transports, tuple) or not self.operand_transports:
      raise ValueError("operand_transports must be a non-empty tuple")
    if any(not isinstance(op, str) or not op or not isinstance(tr, OperandTransport)
           for op, tr in self.operand_transports): raise ValueError("invalid operand transport entry")
    if len({op for op, _ in self.operand_transports}) != len(self.operand_transports):
      raise ValueError("operand transport ids must be unique")
    if not isinstance(self.factors, CandidateFactors): raise ValueError("factors must be CandidateFactors")
    if not isinstance(self.diagnostic, bool): raise ValueError("diagnostic must be bool")
    if any(not isinstance(v, str) or not v for v in self.required_capabilities):
      raise ValueError("required capabilities must be strings")

  @property
  def strategies(self) -> dict[str, str]:
    return {operand: str(transport.declared_strategy) for operand, transport in self.operand_transports}

  def to_json(self) -> dict[str, Any]:
    return {"candidate_id": self.candidate_id,
            "operand_transports": {k: v.to_json() for k, v in self.operand_transports},
            "factors": self.factors.to_json(), "diagnostic": self.diagnostic,
            "required_capabilities": list(self.required_capabilities)}

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "CandidateSpec":
    operand_transports = tuple((operand, OperandTransport.from_json(transport))
      for operand, transport in data.get("operand_transports", {}).items())
    return CandidateSpec(data["candidate_id"], operand_transports, CandidateFactors.from_json(data.get("factors", {})),
      data.get("diagnostic", False), tuple(data.get("required_capabilities", ())))


@dataclass(frozen=True)
class FeasibilityLimits:
  register_bytes_per_workgroup: int | None = None
  predicted_vgpr_per_thread: int | None = None
  predicted_sgpr_per_wave: int | None = None
  max_vgpr_per_thread: int | None = None
  max_sgpr_per_wave: int | None = None
  base_lds_bytes: int = 0
  lds_allocation_granularity: int = 1
  occupancy_floor_waves: int = 1
  predicted_occupancy_waves: int | None = None

  def __post_init__(self):
    for name in ("register_bytes_per_workgroup", "predicted_vgpr_per_thread", "predicted_sgpr_per_wave",
                 "max_vgpr_per_thread", "max_sgpr_per_wave", "base_lds_bytes", "occupancy_floor_waves",
                 "predicted_occupancy_waves"):
      value = getattr(self, name)
      if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a non-negative int")
    if isinstance(self.lds_allocation_granularity, bool) or not isinstance(self.lds_allocation_granularity, int) \
        or self.lds_allocation_granularity <= 0:
      raise ValueError("lds_allocation_granularity must be a positive int")


@dataclass(frozen=True)
class FeasibilityResult:
  candidate: CandidateSpec
  feasible: bool
  reasons: tuple[RejectionReason, ...]
  register_bytes: int
  lds_bytes: int
  predicted_occupancy_waves: int | None
  entangled_factors: tuple[str, ...]

  def to_json(self) -> dict[str, Any]:
    return {"candidate": self.candidate.to_json(), "feasible": self.feasible,
            "reasons": [r.to_json() for r in self.reasons], "register_bytes": self.register_bytes,
            "lds_bytes": self.lds_bytes, "predicted_occupancy_waves": self.predicted_occupancy_waves,
            "entangled_factors": list(self.entangled_factors)}

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "FeasibilityResult":
    return FeasibilityResult(
      CandidateSpec.from_json(data.get("candidate", {})), data.get("feasible", False),
      tuple(RejectionReason.from_json(r) for r in data.get("reasons", ())),
      data.get("register_bytes", 0), data.get("lds_bytes", 0), data.get("predicted_occupancy_waves"),
      tuple(data.get("entangled_factors", ())))


@dataclass(frozen=True)
class CandidatePrediction:
  candidate: CandidateSpec
  interval_ms: tuple[float, float]
  assumptions: tuple[str, ...]
  truth_status: str = "modeled"

  def __post_init__(self):
    if self.truth_status == "measured": raise ValueError("candidate ordering cannot contain measured truth")
    if (not isinstance(self.interval_ms, tuple) or len(self.interval_ms) != 2 or
        any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in self.interval_ms) or
        self.interval_ms[0] < 0 or self.interval_ms[1] < self.interval_ms[0]):
      raise ValueError("invalid prediction interval")

  def to_json(self) -> dict[str, Any]:
    return {"candidate_id": self.candidate.candidate_id, "interval_ms": list(self.interval_ms),
            "assumptions": list(self.assumptions), "truth_status": self.truth_status, "prediction_only": True}


def build_candidate_matrix(*, operand_ids: tuple[str, str] = ("a", "b"),
                           factors: CandidateFactors | None = None,
                           diagnostic_strategies: Iterable[str] = ()) -> tuple[CandidateSpec, ...]:
  """Materialize the authoritative four placements plus explicitly requested diagnostic controls."""
  if len(operand_ids) != 2 or operand_ids[0] == operand_ids[1]: raise ValueError("two distinct operand_ids required")
  factors, out = factors or CandidateFactors(), []
  for row in PLACEMENT_MATRIX:
    remapped = tuple((operand_ids[i], transport) for i, (_, transport) in enumerate(row))
    name = "-".join(f"{op}-{tr.declared_strategy}" for op, tr in remapped)
    out.append(CandidateSpec(name, remapped, factors))
  requested = tuple(dict.fromkeys(diagnostic_strategies))
  illegal = set(requested) - {"cache_streamed", "reloaded"}
  if illegal: raise ValueError(f"unsupported diagnostic strategies: {sorted(illegal)}")
  for strategy in requested:
    transports = tuple((op, OperandTransport(declared_strategy=strategy)) for op in operand_ids)
    out.append(CandidateSpec(f"diagnostic-{strategy}", transports, factors, diagnostic=True))
  return tuple(out)


def prune_candidate(candidate: CandidateSpec, operands: Mapping[str, OperandEstimate], *, target,
                    limits: FeasibilityLimits = FeasibilityLimits(),
                    capabilities: Mapping[str, Any] | None = None) -> FeasibilityResult:
  """Reject statically impossible requests. Unknown limits remain unknown, not invented failures."""
  target_capabilities = getattr(target, "capabilities", None) or {}
  capabilities, reasons = {**target_capabilities, **(capabilities or {})}, []
  register_bytes, raw_lds = 0, limits.base_lds_bytes
  for operand, strategy in candidate.strategies.items():
    if operand not in operands: raise ValueError(f"missing estimate for operand {operand!r}")
    estimate = operands[operand]
    if strategy == "register_resident":
      register_bytes += estimate.fragment_register_bytes
      if estimate.spill_risk: reasons.append(RejectionReason(RejectionCode.SPILL_RISK, "operand estimate predicts spilling", operand))
    if strategy == "lds_staged":
      raw_lds += estimate.lds_bytes * candidate.factors.pipeline_stages
      if estimate.reuse <= 1: reasons.append(RejectionReason(RejectionCode.REUSE_WINDOW, "LDS staging has no reuse window", operand))
      if capabilities.get("synchronization", True) is False:
        reasons.append(RejectionReason(RejectionCode.SYNCHRONIZATION, "LDS staging requires synchronization", operand))
    layouts = capabilities.get("matrix_fragment_layouts")
    if estimate.required_matrix_layout and layouts is not None and estimate.required_matrix_layout not in layouts:
      reasons.append(RejectionReason(RejectionCode.MATRIX_LAYOUT, f"layout {estimate.required_matrix_layout!r} unavailable", operand))
    width = candidate.factors.vector_width_bytes
    if width and estimate.alignment_bytes is not None and estimate.alignment_bytes < width:
      reasons.append(RejectionReason(RejectionCode.ALIGNMENT, f"{width}-byte vector requires stronger alignment", operand))
  granule = max(1, limits.lds_allocation_granularity)
  lds_bytes = math.ceil(raw_lds / granule) * granule
  if limits.register_bytes_per_workgroup is not None and register_bytes > limits.register_bytes_per_workgroup:
    reasons.append(RejectionReason(RejectionCode.REGISTER_BUDGET, f"{register_bytes} > {limits.register_bytes_per_workgroup}"))
  if (limits.predicted_vgpr_per_thread is not None and limits.max_vgpr_per_thread is not None
      and limits.predicted_vgpr_per_thread > limits.max_vgpr_per_thread):
    reasons.append(RejectionReason(RejectionCode.VGPR_BUDGET,
      f"{limits.predicted_vgpr_per_thread} > {limits.max_vgpr_per_thread} VGPR/thread"))
  if (limits.predicted_sgpr_per_wave is not None and limits.max_sgpr_per_wave is not None
      and limits.predicted_sgpr_per_wave > limits.max_sgpr_per_wave):
    reasons.append(RejectionReason(RejectionCode.SGPR_BUDGET,
      f"{limits.predicted_sgpr_per_wave} > {limits.max_sgpr_per_wave} SGPR/wave"))
  if target.lds_bytes_per_cu is not None and lds_bytes > target.lds_bytes_per_cu:
    reasons.append(RejectionReason(RejectionCode.LDS_BUDGET, f"{lds_bytes} > {target.lds_bytes_per_cu}"))
  occupancy = limits.predicted_occupancy_waves
  if occupancy is not None and occupancy < limits.occupancy_floor_waves:
    reasons.append(RejectionReason(RejectionCode.OCCUPANCY, f"{occupancy} < {limits.occupancy_floor_waves} waves"))
  threads = math.prod(candidate.factors.workgroup) if candidate.factors.workgroup else None
  if threads is not None:
    if capabilities.get("max_workgroup_threads") is not None and threads > capabilities["max_workgroup_threads"]:
      reasons.append(RejectionReason(RejectionCode.GEOMETRY, f"{threads} > {capabilities['max_workgroup_threads']} workgroup threads"))
    if threads % target.wave_size: reasons.append(RejectionReason(RejectionCode.GEOMETRY, f"{threads} threads is not wave-size aligned"))
  for requirement in candidate.required_capabilities:
    if capabilities.get(requirement) is not True:
      reasons.append(RejectionReason(RejectionCode.CAPABILITY, f"missing capability {requirement!r}"))
  supported = capabilities.get("operand_strategies")
  if supported is not None:
    for operand, strategy in candidate.strategies.items():
      if strategy not in supported:
        reasons.append(RejectionReason(RejectionCode.CAPABILITY,
          f"operand strategy {strategy!r} is unsupported", operand))
  return FeasibilityResult(candidate, not reasons, tuple(reasons), register_bytes, lds_bytes,
                           occupancy, candidate.factors.entangled)


def prune_candidates(candidates: Iterable[CandidateSpec], operands: Mapping[str, OperandEstimate], **kwargs) -> tuple[FeasibilityResult, ...]:
  return tuple(prune_candidate(candidate, operands, **kwargs) for candidate in candidates)


def predict_and_order(feasible: Iterable[FeasibilityResult], operands: Mapping[str, OperandEstimate], *,
                      target, peak_flops: float, flops: float, bandwidth_relative_uncertainty: float = .1,
                      barrier_s: float = 0.0) -> tuple[CandidatePrediction, ...]:
  """Order feasible compile requests by modeled intervals; this API has no selection outcome."""
  tiers, predictions = registered_memory_tiers(target), []
  for result in feasible:
    if not result.feasible: continue
    global_bytes, lds_bytes = 0, 0
    assumptions = ["roofline lower bound", "serving tier inferred from target capacity"]
    for operand, strategy in result.candidate.strategies.items():
      estimate = operands[operand]
      global_bytes += estimate.compulsory_global_bytes
      if strategy in ("cache_streamed", "reloaded"): global_bytes += estimate.repeated_global_bytes
      if strategy == "lds_staged": lds_bytes += estimate.lds_traffic_bytes
    global_iv = roofline_interval_ms(flops=flops, bytes_moved=global_bytes, peak_flops=peak_flops,
      bandwidth_gbs=tiers["dram_gbs"], bandwidth_relative_uncertainty=bandwidth_relative_uncertainty,
      fixed_cost_s=barrier_s if lds_bytes else 0.0)
    lds_iv = (0.0, 0.0)
    if lds_bytes:
      lds_iv = roofline_interval_ms(flops=0, bytes_moved=lds_bytes, peak_flops=peak_flops,
        bandwidth_gbs=tiers["lds_gbs"], bandwidth_relative_uncertainty=bandwidth_relative_uncertainty)
      assumptions.append("LDS and global costs do not overlap")
    predictions.append(CandidatePrediction(result.candidate,
      (global_iv[0] + lds_iv[0], global_iv[1] + lds_iv[1]), tuple(assumptions)))
  return tuple(sorted(predictions, key=lambda p: (sum(p.interval_ms), p.candidate.candidate_id)))


__all__ = ["CandidateFactors", "CandidatePrediction", "CandidateSpec", "FeasibilityLimits", "FeasibilityResult",
           "OperandEstimate", "RejectionCode", "RejectionReason", "build_candidate_matrix", "predict_and_order",
           "prune_candidate", "prune_candidates"]
