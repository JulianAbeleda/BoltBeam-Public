"""Evidence-request handoff: turn a recommendation or a model profile into a provider-executable request.

BoltBeam never runs a kernel; it *requests* the evidence a provider must produce, then ingests what
comes back. This module closes BoltBeam's half of the loop from two entry points:

  * from the GPU side — a `KernelRecommendation` (what the analyzer decided is missing) becomes an
    `ExperimentManifest` the provider runs to fill that exact gap;
  * from a GGUF — a `boltbeam.model_profile.v1` (roles/shapes/quant, no GPU) becomes one request per
    GEMM role describing the correctness+timing+health evidence to collect.

The provider stamps real identity (binary hash, system snapshot) when it executes; placeholders here
are explicit ("unbound") so a missing identity can never masquerade as a real one.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from boltbeam.core.canonical import sha256_hex
from boltbeam.core.experiment import ExperimentCommand, ExperimentManifest
from boltbeam.core.requirements import EvidenceRequirement, RequirementLevel
from boltbeam.kernel_analysis.model import KernelRecommendation
from boltbeam.kernel_analysis.candidates import CandidateSpec, FeasibilityResult
from boltbeam.kernel_analysis.eligibility import correctness_eligibility
from boltbeam.kernel_analysis.selection import MeasuredMatrixRanking
from boltbeam.ledger.model import CandidateDecision, EvidenceRef, status_for_verdict
from boltbeam.ledger.store import LedgerStore
from boltbeam.vocab import Guardrail, GuardrailState, Verdict
from boltbeam.vocab import SCHEMA_EXECUTION_BRIDGE_REQUEST

UNBOUND = "unbound"  # provider replaces at execution time
SCHEMA_OPERAND_REQUEST_SET = "boltbeam.operand_execution_request_set.v1"

# The standard evidence a correctness+performance comparison needs a provider to produce.
_STANDARD_REQUIREMENTS = (
  EvidenceRequirement("numeric_result", "correctness_metrics_present", RequirementLevel.REQUIRED.value,
                      ("comparison", "promotion"),
                      {"needs": ["element_count", "tolerance_abs", "tolerance_rel", "max_error", "finite_output"]}),
  EvidenceRequirement("timing_samples", "measured_timing_present", RequirementLevel.REQUIRED.value,
                      ("comparison", "promotion"), {"min_samples": 10}),
  EvidenceRequirement("gpu_health", "health_ok", RequirementLevel.RECOMMENDED.value, ("comparison",),
                      {"stages": ["preflight", "postrun"]}),
)

# What each recommendation variable most needs the provider to establish first.
_VARIABLE_REQUIREMENTS = {
  "execution_liveness": (
    EvidenceRequirement("execution_result", "execution_established", RequirementLevel.REQUIRED.value,
                        ("comparison", "diagnosis"), {}),
    EvidenceRequirement("gpu_health", "health_ok", RequirementLevel.REQUIRED.value, ("comparison",),
                        {"stages": ["preflight", "postrun"]}),
  ),
  "correctness": (_STANDARD_REQUIREMENTS[0], _STANDARD_REQUIREMENTS[2]),
  "experiment_identity": (
    EvidenceRequirement("identity_chain", "identity_consistent", RequirementLevel.REQUIRED.value,
                        ("comparison", "diagnosis", "promotion"),
                        {"needs": ["binary_hash", "executed_binary_hash", "experiment_id"]}),
  ),
}


def _shape_from_workload(workload: Mapping[str, Any]) -> dict[str, int]:
  shape = workload.get("shape", {}) if isinstance(workload, Mapping) else {}
  return {k: int(v) for k, v in shape.items() if isinstance(v, int) and not isinstance(v, bool)}


def _command(candidate_id: str, role: str, emit: str) -> ExperimentCommand:
  # A provider-neutral request template; the runner substitutes its own executable/args.
  return ExperimentCommand(
    argv=("PROVIDER_RUN", "--candidate", candidate_id, "--role", role or "unknown", "--emit", emit),
    env={},
  )


def request_from_recommendation(recommendation: KernelRecommendation, *,
                                backend: str = "provider-neutral",
                                system_snapshot_id: str = UNBOUND,
                                seed: int = 0, timeout_s: float | None = 60.0) -> ExperimentManifest:
  """Translate one recommendation into the provider-executable request that would satisfy it."""
  workload = recommendation.invariants.get("workload", {}) if recommendation.invariants else {}
  shape = _shape_from_workload(workload)
  variable = recommendation.variable or "factorial"
  requirements = _VARIABLE_REQUIREMENTS.get(variable, _STANDARD_REQUIREMENTS)
  emit = ",".join(r.requirement_id for r in requirements)
  return ExperimentManifest(
    system_snapshot_id=system_snapshot_id,
    candidate_id=recommendation.candidate or UNBOUND,
    comparator_id=recommendation.comparator or "none",
    backend=backend,
    shape=shape or {"unbound": 0},
    commands=(_command(recommendation.candidate or UNBOUND, str(workload.get("role", "")), emit),),
    requirements=requirements,
    workload=dict(workload) if isinstance(workload, Mapping) else {},
    seed=seed,
    timeout_s=timeout_s,
    policies={
      "intended_variable": recommendation.variable,
      "factorial": recommendation.hints.get("factorial") if recommendation.hints else None,
      "expected_outcomes": list(recommendation.expected_outcomes),
      "stop_condition": recommendation.stop_condition,
      "reopen_condition": recommendation.reopen_condition,
    },
  )


def _role_shape(role: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, int]:
  """Derive a GEMM shape from a model-profile role. rows/cols are the weight dims; n is the token/context
  dimension the provider will bind at run time (left symbolic here)."""
  rows = role.get("rows")
  cols = role.get("cols")
  shape: dict[str, int] = {}
  if isinstance(rows, int) and rows >= 0:
    shape["n"] = rows      # output feature dim
  if isinstance(cols, int) and cols >= 0:
    shape["k"] = cols      # contraction dim
  return shape


def request_from_model_profile(profile: Mapping[str, Any], *,
                               backend: str,
                               system_snapshot_id: str = UNBOUND,
                               context: int = 512,
                               gemm_roles: tuple[str, ...] = ("attn_qo", "attn_kv", "ffn_gate_up", "ffn_down"),
                               seed: int = 0, timeout_s: float | None = 60.0) -> list[ExperimentManifest]:
  """From a GGUF-derived model profile, emit one evidence-request manifest per GEMM role — no GPU needed.

  These describe *what the provider must produce* (correctness + timing + health) so that, once a GPU
  run fulfills them, the evidence flows straight back through investigate-kernels."""
  roles = profile.get("roles") or []
  seen: set[str] = set()
  manifests: list[ExperimentManifest] = []
  for role in roles:
    if not isinstance(role, Mapping):
      continue
    role_name = role.get("role")
    if role_name not in gemm_roles or role_name in seen:
      continue
    seen.add(role_name)
    shape = _role_shape(role, profile)
    shape["m"] = context  # token dimension for prefill
    quant = role.get("quant")
    candidate_id = f"{profile.get('model_id', 'model')}.{role_name}.m{context}"
    workload = {"operation": "gemm", "role": role_name, "shape": shape,
                "dtypes": {"input": quant} if quant else {}, "context": context}
    emit = ",".join(r.requirement_id for r in _STANDARD_REQUIREMENTS)
    manifests.append(ExperimentManifest(
      system_snapshot_id=system_snapshot_id,
      candidate_id=candidate_id,
      comparator_id=f"{candidate_id}.reference",
      backend=backend,
      shape=shape,
      commands=(_command(candidate_id, role_name, emit),),
      requirements=_STANDARD_REQUIREMENTS,
      workload=workload,
      seed=seed,
      timeout_s=timeout_s,
      policies={"source": "model_profile", "model_id": profile.get("model_id"), "quant": quant},
    ))
  return manifests


def _object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
  if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
    raise ValueError(f"{name} must be a string-keyed object")
  try:
    return json.loads(json.dumps(dict(value)))
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{name} must be JSON serializable") from exc


def _digest(value: Mapping[str, Any]) -> str:
  encoded = json.dumps(_object(value, "digest input"), sort_keys=True, separators=(",", ":")).encode()
  return sha256_hex(encoded)


def request_from_candidate(candidate: CandidateSpec, *, workload: Mapping[str, Any],
                           identity: Mapping[str, Any], semantic_abi: Mapping[str, Mapping[str, Any]],
                           protocol: Mapping[str, Any], fixed_invariants: Mapping[str, Any],
                           counter_groups: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
  """Serialize one feasible candidate into a validated provider execution request.

  This function describes compilation and execution requirements only. It does not import a provider,
  compile a kernel, dispatch work, or turn modeled ordering into a selection decision.
  """
  if not isinstance(candidate, CandidateSpec): raise TypeError("candidate must be CandidateSpec")
  workload_row, identity_row = _object(workload, "workload"), _object(identity, "identity")
  protocol_row = _object(protocol, "protocol")
  invariants_row = _object(fixed_invariants, "fixed_invariants")
  abi_row = _object(semantic_abi, "semantic_abi")
  operands = []
  for operand_id, transport in candidate.operand_transports:
    binding = abi_row.get(operand_id)
    if not isinstance(binding, dict): raise ValueError(f"semantic_abi missing operand {operand_id!r}")
    semantic_role, abi_argument = binding.get("semantic_role"), binding.get("abi_argument")
    if not isinstance(semantic_role, str) or not semantic_role: raise ValueError(f"semantic_role required for {operand_id!r}")
    if not isinstance(abi_argument, str) or not abi_argument: raise ValueError(f"abi_argument required for {operand_id!r}")
    operands.append({"operand_id": operand_id, "semantic_role": semantic_role, "abi_argument": abi_argument,
                     "declared_strategy": transport.declared_strategy,
                     "requirements": {"evidence": list(transport.requirements)}})
  required_identity = ("experiment_id", "comparator_id", "schedule_digest", "system_snapshot_id",
                       "reference_identity", "clock_state_id")
  missing = [name for name in required_identity if not isinstance(identity_row.get(name), str) or not identity_row[name]]
  if missing: raise ValueError(f"identity missing required fields: {', '.join(missing)}")
  compiler_context = identity_row.get("compiler")
  if not isinstance(compiler_context, dict) or not isinstance(compiler_context.get("adapter_id"), str) or not compiler_context["adapter_id"]:
    raise ValueError("identity.compiler.adapter_id is required for explicit provider execution")
  execution_transport = compiler_context.get("transport")
  if not isinstance(execution_transport, str) or not execution_transport:
    raise ValueError("identity.compiler.transport is required; transport is provider-owned, never inferred")
  target_id = identity_row.get("target_id")
  if not isinstance(target_id, str) or not target_id: raise ValueError("identity.target_id is required")
  correctness = protocol_row.get("correctness")
  guard, timing = protocol_row.get("guard"), protocol_row.get("timing")
  if not isinstance(correctness, dict) or correctness.get("scope", "full_output") != "full_output":
    raise ValueError("protocol.correctness must require full_output")
  if not isinstance(guard, dict) or not isinstance(timing, dict):
    raise ValueError("protocol requires guard and timing objects")
  request = {
    "schema": SCHEMA_EXECUTION_BRIDGE_REQUEST,
    "experiment_id": identity_row["experiment_id"], "candidate_id": candidate.candidate_id,
    "comparator_id": identity_row["comparator_id"],
    "workload_digest": identity_row.get("workload_digest") or _digest(workload_row),
    "schedule_digest": identity_row["schedule_digest"],
    "transport_plan": {"schema": "execution_bridge.transport_plan.v1", "transport": execution_transport,
                       "schedule_digest": identity_row["schedule_digest"], "operands": operands,
                       "requirements": {"candidate_id": candidate.candidate_id}},
    "target_context": {"target_id": target_id, "workload": workload_row,
                       "system_snapshot_id": identity_row["system_snapshot_id"],
                       "reference_identity": identity_row["reference_identity"],
                       "clock_state_id": identity_row["clock_state_id"]},
    "compiler_context": compiler_context,
    "candidate_knobs": candidate.factors.to_json(),
    "fixed_invariants": invariants_row,
    "artifacts": [{"kind": "final_isa_manifest", "required": True},
                  {"kind": "resource_summary", "required": True}],
    "counter_groups": [_object(group, "counter group") for group in counter_groups],
    "correctness": correctness, "guard": guard, "timing": timing,
  }
  from boltbeam.runtime.execution_bridge import parse_operand_execution_request
  return parse_operand_execution_request(request)


def requests_from_feasibility(results: Iterable[FeasibilityResult], *, workload: Mapping[str, Any],
                              identity: Mapping[str, Any], semantic_abi: Mapping[str, Mapping[str, Any]],
                              protocol: Mapping[str, Any], fixed_invariants: Mapping[str, Any],
                              counter_groups: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
  """Emit requests for feasible rows and typed reasons for every pruned row."""
  requests, infeasible = [], []
  for result in results:
    if not isinstance(result, FeasibilityResult): raise TypeError("results must contain FeasibilityResult values")
    if result.feasible:
      requests.append(request_from_candidate(result.candidate, workload=workload, identity=identity,
        semantic_abi=semantic_abi, protocol=protocol, fixed_invariants=fixed_invariants,
        counter_groups=counter_groups))
    else:
      infeasible.append({"candidate_id": result.candidate.candidate_id,
                         "reasons": [reason.to_json() for reason in result.reasons]})
  return {"schema": SCHEMA_OPERAND_REQUEST_SET, "requests": requests, "infeasible": infeasible,
          "selection_authority": "measured_results_only", "prediction_can_promote": False}


def recommendation_from_measured_matrix(ranking: MeasuredMatrixRanking, evidence: Sequence[Any], *,
                                        model_id: str, workload: str, rollback: Mapping[str, str],
                                        promotion_guardrails: Mapping[str, str],
                                        reopen_condition: str) -> tuple[CandidateDecision, dict[str, Any]]:
  """Build the reversible evaluator/ledger handoff from one measured cohort ranking."""
  if not isinstance(ranking, MeasuredMatrixRanking): raise TypeError("ranking must be MeasuredMatrixRanking")
  by_id = {row.candidate.candidate_id: row for row in evidence}
  eligible_challengers = [row.candidate_id for row in ranking.rows if row.eligible and row.candidate_id != ranking.baseline_id]
  if ranking.status == "promote": candidate_id, verdict = ranking.winner_id, Verdict.PROMOTE.value
  elif ranking.status == "retain":
    candidate_id, verdict = (eligible_challengers[0] if eligible_challengers else ranking.baseline_id), Verdict.REFUTE.value
  elif ranking.status == "inconclusive": candidate_id, verdict = ranking.baseline_id, Verdict.INCONCLUSIVE.value
  else: candidate_id, verdict = ranking.baseline_id, Verdict.DEFER.value
  if candidate_id not in by_id: raise ValueError("ranking candidate is absent from evidence")
  selected = by_id[candidate_id]
  target_id = selected.identity.target_id
  if not isinstance(target_id, str) or not target_id: raise ValueError("measured recommendation requires target identity")
  guardrails = dict(promotion_guardrails)
  guardrails[Guardrail.CORRECTNESS.value] = (GuardrailState.PASS.value if correctness_eligibility(selected).eligible
                                            else GuardrailState.FAIL.value)
  guardrails[Guardrail.ROUTE_BOUND.value] = (GuardrailState.PASS.value if selected.identity.binary_hash is not None and
    selected.identity.binary_hash == selected.identity.executed_binary_hash else GuardrailState.FAIL.value)
  guardrails[Guardrail.SPEED.value] = GuardrailState.PASS.value if ranking.status == "promote" else GuardrailState.FAIL.value
  guardrails[Guardrail.ROLLBACK.value] = GuardrailState.PASS.value if rollback else GuardrailState.FAIL.value
  if verdict == Verdict.PROMOTE.value:
    required = {Guardrail.CORRECTNESS.value, Guardrail.ROUTE_BOUND.value, Guardrail.SPEED.value,
      Guardrail.MEMORY_FIT.value, Guardrail.FALLBACK.value, Guardrail.DETERMINISM.value, Guardrail.ROLLBACK.value}
    failed = sorted(key for key in required if guardrails.get(key) != GuardrailState.PASS.value)
    if failed: raise ValueError(f"promotion guardrails are incomplete: {', '.join(failed)}")
  refs = tuple(EvidenceRef(selected.evidence_id, source.path, source.tool, source.fingerprint,
                           f"measured {ranking.status} in session {ranking.session_id}")
               for source in selected.provenance)
  reason = {"promote": "measured matrix winner with disjoint uncertainty",
            "retain": "measured baseline retained", "inconclusive": "top timing intervals overlap",
            "blocked": "measured matrix is blocked", "unsupported": "measured matrix is unsupported"}[ranking.status]
  decision = CandidateDecision(candidate_id, model_id, target_id, workload, verdict, reason=reason,
    evidence=refs, rollback=dict(rollback), next_action=reopen_condition, guardrails=guardrails)
  scope = {"model": model_id, "target": target_id, "workload": workload,
           "role": selected.workload.role, "shape": dict(selected.workload.shape),
           "dtypes": dict(selected.workload.dtypes), "context": selected.workload.context,
           "system_snapshot_id": selected.identity.system_snapshot_id, "clock_state_id": selected.identity.clock_state_id,
           "candidate_binary_hash": selected.identity.binary_hash, "session_id": selected.identity.session_id,
           "rollback": dict(rollback), "reopen_condition": reopen_condition,
           "invalidated_by": ["compiler_change", "binary_change", "target_change", "workload_change", "correctness_failure"]}
  return decision, scope


def record_measured_recommendation(decision: CandidateDecision, scope: Mapping[str, Any], *,
                                   ledger_path: str | Path, reopen_condition: str) -> dict[str, Any] | None:
  """Append only durable measured outcomes; inconclusive analysis never mutates the ledger."""
  if status_for_verdict(decision.verdict) is None: return None
  return LedgerStore(ledger_path).add(decision, scope=dict(scope), reopen_condition=reopen_condition).to_json()
