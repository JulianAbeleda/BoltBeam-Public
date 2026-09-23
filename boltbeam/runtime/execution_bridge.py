"""Provider-neutral validation for operand execution bridge payloads.

This module deliberately has no workflow or runtime imports so request producers,
runner plans, and external providers share one import-safe contract boundary.
"""
from __future__ import annotations

import json, math
from typing import Any

from boltbeam.vocab import SCHEMA_EXECUTION_BRIDGE_REQUEST, SCHEMA_EXECUTION_BRIDGE_RESULT

_OPERAND_STRATEGIES = frozenset(("register_resident", "lds_staged", "cache_streamed", "reloaded", "unknown"))
_RESULT_STATUSES = frozenset(("not_attempted", "passed", "failed", "timed_out", "unsupported", "blocked"))


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
  if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
    raise ValueError(f"{name} must be a string-keyed object")
  try: return json.loads(json.dumps(value))
  except (TypeError, ValueError) as exc: raise ValueError(f"{name} must be JSON serializable") from exc


def _required_text(row: dict[str, Any], name: str) -> str:
  value = row.get(name)
  if not isinstance(value, str) or not value.strip(): raise ValueError(f"{name} must be a non-empty string")
  return value


def _non_negative_number(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
    raise ValueError(f"{name} must be a finite non-negative number")
  return float(value)


def parse_operand_execution_request(payload: dict[str, Any]) -> dict[str, Any]:
  """Validate a provider-neutral operand request without importing a runtime."""
  row = _json_mapping(payload, "operand execution request")
  if row.get("schema") != SCHEMA_EXECUTION_BRIDGE_REQUEST: raise ValueError("unsupported operand execution request schema")
  for name in ("experiment_id", "candidate_id", "comparator_id", "workload_digest", "schedule_digest"):
    _required_text(row, name)
  for name in ("target_context", "compiler_context", "candidate_knobs", "fixed_invariants"):
    _json_mapping(row.get(name, {}), name)
  plan = _json_mapping(row.get("transport_plan"), "transport_plan")
  if plan.get("schema") not in (None, "execution_bridge.transport_plan.v1"): raise ValueError("unsupported transport_plan schema")
  _required_text(plan, "transport")
  if _required_text(plan, "schedule_digest") != row["schedule_digest"]:
    raise ValueError("transport_plan schedule_digest must match request schedule_digest")
  operands = plan.get("operands", [])
  if not isinstance(operands, list) or not operands: raise ValueError("transport_plan.operands must be a non-empty list")
  ids: set[str] = set(); abi_arguments: set[str] = set()
  for operand in operands:
    op = _json_mapping(operand, "operand"); operand_id = _required_text(op, "operand_id")
    if operand_id in ids: raise ValueError(f"duplicate operand_id: {operand_id}")
    ids.add(operand_id); _required_text(op, "semantic_role"); abi_argument = _required_text(op, "abi_argument")
    if abi_argument in abi_arguments: raise ValueError(f"duplicate abi_argument: {abi_argument}")
    abi_arguments.add(abi_argument)
    if op.get("declared_strategy") not in _OPERAND_STRATEGIES: raise ValueError("invalid declared operand strategy")
    _json_mapping(op.get("requirements", {}), "operand requirements")
  artifacts = row.get("artifacts", [])
  if not isinstance(artifacts, list): raise ValueError("artifacts must be a list")
  artifact_kinds: set[str] = set()
  for value in artifacts:
    artifact = _json_mapping(value, "artifact request"); kind = _required_text(artifact, "kind")
    if kind in artifact_kinds: raise ValueError(f"duplicate artifact kind: {kind}")
    artifact_kinds.add(kind)
    if "required" in artifact and not isinstance(artifact["required"], bool): raise ValueError("artifact.required must be bool")
  groups = row.get("counter_groups", [])
  if not isinstance(groups, list): raise ValueError("counter_groups must be a list")
  group_ids: set[str] = set()
  for value in groups:
    group = _json_mapping(value, "counter group"); group_id = _required_text(group, "group_id")
    if group_id in group_ids: raise ValueError(f"duplicate counter group: {group_id}")
    group_ids.add(group_id); counters = group.get("counters")
    if not isinstance(counters, list) or not counters or any(not isinstance(v, str) or not v for v in counters):
      raise ValueError("counter group counters must be a non-empty string list")
    if len(counters) != len(set(counters)): raise ValueError("counter group counters must be unique")
    for flag in ("optional_when_unsupported", "separate_pass"):
      if flag in group and not isinstance(group[flag], bool): raise ValueError(f"counter group {flag} must be bool")
  correctness = row.get("correctness")
  if correctness is not None:
    correctness = _json_mapping(correctness, "correctness"); _required_text(correctness, "oracle")
    if correctness.get("scope", "full_output") != "full_output": raise ValueError("correctness scope must be full_output")
    for field_name in ("atol", "rtol"): _non_negative_number(correctness.get(field_name, 0.0), f"correctness.{field_name}")
  guard = row.get("guard")
  if guard is not None:
    guard = _json_mapping(guard, "guard")
    if isinstance(guard.get("hard_timeout_ms"), bool) or not isinstance(guard.get("hard_timeout_ms"), int) or guard["hard_timeout_ms"] <= 0:
      raise ValueError("hard_timeout_ms must be positive")
    for flag in ("process_isolation", "guard_buffers", "health_preflight", "health_postflight", "nonconstant_inputs"):
      if flag in guard and not isinstance(guard[flag], bool): raise ValueError(f"guard.{flag} must be bool")
  timing = row.get("timing")
  if timing is not None:
    timing = _json_mapping(timing, "timing")
    if isinstance(timing.get("rounds"), bool) or not isinstance(timing.get("rounds"), int) or timing["rounds"] <= 0:
      raise ValueError("timing rounds must be positive")
    if isinstance(timing.get("warmups", 0), bool) or not isinstance(timing.get("warmups", 0), int) or timing.get("warmups", 0) < 0:
      raise ValueError("timing warmups must be non-negative")
    if isinstance(timing.get("randomization_seed"), bool) or not isinstance(timing.get("randomization_seed"), int):
      raise ValueError("timing randomization_seed must be int")
    _non_negative_number(timing.get("noise_threshold", 0.0), "timing.noise_threshold")
  return row


def parse_operand_execution_result(payload: dict[str, Any]) -> dict[str, Any]:
  """Validate typed phase outcomes while retaining provider extension fields."""
  row = _json_mapping(payload, "operand execution result")
  if row.get("schema") != SCHEMA_EXECUTION_BRIDGE_RESULT: raise ValueError("unsupported operand execution result schema")
  for name in ("experiment_id", "candidate_id", "request_digest"): _required_text(row, name)
  phases = row.get("phases")
  if not isinstance(phases, list) or not phases: raise ValueError("phases must be a non-empty list")
  names: set[str] = set()
  for phase_value in phases:
    phase = _json_mapping(phase_value, "phase result"); name = _required_text(phase, "phase")
    if name in names: raise ValueError(f"duplicate result phase: {name}")
    names.add(name); status = phase.get("status")
    if status not in _RESULT_STATUSES: raise ValueError("invalid phase result status")
    unsupported = phase.get("unsupported", [])
    if not isinstance(unsupported, list): raise ValueError("unsupported outcomes must be a list")
    if status == "unsupported" and not unsupported: raise ValueError("unsupported status requires a typed outcome")
    for outcome_value in unsupported:
      outcome = _json_mapping(outcome_value, "unsupported outcome")
      if outcome.get("status", "unsupported") != "unsupported": raise ValueError("invalid unsupported outcome status")
      for field_name in ("reason", "phase", "feature"): _required_text(outcome, field_name)
  return row


__all__ = ["parse_operand_execution_request", "parse_operand_execution_result"]
