"""Validation for existing pinned pure-prefill authority artifacts."""
from __future__ import annotations
from typing import Any

def validate_typed_pipe_fixture(fixture:dict[str, Any]) -> dict[str, Any]:
  """Validate declarative cross-repo status; never upgrades host data to execution."""
  status = fixture.get("status")
  provenance = fixture.get("provenance")
  errors = []
  if status not in {"typed_host_only", "executed_pure"}: errors.append("status")
  if provenance != "compiler_owned_typed_pipe_ir": errors.append("provenance")
  if status == "executed_pure" and not fixture.get("execution_evidence"):
    errors.append("execution_evidence")
  return {"passed": not errors, "status": status, "provenance": provenance, "errors": errors,
          "execution_claim": status == "executed_pure" and not errors}

def validate_prefill_authority(report:dict[str, Any], *, policy:str, require_sweep:bool=False) -> dict[str, Any]:
  errors:list[str] = []
  if report.get("schema") != "prefill-whole-synced-authority.v1": errors.append("schema")
  if report.get("K") != 8 or report.get("warmups") != 4 or report.get("rounds") != 3: errors.append("protocol")
  if not (report.get("pin_clock") or {}).get("ok"): errors.append("clock_pin")
  route = report.get("route_attribution") or {}
  if route.get("prefill_route_pure") is not True: errors.append("not_pure")
  if route.get("prefill_route_rolled_back") is not False: errors.append("rollback")
  expected = {"gate_up_only": {"ffn_gate_up"}, "all_four": {"ffn_gate_up", "ffn_down", "attn_qo", "attn_kv"}, "s9": None}.get(policy)
  census = report.get("candidate_set_route_census") or {}
  if policy != "s9":
    selected = {x.get("role") for x in census.get("selected", [])}
    if census.get("passed") is not True or selected != expected: errors.append("candidate_census")
  if require_sweep and not {"512", "1024", "2048", "4096"}.issubset((report.get("whole_tok_s") or {}).keys()): errors.append("context_sweep")
  return {"passed": not errors, "policy": policy, "errors": errors,
          "ctx512_present": "512" in (report.get("whole_tok_s") or {}),
          "rollback": route.get("prefill_route_rolled_back"),
          "selected_roles": sorted(x.get("role") for x in census.get("selected", []))}
