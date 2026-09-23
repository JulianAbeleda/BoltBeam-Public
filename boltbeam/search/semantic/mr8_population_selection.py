"""Fail-closed MR7-to-MR8 selection of exact semantic search populations.

MR7 owns which role families may advance. Tinygrad owns exact workload export,
and the existing semantic population builder owns candidate expansion. This
module only verifies those artifacts and joins them without inventing policy.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import pathlib
import re
from typing import Any

from boltbeam.search.role_cost_ranking import SCHEMA as MR7_SCHEMA
from boltbeam.search.semantic.semantic_identity import (canonical_json, content_sha256, normalize_semantic_identity,
                                                validate_exact_semantic_workload)
from boltbeam.search.semantic.semantic_population_export import export_population


REQUEST_SCHEMA = "boltbeam.mr8_population_selection_request.v1"
SCHEMA = "boltbeam.mr8_population_selection.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _selected_role_rows(ranking: Mapping[str, Any]) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
  if ranking.get("schema") != MR7_SCHEMA or ranking.get("status") != "COMPLETE":
    raise ValueError("MR8 requires a complete MR7 role-cost ranking")
  selected = ranking.get("selected_roles")
  roles = ranking.get("roles")
  if not isinstance(selected, list) or not selected or len(selected) > 2 or len(selected) != len(set(selected)):
    raise ValueError("MR8 requires one or two unique selected role families")
  if not isinstance(roles, list) or any(not isinstance(row, Mapping) for row in roles):
    raise ValueError("MR7 role rows are unavailable")
  by_role: dict[str, Mapping[str, Any]] = {}
  for row in roles:
    role = row.get("role")
    if not isinstance(role, str) or not role or role in by_role: raise ValueError("MR7 role rows are duplicated or invalid")
    by_role[role] = row
  gated = {role for role,row in by_role.items() if row.get("gate") == "SELECTED_FOR_MR8"}
  if set(selected) != gated or any(role not in by_role for role in selected):
    raise ValueError("MR7 selected_roles disagree with SELECTED_FOR_MR8 gates")
  return selected, by_role


def _validate_authoritative_ranking(ranking: Mapping[str, Any]) -> None:
  authority = ranking.get("mr7_authority") if isinstance(ranking, Mapping) else None
  if not isinstance(authority, Mapping): raise ValueError("MR8 requires retained canonical MR7 producer authority")
  from boltbeam.search.semantic.mr7_evidence_bundle import ranking_from_authority
  rebuilt = ranking_from_authority(authority) | {"mr7_authority":authority}
  if rebuilt != ranking: raise ValueError("MR7 ranking differs from retained producer authority")


def _population_catalog(population_requests: Sequence[Mapping[str, Any]], *, model_sha256: str,
                        resolved_target_sha256: str) -> dict[str, tuple[dict[str, Any], Mapping[str, Any]]]:
  if not isinstance(population_requests, Sequence) or isinstance(population_requests, (str, bytes)):
    raise ValueError("population_requests must be a sequence")
  catalog: dict[str, tuple[dict[str, Any], Mapping[str, Any]]] = {}
  for request in population_requests:
    if not isinstance(request, Mapping): raise ValueError("population request must be a mapping")
    workload = request.get("semantic_workload")
    if not isinstance(workload, Mapping): raise ValueError("population request lacks semantic_workload")
    target = workload.get("target")
    if workload.get("model_hash") != model_sha256:
      raise ValueError("population workload model hash disagrees with MR8 request")
    if not isinstance(target, Mapping) or target.get("resolved_target_hash") != resolved_target_sha256:
      raise ValueError("population workload resolved target disagrees with MR8 request")
    identity = validate_exact_semantic_workload(workload)
    key = content_sha256(identity)
    if key in catalog: raise ValueError("duplicate exact semantic workload in MR8 population catalog")
    catalog[key] = (identity, request)
  return catalog


def build_mr8_population_selection(ranking: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
  """Select and materialize only the exact populations admitted by MR7."""
  required = {"schema", "mr7_sha256", "model_sha256", "resolved_target_sha256", "population_requests"}
  if not isinstance(request, Mapping) or set(request) != required:
    raise ValueError("MR8 selection request has missing or unknown fields")
  if request.get("schema") != REQUEST_SCHEMA: raise ValueError("unsupported MR8 selection request schema")
  expected_hash = request.get("mr7_sha256")
  actual_hash = content_sha256(ranking)
  if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None or expected_hash != actual_hash:
    raise ValueError("MR7 ranking content hash mismatch")
  model_sha256, resolved_target_sha256 = request.get("model_sha256"), request.get("resolved_target_sha256")
  if not isinstance(model_sha256, str) or _SHA256.fullmatch(model_sha256) is None:
    raise ValueError("MR8 request requires the exact model SHA-256")
  if not isinstance(resolved_target_sha256, str) or _SHA256.fullmatch(resolved_target_sha256) is None:
    raise ValueError("MR8 request requires the exact resolved-target SHA-256")
  selected, role_rows = _selected_role_rows(ranking)
  catalog = _population_catalog(request["population_requests"], model_sha256=model_sha256,
                                resolved_target_sha256=resolved_target_sha256)

  needed: dict[str, tuple[str, dict[str, Any]]] = {}
  for role in selected:
    identities = role_rows[role].get("semantic_identities")
    if not isinstance(identities, list) or not identities:
      raise ValueError("selected MR7 role lacks exact semantic identities")
    for raw_identity in identities:
      identity = normalize_semantic_identity(raw_identity)
      if identity["role"] != role: raise ValueError("selected MR7 role contains a cross-role semantic identity")
      key = content_sha256(identity)
      if key in needed: raise ValueError("duplicate semantic identity in selected MR7 roles")
      needed[key] = (role, identity)
  missing = sorted(set(needed) - set(catalog))
  if missing: raise ValueError(f"MR8 population catalog is missing {len(missing)} selected semantic workload(s)")

  populations = []
  for role in selected:
    role_keys = sorted((key for key,(identity_role,_identity) in needed.items() if identity_role == role),
                       key=lambda key:canonical_json(needed[key][1]))
    for key in role_keys:
      identity, population_request = catalog[key]
      population = export_population(population_request)
      candidates = population.get("candidates")
      if not isinstance(candidates, list) or not candidates:
        raise ValueError("selected semantic population is empty")
      control_count = sum(row.get("candidate", {}).get("schedule", {}).get("plan_kind") == "tinygrad_heuristic.v1"
                          for row in candidates if isinstance(row, Mapping))
      if control_count != 1: raise ValueError("selected semantic population requires exactly one heuristic control")
      populations.append({
        "role":role,
        "semantic_identity_sha256":key,
        "semantic_identity":identity,
        "population_request_sha256":content_sha256(population_request),
        "campaign_request":dict(population_request),
        "population":population,
      })

  basis = [{key:row[key] for key in ("role", "gate", "headroom_fraction", "headroom_lower_95", "semantic_digest")}
           for role in selected for row in (role_rows[role],)]
  used = set(needed)
  result = {
    "schema":SCHEMA,
    "status":"COMPLETE",
    "mr7_sha256":actual_hash,
    "mr7_prerequisites":dict(ranking.get("prerequisites", {})),
    "model_sha256":model_sha256,
    "resolved_target_sha256":resolved_target_sha256,
    "selected_roles":list(selected),
    "selection_basis":basis,
    "selected_semantic_identity_count":len(needed),
    "population_count":len(populations),
    "candidate_count":sum(len(row["population"]["candidates"]) for row in populations),
    "ignored_unselected_population_request_count":len(set(catalog) - used),
    "populations":populations,
    "policy":"MR7 selects at most two role families; MR8 expands only exact joined semantic workloads",
    # MR9 re-runs this pure join and requires byte-equivalent output.  Keeping
    # both authorities prevents a self-hashed, hand-built selection from being
    # mistaken for the result of the MR7-to-MR8 transition.
    "mr7_ranking":json.loads(json.dumps(ranking, sort_keys=True, allow_nan=False)),
    "selection_request":json.loads(json.dumps(request, sort_keys=True, allow_nan=False)),
  }
  result["selection_sha256"] = content_sha256(result)
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Join a complete MR7 ranking to exact MR8 semantic populations")
  parser.add_argument("ranking", type=pathlib.Path)
  parser.add_argument("request", type=pathlib.Path)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args(argv)
  if args.out.exists(): raise FileExistsError(f"MR8 population selection output already exists: {args.out}")
  result = build_mr8_population_selection(json.loads(args.ranking.read_text()), json.loads(args.request.read_text()))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__": raise SystemExit(main())

__all__ = ["REQUEST_SCHEMA", "SCHEMA", "build_mr8_population_selection", "main"]
