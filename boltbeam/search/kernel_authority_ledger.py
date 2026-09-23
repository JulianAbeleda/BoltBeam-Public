"""Strict reader and coverage checks for Boltbeam-owned runtime kernel routes."""
from __future__ import annotations

import json, pathlib
from typing import Any, Iterable

SCHEMA = "boltbeam.kernel_authority_ledger.v1"
DEFAULT_LEDGER = pathlib.Path(__file__).resolve().parents[1] / "data" / "nv_sm120_kernel_authority.json"
STATES = frozenset(("exact", "template", "blocked"))
CHOKE_POINTS = frozenset(("boltbeam_kernel_provider", "kernel_program", "direct_custom_kernel", "scheduler_rewrite"))


def load_kernel_authority_ledger(path: pathlib.Path = DEFAULT_LEDGER) -> dict[str, Any]:
  data = json.loads(path.read_text())
  if not isinstance(data, dict) or set(data) != {"schema", "target", "scope", "states", "routes"}: raise ValueError("invalid kernel authority ledger fields")
  if data["schema"] != SCHEMA: raise ValueError("unsupported kernel authority ledger schema")
  if not isinstance(data["routes"], list) or not data["routes"]: raise ValueError("kernel authority ledger has no routes")
  route_ids: set[str] = set(); policies: set[str] = set()
  for row in data["routes"]:
    required = {"route_policy", "route_id", "phase", "components", "choke_point", "state"}
    if not isinstance(row, dict) or set(row) != required: raise ValueError("invalid kernel authority route row")
    if not isinstance(row["route_id"], str) or not row["route_id"] or row["route_id"] in route_ids: raise ValueError("duplicate or invalid route_id")
    route_ids.add(row["route_id"])
    if row["phase"] not in ("decode", "prefill"): raise ValueError("invalid route phase")
    if not isinstance(row["components"], list) or not row["components"] or len(row["components"]) != len(set(row["components"])): raise ValueError("invalid route components")
    if row["choke_point"] not in CHOKE_POINTS or row["state"] not in STATES: raise ValueError("invalid route authority state")
    policy = row["route_policy"]
    if policy is not None:
      if not isinstance(policy, str) or not policy.endswith("-route-policy.json") or policy in policies: raise ValueError("duplicate or invalid route policy")
      policies.add(policy)
  return data


def assert_promoted_policy_coverage(ledger: dict[str, Any], promoted_policy_names: Iterable[str]) -> None:
  covered = {row["route_policy"] for row in ledger["routes"] if row["route_policy"] is not None}
  promoted = set(promoted_policy_names)
  missing, stale = promoted-covered, covered-promoted
  if missing or stale: raise ValueError(f"kernel authority coverage mismatch: missing={sorted(missing)}, stale={sorted(stale)}")


def migration_summary(ledger: dict[str, Any]) -> dict[str, int]:
  return {state: sum(row["state"] == state for row in ledger["routes"]) for state in sorted(STATES)}


__all__ = ["DEFAULT_LEDGER", "assert_promoted_policy_coverage", "load_kernel_authority_ledger", "migration_summary"]
