from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOUNDARY_SCHEMA = "boltbeam.tinygrad_boundary_plan.v1"

PORTED_COMPAT = {
  "extra/qk_artifact_cache_inventory.py",
  "extra/qk_decode_primitive_candidate_template.py",
  "extra/qk_decode_role_profile.py",
  "extra/qk_descriptor_policy.py",
  "extra/qk_policy_consistency_check.py",
  "extra/qk_route_manifest.py",
  "extra/qk_search_spec.py",
  "extra/qk_search_util.py",
  "extra/qk_semantic_candidate.py",
}

PURE_BOLTBEAM_REVIEW = {
  "extra/amd_isa_decode_attention_ceiling_audit.py",
  "extra/amd_isa_lm_head_q6k_route_audit.py",
  "extra/amd_isa_system_residual_ceiling_audit.py",
  "extra/amd_isa_weight_path_ceiling_audit.py",
  "extra/amd_isa_weight_path_search_scope_builder.py",
  "extra/qk_decode_outer_b_split_contract.py",
  "extra/qk_decode_pressure_search_ownership_audit.py",
  "extra/qk_prefill_theoretical_ceiling_audit.py",
  "extra/qk_project_search_ledger.py",
  "extra/qk_pure_machine_search_gap_audit.py",
  "extra/qk_pure_search_gap_audit.py",
  "extra/qk_tg_p8_delta_audit.py",
  "extra/qk_validate_search_provenance.py",
}

MANIFEST_DEPENDENT_POLICY = {
  "extra/pure_machine_search_default_path_census.py",
  "extra/qk_attention_reopen_gate.py",
  "extra/qk_bandwidth_roofline.py",
  "extra/qk_candidate_evaluator.py",
  "extra/qk_g3_provenance_audit.py",
  "extra/qk_ledger_seed_pms_r4.py",
  "extra/qk_new_profile_search.py",
  "extra/qk_prefill_pipe_template_audit.py",
  "extra/qk_profile_opener.py",
  "extra/qk_profile_regenerate_check.py",
  "extra/qk_pure_search_diagnostic_gate.py",
  "extra/qk_pure_search_guard.py",
  "extra/qk_pure_search_next_candidate.py",
  "extra/qk_search_space_manifest_check.py",
  "extra/qk_template_candidate_gate.py",
  "extra/qk_topology_candidate_author.py",
}


def _decouple_lane(path:str, category:str) -> tuple[str, str]:
  if path in PORTED_COMPAT:
    return "ported_compat_shim", "BoltBeam now owns the pure logic; tinygrad copy should become a temporary compatibility shim or be removed after callers switch"
  if path.startswith("test/"):
    return "split_tests", "keep runtime regression in tinygrad; move policy/search contract tests to BoltBeam"
  if path in PURE_BOLTBEAM_REVIEW:
    return "pure_boltbeam_review", "port as BoltBeam report/audit/search logic; it should consume artifacts and emit decisions without importing tinygrad"
  if path in MANIFEST_DEPENDENT_POLICY:
    return "manifest_dependent_policy", "port after replacing extra.qk_route_manifest/search helpers with BoltBeam policy modules"
  if path.startswith("extra/"):
    return "tinygrad_runner_adapter", "keep tinygrad GPU/runtime execution; move policy/profile/search/eval decision logic to BoltBeam"
  return "manual_decouple", "split mixed ownership at the narrowest JSON contract"


def load_tinygrad_boundary_audit(path:str | Path) -> dict[str, Any]:
  data = json.loads(Path(path).read_text())
  if not isinstance(data, dict) or "files" not in data:
    raise ValueError(f"{path}: expected tinygrad boundary audit JSON with a files list")
  return data


def build_boundary_plan(audit:dict[str, Any]) -> dict[str, Any]:
  files = audit.get("files")
  if not isinstance(files, list):
    raise ValueError("audit['files'] must be a list")

  rows = []
  counts = Counter()
  by_action = defaultdict(list)
  for row in files:
    category = row.get("category")
    if category not in {"decouple", "move_boltbeam"}:
      continue
    path = str(row.get("path", ""))
    action = str(row.get("action", ""))
    if category == "decouple":
      port_kind, seam = _decouple_lane(path, category)
    else:
      port_kind = "boltbeam_owned"
      seam = "migrate artifact, report, search-space, roofline, or policy authority to BoltBeam"
    item = {
      "path": path,
      "category": category,
      "action": action,
      "port_kind": port_kind,
      "seam": seam,
      "reason": row.get("reason", ""),
      "confidence": row.get("confidence", ""),
    }
    rows.append(item)
    counts[category] += 1
    by_action[port_kind].append(path)

  return {
    "schema": BOUNDARY_SCHEMA,
    "source_repo": audit.get("summary", {}).get("repo"),
    "source_head": audit.get("summary", {}).get("git_head"),
    "counts": dict(sorted(counts.items())),
    "port_kinds": {k: len(v) for k, v in sorted(by_action.items())},
    "principle": {
      "tinygrad": "owns runtime execution, kernels, compiler/backend lowering, and hardware gates",
      "boltbeam": "owns model facts, candidate/search space, evaluation policy, ledgers, roofline attribution, and reports",
      "contract": "tinygrad emits evidence JSON; BoltBeam judges and records decisions; tinygrad consumes only promoted runtime policy",
    },
    "items": sorted(rows, key=lambda r: (r["category"], r["path"])),
  }


def boundary_plan_markdown(plan:dict[str, Any]) -> str:
  lines = [
    "# Tinygrad / BoltBeam Boundary Plan",
    "",
    f"Source: `{plan.get('source_repo')}` @ `{plan.get('source_head')}`",
    "",
    "## Principle",
    "",
  ]
  principle = plan["principle"]
  lines += [
    f"- tinygrad: {principle['tinygrad']}",
    f"- BoltBeam: {principle['boltbeam']}",
    f"- contract: {principle['contract']}",
    "",
    "## Counts",
    "",
  ]
  for k, v in plan.get("counts", {}).items():
    lines.append(f"- **{k}**: {v}")
  lines += ["", "## Port Kinds", ""]
  for k, v in plan.get("port_kinds", {}).items():
    lines.append(f"- **{k}**: {v}")
  lines += ["", "## Items", ""]
  for item in plan["items"]:
    lines.append(f"- `{item['path']}` -> `{item['port_kind']}`: {item['seam']}")
  return "\n".join(lines) + "\n"
