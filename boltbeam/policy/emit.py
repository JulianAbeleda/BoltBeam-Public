from __future__ import annotations

from typing import Any

from boltbeam.manifest import Candidate, load_candidates
from boltbeam.eval import promotion_requirements
from boltbeam.profile.ir import ModelProfile, TargetProfile
from boltbeam.search.emit import emit_search_space


_ATTENTION_SHAPE_FIELDS = {
  "Hq": "head_count",
  "Hkv": "head_count_kv",
  "Hd": "head_dim",
}


def _shape_matches_profile(shape:dict[str, Any], profile:ModelProfile) -> bool:
  attention = dict(profile.metadata.get("attention", {}))
  for shape_key, meta_key in _ATTENTION_SHAPE_FIELDS.items():
    if shape_key in shape and attention.get(meta_key) != shape[shape_key]:
      return False
  return bool(shape)


def _structural_shape_ok(rule:dict[str, Any], rows:int, cols:int) -> bool:
  """Per-tensor structural eligibility for a shape_rule candidate. GEMV dims are rows=N (out), cols=K (in)."""
  if not rule: return False
  cols_div = int(rule.get("cols_div", 1)) or 1
  cols_div_mod = int(rule.get("cols_div_mod", 1)) or 1
  rows_mod = int(rule.get("rows_mod", 1)) or 1
  if (cols // cols_div) % cols_div_mod != 0: return False
  if rows % rows_mod != 0: return False
  return True


def _selected_route_row(cand:Candidate, *, role:str, shape:dict[str, Any], quant:str | None) -> dict[str, Any]:
  return {
    "role": role,
    "shape": shape,
    "quant": quant,
    "selected_route": cand.candidate_id,
    "status": cand.status,
    "tier": cand.tier,
    "provenance": cand.provenance or cand.origin,
    "route_family": cand.route_family,
    "route_params": cand.route_params,
    "rollback": cand.rollback,
    "evidence_refs": list(cand.evidence_refs),
    "candidates": [cand.route_family] if cand.route_family else [],
  }


def _selected_default_routes(profile:ModelProfile, target:TargetProfile,
                             candidates:list[Candidate]) -> list[dict[str, Any]]:
  if not (profile.is_complete and target.is_complete): return []
  routes = []
  for cand in candidates:
    if not (cand.default_on and cand.status == "promoted"): continue
    if not cand.applies_to(workload=cand.workload, architecture=profile.architecture_class): continue
    if cand.shape_rule:
      # Per-tensor selection (G3-style): emit one selected row for every profile weight tensor whose role/quant
      # matches this candidate and whose real GEMV shape satisfies the structural rule. Data-driven from the
      # profile's tensor facts — no model-name hardcode.
      for tr in profile.roles:
        if not cand.applies_to(workload=cand.workload, quant=tr.quant, role=tr.role,
                               architecture=profile.architecture_class): continue
        if not _structural_shape_ok(cand.shape_rule, tr.rows, tr.cols): continue
        routes.append(_selected_route_row(cand, role=tr.role, shape={"rows": tr.rows, "cols": tr.cols}, quant=tr.quant))
      continue
    if cand.default_shape and not _shape_matches_profile(cand.default_shape, profile): continue
    routes.append(_selected_route_row(cand, role=cand.roles[0] if cand.roles else cand.route_family,
                                      shape=cand.default_shape, quant=cand.quant[0] if cand.quant else None))
  return routes


def emit_seed_policy(profile:ModelProfile, target:TargetProfile, *, purity_required:bool=False) -> dict[str, Any]:
  search = emit_search_space(profile, target)
  # A9 fail-closed: an unclassified/incomplete architecture yields no unmeasured routes to select — every role
  # is blocked pending classification, so nothing downstream can select or promote a route by default.
  authorized = profile.is_complete
  default_status = "unmeasured" if authorized else "blocked-incomplete-architecture"
  routes = []
  for role in search["roles"]:
    routes.append({
      "role": role["role"],
      "shape": role["shape"],
      "quant": role["quant"],
      "selected_route": None,
      "status": default_status,
      "candidates": [r["family"] for r in role["route_families"]],
    })
  routes = _selected_default_routes(profile, target, load_candidates()) + routes
  policy = {
    "schema": "boltbeam.route_policy.v1",
    "model_id": profile.model_id,
    "architecture_class": profile.architecture_class,
    "authorized": authorized,
    "target": target.to_json(),
    "routes": routes,
    "promotion_required": promotion_requirements(),
  }
  # TG-P6: when the caller requires a pure-machine-search run, tag the policy so the tinygrad consumer can enforce
  # PURE_MACHINE_SEARCH_ONLY. Every selected row already carries provenance + rollback (see _selected_route_row).
  if purity_required:
    policy["purity_required"] = True
  return policy
