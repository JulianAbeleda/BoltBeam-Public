from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.vocab import QuantSupport, RoleGroup, Verdict, is_ssm_role
from boltbeam.quantization.quant import quant_capability
from boltbeam.search.families import family_knobs, moe_families, ssm_families
from boltbeam.profile.ir import ModelProfile, TargetProfile, TensorRole
from boltbeam.search.spec import SearchRow


_FULL_KERNEL_EMIT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "full_kernel_candidates.json"


def _full_kernel_emit_descriptors() -> tuple[dict[str, Any], ...]:
  data = json.loads(_FULL_KERNEL_EMIT_PATH.read_text())
  if data.get("schema") != "boltbeam.full_kernel_emit.v1" or not isinstance(data.get("rows"), list):
    raise ValueError(f"{_FULL_KERNEL_EMIT_PATH} is not a boltbeam.full_kernel_emit.v1 descriptor")
  return tuple(data["rows"])


def emit_full_kernel_search_rows(profile:ModelProfile, target:TargetProfile) -> tuple[SearchRow, ...]:
  """Emit currently reachable full-kernel rows, fail-closed on applicability.

  Full-kernel rows remain separate from the legacy route-family JSON so existing
  consumers retain their wire format. The shared ``SearchRow`` representation is
  the authority consumed by the evaluator/result-store path.
  """
  out = []
  for descriptor in _full_kernel_emit_descriptors():
    match, row_data = descriptor.get("match"), descriptor.get("search_row")
    if not isinstance(match, dict) or not isinstance(row_data, dict):
      raise ValueError("full-kernel emit descriptor requires match and search_row objects")
    role_match = match["role"]
    roles = [role for role in profile.roles if role.role == role_match["name"]
             and role.rows == role_match["rows"] and role.cols == role_match["cols"]
             and role.quant == role_match["quant"]]
    if (profile.model_id == match["profile"] and profile.architecture_class == match["architecture_class"]
        and profile.hidden_size == match["hidden_size"] and profile.ffn_size == match["ffn_size"] and roles
        and target.target_id == match["target_id"] and target.backend == match["backend"]
        and target.wave_size == match["wave_size"] and target.backend_status == "complete"):
      out.append(SearchRow.from_dict(row_data))
  return tuple(out)


def _quant_families(role:TensorRole, target:TargetProfile) -> list[dict[str, Any]]:
  """Quant-GEMV route families for a role, driven by the quant capability registry (audit A4) — NOT by
  hardcoded per-quant name branches. Three outcomes, all explicit: supported (one candidate family per
  registry route_family + knobs), known-no-route (pending row), unsupported quant (search-space-incomplete)."""
  cap = quant_capability(role.quant)
  if cap is None:
    return [{
      "family": "unsupported_quant",
      "status": QuantSupport.UNSUPPORTED.value,
      "reason": f"quant {role.quant!r} is not in the quant capability registry",
      "knobs": {},
    }]
  families = [{
    "family": fam,
    "status": "candidate",
    "knobs": family_knobs(fam, role, target),
  } for fam in cap.route_families]
  if not families:
    families.append({
      "family": "known_no_route",
      "status": QuantSupport.KNOWN_NO_ROUTE.value,
      "reason": f"quant {role.quant!r} is known but has no route family yet",
      "pending_routes": list(cap.pending_routes),
      "knobs": {},
    })
  return families


def _route_families(role:TensorRole, target:TargetProfile, top_k:int | None = None) -> list[dict[str, Any]]:
  """All route families for a role. The router is purely MoE (top-k), no quant GEMV; routed expert weights get
  the quant GEMV family PLUS MoE batched-GEMV/layout/dispatch families; dense and shared-expert roles get the
  quant GEMV family only (audit A5 — a MoE model is not searched as if it were dense FFN)."""
  if is_ssm_role(role.role):
    return ssm_families(role, target)
  if role.role == RoleGroup.MOE_ROUTER.value:
    return moe_families(role, target, top_k)
  return _quant_families(role, target) + moe_families(role, target, top_k)


def emit_search_space(profile:ModelProfile, target:TargetProfile) -> dict[str, Any]:
  sig = profile.metadata.get("architecture_signals", {}) if isinstance(profile.metadata, dict) else {}
  top_k = sig.get("expert_used_count")
  roles = []
  for role in profile.roles:
    families = _route_families(role, target, top_k)
    roles.append({
      "role": role.role,
      "tensor_name": role.tensor_name,
      "shape": [role.rows, role.cols],
      "quant": role.quant,
      "count": role.count,
      "route_families": families,
    })
  space = {
    "schema": "boltbeam.search_space.v1",
    "model_id": profile.model_id,
    "architecture_class": profile.architecture_class,
    "complete": profile.is_complete,
    "target": target.to_json(),
    "roles": roles,
    "notes": [
      "This is a candidate space, not a promotion decision.",
      "Downstream evaluators must measure correctness, speed, memory, and rollback.",
      "Route families are derived from the quant capability registry; unknown quants are marked "
      "search-space-incomplete, never silently omitted.",
    ],
  }
  # A9 fail-closed: an unknown/incomplete architecture does not yield an authorized search space.
  if not profile.is_complete:
    space["status"] = Verdict.SEARCH_SPACE_INCOMPLETE.value
    space["blocked_reason"] = (f"architecture_class={profile.architecture_class!r} is not a recognized "
                               f"transformer class (or the profile has no searchable roles); routes are not "
                               f"authorized until the architecture is classified")
  return space
