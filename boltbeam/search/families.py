"""Route-family knob grammar (audit A4).

`search/` owns the topology/search grammar. Which families a quant offers is DATA in the quant registry
(`boltbeam/quant.py`); the KNOBS each family exposes live here, one authority instead of being inlined in
`emit.py` and duplicated in `analyze.py`. `family_knobs` takes the role and target so later phases (A6 target
caps, A5 MoE) can prune illegal knob values by wave/LDS/shape without touching callers; today it returns the
shipped grammar unchanged so the Qwen search space stays byte-stable.
"""
from __future__ import annotations

from typing import Any

from boltbeam.vocab import RoleGroup
from boltbeam.profile.ir import TensorRole, TargetProfile

# family name -> knob grammar. Values are the shipped literals (byte-stable with the pre-A4 search space).
_FAMILY_KNOBS: dict[str, dict[str, list[Any]]] = {
  "lanemap_gemv": {
    "block_groups": [1, 2, 4, 8, 16, 32],
    "words_per_group": [1, 2, 4, 8, 16, 32],
    "reduction": ["cross_lane", "partials_plus_reduce"],
  },
  "q6k_route": {
    "row_grouping": [1, 2, 4],
    "k_pos_grouping": [16, 32],
    "reduction": ["coop_partial", "direct", "hybrid"],
  },
  "matmul_or_attention": {
    "tile_m": [16, 32],
    "tile_n": [16, 32],
    "tile_k": [16, 32],
  },
}


# MoE-specific families (audit A5). Routed per-expert weights get batched-GEMV + layout + dispatch families;
# the router gets a top-k family. Shared experts are always-active and stay dense-like (quant family only).
_MOE_FAMILY_KNOBS: dict[str, dict[str, list[Any]]] = {
  "moe_expert_batched_gemv": {"expert_tile": [1, 2, 4, 8], "reduction": ["cross_lane", "partials_plus_reduce"]},
  "expert_weight_layout": {"layout": ["stacked", "interleaved", "per_expert"]},
  "active_expert_dispatch": {"dispatch": ["gather", "scatter", "masked"], "cache": ["none", "lru_active"]},
  "router_topk": {"router_reduction": ["argmax_scan", "full_softmax"]},
}


_SSM_FAMILY_KNOBS: dict[str, dict[str, list[Any]]] = {
  "ssm_projection": {"tile_m": [1, 2, 4], "vector_load": [1, 2, 4]},
  "ssm_conv": {"kernel_width": [4], "state_layout": ["contiguous", "interleaved"]},
  "ssm_scan": {"scan": ["serial", "block_prefix"], "state_layout": ["contiguous", "interleaved"]},
  "ssm_state_update": {"state_tile": [1, 2, 4], "state_layout": ["contiguous", "interleaved"]},
}


def _moe_knobs(family:str) -> dict[str, list[Any]]:
  return {k: list(v) for k, v in _MOE_FAMILY_KNOBS.get(family, {}).items()}


def moe_families(role:TensorRole, target:TargetProfile | None = None, top_k:int | None = None
                 ) -> list[dict[str, Any]]:
  """MoE-specific route families for a MoE role, so a MoE model does NOT get searched as if it were dense FFN.
  Returns [] for dense and shared-expert roles (shared experts are always active -> dense-like quant family)."""
  r = role.role
  if r == RoleGroup.MOE_ROUTER.value:
    knobs = _moe_knobs("router_topk")
    knobs["active_expert_count"] = [top_k] if top_k else [1, 2, 4, 8]
    return [{"family": "router_topk", "status": "candidate", "knobs": knobs}]
  if r == RoleGroup.MOE_EXPERT_GATE_UP.value or r == RoleGroup.MOE_EXPERT_DOWN.value:
    fams = [{"family": f, "status": "candidate", "knobs": _moe_knobs(f)}
            for f in ("moe_expert_batched_gemv", "expert_weight_layout", "active_expert_dispatch")]
    if top_k:
      fams[-1]["knobs"]["active_expert_count"] = [top_k]
    fams[0]["knobs"]["n_expert"] = [role.n_expert] if role.n_expert else []
    return fams
  return []


def ssm_families(role:TensorRole, target:TargetProfile | None = None) -> list[dict[str, Any]]:
  """SSM/DeltaNet-specific route families. These roles are not quant GEMVs; they get scan/conv/state
  candidate surfaces and are measured separately from dense attention/FFN routes."""
  if role.role == RoleGroup.SSM_PROJECTION.value:
    return [{"family": "ssm_projection", "status": "candidate", "knobs": _ssm_knobs("ssm_projection")}]
  if role.role == RoleGroup.SSM_CONV.value:
    return [{"family": "ssm_conv", "status": "candidate", "knobs": _ssm_knobs("ssm_conv")}]
  if role.role == RoleGroup.SSM_SCAN.value:
    return [{"family": "ssm_scan", "status": "candidate", "knobs": _ssm_knobs("ssm_scan")}]
  if role.role == RoleGroup.SSM_STATE.value:
    return [{"family": "ssm_state_update", "status": "candidate", "knobs": _ssm_knobs("ssm_state_update")}]
  return []


def _ssm_knobs(family:str) -> dict[str, list[Any]]:
  return {k: list(v) for k, v in _SSM_FAMILY_KNOBS.get(family, {}).items()}


def family_knobs(family:str, role:TensorRole | None = None, target:TargetProfile | None = None
                 ) -> dict[str, list[Any]]:
  """A fresh copy of the knob grammar for a family. role/target are accepted for future capability pruning
  (A5/A6); the shipped grammar is target-independent so callers get identical output today."""
  grammar = _FAMILY_KNOBS.get(family, _SSM_FAMILY_KNOBS.get(family, {}))
  return {k: list(v) for k, v in grammar.items()}
