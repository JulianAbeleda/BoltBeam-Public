"""Pure primitive candidate-template generation for decode attention.

This ports the candidate-template logic from tinygrad's QK decode primitive
generator into BoltBeam without tinygrad imports or filesystem side effects.
"""
from __future__ import annotations

import time
from typing import Any


SCHEMA = "qk_decode_primitive_candidate_templates_v1"
DEFAULT_DATE = "2026-06-26"

DEFAULT_KNOBS = {
  "qk_owner": "per_output_column",
  "d_owner": "global_column",
  "dot_lowering": "scalar_fma",
  "kv_staging": "global_direct",
  "score_broadcast": "none",
  "gqa_reuse": "none",
}

PRIORITY: tuple[dict[str, str], ...] = (
  {
    "qk_owner": "per_query_head",
    "d_owner": "lane_group",
    "dot_lowering": "scalar_fma",
    "kv_staging": "global_direct",
    "score_broadcast": "cross_lane",
    "gqa_reuse": "register_g_vector",
  },
  {
    "qk_owner": "per_kv_tile",
    "d_owner": "lane_group",
    "dot_lowering": "v_dot2",
    "kv_staging": "global_direct",
    "score_broadcast": "cross_lane",
    "gqa_reuse": "register_g_vector",
  },
  {
    "qk_owner": "per_kv_tile",
    "d_owner": "lane_group",
    "dot_lowering": "v_dot2",
    "kv_staging": "lds_tile",
    "score_broadcast": "cross_lane",
    "gqa_reuse": "shared_tile",
  },
)


def required_lowering_support(knobs: dict[str, str]) -> list[str]:
  missing_lowering = []
  if knobs["score_broadcast"] == "cross_lane":
    missing_lowering.append("LaneMap/CrossLane score broadcast lowering")
  if knobs["dot_lowering"] == "v_dot2":
    missing_lowering.append("v_dot2 packed-dot lowering in decode attention")
  if knobs["kv_staging"] == "lds_tile":
    missing_lowering.append("TileMemory LDS K/V cooperative load + barrier lowering")
  return missing_lowering


def intended_fix(knobs: dict[str, str]) -> list[str]:
  return [k for k, v in knobs.items() if v != DEFAULT_KNOBS.get(k)]


def _risk(index: int) -> str:
  return "low" if index == 1 else "medium" if index == 2 else "high"


def _require_contract(contract: dict[str, Any]) -> tuple[str, list[Any]]:
  missing = [k for k in ("candidate_id_prefix", "required_gates") if k not in contract]
  if missing:
    raise ValueError(f"primitive template contract missing required key(s): {', '.join(missing)}")
  prefix = contract["candidate_id_prefix"]
  if not isinstance(prefix, str) or not prefix:
    raise ValueError("primitive template contract candidate_id_prefix must be a non-empty string")
  gates = contract["required_gates"]
  if not isinstance(gates, list):
    raise ValueError("primitive template contract required_gates must be a list")
  return prefix, list(gates)


def build_primitive_candidate_templates(
  contract: dict[str, Any],
  *,
  date: str = DEFAULT_DATE,
  timestamp: str | None = None,
  source_contract: str | None = None,
) -> dict[str, Any]:
  """Build decode-attention physical primitive candidate templates.

  `contract` is the decoded search contract. It must contain `candidate_id_prefix`
  and `required_gates`. Callers may inject `timestamp` for deterministic output.
  """
  prefix, gates = _require_contract(contract)
  candidates = []
  for i, knobs in enumerate(PRIORITY, 1):
    k = dict(knobs)
    candidates.append({
      "candidate_id": f"{prefix}_p{i}",
      "knobs": k,
      "intended_fix": intended_fix(k),
      "required_lowering_support": required_lowering_support(k),
      "gates": list(gates),
      "kill_condition": (
        "Stop if emitted ISA/resource artifact does not show the intended primitive flags or q.k redundancy "
        "remains tied to local output columns."
      ),
      "risk": _risk(i),
    })

  out: dict[str, Any] = {
    "date": date,
    "timestamp": timestamp if timestamp is not None else time.strftime("%Y%m%d-%H%M%S"),
    "schema": SCHEMA,
    "candidates": candidates,
  }
  if source_contract is not None:
    out["source_contract"] = source_contract
  return out


# Short alias for callers that want the original generator's verb.
build = build_primitive_candidate_templates
