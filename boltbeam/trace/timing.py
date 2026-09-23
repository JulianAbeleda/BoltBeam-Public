from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.artifacts.base import sha256_json
from boltbeam.profile.ir import ModelProfile, TargetProfile
from boltbeam.quantization.quant import route_families_for
from boltbeam.roofline.roofline_trace import ingest_roofline
from boltbeam.trace.schedule_trace import ingest_schedule_trace
from boltbeam.vocab import SCHEMA_HW_TRACE, SCHEMA_TIMING_PROFILE, SCHEMA_TIMING_TRACE, SCHEMA_TRACE_REQUEST
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS

_REQUESTED_METRICS = {
  "whole_step": ["wall_us", "tok_s", "total_bytes", "launch_count", "host_sync_pct"],
  "kernel": ["kernel", "kind", "wall_us", "phys_bytes", "gbs"],
  "role": ["role", "quant", "shape", "wall_us", "phys_bytes"],
  "candidate": ["candidate_id", "candidate_us", "baseline_us", "spread_pct", "token_match", "route_bound"],
}


def build_trace_request(profile:ModelProfile, target:TargetProfile, workload_profile:dict[str, Any]) -> dict[str, Any]:
  workload = workload_profile.get("workload", "decode")
  contexts = list(workload_profile.get("contexts", []))
  roles = []
  for role in profile.roles:
    if role.rows <= 0 or role.cols <= 0:
      continue
    roles.append({
      "role": role.role,
      "role_class": role.role_class,
      "tensor_name": role.tensor_name,
      "shape": [role.rows, role.cols],
      "count": role.count,
      "n_expert": role.n_expert,
      "quant": role.quant,
      "route_families": list(route_families_for(role.quant)),
    })
  return {
    "schema": SCHEMA_TRACE_REQUEST,
    "model_id": profile.model_id,
    "target_id": target.target_id,
    "workload": workload,
    "contexts": contexts,
    "provider_neutral": True,
    "execution": {
      "context_mode": "fixed_decode_depth" if workload == "decode" else "fixed_prefill_length",
      "contexts": contexts,
      "warmups": 1,
      "samples": 1,
    },
    "collector_requirements": {
      "required_levels": ["whole_step"],
      "optional_levels": ["command_buffer", "dispatch", "kernel_resources", "dynamic_counters"],
      "scope_rule": "missing optional levels remain unavailable; they are never inferred from another level",
    },
    "requested_metrics": _REQUESTED_METRICS,
    "priority_roles": roles,
    "notes": [
      "External runners should return boltbeam.timing_trace.v1.",
      "Use physical packed bytes for roofline fields; logical/dequantized bytes are not comparable to HBM peak.",
      "Timing rows may be whole_step, kernel, role, or candidate scope.",
    ],
  }


def build_timing_profile(trace:dict[str, Any], *, request:dict[str, Any] | None = None,
                         source_path:str | pathlib.Path | None = None) -> dict[str, Any]:
  _validate_trace(trace)
  rows = [dict(r) for r in trace.get("rows", [])]
  contexts = _contexts(trace, rows)
  summaries = [_context_summary(trace, rows, ctx) for ctx in contexts]
  role_rows = _role_timing(rows, summaries)
  candidate_rows = [_candidate_timing(r) for r in rows if _scope(r) == "candidate"]
  dominant = _dominant_bucket(summaries)
  return {
    "schema": SCHEMA_TIMING_PROFILE,
    "model_id": trace["model_id"],
    "target_id": trace["target_id"],
    "workload": trace["workload"],
    "provider_id": trace.get("provider_id", "external"),
    "timing_source": trace.get("timing_source", "unknown"),
    "source": {
      "path": str(source_path) if source_path is not None else trace.get("source_path"),
      "fingerprint": sha256_json(trace),
      "schema": trace.get("schema"),
    },
    "request_role_count": len((request or {}).get("priority_roles", [])),
    "timing_row_count": len(rows),
    "contexts": [c for c in contexts if c is not None],
    "status": "classified" if rows else "empty",
    "dominant_timing_bucket": dominant,
    "next_actions": _next_actions(dominant, candidate_rows),
    "context_summaries": summaries,
    "role_timing": role_rows,
    "candidate_timing": candidate_rows,
  }


def _validate_trace(trace:dict[str, Any]) -> None:
  if trace.get("schema") not in {SCHEMA_TIMING_TRACE, SCHEMA_HW_TRACE}:
    raise ValueError(f"expected schema {SCHEMA_TIMING_TRACE} or {SCHEMA_HW_TRACE}, got {trace.get('schema')!r}")
  for key in ("model_id", "target_id", "workload"):
    if not trace.get(key):
      raise ValueError(f"timing trace missing required field {key!r}")
  if not isinstance(trace.get("rows", []), list):
    raise ValueError("timing trace field 'rows' must be a list")
  for row in trace.get("rows", []):
    if not isinstance(row, dict):
      raise ValueError("timing trace rows must be JSON objects")


def _fingerprint(obj:dict[str, Any]) -> str:
  return sha256_json(obj)


def _scope(row:dict[str, Any]) -> str:
  scope = row.get("scope")
  if scope:
    return str(scope)
  if row.get("candidate_id") or row.get("candidate_us") is not None:
    return "candidate"
  if row.get("kernel"):
    return "kernel"
  return "whole_step"


def _as_float(row:dict[str, Any] | None, *keys:str) -> float | None:
  if not row:
    return None
  metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
  for key in keys:
    value = row.get(key)
    if value is None:
      value = metrics.get(key)
    if value is None:
      continue
    try:
      return float(value)
    except (TypeError, ValueError):
      return None
  return None


def _as_int(value:Any) -> int | None:
  if value is None:
    return None
  try:
    return int(value)
  except (TypeError, ValueError):
    return None


def _contexts(trace:dict[str, Any], rows:list[dict[str, Any]]) -> list[int | None]:
  out = {_as_int(c) for c in trace.get("contexts", [])}
  out.update(_as_int(r.get("context")) for r in rows if r.get("context") is not None)
  present = sorted(c for c in out if c is not None)
  return present or [None]


def _rows_for_context(rows:list[dict[str, Any]], ctx:int | None) -> list[dict[str, Any]]:
  if ctx is None:
    return rows
  return [r for r in rows if _as_int(r.get("context")) == ctx]


def _context_summary(trace:dict[str, Any], rows:list[dict[str, Any]], ctx:int | None) -> dict[str, Any]:
  scoped = _rows_for_context(rows, ctx)
  whole = next((r for r in scoped if _scope(r) == "whole_step"), None)
  kernels = [r for r in scoped if _scope(r) == "kernel"]
  total_us = _as_float(whole, "wall_us", "total_us", "us")
  if total_us is None:
    total_us = sum(_as_float(r, "wall_us", "us") or 0.0 for r in kernels) or None
  total_bytes = _as_float(whole, "total_bytes", "phys_bytes", "bytes")
  if total_bytes is None:
    total_bytes = sum(_as_float(r, "phys_bytes", "bytes") or 0.0 for r in kernels) or None

  roofline = {}
  schedule = {}
  if kernels:
    roofline = ingest_roofline({
      "label": f"{trace.get('model_id')}:{trace.get('workload')}:{ctx if ctx is not None else 'all'}",
      "timing_source": trace.get("timing_source", "unknown"),
      "total_us": total_us,
      "total_bytes": total_bytes,
      "kernels": [
        {
          "name": r.get("kernel") or r.get("name"),
          "kind": r.get("kind") or "elementwise",
          "us": _as_float(r, "wall_us", "us") or 0.0,
          "phys_bytes": _as_float(r, "phys_bytes", "bytes") or 0.0,
        }
        for r in kernels
      ],
    }, peak_gbs=float(trace.get("peak_gbs", DEFAULT_PEAK_MEM_GBS)))
    schedule = ingest_schedule_trace({
      "label": f"{trace.get('model_id')}:{ctx if ctx is not None else 'all'}",
      "total_us": total_us,
      "kernels": [
        {
          "name": r.get("kernel") or r.get("name"),
          "kind": r.get("kind") or "elementwise",
          "us": _as_float(r, "wall_us", "us") or 0.0,
          "gbs": _as_float(r, "gbs"),
        }
        for r in kernels
      ],
    }, peak_gbs=float(trace.get("peak_gbs", DEFAULT_PEAK_MEM_GBS)))

  return {
    "context": ctx,
    "total_us": total_us,
    "total_bytes": total_bytes,
    "tok_s": _as_float(whole, "tok_s"),
    "launch_count": _as_float(whole, "launch_count"),
    "host_sync_pct": _as_float(whole, "host_sync_pct"),
    "kernel_count": len(kernels),
    "dominant_timing_bucket": roofline.get("dominant_bucket") or schedule.get("diagnosis") or "timing_inconclusive",
    "roofline": roofline,
    "schedule": schedule,
  }


def _shape_key(row:dict[str, Any]) -> tuple[int, ...]:
  try:
    return tuple(int(x) for x in (row.get("shape") or ()))
  except (TypeError, ValueError):
    return ()


def _role_timing(rows:list[dict[str, Any]], summaries:list[dict[str, Any]]) -> list[dict[str, Any]]:
  total_by_context = {s.get("context"): s.get("total_us") for s in summaries}
  grouped: dict[tuple[int | None, str | None, str | None, tuple[int, ...]], dict[str, Any]] = {}
  for row in rows:
    if _scope(row) not in {"role", "kernel"} or not row.get("role"):
      continue
    ctx = _as_int(row.get("context"))
    key = (ctx, row.get("role"), row.get("quant"), _shape_key(row))
    out = grouped.setdefault(key, {
      "context": ctx,
      "role": row.get("role"),
      "quant": row.get("quant"),
      "shape": list(_shape_key(row)),
      "wall_us": 0.0,
      "phys_bytes": 0.0,
      "kernels": [],
    })
    out["wall_us"] += _as_float(row, "wall_us", "us") or 0.0
    out["phys_bytes"] += _as_float(row, "phys_bytes", "bytes") or 0.0
    if row.get("kernel"):
      out["kernels"].append(row.get("kernel"))
  result = []
  for row in grouped.values():
    total_us = total_by_context.get(row["context"])
    row["pct_step"] = (100.0 * row["wall_us"] / total_us) if total_us else None
    row["classification"] = "timing_hot" if (row["pct_step"] or 0.0) >= 10.0 else "timing_observed"
    row["wall_us"] = round(row["wall_us"], 3)
    row["phys_bytes"] = round(row["phys_bytes"], 3)
    result.append(row)
  return sorted(result, key=lambda r: (-(r.get("pct_step") or 0.0), -(r.get("wall_us") or 0.0), r.get("role") or ""))


def _candidate_timing(row:dict[str, Any]) -> dict[str, Any]:
  candidate_us = _as_float(row, "candidate_us", "wall_us")
  baseline_us = _as_float(row, "baseline_us")
  spread_pct = _as_float(row, "spread_pct")
  speedup = None
  classification = "timing_inconclusive"
  if candidate_us is not None and baseline_us and baseline_us > 0:
    speedup = 100.0 * (baseline_us - candidate_us) / baseline_us
    threshold = max(spread_pct or 0.0, 0.5)
    if row.get("token_match") is False:
      classification = "correctness_failed"
    elif row.get("route_bound") is False:
      classification = "route_not_bound"
    elif speedup > threshold:
      classification = "timing_win"
    elif speedup < -threshold:
      classification = "timing_loss"
    else:
      classification = "timing_flat"
  return {
    "context": _as_int(row.get("context")),
    "candidate_id": row.get("candidate_id"),
    "role": row.get("role"),
    "quant": row.get("quant"),
    "shape": list(_shape_key(row)),
    "candidate_us": candidate_us,
    "baseline_us": baseline_us,
    "speedup_pct": speedup,
    "spread_pct": spread_pct,
    "token_match": row.get("token_match"),
    "route_bound": row.get("route_bound"),
    "classification": classification,
  }


def _dominant_bucket(summaries:list[dict[str, Any]]) -> str:
  if not summaries:
    return "timing_inconclusive"
  summary = max(summaries, key=lambda s: s.get("total_us") or 0.0)
  return summary.get("dominant_timing_bucket") or "timing_inconclusive"


def _next_actions(dominant:str, candidates:list[dict[str, Any]]) -> list[str]:
  actions = {
    "gemv_codegen_capped": "prioritize packed-load, dot/dequant, and tiling levers for the dominant weight-read kernels",
    "elementwise_dilution": "look for launch-bound elementwise/reduce work that can fold into neighboring kernels",
    "latency_bound": "avoid treating the dominant timing bucket as a bandwidth lever; gather concurrency evidence first",
    "activation_bound": "measure whether the activation/reduce work is real bandwidth or avoidable materialization",
    "occupancy_starved": "increase independent tiles, residency, or schedule parallelism before adding arithmetic features",
    "mixed": "collect a finer per-kernel trace; no single timing bucket dominates",
    "timing_inconclusive": "rerun the external trace with whole_step and kernel timing rows",
  }
  out = [actions.get(dominant, actions["timing_inconclusive"])]
  if any(c.get("classification") == "timing_win" for c in candidates):
    out.append("candidate timing contains a measurable win; require correctness and route-bound evidence before promotion")
  if any(c.get("classification") == "timing_loss" for c in candidates):
    out.append("candidate timing contains a regression; preserve it as refutation evidence if route-bound")
  return out
