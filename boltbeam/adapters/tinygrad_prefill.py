"""Normalize tinygrad prefill profiler and route-observer evidence.

This module intentionally consumes plain mappings.  Tinygrad's observer dataclasses can be
serialized by the producer without making BoltBeam import tinygrad or execute a model.
"""
from __future__ import annotations

from typing import Any, Iterable

from boltbeam.artifacts.tinygrad_rocprof import classify_tinygrad_kernel
from boltbeam.vocab import SCHEMA_TIMING_TRACE
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS


def _value(event:Any, name:str, default:Any = None) -> Any:
  return event.get(name, default) if isinstance(event, dict) else getattr(event, name, default)


def route_identities(attachments:Iterable[Any] = (), executions:Iterable[Any] = ()) -> list[dict[str, Any]]:
  """Join tinygrad prefill route attachment and execution observations by invocation.

  A route is *bound* only when an execution event names the same route selected by its
  attachment and reports no fallback.  This is evidence normalization, not route selection.
  """
  attached = {str(_value(event, "invocation_id")): event for event in attachments if _value(event, "invocation_id") is not None}
  executed = {str(_value(event, "invocation_id")): event for event in executions if _value(event, "invocation_id") is not None}
  out = []
  for invocation_id in sorted(set(attached) | set(executed)):
    attachment, execution = attached.get(invocation_id), executed.get(invocation_id)
    row: dict[str, Any] = {"invocation_id": invocation_id}
    if attachment is not None:
      for key in ("route_id", "tensor_identity", "selected_policy", "scanned_target_facts", "allocation_owner_identity"):
        value = _value(attachment, key)
        if value is not None: row[key] = value
    if execution is not None:
      for key in ("executed_route_id", "candidate_identity", "program_identity", "fallback_used", "fallback_reason", "execution_evidence"):
        value = _value(execution, key)
        if value is not None: row[key] = value
    if attachment is None:
      row["status"] = "attachment_missing"
    elif execution is None:
      row["status"] = "execution_missing"
    elif row.get("route_id") != row.get("executed_route_id"):
      row["status"] = "route_mismatch"
    elif row.get("fallback_used"):
      row["status"] = "fallback_used"
    elif not row.get("candidate_identity") or not row.get("program_identity"):
      row["status"] = "execution_identity_missing"
    else:
      row["status"] = "bound"
    out.append(row)
  return out


def _profiler_status(events:list[Any]) -> dict[str, Any]:
  if not events:
    return {"status": "missing", "reason": "no_profile_kernel_events", "kernel_event_count": 0}
  with_counters = sum(bool(_value(event, "counters")) for event in events)
  if with_counters != len(events):
    return {"status": "partial", "reason": "missing_hardware_counters", "kernel_event_count": len(events),
            "kernel_events_with_counters": with_counters}
  return {"status": "observed", "reason": None, "kernel_event_count": len(events),
          "kernel_events_with_counters": with_counters}


def adapt_prefill_timing_trace(*, model_id:str, target_id:str, context:int, depth:int,
                               kernel_events:Iterable[Any] = (), authority_wall_us:float | None = None,
                               attachments:Iterable[Any] = (), executions:Iterable[Any] = (),
                               provider_id:str = "tinygrad/profile-events", peak_gbs:float = DEFAULT_PEAK_MEM_GBS) -> dict[str, Any]:
  """Build existing timing-trace rows from a prefill profile pass.

  ``authority_wall_us`` is optional.  When supplied, profile event time is used only as
  attribution and is scaled to that synchronized wall time, matching the native trace rule.
  """
  events = list(kernel_events)
  routes = {row["invocation_id"]: row for row in route_identities(attachments, executions)}
  grouped: dict[tuple[str, str | None], dict[str, Any]] = {}
  for event in events:
    kernel = str(_value(event, "kernel", _value(event, "name", "")))
    if not kernel:
      continue
    invocation_id = _value(event, "invocation_id")
    key = (kernel, str(invocation_id) if invocation_id is not None else None)
    raw_wall_us = float(_value(event, "raw_wall_us", _value(event, "wall_us", 0.0)) or 0.0)
    row = grouped.setdefault(key, {"kernel": kernel, "invocation_id": invocation_id, "raw_wall_us": 0.0, "calls": 0})
    row["raw_wall_us"] += raw_wall_us
    row["calls"] += int(_value(event, "calls", 1) or 1)
    if _value(event, "counters"):
      row["counters"] = _value(event, "counters")
  raw_total = sum(row["raw_wall_us"] for row in grouped.values())
  wall_us = float(authority_wall_us) if authority_wall_us is not None else raw_total
  rows = []
  for row in sorted(grouped.values(), key=lambda item: -item["raw_wall_us"]):
    info = classify_tinygrad_kernel(row["kernel"])
    out = {
      "scope": "kernel", "context": context, "depth": depth, "kernel": row["kernel"],
      "kind": info["kind"], "role": info["role"], "raw_wall_us": row["raw_wall_us"], "calls": row["calls"],
      "wall_us": wall_us * row["raw_wall_us"] / raw_total if authority_wall_us is not None and raw_total else row["raw_wall_us"],
      "time_source": "profile_scaled_to_synced_wall" if authority_wall_us is not None else "tinygrad_profile_events",
    }
    if info.get("quant") is not None: out["quant"] = info["quant"]
    if info.get("shape"): out["shape"] = info["shape"]
    if row.get("counters"): out["counters"] = row["counters"]
    if row["invocation_id"] is not None: out["route"] = routes.get(str(row["invocation_id"]), {"invocation_id": row["invocation_id"], "status": "unobserved"})
    rows.append(out)
  whole = {"scope": "whole_step", "context": context, "depth": depth, "wall_us": wall_us,
           "tok_s": context * 1e6 / wall_us if wall_us > 0 else 0.0,
           "launch_count": sum(row["calls"] for row in grouped.values()),
           "time_source": "synced_authority_wall" if authority_wall_us is not None else "tinygrad_profile_event_sum"}
  return {
    "schema": SCHEMA_TIMING_TRACE, "model_id": model_id, "target_id": target_id, "workload": "prefill",
    "provider_id": provider_id, "timing_source": whole["time_source"], "peak_gbs": peak_gbs, "contexts": [context],
    "metadata": {"prefill_depth": depth, "route_identities": list(routes.values()), "profiler": _profiler_status(events)},
    "rows": [whole] + rows,
  }
