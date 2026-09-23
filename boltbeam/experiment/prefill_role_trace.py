from __future__ import annotations

import copy
import json
import pathlib
from typing import Any


REQUIRED_COUNTERS = (
  "memory_busy_pct",
  "valu_busy_pct",
  "mfma_util_pct",
  "occupancy_pct",
  "l2_hit_pct",
)


def classify_trace_attribution(trace:dict[str, Any]) -> dict[str, Any]:
  """Classify whether a timing trace can support per-role attribution.

  Aggregate whole-step timing remains valid when this returns non_attributive.
  """
  meta = trace.get("metadata") or {}
  flags = meta.get("route_flags") or {}
  profile = str(flags.get("PROFILE", "")) == "1"
  rows = [r for r in trace.get("rows", []) if r.get("scope") == "kernel"]
  role_rows = [r for r in rows if r.get("role") not in (None, "unknown")]
  if not profile:
    return {"status": "non_attributive", "reason": "profile_disabled", "profile_enabled": False,
            "kernel_rows": len(rows), "per_role_kernel_rows": len(role_rows)}
  if not role_rows:
    return {"status": "non_attributive", "reason": "no_per_role_kernel_rows", "profile_enabled": True,
            "kernel_rows": len(rows), "per_role_kernel_rows": 0}
  return {"status": "attributable", "reason": None, "profile_enabled": True,
          "kernel_rows": len(rows), "per_role_kernel_rows": len(role_rows)}


def _f(value:Any) -> float | None:
  try:
    return None if value is None else float(value)
  except (TypeError, ValueError):
    return None


def _shape_arg(value:str | None) -> list[int] | None:
  if not value:
    return None
  try:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
  except ValueError as exc:
    raise ValueError(f"invalid --shape {value!r}; expected M,N,K") from exc
  if len(out) != 3:
    raise ValueError(f"invalid --shape {value!r}; expected M,N,K")
  return out


def _load_inventory(path:str | pathlib.Path | None) -> list[dict[str, Any]]:
  if not path:
    return []
  data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
  return list(data.get("roles", []))


def _role_inventory_match(roles:list[dict[str, Any]], *, role:str, quant:str | None,
                          shape:list[int] | None) -> dict[str, Any] | None:
  for item in roles:
    if item.get("role") != role:
      continue
    if quant and item.get("quant") != quant:
      continue
    inv_shape = item.get("shape") or []
    if shape and len(inv_shape) == 2 and [int(inv_shape[0]), int(inv_shape[1])] != [int(shape[1]), int(shape[2])]:
      continue
    return item
  return None


def _shape_matches(row_shape:Any, shape:list[int] | None) -> bool:
  if shape is None:
    return True
  try:
    return [int(x) for x in (row_shape or [])] == shape
  except (TypeError, ValueError):
    return False


def _resources(row:dict[str, Any]) -> dict[str, Any]:
  return row.get("resources") or {}


def _grid(row:dict[str, Any]) -> list[int] | None:
  res = _resources(row)
  grid = res.get("grid") or res.get("global_size")
  if not grid:
    return None
  try:
    return [int(x) for x in grid]
  except (TypeError, ValueError):
    return None


def _llama_grid_role_match(row:dict[str, Any], *, role_item:dict[str, Any] | None, quant:str | None,
                           shape:list[int] | None, rows_per_grid_x:int) -> bool:
  if not role_item or "mul_mat_q<" not in str(row.get("kernel") or ""):
    return False
  if quant and row.get("quant") != quant:
    return False
  grid = _grid(row)
  inv_shape = role_item.get("shape") or []
  if not grid or len(inv_shape) != 2:
    return False
  output_rows = int(inv_shape[0])
  if shape and output_rows != int(shape[1]):
    return False
  return int(grid[0]) * rows_per_grid_x == output_rows


def _row_matches(row:dict[str, Any], *, role:str, quant:str | None, shape:list[int] | None,
                 role_item:dict[str, Any] | None, rows_per_grid_x:int) -> bool:
  if row.get("scope") != "kernel":
    return False
  if row.get("role") == role and (quant is None or row.get("quant") == quant) and _shape_matches(row.get("shape"), shape):
    return True
  return _llama_grid_role_match(row, role_item=role_item, quant=quant, shape=shape, rows_per_grid_x=rows_per_grid_x)


def _apply_exact_role_metadata(row:dict[str, Any], *, role:str, quant:str | None, shape:list[int] | None,
                               role_item:dict[str, Any] | None, rows_per_grid_x:int) -> dict[str, Any]:
  out = copy.deepcopy(row)
  out["role"] = role
  if quant:
    out["quant"] = quant
  if shape:
    out["shape"] = shape
  if role_item:
    out["phys_bytes"] = float(role_item.get("estimated_weight_bytes") or out.get("phys_bytes") or 0.0)
    out["bytes_source"] = "weight_inventory_exact_role_match"
    grid = _grid(row)
    if grid:
      out["role_match"] = {
        "method": "grid_x_times_rows_per_grid_x_matches_inventory_rows",
        "grid_x": grid[0],
        "rows_per_grid_x": rows_per_grid_x,
        "matched_weight_shape": role_item.get("shape"),
        "confidence": "high",
      }
  wall_us = _f(out.get("wall_us"))
  bytes_ = _f(out.get("phys_bytes"))
  if wall_us and bytes_:
    out["gbs"] = bytes_ / wall_us / 1000.0
  return out


def _counter_coverage(rows:list[dict[str, Any]], required:tuple[str, ...]) -> dict[str, Any]:
  present = sorted({k for row in rows for k in (row.get("counters") or {}) if k in required})
  missing = [k for k in required if k not in present]
  for row in rows:
    counters = row.get("counters") or {}
    row["missing_counters"] = [k for k in required if k not in counters]
  return {
    "required": list(required),
    "present": present,
    "missing": missing,
    "complete": len(missing) == 0,
  }


PMC_GRAPH_BYPASS_HINT = (
  "PMC counters are emitted only from eager AMDProgram.__call__; prefill GEMMs run through "
  "HCQGraph.__call__ (tinygrad/runtime/graph/hcq.py) which bypasses PMC. Counter coverage for "
  "gemm roles is unattainable on this path."
)


def _flag_on(route_flags:dict[str, Any], key:str) -> bool:
  return str(route_flags.get(key)) == "1"


def _coverage_reason(trace:dict[str, Any], kernels:list[dict[str, Any]],
                     coverage:dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
  """Classify WHY counter coverage is incomplete. Returns (reason, reason_detail).

  reason vocabulary:
  - pmc_graph_bypass: PMC+PROFILE were on and PMC produced counters on non-gemm kernels, but every
    gemm role row is empty -> graph-captured GEMMs never hit the eager per-call PMC read.
  - pmc_disabled: a PMC-capable tinygrad trace where PMC/PROFILE were not both on.
  - None: coverage complete, or a trace with no PMC context (e.g. llama / no route_flags).
  """
  if coverage.get("complete"):
    return None, {}
  route_flags = (trace.get("metadata") or {}).get("route_flags") or {}
  pmc_on, profile_on = _flag_on(route_flags, "PMC"), _flag_on(route_flags, "PROFILE")
  gemm = [r for r in kernels if r.get("kind") == "gemm"]
  gemm_with = [r for r in gemm if r.get("counters")]
  nongemm_with = [r for r in trace.get("rows", [])
                  if r.get("scope") == "kernel" and r.get("kind") != "gemm" and r.get("counters")]
  detail = {
    "pmc": pmc_on,
    "profile": profile_on,
    "gemm_rows": len(gemm),
    "gemm_rows_with_counters": len(gemm_with),
    "nongemm_rows_with_counters": len(nongemm_with),
  }
  if pmc_on and profile_on and gemm and not gemm_with and nongemm_with:
    detail["hint"] = PMC_GRAPH_BYPASS_HINT
    return "pmc_graph_bypass", detail
  if "PMC" in route_flags and not (pmc_on and profile_on):
    return "pmc_disabled", detail
  return None, detail


def prefill_role_trace_report(trace:dict[str, Any], *, role:str, quant:str | None = None, shape:str | None = None,
                              weight_inventory:str | pathlib.Path | None = None, rows_per_grid_x:int = 4,
                              required_counters:tuple[str, ...] = REQUIRED_COUNTERS) -> dict[str, Any]:
  parsed_shape = _shape_arg(shape)
  roles = _load_inventory(weight_inventory)
  role_item = _role_inventory_match(roles, role=role, quant=quant, shape=parsed_shape)
  whole = [copy.deepcopy(r) for r in trace.get("rows", []) if r.get("scope") == "whole_step"]
  kernels = []
  for row in trace.get("rows", []):
    if not _row_matches(row, role=role, quant=quant, shape=parsed_shape, role_item=role_item,
                        rows_per_grid_x=rows_per_grid_x):
      continue
    kernels.append(_apply_exact_role_metadata(row, role=role, quant=quant, shape=parsed_shape,
                                              role_item=role_item, rows_per_grid_x=rows_per_grid_x))
  if not kernels:
    raise ValueError(f"no kernel rows matched role={role!r} quant={quant!r} shape={parsed_shape!r}")
  coverage = _counter_coverage(kernels, required_counters)
  reason, reason_detail = _coverage_reason(trace, kernels, coverage)
  coverage["reason"] = reason
  if reason is not None:
    coverage["reason_detail"] = reason_detail
  return {
    "schema": "boltbeam.prefill_role_trace.v1",
    "model_id": trace.get("model_id"),
    "target_id": trace.get("target_id"),
    "workload": trace.get("workload"),
    "provider_id": trace.get("provider_id"),
    "contexts": trace.get("contexts") or [],
    "role_filter": {"role": role, "quant": quant, "shape": parsed_shape},
    "source_trace": trace.get("source_path"),
    "weight_inventory": str(weight_inventory) if weight_inventory else None,
    "counter_requirements": coverage,
    "summary": _summary(kernels, whole),
    "rows": whole + kernels,
  }


def _summary(kernels:list[dict[str, Any]], whole:list[dict[str, Any]]) -> dict[str, Any]:
  wall_us = sum(float(r.get("wall_us") or 0.0) for r in kernels)
  raw_us = sum(float(r.get("raw_wall_us") or r.get("wall_us") or 0.0) for r in kernels)
  bytes_ = sum(float(r.get("phys_bytes") or 0.0) for r in kernels)
  whole_us = _f(whole[0].get("wall_us")) if whole else None
  return {
    "kernel_rows": len(kernels),
    "kernel_wall_us": wall_us,
    "kernel_raw_wall_us": raw_us,
    "phys_bytes": bytes_,
    "effective_gbs": bytes_ / wall_us / 1000.0 if wall_us and bytes_ else None,
    "pct_step": 100.0 * wall_us / whole_us if whole_us else None,
    "resources": [r.get("resources") or {} for r in kernels],
  }


def prefill_role_trace_markdown(report:dict[str, Any]) -> str:
  filt, s, c = report.get("role_filter", {}), report.get("summary", {}), report.get("counter_requirements", {})
  lines = [
    f"# Prefill Role Trace: {report.get('provider_id')}",
    "",
    f"- role: `{filt.get('role')}`",
    f"- quant: `{filt.get('quant')}`",
    f"- shape: `{filt.get('shape')}`",
    f"- kernel wall us: `{_fmt(s.get('kernel_wall_us'))}`",
    f"- effective GB/s: `{_fmt(s.get('effective_gbs'))}`",
    f"- pct step: `{_fmt(s.get('pct_step'))}`",
    f"- counter coverage complete: `{c.get('complete')}`",
    f"- counters present: `{', '.join(c.get('present') or [])}`",
    f"- counters missing: `{', '.join(c.get('missing') or [])}`",
    *( [f"- counter coverage reason: `{c['reason']}`{_reason_suffix(c.get('reason_detail') or {})}"]
       if c.get("reason") else [] ),
    "",
    "## Kernel Rows",
    "",
    "| kernel | wall us | raw us | calls | bytes | GB/s | resources | missing counters |",
    "|---|---:|---:|---:|---:|---:|---|---|",
  ]
  for row in [r for r in report.get("rows", []) if r.get("scope") == "kernel"]:
    lines.append("| " + " | ".join([
      str(row.get("kernel") or ""),
      _fmt(row.get("wall_us")),
      _fmt(row.get("raw_wall_us")),
      str(row.get("calls") or ""),
      _fmt(row.get("phys_bytes")),
      _fmt(row.get("gbs")),
      _resource_text(row.get("resources") or {}),
      ", ".join(row.get("missing_counters") or []),
    ]) + " |")
  return "\n".join(lines) + "\n"


def _reason_suffix(detail:dict[str, Any]) -> str:
  if not detail:
    return ""
  return (f" ({detail.get('gemm_rows', 0)} gemm rows, {detail.get('gemm_rows_with_counters', 0)} with "
          f"counters; PMC live on {detail.get('nongemm_rows_with_counters', 0)} non-gemm)")


def _fmt(value:Any) -> str:
  v = _f(value)
  return "" if v is None else f"{v:.3f}"


def _resource_text(resources:dict[str, Any]) -> str:
  if not resources:
    return "missing"
  parts = []
  for key in ("grid", "global_size", "workgroup", "local_size"):
    if resources.get(key):
      parts.append(f"{key}=" + "x".join(str(x) for x in resources[key]))
  for key in ("workgroup_threads", "lds_bytes", "scratch_bytes", "vgpr", "sgpr"):
    if resources.get(key) is not None:
      parts.append(f"{key}={resources[key]}")
  return ", ".join(parts) or "present"
