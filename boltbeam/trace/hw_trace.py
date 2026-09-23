from __future__ import annotations

import copy
import csv
import pathlib
import re
import statistics
from typing import Any

from boltbeam.trace.timing import build_timing_profile
from boltbeam.trace.timing_compare import compare_timing_profiles
from boltbeam.vocab import SCHEMA_HW_TRACE, SCHEMA_HW_TRACE_COMPARE, SCHEMA_TIMING_TRACE


NORMALIZED_COUNTERS = (
  "occupancy_pct",
  "memory_busy_pct",
  "memory_stall_pct",
  "l2_hit_pct",
  "fetch_kb",
  "write_kb",
  "valu_busy_pct",
  "lds_conflict_pct",
  "lds_stall_pct",
  "mfma_util_pct",
)

_COUNTER_ALIASES = {
  "occupancy_pct": "occupancy_pct",
  "occupancypercent": "occupancy_pct",
  "occupancy": "occupancy_pct",
  "memory_busy_pct": "memory_busy_pct",
  "memunitbusy": "memory_busy_pct",
  "memory_busy": "memory_busy_pct",
  "memory_stall_pct": "memory_stall_pct",
  "memunitstalled": "memory_stall_pct",
  "memory_stall": "memory_stall_pct",
  "l2_hit_pct": "l2_hit_pct",
  "l2cachehit": "l2_hit_pct",
  "l2_hit": "l2_hit_pct",
  "fetch_kb": "fetch_kb",
  "fetch_size": "fetch_kb",
  "fetchsize": "fetch_kb",
  "write_kb": "write_kb",
  "write_size": "write_kb",
  "writesize": "write_kb",
  "valu_busy_pct": "valu_busy_pct",
  "valuutil": "valu_busy_pct",
  "valu_util": "valu_busy_pct",
  "lds_conflict_pct": "lds_conflict_pct",
  "ldsbankconflict": "lds_conflict_pct",
  "lds_bank_conflict": "lds_conflict_pct",
  "lds_stall_pct": "lds_stall_pct",
  "alustalledbylds": "lds_stall_pct",
  "alu_stalled_by_lds": "lds_stall_pct",
  "mfma_util_pct": "mfma_util_pct",
  "mfmautil": "mfma_util_pct",
  "mfma_util": "mfma_util_pct",
}

_KERNEL_KEYS = ("kernel", "Kernel", "Name", "Kernel_Name", "name", "kernel_name")


def _canon_key(key:str) -> str:
  return "".join(ch.lower() for ch in key.strip() if ch.isalnum() or ch == "_")


def _short_name(name:str) -> str:
  head = name.split("(", 1)[0].strip()
  if head.startswith("void "):
    head = head[5:]
  return head


def _as_float(value:Any) -> float | None:
  if value in (None, ""):
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def normalize_counter_row(row:dict[str, Any]) -> dict[str, float]:
  counters: dict[str, float] = {}
  for key, value in row.items():
    norm = _COUNTER_ALIASES.get(_canon_key(str(key)))
    if norm is None:
      continue
    val = _as_float(value)
    if val is not None:
      counters[norm] = val
  return counters


def _row_kernel(row:dict[str, Any]) -> str | None:
  for key in _KERNEL_KEYS:
    value = row.get(key)
    if value:
      return _short_name(str(value))
  return None


def load_counter_csv(path:str | pathlib.Path | None) -> dict[str, dict[str, float]]:
  if not path:
    return {}
  out: dict[str, dict[str, float]] = {}
  with pathlib.Path(path).open(newline="") as f:
    for raw in csv.DictReader(f):
      kernel = _row_kernel(raw)
      if not kernel:
        continue
      counters = normalize_counter_row(raw)
      if counters:
        out[kernel] = counters
  return out


def _match_counters(kernel:str, by_kernel:dict[str, dict[str, float]]) -> dict[str, float] | None:
  if kernel in by_kernel:
    return by_kernel[kernel]
  for key, counters in by_kernel.items():
    if key and (key in kernel or kernel in key):
      return counters
  return None


def timing_trace_to_hw_trace(trace:dict[str, Any], *, backend_counter_csv:str | pathlib.Path | None = None,
                             backend_id:str | None = None, provider_id:str | None = None) -> dict[str, Any]:
  if trace.get("schema") not in {SCHEMA_TIMING_TRACE, SCHEMA_HW_TRACE}:
    raise ValueError(f"expected timing or hw trace schema, got {trace.get('schema')!r}")
  out = copy.deepcopy(trace)
  source_schema = trace.get("schema")
  out["schema"] = SCHEMA_HW_TRACE
  out["source_schema"] = source_schema
  if provider_id:
    out["provider_id"] = provider_id
  out.setdefault("trace_source", "boltbeam_hw_trace")
  out.setdefault("counter_vocab", list(NORMALIZED_COUNTERS))
  out.setdefault("aux_sources", {})
  if backend_counter_csv:
    out["aux_sources"]["backend_counter_csv"] = str(backend_counter_csv)
  if backend_id:
    out.setdefault("backend_adapters", []).append(backend_id)

  by_kernel = load_counter_csv(backend_counter_csv)
  if by_kernel:
    matched = 0
    for row in out.get("rows", []):
      kernel = row.get("kernel")
      if not kernel:
        continue
      counters = _match_counters(str(kernel), by_kernel)
      if counters:
        row["counters"] = counters
        matched += 1
    out["counter_summary"] = {
      "source": str(backend_counter_csv),
      "kernel_counter_rows": len(by_kernel),
      "matched_kernel_rows": matched,
      "normalized_counters": list(NORMALIZED_COUNTERS),
    }
  return out


def hw_trace_to_timing_trace(trace:dict[str, Any]) -> dict[str, Any]:
  if trace.get("schema") == SCHEMA_TIMING_TRACE:
    return copy.deepcopy(trace)
  if trace.get("schema") != SCHEMA_HW_TRACE:
    raise ValueError(f"expected schema {SCHEMA_HW_TRACE}, got {trace.get('schema')!r}")
  out = copy.deepcopy(trace)
  out["schema"] = SCHEMA_TIMING_TRACE
  out["source_schema"] = SCHEMA_HW_TRACE
  for row in out.get("rows", []):
    row.pop("counters", None)
  return out


def build_profile_from_trace(trace:dict[str, Any], *, request:dict[str, Any] | None = None,
                             source_path:str | pathlib.Path | None = None) -> dict[str, Any]:
  return build_timing_profile(hw_trace_to_timing_trace(trace), request=request, source_path=source_path)


def compare_hw_traces(baseline:dict[str, Any], candidate:dict[str, Any], *, context:int | None = None) -> dict[str, Any]:
  bprof = build_profile_from_trace(baseline)
  cprof = build_profile_from_trace(candidate)
  timing_report = compare_timing_profiles(bprof, cprof, context=context)
  return {
    "schema": SCHEMA_HW_TRACE_COMPARE,
    "timing_compare": timing_report,
    "command_buffer_compare": _command_buffer_compare(baseline, candidate, context=context),
    "counter_compare": _counter_compare(baseline, candidate, context=context),
  }


def _command_buffer_compare(baseline:dict[str, Any], candidate:dict[str, Any], *,
                            context:int | None = None) -> dict[str, Any]:
  base, cand = _command_buffer_summary(baseline, context=context), _command_buffer_summary(candidate, context=context)
  b_wall, c_wall = base.get("whole_step_wall_us"), cand.get("whole_step_wall_us")
  return {
    "baseline": base,
    "candidate": cand,
    "candidate_vs_baseline_wall_ratio": c_wall / b_wall if b_wall and c_wall else None,
    "candidate_vs_baseline_tok_s_ratio": (cand.get("tok_s") / base["tok_s"]
                                           if base.get("tok_s") and cand.get("tok_s") else None),
    "scope_rule": "selected command-buffer time is diagnostic; runtime whole-step timing is throughput authority",
  }


def _command_buffer_summary(trace:dict[str, Any], *, context:int | None = None) -> dict[str, Any]:
  def same(row): return context is None or row.get("context") == context
  rows = [row for row in trace.get("rows", []) if row.get("scope") == "command_buffer" and same(row)]
  whole = next((row for row in trace.get("rows", []) if row.get("scope") == "whole_step" and same(row)), {})
  selected = [row for row in rows if row.get("measurement_selected")]
  latencies = [float(row["cpu_to_gpu_latency_us"]) for row in rows if row.get("cpu_to_gpu_latency_us") is not None]
  whole_us = _as_float(whole.get("wall_us"))
  intervals = sorted((float(row["start_us"]), float(row["start_us"]) + float(row["wall_us"]))
                     for row in selected if row.get("start_us") is not None and row.get("wall_us") is not None)
  merged: list[list[float]] = []
  for start, end in intervals:
    if not merged or start > merged[-1][1]: merged.append([start, end])
    else: merged[-1][1] = max(merged[-1][1], end)
  gpu_us = sum(end - start for start, end in merged) if intervals else None
  selected_sum_us = sum(float(row["wall_us"]) for row in selected) if selected else None
  families: dict[str, dict[str, float | int]] = {}
  for row in selected:
    label = str(row.get("label") or "unlabeled")
    if re.match(r"^batched\s+\d+", label): family = "batched"
    elif (match := re.match(r"^([A-Za-z]+)_", label)): family = match.group(1)
    elif label.startswith("Command Buffer"): family = "command_buffer"
    else: family = label.split(":", 1)[0]
    item = families.setdefault(family, {"count": 0, "sum_us": 0.0})
    item["count"] = int(item["count"]) + 1
    item["sum_us"] = float(item["sum_us"]) + float(row.get("wall_us") or 0.0)
  return {
    "provider_id": trace.get("provider_id"),
    "whole_step_wall_us": whole_us,
    "tok_s": _as_float(whole.get("tok_s")),
    "command_buffer_count": len(rows),
    "compute_command_buffer_count": sum(row.get("channel") == "Compute" and "Blit" not in str(row.get("label")) for row in rows),
    "selected_command_buffer_count": len(selected),
    "selected_gpu_us": gpu_us,
    "selected_gpu_sum_us": selected_sum_us,
    "selected_overlap_us": selected_sum_us - gpu_us if selected_sum_us is not None and gpu_us is not None else None,
    "selected_gpu_pct_of_wall": 100.0 * gpu_us / whole_us if gpu_us is not None and whole_us else None,
    "host_or_unattributed_us": whole_us - gpu_us if gpu_us is not None and whole_us else None,
    "median_cpu_to_gpu_latency_us": statistics.median(latencies) if latencies else None,
    "selected_label_families": dict(sorted(families.items(), key=lambda item:-float(item[1]["sum_us"]))),
    "top_selected_command_buffers": [
      {"label": row.get("label"), "wall_us": row.get("wall_us"), "channel": row.get("channel")}
      for row in sorted(selected, key=lambda row:_as_float(row.get("wall_us")) or 0.0, reverse=True)[:8]
    ],
    "selected": selected,
  }


def command_buffer_compare_markdown(report:dict[str, Any]) -> str:
  comparison = report.get("command_buffer_compare", report)
  base, cand = comparison.get("baseline", {}), comparison.get("candidate", {})
  def value(row:dict[str, Any], key:str, digits:int = 3) -> str:
    raw = row.get(key)
    return "unavailable" if raw is None else f"{float(raw):.{digits}f}"
  lines = ["## Command-buffer attribution", "",
           "Runtime whole-step timing is the throughput authority; selected GPU time is interval-union attribution.", "",
           "| metric | baseline | candidate |", "|---|---:|---:|",
           f"| provider | {base.get('provider_id')} | {cand.get('provider_id')} |",
           f"| tok/s | {value(base, 'tok_s')} | {value(cand, 'tok_s')} |",
           f"| whole-step us | {value(base, 'whole_step_wall_us')} | {value(cand, 'whole_step_wall_us')} |",
           f"| selected command buffers | {base.get('selected_command_buffer_count')} | {cand.get('selected_command_buffer_count')} |",
           f"| selected GPU union us | {value(base, 'selected_gpu_us')} | {value(cand, 'selected_gpu_us')} |",
           f"| host/unattributed us | {value(base, 'host_or_unattributed_us')} | {value(cand, 'host_or_unattributed_us')} |",
           "",
           "### Candidate label families", "",
           "| family | buffers | summed GPU ms |", "|---|---:|---:|"]
  for family, item in cand.get("selected_label_families", {}).items():
    lines.append(f"| {family} | {item['count']} | {float(item['sum_us']) / 1000.0:.3f} |")
  return "\n".join(lines) + "\n"


def _counter_compare(baseline:dict[str, Any], candidate:dict[str, Any], *, context:int | None = None) -> dict[str, Any]:
  return {
    "baseline": _counter_summary(baseline, context=context),
    "candidate": _counter_summary(candidate, context=context),
  }


def _counter_summary(trace:dict[str, Any], *, context:int | None = None) -> dict[str, Any]:
  rows = []
  for row in trace.get("rows", []):
    if row.get("scope") != "kernel" or not isinstance(row.get("counters"), dict):
      continue
    if context is not None and row.get("context") != context:
      continue
    rows.append(row)
  totals: dict[str, float] = {}
  weights: dict[str, float] = {}
  for row in rows:
    wall_us = _as_float(row.get("wall_us")) or 0.0
    for key, value in row.get("counters", {}).items():
      if key not in NORMALIZED_COUNTERS:
        continue
      totals[key] = totals.get(key, 0.0) + float(value) * wall_us
      weights[key] = weights.get(key, 0.0) + wall_us
  weighted = {key: (totals[key] / weights[key]) for key in sorted(totals) if weights.get(key)}
  hot = sorted(rows, key=lambda r: _as_float(r.get("wall_us")) or 0.0, reverse=True)[:8]
  return {
    "provider_id": trace.get("provider_id"),
    "kernel_rows_with_counters": len(rows),
    "weighted_counters": weighted,
    "hot_kernels": [
      {
        "kernel": row.get("kernel"),
        "role": row.get("role"),
        "quant": row.get("quant"),
        "wall_us": row.get("wall_us"),
        "counters": row.get("counters"),
      }
      for row in hot
    ],
  }
