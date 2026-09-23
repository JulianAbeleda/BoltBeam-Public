"""Minimal Nsight Compute CSV importer.

Parses a conservative column subset into ``boltbeam.hw_trace.v1`` and records
unsupported fields in trace notes.
"""

from __future__ import annotations

import csv
import pathlib
import re
from collections import OrderedDict
from typing import Any

from boltbeam.trace.hw_trace import NORMALIZED_COUNTERS
from boltbeam.vocab import SCHEMA_HW_TRACE


def _canon_key(name: str) -> str:
  normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
  normalized = re.sub(r"_+", "_", normalized)
  return normalized.strip("_")


def _as_float(raw: Any) -> float | None:
  if raw in (None, ""):
    return None
  clean = str(raw).replace(",", "").replace("%", "").strip()
  m = re.match(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", clean)
  if not m:
    return None
  try:
    return float(m.group(0))
  except ValueError:
    return None


def _as_int(raw: Any) -> int | None:
  value = _as_float(raw)
  if value is None:
    return None
  return int(value)


def _short_name(name: str) -> str:
  head = name.split("(", 1)[0].strip()
  if head.startswith("void "):
    return head[5:]
  return head


_KERNEL_KEYS = {
  "kernel",
  "kernel_name",
  "name",
  "kernel_function",
}

_TIME_KEYS = {
  "duration",
  "duration_us",
  "duration_ns",
  "duration_ms",
  "wall_time",
  "walltime",
  "kernel_duration",
  "kernel_time",
  "time",
}

_CALL_KEYS = {
  "calls",
  "call_count",
  "dispatch_count",
}

_COUNTER_ALIASES = {
  "occupancy_pct": {
    "occupancy",
    "occupancy_percent",
    "occupancy_pct",
    "active_warps_pct",
    "achieved_occupancy",
  },
  "memory_busy_pct": {
    "dram_throughput",
    "memory_busy_pct",
    "memory_busy_percent",
    "memory_util_pct",
    "mem_util",
    "memory_utilization",
  },
  "memory_stall_pct": {
    "memory_stall_pct",
    "memory_stall_percent",
    "mem_stall_pct",
    "memory_stall",
  },
  "l2_hit_pct": {
    "l2_hit",
    "l2_hit_pct",
    "l2_hit_percent",
    "l2_hit_rate",
    "l2_hit_rate_percent",
  },
  "fetch_kb": {
    "read_kb",
    "fetch_kb",
    "dram_read_bytes",
    "read_bytes",
    "memory_read_bytes",
    "read",
  },
  "write_kb": {
    "write_kb",
    "dram_write_bytes",
    "write_bytes",
    "memory_write_bytes",
    "write",
  },
  "valu_busy_pct": {
    "valu_busy_pct",
    "valu_util_pct",
    "vector_unit_util",
  },
  "mfma_util_pct": {
    "mfma_util_pct",
    "mfma_utilization",
    "tensor_core_util_pct",
    "tensor_core_utilization",
    "hmma_util",
    "tensorcore_util",
  },
  "lds_conflict_pct": {
    "lds_conflict_pct",
    "lds_bank_conflict_pct",
    "lds_conflicts",
  },
  "lds_stall_pct": {
    "lds_stall_pct",
    "lds_stall_percent",
    "lds_stalls",
  },
}

_COUNTER_MAP = {alias: target for target, aliases in _COUNTER_ALIASES.items() for alias in aliases}

_WORKGROUP_PREFIXES = (
  "block_size",
  "blockdim",
  "workgroup",
  "threadblock_size",
)

_GRID_PREFIXES = (
  "grid_size",
  "grid",
  "griddim",
)

_RESOURCE_SCALAR_ALIASES = {
  "vgpr": {
    "vgpr",
    "vgpr_count",
    "registers_per_thread",
    "registers",
  },
  "sgpr": {
    "sgpr",
    "sgpr_count",
  },
  "lds_bytes": {
    "lds_bytes",
    "shared_memory_bytes",
    "shared_memory",
    "shared_mem",
    "smem_bytes",
    "l1_shared_bytes",
  },
  "scratch_bytes": {
    "scratch_bytes",
    "dynamic_shared_memory_bytes",
    "scratch_mem",
    "local_memory_bytes",
  },
}


def _as_time_us(canon_key: str, raw: Any) -> float | None:
  scalar = _as_float(raw)
  if scalar is None:
    return None
  key = canon_key
  if key.endswith("ns"):
    return scalar / 1000.0
  if key.endswith("ms"):
    return scalar * 1000.0
  raw_low = str(raw).lower()
  if " ns" in raw_low or raw_low.endswith("ns"):
    return scalar / 1000.0
  if " ms" in raw_low or raw_low.endswith("ms"):
    return scalar * 1000.0
  return scalar


_NCU_LONG_METRICS = {
  "duration": "Duration",
  "achieved_occupancy": "Achieved Occupancy",
  "dram_throughput": "DRAM Throughput",
  "l2_hit_rate": "L2 Hit Rate",
  "registers_per_thread": "Registers Per Thread",
}


def _ncu_unit_scalar(raw: Any, unit: str, *, target: str) -> float | None:
  value = _as_float(raw)
  if value is None:
    return None
  unit_key = _canon_key(unit)
  if target == "duration_us":
    if unit_key in {"ns", "nanosecond", "nanoseconds"}:
      return value / 1000.0
    if unit_key in {"ms", "millisecond", "milliseconds"}:
      return value * 1000.0
    if unit_key in {"s", "second", "seconds"}:
      return value * 1_000_000.0
    return value if unit_key in {"", "us", "microsecond", "microseconds"} else None
  if target == "bytes":
    if unit_key.startswith("kbyte") or unit_key in {"kb", "kib"}:
      return value * 1024.0
    if unit_key.startswith("mbyte") or unit_key in {"mb", "mib"}:
      return value * 1024.0 * 1024.0
    return value if unit_key.startswith("byte") or unit_key == "" else None
  if target == "percent":
    return value if unit_key in {"", "percent"} else None
  return value


def _pivot_long_ncu_rows(source_rows: list[dict[str, str]], notes: list[str]) -> list[dict[str, Any]]:
  """Pivot NCU 2026's one-metric-per-row export to conservative wide rows."""
  groups: OrderedDict[tuple[str, ...], dict[str, Any]] = OrderedDict()
  unsupported_metrics: set[str] = set()
  for row_i, row in enumerate(source_rows, start=2):
    # ID is launch-local only; process/context/stream make it stable in multi-process captures.
    identity = tuple(str(row.get(k, "")).strip() for k in
                     ("Process ID", "Context", "Stream", "ID"))
    group = groups.setdefault(identity, {
      "Kernel Name": row.get("Kernel Name", ""),
      "Block Size": row.get("Block Size", ""),
      "Grid Size": row.get("Grid Size", ""),
      "_metrics": {},
      "_shared": {},
    })
    if group["Kernel Name"] != row.get("Kernel Name", ""):
      notes.append(f"row {row_i}: conflicting kernel name for NCU launch {identity!r}; value ignored")
      continue
    rule_name = str(row.get("Rule Name", "")).strip()
    if rule_name:
      detail = str(row.get("Rule Description", "")).strip()
      rule_type = str(row.get("Rule Type", "")).strip()
      speedup = str(row.get("Estimated Speedup", "")).strip()
      speedup_type = str(row.get("Estimated Speedup Type", "")).strip()
      speedup_note = f"; estimated speedup {speedup} {speedup_type}" if speedup else ""
      notes.append((f"NCU rule [{rule_type or 'unknown'}] {rule_name}: {detail}".rstrip(": ") + speedup_note))
    metric_name = str(row.get("Metric Name", "")).strip()
    if not metric_name:
      continue
    metric_key = _canon_key(metric_name)
    unit = str(row.get("Metric Unit", "")).strip()
    raw_value = row.get("Metric Value", "")
    wide_key = _NCU_LONG_METRICS.get(metric_key)
    value: float | None = None
    if metric_key == "duration":
      value = _ncu_unit_scalar(raw_value, unit, target="duration_us")
      wide_key = "Duration (us)"
    elif metric_key in {"achieved_occupancy", "dram_throughput", "l2_hit_rate"}:
      value = _ncu_unit_scalar(raw_value, unit, target="percent")
    elif metric_key == "registers_per_thread":
      value = _as_float(raw_value)
    elif metric_key in {"dynamic_shared_memory_per_block", "static_shared_memory_per_block"}:
      value = _ncu_unit_scalar(raw_value, unit, target="bytes")
      if value is not None:
        group["_shared"][metric_key] = value
      continue
    else:
      unsupported_metrics.add(metric_name)
      continue
    if value is None:
      notes.append(f"row {row_i}: unsupported unit/value for NCU metric {metric_name!r}: {unit!r}/{raw_value!r}")
      continue
    previous = group["_metrics"].get(wide_key)
    if previous is not None and previous != value:
      notes.append(f"row {row_i}: conflicting duplicate NCU metric {metric_name!r}; launch skipped")
      group["_invalid"] = True
    else:
      group["_metrics"][wide_key] = value

  wide_rows: list[dict[str, Any]] = []
  for group in groups.values():
    if group.get("_invalid"):
      continue
    wide = {k: v for k, v in group.items() if not k.startswith("_")}
    wide.update(group["_metrics"])
    if group["_shared"]:
      wide["Shared Memory Bytes"] = sum(group["_shared"].values())
    wide_rows.append(wide)
  if unsupported_metrics:
    notes.append("Unsupported NCU metrics (ignored): " + ", ".join(sorted(unsupported_metrics)))
  return wide_rows


def _parse_kernel(row: dict[str, str]) -> str | None:
  for raw, value in row.items():
    if _canon_key(raw) in _KERNEL_KEYS:
      value = str(value).strip()
      if value:
        return _short_name(value)
  return None


def _parse_wall_us(row: dict[str, str], notes: list[str], row_i: int) -> float | None:
  for raw, value in row.items():
    canon = _canon_key(raw)
    if canon not in _TIME_KEYS and "time" not in canon and "duration" not in canon and "elapsed" not in canon:
      continue
    wall = _as_time_us(canon, value)
    if wall is None:
      continue
    return wall
  notes.append(f"row {row_i}: missing wall time; kernel row skipped")
  return None


def _consume(row_norm: dict[str, tuple[str, Any]], canon_name: str, used: set[str], parser) -> Any | None:
  hit = row_norm.get(canon_name)
  if hit is None:
    return None
  raw_key, raw_value = hit
  if raw_value in (None, ""):
    return None
  value = parser(raw_value)
  if value is None:
    return None
  used.add(raw_key)
  return value


def _parse_resources(row: dict[str, str], used: set[str]) -> dict[str, Any]:
  row_norm = {_canon_key(k): (k, v) for k, v in row.items()}
  resources: dict[str, Any] = {}

  def parse_axis(prefixes: tuple[str, ...], axis: str) -> int | None:
    for prefix in prefixes:
      for suffix in ("x", "y", "z") if axis == "" else (axis,):
        key = f"{prefix}_{suffix}"
        parsed = _consume(row_norm, key, used, _as_int)
        if parsed is not None:
          return parsed
    return None

  def parse_dim(prefixes: tuple[str, ...], kind: str) -> tuple[int, int, int] | None:
    axis = ["x", "y", "z"]
    values: list[int] = []
    for at in axis:
      val = parse_axis(prefixes, at)
      if val is None:
        # Fallback combined form for key like "block_size" or "grid_size" with "x,y,z".
        for prefix in prefixes:
          raw = row_norm.get(prefix)
          if raw is None:
            continue
          raw_key, raw_value = raw
          parts = re.split(r"[xX,* ]+", str(raw_value).strip().strip("()[]{}"))
          if len(parts) == 3:
            nums = [_as_int(p) for p in parts]
            if all(x is not None for x in nums):
              used.add(raw_key)
              return (int(nums[0]), int(nums[1]), int(nums[2]))
        return None
      values.append(val)
    return int(values[0]), int(values[1]), int(values[2])

  workgroup = parse_dim(_WORKGROUP_PREFIXES, "workgroup")
  if workgroup is not None:
    resources["workgroup"] = list(workgroup)
    resources["workgroup_threads"] = int(workgroup[0] * workgroup[1] * workgroup[2])

  grid = parse_dim(_GRID_PREFIXES, "grid")
  if grid is not None:
    resources["grid"] = list(grid)

  for target, aliases in _RESOURCE_SCALAR_ALIASES.items():
    parsed = None
    for alias in aliases:
      parsed = _consume(row_norm, alias, used, _as_float)
      if parsed is not None:
        break
    if parsed is not None:
      resources[target] = int(parsed) if target in {"vgpr", "sgpr"} else parsed

  return resources


def _is_bytes_counter(counter_key: str, value: Any) -> bool:
  key = _canon_key(counter_key)
  return "byte" in key


def _parse_counter(key: str, value: Any, used: set[str], notes: list[str], row_i: int) -> tuple[str, float] | None:
  canonical = _COUNTER_MAP.get(_canon_key(key))
  if canonical is None:
    return None
  value_f = _as_float(value)
  if value_f is None:
    notes.append(f"row {row_i}: counter {key!r} has non-numeric value")
    return None
  if canonical in {"fetch_kb", "write_kb"} and _is_bytes_counter(key, value):
    return canonical, value_f / 1024.0
  return canonical, value_f


def _parse_counters(row: dict[str, str], used: set[str], notes: list[str], row_i: int) -> dict[str, float]:
  counters: dict[str, float] = {}
  for raw_key, raw_value in row.items():
    if raw_key in used or not raw_value:
      continue
    parsed = _parse_counter(raw_key, raw_value, used, notes, row_i)
    if parsed is None:
      continue
    canonical, value = parsed
    counters[canonical] = value
    used.add(raw_key)
  return counters


def import_profiler_ncu_csv(
  path: str | pathlib.Path,
  *,
  model_id: str,
  target_id: str = "nvidia_unknown",
  workload: str = "decode",
  context: int | None = None,
  provider_id: str = "nvidia/ncu",
) -> dict[str, Any]:
  """Import an NCU CSV into a ``boltbeam.hw_trace.v1`` payload."""
  if not model_id:
    raise ValueError("model_id is required")
  csv_path = pathlib.Path(path)

  rows: list[dict[str, Any]] = []
  notes: list[str] = []
  unsupported_columns: set[str] = set()

  with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
      raise ValueError("NCU CSV is missing a header row")
    source_rows = list(reader)
    field_keys = {_canon_key(k) for k in reader.fieldnames}
    is_long_form = {"metric_name", "metric_unit", "metric_value"}.issubset(field_keys)
    if is_long_form:
      source_rows = _pivot_long_ncu_rows(source_rows, notes)

    for row_i, row in enumerate(source_rows, start=2):
      used: set[str] = set()
      kernel = _parse_kernel(row)
      if not kernel:
        notes.append(f"row {row_i}: missing kernel name; kernel row skipped")
        unsupported_columns.update(set(row.keys()))
        continue

      wall_us = _parse_wall_us(row, notes, row_i)
      if wall_us is None:
        unsupported_columns.update(set(row.keys()))
        continue

      kernel_row: dict[str, Any] = {
        "scope": "kernel",
        "kernel": kernel,
        "wall_us": wall_us,
      }

      if context is not None:
        kernel_row["context"] = int(context)

      row_norm = {_canon_key(k): (k, v) for k, v in row.items()}
      for call_key in _CALL_KEYS:
        calls = _consume(row_norm, call_key, used, _as_int)
        if calls is not None:
          kernel_row["calls"] = calls
          break

      resources = _parse_resources(row, used)
      if resources:
        kernel_row["resources"] = resources

      counters = _parse_counters(row, used, notes, row_i)
      if counters:
        kernel_row["counters"] = counters

      # mark known structural columns as consumed so unsupported columns
      # are truly non-mapped fields only.
      for raw_key in row:
        used.add(raw_key) if _canon_key(raw_key) in _KERNEL_KEYS else None
        used.add(raw_key) if _canon_key(raw_key) in _TIME_KEYS else None
        used.add(raw_key) if _canon_key(raw_key) in _CALL_KEYS else None
        used.add(raw_key) if _canon_key(raw_key) in _COUNTER_MAP else None
        used.add(raw_key) if _parse_dimension_key(raw_key) is not None else None

      unsupported_columns.update(set(row.keys()) - used)
      rows.append(kernel_row)

  if unsupported_columns:
    notes.append("Unsupported NCU columns (ignored): " + ", ".join(sorted(unsupported_columns)))

  out: dict[str, Any] = {
    "schema": SCHEMA_HW_TRACE,
    "model_id": model_id,
    "target_id": target_id,
    "workload": workload,
    "provider_id": provider_id,
    "contexts": [context] if context is not None else [],
    "trace_source": "ncu_csv",
    "source_path": str(csv_path),
    "aux_sources": {
      "ncu_csv": str(csv_path),
    },
    "counter_vocab": list(NORMALIZED_COUNTERS),
    "rows": rows,
    "notes": notes,
  }
  return out


def _parse_dimension_key(raw_key: str) -> str | None:
  key = _canon_key(raw_key)
  if key in {"x", "y", "z"}:
    return None
  if key.startswith(_WORKGROUP_PREFIXES) or key.startswith(_GRID_PREFIXES):
    return key
  if key in {"block_size", "grid_size", "workgroup", "grid"}:
    return key
  if key in {"vgpr", "sgpr", "registers", "registers_per_thread", "shared_memory", "shared_memory_bytes", "scratch_bytes",
             "dynamic_shared_memory_bytes", "shared_memory_size", "scratch_mem", "lds_bytes"}:
    return key
  for aliases in _RESOURCE_SCALAR_ALIASES.values():
    if key in aliases:
      return key
  if key in {"shared", "memory"}:
    return key
  return None
