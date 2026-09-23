from __future__ import annotations

import csv
import json
import pathlib
import re
from typing import Any

from boltbeam.vocab import SCHEMA_TIMING_TRACE
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS


_GGML_QUANTS = {
  12: "Q4_K",
  14: "Q6_K",
}

_ROCPROF_COPY_OPS = {
  2: "host_to_device",
  3: "device_to_host",
}


def _duration_us(row:dict[str, str]) -> float:
  if row.get("TotalDurationNs"):
    return float(row["TotalDurationNs"]) / 1000.0
  if row.get("Start_Timestamp") and row.get("End_Timestamp"):
    return (float(row["End_Timestamp"]) - float(row["Start_Timestamp"])) / 1000.0
  return 0.0


def _calls(row:dict[str, str]) -> int:
  try:
    return int(row.get("Calls") or 1)
  except ValueError:
    return 1


def _int_field(row:dict[str, str], *keys:str) -> int | None:
  for key in keys:
    value = row.get(key)
    if value in (None, ""):
      continue
    try:
      return int(float(value))
    except ValueError:
      continue
  return None


def resource_fields_from_backend_row(row:dict[str, str]) -> dict[str, Any]:
  """Normalize backend resource columns into provider-neutral kernel metadata.

  The fields are optional. Missing columns stay absent; BoltBeam must not turn missing resource evidence into zeroes.
  """
  resources: dict[str, Any] = {}
  wg = [_int_field(row, f"Workgroup_Size_{axis}", f"WorkGroup_Size_{axis}", f"Block_Size_{axis}") for axis in "XYZ"]
  grid = [_int_field(row, f"Grid_Size_{axis}") for axis in "XYZ"]
  if all(v is not None for v in wg):
    resources["workgroup"] = wg
    resources["workgroup_threads"] = int(wg[0] * wg[1] * wg[2])
  if all(v is not None for v in grid):
    resources["grid"] = grid
  scalar_fields = {
    "lds_bytes": ("LDS_Block_Size", "Lds_Block_Size", "group_segment_size"),
    "scratch_bytes": ("Scratch_Size", "Private_Segment_Size", "private_segment_size"),
    "vgpr": ("VGPR_Count", "Vgpr_Count", "vgpr"),
    "sgpr": ("SGPR_Count", "Sgpr_Count", "sgpr"),
  }
  for out_key, keys in scalar_fields.items():
    value = _int_field(row, *keys)
    if value is not None:
      resources[out_key] = value
  return resources


def _short_name(name:str) -> str:
  head = name.split("(", 1)[0].strip()
  if head.startswith("void "):
    head = head[5:]
  return head


def classify_llama_kernel(name:str) -> dict[str, Any]:
  """Classify llama.cpp ROCm kernel names into BoltBeam's provider-neutral timing buckets.

  rocprof kernel names do not include tensor names, so this adapter intentionally avoids pretending to know exact
  roles. It still exposes the dominant physical distinction we need for prefill comparison: quantized matmul vs
  attention vs norm/rope/activation/copy.
  """
  low = name.lower()
  quant = None
  m = re.search(r"ggml_type\)(\d+)", name)
  if m:
    quant = _GGML_QUANTS.get(int(m.group(1)), f"GGML_TYPE_{m.group(1)}")

  if "mul_mat_q<" in low:
    return {"kind": "gemm", "role": "quantized_matmul", "quant": quant}
  if "mul_mat_vec_q<" in low:
    return {"kind": "gemv", "role": "quantized_matvec", "quant": quant}
  if "dequantize_block_q6_k" in low:
    return {"kind": "dequant", "role": "dequant", "quant": "Q6_K"}
  if "dequantize_block_q4_k" in low:
    return {"kind": "dequant", "role": "dequant", "quant": "Q4_K"}
  if name.startswith("Cijk_"):
    return {"kind": "gemm", "role": "dequantized_matmul", "quant": "F16"}
  if "flash_attn" in low:
    return {"kind": "attention", "role": "attention"}
  if "rms_norm" in low or "norm" in low:
    return {"kind": "norm", "role": "norm"}
  if "rope" in low:
    return {"kind": "rope", "role": "rope"}
  if "silu" in low or "unary_gated" in low:
    return {"kind": "activation", "role": "ffn_activation"}
  if "quantize" in low:
    return {"kind": "cast", "role": "activation_quantize", "quant": "Q8_1"}
  if "convert_unary" in low:
    return {"kind": "cast", "role": "copy_cast"}
  if "copy" in low or "set_rows" in low or "get_rows" in low or "fillbuffer" in low:
    return {"kind": "copy", "role": "copy"}
  if "op_add" in low or "bcast" in low or "k_bin_bcast" in low:
    return {"kind": "elementwise", "role": "elementwise"}
  return {"kind": "elementwise", "role": "unknown"}


def _load_llama_bench(path:str | pathlib.Path | None) -> dict[str, Any] | None:
  if not path:
    return None
  data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
  if isinstance(data, list):
    return data[0] if data else None
  if isinstance(data, dict):
    return data
  return None


def _quant_weight_bytes(path:str | pathlib.Path | None) -> dict[str, float]:
  if not path:
    return {}
  data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
  out: dict[str, float] = {}
  for role in data.get("roles", []):
    quant = role.get("quant")
    if not quant:
      continue
    out[quant] = out.get(quant, 0.0) + float(role.get("estimated_weight_bytes") or 0.0)
  return out


def _apply_weight_bytes(rows:list[dict[str, Any]], quant_bytes:dict[str, float]) -> float | None:
  if not quant_bytes:
    return None
  raw_by_quant: dict[str, float] = {}
  for row in rows:
    if row.get("role") not in {"quantized_matmul", "quantized_matvec"}:
      continue
    quant = row.get("quant")
    if quant in quant_bytes:
      raw_by_quant[quant] = raw_by_quant.get(quant, 0.0) + float(row.get("raw_wall_us", row.get("wall_us", 0.0)))
  total = 0.0
  for row in rows:
    quant = row.get("quant")
    denom = raw_by_quant.get(quant, 0.0)
    if row.get("role") not in {"quantized_matmul", "quantized_matvec"} or quant not in quant_bytes or denom <= 0:
      continue
    share = float(row.get("raw_wall_us", row.get("wall_us", 0.0))) / denom
    row["phys_bytes"] = quant_bytes[quant] * share
    row["bytes_source"] = "estimated_weight_inventory_by_quant_time_weighted"
    total += row["phys_bytes"]
  return total


def _rocprof_transfer_rows(path:str | pathlib.Path | None, context:int | None,
                           memory_copy_csv:str | pathlib.Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  if memory_copy_csv:
    return _rocprof_transfer_rows_csv(memory_copy_csv, context)
  if not path:
    return [], {}
  root = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["rocprofiler-sdk-tool"][0]
  copies = root.get("buffer_records", {}).get("memory_copy", [])
  allocs = root.get("buffer_records", {}).get("memory_allocation", [])
  grouped: dict[str, dict[str, Any]] = {}
  for copy in copies:
    direction = _ROCPROF_COPY_OPS.get(copy.get("operation"), f"operation_{copy.get('operation')}")
    row = grouped.setdefault(direction, {
      "scope": "transfer",
      "kind": "copy",
      "role": direction,
      "kernel": f"MEMORY_COPY_{direction.upper()}",
      "wall_us": 0.0,
      "phys_bytes": 0.0,
      "calls": 0,
      "bytes_source": "rocprof_memory_copy_bytes",
      "time_source": "rocprof_memory_copy_trace",
    })
    row["calls"] += 1
    row["phys_bytes"] += float(copy.get("bytes") or 0.0)
    row["wall_us"] += (float(copy.get("end_timestamp") or 0.0) - float(copy.get("start_timestamp") or 0.0)) / 1000.0
  rows = list(grouped.values())
  if context is not None:
    for row in rows:
      row["context"] = context
  summary = {
    "source": str(path),
    "memory_copy_calls": sum(int(r["calls"]) for r in rows),
    "memory_copy_bytes": sum(float(r["phys_bytes"]) for r in rows),
    "memory_copy_us": sum(float(r["wall_us"]) for r in rows),
    "memory_allocation_calls": len(allocs),
  }
  return rows, summary


def _rocprof_transfer_rows_csv(path:str | pathlib.Path, context:int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  grouped: dict[str, dict[str, Any]] = {}
  src = pathlib.Path(path)
  with src.open(newline="") as f:
    for raw in csv.DictReader(f):
      direction = raw.get("Direction") or raw.get("Name") or "MEMORY_COPY"
      calls = _calls(raw)
      us = _duration_us(raw)
      if us == 0.0 and raw.get("Start_Timestamp") and raw.get("End_Timestamp"):
        us = (float(raw["End_Timestamp"]) - float(raw["Start_Timestamp"])) / 1000.0
      row = grouped.setdefault(direction, {
        "scope": "transfer",
        "kind": "copy",
        "role": direction.lower(),
        "kernel": direction,
        "wall_us": 0.0,
        "calls": 0,
        "time_source": "rocprof_memory_copy_csv",
      })
      row["calls"] += calls
      row["wall_us"] += us
      if raw.get("Bytes"):
        row["phys_bytes"] = float(row.get("phys_bytes") or 0.0) + float(raw["Bytes"])
        row["bytes_source"] = "rocprof_memory_copy_csv"
  rows = list(grouped.values())
  if context is not None:
    for row in rows:
      row["context"] = context
  summary = {
    "source": str(src),
    "memory_copy_calls": sum(int(r["calls"]) for r in rows),
    "memory_copy_us": sum(float(r["wall_us"]) for r in rows),
  }
  if any("phys_bytes" in r for r in rows):
    summary["memory_copy_bytes"] = sum(float(r.get("phys_bytes") or 0.0) for r in rows)
  return rows, summary


def _normalize_to_bench(rows:list[dict[str, Any]], bench:dict[str, Any] | None) -> tuple[float | None, float | None]:
  if not bench:
    return None, None
  wall_us = None
  if bench.get("avg_ns") is not None:
    wall_us = float(bench["avg_ns"]) / 1000.0
  raw_total = sum(float(r.get("wall_us") or 0.0) for r in rows)
  if wall_us is None or raw_total <= 0:
    return wall_us, None
  scale = wall_us / raw_total
  for row in rows:
    row["raw_wall_us"] = row["wall_us"]
    row["wall_us"] = row["wall_us"] * scale
    row["time_source"] = "rocprof_kernel_time_scaled_to_llama_bench_wall"
    row["time_normalization_scale"] = scale
  return wall_us, scale


def _freeze(value:Any) -> Any:
  if isinstance(value, dict):
    return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
  if isinstance(value, list):
    return tuple(_freeze(v) for v in value)
  return value


def _aggregate_kernel_rows(rows:list[dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
  order: list[tuple[Any, ...]] = []
  for row in rows:
    key = (
      row.get("scope"), row.get("kernel"), row.get("kind"), row.get("role"), row.get("quant"),
      tuple(row.get("shape") or ()), row.get("context"), _freeze(row.get("resources")),
    )
    if key not in grouped:
      grouped[key] = dict(row)
      order.append(key)
      continue
    cur = grouped[key]
    cur["wall_us"] = float(cur.get("wall_us") or 0.0) + float(row.get("wall_us") or 0.0)
    cur["calls"] = int(cur.get("calls") or 0) + int(row.get("calls") or 1)
  return [grouped[k] for k in order]


def timing_trace_from_rocprof_csv(path:str | pathlib.Path, *, model_id:str, target_id:str = "amd_gfx1100",
                                  workload:str = "prefill", context:int | None = None,
                                  provider_id:str = "llama.cpp/rocprofv3",
                                  peak_gbs:float = DEFAULT_PEAK_MEM_GBS,
                                  llama_bench_json:str | pathlib.Path | None = None,
                                  weight_inventory:str | pathlib.Path | None = None,
                                  rocprof_json:str | pathlib.Path | None = None,
                                  memory_copy_csv:str | pathlib.Path | None = None) -> dict[str, Any]:
  if not model_id:
    raise ValueError("model_id is required; pass --model-id or --run")
  src = pathlib.Path(path)
  rows = []
  with src.open(newline="") as f:
    for raw in csv.DictReader(f):
      name = raw.get("Name") or raw.get("Kernel_Name") or raw.get("name") or raw.get("kernel")
      if not name:
        continue
      us = _duration_us(raw)
      calls = _calls(raw)
      info = classify_llama_kernel(name)
      row = {
        "scope": "kernel",
        "kernel": _short_name(name),
        "kind": info["kind"],
        "role": info["role"],
        "wall_us": us,
        "calls": calls,
      }
      resources = resource_fields_from_backend_row(raw)
      if resources:
        row["resources"] = resources
      if context is not None:
        row["context"] = context
      if info.get("quant"):
        row["quant"] = info["quant"]
      rows.append(row)
  rows = _aggregate_kernel_rows(rows)

  bench = _load_llama_bench(llama_bench_json)
  bench_wall_us, scale = _normalize_to_bench(rows, bench)
  total_us = bench_wall_us if bench_wall_us is not None else sum(float(r["wall_us"]) for r in rows)
  total_bytes = _apply_weight_bytes(rows, _quant_weight_bytes(weight_inventory))
  transfer_rows, transfer_summary = _rocprof_transfer_rows(rocprof_json, context, memory_copy_csv)
  whole = {
    "scope": "whole_step",
    "wall_us": total_us,
    "launch_count": sum(int(r.get("calls", 1)) for r in rows),
    "time_source": "llama_bench_avg_ns" if bench_wall_us is not None else "rocprof_kernel_sum",
  }
  if total_bytes is not None:
    whole["total_bytes"] = total_bytes
    whole["bytes_source"] = "estimated_weight_inventory_by_quant_time_weighted"
  if transfer_summary:
    whole["transfer_us"] = transfer_summary["memory_copy_us"]
    whole["transfer_source"] = "rocprof_memory_copy_csv" if memory_copy_csv else "rocprof_json_memory_copy"
    if "memory_copy_bytes" in transfer_summary:
      whole["transfer_bytes"] = transfer_summary["memory_copy_bytes"]
  if scale is not None:
    whole["kernel_time_normalization_scale"] = scale
  if context is not None:
    whole["context"] = context
    if bench and bench.get("avg_ts") is not None:
      whole["tok_s"] = float(bench["avg_ts"])
      whole["tok_s_source"] = "llama_bench_avg_ts"
    elif total_us > 0:
      whole["tok_s"] = 1_000_000.0 * context / total_us
      whole["tok_s_source"] = whole["time_source"]

  return {
    "schema": SCHEMA_TIMING_TRACE,
    "model_id": model_id,
    "target_id": target_id,
    "workload": workload,
    "provider_id": provider_id,
    "timing_source": "profile",
    "peak_gbs": peak_gbs,
    "contexts": [context] if context is not None else [],
    "source_path": str(src),
    "aux_sources": {
      "llama_bench_json": str(llama_bench_json) if llama_bench_json else None,
      "weight_inventory": str(weight_inventory) if weight_inventory else None,
      "rocprof_json": str(rocprof_json) if rocprof_json else None,
      "memory_copy_csv": str(memory_copy_csv) if memory_copy_csv else None,
    },
    "transfer_summary": transfer_summary,
    "notes": [
      "Generated from rocprofv3 kernel stats/trace CSV for llama.cpp.",
      "Kernel names do not carry exact tensor roles; role fields are coarse buckets inferred from kernel names.",
      "When llama-bench JSON is provided, kernel rows are scaled to the measured llama-bench wall time and keep raw_wall_us.",
      "Bytes are estimated from BoltBeam weight inventory unless a future counter trace supplies measured HBM bytes.",
      "ROCm memory-copy records are imported as scope=transfer rows so model upload/copy traffic is visible without changing prefill kernel timing.",
    ],
    "rows": [whole] + rows + transfer_rows,
  }
