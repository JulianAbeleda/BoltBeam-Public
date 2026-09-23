from __future__ import annotations

import csv
import json
import pathlib
import re
from typing import Any

from boltbeam.artifacts.llama_rocprof import (
  _apply_weight_bytes, _duration_us, _load_llama_bench, _normalize_to_bench, resource_fields_from_backend_row,
)
from boltbeam.artifacts.llama_rocprof import _rocprof_transfer_rows
from boltbeam.vocab import SCHEMA_TIMING_TRACE
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS

_PREFILL_GEMM_RE = re.compile(r"prefill_(?:graph|gen_sched)_gemm_(\d+)_(\d+)_(\d+)")
_PREFILL_DIRECT_PACKED_RE = re.compile(r"prefill_(q4k|q6k)_direct_packed(?:_load)?(?:_direct_out)?_gemm_(\d+)_(\d+)_(\d+)_(\d+)")
_PREFILL_GENERATED_DIRECT_OUT_RE = re.compile(r"(q4k|q6k)_gen_prefill_direct_out_(\d+)_(\d+)_(\d+)(?:_|$)")
_PREFILL_Q4K_WMMA_RE = re.compile(
  r"prefill_q4k_q8_1_wmma(?:_tiled)?_generated_gemm_(attn_kv|attn_qo|ffn_down|ffn_gate_up)_(\d+)_(\d+)_(\d+)")


def _calls(row:dict[str, str]) -> int:
  try:
    return int(row.get("Calls") or 1)
  except ValueError:
    return 1


def _short_name(name:str) -> str:
  head = name.split("(", 1)[0].strip()
  if head.startswith("void "):
    head = head[5:]
  return head


def _shape_index(weight_inventory:str | pathlib.Path | None) -> dict[tuple[int, int], dict[str, Any]]:
  if not weight_inventory:
    return {}
  data = json.loads(pathlib.Path(weight_inventory).read_text(encoding="utf-8"))
  grouped: dict[tuple[int, int], dict[str, Any]] = {}
  for role in data.get("roles", []):
    shape = role.get("shape") or []
    if len(shape) != 2:
      continue
    key = (int(shape[0]), int(shape[1]))
    cur = grouped.setdefault(key, {
      "role": role.get("role"),
      "quant": role.get("quant"),
      "shape": [int(shape[0]), int(shape[1])],
      "estimated_weight_bytes": 0.0,
      "role_count": 0,
      "quants": set(),
      "roles": set(),
    })
    cur["estimated_weight_bytes"] += float(role.get("estimated_weight_bytes") or 0.0)
    cur["role_count"] += int(role.get("count") or 1)
    cur["quants"].add(role.get("quant"))
    cur["roles"].add(role.get("role"))
  out = {}
  for key, cur in grouped.items():
    roles = {x for x in cur.pop("roles") if x}
    quants = {x for x in cur.pop("quants") if x}
    cur["role"] = next(iter(roles)) if len(roles) == 1 else "quantized_matmul"
    cur["quant"] = next(iter(quants)) if len(quants) == 1 else "mixed"
    out[key] = cur
  return out


def classify_tinygrad_kernel(name:str, *, shape_index:dict[tuple[int, int], dict[str, Any]] | None = None) -> dict[str, Any]:
  low = name.lower()
  m = _PREFILL_Q4K_WMMA_RE.search(name)
  if m:
    role, n, k, mb = m.groups()
    return {"kind": "gemm", "role": role, "quant": "Q4_K", "shape": [int(mb), int(n), int(k)]}
  m = _PREFILL_GEMM_RE.search(name)
  if m:
    mb, n, k = (int(x) for x in m.groups())
    info = dict((shape_index or {}).get((n, k), {}))
    return {
      "kind": "gemm",
      "role": info.get("role") or "quantized_matmul",
      "quant": info.get("quant"),
      "shape": [mb, n, k],
    }
  m = _PREFILL_DIRECT_PACKED_RE.search(name)
  if m:
    quant_tag, n, k, mb, _parts = m.groups()
    mb, n, k = int(mb), int(n), int(k)
    info = dict((shape_index or {}).get((n, k), {}))
    kernel_quant = "Q4_K" if quant_tag == "q4k" else "Q6_K"
    return {
      "kind": "gemm",
      "role": info.get("role") or "quantized_matmul",
      "quant": kernel_quant,
      "shape": [mb, n, k],
    }
  m = _PREFILL_GENERATED_DIRECT_OUT_RE.search(name)
  if m:
    quant_tag, n, k, mb = m.groups()
    mb, n, k = int(mb), int(n), int(k)
    info = dict((shape_index or {}).get((n, k), {}))
    return {
      "kind": "gemm",
      "role": info.get("role") or "quantized_matmul",
      "quant": "Q4_K" if quant_tag == "q4k" else "Q6_K",
      "shape": [mb, n, k],
    }
  if "q4k" in low and ("gemm" in low or "matmul" in low):
    return {"kind": "gemm", "role": "quantized_matmul", "quant": "Q4_K"}
  if "q6k" in low and ("gemm" in low or "matmul" in low):
    return {"kind": "gemm", "role": "quantized_matmul", "quant": "Q6_K"}
  if "q4k" in low and "gemv" in low:
    return {"kind": "gemv", "role": "quantized_matvec", "quant": "Q4_K"}
  if "q6k" in low and "gemv" in low:
    return {"kind": "gemv", "role": "quantized_matvec", "quant": "Q6_K"}
  if "dequant" in low or "ggml_data_to_tensor" in low:
    return {"kind": "dequant", "role": "dequant"}
  if "flash" in low or "attn" in low:
    return {"kind": "attention", "role": "attention"}
  if "rope" in low:
    return {"kind": "rope", "role": "rope"}
  if "norm" in low or low.startswith("r_"):
    return {"kind": "norm", "role": "norm"}
  if "silu" in low or "gelu" in low:
    return {"kind": "activation", "role": "ffn_activation"}
  if "copy" in low or "cast" in low or low.startswith("d_"):
    return {"kind": "copy", "role": "copy_cast"}
  if low.startswith("e_"):
    return {"kind": "elementwise", "role": "elementwise"}
  return {"kind": "elementwise", "role": "unknown"}


def _load_tinygrad_wall(path:str | pathlib.Path | None) -> dict[str, Any] | None:
  if not path:
    return None
  data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
  for row in data.get("rows", []):
    if row.get("scope") == "whole_step":
      return {"avg_ns": float(row.get("wall_us") or 0.0) * 1000.0, "avg_ts": row.get("tok_s")}
  return None


def _apply_role_bytes(rows:list[dict[str, Any]], shape_index:dict[tuple[int, int], dict[str, Any]]) -> float | None:
  total = 0.0
  matched = False
  raw_by_shape: dict[tuple[int, int], float] = {}
  for row in rows:
    shape = row.get("shape") or []
    if len(shape) != 3:
      continue
    key = (int(shape[1]), int(shape[2]))
    if key in shape_index:
      raw_by_shape[key] = raw_by_shape.get(key, 0.0) + float(row.get("raw_wall_us", row.get("wall_us", 0.0)))
  for row in rows:
    shape = row.get("shape") or []
    if len(shape) != 3:
      continue
    key = (int(shape[1]), int(shape[2]))
    info = shape_index.get(key)
    denom = raw_by_shape.get(key, 0.0)
    if not info or denom <= 0:
      continue
    share = float(row.get("raw_wall_us", row.get("wall_us", 0.0))) / denom
    row["phys_bytes"] = float(info.get("estimated_weight_bytes") or 0.0) * share
    row["bytes_source"] = "estimated_weight_inventory_by_shape_time_weighted"
    total += row["phys_bytes"]
    matched = True
  return total if matched else None


def timing_trace_from_tinygrad_rocprof_csv(path:str | pathlib.Path, *, model_id:str, target_id:str = "amd_gfx1100",
                                           workload:str = "prefill", context:int | None = None,
                                           provider_id:str = "tinygrad/rocprofv3",
                                           peak_gbs:float = DEFAULT_PEAK_MEM_GBS,
                                           tinygrad_trace_json:str | pathlib.Path | None = None,
                                           weight_inventory:str | pathlib.Path | None = None,
                                           memory_copy_csv:str | pathlib.Path | None = None) -> dict[str, Any]:
  if not model_id:
    raise ValueError("model_id is required; pass --model-id or --run")
  src = pathlib.Path(path)
  shape_index = _shape_index(weight_inventory)
  rows = []
  with src.open(newline="") as f:
    for raw in csv.DictReader(f):
      name = raw.get("Name") or raw.get("Kernel_Name") or raw.get("name") or raw.get("kernel")
      if not name:
        continue
      info = classify_tinygrad_kernel(name, shape_index=shape_index)
      row = {
        "scope": "kernel",
        "kernel": _short_name(name),
        "kind": info["kind"],
        "role": info["role"],
        "wall_us": _duration_us(raw),
        "calls": _calls(raw),
      }
      resources = resource_fields_from_backend_row(raw)
      if resources:
        row["resources"] = resources
      if context is not None:
        row["context"] = context
      if info.get("quant"):
        row["quant"] = info["quant"]
      if info.get("shape"):
        row["shape"] = info["shape"]
      rows.append(row)

  wall = _load_tinygrad_wall(tinygrad_trace_json)
  bench_wall_us, scale = _normalize_to_bench(rows, wall)
  total_us = bench_wall_us if bench_wall_us is not None else sum(float(r["wall_us"]) for r in rows)
  total_bytes = _apply_role_bytes(rows, shape_index)
  if total_bytes is None:
    total_bytes = _apply_weight_bytes(rows, {})
  transfer_rows, transfer_summary = _rocprof_transfer_rows(None, context, memory_copy_csv)
  whole = {
    "scope": "whole_step",
    "wall_us": total_us,
    "launch_count": sum(int(r.get("calls", 1)) for r in rows),
    "time_source": "tinygrad_trace_wall_us" if bench_wall_us is not None else "rocprof_kernel_sum",
  }
  if total_bytes is not None:
    whole["total_bytes"] = total_bytes
    whole["bytes_source"] = "estimated_weight_inventory_by_shape_time_weighted"
  if transfer_summary:
    whole["transfer_us"] = transfer_summary["memory_copy_us"]
    whole["transfer_source"] = "rocprof_memory_copy_csv"
    if "memory_copy_bytes" in transfer_summary:
      whole["transfer_bytes"] = transfer_summary["memory_copy_bytes"]
  if scale is not None:
    whole["kernel_time_normalization_scale"] = scale
  if context is not None:
    whole["context"] = context
    if wall and wall.get("avg_ts") is not None:
      whole["tok_s"] = float(wall["avg_ts"])
      whole["tok_s_source"] = "tinygrad_trace_tok_s"
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
      "tinygrad_trace_json": str(tinygrad_trace_json) if tinygrad_trace_json else None,
      "weight_inventory": str(weight_inventory) if weight_inventory else None,
      "memory_copy_csv": str(memory_copy_csv) if memory_copy_csv else None,
    },
    "transfer_summary": transfer_summary,
    "notes": [
      "Generated from rocprofv3 kernel stats/trace CSV for tinygrad.",
      "When tinygrad timing JSON is provided, kernel rows are scaled to its synced wall time and keep raw_wall_us.",
      "Memory-copy CSV rows include copy timing/calls; bytes are present only if rocprof emits a Bytes column.",
      "Exact tensor role attribution is shape-derived from BoltBeam weight inventory where possible.",
    ],
    "rows": [whole] + rows + transfer_rows,
  }
