from __future__ import annotations

import copy
from typing import Any


def _key(row:dict[str, Any]) -> tuple[Any, ...]:
  return (
    row.get("kernel"), row.get("role"), row.get("quant"),
    tuple(row.get("shape") or ()), row.get("context"),
  )


def _semantic_key(row:dict[str, Any]) -> tuple[Any, ...]:
  return (row.get("role"), row.get("quant"), tuple(row.get("shape") or ()), row.get("context"))


def enrich_trace_resources(trace:dict[str, Any], resource_trace:dict[str, Any]) -> dict[str, Any]:
  out = copy.deepcopy(trace)
  by_key = {}
  by_semantic = {}
  by_kernel = {}
  for row in resource_trace.get("rows", []):
    if row.get("scope") != "kernel":
      continue
    payload = {k: copy.deepcopy(row[k]) for k in (
      "resources", "sources", "counters", "tile_oracle", "resource_constraints", "candidate_geometry",
    ) if k in row}
    if not payload:
      continue
    by_key[_key(row)] = payload
    by_semantic.setdefault(_semantic_key(row), payload)
    if row.get("kernel"):
      by_kernel.setdefault(row.get("kernel"), payload)

  matched = 0
  for row in out.get("rows", []):
    if row.get("scope") != "kernel":
      continue
    payload = by_key.get(_key(row)) or by_semantic.get(_semantic_key(row)) or by_kernel.get(row.get("kernel"))
    if not payload:
      continue
    for name, value in payload.items():
      row.setdefault(name, value)
    matched += 1

  out.setdefault("aux_sources", {})
  out["aux_sources"]["resource_trace_provider"] = resource_trace.get("provider_id")
  out["resource_enrichment"] = {
    "source_provider": resource_trace.get("provider_id"),
    "matched_kernel_rows": matched,
    "resource_rows_available": len(by_key),
  }
  return out
