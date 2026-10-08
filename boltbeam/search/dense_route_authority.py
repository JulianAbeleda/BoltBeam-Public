"""Authority records for tinygrad's promoted dense bf16 projection routes, joined from their qualification evidence.

Three producer documents, each validated strictly (tinygrad-arkey is the producer):
  * the gate (``dense-bf16-candidate-gate.v2``, extra/llm_research/prefill/dense_bf16_candidate_gate.py): per route,
    the candidate's programs vs the safe tensor-core path's, finiteness, error vs the fp32 oracle, bit-exactness,
    timings;
  * the selection (``dense-bf16-geometry-selection.v1``): per (role, rows) the searched geometry, split-K and, for the
    cp.async ring family, the pipeline (stages, async_copy, matrix_fragments, xor_swizzle);
  * the minted route table (``tinygrad.dense_bf16_candidate_routes.v1``): per exact shape the canonical identity of
    the candidate computing one K slice.

A route qualifies when the three agree on its identity/geometry/split and its gate row passes: finite, distinct
programs from the safe path, relative error vs the oracle within ``REL_ERROR_BOUND``, and -- when unsplit (no
reassociation of the K sum) -- bit-exact to the safe tensor-core path.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

SCHEMA = "boltbeam.dense_route_authority.v1"
GATE_SCHEMA = "dense-bf16-candidate-gate.v2"
SELECTION_SCHEMA = "dense-bf16-geometry-selection.v1"
ROUTES_SCHEMA = "tinygrad.dense_bf16_candidate_routes.v1"
# The producer's own selection bound for split-K reassociation (dense_bf16_geometry_search.select).
REL_ERROR_BOUND = 1e-4
_GATE_ROW = {"role", "m", "n", "k", "split_k", "geometry", "canonical_identity", "candidate_programs", "safe_tc_programs",
             "finite", "max_abs_err_vs_oracle", "max_rel_err_vs_oracle", "bitexact_vs_safe_tc", "distinct_programs",
             "candidate_median_us", "safe_tc_median_us", "candidate_tflops", "safe_tc_tflops", "candidate_pct_of_measured_peak"}


def _key(row:Mapping[str, Any]) -> tuple[str, int, int, int]: return (row["role"], row["m"], row["n"], row["k"])


def validate_gate(doc:Mapping[str, Any]) -> list[dict[str, Any]]:
  if doc.get("schema") != GATE_SCHEMA: raise ValueError(f"not a {GATE_SCHEMA} gate")
  if not isinstance(doc.get("rows"), list) or not doc["rows"]: raise ValueError("gate has no rows")
  seen = set()
  for row in doc["rows"]:
    if not isinstance(row, dict) or set(row) != _GATE_ROW: raise ValueError(f"gate row fields differ: {sorted(set(row) ^ _GATE_ROW)}")
    if _key(row) in seen: raise ValueError(f"duplicate gate row {_key(row)}")
    seen.add(_key(row))
  return [dict(row) for row in doc["rows"]]


def gate_verdict(row:Mapping[str, Any]) -> tuple[bool, list[str]]:
  failures = []
  if not row["finite"]: failures.append("non_finite")
  if not row["distinct_programs"]: failures.append("not_distinct_from_safe_path")
  if row["max_rel_err_vs_oracle"] > REL_ERROR_BOUND: failures.append("oracle_error")
  if row["split_k"] == 1 and not row["bitexact_vs_safe_tc"]: failures.append("unsplit_not_bitexact")
  return not failures, failures


def build_authority(gate:Mapping[str, Any], selection:Mapping[str, Any], routes:Mapping[str, Any], *, target:Mapping[str, Any],
                    sources:Mapping[str, str]) -> dict[str, Any]:
  if selection.get("schema") != SELECTION_SCHEMA: raise ValueError(f"not a {SELECTION_SCHEMA} selection")
  if routes.get("schema") != ROUTES_SCHEMA: raise ValueError(f"not a {ROUTES_SCHEMA} route table")
  selected = {(r["role"], r["m"]): r for r in selection["rows"]}
  minted = {_key(r): r for r in routes["routes"]}
  out = []
  for row in validate_gate(gate):
    sel, route = selected.get((row["role"], row["m"])), minted.get(_key(row))
    if sel is None or route is None: raise ValueError(f"gate row {_key(row)} has no selection or minted route")
    if (sel["n"], sel["k"], list(sel["geometry"]), sel["split_k"]) != (row["n"], row["k"], list(row["geometry"]), row["split_k"]):
      raise ValueError(f"gate row {_key(row)} disagrees with the selection")
    if route["canonical_identity"] != row["canonical_identity"] or route["split_k"] != row["split_k"]:
      raise ValueError(f"gate row {_key(row)} disagrees with the minted route")
    ok, failures = gate_verdict(row)
    pipeline = sel.get("pipeline")
    out.append({"route_id": f"dense_bf16.{row['role']}.m{row['m']}", "role": row["role"], "m": row["m"], "n": row["n"], "k": row["k"],
                "canonical_identity": row["canonical_identity"], "geometry": list(row["geometry"]), "split_k": row["split_k"],
                "family": "async_ring" if pipeline else "register_two_buffer", "pipeline": pipeline,
                "qualified": ok, "gate_failures": failures,
                "gate": {k: row[k] for k in ("finite", "distinct_programs", "bitexact_vs_safe_tc", "max_rel_err_vs_oracle",
                                             "candidate_median_us", "safe_tc_median_us")},
                "search_median_us": sel.get("search_median_us"), "ordinary_median_us": sel.get("ordinary_median_us"),
                "speedup_vs_safe_tc": round(row["safe_tc_median_us"] / row["candidate_median_us"], 2)})
  return {"schema": SCHEMA, "target": dict(target), "gate_schema": GATE_SCHEMA, "rel_error_bound": REL_ERROR_BOUND,
          "sources": dict(sources), "routes": sorted(out, key=lambda r: (r["role"], r["m"])),
          "summary": {"gated": len(out), "qualified": sum(r["qualified"] for r in out),
                      "async_ring": sum(r["family"] == "async_ring" for r in out)}}


def promoted_route_ids(authority:Mapping[str, Any], family:str | None=None) -> list[str]:
  return [r["route_id"] for r in authority["routes"] if r["qualified"] and (family is None or r["family"] == family)]


__all__ = ["GATE_SCHEMA", "REL_ERROR_BOUND", "ROUTES_SCHEMA", "SCHEMA", "SELECTION_SCHEMA", "build_authority", "gate_verdict",
           "promoted_route_ids", "validate_gate"]
