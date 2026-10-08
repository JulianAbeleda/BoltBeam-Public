"""Per-shape GEMM scan table: derived strategy vs the promoted route vs the reference (vLLM/cuBLAS) vs the roofline.

One row per logical projection shape (role, M tokens): the GEMM rows our route actually runs (``gemm_rows_per_token``:
2 under hi/lo), the model-best derived candidate and its limiting factor, the promoted route (tinygrad's selection) and
the limiter the model assigns it, the best measured derived candidate when a scan was run, our production time and
the reference's time (both from the kernel audit's JSON), whether the derived space realizes the reference's kernel
choice (or which lowering capability it lacks), and the families the GPU admits but the lowering cannot emit.
Inputs are files; nothing here runs a GPU.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Iterable, Mapping

from boltbeam.search.gemm_strategy import (
  GemmLowering, GemmMachine, Shape, config_from_row, config_json, deferred, derive_space, estimate, explain_exclusion,
  parse_cutlass_kernel, prune, roofline_us, stream_k_proposals)

SCHEMA = "boltbeam.gemm_scan_table.v1"


def _reference_pick(row:Mapping[str, Any]) -> dict[str, Any] | None:
  for kernel in row.get("kernels", ()):
    pick = parse_cutlass_kernel(kernel["name"], kernel["grid"], kernel["block"])
    if pick is not None: return pick
  return None


def _best_measured(rows:Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
  best: dict[tuple[str, int], dict[str, Any]] = {}
  for row in rows:
    if "median_us" not in row or not row.get("finite", False) or row.get("max_rel_vs_unsplit", 0.0) > 1e-4: continue
    key = (row["role"], row["m"])
    if key not in best or row["median_us"] < best[key]["median_us"]: best[key] = dict(row)
  return best


def scan_table(machine:GemmMachine, lowering:GemmLowering, reference:Iterable[Mapping[str, Any]], *,
               selection:Iterable[Mapping[str, Any]]=(), ours:Iterable[Mapping[str, Any]]=(),
               measured:Iterable[Mapping[str, Any]]=(), gemm_rows_per_token:int=2, roles:set[str] | None=None) -> dict[str, Any]:
  promoted = {(r["role"], r["m"]): r for r in selection}
  prod = {(r["role"], r["m"]): r for r in ours if r.get("mode", "prod") == "prod"}
  best_measured = _best_measured(measured)
  out = []
  for ref in reference:
    role, m, n, k = ref["role"], ref["m"], ref["n"], ref["k"]
    if roles is not None and role not in roles: continue
    logical = Shape(m, n, k)
    gemm_rows = m * gemm_rows_per_token
    run = Shape(gemm_rows, n, k)
    space = derive_space(machine, lowering, run)
    kept = prune(machine, lowering, run, space)
    row: dict[str, Any] = {"role": role, "m": m, "n": n, "k": k, "gemm_rows": gemm_rows,
                           "roofline_us": round(roofline_us(machine, lowering, logical), 2),
                           "reference_us": round(ref["gpu_us"], 2) if ref.get("gpu_us") is not None else None,
                           "derived_admitted": len(space), "derived_kept": len(kept)}
    pick = _reference_pick(ref)
    if pick is not None:
      row["reference_pick"] = pick
      row["reference_pick_in_space"] = explain_exclusion(machine, lowering, logical, pick) or "admitted"
    if kept:
      e = estimate(machine, lowering, run, kept[0])
      row["derived_model_best"] = {**config_json(kept[0]), "model_us": e.model_us, "limiter": e.limiter, "ctas": e.ctas,
                                   "ctas_per_sm": e.ctas_per_sm, "waves": e.waves}
    if (p := promoted.get((role, gemm_rows))) is not None:
      c = config_from_row(p)
      e = estimate(machine, lowering, run, c)
      row["promoted"] = {**config_json(c), "search_median_us": p.get("search_median_us"), "limiter": e.limiter, "ctas": e.ctas,
                         "ctas_per_sm": e.ctas_per_sm, "waves": e.waves, "lds_bytes": e.lds_bytes, "in_derived_space": c in set(kept)}
      sk = stream_k_proposals(machine, lowering, run, [c])
      if sk: row["promoted"]["stream_k"] = sk[0]
    if (b := best_measured.get((role, gemm_rows))) is not None:
      row["derived_measured_best"] = {k2: b[k2] for k2 in ("geometry", "split_k", "median_us", "tflops") if k2 in b} | \
                                     {"pipeline": b.get("pipeline")}
    if (o := prod.get((role, m))) is not None:
      row["ours_prod_us"] = round(o["gpu_us"], 2)
      row["ours_gemm_us"] = round(o.get("gemm_us") or 0.0, 2) or None
    row["deferred"] = [d["family"] for d in deferred(machine, lowering, run)]
    if row.get("reference_us") and row.get("ours_prod_us"): row["ours_over_reference"] = round(row["ours_prod_us"] / row["reference_us"], 2)
    out.append(row)
  return {"schema": SCHEMA, "target_id": machine.target_id, "lowering": {"backend": lowering.backend, "arch": lowering.arch},
          "gemm_rows_per_token": gemm_rows_per_token, "rows": out}


def markdown(table:Mapping[str, Any]) -> str:
  head = ("| role | M | GEMM rows | roofline us | vLLM us | ours prod us | ours/vLLM | promoted route (limiter) | promoted us | "
          "derived measured best us | derived model best (limiter) | vLLM pick in derived space | deferred |")
  lines = [head, "|" + "---|" * (head.count("|") - 1)]
  fmt = lambda c: f"{'x'.join(map(str, c['geometry'][:3]))} w{c['geometry'][3]}x{c['geometry'][4]} s{c['split_k']} p{c['pipeline'][0]}{'r' if c['pipeline'][1] else ''}"
  for r in table["rows"]:
    pro, dm, mb = r.get("promoted"), r.get("derived_model_best"), r.get("derived_measured_best")
    lines.append("| " + " | ".join(str(x) for x in (
      r["role"], r["m"], r["gemm_rows"], r["roofline_us"], r.get("reference_us", "-"), r.get("ours_prod_us", "-"), r.get("ours_over_reference", "-"),
      f"{fmt(pro)} ({pro['limiter']})" if pro else "-", pro.get("search_median_us", "-") if pro else "-",
      mb["median_us"] if mb else "-", f"{fmt(dm)} ({dm['limiter']})" if dm else "-",
      r.get("reference_pick_in_space", "-"), ",".join(r["deferred"]) or "-")) + " |")
  return "\n".join(lines) + "\n"


__all__ = ["SCHEMA", "markdown", "scan_table"]
