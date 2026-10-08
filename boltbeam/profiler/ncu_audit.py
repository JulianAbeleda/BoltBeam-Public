"""Side-by-side NCU audit (ours vs a reference, per projection shape) with a research queue.

Per (role, M) measured on both sides: the GEMM kernel and the auxiliary kernels (split-K reductions, hi/lo sums),
their launch facts and counters, the logical op's roofline, and the gap decomposed into KNOWN limiters --
  * ``aux_kernels``: auxiliary kernel time (ours - reference);
  * ``rows``: our GEMM runs ``gemm_rows_per_token`` rows per token (hi/lo); the model floor ratio of the derived
    space at M vs at the rows we run prices the extra rows;
  * ``wave_tail``: SM quantization of each GEMM's measured grid (idle SM share of its time), ours - reference;
  * ``stall_excess:<reason>``: for each named stall reason, the excess share of ours over the reference applied to
    our remaining busy GEMM time;
-- and the remainder is reported as ``unexplained`` with its size, never folded into another bucket (negative means
the known buckets over-attribute).  Buckets plus ``unexplained`` sum exactly to the gap.  Shapes are ranked into a
research queue by unexplained share of our time.

Technique detection: each kernel's SASS opcode census (ncu ``sass__inst_executed_per_opcode`` instances, or a
``cuobjdump -sass`` listing) is mapped through ``data/sass_technique_taxonomy.json``.  An opcode the reference executes
that the taxonomy does not map is an UNKNOWN TECHNIQUE (with counts and kernel name); a mapped technique the reference
uses and ours does not is a technique gap, annotated with the lowering fact that says whether we could emit it.
"""
from __future__ import annotations

import json, math, pathlib, re
from collections import Counter
from typing import Any, Iterable, Mapping

from boltbeam.profiler.ncu_counters import _CENSUS, load_counters

SCHEMA = "boltbeam.ncu_audit.v1"
TAXONOMY_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "sass_technique_taxonomy.json"
NAMED_STALLS = ("mio_throttle", "barrier", "long_scoreboard", "short_scoreboard", "lg_throttle", "math_pipe_throttle",
                "wait", "not_selected", "dispatch_stall", "no_instruction", "tex_throttle", "drain", "membar", "branch_resolving")


# ---- opcode census and techniques ---------------------------------------------------------------------------------

_SASS_LINE = re.compile(r"^\s+/\*[0-9a-f]+\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*)")


def census_from_ncu(value:str) -> dict[str, int]:
  """``sass__inst_executed_per_opcode`` exported with instance values: ``TOTAL (HMMA: 1718528; LDSM: 647736; ...)``."""
  return {op: int(n) for op, n in _CENSUS.findall(value or "")}


def census_from_sass(text:str) -> dict[str, int]:
  """Static census from ``cuobjdump -sass`` text: one count per instruction in the listing."""
  return dict(Counter(m.group(1) for line in text.splitlines() if (m := _SASS_LINE.match(line))))


def load_taxonomy(path:str | pathlib.Path=TAXONOMY_PATH) -> dict[str, Any]:
  data = json.loads(pathlib.Path(path).read_text())
  if data.get("schema") != "boltbeam.sass_technique_taxonomy.v1": raise ValueError("not a SASS technique taxonomy")
  opcode_to = {}
  for technique, spec in data["techniques"].items():
    for op in spec["opcodes"]:
      if op in opcode_to: raise ValueError(f"opcode {op} mapped twice")
      opcode_to[op] = technique
  return {"techniques": data["techniques"], "opcode_to_technique": opcode_to}


def technique_diff(reference:Mapping[str, int], ours:Mapping[str, int] | None, taxonomy:Mapping[str, Any], *,
                   lowering:Mapping[str, Any] | None=None, reference_kernel:str | None=None) -> dict[str, Any]:
  to_tech = taxonomy["opcode_to_technique"]
  unknown = [{"opcode": op, "count": n, "kernel": reference_kernel} for op, n in sorted(reference.items()) if op not in to_tech and n > 0]
  gaps = []
  if ours is not None:
    per = lambda census: Counter({t: 0 for t in taxonomy["techniques"]}) + Counter(
      {to_tech[op]: n for op, n in census.items() if op in to_tech})
    ref_t, ours_t = per(reference), per(ours)
    for technique, n in sorted(ref_t.items()):
      fact = taxonomy["techniques"][technique]["lowering_fact"]
      if fact is not None and n > 0 and ours_t.get(technique, 0) == 0:   # only strategy choices, not epilogue noise
        gaps.append({"technique": technique, "reference_count": n, "lowering_fact": fact,
                     "lowering_can_emit": None if fact is None or lowering is None else bool(lowering.get(fact))})
  return {"unknown_techniques": unknown, "technique_gaps": gaps}


# ---- per-shape side by side ---------------------------------------------------------------------------------------

def _ctas(grid:str | None) -> int | None:
  nums = [int(x) for x in re.findall(r"\d+", grid or "")]
  return math.prod(nums) if nums else None


def _sm_efficiency(ctas:int | None, sm_count:int) -> float | None:
  if not ctas: return None
  return ctas / (math.ceil(ctas / sm_count) * sm_count)


def _stall_shares(row:Mapping[str, Any]) -> dict[str, float]:
  return {s["reason"]: float(s["pct"]) for s in row.get("top_stall_reasons", ())}


def _side(rows:list[dict[str, Any]]) -> dict[str, Any]:
  gemms = [r for r in rows if not r.get("aux")]
  gemm = max(gemms, key=lambda r: r["duration_us"])   # the representative launch for counters/grid
  total, gemm_us = sum(r["duration_us"] for r in rows), sum(r["duration_us"] for r in gemms)
  return {"gemm": gemm, "total_us": total, "gemm_us": gemm_us, "aux_us": total - gemm_us,
          "kernels": len(rows), "gemm_launches": len(gemms)}


def audit(counters:Iterable[Mapping[str, Any]], dims:Mapping[str, tuple[int, int]], *, machine=None, lowering=None,
          gemm_rows_per_token:int=2, census:Mapping[str, Mapping[str, int]] | None=None, taxonomy:Mapping[str, Any] | None=None) -> dict[str, Any]:
  """``counters``: counters documents (either side, either schema); ``dims``: role -> (N, K) of the logical op;
  ``census``: kernel name (or prefix) -> opcode census for kernels whose counters carry none."""
  from boltbeam.search.gemm_strategy import Shape, derive_space, estimate, prune, roofline_us
  taxonomy = taxonomy or load_taxonomy()
  groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
  for doc in counters:
    for r in load_counters(doc)["rows"]:
      if r.get("role") is None: continue
      groups.setdefault((r["side"], r["role"], r["m"]), []).append(r)
  sm_count = machine.sm_count if machine is not None else None
  lowering_facts = None if lowering is None else {f: getattr(lowering, f) for f in lowering.__dataclass_fields__}

  def floor_us(m, n, k):
    shape = Shape(m, n, k)
    kept = prune(machine, lowering, shape, derive_space(machine, lowering, shape))
    return estimate(machine, lowering, shape, kept[0]).model_us if kept else None

  def census_of(row):
    if row.get("opcode_census"): return row["opcode_census"]
    for key, value in (census or {}).items():
      if (row["kernel"] or "").startswith(key) or key in (row["kernel"] or ""): return value
    return None

  out = []
  for (side, role, m), rows in sorted(groups.items()):
    if side != "ours" or ("reference", role, m) not in groups or role not in dims: continue
    ours, ref = _side(rows), _side(groups[("reference", role, m)])
    n, k = dims[role]
    row: dict[str, Any] = {"role": role, "m": m, "n": n, "k": k, "gemm_rows": m * gemm_rows_per_token}
    for label, s in (("ours", ours), ("reference", ref)):
      g = s["gemm"]
      row[label] = {"total_us": round(s["total_us"], 3), "gemm_us": round(s["gemm_us"], 3), "aux_us": round(s["aux_us"], 3),
                    "kernels": s["kernels"], "gemm_kernel": g["kernel"], "grid": g["grid"], "block": g["block"],
                    "registers_per_thread": g["registers_per_thread"],
                    "shared_bytes": None if g.get("static_shared_bytes") is None else g["static_shared_bytes"] + g["dynamic_shared_bytes"],
                    "achieved_occupancy_pct": g["achieved_occupancy_pct"], "tensor_pipe_util_pct": g["tensor_pipe_util_pct"],
                    "dram_tbps": round(g["dram_throughput_bytes_per_sec"] / 1e12, 3), "top_stall_reasons": g["top_stall_reasons"]}
    gap = ours["total_us"] - ref["total_us"]
    buckets: dict[str, float] = {"aux_kernels": ours["aux_us"] - ref["aux_us"]}
    rows_share = 0.0
    if machine is not None and lowering is not None:
      row["roofline_us"] = round(roofline_us(machine, lowering, Shape(m, n, k)), 3)
      if row["gemm_rows"] != m:
        f_m, f_rows = floor_us(m, n, k), floor_us(row["gemm_rows"], n, k)
        if f_m and f_rows: rows_share = max(0.0, 1.0 - f_m / f_rows)
    buckets["rows"] = ours["gemm_us"] * rows_share
    eff_o = _sm_efficiency(_ctas(ours["gemm"]["grid"]), sm_count) if sm_count else None
    eff_r = _sm_efficiency(_ctas(ref["gemm"]["grid"]), sm_count) if sm_count else None
    tail_o = ours["gemm_us"] * (1 - eff_o) if eff_o is not None else 0.0
    tail_r = ref["gemm_us"] * (1 - eff_r) if eff_r is not None else 0.0
    buckets["wave_tail"] = tail_o - tail_r
    busy = max(0.0, ours["gemm_us"] - buckets["rows"] - tail_o)
    so, sr = _stall_shares(ours["gemm"]), _stall_shares(ref["gemm"])
    for reason in NAMED_STALLS:
      excess = so.get(reason, 0.0) - sr.get(reason, 0.0)
      if excess > 0: buckets[f"stall_excess:{reason}"] = busy * excess / 100.0
    unexplained = gap - sum(buckets.values())
    row["gap_vs_reference"] = {"total_us": round(gap, 3), "known": {k2: round(v, 3) for k2, v in buckets.items()},
                               "unexplained_us": round(unexplained, 3),
                               "unexplained_share_of_ours": round(unexplained / ours["total_us"], 4) if ours["total_us"] else None}
    if "roofline_us" in row:
      gap_r = ours["total_us"] - row["roofline_us"]
      known_r = {"aux_kernels": ours["aux_us"], "rows": buckets["rows"], "wave_tail": tail_o}
      row["gap_vs_roofline"] = {"total_us": round(gap_r, 3), "known": {k2: round(v, 3) for k2, v in known_r.items()},
                                "unexplained_us": round(gap_r - sum(known_r.values()), 3)}
    ref_census, ours_census = census_of(ref["gemm"]), census_of(ours["gemm"])
    if ref_census:
      row["techniques"] = technique_diff(ref_census, ours_census, taxonomy, lowering=lowering_facts, reference_kernel=ref["gemm"]["kernel"])
    out.append(row)
  queue = sorted((r for r in out), key=lambda r: -abs(r["gap_vs_reference"]["unexplained_share_of_ours"] or 0.0))
  unknown = [dict(u, role=r["role"], m=r["m"]) for r in out for u in r.get("techniques", {}).get("unknown_techniques", ())]
  return {"schema": SCHEMA, "gemm_rows_per_token": gemm_rows_per_token, "named_stalls": list(NAMED_STALLS), "rows": out,
          "research_queue": {"unexplained": [{"role": r["role"], "m": r["m"], **{k2: r["gap_vs_reference"][k2] for k2 in
                                               ("unexplained_us", "unexplained_share_of_ours", "total_us")}} for r in queue],
                             "unknown_techniques": unknown,
                             "technique_gaps": [dict(g, role=r["role"], m=r["m"]) for r in out for g in r.get("techniques", {}).get("technique_gaps", ())]}}


def markdown(doc:Mapping[str, Any]) -> str:
  lines = ["| role | M | ours us (GEMM+aux) | vLLM us | roofline us | ours grid/regs/occ/TP/DRAM | ref grid/regs/occ/TP/DRAM | ours top stalls | gap us | known buckets | UNEXPLAINED us |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
  side = lambda s: f"{s['grid']} / {s['registers_per_thread']} / {s['achieved_occupancy_pct']:.0f}% / {s['tensor_pipe_util_pct']:.0f}% / {s['dram_tbps']}"
  for r in doc["rows"]:
    g = r["gap_vs_reference"]
    known = ", ".join(f"{k}={v}" for k, v in g["known"].items() if abs(v) >= 0.05)
    stalls = ", ".join(f"{s['reason']} {s['pct']:.0f}%" for s in r["ours"]["top_stall_reasons"])
    lines.append(f"| {r['role']} | {r['m']} | {r['ours']['total_us']} ({r['ours']['gemm_us']}+{r['ours']['aux_us']}) | {r['reference']['total_us']} | "
                 f"{r.get('roofline_us', '-')} | {side(r['ours'])} | {side(r['reference'])} | {stalls} | {g['total_us']} | {known} | {g['unexplained_us']} |")
  q = doc["research_queue"]
  lines += ["", "## Research queue", "", "Unexplained residual, largest share of our time first:", ""]
  lines += [f"- {u['role']} M={u['m']}: {u['unexplained_us']} us unexplained of a {u['total_us']} us gap "
            f"({(u['unexplained_share_of_ours'] or 0) * 100:.1f}% of ours)" for u in q["unexplained"][:15]]
  lines += ["", "Unknown techniques (reference opcodes the taxonomy does not map):", ""]
  lines += [f"- {u['opcode']} x{u['count']} in {u['kernel']} ({u['role']} M={u['m']})" for u in q["unknown_techniques"]] or ["- none"]
  lines += ["", "Technique gaps (reference uses, ours does not):", ""]
  lines += [f"- {g['technique']} ({g['role']} M={g['m']}): reference {g['reference_count']}; lowering fact {g['lowering_fact']} = "
            f"{g['lowering_can_emit']}" for g in q["technique_gaps"]] or ["- none"]
  return "\n".join(lines) + "\n"


__all__ = ["NAMED_STALLS", "SCHEMA", "audit", "census_from_ncu", "census_from_sass", "load_taxonomy", "markdown", "technique_diff"]
