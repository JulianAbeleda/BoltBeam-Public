"""Model-aware profiler report over boltbeam.hw_trace.v1.

Emits an honest speed-of-light / roofline / causal report. Every derived metric carries provenance;
anything not measurable is `missing`/`unsupported` (never zero). Reuses the target registry for peaks.
See docs/profiler-ontology-architecture-20260704.md.
"""
from __future__ import annotations

from typing import Any

from boltbeam.target.targets import TARGETS, target_capability
from boltbeam.profiler import ontology as o
from boltbeam.math.roofline import dual_roofline_placement

SCHEMA = "boltbeam.profiler_report.v1"

# tinygrad-native-pmc / legacy counter names -> ontology concept keys we surface in a section.
_COUNTER_CONCEPT = {
  "occupancy_pct": "occupancy",
  "valu_busy_pct": "vector_unit",
  "tensor_core_util_pct": "matrix_unit",
  "mfma_util_pct": "matrix_unit",       # legacy alias; concept is matrix_unit
  "memory_busy_pct": "memory_level",
  "l2_hit_pct": "memory_level",
  "lds_conflict_pct": "memory_level",
}


def _f(v: Any) -> float | None:
  try:
    return None if v is None else float(v)
  except (TypeError, ValueError):
    return None


def _target_id(trace: dict[str, Any], override: str | None) -> str | None:
  tid = override or trace.get("target_id")
  # traces use "amd_gfx1100"; registry key is the same
  return tid if tid in TARGETS else (tid or None)


def _rows(trace: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
  whole = next((r for r in trace.get("rows", []) if r.get("scope") == "whole_step"), None)
  kernels = [r for r in trace.get("rows", []) if r.get("scope") == "kernel"]
  return whole, kernels


def _work_flops(row: dict[str, Any]) -> tuple[float | None, str]:
  # GEMM work: 2*M*N*K per dispatch, times call count. shape is [M,N,K].
  shape = row.get("shape") or []
  calls = int(row.get("calls") or 1)
  if row.get("kind") == "gemm" and len(shape) == 3:
    try:
      m, n, k = (int(x) for x in shape)
    except (TypeError, ValueError):
      return None, "unparseable shape"
    return 2.0 * m * n * k * calls, f"2*M*N*K*calls (shape={shape}, calls={calls})"
  return None, f"work model only defined for gemm rows (kind={row.get('kind')})"


def _tile_oracle_section(row: dict[str, Any]) -> dict[str, Any] | None:
  oracle = row.get("tile_oracle")
  if not isinstance(oracle, dict):
    return None
  out = dict(oracle)
  resources = row.get("resources") or {}
  measured = {k: resources.get(k) for k in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "workgroup_threads")
              if resources.get(k) is not None}
  if measured:
    out.setdefault("resources_measured", measured)
  if row.get("resource_constraints"):
    out.setdefault("resource_constraints", row["resource_constraints"])
  if row.get("candidate_geometry"):
    out.setdefault("candidate_geometry", row["candidate_geometry"])
  return out


def _oracle_geometry(oracle: dict[str, Any]) -> dict[str, Any]:
  geom = oracle.get("geometry") if isinstance(oracle.get("geometry"), dict) else {}
  cand = oracle.get("candidate_geometry") if isinstance(oracle.get("candidate_geometry"), dict) else {}
  merged = dict(geom)
  merged.update(cand)
  return merged


def _candidate_geometry_recommendation(section: dict[str, Any]) -> dict[str, Any] | None:
  oracle = section.get("cooperative_tile_oracle")
  if not isinstance(oracle, dict):
    return None
  geom = _oracle_geometry(oracle)
  candidate_id = oracle.get("candidate_id") or section["dispatch"].get("kernel") or "cooperative_tile_candidate"
  resources = oracle.get("resources_measured") or {}
  constraints = {
    "mmq_x": geom.get("mmq_x"),
    "mmq_y": geom.get("mmq_y"),
    "iter_k": geom.get("iter_k"),
    "nwarps": geom.get("nwarps") or geom.get("waves_per_workgroup"),
    "warp_size": geom.get("warp_size"),
    "tile_c_i": geom.get("tile_c_i"),
    "tile_c_j": geom.get("tile_c_j"),
    "scratch_bytes": {"eq": 0},
  }
  if resources.get("lds_bytes") is not None:
    constraints["lds_bytes"] = {"lte": resources["lds_bytes"]}
  if resources.get("vgpr") is not None:
    constraints["vgpr"] = {"lte": resources["vgpr"]}
  return {
    "id": f"{candidate_id}_coop_tile_constraints",
    "applies_to_kernel": section["dispatch"].get("kernel"),
    "route_family": oracle.get("route_family") or "llama_mmq_cooperative_tile",
    "action": "search",
    "constraints": {k: v for k, v in constraints.items() if v is not None},
    "reject_if": [
      "production_dispatch_changed != false",
      "duplicate_store_count > 0",
      "missing_store_count > 0",
      "scratch_bytes > 0",
    ],
    "evidence": ["tile_oracle", "hw_trace.resources" if resources else "hw_trace.resources_missing"],
  }


def _operand_path_rows(row: dict[str, Any]) -> list[dict[str, Any]] | None:
  """Render normalized kernel-analysis operand evidence without reclassifying it."""
  paths = row.get("operand_paths")
  if paths is None:
    return None
  if not isinstance(paths, dict):
    return []

  rendered = []
  for key, value in paths.items():
    if not isinstance(value, dict):
      continue
    static = value.get("static") if isinstance(value.get("static"), dict) else {}
    dynamic = value.get("dynamic") if isinstance(value.get("dynamic"), dict) else {}
    classification = value.get("classification") if isinstance(value.get("classification"), dict) else {}

    site_counts: dict[str, int] = {}
    for site in static.get("instruction_sites", []):
      if isinstance(site, dict) and isinstance(site.get("kind"), str):
        kind = site["kind"]
        site_counts[kind] = site_counts.get(kind, 0) + 1
    static_summary = {
      "attribution_status": static.get("attribution_status", "unknown"),
      "complete_control_flow": static.get("complete_control_flow"),
      "instruction_site_counts": site_counts,
      "compulsory_fetch_group_count": (len(static["compulsory_fetch_groups"])
        if isinstance(static.get("compulsory_fetch_groups"), list) else None),
    }
    for name in ("expected_semantic_bytes", "reload_factor", "fragment_bytes_per_lane",
                 "fragment_bytes_per_wave", "fragment_bytes_per_workgroup", "operand_lds_bytes",
                 "accumulator_register_bytes", "temporary_register_bytes", "spill_bytes", "barriers",
                 "waits", "loops", "pipeline_stages", "buffering"):
      if name in static:
        static_summary[name] = static[name]

    rendered.append({
      "operand_id": value.get("operand_id", str(key)),
      "semantic_role": value.get("semantic_role", str(key)),
      "scope": value.get("scope", "unknown"),
      "declared_strategy": value.get("declared_strategy", "unknown"),
      "classified_strategy": classification.get("strategy", "unknown"),
      "truth_status": classification.get("status", "unknown"),
      "confidence": classification.get("confidence", "unknown"),
      "static_evidence_summary": static_summary,
      "dynamic_serving_tiers": [dict(t) for t in dynamic.get("serving_tiers", []) if isinstance(t, dict)],
      "missing_evidence": list(classification.get("missing_discriminators", [])),
    })
  return rendered


def _kernel_section(row: dict[str, Any], whole: dict[str, Any] | None, target) -> dict[str, Any]:
  wall_us = _f(row.get("wall_us"))
  wall_s = (wall_us / 1e6) if wall_us else None
  whole_us = _f(whole.get("wall_us")) if whole else None
  counters = row.get("counters") or {}
  resources = row.get("resources") or {}

  peak_tf = (target.peak_tflops or {}).get("fp16") if target else None
  mem_bw = target.memory_bandwidth_gbs if target else None

  flops, flops_src = _work_flops(row)
  phys_bytes = _f(row.get("phys_bytes"))

  sec: dict[str, Any] = {
    "dispatch": {"kernel": row.get("kernel"), "kind": row.get("kind"), "calls": row.get("calls"),
                 "grid": resources.get("grid") or resources.get("global_size"),
                 "workgroup": resources.get("workgroup") or resources.get("local_size")},
    "program": {"role": row.get("role"), "quant": row.get("quant"), "shape": row.get("shape"),
                "vgpr": resources.get("vgpr"), "sgpr": resources.get("sgpr"),
                "lds_bytes": resources.get("lds_bytes"), "scratch_bytes": resources.get("scratch_bytes"),
                "operand_path": {
                  "static": ("explicit_lds_allocation" if (_f(resources.get("lds_bytes")) or 0) > 0
                             else "no_lds_allocation" if resources.get("lds_bytes") == 0 else "unknown"),
                  "dynamic_cache_residency": "unknown",
                  "note": "resource allocation is static evidence; L0/L1/L2/last-level/DRAM residency needs measured evidence",
                }},
    "time": {},
    "work_done": {}, "bytes_moved": {}, "speed_of_light": {}, "occupancy": {},
    "vector_unit": {}, "matrix_unit": {}, "memory_level": {}, "missing_evidence": [],
  }

  # time
  if wall_us is not None:
    sec["time"]["wall"] = o.metric(round(wall_us, 1), "us", "trace", o.MEASURED)
    if whole_us:
      sec["time"]["pct_step"] = o.metric(round(100.0 * wall_us / whole_us, 2), "%", "trace", o.DERIVED)

  # work_done / arithmetic intensity
  if flops is not None:
    sec["work_done"]["flops"] = o.metric(flops, "flop", flops_src, o.MODELED)
    if phys_bytes:
      sec["work_done"]["arithmetic_intensity"] = o.metric(round(flops / phys_bytes, 3), "flop/byte",
                                                          "flops/phys_bytes", o.DERIVED)
  # bytes_moved (weight bytes; model-derived, not a measured DRAM counter)
  if phys_bytes:
    sec["bytes_moved"]["dram_bytes"] = o.metric(phys_bytes, "byte", row.get("bytes_source") or "weight_inventory", o.MODELED)

  # speed-of-light
  sol = sec["speed_of_light"]
  if flops is not None and wall_s:
    ach_tf = flops / wall_s / 1e12
    sol["achieved_tflops"] = o.metric(round(ach_tf, 3), "tflops", "flops/wall", o.DERIVED)
    if peak_tf:
      sol["pct_peak_compute"] = o.metric(round(100.0 * ach_tf / peak_tf, 2), "%",
                                         f"vs fp16 peak {peak_tf} TFLOPS (post-dequant MAC)", o.DERIVED)
  if phys_bytes and wall_s:
    ach_gbs = phys_bytes / wall_s / 1e9
    sol["achieved_gbs"] = o.metric(round(ach_gbs, 2), "gb/s", "phys_bytes/wall", o.DERIVED)
    if mem_bw:
      sol["pct_peak_mem"] = o.metric(round(100.0 * ach_gbs / mem_bw, 2), "%", f"vs {mem_bw} GB/s HBM", o.DERIVED)

  # measured hardware efficiency concepts from counters, else missing
  present_concepts = set()
  for cname, cval in counters.items():
    concept = _COUNTER_CONCEPT.get(cname)
    if concept and _f(cval) is not None:
      sec[concept][cname] = o.metric(_f(cval), "%", "tinygrad_pmc", o.MEASURED)
      present_concepts.add(concept)

  # occupancy proxy from SQ busy cycles if no occupancy_pct counter
  if "occupancy" not in present_concepts and counters.get("sq_busy_cycles") is not None:
    sec["occupancy"]["sq_busy_cycles"] = o.metric(_f(counters["sq_busy_cycles"]), "cycles",
                                                  "tinygrad_pmc", o.MEASURED)
    present_concepts.add("occupancy")

  # missing/unsupported evidence. As of the amdgpu-dkms 6.16 driver these counters ARE collectable
  # on gfx11, so a concept is "missing" only when this particular trace didn't carry it.
  matrix_unit = target_capability(target.target_id, "matrix_unit") if target else None
  for concept, need in (("occupancy", "occupancy/residency"), ("vector_unit", "valu_busy_pct"),
                        ("memory_level", "l2/memory counters")):
    if concept not in present_concepts:
      sec["missing_evidence"].append({"concept": concept, "need": need,
                                      "reason": "counter not present in this trace"})
  if "matrix_unit" not in present_concepts:
    if matrix_unit:
      sec["missing_evidence"].append({"concept": "matrix_unit", "need": "tensor_core_util_pct",
                                      "reason": f"target has {matrix_unit.upper()} but a matrix-unit util counter was not collected"})
    else:
      sec["matrix_unit"] = o.unsupported("target has no matrix unit")

  tile_oracle = _tile_oracle_section(row)
  if tile_oracle:
    sec["cooperative_tile_oracle"] = tile_oracle

  operand_paths = _operand_path_rows(row)
  if operand_paths is not None:
    sec["operand_paths"] = operand_paths

  sec["classification"] = _classify_kernel(sol, sec, target)
  return sec


def _measured(block: dict[str, Any], key: str) -> float | None:
  m = block.get(key)
  return m.get("value") if isinstance(m, dict) else None


def _classify_kernel(sol: dict[str, Any], sec: dict[str, Any], target) -> dict[str, Any]:
  pc = sol.get("pct_peak_compute", {}).get("value")
  pm = sol.get("pct_peak_mem", {}).get("value")
  if pm is not None and pm >= 50:
    return {"limiter": "memory_bandwidth_bound", "confidence": "medium",
            "evidence": f"{pm}% of HBM peak"}
  if pc is not None and pc >= 50:
    return {"limiter": "compute_bound", "confidence": "medium", "evidence": f"{pc}% of fp16 peak"}

  # Both speed-of-light numbers low. Use the measured efficiency counters (now unlocked on gfx11)
  # to name the limiter instead of declaring the cause unresolved.
  occ = _measured(sec["occupancy"], "occupancy_pct")
  valu = _measured(sec["vector_unit"], "valu_busy_pct")
  l2 = _measured(sec["memory_level"], "l2_hit_pct")
  ev = [f"pct_peak_compute={pc}", f"pct_peak_mem={pm}"]
  if occ is not None: ev.append(f"occupancy={occ:.0f}%")
  if valu is not None: ev.append(f"valu_busy={valu:.0f}%")
  if l2 is not None: ev.append(f"l2_hit={l2:.0f}%")

  if occ is None and valu is None and l2 is None:
    return {"limiter": "low_efficiency_cause_unresolved", "confidence": "low",
            "evidence": "; ".join(ev) + "; both low",
            "missing_reason": "occupancy/VALU/memory efficiency counters not collected in this trace"}

  low_occ = occ is not None and occ < 60.0
  low_valu = valu is not None and valu < 50.0
  l2_resident = l2 is not None and l2 >= 90.0
  if l2_resident and (low_occ or low_valu):
    limiter, detail = "low_efficiency_underutilized", (
      "working set stays resident in L2 (high l2_hit, negligible HBM traffic) so this is not "
      "HBM-bandwidth bound; the low occupancy/VALU occupancy points to under-filled execution "
      "units — latency/occupancy/codegen bound, not proportional extra work")
  elif low_occ:
    limiter, detail = "occupancy_bound", "measured occupancy below the fill threshold with both speed-of-light numbers low"
  elif low_valu:
    limiter, detail = "vector_unit_underutilized", "VALU busy fraction low with both speed-of-light numbers low"
  else:
    limiter, detail = "low_efficiency_cause_unresolved", "efficiency counters present but none crosses a decisive threshold"
  return {"limiter": limiter, "confidence": "medium", "evidence": "; ".join(ev), "detail": detail,
          "measured_efficiency_counters": True}


def _overall(trace, whole, sections, baseline_summary) -> dict[str, Any]:
  hot = sections[0] if sections else None
  if baseline_summary is None:
    verdict = "insufficient_single_trace"
    cls = (hot or {}).get("classification", {})
    if cls.get("measured_efficiency_counters"):
      detail = ("single trace: hot-role speed-of-light + roofline placement produced, and per-kernel "
                f"efficiency counters ARE measured (limiter={cls.get('limiter')}: {cls.get('detail','')}). "
                "Separating 'more work' from 'lost efficiency' still needs a baseline (8B or llama) for "
                "an apples-to-apples comparison.")
    else:
      detail = ("single trace: hot-role speed-of-light + roofline placement produced, but attributing "
                "'more work' vs 'lost efficiency' needs a baseline (8B or llama), and this trace did not "
                "carry per-kernel efficiency counters (occupancy/VALU/tensor/memory).")
  else:
    verdict = baseline_summary.get("verdict", "insufficient")
    detail = baseline_summary.get("detail", "")
  return {"verdict": verdict, "detail": detail,
          "hot_kernel": hot["dispatch"]["kernel"] if hot else None,
          "hot_role": hot["program"]["role"] if hot else None}


def profiler_report(trace: dict[str, Any], *, target_id: str | None = None, top: int = 5,
                    baseline: dict[str, Any] | None = None,
                    roofline_input: dict[str, Any] | None = None,
                    tiered_roofline: dict[str, Any] | None = None) -> dict[str, Any]:
  if trace.get("schema") not in ("boltbeam.hw_trace.v1",):
    raise ValueError(f"expected boltbeam.hw_trace.v1, got {trace.get('schema')!r}")
  tid = _target_id(trace, target_id)
  target = TARGETS.get(tid) if tid else None
  whole, kernels = _rows(trace)
  kernels = sorted(kernels, key=lambda r: _f(r.get("wall_us")) or 0.0, reverse=True)
  hot = kernels[:top]
  sections = [_kernel_section(r, whole, target) for r in hot]
  recommendations = [rec for rec in (_candidate_geometry_recommendation(s) for s in sections) if rec is not None]

  baseline_summary = _baseline_compare(trace, hot, baseline) if baseline else None
  roofline = None
  if roofline_input is not None:
    for key in ("model_id", "workload"):
      expected, actual = roofline_input.get(key), trace.get(key)
      if expected is not None and expected != actual:
        raise ValueError(f"roofline input {key}={expected!r} does not match trace {actual!r}")
    roofline = dual_roofline_placement(
      measured=float(roofline_input["measured"]), raw_ceiling=float(roofline_input["raw_ceiling"]),
      practical_ceiling=float(roofline_input["practical_ceiling"]),
      metric=str(roofline_input.get("metric") or "tok_s"), raw_basis=roofline_input.get("raw_basis"),
      practical_basis=roofline_input.get("practical_basis"),
    )
  hierarchy = None
  if tiered_roofline is not None:
    if tiered_roofline.get("model_id") not in (None, trace.get("model_id")):
      raise ValueError(f"tiered roofline model_id={tiered_roofline.get('model_id')!r} does not match trace {trace.get('model_id')!r}")
    workload = str(trace.get("workload") or "")
    selected = tiered_roofline.get(workload) if workload in ("prefill", "decode") else None
    if not isinstance(selected, dict):
      raise ValueError(f"tiered roofline has no {workload!r} workload section")
    hierarchy = {
      "truth_status": tiered_roofline.get("truth_status", "unknown"),
      "workload": workload,
      "context": selected.get("context"),
      "modeled_tok_s": selected.get("tok_s"),
      "tiers": selected.get("tiers") or {},
      "roles": selected.get("roles") or [],
      "assumptions": selected.get("assumptions") or [],
      "dynamic_residency": "modeled_not_measured",
    }

  out = {
    "schema": SCHEMA,
    "model_id": trace.get("model_id"),
    "target_id": tid,
    "target_known": target is not None,
    "provider_id": trace.get("provider_id"),
    "workload": trace.get("workload"),
    "whole_step": {"wall_us": _f(whole.get("wall_us")) if whole else None,
                   "tok_s": _f(whole.get("tok_s")) if whole else None} if whole else None,
    "top_kernels": sections,
    "candidate_geometry_recommendations": recommendations,
    "overall": _overall(trace, whole, sections, baseline_summary),
    "notes": [
      "Derived metrics carry provenance (measured/derived/modeled). Missing concepts are explicit.",
      "bytes_moved and work_done are model/weight-derived, not measured DRAM/FLOP counters.",
    ],
  }
  if roofline is not None:
    out["roofline"] = roofline
  if hierarchy is not None:
    out["memory_hierarchy"] = hierarchy
  return out


def _m(metric: dict[str, Any] | None) -> str:
  if not metric or metric.get("value") is None:
    return f"_{(metric or {}).get('status', 'missing')}_"
  return f"{metric['value']} {metric.get('unit','')}".strip()


def profiler_report_markdown(report: dict[str, Any]) -> str:
  L = [f"# Profiler report: {report.get('model_id')} ({report.get('provider_id')})", "",
       f"- target: `{report.get('target_id')}` (known: {report.get('target_known')})",
       f"- workload: `{report.get('workload')}`"]
  ws = report.get("whole_step")
  if ws:
    if report.get("roofline"):
      L.append(f"- PROFILE capture (instrumented, not throughput authority): wall={ws.get('wall_us')} us, "
               f"derived tok/s={ws.get('tok_s')}")
    else:
      L.append(f"- whole step: wall={ws.get('wall_us')} us, tok/s={ws.get('tok_s')}")
  roof = report.get("roofline")
  if roof:
    raw, practical = roof["raw_hardware"], roof["practical"]
    L += ["", "## Roofline placement", "",
          f"- authority: `{roof['measured']:.3f} {roof['metric']}`",
          f"- Raw hardware roofline: `{raw['ceiling']:.3f} {roof['metric']}` — "
          f"`{raw['measured_pct_of_ceiling']:.1f}%`, `{raw['headroom_x']:.2f}x` headroom; {raw.get('basis')}",
          f"- Practical roofline: `{practical['ceiling']:.3f} {roof['metric']}` — "
          f"`{practical['measured_pct_of_ceiling']:.1f}%`, `{practical['headroom_x']:.2f}x` headroom; "
          f"{practical.get('basis')}"]
  hierarchy = report.get("memory_hierarchy")
  if hierarchy:
    tiers = hierarchy.get("tiers") or {}
    L += ["", "## Operand path and memory hierarchy", "",
          f"- truth status: `{hierarchy.get('truth_status')}`; dynamic residency: `{hierarchy.get('dynamic_residency')}`",
          f"- L2: `{tiers.get('l2_mib')} MiB`; last-level cache: `{tiers.get('last_level_cache_name')}` "
          f"`{tiers.get('mall_mib')} MiB`; DRAM: `{tiers.get('dram_gbs')} GB/s`",
          f"- cache bandwidth status: L2=`{tiers.get('l2_bandwidth_status')}`, "
          f"last-level=`{tiers.get('mall_bandwidth_status')}`; {tiers.get('calibration')}", "",
          "| role | transport | physical weight | tile M | fetch groups | fit tier band | serving tier | regime |",
          "|---|---|---:|---:|---:|---|---|---|"]
    for role in hierarchy.get("roles", []):
      band = "/".join(role.get("cache_tier_candidates") or [str(role.get("cache_tier") or "unknown")])
      L.append(f"| {role.get('role')} | {role.get('transport')} | {role.get('weight_mib')} MiB | "
               f"{role.get('tile_m')} | {role.get('global_weight_fetch_groups')} | {band} "
               f"({role.get('cache_tier_confidence')}) | {role.get('serving_tier')} | {role.get('regime')} |")
  ov = report.get("overall", {})
  L += ["", f"## Verdict: `{ov.get('verdict')}`", "", ov.get("detail", ""),
        f"\nHot kernel: `{ov.get('hot_kernel')}` (role `{ov.get('hot_role')}`)", "", "## Top kernels", ""]
  for s in report.get("top_kernels", []):
    d, p, sol, cls = s["dispatch"], s["program"], s["speed_of_light"], s.get("classification", {})
    L.append(f"### {d.get('kernel')}  —  role `{p.get('role')}` {p.get('quant') or ''} {p.get('shape')}")
    L.append(f"- time: {_m(s['time'].get('wall'))}, {_m(s['time'].get('pct_step'))} of step; calls={d.get('calls')}")
    path = p.get("operand_path") or {}
    L.append(f"- operand path: static `{path.get('static')}`; dynamic cache residency `{path.get('dynamic_cache_residency')}`")
    for operand in s.get("operand_paths", []):
      static = operand.get("static_evidence_summary") or {}
      tiers = operand.get("dynamic_serving_tiers") or []
      tier_text = ", ".join(f"{t.get('tier', 'unknown')}:{t.get('status', 'unknown')}" for t in tiers) or "unknown"
      missing = ", ".join(str(x) for x in operand.get("missing_evidence", [])) or "none"
      L.append(f"  - operand `{operand.get('operand_id')}` ({operand.get('semantic_role')}): declared "
               f"`{operand.get('declared_strategy')}`, classified `{operand.get('classified_strategy')}` "
               f"({operand.get('confidence')}/{operand.get('truth_status')}); static={static}; "
               f"serving tiers={tier_text}; missing={missing}")
    L.append(f"- speed-of-light: compute {_m(sol.get('achieved_tflops'))} ({_m(sol.get('pct_peak_compute'))}), "
             f"mem {_m(sol.get('achieved_gbs'))} ({_m(sol.get('pct_peak_mem'))}), AI={_m(s['work_done'].get('arithmetic_intensity'))}")
    ru = ", ".join(f"{c}={_m(list(s[c].values())[0]) if s[c] else None}" for c in ("occupancy", "vector_unit", "matrix_unit", "memory_level"))
    L.append(f"- hw: {ru}")
    L.append(f"- **limiter: `{cls.get('limiter')}`** ({cls.get('confidence')}) — {cls.get('evidence') or cls.get('missing_reason','')}")
    if s.get("missing_evidence"):
      L.append(f"- missing: " + "; ".join(f"{m['concept']} ({m['reason']})" for m in s["missing_evidence"]))
    if s.get("cooperative_tile_oracle"):
      oracle = s["cooperative_tile_oracle"]
      geom = oracle.get("geometry") or {}
      L.append(f"- cooperative tile oracle: `{oracle.get('candidate_id') or oracle.get('backend_id') or oracle.get('kind')}` "
               f"geometry={geom}")
    L.append("")
  if report.get("candidate_geometry_recommendations"):
    L.append("## Candidate geometry recommendations")
    L.append("")
    for rec in report["candidate_geometry_recommendations"]:
      L.append(f"- `{rec.get('id')}`: action={rec.get('action')}, constraints={rec.get('constraints')}")
  return "\n".join(L) + "\n"


def _baseline_compare(trace, hot, baseline) -> dict[str, Any]:
  # Compare hot roles by work vs achieved time to separate more-work from lost-efficiency.
  _, base_kernels = _rows(baseline)
  base_by_role = {}
  for r in base_kernels:
    base_by_role.setdefault(r.get("role"), r)
  rows = []
  more_work = lost_eff = 0
  cand_kernels = _rows(trace)[1]
  cand_by_role = {}
  for r in cand_kernels:
    cand_by_role.setdefault(r.get("role"), r)
  for role, ck in cand_by_role.items():
    bk = base_by_role.get(role)
    if not bk:
      continue
    cf, _ = _work_flops(ck); bf, _ = _work_flops(bk)
    cw = _f(ck.get("wall_us")); bw = _f(bk.get("wall_us"))
    if not (cf and bf and cw and bw):
      continue
    work_ratio = cf / bf
    time_ratio = cw / bw
    eff_ratio = work_ratio / time_ratio if time_ratio else None  # >1 => candidate MORE efficient/FLOP
    # Direction matters: eff_ratio<0.85 is a real efficiency LOSS (candidate slower per FLOP);
    # eff_ratio>1.18 is a favorable GAIN (bigger GEMM amortizes overhead), not a regression.
    if eff_ratio is None:
      continue
    if eff_ratio < 0.85:
      verdict = "efficiency_loss"; lost_eff += 1
    elif eff_ratio > 1.18:
      verdict = "more_work_efficiency_gain"; more_work += 1
    else:
      verdict = "more_work"; more_work += 1
    rows.append({"role": role, "work_ratio": round(work_ratio, 3), "time_ratio": round(time_ratio, 3),
                 "efficiency_ratio": round(eff_ratio, 3), "verdict": verdict})
  overall = ("proportional_more_work" if more_work and not lost_eff
             else "efficiency_loss" if lost_eff and not more_work
             else "mixed" if rows else "insufficient")
  return {"verdict": overall,
          "detail": f"compared {len(rows)} matched roles vs baseline {baseline.get('model_id')}",
          "per_role": rows}
