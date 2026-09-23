from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.policy.emit import emit_seed_policy
from boltbeam.quantization.quant_gemv import build_probe_request
from boltbeam.quantization.quant import route_families_for
from boltbeam.search.emit import emit_search_space
from boltbeam.synth.fixtures import emit_fixture_manifest
from boltbeam.target.targets import get_target
from boltbeam.trace.timing import build_trace_request
from boltbeam.vocab import SCHEMA_ANALYSIS_REPORT, SCHEMA_MEASUREMENT_PLAN, role_rank_priority
from boltbeam.workflow.common import load_manifest, load_profile, read_json, run_dir, update_manifest, write_json

ANALYZE_ARTIFACTS = ("search_space.json", "route_policy.json", "fixture_manifest.json",
                     "measurement_plan.json", "analysis_report.json", "probe_request.json", "trace_request.json")


def _rank_by_routability(role) -> tuple[float, int, int, str]:
  """Answers: which role can we act on right now, by route-family support?

  Quant priority is binary: any known route family beats none (F16 ranks before Q8_0 if, e.g., Q8_0 has no
  route family registered while F16 does) — derived from route_families_for, not a hardcoded per-quant
  table.

  This is a different question from boltbeam/analyze.py:_rank_by_memory_cost, which answers "which role is
  hottest to keep resident" by bytes/elem (Q8_0 ranks before F16 there). Both are correct; do not merge
  them. They share only the coarse-role tiebreak table, centralized in vocab.role_rank_priority.
  """
  supported = bool(route_families_for(role.quant))
  return (0 if supported else 1, role_rank_priority(role.role), -(role.rows * role.cols * role.count), role.tensor_name)


def _route_focus(search:dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
  focus: dict[str, list[dict[str, Any]]] = {}
  for role in search.get("roles", []):
    for family in role.get("route_families", []):
      focus.setdefault(family["family"], []).append({
        "role": role["role"],
        "shape": role["shape"],
        "quant": role["quant"],
        "status": family.get("status", "candidate"),
      })
  return focus


def _provider_summary(run:pathlib.Path) -> dict[str, Any]:
  path = run / "provider_capabilities.json"
  if not path.exists():
    return {"status": "not_scanned", "providers": []}
  caps = read_json(path)
  return {
    "status": "scanned",
    "providers": [
      {"provider_id": p["provider_id"], "status": p["status"], "capabilities": p.get("capabilities", {})}
      for p in caps.get("providers", [])
    ],
  }


def _primitive_profile(run:pathlib.Path) -> dict[str, Any] | None:
  path = run / "primitive_profile.json"
  return read_json(path) if path.exists() else None


def _timing_profile(run:pathlib.Path) -> dict[str, Any] | None:
  path = run / "timing_profile.json"
  return read_json(path) if path.exists() else None


def _regime_index(primitive_profile:dict[str, Any] | None) -> dict[tuple[str | None, str | None, tuple[int, ...]], dict[str, Any]]:
  idx: dict[tuple[str | None, str | None, tuple[int, ...]], dict[str, Any]] = {}
  for row in (primitive_profile or {}).get("quant_gemv_regimes", []):
    try:
      shape = tuple(int(x) for x in (row.get("shape") or ()))
    except (TypeError, ValueError):
      continue
    idx[(row.get("role"), row.get("quant"), shape)] = row
  return idx


def _role_timing_index(timing_profile:dict[str, Any] | None) -> dict[tuple[str | None, str | None, tuple[int, ...]], dict[str, Any]]:
  idx: dict[tuple[str | None, str | None, tuple[int, ...]], dict[str, Any]] = {}
  for row in (timing_profile or {}).get("role_timing", []):
    try:
      shape = tuple(int(x) for x in (row.get("shape") or ()))
    except (TypeError, ValueError):
      continue
    key = (row.get("role"), row.get("quant"), shape)
    if key not in idx or (row.get("pct_step") or 0.0) > (idx[key].get("pct_step") or 0.0):
      idx[key] = row
  return idx


def _measurement_plan(profile, target, search:dict[str, Any], provider_summary:dict[str, Any],
                      workload_profile:dict[str, Any], primitive_profile:dict[str, Any] | None,
                      probe_request:dict[str, Any], timing_profile:dict[str, Any] | None,
                      trace_request:dict[str, Any]) -> dict[str, Any]:
  regimes = _regime_index(primitive_profile)
  timings = _role_timing_index(timing_profile)
  regime_rank = {
    "dequant_bound": 0,
    "metadata_bound": 1,
    "occupancy_starved": 2,
    "latency_bound": 3,
    "streaming_bound": 4,
    "compute_bound": 5,
    "inconclusive": 8,
    None: 9,
  }
  priority = []
  for role in sorted(profile.roles, key=_rank_by_routability):
    regime = regimes.get((role.role, role.quant, (role.rows, role.cols)))
    timing = timings.get((role.role, role.quant, (role.rows, role.cols)))
    priority.append({
      "role": role.role,
      "tensor_name": role.tensor_name,
      "shape": [role.rows, role.cols],
      "quant": role.quant,
      "count": role.count,
      "route_families": list(route_families_for(role.quant)) or ["profile_first"],
      "quant_gemv_regime": regime["classification"] if regime else None,
      "visible_bottleneck": regime["visible_bottleneck"] if regime else None,
      "probe_id": regime["probe_id"] if regime else None,
      "timing_classification": timing["classification"] if timing else None,
      "timing_pct_step": timing.get("pct_step") if timing else None,
      "timing_wall_us": timing.get("wall_us") if timing else None,
      "timing_context": timing.get("context") if timing else None,
    })
  priority.sort(key=lambda r: (0 if r.get("timing_pct_step") is not None else 1,
                               regime_rank.get(r["quant_gemv_regime"], 7),
                               -(r.get("timing_pct_step") or 0.0), r["role"], r["shape"]))
  primitive_status = "classified" if primitive_profile else ("requested" if probe_request.get("probes") else "not_applicable")
  timing_status = "classified" if timing_profile else ("requested" if trace_request.get("priority_roles") else "not_applicable")
  return {
    "schema": SCHEMA_MEASUREMENT_PLAN,
    "model_id": profile.model_id,
    "target": target.to_json(),
    "workload": workload_profile.get("workload", "decode"),
    "contexts": workload_profile.get("contexts", []),
    "provider_summary": provider_summary,
    "primitive_profile": {
      "status": primitive_status,
      "artifact": "primitive_profile.json" if primitive_profile else None,
      "probe_request": "probe_request.json" if probe_request.get("probes") else None,
      "quant_gemv_regime_count": len((primitive_profile or {}).get("quant_gemv_regimes", [])),
    },
    "timing_profile": {
      "status": timing_status,
      "artifact": "timing_profile.json" if timing_profile else None,
      "trace_request": "trace_request.json" if trace_request.get("priority_roles") else None,
      "context_count": len((timing_profile or {}).get("contexts", [])),
      "dominant_timing_bucket": (timing_profile or {}).get("dominant_timing_bucket"),
      "role_timing_count": len((timing_profile or {}).get("role_timing", [])),
    },
    "phase_order": [
      {
        "id": "M0_validate_profile",
        "purpose": "Confirm role, shape, quant, and architecture facts before route measurement.",
        "required_evidence": ["profile_validation"],
      },
      {
        "id": "M1_probe_primitives",
        "purpose": "Measure provider/hardware primitives only where model facts expose route headroom.",
        "required_evidence": ["probe_evidence", "runtime_overhead"],
        "request_artifact": "probe_request.json" if probe_request.get("probes") else None,
      },
      {
        "id": "M2_trace_timing",
        "purpose": "Trace whole-step, per-role, per-kernel, and candidate timing before exploiting route families.",
        "required_evidence": ["timing_trace"],
        "request_artifact": "trace_request.json" if trace_request.get("priority_roles") else None,
      },
      {
        "id": "M3_measure_candidates",
        "purpose": "Measure prioritized route families against correctness, route binding, speed, memory, and rollback gates.",
        "required_evidence": ["wd_speed"],
      },
      {
        "id": "M4_decide",
        "purpose": "Promote, refute, defer, or request more evidence through the provider-neutral evaluator.",
        "required_evidence": ["candidate_decision"],
      },
    ],
    "priority_roles": priority,
    "route_focus": {"by_route_family": _route_focus(search)},
    "notes": [
      "This plan is provider-neutral. Provider adapters may translate phases into concrete commands.",
      "Output may remain a measurement request until enough normalized evidence exists to promote a route.",
    ],
  }


def _analysis_report(profile, search:dict[str, Any], policy:dict[str, Any],
                     measurement_plan:dict[str, Any], primitive_profile:dict[str, Any] | None,
                     timing_profile:dict[str, Any] | None) -> dict[str, Any]:
  blocked = []
  for role in search.get("roles", []):
    fams = role.get("route_families", [])
    if any(f.get("status") in ("unsupported_quant", "known_no_route") for f in fams):
      blocked.append({"role": role["role"], "quant": role["quant"], "families": fams})
  selected = [r for r in policy.get("routes", []) if r.get("selected_route")]
  regimes = (primitive_profile or {}).get("quant_gemv_regimes", [])
  timing = timing_profile or {}
  return {
    "schema": SCHEMA_ANALYSIS_REPORT,
    "model_id": profile.model_id,
    "architecture_class": profile.architecture_class,
    "status": "needs_measurement" if not selected else "policy_seeded",
    "complete": profile.is_complete,
    "role_count": len(profile.roles),
    "blocked_roles": blocked,
    "selected_routes": selected,
    "quant_gemv_regimes": regimes,
    "timing": {
      "status": timing.get("status", "not_ingested"),
      "dominant_timing_bucket": timing.get("dominant_timing_bucket"),
      "context_count": len(timing.get("contexts", [])),
      "role_timing_count": len(timing.get("role_timing", [])),
      "candidate_timing_count": len(timing.get("candidate_timing", [])),
    },
    "next_artifact": "measurement_plan.json",
    "measurement_phase_count": len(measurement_plan["phase_order"]),
  }


def analyze_run(run:str | pathlib.Path) -> dict[str, Any]:
  out = run_dir(run)
  manifest = load_manifest(out)
  profile = load_profile(out)
  target = get_target(manifest.get("target_id", "amd_gfx1100"))
  workload_profile = read_json(out / "workload_profile.json")
  search = emit_search_space(profile, target)
  policy = emit_seed_policy(profile, target)
  fixtures = emit_fixture_manifest(profile)
  providers = _provider_summary(out)
  primitive = _primitive_profile(out)
  timing = _timing_profile(out)
  probe_request = build_probe_request(profile, target, workload_profile)
  trace_request = build_trace_request(profile, target, workload_profile)
  plan = _measurement_plan(profile, target, search, providers, workload_profile, primitive, probe_request,
                           timing, trace_request)
  report = _analysis_report(profile, search, policy, plan, primitive, timing)
  write_json(out / "search_space.json", search)
  write_json(out / "route_policy.json", policy)
  write_json(out / "fixture_manifest.json", fixtures)
  write_json(out / "measurement_plan.json", plan)
  write_json(out / "analysis_report.json", report)
  write_json(out / "probe_request.json", probe_request)
  write_json(out / "trace_request.json", trace_request)
  return update_manifest(out, stage="analyze", artifacts=list(ANALYZE_ARTIFACTS))
