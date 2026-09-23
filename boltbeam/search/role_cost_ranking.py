"""Fail-closed MR7 role-cost screening from semantic census and exact isolated evidence.

Outer graph durations are enclosing bounds only: this module never divides a graph duration among its members.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from boltbeam.core.canonical import canonical_json as _canonical, sha256_json as _digest
from boltbeam.search.semantic.semantic_identity import SEMANTIC_IDENTITY_FIELDS, normalize_semantic_identity

SCHEMA = "boltbeam.role_cost_ranking.v1"
MEASUREMENTS_SCHEMA = "boltbeam.mr7_isolated_measurements.v1"
WHOLE_STEP_SCHEMA = "boltbeam.mr7_whole_step.v1"
ENCLOSURES_SCHEMA = "boltbeam.mr7_outer_enclosures.v1"
MR4_SUMMARY_SCHEMA = "boltbeam.replay_ab_summary.v1"
SEMANTIC_FIELDS = SEMANTIC_IDENTITY_FIELDS
PROGRAM_FIELDS = ("program_hash", "source_sha256", "binary_sha256")
PREREQUISITES = ("mr0", "mr4", "mr5", "mr6")
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")


def _finite_positive(value:Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _blocked(status:str, reason:str, prerequisites:Mapping[str, Any] | None = None) -> dict[str, Any]:
  return {"schema":SCHEMA, "status":status, "reason":reason, "prerequisites":dict(prerequisites or {}),
          "roles":[], "selected_roles":[]}


def _semantic(identity:Any) -> dict[str, Any] | None:
  try: return normalize_semantic_identity(identity)
  except ValueError: return None


def _program(record:Mapping[str, Any]) -> dict[str, str] | None:
  out = {field:record.get(field) for field in PROGRAM_FIELDS}
  return out if all(isinstance(value, str) and _SHA256.fullmatch(value) for value in out.values()) else None


def _execution_identity(program:Mapping[str, str], identities:Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  return {**program, "semantic_identities":sorted((dict(identity) for identity in identities), key=_canonical)}


def _samples(measurement:Any, *, minimum:int = 5) -> tuple[list[float], float] | None:
  if not isinstance(measurement, Mapping): return None
  raw = measurement.get("samples_ns")
  if not isinstance(raw, list) or len(raw) < minimum or not all(_finite_positive(value) for value in raw): return None
  values = [float(value) for value in raw]
  median = statistics.median(values)
  claimed = measurement.get("median_ns")
  if not _finite_positive(claimed) or float(claimed) != median: return None
  return values, median


def whole_step_from_mr4_summary(summary: Mapping[str, Any], arm: str) -> dict[str, Any]:
  """Adapt one complete live MR4 arm without discarding its five paired raw samples."""
  if arm not in ("control", "hybrid"): raise ValueError("MR4 arm must be control or hybrid")
  if summary.get("schema") != MR4_SUMMARY_SCHEMA or summary.get("status") != "complete":
    raise ValueError("MR7 requires a complete MR4 replay summary")
  if summary.get("evidence_readiness", {}).get("status") != "ready":
    raise ValueError("MR7 requires MR4 evidence readiness")
  order, required_order = summary.get("order"), summary.get("required_order")
  expected_order = [name for index in range(1, 6) for name in (f"control-{index:02d}", f"hybrid-{index:02d}")]
  if order != required_order or order != expected_order:
    raise ValueError("MR7 requires the exact five-pair MR4 interleaved order")
  stats = summary.get("paired_statistics")
  pairs = stats.get("pairs") if isinstance(stats, Mapping) else None
  if not isinstance(pairs, list) or stats.get("complete_pairs") != 5 or len(pairs) != 5:
    raise ValueError("MR7 requires five complete MR4 pairs")
  samples = []
  for expected_index,row in enumerate(pairs, 1):
    if not isinstance(row, Mapping) or row.get("pair_index") != expected_index:
      raise ValueError("MR4 pair identities are incomplete or duplicated")
    value = row.get(f"{arm}_wall_ns")
    if not _finite_positive(value): raise ValueError("MR4 arm wall sample is unavailable")
    samples.append(float(value))
  return {"schema":WHOLE_STEP_SCHEMA, "source_schema":MR4_SUMMARY_SCHEMA, "source_sha256":_digest(summary),
          "arm":arm, "samples_ns":samples, "median_ns":statistics.median(samples)}


def _reconcile_material(census:Mapping[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
  if census.get("schema") != "tinygrad.graph_admission_census.v1": return None, "unsupported_census_schema"
  records, counts = census.get("records"), census.get("counts")
  if not isinstance(records, list) or not isinstance(counts, Mapping): return None, "missing_census"
  try:
    logical = int(counts["logical_calls"])
    assigned = sum(int(counts.get(key, 0)) for key in ("graph_members", "direct_calls", "ignored_slice_nodes", "constructor_failures"))
  except (KeyError, TypeError, ValueError): return None, "malformed_census_counts"
  if logical != assigned or len(records) != logical: return None, "incomplete_census_reconciliation"
  if any(row.get("decision") == "unknown" or row.get("reason") == "unknown" for row in records if isinstance(row, Mapping)):
    return None, "unknown_admission_decision"
  material = []
  for row in records:
    if not isinstance(row, Mapping) or row.get("assignment") not in ("graph", "direct"): continue
    identities = [_semantic(value) for value in row.get("semantic_identities", [])]
    if row.get("metadata_unavailable") is True or row.get("metadata_status") != "semantic" or not identities or any(value is None for value in identities):
      return None, "material_call_metadata_unavailable"
    program = _program(row)
    if program is None: return None, "material_program_identity_unavailable"
    call_index = row.get("call_index")
    if not isinstance(call_index, int) or isinstance(call_index, bool): return None, "material_call_index_unavailable"
    outer = f"graph:{row.get('batch_index')}" if row.get("assignment") == "graph" else f"direct:{row.get('direct_call_index')}"
    material.append({"call_index":call_index, "outer_id":outer,
      "execution_identity":_execution_identity(program, [value for value in identities if value is not None]),
      "roles":sorted({value["role"] for value in identities if value is not None}),
      "placement":{"assignment":row.get("assignment"), "batch_index":row.get("batch_index"),
        "batch_member_index":row.get("batch_member_index"), "direct_call_index":row.get("direct_call_index")}})
  if not material: return None, "no_material_semantic_calls"
  if len({row["call_index"] for row in material}) != len(material): return None, "duplicate_material_call_index"
  return material, None


def reconcile_material_calls(census: Mapping[str, Any]) -> list[dict[str, Any]]:
  """Public exact MR5 adapter shared by ranking and evidence production."""
  material, error = _reconcile_material(census)
  if error is not None: raise ValueError(error)
  assert material is not None
  return material


def _measurement_map(measurements:Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
  out: dict[str, dict[str, Any]] = {}
  for row in measurements:
    identity = row.get("execution_identity") if isinstance(row, Mapping) else None
    if not isinstance(identity, Mapping): return None, "measurement_execution_identity_missing"
    program = _program(identity)
    semantic = [_semantic(value) for value in identity.get("semantic_identities", [])]
    if program is None or not semantic or any(value is None for value in semantic): return None, "measurement_execution_identity_invalid"
    normalized = _execution_identity(program, [value for value in semantic if value is not None])
    key = _digest(normalized)
    if key in out: return None, "many_to_one_measurement_join"
    timing = _samples(row.get("performance", {}).get("measurement") if isinstance(row.get("performance"), Mapping) else None)
    hashes = (row.get("correctness", {}).get("evidence_hash"), row.get("performance", {}).get("evidence_hash"),
      row.get("generated", {}).get("source_hash"), row.get("generated", {}).get("plan_hash"), row.get("generated", {}).get("binary_hash"))
    if timing is None: return None, "isolated_measurement_requires_five_exact_samples"
    if row.get("execution", {}).get("shape_mode") != "exact_workload": return None, "isolated_measurement_not_exact_workload"
    if row.get("correctness", {}).get("passed") is not True: return None, "isolated_measurement_correctness_failed"
    if not all(isinstance(value, str) and _SHA256.fullmatch(value) for value in hashes): return None, "isolated_evidence_hash_missing"
    if row["generated"]["source_hash"] != program["source_sha256"] or row["generated"]["binary_hash"] != program["binary_sha256"]:
      return None, "isolated_generated_identity_mismatch"
    samples, median = timing
    mad = statistics.median(abs(value-median) for value in samples)
    out[key] = {"identity":normalized, "samples_ns":samples, "median_ns":median,
      "relative_mad":mad/median, "evidence_hashes":list(hashes)}
  return out, None


def _outer_map(enclosures:Mapping[str, Any], material:list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
  if not isinstance(enclosures, Mapping) or enclosures.get("schema") != ENCLOSURES_SCHEMA:
    return None, "unsupported_outer_enclosure_schema"
  rows = enclosures.get("rows") if isinstance(enclosures, Mapping) else None
  if not isinstance(rows, list): return None, "outer_enclosures_missing"
  out = {}
  for row in rows:
    outer_id = row.get("outer_id") if isinstance(row, Mapping) else None
    samples = row.get("samples_ns") if isinstance(row, Mapping) else None
    if not isinstance(outer_id, str) or outer_id in out: return None, "outer_enclosure_identity_invalid"
    if not isinstance(samples, list) or len(samples) < 5 or not all(_finite_positive(value) for value in samples):
      return None, "outer_enclosure_requires_five_samples"
    evidence_hash = row.get("evidence_hash")
    if not isinstance(evidence_hash, str) or _SHA256.fullmatch(evidence_hash) is None:
      return None, "outer_enclosure_evidence_hash_missing"
    call_indices = row.get("call_indices")
    if not isinstance(call_indices, list) or any(not isinstance(value, int) for value in call_indices):
      return None, "outer_enclosure_call_indices_invalid"
    out[outer_id] = {"outer_id":outer_id, "samples_ns":[float(value) for value in samples],
                     "median_ns":statistics.median(samples), "call_indices":sorted(call_indices),
                     "evidence_hash":evidence_hash}
  expected = {row["outer_id"]:{entry["call_index"] for entry in material if entry["outer_id"] == row["outer_id"]}
              for row in material}
  if set(out) != set(expected): return None, "outer_enclosure_coverage_mismatch"
  if any(out[key]["call_indices"] != sorted(expected[key]) for key in expected): return None, "outer_enclosure_call_join_mismatch"
  return out, None


def _honest_classification(role:str, evidence:Mapping[str, Any] | None, mr4_hash:str) -> tuple[str, str]:
  if not isinstance(evidence, Mapping): return "unknown", "no controlled classification evidence"
  requested, proof = evidence.get("classification"), evidence.get("proof")
  points = evidence.get("timing_points")
  valid_points = isinstance(points, list) and len(points) >= 2 and all(isinstance(point, Mapping) and
    _finite_positive(point.get("median_ns")) for point in points)
  if requested == "memory_bound" and proof == "controlled_bytes_scaling" and evidence.get("compute_held_constant") is True \
      and _finite_positive(evidence.get("exact_packed_bytes")) and _finite_positive(evidence.get("exact_logical_bytes")) \
      and valid_points and all(_finite_positive(point.get("bytes")) for point in points) \
      and len({float(point["bytes"]) for point in points}) >= 2:
    return requested, "controlled exact-byte scaling"
  if requested == "compute_bound" and proof == "controlled_compute_scaling" and evidence.get("bytes_held_constant") is True \
      and _finite_positive(evidence.get("exact_semantic_operations")) and valid_points \
      and all(_finite_positive(point.get("operations")) for point in points) \
      and len({float(point["operations"]) for point in points}) >= 2:
    return requested, "controlled exact-operation scaling"
  if requested == "latency_dispatch_sensitive" and proof == "mr4_grouping_ab" and evidence.get("mr4_sha256") == mr4_hash \
      and evidence.get("compiled_identity_preserved") is True and evidence.get("resident_buffers_preserved") is True \
      and evidence.get("submission_count_reduced") is True and evidence.get("whole_wall_reduced") is True \
      and evidence.get("gpu_union_reduced") is True:
    return requested, "matched MR4 grouping-only A/B"
  return "unknown", f"unsupported or incomplete {requested or 'unknown'} proof"


def _bootstrap_lower(role_calls:list[dict[str, Any]], measurement_map:Mapping[str, dict[str, Any]],
                     role_outer_ids:list[str], outer_map:Mapping[str, dict[str, Any]], whole_samples:list[float],
                     *, resamples:int = 10_000, seed:int = 20260730) -> float:
  rng, values = random.Random(seed), []
  execution_counts: dict[str, int] = {}
  for call in role_calls:
    key = _digest(call["execution_identity"]); execution_counts[key] = execution_counts.get(key, 0) + 1
  paired_count = min([len(whole_samples), *(len(outer_map[outer]["samples_ns"]) for outer in role_outer_ids)])
  for _ in range(resamples):
    sample_index = rng.randrange(paired_count)
    wall = whole_samples[sample_index]
    isolated = sum(count * rng.choice(measurement_map[key]["samples_ns"]) for key,count in execution_counts.items())
    enclosing = sum(outer_map[outer]["samples_ns"][sample_index] for outer in role_outer_ids)
    values.append(min(wall, isolated, enclosing) / wall)
  values.sort()
  return values[int(0.05 * resamples)]


def build_role_cost_ranking(census:Mapping[str, Any], measurements:Sequence[Mapping[str, Any]], whole_step:Mapping[str, Any], *,
                            prerequisites:Mapping[str, Any], enclosures:Mapping[str, Any],
                            classification_evidence:Sequence[Mapping[str, Any]] = (), max_roles:int = 2) -> dict[str, Any]:
  if max_roles != 2: raise ValueError("MR7 ranking is bounded to exactly a two-role selection budget")
  if not isinstance(prerequisites, Mapping) or any(not isinstance(prerequisites.get(key), str) or
      _SHA256.fullmatch(prerequisites[key]) is None for key in PREREQUISITES):
    return _blocked("BLOCKED_PREREQUISITE", "missing_mr0_mr4_mr5_mr6_hash", prerequisites)
  if prerequisites["mr5"] != _digest(census):
    return _blocked("BLOCKED_PREREQUISITE", "mr5_census_hash_mismatch", prerequisites)
  if not isinstance(whole_step, Mapping) or whole_step.get("schema") != WHOLE_STEP_SCHEMA:
    return _blocked("INCONCLUSIVE_MEASUREMENT", "unsupported_whole_step_schema", prerequisites)
  if whole_step.get("source_schema") == MR4_SUMMARY_SCHEMA and whole_step.get("source_sha256") != prerequisites["mr4"]:
    return _blocked("BLOCKED_PREREQUISITE", "mr4_whole_step_hash_mismatch", prerequisites)
  whole = _samples(whole_step)
  if whole is None: return _blocked("INCONCLUSIVE_MEASUREMENT", "whole_step_requires_five_raw_samples", prerequisites)
  try: material = reconcile_material_calls(census)
  except ValueError as exc: return _blocked("BLOCKED_IDENTITY", str(exc), prerequisites)
  measured, error = _measurement_map(measurements)
  if error: return _blocked("BLOCKED_PROVIDER" if error.startswith("isolated") else "BLOCKED_IDENTITY", error, prerequisites)
  assert measured is not None
  needed = {_digest(row["execution_identity"]) for row in material}
  if set(measured) != needed: return _blocked("BLOCKED_IDENTITY", "incomplete_or_extra_exact_execution_identity_join", prerequisites)
  outer, error = _outer_map(enclosures, material)
  if error: return _blocked("INCONCLUSIVE_BOUND", error, prerequisites)
  assert outer is not None
  wall_samples, wall_median = whole
  class_by_role = {row.get("role"):row for row in classification_evidence if isinstance(row, Mapping) and isinstance(row.get("role"), str)}
  roles = []
  for role in sorted({role for row in material for role in row["roles"]}):
    calls = [row for row in material if role in row["roles"]]
    outer_ids = sorted({row["outer_id"] for row in calls})
    isolated_ns = sum(measured[_digest(row["execution_identity"])]["median_ns"] for row in calls)
    enclosing_uncapped_ns = sum(outer[key]["median_ns"] for key in outer_ids)
    enclosing_ns = min(wall_median, enclosing_uncapped_ns)
    headroom = min(wall_median, isolated_ns, enclosing_ns) / wall_median
    lower = _bootstrap_lower(calls, measured, outer_ids, outer, wall_samples)
    classification, reason = _honest_classification(role, class_by_role.get(role), prerequisites["mr4"])
    execution_keys = sorted({_digest(row["execution_identity"]) for row in calls})
    dispersion = max(measured[key]["relative_mad"] for key in execution_keys)
    eligible = headroom >= 0.05 and lower >= 0.05
    semantic_identities = sorted({_canonical(identity):identity for call in calls
      for identity in call["execution_identity"]["semantic_identities"] if identity["role"] == role}.values(), key=_canonical)
    isolated_evidence = [{"execution_identity_digest":key, "execution_identity":measured[key]["identity"],
      "median_ns":measured[key]["median_ns"],
      "samples_ns":measured[key]["samples_ns"], "relative_mad":measured[key]["relative_mad"],
      "evidence_hashes":measured[key]["evidence_hashes"]} for key in execution_keys]
    enclosure_evidence = [dict(outer[key]) for key in outer_ids]
    roles.append({"role":role, "semantic_digest":_digest([row["execution_identity"] for row in calls]),
      "call_indices":[row["call_index"] for row in calls], "occurrence_count":len(calls), "execution_identity_digests":execution_keys,
      "semantic_identities":semantic_identities,
      "placements":[{"call_index":row["call_index"], **row["placement"]} for row in calls],
      "graph_direct_enclosures":outer_ids, "isolated_measurements":isolated_evidence,
      "outer_enclosure_evidence":enclosure_evidence, "isolated_upper_bound_ns":isolated_ns,
      "enclosing_upper_bound_ns":enclosing_ns, "enclosing_uncapped_sum_ns":enclosing_uncapped_ns,
      "whole_step_median_ns":wall_median, "headroom_fraction":headroom, "headroom_lower_95":lower,
      "maximum_modeled_saved_ns":headroom*wall_median,
      "maximum_modeled_throughput_gain":None if headroom >= 1 else 1/(1-headroom)-1,
      "normalized_measurement_dispersion":dispersion, "bottleneck_class":classification,
      "bottleneck_reason":reason, "gate":"ELIGIBLE" if eligible else "EXCLUDED_NO_HEADROOM",
      "reopen_condition":("re-run after a new exact identity/outer-bound measurement raises both median and lower headroom to 5%"
                          if not eligible else "advance only through bounded MR8 population and later whole-model validation"),
      "note":"complete enclosing outer durations; never graph-duration/member splitting"})
  eligible = sorted((row for row in roles if row["gate"] == "ELIGIBLE"),
    key=lambda row:(-row["headroom_lower_95"], -row["headroom_fraction"], row["normalized_measurement_dispersion"], row["semantic_digest"]))
  for index,row in enumerate(eligible): row["gate"] = "SELECTED_FOR_MR8" if index < 2 else "DEFERRED_BY_BUDGET"
  return {"schema":SCHEMA, "status":"COMPLETE", "prerequisites":dict(prerequisites),
    "whole_step":{"samples_ns":wall_samples, "median_ns":wall_median}, "material_call_count":len(material),
    "measured_execution_identity_count":len(measured), "outer_enclosure_count":len(outer), "bootstrap_resamples":10_000,
    "roles":roles, "selected_roles":[row["role"] for row in eligible[:2]],
    "accounting":"I=sum exact isolated occurrence medians; G=sum complete unique enclosing outers capped by H; no graph-duration splitting"}


def main(argv:list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Build the fail-closed MR7 semantic role-cost ranking")
  for name in ("census", "measurements", "whole-step", "enclosures", "prerequisites"):
    parser.add_argument(f"--{name}", type=pathlib.Path)
  parser.add_argument("--authority", type=pathlib.Path,
                      help="canonical MR7 authority packet; mutually exclusive with individual inputs")
  parser.add_argument("--classification-evidence", type=pathlib.Path)
  parser.add_argument("--mr4-arm", choices=("control", "hybrid"), help="adapt --whole-step directly from this MR4 arm")
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args(argv)
  if args.out.exists(): raise FileExistsError(f"ranking output already exists: {args.out}")
  load = lambda path: json.loads(path.read_text())
  individual = [getattr(args, name.replace("-", "_")) for name in ("census","measurements","whole-step","enclosures","prerequisites")]
  if args.authority is not None:
    if any(value is not None for value in individual) or args.classification_evidence is not None or args.mr4_arm is not None:
      raise ValueError("--authority is mutually exclusive with individual MR7 ranker inputs")
    from boltbeam.search.semantic.mr7_evidence_bundle import ranking_from_authority
    authority = load(args.authority)
    result = ranking_from_authority(authority) | {"mr7_authority":authority}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"] == "COMPLETE" else 2
  if any(value is None for value in individual):
    raise ValueError("individual MR7 ranking requires census, measurements, whole-step, enclosures, and prerequisites")
  measurements = load(args.measurements)
  if not isinstance(measurements, Mapping) or measurements.get("schema") != MEASUREMENTS_SCHEMA:
    raise ValueError("measurements artifact must use the MR7 isolated-measurements schema")
  measurements = measurements.get("measurements")
  if not isinstance(measurements, list): raise ValueError("measurements artifact must contain measurements")
  classifications = load(args.classification_evidence) if args.classification_evidence else []
  if isinstance(classifications, Mapping): classifications = classifications.get("classifications", [])
  whole_step = load(getattr(args, "whole_step"))
  if whole_step.get("schema") == MR4_SUMMARY_SCHEMA:
    if args.mr4_arm is None: raise ValueError("--mr4-arm is required when --whole-step is an MR4 replay summary")
    whole_step = whole_step_from_mr4_summary(whole_step, args.mr4_arm)
  elif args.mr4_arm is not None: raise ValueError("--mr4-arm only applies to an MR4 replay summary")
  result = build_role_cost_ranking(load(args.census), measurements, whole_step,
    prerequisites=load(args.prerequisites), enclosures=load(args.enclosures), classification_evidence=classifications)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__": raise SystemExit(main())

__all__ = ["ENCLOSURES_SCHEMA", "MEASUREMENTS_SCHEMA", "MR4_SUMMARY_SCHEMA", "SCHEMA", "SEMANTIC_FIELDS",
           "WHOLE_STEP_SCHEMA", "build_role_cost_ranking", "main", "reconcile_material_calls",
           "whole_step_from_mr4_summary"]
