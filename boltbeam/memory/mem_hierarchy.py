"""Ingest hierarchy calibration sweeps without converting them to candidate residency evidence."""
from __future__ import annotations

import statistics
from typing import Any

from boltbeam.artifacts.base import EvidenceSource, sha256_json
from boltbeam.core.facts import Fact
from boltbeam.profiler.ontology import tier_observation, unsupported
from boltbeam.roofline.roofline_ceiling import resolve_achieved_ceiling
from boltbeam.vocab import SCHEMA_MEMORY_HIERARCHY_PROFILE

_PLATEAU_BAND = 0.08
_CANONICAL_CACHE_TIERS = ("l0", "l1", "l2", "last_level_cache")


def _sample_band(values: list[float]) -> dict[str, float]:
  ordered = sorted(float(v) for v in values)
  return {"low": ordered[0], "high": ordered[-1], "median": statistics.median(ordered)}


def _build_curve(points: list[dict], *, mode: str = "bandwidth", traffic: str = "read") -> list[tuple[int, float]]:
  curve = []
  for p in points:
    point_mode = p.get("mode", "bandwidth" if p.get("kind") == "copy" else p.get("kind"))
    if point_mode != mode or p.get("traffic", "read") != traffic: continue
    if mode == "pointer_chase":
      samples = p.get("warm_latency_ns") or p.get("latency_ns") or []
      if samples: curve.append((int(p["bytes"]), float(statistics.median(samples))))
    else:
      res = resolve_achieved_ceiling(cold_samples=p.get("cold_gbs"), sustained_samples=p.get("sustained_gbs"))
      curve.append((int(p["bytes"]), res.achieved_gbs))
  return sorted(curve)


def _detect_plateaus(curve: list[tuple[int, float]]) -> list[list[tuple[int, float]]]:
  plateaus, run = [], []
  for point in curve:
    trial = run + [point]
    med = statistics.median(g for _, g in trial)
    if not run or all(abs(g - med) <= _PLATEAU_BAND * abs(med) for _, g in trial):
      run = trial
    else:
      if len(run) >= 2: plateaus.append(run)
      run = [point]
  if len(run) >= 2: plateaus.append(run)
  return plateaus


def _tier_names(count: int, declared: list[str] | None) -> list[str]:
  if declared:
    names = ["last_level_cache" if t in {"mall", "llc"} else t for t in declared]
    if len(names) != count or len(set(names)) != len(names) or any(t not in (*_CANONICAL_CACHE_TIERS, "dram") for t in names):
      raise ValueError("separable_tiers must name exactly one supported tier per plateau")
    return names
  if count == 2: return ["l2_last_level_cache", "dram"]
  if count == 3: return ["l2", "last_level_cache", "dram"]
  if count in {4, 5}: return list((*_CANONICAL_CACHE_TIERS, "dram")[-count:])
  raise ValueError(f"cannot assign {count} hierarchy plateaus without separable_tiers")


def classify_ws_sweep(samples: dict, *, system_snapshot_id: str, tool: str = "tinygrad_ws_sweep") -> dict:
  """Normalize calibrated bands. Output is explicitly forbidden as candidate hit-rate evidence."""
  sample_snapshot = samples.get("system_snapshot_id")
  if sample_snapshot not in (None, system_snapshot_id):
    return {"schema": SCHEMA_MEMORY_HIERARCHY_PROFILE, "system_snapshot_id": system_snapshot_id,
            "facts": [], "tier_observations": [], "blockers": ["sweep-identity-mismatch: system_snapshot_id"]}
  curve = _build_curve(samples.get("points") or [])
  plateaus = _detect_plateaus(curve)
  if len(plateaus) < 2:
    return {"schema": SCHEMA_MEMORY_HIERARCHY_PROFILE, "system_snapshot_id": system_snapshot_id,
            "facts": [], "tier_observations": [], "blockers": [f"sweep-underspecified: {len(plateaus)} plateaus"]}
  ranked = sorted(plateaus, key=lambda run: statistics.median(g for _, g in run), reverse=True)
  try: names = _tier_names(len(ranked), samples.get("separable_tiers"))
  except ValueError as exc:
    return {"schema": SCHEMA_MEMORY_HIERARCHY_PROFILE, "system_snapshot_id": system_snapshot_id,
            "facts": [], "tier_observations": [], "blockers": [f"sweep-tier-ambiguity: {exc}"]}
  src = (EvidenceSource("boltbeam", tool, "", sha256_json(samples),
                        {"system_snapshot_id": system_snapshot_id, "scope": "hierarchy_calibration"}),)
  facts, observations, bandwidth_facts = [], [], {}
  for name, run in zip(names, ranked):
    values = [g for _, g in run]
    fact = Fact(f"mem.tier.{name}.bandwidth_gbs", statistics.median(values), "measured", unit="GB/s",
                sources=src, uncertainty=_sample_band(values))
    facts.append(fact); bandwidth_facts[name] = fact
    observations.append(tier_observation(name if name != "l2_last_level_cache" else "unknown",
                                         scope="global", source="hierarchy_calibration_sweep", status="derived",
                                         uncertainty={"bandwidth_gbs": _sample_band(values)},
                                         candidate_hit_rate_eligible=False, combined_tiers=["l2", "last_level_cache"] if name == "l2_last_level_cache" else []))
  for (fast_name, fast), (slow_name, slow) in zip(zip(names, ranked), zip(names[1:], ranked[1:])):
    if slow_name == "dram": capacity_name = fast_name
    else: capacity_name = fast_name
    low, high = fast[-1][0], slow[0][0]
    facts.append(Fact(f"mem.tier.{capacity_name}.bytes", (low * high) ** 0.5, "derived", unit="bytes",
                      derivation="sweep transition-band midpoint", sources=src,
                      input_fact_ids=(bandwidth_facts[fast_name].fact_id, bandwidth_facts[slow_name].fact_id),
                      uncertainty={"low": low, "high": high}))
  # Preserve AMD's historical MALL aliases while canonical LLC remains authoritative.
  by_name = {f.name: f for f in facts}
  for suffix in ("bandwidth_gbs", "bytes"):
    canonical = by_name.get(f"mem.tier.last_level_cache.{suffix}")
    if canonical:
      facts.append(Fact(f"mem.tier.mall.{suffix}", canonical.value, "derived", unit=canonical.unit,
                        derivation="AMD MALL alias of provider-neutral last_level_cache", sources=src,
                        input_fact_ids=(canonical.fact_id,), uncertainty=canonical.uncertainty))
  lds_values = [float(v) for v in ((samples.get("lds_probe") or {}).get("sustained_gbs") or [])]
  if lds_values:
    facts.append(Fact("mem.tier.lds.bandwidth_gbs", statistics.median(lds_values), "measured", unit="GB/s",
                      sources=src, uncertainty=_sample_band(lds_values)))
  unsupported_tiers = samples.get("unsupported_tiers") or {}
  for tier in _CANONICAL_CACHE_TIERS:
    if tier not in names:
      reason = unsupported_tiers.get(tier, "not_separable_by_calibration_sweep")
      observations.append(tier_observation(tier, scope="global", source=tool, status="unsupported", reason=reason,
                                           candidate_hit_rate_eligible=False))
  counter_outcomes = [unsupported(reason, counter=f"{tier}.hit_rate", tier=tier)
                      for tier, reason in (samples.get("unsupported_hit_rate_counters") or {}).items()]
  if not counter_outcomes:
    counter_outcomes = [{"value": None, "status": "missing", "reason": "not_measured_by_hierarchy_sweep",
                         "counter": f"{tier}.hit_rate", "tier": tier} for tier in _CANONICAL_CACHE_TIERS]
  # Legacy unknown fact retained, but its derivation no longer assumes an AMD target.
  facts.append(Fact("mem.l2.hit_rate", None, "unknown", derivation="not_measured_by_hierarchy_sweep"))
  return {"schema": SCHEMA_MEMORY_HIERARCHY_PROFILE, "system_snapshot_id": system_snapshot_id,
          "evidence_scope": "hierarchy_calibration", "candidate_hit_rate_eligible": False,
          "facts": [f.to_json() for f in facts], "tier_observations": observations,
          "counter_outcomes": counter_outcomes, "blockers": []}
