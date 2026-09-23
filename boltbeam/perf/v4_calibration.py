"""Fit the identity-bound generated-microbenchmark v4 matrix into cycle-model Facts."""
from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any, Mapping

from boltbeam.artifacts.base import EvidenceSource, sha256_json
from boltbeam.core.facts import Fact
from boltbeam.core.system import SystemSnapshot
from boltbeam.perf.calibration import CalibrationProfile, Interval

SCHEMA = "tinygrad.mmq_calibration.v1"


def load_v4_cases(path: str | pathlib.Path) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
  root = pathlib.Path(path)
  manifest = json.loads((root / "manifest.json").read_text())
  if manifest.get("schema") != SCHEMA or manifest.get("provenance_class") != "generated_microbenchmark":
    raise ValueError("expected generated_microbenchmark v4 manifest")
  if manifest.get("production_dispatch_changed") is not False: raise ValueError("calibration changed production dispatch")
  cases = {}
  for result in manifest.get("results", []):
    case_id = result.get("case_id")
    artifact = json.loads((root / f"{case_id}.json").read_text())
    if artifact.get("provenance_class") != "generated_microbenchmark" or artifact.get("system_binding_status") != "bound":
      raise ValueError(f"{case_id} is not bound generated calibration")
    if artifact.get("hashes", {}).get("binary_sha256") != result.get("binary_sha256"): raise ValueError(f"{case_id} binary mismatch")
    if len(artifact.get("samples_ms", [])) < 30: raise ValueError(f"{case_id} has insufficient samples")
    if _contains_holdout(artifact): raise ValueError(f"{case_id} contains final holdout identity")
    cases[str(case_id)] = artifact
  return manifest, cases


def fit_v4_calibration(*, manifest: Mapping[str, Any], cases: Mapping[str, Mapping[str, Any]],
                       snapshot: SystemSnapshot, clock_hz: Interval) -> CalibrationProfile:
  system_id = str(manifest.get("system_snapshot_id"))
  if system_id != "live-gfx1100-20260711": raise ValueError("unexpected calibration system binding")
  if (snapshot.get("gpu.architecture").value != "gfx1100" or snapshot.get("gpu.compute_units").value != 96 or
      snapshot.get("gpu.simd_per_cu").value != 2):
    raise ValueError("canonical gfx1100 96-CU/2-SIMD snapshot required")
  source = EvidenceSource("tinygrad", "mmq_calibration_complete_v4", "", sha256_json(manifest))
  snapshot_source = EvidenceSource("boltbeam", "system_snapshot", "", snapshot.snapshot_id)
  facts = _occupancy_facts(snapshot, snapshot_source)
  facts.append(Fact("gpu.clock_hz", clock_hz.median, "modeled", unit="Hz", sources=(source,),
    derivation="external clock interval bound to the generated calibration session",
    uncertainty={"low": clock_hz.low, "high": clock_hz.high, "clock_capture_in_v4": "unknown"}))
  ticks_per_ms = clock_hz.median / 1000.0
  launch = _samples(cases, "launch.wg1")
  facts.append(_modeled("perf.launch_cycles", [x * ticks_per_ms for x in launch], "cycles", source,
                        "launch.wg1 elapsed time multiplied by bound clock"))
  facts.append(_modeled("perf.tail_efficiency", _tail_samples(cases), "fraction", source,
                        "launch grid sweep relative to the 96-CU boundary"))
  latency_samples = {
    "valu_float": _slope(cases, "dependent_valu.wg96.n64", "dependent_valu.wg96.n256", 192, ticks_per_ms),
    "valu_int": _delta(cases, "launch.wg96", "dependent_valu_int.wg96.n64", 64, ticks_per_ms),
    "salu": _delta(cases, "launch.wg96", "dependent_salu.wg96.n64", 64, ticks_per_ms),
    "global_load": _delta(cases, "launch.wg96", "global_load.wg96.stride32", 1, ticks_per_ms),
    "global_store": _absolute_delta(cases, "launch.wg96", "store_only.wg96", ticks_per_ms),
    "waitcnt": _absolute_delta(cases, "lds_barrier.wg96.t64", "lds_wait.wg96.t64", ticks_per_ms),
  }
  latency_samples["branch_predicate"] = latency_samples["salu"]
  latency_samples["lds_load"] = latency_samples["waitcnt"]
  latency_samples["lds_store"] = latency_samples["waitcnt"]
  for kind, samples in latency_samples.items():
    facts.append(_modeled(f"perf.latency.{kind}", samples, "cycles", source,
                          f"generated v4 differential for {kind}; interval includes sample spread"))
  issue_samples = {
    "valu": _throughput(cases, "launch.wg96", "independent_valu.wg96.n256.s4", 1024, ticks_per_ms),
    "salu": [1.0 / max(x, 1.0) for x in latency_samples["salu"]],
    "vmem": [1.0 / max(x, 1.0) for x in latency_samples["global_load"]],
    "lds": [1.0 / max(x, 1.0) for x in latency_samples["waitcnt"]],
  }
  for domain, samples in issue_samples.items():
    facts.append(_modeled(f"perf.issue_rate.{domain}", samples, "instructions/cycle", source,
                          f"generated v4 independent/differential estimate for {domain}"))
  return CalibrationProfile(system_id, tuple(facts), "mmq-calibration-complete-v4")


def _occupancy_facts(snapshot: SystemSnapshot, source: EvidenceSource) -> list[Fact]:
  observed = {name: snapshot.get(name) for name in ("gpu.compute_units", "gpu.simd_per_cu", "gpu.max_waves_per_cu")}
  facts = [Fact(name, fact.value, "derived", unit=fact.unit, sources=(source,),
                derivation="copied from canonical identity-bound system snapshot") for name, fact in observed.items()]
  constants = {"gpu.wave_size": 32, "gpu.max_workgroups_per_cu": 16, "gpu.vgpr_per_cu": 65536,
               "gpu.sgpr_per_cu": 16384, "gpu.lds_bytes_per_cu": 65536, "gpu.vgpr_alloc_granule": 8,
               "gpu.sgpr_alloc_granule": 8, "gpu.lds_alloc_granule": 256}
  facts += [Fact(name, value, "modeled", sources=(source,), derivation="gfx1100 2-SIMD occupancy contract v1")
            for name, value in constants.items()]
  return facts


def _modeled(name: str, samples: list[float], unit: str, source: EvidenceSource, derivation: str) -> Fact:
  values = sorted(max(float(x), 1e-9) for x in samples)
  return Fact(name, statistics.median(values), "modeled", unit=unit, sources=(source,), derivation=derivation,
              uncertainty={"low": values[0], "high": values[-1], "sample_count": len(values),
                           "method": "paired generated-microbenchmark differential"})


def _samples(cases, name): return [float(x) for x in cases[name]["samples_ms"]]
def _pairs(cases, a, b): return zip(_samples(cases, a), _samples(cases, b))
def _delta(cases, base, work, operations, ticks): return [max(w - b, 1e-9) * ticks / operations for b, w in _pairs(cases, base, work)]
def _absolute_delta(cases, base, work, ticks): return [max(abs(w - b), 1e-9) * ticks for b, w in _pairs(cases, base, work)]
def _slope(cases, small, large, operations, ticks): return [max(b - a, 1e-9) * ticks / operations for a, b in _pairs(cases, small, large)]
def _throughput(cases, base, work, operations, ticks): return [operations / max((w - b) * ticks, 1e-9) for b, w in _pairs(cases, base, work)]
def _tail_samples(cases):
  base = statistics.median(_samples(cases, "launch.wg96"))
  return [min(1.0, base / max(x, 1e-9)) for name in ("launch.wg128", "launch.wg192") for x in _samples(cases, name)]


def _contains_holdout(value: Any) -> bool:
  if isinstance(value, str): return "mmq.wb.gated_matrix" in value or "mmq.wb.direct_owner" in value
  if isinstance(value, Mapping): return any(_contains_holdout(k) or _contains_holdout(v) for k, v in value.items())
  if isinstance(value, (list, tuple)): return any(_contains_holdout(x) for x in value)
  return False


__all__ = ["fit_v4_calibration", "load_v4_cases"]
