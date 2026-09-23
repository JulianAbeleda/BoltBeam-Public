"""Deterministic bootstrap fitting of calibration measurements into Facts."""
from __future__ import annotations

import random
import statistics
from typing import Iterable

from boltbeam.artifacts.base import EvidenceSource
from boltbeam.core.facts import Fact
from boltbeam.perf.calibration import CalibrationProfile
from boltbeam.perf.microbench import CalibrationMeasurement


def fit_calibration(measurements: Iterable[CalibrationMeasurement], *, system_snapshot_id: str,
                    protocol_id: str, static_facts: tuple[Fact, ...] = (), bootstrap_rounds: int = 1000,
                    seed: int = 0) -> CalibrationProfile:
  rows = tuple(measurements)
  facts = list(static_facts)
  names = {fact.name for fact in facts}
  for measurement in rows:
    if measurement.spec.system_snapshot_id != system_snapshot_id: raise ValueError("calibration system identity mismatch")
    if measurement.status != "measured": continue
    if measurement.spec.controls.get("training_scope") != "generated_microbenchmark":
      raise ValueError("calibration fit accepts generated microbenchmarks only, never final candidates")
    name = measurement.spec.parameter_name
    if name in names: raise ValueError(f"duplicate fitted parameter {name}")
    low, median, high = bootstrap_median_interval(measurement.samples, rounds=bootstrap_rounds,
                                                  seed=seed ^ int(measurement.measurement_id[-8:], 16))
    source = EvidenceSource("boltbeam", measurement.producer, "", measurement.measurement_id)
    facts.append(Fact(name, median, "measured", unit=measurement.spec.unit, sources=(source,),
                      uncertainty={"method": "bootstrap_median", "confidence": .95, "low": low, "high": high,
                                   "sample_count": len(measurement.samples)}))
    names.add(name)
  return CalibrationProfile(system_snapshot_id, tuple(facts), protocol_id)


def bootstrap_median_interval(samples: Iterable[float], *, rounds: int = 1000, seed: int = 0) -> tuple[float, float, float]:
  values = tuple(float(x) for x in samples)
  if len(values) < 30: raise ValueError("bootstrap fit requires at least 30 samples")
  if rounds < 100: raise ValueError("bootstrap fit requires at least 100 rounds")
  rng = random.Random(seed)
  medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(rounds))
  return medians[int(.025 * (rounds - 1))], statistics.median(values), medians[int(.975 * (rounds - 1))]


__all__ = ["bootstrap_median_interval", "fit_calibration"]
