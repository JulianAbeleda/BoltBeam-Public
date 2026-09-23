"""Independent aggregate-wave scheduling model fitted from generated SQ calibration."""
from __future__ import annotations

import math
import json
import os
import pathlib
import statistics
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping

from boltbeam.artifacts.base import sha256_json
from boltbeam.perf.calibration import Interval
from boltbeam.perf.isa_graph import ISAGraph

SCHEMA = "boltbeam.mmq_scheduling_model.v2"


@dataclass(frozen=True)
class SchedulingModelV2:
  system_snapshot_id: str
  static_to_executed_valu: float
  static_to_executed_salu: float
  wave_coefficients: tuple[float, ...]  # intercept,waves,VALU,SALU,resident_batches
  wall_coefficients: tuple[float, float] # intercept_ms, ms/aggregate-wave-cycle
  relative_error: float
  source_id: str

  @property
  def model_id(self) -> str: return sha256_json(self.to_json(include_id=False))

  def to_json(self, include_id: bool = True) -> dict[str, Any]:
    out = {"schema": SCHEMA, "system_snapshot_id": self.system_snapshot_id,
      "static_to_executed": {"valu": self.static_to_executed_valu, "salu": self.static_to_executed_salu},
      "wave_coefficients": list(self.wave_coefficients), "wall_coefficients": list(self.wall_coefficients),
      "relative_error": self.relative_error, "source_id": self.source_id,
      "sq_wait_any_policy": "overlap_diagnostic_only_not_additive"}
    if include_id: out["model_id"] = self.model_id
    return out


def fit_scheduling_v2(artifact: Mapping[str, Any], cases: Mapping[str, Mapping[str, Any]], *, compute_units: int = 96) -> SchedulingModelV2:
  if (artifact.get("schema") != "tinygrad.mmq_scheduling_calibration.v1" or
      artifact.get("provenance_class") != "generated_microbenchmark" or
      artifact.get("candidate_timing_used_for_fit") is not False):
    raise ValueError("scheduling fit requires independent generated calibration")
  rows, valu_ratios, salu_ratios = [], [], []
  for case_id, relation in artifact.get("relationships", {}).get("cases", {}).items():
    case = cases.get(case_id)
    if not case: continue
    counters = relation["median_counters"]
    waves = float(counters["SQ_WAVES"])
    static_valu, static_salu = _static_domains(case)
    if static_valu: valu_ratios.append(counters["SQ_INSTS_VALU"] / (static_valu * waves))
    if static_salu: salu_ratios.append(counters["SQ_INSTS_SALU"] / (static_salu * waves))
    rows.append((waves, static_valu * waves, static_salu * waves,
                 math.ceil(waves / compute_units), float(counters["SQ_WAVE_CYCLES"]), float(relation["median_ms"])))
  if len(rows) < 10 or not valu_ratios or not salu_ratios: raise ValueError("insufficient scheduling calibration coverage")
  wave_x = [[1.0, w, valu, salu, batches] for w, valu, salu, batches, _, _ in rows]
  wave_y = [cycles for *_, cycles, _ in rows]
  wave_coeff = _least_squares(wave_x, wave_y, ridge=1e-6)
  wall_coeff = _least_squares([[1.0, cycles] for *_, cycles, _ in rows], [ms for *_, ms in rows], ridge=1e-9)
  errors = []
  for row, actual in zip(wave_x, wave_y): errors.append(abs(_dot(wave_coeff, row) - actual) / actual)
  source_id = sha256_json({k: v for k, v in artifact.items() if k != "samples"})
  return SchedulingModelV2(str(artifact["system_snapshot_id"]), statistics.median(valu_ratios),
    statistics.median(salu_ratios), tuple(wave_coeff), tuple(wall_coeff), max(errors), source_id)


def predict_scheduling_v2(graph: ISAGraph, *, workgroups: int, model: SchedulingModelV2,
                          candidate_id: str, binary_sha256: str) -> dict[str, Any]:
  counts = graph.counts()
  static_valu = sum(counts[k] for k in ("valu_int", "valu_float", "dot_mfma"))
  static_salu = sum(counts[k] for k in ("salu", "branch_predicate", "barrier", "waitcnt"))
  static_valu_total, static_salu_total = static_valu * workgroups, static_salu * workgroups
  executed_valu = static_valu_total * model.static_to_executed_valu
  executed_salu = static_salu_total * model.static_to_executed_salu
  batches = math.ceil(workgroups / 96)
  aggregate_cycles = max(1.0, _dot(model.wave_coefficients, [1, workgroups, static_valu_total, static_salu_total, batches]))
  median_ms = max(1e-9, _dot(model.wall_coefficients, [1, aggregate_cycles]))
  error = max(.05, model.relative_error)
  interval = Interval(median_ms * max(.01, 1 - error), median_ms, median_ms * (1 + error))
  body = {"schema": "boltbeam.mmq_prediction.v2", "model_id": model.model_id, "candidate_id": candidate_id,
    "binary_sha256": binary_sha256, "milliseconds": interval.to_json(), "aggregate_wave_cycles": aggregate_cycles,
    "estimated_executed": {"valu": executed_valu, "salu": executed_salu, "waves": workgroups,
                           "resident_batches": batches},
    "sq_wait_any_policy": "overlap_diagnostic_only_not_additive", "holdout_timing_used": False}
  return {**body, "prediction_id": sha256_json(body)}


def freeze_scheduling_v2(prediction: Mapping[str, Any], output: str | pathlib.Path) -> dict[str, Any]:
  if prediction.get("schema") != "boltbeam.mmq_prediction.v2" or prediction.get("holdout_timing_used") is not False:
    raise ValueError("only timing-free v2 predictions may freeze")
  body = {"schema": "boltbeam.mmq_prediction_freeze.v2", "prediction": dict(prediction),
          "frozen_before_holdout_read": True}
  artifact = {**body, "freeze_id": sha256_json(body)}
  path = pathlib.Path(output)
  if path.exists(): raise FileExistsError(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  try:
    with os.fdopen(fd, "w") as file: json.dump(artifact, file, indent=2, sort_keys=True); file.write("\n")
    os.replace(temporary, path)
  except Exception:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise
  return artifact


def _static_domains(case: Mapping[str, Any]) -> tuple[int, int]:
  instructions = case.get("isa", {}).get("instructions", [])
  valu = sum(row.get("instruction_class") in ("valu_int", "valu_float", "dot_mfma") for row in instructions)
  salu = sum(row.get("instruction_class") in ("salu", "branch_predicate", "barrier", "waitcnt") for row in instructions)
  return valu, salu


def _least_squares(x, y, ridge=0.0):
  n = len(x[0]); a = [[sum(row[i] * row[j] for row in x) + (ridge if i == j else 0) for j in range(n)] for i in range(n)]
  b = [sum(row[i] * value for row, value in zip(x, y)) for i in range(n)]
  for i in range(n):
    pivot = max(range(i, n), key=lambda r: abs(a[r][i])); a[i], a[pivot], b[i], b[pivot] = a[pivot], a[i], b[pivot], b[i]
    scale = a[i][i]
    if abs(scale) < 1e-12: continue
    a[i] = [v / scale for v in a[i]]; b[i] /= scale
    for r in range(n):
      if r == i: continue
      factor = a[r][i]; a[r] = [v - factor * q for v, q in zip(a[r], a[i])]; b[r] -= factor * b[i]
  return b


def _dot(a, b): return sum(x * y for x, y in zip(a, b))


__all__ = ["SchedulingModelV2", "fit_scheduling_v2", "freeze_scheduling_v2", "predict_scheduling_v2"]
