"""Holdout validation and explicit model falsification."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from boltbeam.perf.cycle_model import CyclePrediction

SCHEMA = "boltbeam.mmq_model_validation.v1"


@dataclass(frozen=True)
class ModelValidation:
  status: str
  reasons: tuple[str, ...]
  metrics: Mapping[str, Any]

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA, "status": self.status, "reasons": list(self.reasons), "metrics": dict(self.metrics)}


def validate_prediction(prediction: CyclePrediction, samples_ms: Sequence[float], *, system_snapshot_id: str,
                        binary_sha256: str, max_absolute_error: float = .15) -> ModelValidation:
  if not prediction.complete: return ModelValidation("inconclusive", ("prediction incomplete",), {})
  if system_snapshot_id != prediction.system_snapshot_id or binary_sha256 != prediction.binary_sha256:
    return ModelValidation("inconclusive", ("measurement identity mismatch",), {})
  if len(samples_ms) < 10 or any(not isinstance(x, (int, float)) or x <= 0 for x in samples_ms):
    return ModelValidation("inconclusive", ("insufficient measured samples",), {})
  measured = statistics.median(samples_ms)
  predicted = prediction.milliseconds.median
  relative_error = abs(predicted - measured) / measured
  reasons = []
  if not prediction.milliseconds.low <= measured <= prediction.milliseconds.high: reasons.append("measured median outside 95% interval")
  if relative_error > max_absolute_error: reasons.append("absolute timing error exceeds threshold")
  metrics = {"measured_median_ms": measured, "predicted_median_ms": predicted, "relative_error": relative_error,
             "interval_ms": prediction.milliseconds.to_json()}
  return ModelValidation("falsified" if reasons else "validated", tuple(reasons), metrics)


def validate_pair(gated_prediction: CyclePrediction, direct_prediction: CyclePrediction,
                  gated_samples: Sequence[float], direct_samples: Sequence[float], *, max_ratio_error: float = .10) -> ModelValidation:
  if (gated_prediction.system_snapshot_id != direct_prediction.system_snapshot_id or
      gated_prediction.calibration_id != direct_prediction.calibration_id or
      gated_prediction.model_version != direct_prediction.model_version):
    return ModelValidation("inconclusive", ("pair prediction identity mismatch",), {})
  gv = validate_prediction(gated_prediction, gated_samples, system_snapshot_id=gated_prediction.system_snapshot_id,
                           binary_sha256=gated_prediction.binary_sha256)
  dv = validate_prediction(direct_prediction, direct_samples, system_snapshot_id=direct_prediction.system_snapshot_id,
                           binary_sha256=direct_prediction.binary_sha256)
  if "inconclusive" in (gv.status, dv.status): return ModelValidation("inconclusive", gv.reasons + dv.reasons, {})
  measured_ratio = statistics.median(gated_samples) / statistics.median(direct_samples)
  predicted_ratio = gated_prediction.milliseconds.median / direct_prediction.milliseconds.median
  reasons = list(gv.reasons + dv.reasons)
  if (predicted_ratio > 1) != (measured_ratio > 1): reasons.append("predicted candidate ordering is wrong")
  ratio_error = abs(predicted_ratio - measured_ratio) / measured_ratio
  if ratio_error > max_ratio_error: reasons.append("pair ratio error exceeds threshold")
  return ModelValidation("falsified" if reasons else "validated", tuple(reasons),
                         {"measured_ratio": measured_ratio, "predicted_ratio": predicted_ratio, "ratio_error": ratio_error})


__all__ = ["ModelValidation", "validate_pair", "validate_prediction"]
