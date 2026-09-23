"""Device-scoped SQ-to-GPU-time conversion fitted only from generated calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from boltbeam.artifacts.base import sha256_json
from boltbeam.perf.calibration import Interval
from boltbeam.perf.scheduling_v3 import fit_scheduling_v3
from boltbeam.perf.timing_scope import require_kernel_target


@dataclass(frozen=True)
class SchedulingModelV4:
  intercept_ms: float
  ms_per_wave_cycle: float
  max_training_per_wave_cycles: float
  relative_error: float
  source_id: str

  @property
  def model_id(self): return sha256_json(self.to_json(False))
  def to_json(self, include_id=True):
    out = {"schema":"boltbeam.mmq_scheduling_model.v4", "target_metric":"kernel_device_ms",
      "intercept_ms":self.intercept_ms, "ms_per_wave_cycle":self.ms_per_wave_cycle,
      "max_training_per_wave_cycles":self.max_training_per_wave_cycles,
      "relative_error":self.relative_error, "source_id":self.source_id,
      "resident_batches_are_not_additive_time":True}
    if include_id: out["model_id"] = self.model_id
    return out


def fit_scheduling_v4(artifact: Mapping[str,Any]) -> SchedulingModelV4:
  require_kernel_target("kernel_device_ms")
  v3 = fit_scheduling_v3(artifact)
  return SchedulingModelV4(v3.intercept_ms, v3.ms_per_wave_cycle, v3.max_training_per_wave_cycles,
                           v3.relative_error, v3.source_id)


def predict_scheduling_v4(v2_prediction: Mapping[str,Any], model: SchedulingModelV4) -> dict[str,Any]:
  waves=float(v2_prediction["estimated_executed"]["waves"])
  per_wave=float(v2_prediction["aggregate_wave_cycles"])/waves
  median=model.intercept_ms+max(0.0, model.ms_per_wave_cycle*per_wave)
  envelope=per_wave/model.max_training_per_wave_cycles
  error=max(.05,model.relative_error)*max(1.0,envelope)
  if envelope>1: error=max(error,.75*(envelope-1)+.25)
  interval=Interval(max(1e-9,median*(1-error)),median,median*(1+error))
  body={"schema":"boltbeam.mmq_prediction.v4", "model_id":model.model_id,
    "candidate_id":v2_prediction["candidate_id"], "binary_sha256":v2_prediction["binary_sha256"],
    "target_metric":"kernel_device_ms", "milliseconds":interval.to_json(), "per_wave_cycles":per_wave,
    "training_envelope_ratio":envelope, "extrapolated":envelope>1,
    "resident_batches_applied_to_time":False, "candidate_timing_used":False,
    "constructed_after_metric_scope_correction":True}
  return {**body,"prediction_id":sha256_json(body)}


__all__=["SchedulingModelV4","fit_scheduling_v4","predict_scheduling_v4"]
