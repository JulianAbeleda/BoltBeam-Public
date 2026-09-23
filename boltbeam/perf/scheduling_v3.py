"""Long-chain per-wave SQ-to-wall conversion with explicit extrapolation bounds."""
from __future__ import annotations

import json, os, pathlib, statistics, tempfile
from dataclasses import dataclass
from typing import Any, Mapping

from boltbeam.artifacts.base import sha256_json
from boltbeam.perf.calibration import Interval


@dataclass(frozen=True)
class SchedulingModelV3:
  intercept_ms: float
  ms_per_wave_cycle: float
  max_training_per_wave_cycles: float
  relative_error: float
  blocked_chain_lengths: tuple[int, ...]
  source_id: str

  @property
  def model_id(self) -> str: return sha256_json(self.to_json(False))
  def to_json(self, include_id=True):
    out={"schema":"boltbeam.mmq_scheduling_model.v3","intercept_ms":self.intercept_ms,
      "ms_per_wave_cycle":self.ms_per_wave_cycle,"max_training_per_wave_cycles":self.max_training_per_wave_cycles,
      "relative_error":self.relative_error,"blocked_chain_lengths":list(self.blocked_chain_lengths),
      "source_id":self.source_id,"sq_wait_any_policy":"overlap_diagnostic_only_not_additive"}
    if include_id: out["model_id"]=self.model_id
    return out


def fit_scheduling_v3(artifact: Mapping[str,Any]) -> SchedulingModelV3:
  if artifact.get("schema")!="tinygrad.mmq_long_chain_calibration.v1" or artifact.get("provenance_class")!="generated_microbenchmark" or artifact.get("relationships",{}).get("candidate_timing_used_for_fit") is not False:
    raise ValueError("v3 requires generated long-chain calibration")
  rows=sorted(artifact.get("joined_cases",[]),key=lambda r:r.get("chain_length",0))
  if [r.get("chain_length") for r in rows] != [128,256,512]: raise ValueError("v3 training envelope must be exactly 128/256/512")
  x=[float(r["per_wave_cycles"]) for r in rows]; y=[float(r["auto_median_ms"]) for r in rows]
  slope=sum((a-statistics.mean(x))*(b-statistics.mean(y)) for a,b in zip(x,y))/sum((a-statistics.mean(x))**2 for a in x)
  intercept=statistics.mean(y)-slope*statistics.mean(x)
  errors=[abs((intercept+slope*a)-b)/b for a,b in zip(x,y)]
  blocked=tuple(r["chain_length"] for r in artifact.get("requested_long_chains",[]) if str(r.get("auto_status","")).startswith("blocked"))
  return SchedulingModelV3(intercept,slope,max(x),max(errors),blocked,sha256_json({k:v for k,v in artifact.items() if k not in ("modes",)}))


def predict_scheduling_v3(v2_prediction: Mapping[str,Any],model:SchedulingModelV3,*,compute_units=96)->dict[str,Any]:
  waves=float(v2_prediction["estimated_executed"]["waves"]); per_wave=float(v2_prediction["aggregate_wave_cycles"])/waves
  batches=int(v2_prediction["estimated_executed"]["resident_batches"])
  work=max(0.0,model.ms_per_wave_cycle*per_wave)*batches
  median=model.intercept_ms+work
  extrapolation=per_wave/model.max_training_per_wave_cycles
  error=max(.05,model.relative_error)*(max(1.0,extrapolation))
  if extrapolation>1: error=max(error,.75*(extrapolation-1)+.25)
  interval=Interval(max(1e-9,median*(1-error)),median,median*(1+error))
  body={"schema":"boltbeam.mmq_prediction.v3","model_id":model.model_id,"candidate_id":v2_prediction["candidate_id"],
    "binary_sha256":v2_prediction["binary_sha256"],"milliseconds":interval.to_json(),"per_wave_cycles":per_wave,
    "resident_batches":batches,"training_envelope_ratio":extrapolation,"extrapolated":extrapolation>1,
    "blocked_larger_chains":list(model.blocked_chain_lengths),"sq_wait_any_policy":"overlap_diagnostic_only_not_additive",
    "holdout_timing_used":False}
  return {**body,"prediction_id":sha256_json(body)}


def freeze_scheduling_v3(prediction:Mapping[str,Any],output:str|pathlib.Path)->dict[str,Any]:
  if prediction.get("schema")!="boltbeam.mmq_prediction.v3" or prediction.get("holdout_timing_used") is not False: raise ValueError("invalid v3 freeze")
  body={"schema":"boltbeam.mmq_prediction_freeze.v3","prediction":dict(prediction),"frozen_before_holdout_read":True}; artifact={**body,"freeze_id":sha256_json(body)}
  path=pathlib.Path(output)
  if path.exists(): raise FileExistsError(path)
  fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
  try:
    with os.fdopen(fd,"w") as f: json.dump(artifact,f,indent=2,sort_keys=True);f.write("\n")
    os.replace(tmp,path)
  except Exception:
    try: os.unlink(tmp)
    except FileNotFoundError: pass
    raise
  return artifact

__all__=["SchedulingModelV3","fit_scheduling_v3","freeze_scheduling_v3","predict_scheduling_v3"]
