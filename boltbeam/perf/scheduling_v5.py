"""Independent false-site/LDS interaction correction for device-scoped MMQ time."""
from __future__ import annotations

import json, os, pathlib, tempfile
from dataclasses import dataclass
from typing import Any, Mapping

from boltbeam.artifacts.base import sha256_json
from boltbeam.perf.calibration import Interval


@dataclass(frozen=True)
class SchedulingModelV5:
  false_site_ms: float
  lds_offset_ms: float
  false_site_lds_ms: float
  source_id: str

  @property
  def model_id(self): return sha256_json(self.to_json(False))
  def to_json(self, include_id=True):
    out={"schema":"boltbeam.mmq_scheduling_model.v5","target_metric":"kernel_device_ms",
      "false_site_ms":self.false_site_ms,"lds_offset_ms":self.lds_offset_ms,
      "false_site_lds_ms":self.false_site_lds_ms,"source_id":self.source_id,
      "candidate_timing_used_for_fit":False}
    if include_id: out["model_id"]=self.model_id
    return out


def fit_scheduling_v5(manifest: Mapping[str,Any], cases: Mapping[str,Mapping[str,Any]]) -> SchedulingModelV5:
  if manifest.get("schema")!="tinygrad.mmq_residual_probe.v4" or manifest.get("cases")!=16: raise ValueError("expected complete residual probe v4")
  rows=manifest.get("rows",[])
  if len(rows)!=16 or set(cases)!={r["case_id"] for r in rows}: raise ValueError("all residual cases are required")
  if any(c.get("provenance_class")!="generated_microbenchmark" or not c.get("protocol",{}).get("gpu_timestamp_only") or c.get("production_dispatch_changed") is not False for c in cases.values()):
    raise ValueError("v5 accepts independent GPU-timestamp microbenchmarks only")
  terms=manifest["fit"]["terms"]; coeff=manifest["fit"]["coefficients_ms"]
  by=dict(zip(terms,map(float,coeff)))
  false_site=by["static_store_sites"]+2*by["branch_sites"]
  if not 0.000015 <= false_site <= 0.000021: raise ValueError("combined false-site coefficient outside calibrated scope")
  return SchedulingModelV5(false_site,by["lds"],by["store_sites_x_lds"],sha256_json(manifest))


def predict_scheduling_v5(base_prediction: Mapping[str,Any], model: SchedulingModelV5, *, candidate_id: str,
                          binary_sha256: str, false_sites: int, lds_stage: bool) -> dict[str,Any]:
  base=float(base_prediction["milliseconds"]["median"])
  correction=false_sites*model.false_site_ms
  if lds_stage: correction += model.lds_offset_ms+false_sites*model.false_site_lds_ms
  median=base+correction; error=.15
  body={"schema":"boltbeam.mmq_prediction.v5","model_id":model.model_id,"candidate_id":candidate_id,
    "binary_sha256":binary_sha256,"target_metric":"kernel_device_ms","milliseconds":Interval(median*(1-error),median,median*(1+error)).to_json(),
    "base_prediction_id":base_prediction["prediction_id"],"false_sites":false_sites,"lds_stage":lds_stage,
    "interaction_correction_ms":correction,"candidate_timing_used":False,"holdout_timing_used":False}
  return {**body,"prediction_id":sha256_json(body)}


def freeze_scheduling_v5(prediction:Mapping[str,Any], output:str|pathlib.Path):
  if prediction.get("candidate_timing_used") is not False or prediction.get("holdout_timing_used") is not False: raise ValueError("prediction is not holdout-clean")
  body={"schema":"boltbeam.mmq_prediction_freeze.v5","prediction":dict(prediction),"frozen_before_device_holdout_read":True}
  artifact={**body,"freeze_id":sha256_json(body)}; path=pathlib.Path(output)
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


__all__=["SchedulingModelV5","fit_scheduling_v5","freeze_scheduling_v5","predict_scheduling_v5"]
