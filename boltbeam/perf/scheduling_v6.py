"""Bounded grid-transfer correction for generated false branch/store sites."""
from __future__ import annotations

import json,os,pathlib,tempfile
from dataclasses import dataclass
from typing import Any, Mapping
from boltbeam.artifacts.base import sha256_json


@dataclass(frozen=True)
class SchedulingModelV6:
  coefficients_ms: tuple[float,...]
  intervals_ms: tuple[tuple[float,float],...]
  source_id: str
  @property
  def model_id(self): return sha256_json(self.to_json(False))
  def to_json(self,include_id=True):
    out={"schema":"boltbeam.mmq_scheduling_model.v6","target_metric":"kernel_device_ms",
      "terms":["intercept","workgroups","false_sites","lds","workgroups_x_false_sites","false_sites_x_lds","workgroups_x_false_sites_x_lds"],
      "coefficients_ms":list(self.coefficients_ms),"interval95_ms":[{"low":a,"high":b} for a,b in self.intervals_ms],
      "source_id":self.source_id,"candidate_timing_used_for_fit":False}
    if include_id:out["model_id"]=self.model_id
    return out


def fit_scheduling_v6(manifest:Mapping[str,Any],cases:Mapping[str,Mapping[str,Any]])->SchedulingModelV6:
  if manifest.get("schema")!="tinygrad.mmq_residual_grid_factorial.v5" or manifest.get("cells")!=24 or manifest.get("provenance_class")!="generated_microbenchmark" or manifest.get("production_dispatch_changed") is not False:
    raise ValueError("expected independent residual grid factorial")
  if set(cases)!=set(manifest.get("case_ids",[])) or any(not c.get("protocol",{}).get("gpu_timestamp_only") or c.get("provenance_class")!="generated_microbenchmark" for c in cases.values()):
    raise ValueError("all 24 GPU-timestamp cases are required")
  fit=manifest["fit"]
  expected=["intercept","workgroups","false_sites","lds","workgroups_x_false_sites","false_sites_x_lds","workgroups_x_false_sites_x_lds"]
  if fit["terms"]!=expected or fit.get("bootstrap_replicates",0)<1000:raise ValueError("invalid bounded grid fit")
  return SchedulingModelV6(tuple(map(float,fit["coefficients_ms"])),tuple((float(x["low"]),float(x["high"])) for x in fit["interval95_ms"]),sha256_json(manifest))


def predict_scheduling_v6(base:Mapping[str,Any],model:SchedulingModelV6,*,candidate_id:str,binary_sha256:str,false_sites:int,workgroups:int=256,lds_stage=True)->dict[str,Any]:
  c=model.coefficients_ms;i=model.intervals_ms;n=false_sites;w=workgroups
  point=n*(c[2]+w*c[4]+(c[5] if lds_stage else 0)+(w*c[6] if lds_stage else 0))
  low=n*(i[2][0]+w*i[4][0]+(i[5][0] if lds_stage else 0)+(w*i[6][0] if lds_stage else 0))
  high=n*(i[2][1]+w*i[4][1]+(i[5][1] if lds_stage else 0)+(w*i[6][1] if lds_stage else 0))
  b=base["milliseconds"]
  body={"schema":"boltbeam.mmq_prediction.v6","model_id":model.model_id,"candidate_id":candidate_id,"binary_sha256":binary_sha256,
    "target_metric":"kernel_device_ms","milliseconds":{"low":b["low"]+low,"median":b["median"]+point,"high":b["high"]+high},
    "grid_correction_ms":{"low":low,"median":point,"high":high},"false_sites":n,"workgroups":w,"lds_stage":lds_stage,
    "candidate_timing_used":False,"holdout_timing_used":False}
  return {**body,"prediction_id":sha256_json(body)}


def freeze_scheduling_v6(prediction:Mapping[str,Any],output:str|pathlib.Path):
  if prediction.get("schema")!="boltbeam.mmq_prediction.v6" or prediction.get("candidate_timing_used") is not False or prediction.get("holdout_timing_used") is not False:raise ValueError("prediction is not holdout-clean")
  body={"schema":"boltbeam.mmq_prediction_freeze.v6","prediction":dict(prediction),"frozen_before_device_holdout_read":True};out={**body,"freeze_id":sha256_json(body)};path=pathlib.Path(output)
  if path.exists():raise FileExistsError(path)
  fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
  try:
    with os.fdopen(fd,"w") as f:json.dump(out,f,indent=2,sort_keys=True);f.write("\n")
    os.replace(tmp,path)
  except Exception:
    try:os.unlink(tmp)
    except FileNotFoundError:pass
    raise
  return out


__all__=["SchedulingModelV6","fit_scheduling_v6","freeze_scheduling_v6","predict_scheduling_v6"]
