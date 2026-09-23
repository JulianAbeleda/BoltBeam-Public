"""Composable MMQ invocation phases; device execution remains a separate v7 objective."""
from __future__ import annotations
import json,os,pathlib,tempfile
from dataclasses import dataclass
from typing import Any,Mapping
from boltbeam.artifacts.base import sha256_json

PHASES=("source_construction","schedule_creation","compile_cache_lookup","output_readback","kernel_device")

@dataclass(frozen=True)
class InvocationModel:
  coefficients:Mapping[str,tuple[float,float]]
  max_fit_residual_ms:Mapping[str,float]
  source_ids:tuple[str,str]
  @property
  def model_id(self):return sha256_json(self.to_json(False))
  def to_json(self,include_id=True):
    out={"schema":"boltbeam.mmq_invocation_model.v1","phases":list(PHASES),"coefficients_ms":{k:list(v) for k,v in self.coefficients.items()},
      "max_fit_residual_ms":dict(self.max_fit_residual_ms),"source_ids":list(self.source_ids),"candidate_timing_used_for_fit":False,
      "uop_construction_relationship":"excluded_overlaps_source_construction"}
    if include_id:out["model_id"]=self.model_id
    return out


def fit_invocation_model(source:Mapping[str,Any],host:Mapping[str,Any])->InvocationModel:
  if source.get("schema")!="tinygrad.mmq_source_construction_factorial.v1" or source.get("provenance_class")!="generated_microbenchmark" or source.get("candidate_id") is not None:raise ValueError("independent source factorial required")
  if host.get("schema")!="tinygrad.mmq_host_invocation_calibration.v1" or host.get("provenance_class")!="generated_host_microbenchmark" or host.get("production_dispatch_changed") is not False:raise ValueError("independent host calibration required")
  sf=source["fit"];coeff={"source_construction":(float(sf["intercept_ms"]),float(sf["per_site_ms"]))}
  for phase in ("schedule_creation","compile_cache_lookup","output_readback"):
    f=host["fits"][phase];coeff[phase]=(float(f["intercept_ns"])/1e6,float(f["per_false_site_ns"])/1e6)
  residual={}
  residual["source_construction"]=max(abs(coeff["source_construction"][0]+coeff["source_construction"][1]*r["sites"]-r["median_ms"]) for r in source["points"])
  for phase in ("schedule_creation","compile_cache_lookup","output_readback"):
    residual[phase]=max(abs(coeff[phase][0]+coeff[phase][1]*r["false_sites"]-r["phases"][phase]["corrected_median_ns"]/1e6) for r in host["rows"])
  return InvocationModel(coeff,residual,(sha256_json(source),sha256_json(host)))


def predict_invocation(model:InvocationModel,device_prediction:Mapping[str,Any],*,false_sites:int)->dict[str,Any]:
  phases={}
  for name,(a,b) in model.coefficients.items():
    median=a+b*false_sites;e=model.max_fit_residual_ms[name];phases[name]={"low":max(0,median-e),"median":median,"high":median+e}
  phases["kernel_device"]=dict(device_prediction["milliseconds"])
  total={k:sum(x[k] for x in phases.values()) for k in ("low","median","high")}
  body={"schema":"boltbeam.mmq_invocation_prediction.v1","model_id":model.model_id,"candidate_id":device_prediction["candidate_id"],
    "binary_sha256":device_prediction["binary_sha256"],"false_sites":false_sites,"phases_ms":phases,"total_ms":total,
    "device_prediction_id":device_prediction["prediction_id"],"candidate_timing_used":False,"holdout_timing_used":False}
  return {**body,"prediction_id":sha256_json(body)}


def freeze_invocation(prediction:Mapping[str,Any],output:str|pathlib.Path):
  if prediction.get("schema")!="boltbeam.mmq_invocation_prediction.v1" or prediction.get("candidate_timing_used") is not False or prediction.get("holdout_timing_used") is not False:raise ValueError("prediction is not holdout-clean")
  body={"schema":"boltbeam.mmq_invocation_freeze.v1","prediction":dict(prediction),"frozen_before_invocation_holdout_read":True};out={**body,"freeze_id":sha256_json(body)};path=pathlib.Path(output)
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


__all__=["InvocationModel","PHASES","fit_invocation_model","freeze_invocation","predict_invocation"]
