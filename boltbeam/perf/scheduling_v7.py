"""Exact-final-ISA residual correction with explicit resource-transfer uncertainty."""
from __future__ import annotations
import json,os,pathlib,tempfile
from dataclasses import dataclass
from typing import Any,Mapping
from boltbeam.artifacts.base import sha256_json


@dataclass(frozen=True)
class SchedulingModelV7:
  per_false_site_ms:float
  max_fit_residual_ms:float
  source_id:str
  probe_resource_scope:Mapping[str,Any]
  @property
  def model_id(self):return sha256_json(self.to_json(False))
  def to_json(self,include_id=True):
    out={"schema":"boltbeam.mmq_scheduling_model.v7","target_metric":"kernel_device_ms","per_false_site_ms":self.per_false_site_ms,
      "max_fit_residual_ms":self.max_fit_residual_ms,"resource_transfer_uncertainty_ms":self.max_fit_residual_ms,
      "probe_resource_scope":dict(self.probe_resource_scope),"source_id":self.source_id,"candidate_timing_used_for_fit":False}
    if include_id:out["model_id"]=self.model_id
    return out


def fit_scheduling_v7(manifest:Mapping[str,Any],points:Mapping[int,Mapping[str,Any]])->SchedulingModelV7:
  if manifest.get("schema")!="tinygrad.mmq_exact_isa_residual.v6" or manifest.get("admitted_points")!=4 or set(points)!={0,64,128,256}:raise ValueError("expected complete exact-ISA series")
  for n,p in points.items():
    if p.get("provenance_class")!="generated_microbenchmark" or p.get("admitted") is not True or p.get("actual")!=p.get("expected") or not p.get("protocol",{}).get("gpu_timestamp_only") or p.get("production_dispatch_changed") is not False:raise ValueError("point failed exact-ISA admission")
    if n and p.get("store_mnemonics")!=["global_store_b32"]:raise ValueError("scalar b32 stores required")
  slope=float(manifest["fit"]["per_false_site_ms"]);intercept=float(manifest["fit"]["intercept_ms"])
  residual=max(abs(intercept+slope*n-float(points[n]["median_ms"])) for n in points)
  resources={"vgpr":[min(p["resources"]["vgpr"] for p in points.values()),max(p["resources"]["vgpr"] for p in points.values())],
    "sgpr":[min(p["resources"]["sgpr"] for p in points.values()),max(p["resources"]["sgpr"] for p in points.values())],
    "max_workgroup_threads":sorted({p["resources"]["max_workgroup_threads"] for p in points.values()})}
  return SchedulingModelV7(slope,residual,sha256_json(manifest),resources)


def predict_scheduling_v7(v5_prediction:Mapping[str,Any],model:SchedulingModelV7)->dict[str,Any]:
  n=int(v5_prediction["false_sites"]);correction=n*model.per_false_site_ms;b=v5_prediction["milliseconds"]
  uncertainty=model.max_fit_residual_ms if n else 0.0
  body={"schema":"boltbeam.mmq_prediction.v7","model_id":model.model_id,"candidate_id":v5_prediction["candidate_id"],"binary_sha256":v5_prediction["binary_sha256"],
    "target_metric":"kernel_device_ms","milliseconds":{"low":max(1e-9,b["low"]+correction-uncertainty),"median":b["median"]+correction,"high":b["high"]+correction+uncertainty},
    "base_prediction_id":v5_prediction["prediction_id"],"exact_isa_correction_ms":correction,"resource_transfer_uncertainty_ms":uncertainty,
    "candidate_timing_used":False,"holdout_timing_used":False}
  return {**body,"prediction_id":sha256_json(body)}


def freeze_scheduling_v7(prediction:Mapping[str,Any],output:str|pathlib.Path):
  if prediction.get("schema")!="boltbeam.mmq_prediction.v7" or prediction.get("candidate_timing_used") is not False or prediction.get("holdout_timing_used") is not False:raise ValueError("prediction is not holdout-clean")
  body={"schema":"boltbeam.mmq_prediction_freeze.v7","prediction":dict(prediction),"frozen_before_device_holdout_read":True};out={**body,"freeze_id":sha256_json(body)};path=pathlib.Path(output)
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


__all__=["SchedulingModelV7","fit_scheduling_v7","freeze_scheduling_v7","predict_scheduling_v7"]
