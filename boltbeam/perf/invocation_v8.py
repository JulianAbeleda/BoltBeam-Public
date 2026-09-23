"""Candidate-shaped operand fixed cost plus bounded writeback interaction.

Verdict: operational_total_validated_with_minor_phase_scope_gaps -- the surviving
invocation model after v2..v7 were falsified against measurement. Evidence:
bench/mmq-invocation-v8-validation-20260711.json and
bench/mmq-combined-operational-model-v1-20260711.json (combined model: validated).
Falsification record for the deleted v2..v7: bench/mmq-invocation-v{2..7}-validation-20260711.json
and docs/task_workflow/output/perf-model-verdict-record-20260731.md.
"""
from __future__ import annotations
import json,os,pathlib,tempfile
from dataclasses import dataclass
from typing import Any,Mapping
from boltbeam.artifacts.base import sha256_json

def _q(x,q):
  x=sorted(map(float,x));p=(len(x)-1)*q;a=int(p);b=min(a+1,len(x)-1);return (x[a]+(x[b]-x[a])*(p-a))/1e6

@dataclass(frozen=True)
class InvocationModelV8:
  fixed_operand_ms:Mapping[str,float]
  writeback_interaction_ms:Mapping[str,float]
  gated_schedule_cache:Mapping[str,Mapping[str,float]]
  source_ids:tuple[str,str]
  @property
  def model_id(self):return sha256_json(self.to_json(False))
  def to_json(self,include_id=True):
    out={"schema":"boltbeam.mmq_invocation_model.v8","objective":"host_invocation_ms","device_time_excluded":True,
      "shared_candidate_operand_fixed_ms":dict(self.fixed_operand_ms),"gated_writeback_interaction_ms":dict(self.writeback_interaction_ms),
      "gated_schedule_cache":dict(self.gated_schedule_cache),"uncertainty_policy":"conservative empirical difference bounds from independent cell percentiles",
      "source_ids":list(self.source_ids),"candidate_timing_used_for_fit":False}
    if include_id:out["model_id"]=self.model_id
    return out

def _difference(a,b):
  sa=a["channels"]["total"]["samples_ns"];sb=b["channels"]["total"]["samples_ns"]
  return {"low":_q(sa,.025)-_q(sb,.975),"median":_q(sa,.5)-_q(sb,.5),"high":_q(sa,.975)-_q(sb,.025)}

def fit_invocation_v8(operand:Mapping[str,Any],scaffold:Mapping[str,Any])->InvocationModelV8:
  if operand.get("schema")!="tinygrad.mmq_invocation_v7.writeback_builder.v1" or operand.get("candidate_ids")!=[] or operand.get("candidate_timings")!=[] or operand.get("production_dispatch_changed") is not False:raise ValueError("independent operand interaction required")
  rows={(r["style"],r["writeback_iterations"]):r for r in operand["rows"]}
  fixed=_difference(rows[("candidate_shaped",1)],rows[("simple",1)])
  wb=_difference(rows[("candidate_shaped",256)],rows[("candidate_shaped",1)])
  pooled={p:[x for r in scaffold["rows"] for x in r["phases"][p]["samples_ns"]] for p in ("schedule_creation","warmed_compile_cache_lookup")}
  gs={p:{"low":_q(s,.025),"median":_q(s,.5),"high":_q(s,.975)} for p,s in pooled.items()}
  return InvocationModelV8(fixed,wb,gs,(sha256_json(operand),sha256_json(scaffold)))

def predict_invocation_v8(model:InvocationModelV8,direct:Mapping[str,Any],*,candidate_id:str,binary_sha256:str,gated:bool)->dict[str,Any]:
  phases={k:dict(v) for k,v in direct["phases_ms"].items()};base=direct["phases_ms"]["uop_construction"]
  corr={k:model.fixed_operand_ms[k]+(model.writeback_interaction_ms[k] if gated else 0) for k in ("low","median","high")}
  phases["uop_construction"]={k:max(0,base[k]+corr[k]) for k in ("low","median","high")}
  if gated:phases.update({k:dict(v) for k,v in model.gated_schedule_cache.items()})
  total={k:sum(v[k] for v in phases.values()) for k in ("low","median","high")}
  body={"schema":"boltbeam.mmq_invocation_prediction.v8","model_id":model.model_id,"candidate_id":candidate_id,"binary_sha256":binary_sha256,"gated":gated,
    "phases_ms":phases,"total_host_ms":total,"uop_correction_ms":corr,"shared_fixed_applied_once":True,"device_time_excluded":True,"candidate_timing_used":False,"holdout_timing_used":False}
  return {**body,"prediction_id":sha256_json(body)}

def freeze_invocation_v8(prediction:Mapping[str,Any],output:str|pathlib.Path):
  if prediction.get("schema")!="boltbeam.mmq_invocation_prediction.v8" or prediction.get("candidate_timing_used") is not False or prediction.get("holdout_timing_used") is not False:raise ValueError("prediction is not holdout-clean")
  body={"schema":"boltbeam.mmq_invocation_freeze.v8","prediction":dict(prediction),"frozen_before_invocation_holdout_read":True};out={**body,"freeze_id":sha256_json(body)};path=pathlib.Path(output)
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

__all__=["InvocationModelV8","fit_invocation_v8","freeze_invocation_v8","predict_invocation_v8"]
