"""Validate external FutureSight static evidence against canonical BoltBeam IDs."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
from boltbeam.core.canonical import sha256_json as _digest
ASSESSMENT_VERSION="bubblebeam.futuresight.static.v1"
def row_hash(row:Mapping[str,Any])->str: return _digest(row)
def population_hash(candidates:Sequence[Any], coupled_rows:Sequence[Mapping[str,Any]]=())->str:
  return _digest({"candidates":[c.canonical_json() if hasattr(c,"canonical_json") else c for c in candidates],"rows":[row_hash(r) for r in coupled_rows]})
def bind_futuresight_evidence(candidates:Sequence[Any], evidence:Mapping[str,Any], coupled_rows:Sequence[Mapping[str,Any]]=())->dict[str,Any]:
  hashes={c.candidate_hash for c in candidates}; rows={row_hash(r) for r in coupled_rows}
  if not isinstance(evidence, Mapping) or not {"assessments","rejections"}.issubset(evidence) or set(evidence)-{"assessment_version","assessments","rejections","population_hash","rejected_coupled_rows"}: raise ValueError("FutureSight evidence requires exact fields")
  if evidence.get("assessment_version",ASSESSMENT_VERSION) != ASSESSMENT_VERSION: raise ValueError(f"FutureSight evidence version must be {ASSESSMENT_VERSION}")
  assessments=evidence["assessments"]; rejections=evidence["rejections"]
  if "population_hash" in evidence and evidence["population_hash"] != population_hash(candidates,coupled_rows): raise ValueError(f"FutureSight population hash mismatch: {evidence['population_hash']} != {population_hash(candidates,coupled_rows)}")
  if not isinstance(assessments,list) or not isinstance(rejections,list): raise ValueError("FutureSight evidence requires lists")
  for item in assessments:
    if not isinstance(item,Mapping) or set(item) != {"candidate_hash","static_score","static_reason"} or item.get("candidate_hash") not in hashes or not isinstance(item["static_score"],int) or not isinstance(item["static_reason"],str): raise ValueError("unknown FutureSight candidate hash")
  for item in rejections:
    if not isinstance(item,Mapping) or (item.get("row_hash") not in rows and item.get("candidate_hash") not in hashes): raise ValueError("unknown FutureSight row hash")
  ids=[x["candidate_hash"] for x in assessments]+[x.get("candidate_hash") for x in rejections if "candidate_hash" in x]
  if len(ids)!=len(set(ids)) or set(ids)!=hashes: raise ValueError("incomplete FutureSight candidate coverage")
  return {"assessment_version":ASSESSMENT_VERSION,"assessments":[dict(x) for x in assessments],"rejections":[dict(x) for x in rejections]}
