"""Materialize canonical semantic population IDs for external static assessment."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from boltbeam.search.semantic.semantic_candidate_plan import build_semantic_population, semantic_workload_facts
from boltbeam.search.futuresight_evidence import row_hash, population_hash
POPULATION_SCHEMA="boltbeam.semantic_population.v1"
def export_population(request:Mapping[str,Any])->dict[str,Any]:
  population=build_semantic_population(request["semantic_workload"],request["schedule"],compiler_facts=request["compiler_facts"],propose_dimensions=lambda _w,_f: request["dimensions"],coupled_rows=request.get("legal_coupled_rows",[]),max_candidates=request["budget"]["max_candidates"])
  candidates=population["candidates"]
  rows=request.get("legal_coupled_rows",[]); rejected=request.get("rejected_coupled_rows",[])
  result={"schema":POPULATION_SCHEMA,"workload_facts":semantic_workload_facts(request["semantic_workload"]),"compiler_facts":dict(request["compiler_facts"]),"candidates":[{"candidate_hash":c.candidate_hash,"candidate":c.to_dict()} for c in candidates],"coupled_rows":[{"row_hash":row_hash(r),"row":dict(r)} for r in rows],"rejected_coupled_rows":[{"row_hash":row_hash(r["row"]),"row":dict(r["row"]),"reason":r["reason"]} for r in rejected]}
  result["population_hash"]=population_hash(candidates,rows)
  return result
