"""Serialized, cross-repository entry point for an exact semantic campaign."""
from __future__ import annotations
import argparse, json, pathlib
from typing import Any
from boltbeam.search.semantic.semantic_search_campaign import run_semantic_search_campaign
from boltbeam.search.semantic.semantic_candidate_plan import build_semantic_population
from boltbeam.search.full_kernel.tinygrad_full_kernel import PersistentJSONLSession

# The exact field set of a semantic campaign request: run_request consumes it, propose_request produces it.
REQUEST_FIELDS=frozenset({"semantic_workload","schedule","dimensions","legal_coupled_rows","rejected_coupled_rows","futuresight_evidence","compiler_facts","resolved_target","gguf_path","execution","request_id","run_id","timestamp","budget"})

def run_request(request: dict[str, Any], *, provider_command: list[str], futuresight_evidence: dict[str, Any] | None = None,
                finalist_count:int = 0, finalist_repeats:int = 0,
                provider_session:PersistentJSONLSession|None = None,
                requested_provider_revision:str|None = None,
                requested_boltbeam_revision:str|None = None) -> dict[str, Any]:
  if set(request) != REQUEST_FIELDS: raise ValueError("semantic campaign request has missing or unknown fields")
  budget=request["budget"]
  if not isinstance(budget,dict) or set(budget)!={"max_candidates","timeout_s"}: raise ValueError("budget requires max_candidates and timeout_s")
  if futuresight_evidence is not None: request = dict(request) | {"futuresight_evidence": futuresight_evidence}
  def bridge(workload, schedule, facts, maximum):
    return build_semantic_population(workload, schedule, compiler_facts=facts, propose_dimensions=lambda _w,_f: request["dimensions"], coupled_rows=request["legal_coupled_rows"], max_candidates=maximum)
  session = provider_session or PersistentJSONLSession(tuple(provider_command))
  return run_semantic_search_campaign(request["semantic_workload"],request["schedule"],request["dimensions"],worker=session,request_id=request["request_id"],run_id=request["run_id"],timestamp=request["timestamp"],execution=request["execution"],resolved_target=request["resolved_target"],gguf_path=request["gguf_path"],max_candidates=budget["max_candidates"],population_builder=bridge,compiler_facts=request["compiler_facts"],timeout_s=budget["timeout_s"],futuresight_evidence=request["futuresight_evidence"],coupled_rows=request["legal_coupled_rows"],rejected_coupled_rows=request["rejected_coupled_rows"],finalist_count=finalist_count,finalist_repeats=finalist_repeats,manage_provider_session=provider_session is None,requested_provider_revision=requested_provider_revision,requested_boltbeam_revision=requested_boltbeam_revision)

def main(argv:list[str]|None=None)->int:
  ap=argparse.ArgumentParser(description="Run a finite exact semantic campaign from serialized BubbleBeam/FutureSight output")
  ap.add_argument("request",type=pathlib.Path); ap.add_argument("--provider",nargs="+",required=True); ap.add_argument("--futuresight-evidence",type=pathlib.Path); ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args(argv); result=run_request(json.loads(args.request.read_text()),provider_command=args.provider,futuresight_evidence=json.loads(args.futuresight_evidence.read_text()) if args.futuresight_evidence else None)
  args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n"); return 0
if __name__=="__main__": raise SystemExit(main())
