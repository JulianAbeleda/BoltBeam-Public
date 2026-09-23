"""Replayable, bounded full-kernel search driven only by worker evidence."""
from __future__ import annotations
import datetime as _dt
import functools, json, math, pathlib, re, subprocess
from collections.abc import Callable, Mapping
from typing import Any
from boltbeam.core.canonical import canonical_json as _canonical, sha256_json as _digest
from boltbeam.runtime.process_isolated import run_isolated
from boltbeam.plan.resolved_target import validate_candidate_target
from boltbeam.search.full_kernel.full_kernel_candidates import instantiate_candidate_rows, instantiate_candidates
from boltbeam.search.spec import FullKernelCandidate
from boltbeam.search.full_kernel.tinygrad_full_kernel import TinygradSearchProviderWorker, validate_execution

SCHEMA = "boltbeam.full_kernel_search_result.v1"
REQUEST_SCHEMA = "boltbeam.full_kernel_search_request.v1"
MISSING = "MISSING"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT = re.compile(r"^[0-9a-f]{40}$")

def _git_revision() -> str:
  try: return subprocess.run(["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).parents[3], text=True, capture_output=True, check=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError): return "UNKNOWN"
def _git_dirty() -> bool:
  try:
    return bool(subprocess.run(["git", "status", "--porcelain"], cwd=pathlib.Path(__file__).parents[3], text=True,
                               capture_output=True, check=True).stdout.strip())
  except (OSError, subprocess.CalledProcessError): return True
def _sha(v: Any) -> bool:
  # Repeated digit placeholders ("aaaa…", "0000…") are not evidence.
  return isinstance(v, str) and bool(_SHA256.fullmatch(v)) and len(set(v)) > 1
def _timestamp(v: Any) -> bool:
  if not isinstance(v, str) or not v: return False
  try: _dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
  except ValueError: return False
  return True

def _exact_target(v: Any) -> bool:
  if not isinstance(v, Mapping): return False
  keys = set(v)
  if keys == {"backend", "arch", "wave_size"}:
    return (isinstance(v["backend"], str) and bool(v["backend"]) and isinstance(v["arch"], str) and bool(v["arch"])
            and isinstance(v["wave_size"], int) and not isinstance(v["wave_size"], bool) and v["wave_size"] > 0)
  if keys == {"target_id", "backend", "arch", "subgroup_size", "resolved_target_hash"}:
    return (all(isinstance(v[key], str) and v[key] for key in ("target_id", "backend", "arch"))
            and isinstance(v["subgroup_size"], int) and not isinstance(v["subgroup_size"], bool) and v["subgroup_size"] > 0
            and isinstance(v["resolved_target_hash"], str) and bool(_SHA256.fullmatch(v["resolved_target_hash"])))
  return False

SubprocessWorker = TinygradSearchProviderWorker

def subprocess_worker(command:list[str], *, cwd:str|None=None, timeout_s:float=60.0,
                      resolved_target:Mapping[str, Any]|None=None,
                      requested_provider_revision:str|None=None,
                      execution:Mapping[str, Any]|None=None) -> TinygradSearchProviderWorker:
  return TinygradSearchProviderWorker(tuple(command), cwd, timeout_s, resolved_target, requested_provider_revision, execution)

def _candidates(space: Mapping[str, Any], budget: int) -> tuple[FullKernelCandidate, ...]:
  keys = set(space)
  if keys == {"candidates"}:
    raw = space["candidates"]
    if not isinstance(raw, list) or not raw: raise ValueError("candidate_space.candidates must be non-empty")
    out = tuple(FullKernelCandidate.from_dict(x) for x in raw)
  elif keys == {"seed", "dimensions"}:
    out = instantiate_candidates(FullKernelCandidate.from_dict(space["seed"]), space["dimensions"], max_candidates=budget)
  elif keys == {"seed", "rows"}:
    out = instantiate_candidate_rows(FullKernelCandidate.from_dict(space["seed"]), space["rows"], max_candidates=budget)
  else: raise ValueError("candidate_space must be explicit candidates, seed+dimensions, or seed+rows")
  if len(out) > budget: raise ValueError("candidate space exceeds declared budget")
  if len({x.candidate_hash for x in out}) != len(out): raise ValueError("candidate space contains duplicate candidates")
  return tuple(sorted(out, key=lambda x:x.candidate_hash))

def validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
  if not isinstance(request, Mapping): raise ValueError("search request must be object")
  if request.get("schema") != REQUEST_SCHEMA: raise ValueError("request schema/version is required")
  if request.get("candidate_space_status", MISSING) == MISSING: raise ValueError("candidate_space_status=MISSING: refusing unspecified space")
  required={"schema","request_id","run_id","timestamp","candidate_space_status","candidate_space","target","workloads","budget","objective"}
  if set(request) - (required|{"requested_boltbeam_revision","requested_provider_revision","resolved_target","require_control","execution","execution_order"}) or required-set(request): raise ValueError("request has missing or unknown public fields")
  if request["candidate_space_status"] != "FINITE": raise ValueError("candidate_space_status must be FINITE")
  if not all(isinstance(request[k],str) and request[k] for k in ("request_id","run_id")) or not _timestamp(request["timestamp"]): raise ValueError("request_id, run_id, and ISO-8601 timestamp required")
  for field in ("requested_boltbeam_revision", "requested_provider_revision"):
    if field in request and (not isinstance(request[field], str) or not request[field]): raise ValueError(f"{field} must be non-empty")
  if "require_control" in request and not isinstance(request["require_control"], bool): raise ValueError("require_control must be bool")
  if not _exact_target(request["target"]): raise ValueError("target must be an exact v1 or resolved v2 target")
  budget=request["budget"]
  if not isinstance(budget,Mapping) or set(budget)!={"max_candidates"} or not isinstance(budget["max_candidates"],int) or isinstance(budget["max_candidates"],bool) or budget["max_candidates"]<=0: raise ValueError("budget.max_candidates required")
  objective=request["objective"]
  if not isinstance(objective,Mapping) or set(objective)!={"metric","direction","tie_break"} or not isinstance(objective.get("metric"),str) or not objective["metric"] or objective.get("direction") not in ("minimize","maximize") or objective.get("tie_break")!="candidate_hash": raise ValueError("invalid objective")
  if not isinstance(request["workloads"],list) or not request["workloads"] or any(not isinstance(x,Mapping) for x in request["workloads"]): raise ValueError("exact workloads required")
  candidates=_candidates(request["candidate_space"], budget["max_candidates"])
  order=request.get("execution_order")
  if order is not None:
    if not isinstance(order,list) or len(order)!=len(candidates) or len(set(order))!=len(order) or set(order)!={c.candidate_hash for c in candidates}: raise ValueError("execution_order must be an exact candidate-hash permutation")
    candidates=tuple(next(c for c in candidates if c.candidate_hash==h) for h in order)
  uses_resolved_target = any(candidate.schema_version in ("boltbeam.full_kernel_candidate.v2", "boltbeam.full_kernel_candidate.v3") for candidate in candidates)
  if uses_resolved_target:
    if "resolved_target" not in request: raise ValueError("v2 request requires resolved_target document")
    validate_candidate_target(request["target"], request["resolved_target"])
    if "execution" not in request: raise ValueError("v2 request requires explicit execution regime")
  elif "resolved_target" in request: raise ValueError("v1 request cannot carry a resolved-target document")
  execution = validate_execution(request["execution"]) if "execution" in request else None
  if request.get("require_control", False):
    controls = [candidate for candidate in candidates if candidate.payload["schedule"].get("plan_kind") == "tinygrad_heuristic.v1"]
    if len(controls) != 1: raise ValueError("finite candidate space must contain exactly one tinygrad_heuristic.v1 control")
  workload_keys={_canonical(x) for x in request["workloads"]}
  candidate_workload_keys={_canonical(c.workload) for c in candidates}
  if candidate_workload_keys != workload_keys: raise ValueError("candidate space must cover every declared workload exactly")
  for c in candidates:
    if c.target != dict(request["target"]) or _canonical(c.workload) not in workload_keys: raise ValueError("candidate target/workload outside declared request")
  actual=_git_revision(); requested=request.get("requested_boltbeam_revision")
  if not _GIT.fullmatch(actual): raise ValueError("BoltBeam exact git revision is unavailable; executable FINITE run is BLOCKED")
  if requested is not None and requested != actual: raise ValueError("requested_boltbeam_revision does not match BoltBeam revision")
  if requested is not None and _git_dirty(): raise ValueError("requested BoltBeam revision requires a clean worktree")
  return {"candidates":candidates,"objective":dict(objective),"revision":actual,"budget":budget,"execution":execution}

def _outcome(c, worker, timeout, metric):
  base={"candidate_hash":c.candidate_hash,"candidate":c.to_dict()}
  if isinstance(worker, SubprocessWorker) or getattr(worker,"direct_worker",False): value=worker(c)
  else:
    try: stage=run_isolated(functools.partial(worker, c), timeout_s=timeout)
    except Exception as exc: return base|{"state":"BLOCKED","reason":f"worker_isolation_error: {exc}"}
    if not stage.ok: return base|{"state":"BLOCKED","reason":stage.error or stage.status}
    value=stage.value
  if not isinstance(value,Mapping) or value.get("candidate_hash") != c.candidate_hash: return base|{"state":"BLOCKED","reason":"invalid worker identity"}
  rejection=value.get("candidate_rejection")
  if rejection is not None:
    if not isinstance(rejection,Mapping) or rejection.get("state") not in {"REJECTED_INVALID","REJECTED_UNSUPPORTED","REJECTED_STATIC"} \
        or not all(isinstance(rejection.get(key),str) and rejection[key] for key in ("stage","code","reason")):
      return base|{"state":"BLOCKED","reason":"invalid candidate rejection evidence","worker":dict(value)}
    return base|{"state":rejection["state"],"reason":rejection["reason"],"worker":dict(value)}
  if value.get("blocked_reason"): return base|{"state":"BLOCKED","reason":str(value["blocked_reason"]),"worker":dict(value)}
  w=c.workload
  required_hashes=(value.get("correctness",{}).get("evidence_hash"), value.get("performance",{}).get("evidence_hash"), value.get("generated",{}).get("source_hash"), value.get("generated",{}).get("plan_hash"), value.get("generated",{}).get("binary_hash"))
  if value.get("target")!=w["target"] or value.get("workload")!=w or not all(_sha(x) for x in required_hashes): return base|{"state":"BLOCKED","reason":"missing target-bound hardware/evidence/generated hashes","worker":dict(value)}
  hw=value.get("hardware")
  if not isinstance(hw,Mapping) or hw.get("target")!=w["target"] or not isinstance(hw.get("device_id"),str) or not hw["device_id"]: return base|{"state":"BLOCKED","reason":"missing target-bound hardware facts","worker":dict(value)}
  if value.get("correctness",{}).get("passed") is not True: return base|{"state":"REJECTED_INCORRECT","worker":dict(value)}
  measurement=value.get("performance",{}).get("measurement",{})
  score=measurement.get(metric) if isinstance(measurement,Mapping) else None
  if not isinstance(score,(int,float)) or isinstance(score,bool) or not math.isfinite(score): return base|{"state":"BLOCKED","reason":"missing finite measured objective","worker":dict(value)}
  return base|{"state":"MEASURED","score":score,"measurement":dict(measurement),"worker":dict(value)}

def run_full_kernel_search(request: Mapping[str, Any], worker: Callable[[FullKernelCandidate],Any], *, timeout_s:float=60.0) -> dict[str,Any]:
  plan=validate_request(request)
  rows=[_outcome(c,worker,timeout_s,plan["objective"]["metric"]) for c in plan["candidates"]]
  measured=sorted((x for x in rows if x["state"]=="MEASURED"), key=lambda x:((-x["score"] if plan["objective"]["direction"]=="maximize" else x["score"]),x["candidate_hash"]))
  for i,x in enumerate(measured,1): x["rank"]=i
  selected={}
  for x in measured:
    key=_canonical(x["candidate"]["workload"]); selected.setdefault(key,{"shape_key":json.loads(key),"candidate_hash":x["candidate_hash"],"candidate":x["candidate"],"rank":x["rank"],"objective":plan["objective"],"measurement":x["measurement"]})
  blocked=sum(x["state"]=="BLOCKED" for x in rows)
  complete=bool(measured) and blocked == 0 and len(selected) == len(request["workloads"])
  return {"schema":SCHEMA,"run_id":request["run_id"],"request_id":request["request_id"],"timestamp":request["timestamp"],"boltbeam_revision":plan["revision"],"request_digest":_digest(request),"candidate_space_digest":_digest(request["candidate_space"]),"candidate_space_status":"FINITE","execution":plan["execution"],"execution_digest":_digest(plan["execution"]) if plan["execution"] is not None else None,"objective":plan["objective"],"budget":dict(plan["budget"])|{"enumerated":len(rows)},"counts":{"total":len(rows),"measured_correct":len(measured),"blocked":blocked,"rejected_invalid":sum(x["state"]=="REJECTED_INVALID" for x in rows),"rejected_unsupported":sum(x["state"]=="REJECTED_UNSUPPORTED" for x in rows),"rejected_static":sum(x["state"]=="REJECTED_STATIC" for x in rows),"rejected_incorrect":sum(x["state"]=="REJECTED_INCORRECT" for x in rows)},"population":rows,"selected_shape_key_plans":[selected[k] for k in sorted(selected)],"status":"COMPLETE" if complete else "BLOCKED"}

def deterministic_json(result: Mapping[str,Any])->str: return json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
__all__=["MISSING","REQUEST_SCHEMA","SCHEMA","SubprocessWorker","deterministic_json","run_full_kernel_search","subprocess_worker","validate_request"]
