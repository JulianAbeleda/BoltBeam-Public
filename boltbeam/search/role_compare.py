"""Compare kernels per role on Metal: search, matched whole-model A/B, decision, route status.

For every role in a run's route_policy.json this module
  1. builds a finite full-kernel search request (the same 13-row space as the 2026-07-29 Apple M4 record),
  2. runs it through the tinygrad fork's search provider (`search-full-kernel`'s own runner),
  3. times the winner against the default kernel in a matched whole-model decode A/B, in tinygrad's own Metal
     runtime (default, plan, default, plan), with the greedy tokens compared,
  4. turns that into a CandidateDecision with the existing evaluator, appends it to the run's route ledger, and
     writes the route's status into route_policy.json.

Every time it records is a tinygrad runtime time. None of them is a llama.cpp time.
"""
from __future__ import annotations

import datetime as _dt
import hashlib, json, os, pathlib, statistics, subprocess
from collections.abc import Callable, Mapping
from typing import Any

from boltbeam.artifacts.base import EvidenceFlags, EvidenceRow, EvidenceSource, NormalizedEvidence
from boltbeam.eval.evaluator import evaluate
from boltbeam.ledger.model import CandidateDecision, EvidenceRef, status_for_verdict
from boltbeam.ledger.store import LedgerStore
from boltbeam.manifest import Candidate
from boltbeam.plan.resolved_target import candidate_target, resolved_target_document
from boltbeam.search.full_kernel.full_kernel_search import run_full_kernel_search, subprocess_worker
from boltbeam.vocab import Verdict

SCHEMA = "boltbeam.kernel_compare.v1"
FOLDER = "kernel_compare"
TIMING_SOURCE = "tinygrad Metal runtime"
# The backends whose kernels this module can compare (the fork's search provider runs on Metal only).
COMPARE_BACKENDS = ("metal",)
FORK_URL = "https://github.com/JulianAbeleda/tinygrad-arkey"
PROVIDER = "extra/llm_research/search_provider.py"
AB_DRIVER = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "tinygrad_decode_ab.py"
# The fork's default Metal replay cannot encode Qwen3-8B decode (ICB offset over 32 bits); the fork's own
# EXP switch direct-encodes those calls. Both A/B arms run with it, so the comparison stays matched.
RUNTIME_ENV = {"DEV": "METAL", "METAL_HYBRID_REPLAY": "1"}
REOPEN = "reopen when the plan binds by model-graph role identity or the A/B is re-measured at a new revision"

# One row per candidate; the same finite space as bench/metal-qwen3-8b-20260729 (Apple M4).
ROWS: tuple[dict[str, Any], ...] = (
  {"schedule.plan_kind": "tinygrad_heuristic.v1", "schedule.transforms": [], "schedule.launch.threads": 32},
  {"schedule.plan_kind": "tinygrad_opt_sequence.v1", "schedule.transforms": [], "schedule.launch.threads": 1},
  *({"schedule.transforms": [{"op": "UPCAST", "axis": 0, "arg": n}], "schedule.launch.threads": 1} for n in (2, 3, 4)),
  *({"schedule.transforms": [{"op": "LOCAL", "axis": 0, "arg": n}], "schedule.launch.threads": n}
    for n in (8, 16, 32, 64, 128, 256, 512, 1024)),
)
DEFAULT_KIND = "tinygrad_heuristic.v1"


def _sha256_file(path:pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""): digest.update(chunk)
  return digest.hexdigest()


def _now() -> str:
  return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- readiness: what the screen says before anything starts ---------------------------------------------------

def default_fork_root() -> pathlib.Path:
  """$BOLTBEAM_TINYGRAD_ROOT, else tinygrad-arkey-exp next to this BoltBeam checkout."""
  if env := os.environ.get("BOLTBEAM_TINYGRAD_ROOT"):
    return pathlib.Path(env).expanduser()
  return pathlib.Path(__file__).resolve().parents[2].parent / "tinygrad-arkey-exp"


def fork_python(root:pathlib.Path) -> pathlib.Path:
  return root / ".venv" / "bin" / "python"


def readiness(root:pathlib.Path | None = None, model:str | None = None) -> dict[str, Any]:
  """Can this machine compare kernels now? Each missing piece names the one command that fixes it."""
  root = pathlib.Path(root) if root else default_fork_root()
  venv = f"cd {root} && python3.12 -m venv .venv && .venv/bin/pip install numpy"
  out: dict[str, Any] = {"ready": False, "fork": str(root), "missing": None, "message": None, "fix": None}
  if not (root / PROVIDER).is_file():
    return out | {"missing": "fork", "message": f"The tinygrad fork is not at {root}.",
                  "fix": f"git clone -b exp {FORK_URL} {root}"}
  if not fork_python(root).is_file():
    return out | {"missing": "venv", "message": f"The fork has no venv at {root / '.venv'}.", "fix": venv}
  if not any((root / ".venv" / "lib").glob("python3*/site-packages/numpy")):
    return out | {"missing": "numpy", "message": "The fork's venv has no numpy.",
                  "fix": f"{fork_python(root)} -m pip install numpy"}
  if model is not None and not pathlib.Path(model).expanduser().is_file():
    return out | {"missing": "model", "message": f"The model file is not at {model}.", "fix": None}
  return out | {"ready": True}


# --- the request for one role ---------------------------------------------------------------------------------

def describe(root:pathlib.Path, timeout_s:float = 180.0) -> dict[str, Any]:
  line = json.dumps({"protocol": "tinygrad.search_provider.v1", "request_id": "describe", "action": "describe", "payload": {}})
  proc = subprocess.run([str(fork_python(root)), PROVIDER, "--backend", "METAL"], cwd=root, input=line + "\n",
                        capture_output=True, text=True, timeout=timeout_s, env={**os.environ, "PYTHONPATH": "."})
  reply = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
  if reply.get("status") != "ok": raise RuntimeError(f"provider describe failed: {proc.stderr.strip()[-300:] or reply}")
  return reply["result"]


def _workload(role:str, quant:str, n:int, k:int, profile:str, model_sha:str, target:dict[str, Any]) -> dict[str, Any]:
  return {"profile": profile, "model_sha256": model_sha, "phase": "decode", "role": role, "operation": "matmul",
          "shape": {"m": 1, "n": n, "k": k},
          "operands": {"a": {"dtype": "fp16", "layout": "row_major", "quantization": "none"},
                       "b": {"dtype": quant.lower(), "layout": "transposed_row_major", "quantization": quant.lower()},
                       "c": {"dtype": "fp16", "layout": "row_major", "quantization": "none"}},
          "accumulator_dtype": "fp32", "target": target}


def route_request(route:Mapping[str, Any], *, model_id:str, target_id:str, observed:Mapping[str, Any], model_sha:str,
                  provider_revision:str, boltbeam_revision:str | None, run_id:str, timestamp:str) -> dict[str, Any]:
  """The search request for one route_policy row. Shape [N, K] is the GEMV: N output rows, K inputs."""
  role, quant = str(route["role"]), str(route["quant"])
  n, k = (int(x) for x in route["shape"])
  resolved = resolved_target_document(target_id, observed)
  target = candidate_target(resolved)
  slug = "".join(ch if ch.isalnum() else "_" for ch in model_id.lower()).strip("_")
  profile = f"{slug}_{quant.lower().replace('_', '')}_{target_id}"
  workload = _workload(role, quant, n, k, profile, model_sha, target)
  seed = {"schema_version": "boltbeam.full_kernel_candidate.v2", "workload": workload,
          "schedule": {"plan_kind": "tinygrad_opt_sequence.v1", "transforms": [], "tile": {"m": 1, "n": 32, "k": 32},
                       "launch": {"threads": 1}, "mapping": {"lane_policy": "subgroup_contiguous"},
                       "memory": {"a": {"space": "global", "vector_width": 1, "alignment": 2},
                                  "b": {"space": "global", "vector_width": 4, "alignment": 16},
                                  "c": {"space": "global", "vector_width": 1, "alignment": 2}},
                       "pipeline": {"stage_count": 1}, "compute": {"family": "generic_matvec"},
                       "numerical_mode": "fp16_acc_fp32"},
          "static_constraints": {"max_local_memory_bytes": None, "max_registers_per_thread": None, "spill_policy": "unknown"},
          "correctness": {"oracle": "canonical_packed_reference", "atol": 0.001, "rtol": 0.001},
          "memory_budget": {"status": "unavailable", "bytes": None},
          "provenance": {"generator_id": "bubblebeam_futuresight", "generator_revision": provider_revision,
                         "schema_revision": "boltbeam.full_kernel_candidate.v2"},
          "applicability": {"exact_shape": True, "profiles": [profile], "roles": [role],
                            "targets": [f"{target_id}:subgroup{target['subgroup_size']}"]}}
  request = {"schema": "boltbeam.full_kernel_search_request.v1",
             "request_id": f"{run_id}-{role}-{quant.lower()}", "run_id": run_id, "timestamp": timestamp,
             "candidate_space_status": "FINITE", "requested_provider_revision": provider_revision,
             "require_control": True, "execution": {"shape_mode": "exact_workload", "warmups": 2, "samples": 7},
             "target": target, "resolved_target": resolved,
             "candidate_space": {"seed": seed, "rows": [dict(r) for r in ROWS]},
             "workloads": [workload], "budget": {"max_candidates": len(ROWS)},
             "objective": {"metric": "median_ns", "direction": "minimize", "tie_break": "candidate_hash"}}
  if boltbeam_revision: request["requested_boltbeam_revision"] = boltbeam_revision
  return request


# --- reading a search result ----------------------------------------------------------------------------------

def _plan_text(transforms:list[Mapping[str, Any]], kind:str) -> str:
  if kind == DEFAULT_KIND: return "default kernel"
  if not transforms: return "no opts"
  return " ".join(f"{t['op']} {t['arg']}" + (f" on axis {t['axis']}" if t.get("axis") else "") for t in transforms)


def summarize_search(result:Mapping[str, Any]) -> dict[str, Any]:
  """Winner, the default kernel's time, and the counts, from one full-kernel search result."""
  measured = [r for r in result.get("population", []) if r.get("state") == "MEASURED"]
  measured.sort(key=lambda r: (r.get("rank") or 1 << 30))
  def median(r): return (r.get("measurement") or {}).get("median_ns")
  default = next((r for r in measured if r["candidate"]["schedule"]["plan_kind"] == DEFAULT_KIND), None)
  out = {"status": result.get("status"), "counts": dict(result.get("counts", {})), "winner": None,
         "default_median_ns": median(default) if default else None}
  if measured:
    best = measured[0]
    sched = best["candidate"]["schedule"]
    generated = (best.get("worker") or {}).get("generated") or {}
    out["winner"] = {"candidate_hash": best["candidate_hash"], "plan_kind": sched["plan_kind"],
                     "transforms": list(sched.get("transforms", [])), "plan": _plan_text(sched.get("transforms", []), sched["plan_kind"]),
                     "plan_hash": generated.get("plan_hash"), "median_ns": median(best),
                     "is_default": sched["plan_kind"] == DEFAULT_KIND}
  return out


def time_roles(run:pathlib.Path, *, root:pathlib.Path | None = None) -> dict[str, Any]:
  """Time every role inside a real decode in tinygrad's runtime (collectors/tinygrad_role_time.py)."""
  from boltbeam.collectors import tinygrad_role_time
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest
  root = pathlib.Path(root) if root else default_fork_root()
  manifest = load_manifest(run)
  model = str(manifest.get("model_path") or "")
  ready = readiness(root, model)
  if not ready["ready"]:
    raise RuntimeError(f"{ready['message']} Fix: {ready['fix']}" if ready.get("fix") else ready["message"])
  trace = tinygrad_role_time.collect(run, root=root, python=fork_python(root), model=model,
                                     model_id=str(manifest.get("model_id")), target=get_target(manifest.get("target_id")))
  from boltbeam.workflow.screen import _measured_vs_ceiling, _optional
  ceil = _measured_vs_ceiling(manifest, _optional(run, "model_profile.json"))
  table = tinygrad_role_time.loss(ceil.get("_roles") or [], trace, ceil.get("floor_ms"))
  if table and table["status"] != "measured":  # say it where the job ends, not only on the next screen
    raise RuntimeError(table["reason"])
  return trace


def role_losses(run:pathlib.Path) -> dict[tuple[str, str], dict[str, Any]]:
  """The per-role loss rows of a run, by (role, quant); empty before the roles were timed."""
  from boltbeam.collectors import tinygrad_role_time
  from boltbeam.workflow.common import read_json
  from boltbeam.workflow.screen import _measured_vs_ceiling, _optional
  from boltbeam.workflow.common import load_manifest
  path = run / tinygrad_role_time.TRACE
  if not path.is_file(): return {}
  ceil = _measured_vs_ceiling(load_manifest(run), _optional(run, "model_profile.json"))
  table = tinygrad_role_time.loss(ceil.get("_roles") or [], read_json(path), ceil.get("floor_ms")) or {"roles": []}
  return {(r["role"], r["quant"]): r for r in table["roles"]}  # an incomplete table has no roles


def kernel_numbers(summary:Mapping[str, Any], in_model:Mapping[str, Any] | None = None) -> dict[str, Any] | None:
  """Number one of two: the winning plan alone vs the search's reference kernel, at the role shape, same harness.
  The reference is tinygrad's heuristic on the search fixture, not the kernel the model's decode runs (lm_head's
  reference alone took longer than a whole decode token), so this is never read as faster or slower than the
  model. Never a stand-in for the whole-model number."""
  w, d = summary.get("winner"), summary.get("default_median_ns")
  if not w or w.get("median_ns") is None: return None
  out = {"plan_us": w["median_ns"] / 1000.0, "reference_us": d / 1000.0 if d is not None else None,
         "reference": "search fixture heuristic", "model_us_per_call": None, "faster_than_model": None}
  if in_model and in_model.get("calls_per_token"):
    # the baseline: the role's own kernel inside the running model, GPU time per call at the same shape
    per_call = in_model["actual_ms"] * 1000.0 / in_model["calls_per_token"]
    out.update(model_us_per_call=per_call, faster_than_model=out["plan_us"] < per_call)
  return out


def plan_id(role:str, quant:str, winner:Mapping[str, Any]) -> str:
  return f"metal-plan:{role}:{quant}:{(winner.get('plan_hash') or winner['candidate_hash'])[:12]}"


# --- the matched whole-model A/B ------------------------------------------------------------------------------

def run_ab(root:pathlib.Path, model:str, *, transforms:list[Mapping[str, Any]], shape:tuple[int, int],
           pairs:int = 3, timeout_s:float = 1800.0) -> dict[str, Any]:
  """The driver loads the model once and alternates default and plan arms in that one process."""
  argv = [str(fork_python(root)), str(AB_DRIVER), "--model", str(model), "--opts", json.dumps(list(transforms)),
          "--shape", f"{shape[0]},{shape[1]}", "--pairs", str(pairs)]
  proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout_s,
                        env={**os.environ, "PYTHONPATH": ".", **RUNTIME_ENV})
  lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise RuntimeError(f"decode A/B failed: {(proc.stderr or proc.stdout).strip()[-400:]}")
  return json.loads(lines[-1])


def matched_ab(raw:Mapping[str, Any]) -> dict[str, Any]:
  """Medians and noise from the interleaved arms. The plan counts as bound only when it reached decode calls:
  a plan applied to other kernels leaves both arms running the same decode."""
  b = [x for arm in raw["baseline"] for x in arm["tok_s_samples"]]
  c = [x for arm in raw["candidate"] for x in arm["tok_s_samples"]]
  if not c: c = list(b)
  bm, cm = statistics.median(b), statistics.median(c)
  first = raw["baseline"][0]["tokens"]
  binding = dict(raw["binding"])
  return {"timing_source": TIMING_SOURCE, "context": raw["context"], "runtime_env": dict(RUNTIME_ENV),
          "pairs": raw["pairs"], "baseline_tok_s": bm, "candidate_tok_s": cm, "delta_pct": (cm - bm) / bm * 100.0,
          # noise is the wider of the two sides' ranges; overlapping ranges mean no measurable change
          "spread_pct": max(max(b) - min(b), max(c) - min(c)) / bm * 100.0,
          "ranges_overlap": min(c) <= max(b) and min(b) <= max(c),
          "baseline_samples": b, "candidate_samples": c,
          "token_match": all(arm["tokens"] == first for arm in raw["candidate"]),
          "deterministic": all(arm["tokens"] == first for arm in raw["baseline"]),
          "binding": binding, "route_bound": binding.get("calls_reached", 0) > 0 and binding.get("errors", 0) == 0,
          "load_s": raw.get("load_s"), "max_rss_bytes": raw.get("max_rss_bytes")}


# --- the decision ---------------------------------------------------------------------------------------------

def decide(route:Mapping[str, Any], winner:Mapping[str, Any], ab:Mapping[str, Any], *, model_id:str, target_id:str,
           ab_path:str, ab_sha:str) -> CandidateDecision:
  """The existing evaluator judges the A/B: correctness, binding, speed tier, rollback, target completeness."""
  role, quant = str(route["role"]), str(route["quant"])
  cid = plan_id(role, quant, winner)
  candidate = Candidate(candidate_id=cid, workload="decode", quant=(quant,), roles=(role,),
                        required_evidence_kinds=("wd_speed",), rollback={"kernel_plan": "default"},
                        route_family=(route.get("candidates") or [""])[0], origin="search",
                        description=f"{winner['plan']} for {role} {quant}")
  row = EvidenceRow(kind="wd_speed", metric="tok_s", value=float(ab["candidate_tok_s"]), unit="tok/s", role=role,
                    quant=quant, context=int(ab["context"]),
                    extra={"baseline_tok_s": float(ab["baseline_tok_s"]), "spread_pct": float(ab["spread_pct"]),
                           "timing_source": TIMING_SOURCE})
  evidence = NormalizedEvidence(model_id=model_id, target_id=target_id, workload="decode",
                                source=EvidenceSource("tinygrad", "boltbeam/runtime/tinygrad_decode_ab.py", ab_path, f"sha256:{ab_sha}"),
                                rows=(row,), contexts=(int(ab["context"]),),
                                flags=EvidenceFlags(route_bound=bool(ab["route_bound"]), token_match=bool(ab["token_match"]),
                                                    deterministic=bool(ab["deterministic"])))
  return evaluate(candidate, [evidence], model_id=model_id, target_id=target_id, workload="decode")


def default_wins(route:Mapping[str, Any], winner:Mapping[str, Any], *, model_id:str, target_id:str,
                 search_path:str, search_sha:str) -> CandidateDecision:
  """The isolated search already put the default kernel first: no plan to try, a firm refutation."""
  role, quant = str(route["role"]), str(route["quant"])
  ref = EvidenceRef(evidence_id=search_sha, path=search_path, kind="full_kernel_search", fingerprint=f"sha256:{search_sha}",
                    claim=f"the search reference is the fastest measured {role} {quant} fixture kernel")
  return CandidateDecision(candidate_id=plan_id(role, quant, winner), model_id=model_id, target_id=target_id,
                           workload="decode", verdict=Verdict.REFUTE.value, evidence=(ref,),
                           reason="no searched plan beat the search reference kernel alone",
                           next_action="keep the default kernel")


def route_status(verdict:str) -> str:
  """route_policy.json status for a verdict: promote and refute are firm; anything else is not decided yet."""
  return {Verdict.PROMOTE.value: "promoted", Verdict.REFUTE.value: "refuted"}.get(verdict, "blocked")


# --- one run ----------------------------------------------------------------------------------------------------

def _calls(row:Mapping[str, Any] | None) -> str:
  return f"{row['calls_per_token']:.0f}" if row else "unknown number of"


def _git_state(repo:pathlib.Path) -> tuple[str | None, bool]:
  try:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
    return rev, dirty
  except (OSError, subprocess.CalledProcessError):
    return None, True


def _write(path:pathlib.Path, obj:Any) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
  path.write_text(text)
  return hashlib.sha256(text.encode()).hexdigest()


def compare_run(run:pathlib.Path, *, root:pathlib.Path | None = None, say:Callable[[str], None] = print,
                search:Callable[..., dict[str, Any]] | None = None, ab_run:Callable[..., dict[str, Any]] | None = None,
                describe_fn:Callable[[pathlib.Path], dict[str, Any]] | None = None,
                only:set[str] | None = None) -> dict[str, Any]:
  """Compare kernels for every route of one run and write the outcome into the run folder.

  `search`, `ab_run` and `describe_fn` default to the real provider and driver; tests pass doubles.
  """
  from boltbeam.workflow.common import load_manifest, read_json, write_json
  root = pathlib.Path(root) if root else default_fork_root()
  manifest = load_manifest(run)
  model = str(manifest.get("model_path") or "")
  ready = readiness(root, model) if search is None else {"ready": True}
  if not ready["ready"]:
    raise RuntimeError(f"{ready['message']} Fix: {ready['fix']}" if ready.get("fix") else ready["message"])
  target_id, model_id = str(manifest.get("target_id")), str(manifest.get("model_id"))
  policy = read_json(run / "route_policy.json")
  routes = [r for r in policy.get("routes", []) if r.get("role") and r.get("quant") and r.get("shape")]
  if only: routes = [r for r in routes if f"{r['role']}:{r['quant']}" in only]
  if search is None and not (run / "tinygrad_timing_trace.json").is_file():
    say("role-time: start")
    time_roles(run, root=root)
  losses = role_losses(run)
  # the biggest loss first: that is where a better kernel can save the most
  routes.sort(key=lambda r: -losses.get((r["role"], r["quant"]), {}).get("lost_ms", float("-inf")))
  facts = (describe_fn or describe)(root)
  provider_rev = (facts.get("provider_revision") or {}).get("revision")
  bb_rev, bb_dirty = _git_state(pathlib.Path(__file__).resolve().parents[2])
  model_sha = _sha256_file(pathlib.Path(model)) if search is None else "0" * 63 + "1"
  folder = run / FOLDER
  timestamp = _now()
  if search is None:
    def search(request):  # noqa: E306 - the real provider, one process per stage
      worker = subprocess_worker([str(fork_python(root)), "-m", "extra.llm_research.search_provider"], cwd=str(root),
                                 timeout_s=600.0, resolved_target=request["resolved_target"],
                                 requested_provider_revision=request["requested_provider_revision"],
                                 execution=request["execution"])
      return run_full_kernel_search(request, worker, timeout_s=600.0)
  if ab_run is None:
    def ab_run(transforms, shape):  # noqa: E306
      return run_ab(root, model, transforms=transforms, shape=shape)
  record = {"schema": SCHEMA, "run": run.name, "model_id": model_id, "target_id": target_id, "model_path": model,
            "model_sha256": model_sha, "provider_revision": provider_rev, "boltbeam_revision": bb_rev,
            "boltbeam_dirty": bb_dirty, "fork": str(root), "timestamp": timestamp, "timing_source": TIMING_SOURCE,
            "note": "Every time here is a tinygrad Metal runtime time, not a llama.cpp time.", "roles": []}
  ledger = LedgerStore(run / "route_ledger.jsonl")
  say(f"compare roles: {len(routes)}")
  for route in routes:
    role, quant = str(route["role"]), str(route["quant"])
    key, stem = f"{role} {quant}", f"{role}-{quant.lower()}"
    n, k = (int(x) for x in route["shape"])
    say(f"role {key}: search")
    request = route_request(route, model_id=model_id, target_id=target_id, observed=facts["target"], model_sha=model_sha,
                            provider_revision=provider_rev, boltbeam_revision=None if bb_dirty else bb_rev,
                            run_id=f"{run.name}-compare", timestamp=timestamp)
    _write(folder / f"{stem}-search-request.json", request)
    try:
      result = search(request)
    except Exception as exc:  # a provider failure blocks this role, not the others
      result = {"status": "BLOCKED", "counts": {}, "population": [], "error": str(exc)}
    search_sha = _write(folder / f"{stem}-search-result.json", result)
    summary = summarize_search(result)
    row: dict[str, Any] = {"role": role, "quant": quant, "shape": [n, k], "search": summary,
                           "search_result": f"{FOLDER}/{stem}-search-result.json", "ab": None, "decision": None}
    winner = summary["winner"]
    decision: CandidateDecision | None = None
    if winner is None:
      reason = result.get("error") or "no candidate compiled and passed the correctness check"
      row.update(status="blocked", reason=f"decided by the search: {reason}", decided_by="search")
    elif winner["is_default"]:
      decision = default_wins(route, winner, model_id=model_id, target_id=target_id,
                              search_path=row["search_result"], search_sha=search_sha)
    else:
      say(f"role {key}: ab")
      try:
        ab = matched_ab(ab_run(winner["transforms"], (n, k)))
      except Exception as exc:
        ab = None
        row.update(status="blocked", reason=f"decided by the whole model: the A/B did not run: {exc}",
                   decided_by="whole model")
      if ab is not None:
        ab_sha = _write(folder / f"{stem}-ab.json", ab)
        row["ab"] = {key_: ab[key_] for key_ in ("baseline_tok_s", "candidate_tok_s", "delta_pct", "spread_pct", "ranges_overlap", "pairs",
                                                 "token_match", "route_bound", "context", "timing_source")}
        row["ab_result"] = f"{FOLDER}/{stem}-ab.json"
        row["ab"]["binding"] = ab["binding"]
      if ab is not None and not ab["route_bound"]:
        # Both arms ran the same code, so neither number can judge the plan. Not tested is not refuted.
        row.update(status="blocked", decided_by="binding",
                   reason=f"decided by binding: the plan reached {ab['binding'].get('calls_reached', 0)} of the "
                          f"{_calls(losses.get((role, quant)))} {role} {quant} calls per token "
                          f"({ab['binding'].get('decode_graph_calls', 0)} decode calls in all) "
                          f"(applied to {ab['binding'].get('applied', 0)} other kernels); "
                          "the A/B compared the default decode with itself")
      elif ab is not None:
        decision = decide(route, winner, ab, model_id=model_id, target_id=target_id, ab_path=row["ab_result"], ab_sha=ab_sha)
    row["kernel"] = kernel_numbers(summary, losses.get((role, quant)))
    if row["kernel"] and row["kernel"]["model_us_per_call"] is not None:
      row["role_calls_per_token"] = losses[(role, quant)]["calls_per_token"]
    if decision is not None:
      by = "kernel alone" if row["ab"] is None else "whole model"
      row.update(status=route_status(decision.verdict), reason=f"decided by the {by}: {decision.reason}",
                 decided_by=by, decision=decision.to_json())
      if status_for_verdict(decision.verdict) is not None:
        ledger.add(decision, scope={"model": model_id, "target": target_id, "workload": "decode", "role": role, "quant": quant},
                   reopen_condition=REOPEN if decision.verdict == Verdict.REFUTE.value else "")
    if winner is not None: row["plan_id"] = plan_id(role, quant, winner)
    record["roles"].append(row)
    _apply(route, row)
    write_json(run / "route_policy.json", policy)
    _write(folder / "compare.json", record)
    say(f"role {key}: done {row['status']}")
  say(f"compare done: {run}")
  return record


def _apply(route:dict[str, Any], row:Mapping[str, Any]) -> None:
  """Write one role's outcome into its route_policy.json row. selected_route is set only for a promoted plan,
  because consumers of route_policy.json run a selected route."""
  route["status"] = row["status"]
  route["selected_route"] = row.get("plan_id") if row["status"] == "promoted" else None
  route["evidence_refs"] = [x for x in (row.get("search_result"), row.get("ab_result")) if x]
  winner = row["search"]["winner"] or {}
  route["compare"] = {"plan_id": row.get("plan_id"), "plan": winner.get("plan"), "plan_hash": winner.get("plan_hash"),
                      "search_median_ns": winner.get("median_ns"), "default_median_ns": row["search"]["default_median_ns"],
                      "kernel": row.get("kernel"), "decided_by": row.get("decided_by"),
                      "role_calls_per_token": row.get("role_calls_per_token"),
                      "measured_correct": row["search"]["counts"].get("measured_correct"),
                      "candidates": row["search"]["counts"].get("total"), "ab": row.get("ab"), "reason": row.get("reason"),
                      "timing_source": TIMING_SOURCE}

