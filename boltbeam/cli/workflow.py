from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

from boltbeam.cli._common import (_write, _profile, _parse_ctxs, _parse_route_flags,
  _run_manifest_defaults, _fail, _load_json, _load_evidences, _ensure_ledger_file, _add_common)
from boltbeam.analyze import emit_analysis_bundle
from boltbeam.profile.loaders import profile_from_model
from boltbeam.search.emit import emit_search_space
from boltbeam.policy.emit import emit_seed_policy
from boltbeam.synth.fixtures import emit_fixture_manifest
from boltbeam.target.targets import get_target

from boltbeam.vocab import Verdict, SCHEMA_MODEL_PROFILE, SCHEMA_EVIDENCE_BUNDLE, SCHEMA_INGEST_ERROR
from boltbeam.artifacts.tinygrad import normalize, normalize_dir
from boltbeam.artifacts.base import NormalizedEvidence, AdapterIncomplete, UnsupportedArtifact
from boltbeam.ledger.model import CandidateDecision, LedgerEntry, status_for_verdict
from boltbeam.ledger.store import LedgerStore
from boltbeam.eval.evaluator import (evaluate, evaluate_all, candidate_in_search_space,
                                     candidates_in_search_space, primary_decision)
from boltbeam.manifest import candidate_by_id, load_candidates
from boltbeam.policy.check import check_policy
from boltbeam.report.markdown import render_report
from boltbeam.cache.store import CacheStore
from boltbeam.workflow import (load_run, autoscan_run, analyze_run, output_run, ingest_probe_run,
                               ingest_timing_run, runner_plan_run)

def _register_inspect(sub) -> None:
  p = sub.add_parser("inspect", help="emit model profile JSON from GGUF/safetensors/AWQ/GPTQ/ONNX/MLX")
  _add_common(p); p.set_defaults(fn=cmd_inspect)



def cmd_inspect(args) -> int:
  profile, _target = _profile(args)
  _write(profile.to_json(), args.out)
  return 0

def _register_emit_search(sub) -> None:
  p = sub.add_parser("emit-search", help="emit candidate search space JSON")
  _add_common(p); p.set_defaults(fn=cmd_emit_search)



def cmd_emit_search(args) -> int:
  profile, target = _profile(args)
  _write(emit_search_space(profile, target), args.out)
  return 0

def _register_vocab(sub) -> None:
  p = sub.add_parser("vocab", help="derive the model's complete required codegen-primitive vocabulary + lowered/backlog coverage")
  _add_common(p); p.set_defaults(fn=cmd_vocab)



def cmd_vocab(args) -> int:
  from boltbeam.vocabulary.model_vocab import extract_model_vocabulary
  profile, target = _profile(args)
  _write(extract_model_vocabulary(profile, target).to_json(), args.out)
  return 0

def _register_vocab_capture(sub) -> None:
  p = sub.add_parser("vocab-capture", help="ingest an empirical vocab capture (real scheduled ISA) -> used/lowered-but-unused/backlog")
  p.add_argument("capture", help="capture JSON from tinygrad extra/qk_vocab_capture.py")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_vocab_capture)



def cmd_vocab_capture(args) -> int:
  from boltbeam.vocabulary.vocab_capture import ingest_capture
  cap = _load_json(args.capture)
  _write(ingest_capture(cap), args.out)
  return 0

def _register_decode_role_profile(sub) -> None:
  p = sub.add_parser("decode-role-profile", help="profile GGUF decode weight roles without loading tinygrad")
  p.add_argument("model", help="GGUF model path")
  p.add_argument("--id", default=None, help="profile/model id override")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_decode_role_profile)



def cmd_decode_role_profile(args) -> int:
  from boltbeam.profile.decode_roles import profile_from_gguf
  _write(profile_from_gguf(args.model, model_id=args.id).to_json(), args.out)
  return 0

def _register_boundary_plan(sub) -> None:
  p = sub.add_parser("boundary-plan", help="convert tinygrad's boundary audit into an actionable BoltBeam migration plan")
  p.add_argument("audit", help="tinygrad boltbeam_boundary_audit.json")
  p.add_argument("--out", default=None, help="write plan JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="optional markdown report path")
  p.set_defaults(fn=cmd_boundary_plan)



def cmd_boundary_plan(args) -> int:
  from boltbeam.plan.boundary import build_boundary_plan, boundary_plan_markdown, load_tinygrad_boundary_audit
  plan = build_boundary_plan(load_tinygrad_boundary_audit(args.audit))
  _write(plan, args.out)
  if args.markdown:
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(boundary_plan_markdown(plan))
  return 0

def _register_route_manifest(sub) -> None:
  p = sub.add_parser("route-manifest", help="emit/validate the BoltBeam-owned tinygrad route manifest")
  p.add_argument("--out", default=None, help="write manifest JSON here instead of stdout")
  p.add_argument("--dump-default", default=None, help="optional path for default-route manifest JSON")
  p.add_argument("--dump-refuted", default=None, help="optional path for refuted-axes JSON")
  p.set_defaults(fn=cmd_route_manifest)



def cmd_route_manifest(args) -> int:
  from boltbeam.policy.route_manifest import dump, dump_refuted, to_manifest_dict, validate_manifest
  errors = validate_manifest()
  if errors:
    return _fail("route-manifest: validation failed:\n- " + "\n- ".join(errors))
  if args.out:
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(to_manifest_dict(), indent=2, sort_keys=True) + "\n")
  else:
    _write(to_manifest_dict(), None)
  if args.dump_refuted:
    dump_refuted(args.dump_refuted)
  if args.dump_default:
    dump(args.dump_default)
  return 0

def _register_emit_policy(sub) -> None:
  p = sub.add_parser("emit-policy", help="emit seed route policy JSON")
  _add_common(p); p.set_defaults(fn=cmd_emit_policy)



def cmd_emit_policy(args) -> int:
  profile, target = _profile(args)
  _write(emit_seed_policy(profile, target), args.out)
  return 0

def _register_synth(sub) -> None:
  p = sub.add_parser("synth", help="emit synthetic fixture manifest")
  _add_common(p); p.set_defaults(fn=cmd_synth)



def cmd_synth(args) -> int:
  profile, _target = _profile(args)
  _write(emit_fixture_manifest(profile), args.out)
  return 0

def _register_load(sub) -> None:
  p = sub.add_parser("load", help="stage model/weight/workload facts into a provider-neutral run directory")
  p.add_argument("model", help="model artifact path")
  p.add_argument("--run", required=True, help="run directory for staged artifacts")
  p.add_argument("--target", default=None,
                 help="explicit target id; when omitted autoscan may replace the legacy AMD default")
  p.add_argument("--id", default=None, help="profile/model id override")
  p.add_argument("--workload", default="decode", choices=["decode", "prefill"])
  p.add_argument("--ctxs", default="128,512", help="comma-separated contexts to seed the workload profile")
  p.add_argument("--out", default=None, help="write run manifest JSON here instead of stdout")
  p.set_defaults(fn=cmd_load)



def cmd_load(args) -> int:
  manifest = load_run(args.model, args.run, target_id=args.target or "amd_gfx1100", model_id=args.id,
                      workload=args.workload, contexts=_parse_ctxs(args.ctxs),
                      target_source="user" if args.target else "default")
  _write(manifest, args.out)
  return 0

def _register_autoscan(sub) -> None:
  p = sub.add_parser("autoscan", help="record cheap machine/provider facts for a staged run")
  p.add_argument("--run", required=True, help="run directory created by `boltbeam load`")
  p.add_argument("--provider", action="append", default=[],
                 help="provider hint, either NAME or NAME=/path (repeatable)")
  p.add_argument("--out", default=None, help="write run manifest JSON here instead of stdout")
  p.set_defaults(fn=cmd_autoscan)



def cmd_autoscan(args) -> int:
  manifest = autoscan_run(args.run, providers=tuple(args.provider or ()))
  _write(manifest, args.out)
  return 0

def _register_analyze(sub) -> None:
  p = sub.add_parser("analyze", help="emit a Claude-ready profile/search/measurement bundle")
  p.add_argument("model", nargs="?", help="GGUF model path (legacy mode; omit when using --run)")
  p.add_argument("--run", default=None, help="provider-neutral run directory created by `boltbeam load`")
  p.add_argument("--target", default="amd_gfx1100")
  p.add_argument("--id", default=None, help="profile/model id override")
  p.add_argument("--out-dir", default=None, help="directory for legacy bundle artifacts")
  p.add_argument("--tinygrad-root", default=None,
                 help="tinygrad checkout used in generated commands (default: TINYGRAD_ROOT or recognized sibling)")
  p.add_argument("--ctxs", default="128,512", help="comma-separated decode contexts for measurement commands")
  p.add_argument("--max-context", type=int, default=4608)
  p.add_argument("--route-flags", default="DECODE_Q4K_G3_ANYSHAPE=1", help="comma-separated KEY=VALUE env flags for route-on captures")
  p.add_argument("--out", default=None, help="write run manifest JSON here instead of stdout in --run mode")
  p.set_defaults(fn=cmd_analyze)



def cmd_analyze(args) -> int:
  if getattr(args, "run", None):
    manifest = analyze_run(args.run)
    _write(manifest, args.out)
    return 0
  if not args.model:
    return _fail("analyze: MODEL is required unless --run is supplied")
  if not args.out_dir:
    return _fail("analyze: --out-dir is required unless --run is supplied")
  profile, target = _profile(args)
  manifest = emit_analysis_bundle(
    profile=profile,
    target=target,
    model_path=args.model,
    out_dir=args.out_dir,
    ctxs=_parse_ctxs(args.ctxs),
    tinygrad_root=args.tinygrad_root,
    max_context=args.max_context,
    route_flags=_parse_route_flags(args.route_flags),
  )
  sys.stdout.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  return 0

def _register_output(sub) -> None:
  p = sub.add_parser("output", help="package a staged run into policy/report/provider-plan artifacts")
  p.add_argument("--run", required=True, help="run directory created by `boltbeam load`")
  p.add_argument("--out", default=None, help="write run manifest JSON here instead of stdout")
  p.set_defaults(fn=cmd_output)



def cmd_output(args) -> int:
  manifest = output_run(args.run)
  _write(manifest, args.out)
  return 0

def _register_ingest_probe(sub) -> None:
  p = sub.add_parser("ingest-probe", help="ingest provider-neutral primitive probe evidence into a staged run")
  p.add_argument("evidence", help="boltbeam.probe_evidence.v1 JSON from an external runner")
  p.add_argument("--run", required=True, help="run directory created by `boltbeam load`")
  p.add_argument("--out", default=None, help="write primitive profile JSON here instead of stdout")
  p.set_defaults(fn=cmd_ingest_probe)



def cmd_ingest_probe(args) -> int:
  try:
    profile = ingest_probe_run(args.run, args.evidence)
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"ingest-probe: {exc}")
  _write(profile, args.out)
  return 0

def _register_ingest_timing(sub) -> None:
  p = sub.add_parser("ingest-timing", help="ingest provider-neutral timing trace evidence into a staged run")
  p.add_argument("trace", help="boltbeam.timing_trace.v1 JSON from an external runner")
  p.add_argument("--run", required=True, help="run directory created by `boltbeam load`")
  p.add_argument("--out", default=None, help="write timing profile JSON here instead of stdout")
  p.set_defaults(fn=cmd_ingest_timing)



def cmd_ingest_timing(args) -> int:
  try:
    profile = ingest_timing_run(args.run, args.trace)
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"ingest-timing: {exc}")
  _write(profile, args.out)
  return 0

def _register_runner_plan(sub) -> None:
  p = sub.add_parser("runner-plan", help="prepare an audit-tracer handoff bundle for a tinygrad-side runtime executor")
  p.add_argument("--run", required=True, help="run directory created by `boltbeam load` and `boltbeam analyze`")
  p.add_argument("--provider", action="append", default=[],
                 help="runtime executor hint, usually tinygrad or tinygrad=/path (repeatable)")
  p.add_argument("--bundle-dir", default=None,
                 help="optional bundle directory; defaults to <run>/runner_bundle")
  p.add_argument("--out", default=None, help="write runner plan JSON here instead of stdout")
  p.set_defaults(fn=cmd_runner_plan)



def cmd_runner_plan(args) -> int:
  try:
    plan = runner_plan_run(args.run, providers=tuple(args.provider or ()), bundle_dir=args.bundle_dir)
  except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
    return _fail(f"runner-plan: {exc}")
  _write(plan, args.out)
  return 0

def _register_matched_control(sub) -> None:
  p = sub.add_parser("matched-control", help="run a serial, identity-complete tinygrad/llama fixed-depth control")
  p.add_argument("--model", required=True, help="GGUF model path")
  p.add_argument("--tinygrad-root", required=True, help="clean tinygrad EXP checkout")
  p.add_argument("--llama-root", required=True, help="clean llama.cpp checkout")
  p.add_argument("--llama-bench", required=True, help="pinned llama-bench timing executable")
  p.add_argument("--llama-server", required=True, help="pinned llama-server correctness-canary executable")
  p.add_argument("--output-root", required=True, help="new output directory; must not already exist")
  p.add_argument("--target-id", default="apple_m4_10c")
  p.add_argument("--depth", type=int, default=128)
  p.add_argument("--warmups", type=int, default=2)
  p.add_argument("--samples", type=int, default=5)
  p.add_argument("--minimum-free-memory-percent", type=int, default=10)
  p.add_argument("--run-id", default=None)
  p.add_argument("--plan-only", action="store_true", help="capture preflight identities and print the plan without runtimes")
  p.add_argument("--out", default=None, help="optional plan/summary JSON path")
  p.set_defaults(fn=cmd_matched_control)



def cmd_matched_control(args) -> int:
  """Build or execute the fail-closed fixed-depth matched-control protocol."""
  from boltbeam.control.matched_control import build_protocol, execute_protocol
  try:
    plan = build_protocol(model=args.model, tinygrad_root=args.tinygrad_root, llama_root=args.llama_root,
                          llama_bench=args.llama_bench, llama_server=args.llama_server, output_root=args.output_root,
                          target_id=args.target_id, depth=args.depth, warmups=args.warmups,
                          samples=args.samples, minimum_free_memory_percent=args.minimum_free_memory_percent,
                          run_id=args.run_id)
    if args.plan_only:
      _write(plan, args.out)
      return 0 if plan["preflight"]["status"] == "ready" else 2
    summary = execute_protocol(plan)
  except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError,
          json.JSONDecodeError) as exc:
    return _fail(f"matched-control: {exc}")
  _write(summary, args.out)
  return 0 if summary["status"] == "complete" else 2

def _register_replay_ab(sub) -> None:
  p = sub.add_parser("replay-ab", help="run paired serial METAL_HYBRID_REPLAY=0/1 fixed-depth rows")
  p.add_argument("--model", required=True, help="Qwen3-8B GGUF model path")
  p.add_argument("--tinygrad-root", required=True, help="clean tinygrad EXP checkout")
  p.add_argument("--output-root", required=True, help="new output directory; must not already exist")
  p.add_argument("--target-id", default="apple_m4_10c")
  p.add_argument("--depth", type=int, default=128)
  p.add_argument("--warmups", type=int, default=2)
  p.add_argument("--samples", type=int, default=5)
  p.add_argument("--minimum-free-memory-percent", type=int, default=10)
  p.add_argument("--run-id", default=None)
  p.add_argument("--plan-only", action="store_true", help="capture identities and print the plan without runtimes")
  p.add_argument("--out", default=None, help="optional plan/summary JSON path")
  p.set_defaults(fn=cmd_replay_ab)



def cmd_replay_ab(args) -> int:
  """Build or execute the paired Metal hybrid-replay protocol."""
  from boltbeam.control.replay_ab import build_protocol, execute_protocol
  try:
    plan = build_protocol(model=args.model, tinygrad_root=args.tinygrad_root, output_root=args.output_root,
                          target_id=args.target_id, depth=args.depth, warmups=args.warmups,
                          samples=args.samples, minimum_free_memory_percent=args.minimum_free_memory_percent,
                          run_id=args.run_id)
    if args.plan_only:
      _write(plan, args.out)
      return 0 if plan["preflight"]["status"] == "ready" else 2
    summary = execute_protocol(plan)
  except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError,
          json.JSONDecodeError) as exc:
    return _fail(f"replay-ab: {exc}")
  _write(summary, args.out)
  return 0 if summary["status"] == "complete" and summary.get("verdict") != "INCONCLUSIVE" else 2


# ---- machine-contract helpers (all commands exit nonzero on malformed contracts) -------------------------

def _register_ingest(sub) -> None:
  # ---- BB8 audit-loop commands ---------------------------------------------------------------------------
  p = sub.add_parser("ingest", help="normalize a tinygrad artifact (or directory) into evidence JSON")
  p.add_argument("path", help="artifact file or directory")
  p.add_argument("--kind", default="auto", help="artifact kind (only 'auto' supported in v1)")
  p.add_argument("--model-id", default=None, dest="model_id",
                 help="override/fill model_id for artifacts that do not carry one (e.g. reduce-source traces)")
  p.add_argument("--target-id", default=None, dest="target_id",
                 help="override/fill target_id for non-gfx1100 artifacts (avoids silently defaulting to amd_gfx1100)")
  p.add_argument("--out", default=None, help="write evidence JSON here instead of stdout")
  p.set_defaults(fn=cmd_ingest)



def cmd_ingest(args) -> int:
  p = pathlib.Path(args.path)
  if not p.exists():
    return _fail(f"ingest: no such path {args.path!r}")
  if args.kind and args.kind != "auto":
    return _fail(f"ingest: only --kind auto is supported in v1, got {args.kind!r}")
  model_id = getattr(args, "model_id", None)
  target_id = getattr(args, "target_id", None)
  if p.is_dir():
    evs = normalize_dir(str(p), model_id=model_id, target_id=target_id)
    bundle = {"schema": SCHEMA_EVIDENCE_BUNDLE, "count": len(evs), "evidence": [e.to_json() for e in evs],
              "skipped": [{"path": sp, "reason": r} for sp, r in getattr(normalize_dir, "last_skipped", [])]}
    _write(bundle, args.out)
    return 0
  try:
    ev = normalize(str(p), model_id=model_id, target_id=target_id)
  except AdapterIncomplete as e:
    _write({"schema": SCHEMA_INGEST_ERROR, "verdict": Verdict.ADAPTER_INCOMPLETE.value,
            "reason": e.reason, "path": args.path}, args.out)
    return 3
  except UnsupportedArtifact as e:
    # an unrecognized artifact is still "cannot be safely read yet" -> the adapter-incomplete verdict.
    _write({"schema": SCHEMA_INGEST_ERROR, "verdict": Verdict.ADAPTER_INCOMPLETE.value,
            "reason": e.reason, "path": args.path}, args.out)
    return 4
  _write(ev.to_json(), args.out)
  return 0


# ---- BB8: evaluate ---------------------------------------------------------------------------------------

def _register_evaluate(sub) -> None:
  p = sub.add_parser("evaluate", help="classify normalized evidence into a candidate decision")
  p.add_argument("--profile", required=True, help="model_profile.json")
  p.add_argument("--search", required=True, help="search_space.json")
  p.add_argument("--evidence", required=True, help="evidence.json (single or bundle)")
  p.add_argument("--candidate", default=None, help="evaluate only this candidate id (optional)")
  p.add_argument("--out", default=None, help="write decision JSON here instead of stdout")
  p.set_defaults(fn=cmd_evaluate)



def cmd_evaluate(args) -> int:
  prof = _load_json(args.profile)
  if not (isinstance(prof, dict) and prof.get("schema") == SCHEMA_MODEL_PROFILE):
    return _fail(f"evaluate: {args.profile!r} is not a {SCHEMA_MODEL_PROFILE} profile")
  space = _load_json(args.search)
  if not isinstance(space, dict):
    return _fail(f"evaluate: {args.search!r} is not a JSON object search space")
  evidences = _load_evidences(args.evidence)
  if not evidences:
    return _fail(f"evaluate: {args.evidence!r} carried no normalized evidence")
  model_id, target_id, workload = evidences[0].model_id, evidences[0].target_id, evidences[0].workload

  # profile/search/evidence must agree on model + target — a stale/wrong search file must be able to BLOCK a
  # promotion, not be silently ignored (the profile/search/evidence contract).
  pm, sm, st = prof.get("model_id"), space.get("model_id"), (space.get("target") or {}).get("target_id")
  for a, b, what in ((pm, sm, "profile/search model_id"), (model_id, sm, "evidence/search model_id"),
                     (pm, model_id, "profile/evidence model_id"), (target_id, st, "evidence/search target_id")):
    if a and b and a != b:
      return _fail(f"evaluate: {what} mismatch ({a!r} != {b!r}); refusing to evaluate an inconsistent contract")

  if args.candidate:
    cand = candidate_by_id(args.candidate)
    if cand is None:
      return _fail(f"evaluate: unknown candidate id {args.candidate!r}")
    if not candidate_in_search_space(cand, space):
      decision = CandidateDecision(candidate_id=cand.candidate_id, model_id=model_id, target_id=target_id,
                                   workload=cand.workload, verdict=Verdict.INCONCLUSIVE.value,
                                   reason=f"candidate {cand.candidate_id!r} is not authorized by the supplied search space",
                                   next_action="add its role/quant to the search space or evaluate a candidate it lists")
      _write(decision.to_json(), args.out)
      return 0
    decision = evaluate(cand, evidences, model_id=model_id, target_id=target_id, workload=cand.workload)
  else:
    allowed = candidates_in_search_space(load_candidates(), space)
    decisions = evaluate_all(evidences, model_id, target_id, workload, candidates=allowed)
    if not decisions:
      decision = CandidateDecision(candidate_id="none", model_id=model_id, target_id=target_id,
                                   workload=workload, verdict=Verdict.INCONCLUSIVE.value,
                                   reason="no candidate in the manifest matched this evidence",
                                   next_action="add a matching candidate or capture more evidence kinds")
    else:
      decision = primary_decision(decisions)
  _write(decision.to_json(), args.out)
  return 0


# ---- reopen-check: audit a specific (possibly refuted) candidate against fresh evidence ------------------

def _register_reopen_check(sub) -> None:
  p = sub.add_parser("reopen-check", help="audit ONE candidate (incl. refuted) against fresh evidence, bypassing "
                     "search-space authorization; a would-be promote is downgraded to candidate (audit, not authorization)")
  p.add_argument("--candidate", required=True, help="candidate id to audit (may be refuted)")
  p.add_argument("--evidence", required=True, help="evidence.json (single or bundle)")
  p.add_argument("--profile", default=None, help="optional model_profile.json to enforce model_id agreement")
  p.add_argument("--out", default=None, help="write decision JSON here instead of stdout")
  p.set_defaults(fn=cmd_reopen_check)



def cmd_reopen_check(args) -> int:
  """Audit ONE candidate -- INCLUDING a refuted one -- against fresh evidence, bypassing the search-space
  authorization gate that `evaluate` enforces. This answers the reopen question directly: 'given this new
  evidence, does this candidate clear the bar to reopen, or does the evidence confirm the refute?'

  `evaluate` intentionally returns INCONCLUSIVE ('not authorized by the search space') for a refuted candidate,
  because a refuted candidate is excluded from the emitted search space. That is correct for a promotion pass,
  but it makes the refuted candidate un-auditable against new measurements (the evidence gets mis-attributed to
  the nearest authorized candidate). reopen-check evaluates the named candidate itself so the guardrails/tier
  logic and its refuted_axis_tags apply to the right candidate.

  The promotion contract is preserved: a would-be PROMOTE is downgraded to CANDIDATE (reopen-eligible), because
  a reopen-check is an audit, not an authorization -- promotion still requires the authorized `evaluate` path."""
  cand = candidate_by_id(args.candidate)
  if cand is None:
    return _fail(f"reopen-check: unknown candidate id {args.candidate!r}")
  evidences = _load_evidences(args.evidence)
  if not evidences:
    return _fail(f"reopen-check: {args.evidence!r} carried no normalized evidence")
  model_id, target_id, workload = evidences[0].model_id, evidences[0].target_id, cand.workload
  # optional profile: enforce the model_id agreement contract so stale evidence cannot audit the wrong model
  if args.profile:
    prof = _load_json(args.profile)
    if not (isinstance(prof, dict) and prof.get("schema") == SCHEMA_MODEL_PROFILE):
      return _fail(f"reopen-check: {args.profile!r} is not a {SCHEMA_MODEL_PROFILE} profile")
    pm = prof.get("model_id")
    if pm and model_id and pm != model_id:
      return _fail(f"reopen-check: profile/evidence model_id mismatch ({pm!r} != {model_id!r})")
  dec = evaluate(cand, evidences, model_id=model_id, target_id=target_id, workload=workload)
  d = dec.to_json()
  d["reopen_check"] = True
  d["authorization"] = "unauthorized_audit"
  if dec.verdict == Verdict.PROMOTE.value:
    d["reopen_check_original_verdict"] = Verdict.PROMOTE.value
    d["verdict"] = Verdict.CANDIDATE.value
    d["next_action"] = ("evidence clears the promotion bar; add this candidate to the search space and run "
                        "`evaluate` to authorize promotion (reopen-check is an unauthorized audit)")
  _write(d, args.out)
  return 0


# ---- BB8: ledger add / report / import -------------------------------------------------------------------

def _register_ledger(sub) -> None:
  lg = sub.add_parser("ledger", help="durable route ledger operations")
  lgsub = lg.add_subparsers(dest="ledger_cmd", required=True)

  p = lgsub.add_parser("add", help="append a candidate decision to the ledger")
  p.add_argument("decision", help="decision.json")
  p.add_argument("--ledger", required=True, help="route_ledger.jsonl")
  p.add_argument("--do-not-retry", action="store_true", help="mark the entry as do_not_retry")
  p.add_argument("--out", default=None, help="write the stored entry JSON here instead of stdout")
  p.set_defaults(fn=cmd_ledger_add)

  p = lgsub.add_parser("report", help="render a deterministic markdown report from the ledger")
  p.add_argument("ledger", help="route_ledger.jsonl")
  p.add_argument("--out", default=None, help="write summary markdown here instead of stdout")
  p.set_defaults(fn=cmd_ledger_report)

  p = lgsub.add_parser("import", help="import a JSONL of LedgerEntry rows into the ledger")
  p.add_argument("source", help="seed JSONL of LedgerEntry rows")
  p.add_argument("--ledger", required=True, help="route_ledger.jsonl")
  p.set_defaults(fn=cmd_ledger_import)



def cmd_ledger_add(args) -> int:
  try:
    dec = CandidateDecision.from_json(_load_json(args.decision))
  except (ValueError, KeyError, json.JSONDecodeError) as e:
    return _fail(f"ledger add: {args.decision!r} is not a valid candidate decision: {e}")
  status = status_for_verdict(dec.verdict)
  if status is None:
    _ensure_ledger_file(args.ledger)
    sys.stderr.write(f"ledger add: decision {dec.candidate_id!r} verdict {dec.verdict!r} is not a durable "
                     f"ledger status; nothing appended.\n")
    return 0
  reopen = dec.next_action or dec.reason or "reopen when new evidence supersedes this decision"
  entry = LedgerStore(args.ledger).add(dec, reopen_condition=reopen, do_not_retry=args.do_not_retry)
  _write(entry.to_json(), args.out)
  return 0


def cmd_ledger_report(args) -> int:
  store = LedgerStore(args.ledger)
  try:
    entries = list(store.latest_entries().values())
  except (ValueError, KeyError, json.JSONDecodeError) as e:
    return _fail(f"ledger report: {args.ledger!r} contains a malformed ledger entry: {e}")
  text = render_report(entries)
  if args.out:
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(text)
  else:
    sys.stdout.write(text)
  return 0


def cmd_ledger_import(args) -> int:
  src = pathlib.Path(args.source)
  if not src.exists():
    return _fail(f"ledger import: no such file {args.source!r}")
  store = LedgerStore(args.ledger)
  n = 0
  for i, line in enumerate(src.read_text().splitlines(), 1):
    line = line.strip()
    if not line:
      continue
    try:
      entry = LedgerEntry.from_json(json.loads(line))
    except (ValueError, json.JSONDecodeError) as e:
      return _fail(f"ledger import: line {i} is not a valid LedgerEntry: {e}")
    store.add(entry)
    n += 1
  sys.stderr.write(f"ledger import: imported {n} entries into {args.ledger}\n")
  return 0


# ---- BB8: cache status -----------------------------------------------------------------------------------

def _register_cache_status(sub) -> None:
  cache = sub.add_parser("cache", help="artifact cache operations")
  cachesub = cache.add_subparsers(dest="cache_cmd", required=True)
  p = cachesub.add_parser("status", help="report cache entries, producers, model ids, and missing/stale paths")
  p.add_argument("--cache", default="cache.json", help="cache index JSON")
  p.add_argument("--out", default=None, help="write status JSON here instead of stdout")
  p.set_defaults(fn=cmd_cache_status)



def cmd_cache_status(args) -> int:
  cache = CacheStore(args.cache)
  if pathlib.Path(args.cache).exists():
    try:
      cache.load()
    except (ValueError, KeyError, json.JSONDecodeError) as e:
      return _fail(f"cache status: {args.cache!r} is not a valid artifact cache: {e}")
  _write(cache.status(), args.out)
  return 0


# ---- BB8: check-policy -----------------------------------------------------------------------------------

def _register_check_policy(sub) -> None:
  p = sub.add_parser("check-policy", help="check the route ledger for policy violations")
  p.add_argument("--ledger", required=True, help="route_ledger.jsonl")
  p.set_defaults(fn=cmd_check_policy)



def cmd_check_policy(args) -> int:
  store = LedgerStore(args.ledger)
  try:
    violations = check_policy(store, selected_candidate_ids=None)
  except (ValueError, KeyError, json.JSONDecodeError) as e:
    return _fail(f"check-policy: {args.ledger!r} contains a malformed ledger entry: {e}")
  if violations:
    sys.stderr.write(f"check-policy: {len(violations)} violation(s):\n")
    for v in violations:
      sys.stderr.write(f"  [{v.rule}] {v.candidate_id}: {v.detail}\n")
    return 1
  sys.stdout.write("check-policy: ok (no violations)\n")
  return 0

def _register_mem_sweep_request(sub) -> None:
  p = sub.add_parser("mem-sweep-request", help="emit a working-set sweep request for locating cache tiers")
  p.add_argument("--target", required=True, help="target id")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_mem_sweep_request)



def cmd_mem_sweep_request(args) -> int:
  from boltbeam.memory.mem_sweep import build_ws_sweep_request
  _write(build_ws_sweep_request(args.target), args.out)
  return 0

def _register_ingest_mem_sweep(sub) -> None:
  p = sub.add_parser("ingest-mem-sweep", help="classify a working-set sweep sample file into a memory hierarchy profile")
  p.add_argument("samples", help="boltbeam.ws_sweep_samples JSON from the runner")
  p.add_argument("--snapshot", required=True, help="system snapshot id")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_ingest_mem_sweep)



def cmd_ingest_mem_sweep(args) -> int:
  from boltbeam.memory.mem_hierarchy import classify_ws_sweep
  _write(classify_ws_sweep(_load_json(args.samples), system_snapshot_id=args.snapshot), args.out)
  return 0

def _register_lds_l2(sub) -> None:
  p = sub.add_parser("lds-l2", help="decide LDS vs L2 residency for an operand tile against a memory hierarchy profile")
  p.add_argument("--profile", required=True, help="memory_hierarchy_profile JSON")
  p.add_argument("--tile-bytes", type=float, required=True)
  p.add_argument("--traffic-bytes", type=float, required=True)
  p.add_argument("--reuse", type=float, required=True)
  p.add_argument("--role", default=None)
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_lds_l2)



def cmd_lds_l2(args) -> int:
  from boltbeam.memory.lds_l2 import lds_l2_for_operand
  _write(lds_l2_for_operand(_load_json(args.profile), tile_bytes=args.tile_bytes,
    traffic_bytes=args.traffic_bytes, reuse=args.reuse, role=args.role or ""), args.out)
  return 0


def register(sub) -> None:
  _register_inspect(sub)
  _register_emit_search(sub)
  _register_vocab(sub)
  _register_vocab_capture(sub)
  _register_decode_role_profile(sub)
  _register_boundary_plan(sub)
  _register_route_manifest(sub)
  _register_emit_policy(sub)
  _register_synth(sub)
  _register_load(sub)
  _register_autoscan(sub)
  _register_analyze(sub)
  _register_output(sub)
  _register_ingest_probe(sub)
  _register_ingest_timing(sub)
  _register_runner_plan(sub)
  _register_matched_control(sub)
  _register_replay_ab(sub)
  _register_ingest(sub)
  _register_evaluate(sub)
  _register_reopen_check(sub)
  _register_ledger(sub)
  _register_cache_status(sub)
  _register_check_policy(sub)
  _register_mem_sweep_request(sub)
  _register_ingest_mem_sweep(sub)
  _register_lds_l2(sub)
