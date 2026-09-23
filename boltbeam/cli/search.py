from __future__ import annotations

import json
import pathlib
import sys

from boltbeam.cli._common import _write, _fail, _load_json
from boltbeam.full_kernel_candidate_set import (build_qwen3_8b_buffer2_candidate_set,
  candidate_from_proven_document, deterministic_candidate_set_json)
from boltbeam.search.full_kernel.full_kernel_search import (deterministic_json as full_kernel_search_json,
  run_full_kernel_search, subprocess_worker)

def _register_mr13_closure(sub) -> None:
  p = sub.add_parser("mr13-closure", help="write a fresh, hashed MR13 closure manifest (no measurements)")
  p.add_argument("--artifact", action="append", nargs=4, metavar=("NAME", "PATH", "SCHEMA", "STATUS"), required=True,
                 help="required named JSON packet and accepted comma-separated statuses")
  p.add_argument("--mr9-disposition", choices=("winner", "refuted"), required=True)
  p.add_argument("--omit", action="append", default=[], metavar="NAME=REASON", help="required for each winner-only packet on refutation")
  p.add_argument("--out", required=True, help="new destination; existing outputs are refused")
  p.set_defaults(fn=cmd_mr13_closure)



def cmd_mr13_closure(args) -> int:
  from boltbeam.artifacts.mr13_closure import PacketRequirement, build, write_fresh
  rows = [PacketRequirement(name, pathlib.Path(path), schema, tuple(status.split(",")))
          for name, path, schema, status in args.artifact]
  omissions = dict(item.split("=", 1) for item in args.omit)
  manifest = build(rows, mr9_winner=args.mr9_disposition == "winner", omissions=omissions or None)
  write_fresh(manifest, pathlib.Path(args.out))
  return 0

def _register_mr12_static_audit(sub) -> None:
  p = sub.add_parser("mr12-static-audit", help="emit the non-hardware MR12 duplicate-policy and clean-revision packet")
  p.add_argument("--tinygrad-root", required=True, help="clean tinygrad EXP checkout containing the generated policy snapshot")
  p.add_argument("--out", required=True, help="new destination; existing outputs are refused")
  p.set_defaults(fn=cmd_mr12_static_audit)



def cmd_mr12_static_audit(args) -> int:
  from boltbeam.artifacts.mr12_static_audit import build, write_fresh
  packet = build(boltbeam_root=pathlib.Path(__file__).resolve().parents[1], tinygrad_root=pathlib.Path(args.tinygrad_root))
  write_fresh(packet, pathlib.Path(args.out))
  return 0 if packet["status"] == "pass" else 1

def _register_build_full_kernel_candidate_set(sub) -> None:
  p = sub.add_parser("build-full-kernel-candidate-set",
    help="persist exact qwen3-8B Tinygrad candidates from a proven gate/up schedule")
  p.add_argument("candidate", help="proven gate/up candidate JSON or BoltBeam candidate manifest")
  p.add_argument("--target-id", default="amd_gfx1100",
    help="registry target to stamp into workload/applicability (default amd_gfx1100)")
  p.add_argument("--schedule-template", default=None,
    help="tinygrad-derived typed schedule JSON carrying schedule + static_constraints; "
         "the builder stops cloning the seed's schedule when given")
  p.add_argument("--out", required=True, help="write deterministic candidate-set JSON here")
  p.set_defaults(fn=cmd_build_full_kernel_candidate_set)



def cmd_build_full_kernel_candidate_set(args) -> int:
  """Persist exact 8B role candidates derived from one admitted gate/up schedule."""
  try:
    seed = candidate_from_proven_document(_load_json(args.candidate))
    template = _load_json(args.schedule_template) if args.schedule_template else None
    result = build_qwen3_8b_buffer2_candidate_set(seed, target_id=args.target_id,
                                                  schedule_template=template)
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(deterministic_candidate_set_json(result))
  except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
    return _fail(f"build-full-kernel-candidate-set: {exc}")
  return 0

def _register_search_full_kernel(sub) -> None:
  p = sub.add_parser("search-full-kernel", help="exhaustively measure an explicit finite full-kernel candidate space")
  p.add_argument("request", help="public full-kernel search request JSON")
  p.add_argument("--worker-command", required=True, help="JSON argv for the isolated tinygrad.search_provider.v1 entrypoint")
  p.add_argument("--tinygrad-root", default=None, help="optional worker cwd; BoltBeam never modifies it")
  p.add_argument("--timeout", type=float, default=60.0, help="per-candidate worker deadline in seconds")
  p.add_argument("--out", default=None, help="write complete deterministic result JSON here")
  p.set_defaults(fn=cmd_search_full_kernel)



def cmd_search_full_kernel(args) -> int:
  """Run an explicitly finite full-kernel population through an external worker."""
  try:
    request = _load_json(args.request)
    command = json.loads(args.worker_command)
    if not isinstance(command, list): raise ValueError("--worker-command must be a JSON argv array")
    result = run_full_kernel_search(request, subprocess_worker(command, cwd=args.tinygrad_root, timeout_s=args.timeout,
                                    resolved_target=request.get("resolved_target"),
                                    requested_provider_revision=request.get("requested_provider_revision"),
                                    execution=request.get("execution")),
                                    timeout_s=args.timeout)
    text = full_kernel_search_json(result)
    if args.out:
      output = pathlib.Path(args.out); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(text)
    else: sys.stdout.write(text)
  except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
    return _fail(f"search-full-kernel: {exc}")
  # BLOCKED is a valid, evidence-preserving result rather than a fabricated success.
  return 0

def _register_semantic_campaign(sub) -> None:
  p = sub.add_parser("semantic-campaign", help="run a finite exact semantic GGUF campaign request")
  p.add_argument("request", help="serialized semantic campaign request JSON")
  p.add_argument("--provider-command", required=True, help="JSON argv for persistent tinygrad provider")
  p.add_argument("--futuresight-evidence", default=None, help="optional static evidence JSON from assess-semantic-population")
  p.add_argument("--out", required=True, help="write canonical campaign ledger JSON")
  p.set_defaults(fn=cmd_semantic_campaign)



def cmd_semantic_campaign(args) -> int:
  from boltbeam.search.semantic_campaign_cli import run_request
  request = _load_json(args.request)
  evidence = _load_json(args.futuresight_evidence) if args.futuresight_evidence else None
  result = run_request(request, provider_command=json.loads(args.provider_command), futuresight_evidence=evidence)
  pathlib.Path(args.out).write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
  return 0

def _register_export_semantic_population(sub) -> None:
  p = sub.add_parser("export-semantic-population", help="materialize canonical semantic candidate/row IDs without measurement")
  p.add_argument("request", help="serialized semantic campaign request JSON")
  p.add_argument("--out", required=True, help="write canonical population JSON")
  p.set_defaults(fn=cmd_export_semantic_population)



def cmd_export_semantic_population(args) -> int:
  from boltbeam.search.semantic.semantic_population_export import export_population
  result = export_population(_load_json(args.request))
  pathlib.Path(args.out).write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
  return 0

def _register_propose_semantic_dimensions(sub) -> None:
  p = sub.add_parser("propose-semantic-dimensions",
    help="BubbleBeam: propose legal dimensions and coupled rows from target facts; writes a semantic-campaign request")
  p.add_argument("spec", help="proposal spec JSON: semantic workload, baseline schedule, axis_choices, coupled_rows, campaign fields")
  p.add_argument("--describe", default=None,
    help="provider describe result JSON; its target must reproduce the spec's resolved_target, "
         "and it adds compiler_transforms and supported_plan_kinds to the target facts")
  p.add_argument("--out", required=True, help="write the semantic campaign request JSON")
  p.set_defaults(fn=cmd_propose_semantic_dimensions)



def cmd_propose_semantic_dimensions(args) -> int:
  from boltbeam.search.futuresight_adapter import propose_request
  try:
    request, missing = propose_request(_load_json(args.spec), _load_json(args.describe) if args.describe else None)
    _write(request, args.out)
  except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
    return _fail(f"propose-semantic-dimensions: {exc}")
  if missing: sys.stderr.write(f"propose-semantic-dimensions: target facts missing from registry and provider: {', '.join(missing)}\n")
  return 0

def _register_assess_semantic_population(sub) -> None:
  p = sub.add_parser("assess-semantic-population",
    help="FutureSight: statically reject and order an exported population; writes semantic-campaign --futuresight-evidence")
  p.add_argument("population", help="export-semantic-population JSON")
  p.add_argument("--preferences", default=None, help="JSON object: candidate field path -> preferred values, best first")
  p.add_argument("--out", required=True, help="write the FutureSight evidence JSON")
  p.set_defaults(fn=cmd_assess_semantic_population)



def cmd_assess_semantic_population(args) -> int:
  from boltbeam.search.futuresight_adapter import assess_population
  try:
    evidence = assess_population(_load_json(args.population), _load_json(args.preferences) if args.preferences else None)
    _write(evidence, args.out)
  except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
    return _fail(f"assess-semantic-population: {exc}")
  return 0

def _register_mr8_population_selection(sub) -> None:
  p = sub.add_parser("mr8-population-selection", help="join a complete MR7 ranking to exact semantic populations")
  p.add_argument("ranking", help="complete MR7 role-cost ranking JSON")
  p.add_argument("request", help="MR8 population selection request JSON")
  p.add_argument("--out", required=True, help="write the verified MR8 selection handoff JSON")
  p.set_defaults(fn=cmd_mr8_population_selection)



def cmd_mr8_population_selection(args) -> int:
  from boltbeam.search.semantic.mr8_population_selection import build_mr8_population_selection
  ranking = _load_json(args.ranking)
  request = _load_json(args.request)
  result = build_mr8_population_selection(ranking, request)
  output = pathlib.Path(args.out)
  if output.exists(): raise FileExistsError(f"MR8 population selection output already exists: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
  return 0

def _register_mr7_evidence_plan(sub) -> None:
  p = sub.add_parser("mr7-evidence-plan", help="derive exact isolated/outer provider requests from MR4 and MR5")
  p.add_argument("mr4_summary", help="complete MR4 replay summary JSON")
  p.add_argument("census", help="finalized MR5 graph-admission census JSON")
  p.add_argument("--arm", choices=("control", "hybrid"), required=True)
  p.add_argument("--out", required=True, help="write immutable provider plan JSON")
  p.set_defaults(fn=cmd_mr7_evidence_plan)



def cmd_mr7_evidence_plan(args) -> int:
  from boltbeam.search.semantic.mr7_evidence_bundle import build_evidence_plan
  output = pathlib.Path(args.out)
  if output.exists(): raise FileExistsError(f"MR7 evidence plan already exists: {output}")
  plan = build_evidence_plan(_load_json(args.mr4_summary), _load_json(args.census), arm=args.arm)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
  return 0

def _register_mr7_evidence_bundle(sub) -> None:
  p = sub.add_parser("mr7-evidence-bundle", help="bind exact provider results into ranker-ready MR7 inputs")
  p.add_argument("plan"); p.add_argument("provider_results")
  p.add_argument("--mr4-summary", required=True); p.add_argument("--census", required=True)
  p.add_argument("--mr0", required=True); p.add_argument("--mr6", required=True)
  p.add_argument("--out-dir", required=True)
  p.set_defaults(fn=cmd_mr7_evidence_bundle)



def cmd_mr7_evidence_bundle(args) -> int:
  from boltbeam.search.semantic.mr7_evidence_bundle import write_evidence_bundle
  write_evidence_bundle(pathlib.Path(args.out_dir), mr4_summary=_load_json(args.mr4_summary),
    census=_load_json(args.census), mr0=_load_json(args.mr0), mr6=_load_json(args.mr6),
    plan=_load_json(args.plan), provider_results=_load_json(args.provider_results))
  return 0

def _register_mr7_provider_run(sub) -> None:
  p = sub.add_parser("mr7-provider-run", help="execute every immutable MR7 provider request with exact JSON binding")
  p.add_argument("plan"); p.add_argument("--provider-command", required=True, help="JSON argv for one-request MR7 provider")
  p.add_argument("--timeout", type=float, default=60.0); p.add_argument("--out", required=True)
  p.set_defaults(fn=cmd_mr7_provider_run)



def cmd_mr7_provider_run(args) -> int:
  from boltbeam.search.semantic.mr7_evidence_bundle import collect_provider_results
  output = pathlib.Path(args.out)
  if output.exists(): raise FileExistsError(f"MR7 provider results already exist: {output}")
  command = json.loads(args.provider_command)
  if not isinstance(command, list) or not command: raise ValueError("--provider-command must be a JSON argv array")
  results = collect_provider_results(_load_json(args.plan), command, timeout_s=args.timeout)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
  return 0

def _register_mr7_evidence_run(sub) -> None:
  p = sub.add_parser("mr7-evidence-run", help="plan, run exact provider requests, and emit a fresh MR7 bundle")
  p.add_argument("mr4_summary"); p.add_argument("census")
  p.add_argument("--mr0", required=True); p.add_argument("--mr6", required=True)
  p.add_argument("--arm", choices=("control", "hybrid"), required=True)
  p.add_argument("--provider-command", required=True, help="JSON argv for one-request MR7 provider")
  p.add_argument("--timeout", type=float, default=60.0); p.add_argument("--out-dir", required=True)
  p.set_defaults(fn=cmd_mr7_evidence_run)



def cmd_mr7_evidence_run(args) -> int:
  from boltbeam.search.semantic.mr7_evidence_bundle import build_evidence_plan, collect_provider_results, write_evidence_bundle
  command = json.loads(args.provider_command)
  if not isinstance(command, list) or not command: raise ValueError("--provider-command must be a JSON argv array")
  output_dir = pathlib.Path(args.out_dir)
  if output_dir.exists(): raise FileExistsError(f"MR7 evidence output already exists: {output_dir}")
  mr4, census = _load_json(args.mr4_summary), _load_json(args.census)
  plan = build_evidence_plan(mr4, census, arm=args.arm)
  results = collect_provider_results(plan, command, timeout_s=args.timeout)
  write_evidence_bundle(output_dir, mr4_summary=mr4, census=census,
    mr0=_load_json(args.mr0), mr6=_load_json(args.mr6), plan=plan, provider_results=results)
  return 0

def _register_mr9_semantic_search(sub) -> None:
  p = sub.add_parser("mr9-semantic-search", help="execute complete MR8 populations and repeat bounded finalists")
  p.add_argument("selection"); p.add_argument("--provider-command", required=True,
    help="JSON argv for persistent tinygrad.search_provider.v1")
  p.add_argument("--provider-revision", required=True, help="clean pinned Tinygrad git revision")
  p.add_argument("--boltbeam-revision", required=True, help="clean pinned BoltBeam git revision")
  p.add_argument("--finalist-repeats", type=int, default=2); p.add_argument("--minimum-win", type=float, default=0.03)
  p.add_argument("--maximum-relative-mad", type=float, default=0.05); p.add_argument("--out", required=True)
  p.set_defaults(fn=cmd_mr9_semantic_search)



def cmd_mr9_semantic_search(args) -> int:
  from boltbeam.search.semantic.mr9_semantic_search import run_mr9_semantic_search
  output = pathlib.Path(args.out)
  if output.exists(): raise FileExistsError(f"MR9 output already exists: {output}")
  command = json.loads(args.provider_command)
  result = run_mr9_semantic_search(_load_json(args.selection), provider_command=command,
    finalist_repeats=args.finalist_repeats, minimum_win_fraction=args.minimum_win,
    maximum_relative_mad=args.maximum_relative_mad, requested_provider_revision=args.provider_revision,
    requested_boltbeam_revision=args.boltbeam_revision)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
  return 0 if result["status"] == "COMPLETE" else 2



# ---- BB8: ingest -----------------------------------------------------------------------------------------

def _register_artifact_cache_inventory(sub) -> None:
  p = sub.add_parser("artifact-cache-inventory", help="classify tinygrad search artifacts into cache reuse classes")
  p.add_argument("root", help="tinygrad checkout root")
  p.add_argument("--out", default=None, help="write inventory JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="optional markdown report path")
  p.set_defaults(fn=cmd_artifact_cache_inventory)



def cmd_artifact_cache_inventory(args) -> int:
  from boltbeam.artifacts.cache_inventory import build_inventory, inventory_markdown
  report = build_inventory(args.root)
  _write(report, args.out)
  if args.markdown:
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(inventory_markdown(report))
  return 0


def register(sub) -> None:
  _register_mr13_closure(sub)
  _register_mr12_static_audit(sub)
  _register_build_full_kernel_candidate_set(sub)
  _register_search_full_kernel(sub)
  _register_semantic_campaign(sub)
  _register_propose_semantic_dimensions(sub)
  _register_export_semantic_population(sub)
  _register_assess_semantic_population(sub)
  _register_mr8_population_selection(sub)
  _register_mr7_evidence_plan(sub)
  _register_mr7_evidence_bundle(sub)
  _register_mr7_provider_run(sub)
  _register_mr7_evidence_run(sub)
  _register_mr9_semantic_search(sub)
  _register_artifact_cache_inventory(sub)
