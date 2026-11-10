"""The fusions: kernels that replace several adjacent decode-graph nodes with one launch, proposed by BubbleBeam from
the provider's fused emitters and BoltBeam's own decode graph (search/fusion_space.py), run through the same campaign
as the roles and checked the same way (role_check.Checker.fusion), each judged on the in-model time of the tie-out rows
it replaces and on BoltBeam's unfused side timed the same way. Writes kernel_compare/fusions.json.
"""
from __future__ import annotations

import pathlib
from collections.abc import Callable, Mapping
from typing import Any

from boltbeam.plan.resolved_target import candidate_target, resolved_target_document
from boltbeam.search import compare_record, provider_check, role_check, role_numbers, role_proposal, role_space

FUSIONS_SCHEMA = "boltbeam.kernel_fusions.v1"


def _fusion_stem(proposal:Mapping[str, Any]) -> str:
  return f"fusion-{proposal['name'].replace('+', '-')}-{str(proposal['quant']).lower()}"


def fusion_verdict(proposal:Mapping[str, Any], check:Mapping[str, Any] | None, losses:Mapping[tuple[str, str], Mapping[str, Any]],
                   band:float) -> dict[str, Any]:
  """One fusion's numbers and verdict. The bar is the in-model time of the tie-out rows it replaces (the "where the
  token goes" rows), per token; BoltBeam's fused kernel alone (less the floor) times the launches per token must beat it
  by more than the chip's band. It must also beat BoltBeam's unfused side timed the same way (each covered node's
  kernel alone, less the floor; a lower bound when a node's reading was not reproduced): an isolated time against an
  in-model one alone can show a gain that is only the difference between the two ways of timing. A covered node with
  no row of its own counts 0 ms in the model (fusion_space.NOT_IN_MODEL)."""
  from boltbeam.search import fusion_space
  inst = proposal.get("instance") or {}
  rows = [losses.get(tuple(r)) for r in inst.get("tie_out_rows") or []]
  calls = inst.get("calls_per_token")
  out: dict[str, Any] = {"in_model_ms": None, "lost_ms": None, "fused": None, "unfused": None}
  if rows and all(rows):
    out["in_model_ms"] = sum(r["actual_ms"] for r in rows)
    out["lost_ms"] = sum(r["lost_ms"] for r in rows)
    out["in_model_rows"] = [{"row": f"{r['role']} {r['quant']}", "ms": r["actual_ms"], "lost_ms": r["lost_ms"],
                             "calls_per_token": r["calls_per_token"]} for r in rows]
  weights = set(proposal.get("weights") or [])
  zero = [n for n in proposal.get("covers") or [] if n not in weights]
  if zero:
    out["in_model_note"] = f"{', '.join(zero)} {fusion_space.NOT_IN_MODEL}"
  if check is None or check.get("error"):
    return out | {"verdict": "not_reproduced" if check else "not_searched",
                  "reason": (check or {}).get("error") or "the search measured no fused candidate"}
  best = role_check.pick_winner(check.get("candidates") or [])
  if best is None or best.get("verdict") != provider_check.REPRODUCED:
    why = "; ".join(sorted({str(c.get("reason")) for c in check.get("candidates") or []})) or "the search measured no fused candidate"
    return out | {"verdict": "not_reproduced", "reason": f"{provider_check.NOT_REPRODUCED}: {why}"}
  fused_ms = best["boltbeam_us_less_floor"] * calls / 1000.0
  out["fused"] = {"plan": best["plan"], "candidate_hash": best["candidate_hash"], "us_less_floor": best["boltbeam_us_less_floor"],
                  "ms_per_token": fused_ms, "correctness": (best.get("boltbeam") or {}).get("correctness"),
                  "provider_note": best.get("provider_note")}
  alone = check.get("unfused") or []
  good_alone = [u for u in alone if u.get("verdict") == provider_check.REPRODUCED]
  every = bool(alone) and len(good_alone) == len(alone) and {u["node"] for u in alone} == set(proposal.get("covers") or [])
  # the unfused side alone: its sum when BoltBeam reproduced every node; the reproduced nodes' sum is a lower bound
  # otherwise (a node's time is never below 0), and like for like a fused kernel must beat even that
  lower = sum(u["boltbeam_us_less_floor"] for u in good_alone) * calls / 1000.0 if good_alone else None
  out["unfused"] = {"nodes": [{"node": u.get("node"), "verdict": u.get("verdict"), "us_less_floor": u.get("boltbeam_us_less_floor"),
                               "reason": u.get("reason")} for u in alone],
                    "ms_per_token": lower if every else None, "lower_bound_ms": lower}
  if not every:
    out["unfused"]["why"] = ("BoltBeam did not reproduce every covered node's kernel alone; the ones it did sum to a lower bound"
                             if alone else "the provider returned no unfused kernels")
  if not every:
    return out | {"verdict": "not_reproduced", "reason": "not decided: " + out["unfused"]["why"]}
  if check.get("unfused_model_bound") is not True:
    return out | {"verdict": "not_reproduced", "reason": "not decided: unfused kernels are not identified as model-bound"}
  if out["in_model_ms"] is None:
    return out | {"verdict": "not_reproduced", "reason": "the tie-out has no in-model time for the rows this fusion replaces"}
  if lower is not None and fused_ms >= lower * (1.0 - band):
    return out | {"verdict": "none_faster", "decided_by": "kernel alone",
                  "reason": (f"decided by the kernel alone: BoltBeam's fastest fused kernel ({best['plan']}) takes {fused_ms:.3f} ms per token, "
                             f"not faster than the nodes it replaces timed alone the same way ({lower:.3f} ms"
                             f"{'' if every else ', a lower bound'}); the model's {out['in_model_ms']:.3f} ms in the model is not a like-for-like bar")}
  if fused_ms < out["in_model_ms"] * (1.0 - band):
    return out | {"verdict": "found_not_applied", "decided_by": "binding",
                  "reason": (f"decided by binding: BoltBeam's fused kernel ({best['plan']}) takes {fused_ms:.3f} ms per token against "
                             f"the model's {out['in_model_ms']:.3f} ms for these rows; the whole-model A/B does not install a fused emitter yet")}
  return out | {"verdict": "none_faster", "decided_by": "kernel alone",
                "reason": (f"decided by the kernel alone: BoltBeam's fastest fused kernel ({best['plan']}) takes {fused_ms:.3f} ms per token "
                           f"against the model's {out['in_model_ms']:.3f} ms for these rows (the chip's band is ±{100 * band:.1f}%)")}


def compare_fusions(run:pathlib.Path, *, facts:Mapping[str, Any], describe_shapes:Callable[[list[dict[str, int]]], list[dict[str, Any]]],
                    search:Callable[..., dict[str, Any]], checker:Any, model:str, model_sha:str, target_id:str,
                    losses:Mapping[tuple[str, str], Mapping[str, Any]], timestamp:str, say:Callable[[str], None],
                    tensors:list[tuple[str, tuple[int, ...], int, int]] | None = None) -> dict[str, Any]:
  """BubbleBeam proposes the fusions (fusion_space.propose), the population is exported, FutureSight assesses it, the
  campaign measures it, BoltBeam checks and times both sides, and fusion_verdict judges. Writes
  kernel_compare/fusion-<nodes>-<quant>-{search-request,population,search-result,check}.json and fusions.json."""
  from boltbeam.search import fusion_space
  from boltbeam.search.futuresight_adapter import assess_population, propose_request, target_facts
  from boltbeam.search.semantic.semantic_population_export import export_population
  folder = run / compare_record.FOLDER
  families = [f for f in facts.get("decode_fusions") or [] if isinstance(f, Mapping)]
  record: dict[str, Any] = {"schema": FUSIONS_SCHEMA, "run": run.name, "timing_source": compare_record.TIMING_SOURCE, "fusions": []}
  if not families:
    record["note"] = "the provider describes no fused emitter on this device"
    compare_record.write(folder / compare_record.FUSIONS_FILE, record)
    return record
  if tensors is None:
    from boltbeam.profile.gguf import read_gguf_split
    tensors = read_gguf_split(pathlib.Path(model))[1]
  by_layer = fusion_space.layers(tensors)
  first = fusion_space.propose(families, by_layer, set(losses))
  shapes = sorted({tuple(p["instance"]["shape"]) for p in first if p.get("instance")})
  verdicts = describe_shapes([{"n": n, "k": k} for n, k in shapes]) if shapes else []
  proposals = fusion_space.propose(families, by_layer, set(losses), verdicts)
  resolved = resolved_target_document(target_id, facts["target"])
  target = candidate_target(resolved)
  chip, _missing = target_facts(target_id, facts)
  say(f"compare fusions: {len(proposals)} proposed by the provider's fused emitters")
  for proposal in proposals:
    stem = _fusion_stem(proposal)
    row: dict[str, Any] = {"name": proposal["name"], "family": proposal["family"], "covers": proposal["covers"], "quant": proposal["quant"],
                           "instance": proposal.get("instance"), "proposed_rows": len(proposal.get("rows") or []),
                           "rejected_rows": proposal.get("rejected_rows") or [], "stem": stem}
    if proposal.get("why"):
      row.update(verdict="rejected", reason=f"rejected by BubbleBeam: {proposal['why']}", decided_by="proposal")
      record["fusions"].append(row | fusion_verdict(proposal, None, losses, checker.band) | {"verdict": "rejected", "reason": row["reason"]})
      say(f"fusion {proposal['name']} {proposal['quant']}: rejected ({proposal['why']})")
      continue
    say(f"fusion {proposal['name']} {proposal['quant']}: search")
    work = fusion_space.workload(proposal, model_sha=model_sha, target=target, tolerance=role_proposal.tolerances())
    base = {"schedule.plan_kind": role_space.EMITTER_PLAN_KIND, "schedule.compute.family": proposal["family"]}
    spec = {"semantic_workload": work, "schedule": dict(role_space.SEED_SCHEDULE), "resolved_target": resolved, "gguf_path": str(model),
            "execution": {"shape_mode": "exact_workload", "warmups": 2, "samples": 7},
            "request_id": f"{run.name}-{stem}", "run_id": f"{run.name}-compare", "timestamp": timestamp,
            "budget": {"max_candidates": len(proposal["rows"]) + 2, "timeout_s": 600.0},
            "axis_choices": {}, "coupled_rows": [{**base, **r} for r in proposal["rows"]]}
    try:
      request, _missing = propose_request(spec, facts)                                     # 1. BubbleBeam
      request["rejected_coupled_rows"] = list(request["rejected_coupled_rows"]) + [
        {"row": {**base, **r["row"]}, "reason": r["reason"]} for r in proposal.get("rejected_rows") or []]
      population = export_population(request)                                              # 2. the population
      evidence = assess_population(population)                                             # 3. FutureSight
    except Exception as exc:  # no space for this fusion: say why, the others go on
      row.update(verdict="not_searched", reason=f"no space: {exc}")
      record["fusions"].append(row); continue
    compare_record.write(folder / f"{stem}-search-request.json", request | {"futuresight_evidence": evidence})
    compare_record.write(folder / f"{stem}-population.json", population)
    row["futuresight_rejected"] = [{"candidate_hash": r.get("candidate_hash"), "reason": r.get("reason")}
                                   for r in evidence.get("rejections") or [] if "candidate_hash" in r]
    try:
      result = search(request, evidence)                                                   # 4. the campaign
    except Exception as exc:
      result = {"status": "BLOCKED", "counts": {}, "population": [], "error": str(exc), "futuresight_static_evidence": evidence}
    compare_record.write(folder / f"{stem}-search-result.json", result)
    row["search"] = {"status": result.get("status"), "counts": dict(result.get("counts") or {}), "error": result.get("error"),
                     "refused": [{"candidate_hash": r.get("candidate_hash"), "state": r.get("state"), "reason": r.get("reason"),
                                  "plan": role_numbers.plan_text((r.get("candidate") or {}).get("schedule") or {})}
                                 for r in result.get("population") or [] if r.get("state") != "MEASURED"]}
    check = None
    if any(r.get("state") == "MEASURED" for r in result.get("population") or []):
      say(f"fusion {proposal['name']} {proposal['quant']}: check")
      try:
        check = checker.fusion(result, model=model, proposal=proposal)
      except Exception as exc:  # BoltBeam could not run its own check: nothing the provider says is taken
        check = {"error": f"{type(exc).__name__}: {exc}"[:400]}
      compare_record.write(folder / f"{stem}-check.json", check)
      row["check_result"] = f"{compare_record.FOLDER}/{stem}-check.json"
    judged = fusion_verdict(proposal, check, losses, checker.band)
    if check is None:
      judged.update(verdict="not_reproduced" if result.get("population") else "not_searched",
                    reason=f"the provider measured no fused candidate: {result.get('error') or 'every candidate was refused'}")
    row.update(judged, search_result=f"{compare_record.FOLDER}/{stem}-search-result.json")
    record["fusions"].append(row)
    say(f"fusion {proposal['name']} {proposal['quant']}: done {row['verdict']}")
  compare_record.write(folder / compare_record.FUSIONS_FILE, record)
  return record
