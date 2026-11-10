"""Compare kernels per role on any tinygrad GPU: search, BoltBeam's own check, matched whole-model A/B, decision.

For every role in a run's route_policy.json this module
  1. asks the tinygrad fork's search provider which devices it can run (its capability action) and picks the first
     of the run's backend's devices that opened (boltbeam/data/search_runtime.json, a table, not code),
  2. checks the provider's device identity against BoltBeam's own reading of the GPU (search/provider_check.py),
  3. builds the role's finite space from what the model's decode binds through on that device (search/role_space.py:
     the fork's decode emitters, or Opt sequences on tinygrad's scheduler), with BubbleBeam's proposal and
     FutureSight's static rejections recorded,
  4. runs it through the provider (`search-full-kernel`'s own runner),
  5. rebuilds every candidate the provider measured from the source it returned, on the model's weight bytes, checks
     it against BoltBeam's reference and times it with BoltBeam's kernel timer: BoltBeam's own time decides, the
     provider's only ordered the search; a kernel BoltBeam cannot rebuild, finds incorrect or reads noisily is "not
     reproduced by BoltBeam" and the role is never promoted,
  6. times the winner against the default in a matched whole-model decode A/B in tinygrad's runtime,
  7. turns that into a CandidateDecision with the existing evaluator, appends it to the run's route ledger, and
     writes the route's status into route_policy.json.

Then the fusions (compare_fusions): kernels that replace several adjacent decode-graph nodes with one launch, proposed
by BubbleBeam from the provider's fused emitters and BoltBeam's own decode graph (search/fusion_space.py), run through
the same campaign and checked the same way, each judged on the in-model time of the tie-out rows it replaces.

The provider's numbers are claims. The kernel numbers this module judges by are BoltBeam's own (the kernel timer);
the whole-model numbers are tinygrad runtime times. None of them is a llama.cpp time.

This module is the per-role comparison itself (compare_run) and what it writes into route_policy.json. Each step
above has its own module: the fork and its provider (search/tinygrad_fork.py: steps 1 and 4's process), the role's
proposal (search/role_proposal.py: step 3), the numbers a role is judged by (search/role_numbers.py), BoltBeam's own
check (search/role_check.py: step 5), the A/B and the decision (search/role_decision.py: steps 6 and 7), the fusions
and flash decode (search/fusion_compare.py, search/flash_compare.py), the files a run holds (search/compare_record.py)
and the search as a stage of Run with the per-role verdict the screens show (search/search_stage.py).
"""
from __future__ import annotations

import pathlib
from collections.abc import Callable, Mapping
from typing import Any

from boltbeam.role_key import role_key, role_label, role_stem
from boltbeam.ledger.model import CandidateDecision, status_for_verdict
from boltbeam.ledger.store import LedgerStore
from boltbeam.search import (compare_record, flash_compare, fusion_compare, provider_check, role_check, role_decision, role_numbers,
                             role_proposal, tinygrad_fork)
from boltbeam.vocab import Verdict

SCHEMA = "boltbeam.kernel_compare.v2"
REOPEN = "reopen when the plan binds by model-graph role identity or the A/B is re-measured at a new revision"


def _calls(row:Mapping[str, Any] | None) -> str:
  return f"{row['calls_per_token']:.0f}" if row else "unknown number of"


class ProviderRefused(RuntimeError):
  """The provider's device is not the GPU BoltBeam reads, or the provider runs on none of this machine's devices."""



def compare_run(run:pathlib.Path, *, root:pathlib.Path | None = None, say:Callable[[str], None] = print,
                search:Callable[..., dict[str, Any]] | None = None, ab_run:Callable[..., dict[str, Any]] | None = None,
                describe_fn:Callable[..., dict[str, Any]] | None = None,
                capability_fn:Callable[[pathlib.Path, list[str]], dict[str, Any]] | None = None,
                checker:Any = None, only:set[str] | None = None, ab_only_if_faster:bool = False,
                step:Callable[[int, int], None] | None = None, fusions:bool = True,
                tensors_of_model:list[tuple[str, tuple[int, ...], int, int]] | None = None,
                flash_fn:Callable[..., dict[str, Any]] | bool | None = None) -> dict[str, Any]:
  """Compare kernels for every route of one run and write the outcome into the run folder.

  `search`, `ab_run`, `describe_fn`, `capability_fn` and `checker` default to the real provider, driver and BoltBeam
  bridge; tests pass doubles (and `tensors_of_model`, the GGUF's tensor list, for the fusions). fusions=False skips
  the fusion stage. ab_only_if_faster skips the whole-model A/B for a role whose best plan alone is not
  faster than the model's own kernel per call (Run's search stage). step(done, total) is called after each role.
  """
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest, read_json, write_json
  root = pathlib.Path(root) if root else tinygrad_fork.default_fork_root()
  manifest = load_manifest(run)
  model = str(manifest.get("model_path") or "")
  ready = tinygrad_fork.readiness(root, model) if search is None else {"ready": True}
  if not ready["ready"]:
    raise RuntimeError(f"{ready['message']} Fix: {ready['fix']}" if ready.get("fix") else ready["message"])
  if search is None and (why := tinygrad_fork.clean_refusal(root)):
    raise RuntimeError(why)
  target_id, model_id = str(manifest.get("target_id")), str(manifest.get("model_id"))
  backend = get_target(target_id).backend
  # the device: the first of this backend's devices the provider says it opened
  row = tinygrad_fork.runtime_row(backend)
  reply = (capability_fn or tinygrad_fork.capability)(root, list((row or {}).get("tinygrad_devices") or []))
  device, why = tinygrad_fork.pick_device(backend, reply)
  if device is None:
    raise ProviderRefused(why)
  facts = (describe_fn or tinygrad_fork.describe)(root, device)
  # the provider's device against BoltBeam's own reading of this machine's GPU
  measured_band = None
  if checker is None:
    from boltbeam.workflow.repeat_bands import for_decode
    measured_band = for_decode(run)
    if measured_band is None:
      raise ProviderRefused("not checked: no matching repeat evidence for this setup")
    checker = role_check.Checker(backend, None)
  ident = provider_check.identity(facts.get("target") or {}, checker.facts())
  if not ident["passed"]:
    raise ProviderRefused(ident["reason"])
  policy = read_json(run / "route_policy.json")
  routes = [r for r in policy.get("routes", []) if r.get("role") and r.get("quant") and r.get("shape")]
  if only: routes = [r for r in routes if f"{r['role']}:{r['quant']}" in only]
  if search is None and not (run / "tinygrad_timing_trace.json").is_file():
    say("role-time: start")
    role_numbers.time_roles(run, root=root)
  losses = role_numbers.role_losses(run)
  profile = read_json(run / "model_profile.json") if (run / "model_profile.json").is_file() else {}
  tensors = {role_key(r): r.get("tensor_name") for r in profile.get("roles", []) if r.get("tensor_name")}
  counts = {role_key(r): int(r["count"]) for r in profile.get("roles", []) if r.get("count")}
  # the biggest loss first: that is where a better kernel can save the most
  routes.sort(key=lambda r: -losses.get(role_key(r), {}).get("lost_ms", float("-inf")))
  provider_rev = (facts.get("provider_revision") or {}).get("revision")
  bb_rev, bb_dirty = compare_record.git_state(pathlib.Path(__file__).resolve().parents[2])
  model_sha = compare_record.sha256_file(pathlib.Path(model)) if search is None else "0" * 63 + "1"
  folder = run / compare_record.FOLDER
  timestamp = compare_record.now()
  env = tinygrad_fork.decode_env(backend, device)
  live_search = search is None
  if search is None:
    def search(request, evidence):  # noqa: E306 - the documented campaign, one provider process per role
      from boltbeam.search.full_kernel.tinygrad_full_kernel import PersistentJSONLSession
      from boltbeam.search.semantic_campaign_cli import run_request
      with PersistentJSONLSession((str(tinygrad_fork.fork_python(root)), "-m", "extra.llm_research.search_provider"), cwd=str(root)) as session:
        return run_request(request, provider_command=[], futuresight_evidence=evidence, provider_session=session,
                           requested_provider_revision=provider_rev)
  if ab_run is None:
    def ab_run(transforms, shape):  # noqa: E306
      return role_decision.run_ab(root, model, transforms=transforms, shape=shape, env=env)
  record = {"schema": SCHEMA, "run": run.name, "model_id": model_id, "target_id": target_id, "model_path": model,
            "model_sha256": model_sha, "provider_revision": provider_rev, "boltbeam_revision": bb_rev,
            "boltbeam_dirty": bb_dirty, "fork": str(root), "timestamp": timestamp, "timing_source": compare_record.TIMING_SOURCE,
            "device": device, "capability": reply, "identity": ident, "band": checker.band, "runtime_env": env,
            "note": "Kernel times are BoltBeam's kernel timer on the source the provider returned; whole-model times are "
                    "tinygrad runtime times; none is a llama.cpp time. Provider times are claims, kept beside.",
            "roles": []}
  ledger = LedgerStore(run / "route_ledger.jsonl")
  say(f"compare roles: {len(routes)} on {device}")
  for route in routes:
    role, quant = str(route["role"]), str(route["quant"])
    key, stem, rk = role_label(role, quant, route["shape"]), role_stem(role, quant, route["shape"]), role_key(route)
    n, k = (int(x) for x in route["shape"])
    if measured_band is not None:
      repeat_row = measured_band.get("rows", {}).get(key, {})
      reference = repeat_row.get("reference_ms")
      checker.band = repeat_row.get("width_ms", 0) / reference if reference else None
    say(f"role {key}: search")
    tensor_name = route.get("tensor_name") or tensors.get(rk)
    try:
      request, population, evidence, inputs = role_proposal.propose(route, tensor_name=str(tensor_name), model=model, model_id=model_id,
                                                      target_id=target_id, describe_result=facts, model_sha=model_sha,
                                                      run_id=f"{run.name}-compare", timestamp=timestamp,
                                                      sm_count=checker.facts().get("sm_count"))  # BoltBeam's bridge, after the identity check
    except Exception as exc:  # no space for this role: say why, the other roles go on
      request, population, evidence, inputs = None, None, None, {"why": str(exc)}
    if request is not None:
      compare_record.write(folder / f"{stem}-search-request.json", request | {"futuresight_evidence": evidence})
      compare_record.write(folder / f"{stem}-population.json", population)
      try:
        result = search(request, evidence)                           # 4. the campaign measures the survivors
      except Exception as exc:  # a provider failure blocks this role, not the others
        result = {"status": "BLOCKED", "counts": {}, "population": [], "error": str(exc),
                  "futuresight_static_evidence": evidence}
    else:
      result = {"status": "BLOCKED", "counts": {}, "population": [], "error": inputs["why"]}
    compare_record.write(folder / f"{stem}-search-result.json", result)
    summary = role_numbers.summarize_search(result, inputs.get("installed_row"))
    rejected = [r for r in (evidence or {}).get("rejections") or [] if "candidate_hash" in r]
    row: dict[str, Any] = {"role": role, "quant": quant, "shape": [n, k], "search": summary, "space": {
                             "binds_through": inputs.get("binds_through"), "family": inputs.get("family"), "generator": inputs.get("generator"),
                             "proposed": len((population or {}).get("candidates") or []),
                             "bubblebeam_rejected_rows": len((request or {}).get("rejected_coupled_rows") or []),
                             "futuresight_rejected": len(rejected),
                             "futuresight_reasons": sorted({str(r.get("reason")) for r in rejected})},
                           "search_result": f"{compare_record.FOLDER}/{stem}-search-result.json", "ab": None, "decision": None}
    winner = summary["winner"]
    decision: CandidateDecision | None = None
    verify, judged = None, None
    if winner is not None:
      say(f"role {key}: check")
      try:
        verify = checker(summary, result, model=model, quant=quant, rows=n, cols=k, role=role, tensor_name=tensor_name)
      except Exception as exc:  # BoltBeam could not run its own check: nothing the provider says is taken
        verify = {"error": f"{type(exc).__name__}: {exc}"[:400]}
      compare_record.write(folder / f"{stem}-check.json", verify)
      judged = role_check.checked(summary, verify) if "error" not in verify else {}
      short = lambda v: {kk: v[kk] for kk in ("plan", "verdict", "provider_us", "boltbeam_us", "boltbeam_us_less_floor", "ratio", "band", "reason") if kk in v} if isinstance(v, dict) else v  # noqa: E731
      row["check"] = {"winner": short(judged.get("winner")), "model_kernel": short(judged.get("model_kernel")),
                      "boltbeam_winner": short(judged.get("boltbeam_winner")), "reproduced": judged.get("reproduced"),
                      "not_reproduced": judged.get("not_reproduced"), "error": verify.get("error"), "tensor": verify.get("tensor"),
                      "floor_us": verify.get("floor_us"), "chip_band": verify.get("band")}
      row["check_result"] = f"{compare_record.FOLDER}/{stem}-check.json"
      if judged.get("boltbeam_winner"):  # BoltBeam's pick (its own fastest) replaces the provider's rank
        pick = judged["boltbeam_winner"]["candidate_hash"]
        row["provider_winner"] = winner
        winner = next(role_numbers.entry(r) for r in result["population"] if r.get("candidate_hash") == pick)
        model_row = summary.get("model_kernel")
        winner["is_model_kernel"] = bool(model_row) and model_row["candidate_hash"] == pick
        summary["winner"] = winner
        verify = {"winner": judged["boltbeam_winner"], "model_kernel": judged.get("model_kernel"), "band": verify.get("band")}
    reproduced = bool(verify) and (verify.get("winner") or {}).get("verdict") == provider_check.REPRODUCED
    k_numbers = role_numbers.kernel_numbers(summary, losses.get(rk), verify if reproduced else None, counts.get(rk))
    if winner is None:
      reason = result.get("error") or "no candidate compiled and passed the correctness check"
      row.update(status="blocked", reason=f"decided by the search: {reason}", decided_by="search")
    elif not reproduced:
      best = (verify or {}).get("winner") or {}
      why = (verify or {}).get("error") or (f"BoltBeam's fastest candidate, {best.get('plan')}: {best.get('reason')}" if best
                                            else "BoltBeam measured no candidate correct" if verify is not None else "BoltBeam's check did not run")
      row.update(status="blocked", decided_by="check", reason=f"{provider_check.NOT_REPRODUCED}: {why}", not_reproduced=True)
    elif winner["is_default"] or winner.get("is_model_kernel"):
      decision = role_decision.default_wins(route, winner, model_id=model_id, target_id=target_id,
                              search_path=row["check_result"], search_sha=compare_record.sha256_file(run / row["check_result"]),
                              model_kernel=bool(winner.get("is_model_kernel")))
    elif ab_only_if_faster and not (k_numbers or {}).get("faster_than_model"):
      model_us = (k_numbers or {}).get("model_kernel_us")
      row.update(status="blocked", decided_by="kernel alone",
                 reason=f"decided by the kernel alone: the best plan took {(k_numbers or {}).get('plan_us', 0):.1f} µs per call by BoltBeam's timer, "
                        + (f"the model's own kernel {model_us:.1f} µs; no whole-model A/B" if model_us is not None
                           else "with no model time to compare; no whole-model A/B"))
    elif winner["plan_kind"] != "tinygrad_opt_sequence.v1":
      row.update(status="blocked", decided_by="binding",
                 reason="decided by binding: the whole-model A/B binds Opt sequences only; an emitter plan reaches the "
                        "model through the fork's route admission, which the A/B driver does not install yet")
    else:
      say(f"role {key}: ab")
      try:
        ab = role_decision.matched_ab(ab_run(winner["transforms"], (n, k)), env)
      except Exception as exc:
        ab = None
        row.update(status="blocked", reason=f"decided by the whole model: the A/B did not run: {exc}",
                   decided_by="whole model")
      if ab is not None:
        ab_sha = compare_record.write(folder / f"{stem}-ab.json", ab)
        row["ab"] = {key_: ab[key_] for key_ in ("baseline_tok_s", "candidate_tok_s", "delta_pct", "spread_pct", "ranges_overlap", "pairs",
                                                 "token_match", "route_bound", "context", "timing_source")}
        row["ab_result"] = f"{compare_record.FOLDER}/{stem}-ab.json"
        row["ab"]["binding"] = ab["binding"]
      if ab is not None and not ab["route_bound"]:
        # Both arms ran the same code, so neither number can judge the plan. Not tested is not refuted.
        row.update(status="blocked", decided_by="binding",
                   reason=f"decided by binding: the plan reached {ab['binding'].get('calls_reached', 0)} of the "
                          f"{_calls(losses.get(rk))} {role} {quant} calls per token "
                          f"({ab['binding'].get('decode_graph_calls', 0)} decode calls in all) "
                          f"(applied to {ab['binding'].get('applied', 0)} other kernels); "
                          "the A/B compared the default decode with itself")
      elif ab is not None:
        decision = role_decision.decide(route, winner, ab, model_id=model_id, target_id=target_id, ab_path=row["ab_result"], ab_sha=ab_sha)
    row["kernel"] = k_numbers
    if row["kernel"] and row["kernel"]["model_us_per_call"] is not None:
      row["role_calls_per_token"] = row["kernel"]["tensors_per_token"]
    if decision is not None:
      by = "kernel alone" if row["ab"] is None else "whole model"
      row.update(status=role_decision.route_status(decision.verdict), reason=f"decided by the {by}: {decision.reason}",
                 decided_by=by, decision=decision.to_json())
      if status_for_verdict(decision.verdict) is not None:
        ledger.add(decision, scope={"model": model_id, "target": target_id, "workload": "decode", "role": role, "quant": quant},
                   reopen_condition=REOPEN if decision.verdict == Verdict.REFUTE.value else "")
    if winner is not None: row["plan_id"] = role_numbers.plan_id(role, quant, winner)
    record["roles"].append(row)
    _apply(route, row)
    write_json(run / "route_policy.json", policy)
    compare_record.write(folder / "compare.json", record)
    say(f"role {key}: done {row['status']}")
    if step: step(len(record["roles"]), len(routes))
  if fusions and measured_band is not None:
    say("fusions not checked: no repeat evidence for the fused workload")
  if fusions and measured_band is None:
    try:
      found = fusion_compare.compare_fusions(run, facts=facts, describe_shapes=lambda shapes: ((describe_fn or tinygrad_fork.describe)(root, device, shapes=shapes)
                                                                                 .get("fusion_shape_verdicts") or []),
                              search=search, checker=checker, model=model, model_sha=model_sha, target_id=target_id, losses=losses,
                              timestamp=timestamp, say=say, tensors=tensors_of_model)
    except Exception as exc:  # the fusions are their own stage: a failure there leaves the roles' results standing
      found = {"schema": fusion_compare.FUSIONS_SCHEMA, "run": run.name, "fusions": [], "error": f"{type(exc).__name__}: {exc}"[:400]}
      compare_record.write(folder / compare_record.FUSIONS_FILE, found)
      say(f"compare fusions: failed: {found['error']}")
    # the flash stage runs on the real provider, or on a double a caller passes; a test double for the search alone skips it
    if flash_fn is not False and (flash_fn is not None or live_search):
      try:
        flash_row = (flash_fn or flash_compare.compare_flash)(run, root=root, device=device, checker=checker, band=checker.band, say=say)
      except Exception as exc:  # its own stage: a failure there leaves the roles and fusions standing
        flash_row = {"name": "flash", "verdict": "not_searched", "reason": f"the flash stage failed: {type(exc).__name__}: {exc}"[:400]}
      found.setdefault("fusions", []).append(flash_row)
      compare_record.write(folder / compare_record.FUSIONS_FILE, found)
      say(f"flash: done {flash_row.get('verdict')}")
    record["fusions"] = [{k: f.get(k) for k in ("name", "quant", "verdict", "reason", "in_model_ms", "lost_ms")} | {
      "fused_ms": (f.get("fused") or {}).get("ms_per_token"), "unfused_ms": (f.get("unfused") or {}).get("ms_per_token")}
      for f in found.get("fusions") or []]
    record["fusions_result"] = f"{compare_record.FOLDER}/{compare_record.FUSIONS_FILE}"
    compare_record.write(folder / "compare.json", record)
  say(f"compare done: {run}")
  return record


def _apply(route:dict[str, Any], row:Mapping[str, Any]) -> None:
  """Write one role's outcome into its route_policy.json row. selected_route is set only for a promoted plan,
  because consumers of route_policy.json run a selected route."""
  route["status"] = row["status"]
  route["selected_route"] = row.get("plan_id") if row["status"] == "promoted" else None
  route["evidence_refs"] = [x for x in (row.get("check_result"), row.get("ab_result")) if x]
  winner = row["search"]["winner"] or {}
  route["compare"] = {"plan_id": row.get("plan_id"), "plan": winner.get("plan"), "plan_hash": winner.get("plan_hash"),
                      "search_median_ns": winner.get("median_ns"), "default_median_ns": row["search"]["default_median_ns"],
                      "kernel": row.get("kernel"), "decided_by": row.get("decided_by"),
                      "role_calls_per_token": row.get("role_calls_per_token"),
                      "measured_correct": row["search"]["counts"].get("measured_correct"),
                      "candidates": row["search"]["counts"].get("total"), "ab": row.get("ab"), "reason": row.get("reason"),
                      "timing_source": compare_record.TIMING_SOURCE, "check": row.get("check"), "not_reproduced": bool(row.get("not_reproduced")),
                      "space": row.get("space")}
