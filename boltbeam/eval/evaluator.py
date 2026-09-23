"""Evaluator: classify normalized evidence into a CandidateDecision (audit-brain-build-scope BB5).

This module is the promotion contract. It reads NormalizedEvidence rows only (never raw producer fields),
compares against the ONE threshold table (eval/thresholds.py) and the ONE candidate manifest (manifest.py),
and emits a typed CandidateDecision whose invariants are enforced by that dataclass (notably: no PROMOTE
without a rollback). It measures nothing and runs no GPU/model code — a search result is not a win until
downstream measurement proves it, and here we only read the proof.

Guardrails (all seven from vocab.Guardrail) resolve to PASS / FAIL / UNKNOWN. Tri-state evidence flags mean
True=proven, False=proven-absent, None=unknown; UNKNOWN never silently becomes a pass. Promotion requires
every guardrail to PASS, a tier-A/B speed win, and a rollback knob.
"""
from __future__ import annotations

from typing import Any

from boltbeam.vocab import (Verdict, Tier, Guardrail, GuardrailState, BackendStatus, RowKind, KVPolicyClass,
                            TargetCapability)
from boltbeam.target.targets import target_backend_status, target_capability
from boltbeam.artifacts.base import NormalizedEvidence, EvidenceRow, EvidenceFlags
from boltbeam.ledger.model import CandidateDecision, EvidenceRef
from boltbeam.manifest import Candidate, load_candidates
from boltbeam.eval.thresholds import classify_tier, is_protected_regression, TierResult, SPEED_EQUIVALENT_BAND_PCT
from boltbeam.eval.contracts import promotion_requirements
from boltbeam.math.kv_cache import classify_kv_slope

_PASS, _FAIL, _UNKNOWN = GuardrailState.PASS.value, GuardrailState.FAIL.value, GuardrailState.UNKNOWN.value

# required guardrails must PASS to promote; conditional guardrails only block on a firm FAIL (checked earlier)
_REQUIRED_GUARDRAILS = tuple(promotion_requirements())  # single source: eval/contracts.py
_CONDITIONAL_GUARDRAILS = (Guardrail.MEMORY_FIT.value, Guardrail.FALLBACK.value, Guardrail.DETERMINISM.value)


# ---- flag / row helpers -----------------------------------------------------------------------------------

def _merge_flag(vals:list[bool | None]) -> bool | None:
  """Combine one tri-state flag across evidences: a proven-absent (False) dominates, else any proven True wins,
  else unknown. UNKNOWN is never upgraded to a pass by absence of a contradicting row."""
  present = [v for v in vals if v is not None]
  if not present: return None
  if False in present: return False
  return True


def _merge_flags(evidences:list[NormalizedEvidence]) -> EvidenceFlags:
  fs = [e.flags for e in evidences]
  return EvidenceFlags(route_bound=_merge_flag([f.route_bound for f in fs]),
                       token_match=_merge_flag([f.token_match for f in fs]),
                       deterministic=_merge_flag([f.deterministic for f in fs]),
                       hidden_fallback=_merge_flag([f.hidden_fallback for f in fs]))


def _all_rows(evidences:list[NormalizedEvidence]) -> list[EvidenceRow]:
  return [r for e in evidences for r in e.rows]


def _candidate_rows(candidate:Candidate, rows:list[EvidenceRow]) -> list[EvidenceRow]:
  """Only the rows whose (quant, role) the candidate applies to. A row with quant/role None is
  candidate-agnostic (applies_to treats None as 'any') and is kept — e.g. whole-workload prefill tok/s.
  This stops a row for a DIFFERENT quant/role (a Q6_K/ffn_down regression) from scoring or refuting a
  Q4_K/attn_kv candidate: matching authorizes the candidate, scoping decides what evidence judges it."""
  return [r for r in rows
          if candidate.applies_to(workload=candidate.workload, quant=r.quant, role=r.role)]


_SPEED_KINDS = ("wd_speed", "prefill_speed")


def _row_delta_pct(row:EvidenceRow) -> float | None:
  """Signed whole-workload %delta vs baseline (positive = faster) from one speed row.

  - metric "delta_pct": the value IS the signed gain.
  - metric "lag_pct": percent BEHIND baseline, so the gain is its negation.
  - metric "tok_s": needs a baseline tok/s in row.extra["baseline_tok_s"]; delta = (cand-base)/base*100.
  """
  m = row.metric
  if m == "delta_pct": return float(row.value)
  if m == "lag_pct": return -float(row.value)
  if m == "tok_s":
    base = row.extra.get("baseline_tok_s")
    if base in (None, 0): return None
    return (float(row.value) - float(base)) / float(base) * 100.0
  return None


def _speed_points(rows:list[EvidenceRow]) -> list[tuple[float, float | None, int | None]]:
  """Every computable speed measurement as (signed %delta, spread_pct, context). ALL contexts, not just the
  first — a regression at one context must not be hidden by a win at another (the 'protected regression on any
  context' principle). spread_pct feeds the tier noise floor."""
  out:list[tuple[float, float | None, int | None]] = []
  for r in rows:
    if r.kind in _SPEED_KINDS:
      d = _row_delta_pct(r)
      if d is not None:
        sp = r.extra.get("spread_pct")
        out.append((d, (float(sp) if sp is not None else None), r.context))
  return out


def _first_protected_regression(points:list[tuple[float, float | None, int | None]]) -> tuple[float, int | None] | None:
  """The worst noise-qualified protected-context regression, if any. A regression smaller than the observed
  spread at that context is noise (skipped); one beyond the protected guard fails the whole candidate."""
  worst:tuple[float, int | None] | None = None
  for delta, spread, ctx in points:
    tr = classify_tier(delta, spread_pct=spread)
    if tr.is_regression and is_protected_regression(delta):
      if worst is None or delta < worst[0]: worst = (delta, ctx)
  return worst


def _representative(points:list[tuple[float, float | None, int | None]]) -> tuple[float | None, float | None]:
  """The delta used for tiering the promotable case: the MEDIAN across contexts (robust to one noisy ctx),
  paired with the MAX spread across contexts (conservative noise floor)."""
  if not points: return None, None
  deltas = sorted(p[0] for p in points)
  n = len(deltas)
  med = deltas[n // 2] if n % 2 else (deltas[n // 2 - 1] + deltas[n // 2]) / 2.0
  spreads = [p[1] for p in points if p[1] is not None]
  return med, (max(spreads) if spreads else None)


def _refuted_reopen_note(candidate:Candidate, delta:float | None) -> str:
  """A note appended to a PROMOTE reason/next_action when the candidate re-opens a previously-refuted axis on
  new positive evidence. The evaluator does not receive the ledger, so this SURFACES the reopen requirement
  rather than hard-gating on it."""
  if candidate.refuted_axis_tags and delta is not None and delta > 0:
    tag = candidate.refuted_axis_tags[0]
    return (f" re-opening previously-refuted axis {tag!r} on new positive evidence; "
            f"verify ledger reopen_condition before acting")
  return ""


def _memory_state(rows:list[EvidenceRow]) -> str:
  """memory_fit is UNKNOWN unless evidence carries it (row extra "memory_fit": bool)."""
  vals = [r.extra.get("memory_fit") for r in rows if "memory_fit" in r.extra]
  if any(v is False for v in vals): return _FAIL
  if any(v is True for v in vals): return _PASS
  return _UNKNOWN


def _missing_knob(candidate:Candidate, rows:list[EvidenceRow]) -> str | None:
  """A needed topology knob the search space cannot express yet, from an explicit candidate input
  (thresholds["requires_knob"]) or an evidence row extra ("missing_knob")."""
  req = candidate.thresholds.get("requires_knob")
  if req:
    available:set[str] = set()
    for r in rows:
      ks = r.extra.get("knobs_available") or r.extra.get("available_knobs")
      if isinstance(ks, (list, tuple)): available.update(ks)
    if req not in available: return str(req)
  for r in rows:
    mk = r.extra.get("missing_knob")
    if mk: return str(mk)
  return None


def _reduce_eliminated_state(candidate:Candidate, rows:list[EvidenceRow]) -> str | None:
  """Mechanism guardrail for reduce-elimination candidates. Returns None when the candidate does not target a
  reduce class (guardrail N/A). Otherwise it requires a reduce_source row tagged with the targeted class
  (extra['reduce_class']) carrying the baseline bucket% (extra['baseline_pct_gpu']) and the candidate bucket% as
  the row value: PASS if the bucket shrank by at least min_reduce_shrink_pct, FAIL if it did not shrink,
  UNKNOWN if no such before/after evidence exists. This stops a noisy tok/s bump from being credited to a
  fusion that did not actually remove the reduce."""
  cls = candidate.thresholds.get("targets_reduce_class")
  if not cls: return None
  min_shrink = float(candidate.thresholds.get("min_reduce_shrink_pct", 0.5))
  for r in rows:
    if r.kind != RowKind.REDUCE_SOURCE.value: continue
    if r.extra.get("reduce_class") != cls: continue
    base = r.extra.get("baseline_pct_gpu")
    if base is None: continue
    return _PASS if (float(base) - float(r.value)) >= min_shrink else _FAIL
  return _UNKNOWN


def _reduces_surface(candidate:Candidate, rows:list[EvidenceRow]) -> bool:
  if candidate.thresholds.get("reduces_surface") or candidate.thresholds.get("purity"): return True
  return any(r.extra.get("reduces_surface") or r.extra.get("purity") for r in rows)


def _fallback_state(v:bool | None) -> str:
  if v is False: return _PASS          # no hidden fallback -> pass
  if v is True: return _FAIL
  return _UNKNOWN


def _tri(v:bool | None) -> str:
  if v is True: return _PASS
  if v is False: return _FAIL
  return _UNKNOWN


def _speed_state(tr:TierResult | None) -> str:
  if tr is None: return _UNKNOWN
  if tr.is_regression: return _FAIL
  if tr.is_win: return _PASS
  return _UNKNOWN                        # equivalent or sub-threshold (tier C)


def _citations(evidences:list[NormalizedEvidence]) -> tuple[EvidenceRef, ...]:
  refs = []
  for e in evidences:
    kind = e.rows[0].kind if e.rows else ""
    claim = f"{e.source.producer} {e.source.tool} reports {kind} for {e.workload} on {e.target_id}"
    refs.append(EvidenceRef(evidence_id=e.evidence_id, path=e.source.path, kind=kind,
                            fingerprint=e.source.fingerprint, claim=claim))
  return tuple(refs)


def _kv_policy_result(candidate:Candidate, rows:list[EvidenceRow], target_id:str, speed_points:list[tuple[float, float | None, int | None]]):
  """Classify KV-cache candidates from KV slope evidence. Returns None for non-KV candidates."""
  policy = candidate.thresholds.get("kv_cache_policy")
  if not policy:
    return None
  kv_rows = [r for r in rows if r.kind == RowKind.KV_CTX_SLOPE.value and r.metric == "b_ms_per_ctx"]
  if not kv_rows:
    return (KVPolicyClass.INCONCLUSIVE.value, "missing kv_ctx_slope evidence", None)
  row = kv_rows[0]
  extra = row.extra
  storage_b = extra.get("storage_only_b_ms_per_ctx")
  baseline_b = extra.get("baseline_b_ms_per_ctx")
  baseline_bytes = extra.get("baseline_kv_bytes_per_ctx_token")
  r2 = extra.get("r2")
  kv_bytes = extra.get("kv_bytes_per_ctx_token")
  if storage_b is None or r2 is None or kv_bytes is None:
    return (KVPolicyClass.INCONCLUSIVE.value, "kv_ctx_slope row lacks storage/r2/bytes residual inputs", None)
  mixed_fastpath = bool(target_capability(target_id, TargetCapability.MIXED_QUANT_KV_FLASH_FASTPATH.value, False))
  rep_delta, _spread = _representative(speed_points)
  cap_required = bool(candidate.thresholds.get("capacity_required") or extra.get("capacity_required"))
  res = classify_kv_slope(
    measured_b_ms_per_ctx=float(row.value),
    storage_only_b_ms_per_ctx=float(storage_b),
    r2=float(r2),
    kv_bytes_per_ctx_token=float(kv_bytes),
    baseline_b_ms_per_ctx=float(baseline_b) if baseline_b is not None else None,
    baseline_kv_bytes_per_ctx_token=float(baseline_bytes) if baseline_bytes is not None else None,
    cache_type_k=str(extra.get("cache_type_k") or candidate.thresholds.get("cache_type_k") or ""),
    cache_type_v=str(extra.get("cache_type_v") or candidate.thresholds.get("cache_type_v") or ""),
    mixed_fastpath=mixed_fastpath,
    whole_decode_delta_pct=rep_delta,
    capacity_required=cap_required)
  return (res.policy_class, res.reason, res)


# ---- public API -------------------------------------------------------------------------------------------

def evaluate(candidate:Candidate, evidences:list[NormalizedEvidence], *, model_id:str, target_id:str,
             workload:str) -> CandidateDecision:
  """Classify the evidence for one candidate into a CandidateDecision."""
  # scope scoring rows to this candidate's (quant, role): flags/citations stay evidence-level, but the speed /
  # memory / knob signals must come only from rows that actually describe this candidate (audit row-scope invariant).
  rows = _candidate_rows(candidate, _all_rows(evidences))
  flags = _merge_flags(evidences)
  cites = _citations(evidences)

  target_complete = target_backend_status(target_id) == BackendStatus.COMPLETE.value

  def decide(verdict:str, reason:str, *, tier:str = Tier.NONE.value, next_action:str = "",
             guardrails:dict[str, str] | None = None, rollback:dict[str, str] | None = None) -> CandidateDecision:
    # A6: a would-be promote on a descriptor-only/unsupported backend is target-backend-incomplete -> DEFER.
    if verdict == Verdict.PROMOTE.value and not target_complete:
      verdict, next_action = Verdict.DEFER.value, f"build the {target_id} backend evaluator before promoting"
      reason = f"{reason}; blocked: target backend {target_id!r} is not complete (target-backend-incomplete)"
    return CandidateDecision(candidate_id=candidate.candidate_id, model_id=model_id, target_id=target_id,
                             workload=workload, verdict=verdict, tier=tier, reason=reason, evidence=cites,
                             rollback=rollback if rollback is not None else dict(candidate.rollback),
                             next_action=next_action, guardrails=guardrails or {})

  # 1) required evidence must be present, else we cannot judge this candidate at all
  present = {r.kind for r in rows}
  for kind in candidate.required_evidence_kinds:
    if kind not in present:
      return decide(Verdict.INCONCLUSIVE.value, f"missing required evidence kind {kind!r} for {candidate.candidate_id}",
                    next_action=f"produce a {kind} artifact for this candidate")

  # 2) a needed topology knob the search space cannot express yet
  knob = _missing_knob(candidate, rows)
  if knob:
    return decide(Verdict.SEARCH_SPACE_INCOMPLETE.value,
                  f"candidate {candidate.candidate_id} needs knob {knob!r} which the search space cannot express yet",
                  next_action=f"extend the emitter/target to expose {knob!r}")

  points = _speed_points(rows)
  kv_policy = _kv_policy_result(candidate, rows, target_id, points)
  if kv_policy is not None:
    cls, kv_reason, kv_detail = kv_policy
    if cls == KVPolicyClass.STORAGE_WIN_SPEED_LOSS.value:
      return decide(Verdict.REFUTE.value,
                    f"quantized KV is smaller but speed-negative: {kv_reason}",
                    next_action="keep f16/bf16 KV as the speed default; use this dtype only for capacity")
    if cls == KVPolicyClass.CAPACITY_ONLY.value:
      return decide(Verdict.CANDIDATE.value,
                    f"capacity-only KV route: {kv_reason}",
                    next_action="select only when f16/bf16 KV would exceed the context or memory budget")
    if cls == KVPolicyClass.STORAGE_WIN_SPEED_WIN.value:
      # Fall through to ordinary promotion if whole-decode speed evidence exists; otherwise the classifier would
      # have returned inconclusive. Guardrails below still require rollback/correctness/route where applicable.
      pass
    elif cls in (KVPolicyClass.MIXED_FASTPATH_UNKNOWN.value, KVPolicyClass.FASTPATH_MISSING.value):
      return decide(Verdict.DEFER.value,
                    f"quantized KV fastpath not proven: {kv_reason}",
                    next_action="prove the target's quantized FlashAttention fastpath before measuring this candidate")
    elif cls == KVPolicyClass.INCONCLUSIVE.value:
      return decide(Verdict.INCONCLUSIVE.value,
                    f"quantized KV policy inconclusive: {kv_reason}",
                    next_action="capture a KV context-slope artifact with baseline residual fields")
  # a protected-context regression at ANY context refutes the candidate BEFORE tiering the median/best case,
  # so a win at one context cannot mask a regression at another.
  worst_reg = _first_protected_regression(points)
  delta, spread = _representative(points)
  tr = classify_tier(delta, spread_pct=spread) if delta is not None else None

  guardrails = {
    Guardrail.CORRECTNESS.value: _tri(flags.token_match),
    Guardrail.ROUTE_BOUND.value: _tri(flags.route_bound),
    Guardrail.SPEED.value: _speed_state(tr),
    Guardrail.MEMORY_FIT.value: _memory_state(rows),
    Guardrail.FALLBACK.value: _fallback_state(flags.hidden_fallback),
    # byte-identical greedy output (token_match) proves determinism; else fall back to an explicit flag
    Guardrail.DETERMINISM.value: (_PASS if flags.token_match is True else _tri(flags.deterministic)),
    Guardrail.ROLLBACK.value: _PASS if candidate.rollback else _FAIL,
  }
  # mechanism check (only for candidates that declare a targeted reduce class): the bucket must actually shrink.
  reduce_state = _reduce_eliminated_state(candidate, rows)
  if reduce_state is not None:
    guardrails[Guardrail.REDUCE_ELIMINATED.value] = reduce_state
  tier_for = tr.tier if tr is not None else Tier.NONE.value

  # 3) firm negative gates -> REFUTE (failed a gate with firm evidence)
  if flags.token_match is False:
    return decide(Verdict.REFUTE.value, "correctness gate failed: token mismatch vs reference",
                  tier=tier_for, guardrails=guardrails, next_action="fix output equivalence before re-measuring")
  if worst_reg is not None:
    d, ctx = worst_reg
    tag = f" (refuted axis {candidate.refuted_axis_tags[0]!r})" if candidate.refuted_axis_tags else ""
    at = f" at ctx{ctx}" if ctx is not None else ""
    return decide(Verdict.REFUTE.value, f"protected-context regression of {d:.1f}%{at}{tag}", tier=tier_for,
                  guardrails=guardrails, next_action="record refuted axis; reopen on a new primitive")
  if tr is not None and (tr.is_regression or (candidate.refuted_axis_tags and delta < 0)):
    tag = f" (refuted axis {candidate.refuted_axis_tags[0]!r})" if candidate.refuted_axis_tags else ""
    return decide(Verdict.REFUTE.value, f"speed regression of {delta:.1f}%{tag}", tier=tier_for,
                  guardrails=guardrails, next_action="record refuted axis; reopen on a new primitive")
  if guardrails[Guardrail.MEMORY_FIT.value] == _FAIL:
    return decide(Verdict.REFUTE.value, "memory_fit gate failed: candidate exceeds available memory",
                  tier=tier_for, guardrails=guardrails, next_action="shrink footprint or defer")
  if flags.hidden_fallback is True:
    return decide(Verdict.REFUTE.value, "hidden fallback detected: the intended route is not actually taken",
                  tier=tier_for, guardrails=guardrails, next_action="remove the silent fallback path")
  if flags.deterministic is False:
    return decide(Verdict.REFUTE.value, "determinism gate failed: output is not reproducible", tier=tier_for,
                  guardrails=guardrails, next_action="make the route deterministic before re-measuring")
  if flags.route_bound is False:
    return decide(Verdict.REFUTE.value, "route_bound gate failed: the generated route did not bind", tier=tier_for,
                  guardrails=guardrails, next_action="ensure the route is selected before crediting its speed")
  if reduce_state == _FAIL:
    cls = candidate.thresholds.get("targets_reduce_class")
    return decide(Verdict.REFUTE.value,
                  f"reduce_eliminated gate failed: the targeted {cls!r} reduce bucket did not shrink in the "
                  f"before/after profile; the speed change is not attributable to the claimed fusion",
                  tier=tier_for, guardrails=guardrails,
                  next_action="the mechanism was not shown in the profile; do not credit tok/s to this fusion")

  # 4) needed correctness/route flags unknown -> INCONCLUSIVE (do not infer a pass from absence)
  if flags.token_match is None:
    return decide(Verdict.INCONCLUSIVE.value, "correctness unknown: no token_match evidence", tier=tier_for,
                  guardrails=guardrails, next_action="capture a token-match check for this candidate")
  if flags.route_bound is None:
    return decide(Verdict.INCONCLUSIVE.value, "route binding unknown: no route_bound evidence", tier=tier_for,
                  guardrails=guardrails, next_action="capture route-binding evidence for this candidate")
  if tr is None:
    return decide(Verdict.INCONCLUSIVE.value, "speed delta not computable from the evidence rows", tier=tier_for,
                  guardrails=guardrails, next_action="produce a wd_speed/prefill_speed row with a baseline")

  # 5) speed-equivalent: not a win, not a regression. A GENERATED route that is speed-equivalent to the
  #    hand-written one still PROMOTES when it reduces hand-written surface AND the hard guardrails
  #    (correctness / route-binding / rollback) hold — this is the real g3 case (BB5): +/-noise speed,
  #    token-identical, route-bound, replacing a hand-written kernel. Otherwise it stays a purity candidate
  #    (or a plain diagnostic when it does not reduce surface).
  if tr.is_equivalent:
    if _reduces_surface(candidate, rows):
      guardrails_ok = (guardrails[Guardrail.CORRECTNESS.value] == _PASS
                       and guardrails[Guardrail.ROUTE_BOUND.value] == _PASS and bool(candidate.rollback)
                       and (reduce_state is None or reduce_state == _PASS))
      if guardrails_ok:
        note = _refuted_reopen_note(candidate, delta)
        return decide(Verdict.PROMOTE.value,
                      f"speed-equivalent generated replacement (reduces hand-written surface){note}",
                      tier=Tier.NONE.value, guardrails=guardrails,
                      next_action="promote and record in the ledger" + note)
      return decide(Verdict.CANDIDATE.value,
                    f"speed-equivalent ({delta:.1f}%) reduces hand-written surface but correctness/route/rollback not all confirmed",
                    tier=tier_for, guardrails=guardrails,
                    next_action="confirm correctness, route-binding, and a rollback knob to promote the purity replacement")
    return decide(Verdict.DIAGNOSTIC.value, f"correct but speed-equivalent ({delta:.1f}%); not a promotable win",
                  tier=tier_for, guardrails=guardrails, next_action="explains behavior; not a candidate for promotion")

  # 6) sub-threshold movement (tier C) -> diagnostic only unless strategic
  if not tr.is_win:
    return decide(Verdict.DIAGNOSTIC.value, f"sub-threshold movement ({delta:.1f}%, tier C)", tier=tier_for,
                  guardrails=guardrails, next_action="below tier-B; diagnostic unless strategic")

  # 7) tier A/B win. Never promote without a rollback.
  if not candidate.rollback:
    return decide(Verdict.DEFER.value, f"tier-{tr.tier} win ({delta:.1f}%) but no rollback knob; cannot promote",
                  tier=tier_for, guardrails=guardrails, rollback={},
                  next_action="add a rollback knob to the candidate before promotion")

  # promotion gate: the REQUIRED guardrails must PASS (correctness/route_bound/speed/rollback). The CONDITIONAL
  # guardrails (memory_fit "if evidence exists", hidden_fallback "absent", determinism "or noise-qualified") have
  # already been REFUTEd above on any firm FAIL, so here they are pass-or-unknown and do not block — an unknown
  # conditional is noted, not treated as a failed proof (audit-brain-build-scope BB5).
  # a candidate that DECLARES a targeted reduce class must also prove the bucket shrank (reduce_eliminated PASS):
  # UNKNOWN here means "no before/after profile" -> not promotable until the mechanism is shown.
  required = _REQUIRED_GUARDRAILS + ((Guardrail.REDUCE_ELIMINATED.value,) if reduce_state is not None else ())
  required_fail = [g for g in required if guardrails[g] != _PASS]
  if required_fail:
    return decide(Verdict.CANDIDATE.value,
                  f"tier-{tr.tier} win ({delta:.1f}%) but required guardrails not confirmed: {', '.join(sorted(required_fail))}",
                  tier=tier_for, guardrails=guardrails, next_action=f"confirm {', '.join(sorted(required_fail))} to promote")

  unconfirmed = [g for g in _CONDITIONAL_GUARDRAILS if guardrails[g] == _UNKNOWN]
  note = f" (conditional guardrails unconfirmed: {', '.join(sorted(unconfirmed))})" if unconfirmed else ""
  reopen_note = _refuted_reopen_note(candidate, delta)
  return decide(Verdict.PROMOTE.value,
                f"tier-{tr.tier} win ({delta:.1f}%); required guardrails pass with rollback{note}{reopen_note}",
                tier=tr.tier, guardrails=guardrails, next_action="promote and record in the ledger" + reopen_note)


def _matches(candidate:Candidate, ev:NormalizedEvidence) -> bool:
  """A candidate matches evidence only if a SINGLE row satisfies the candidate's workload+quant+role together.
  Collecting quants and roles separately and cross-producting them would falsely match e.g. a Q4_K/attn_kv
  candidate against evidence that only proves Q6_K/attn_kv and Q4_K/ffn_down (no row proves Q4_K/attn_kv)."""
  if ev.workload != candidate.workload: return False
  for r in ev.rows:
    if candidate.applies_to(workload=ev.workload, quant=r.quant, role=r.role): return True
  return False


def search_space_pairs(search:dict[str, Any]) -> set[tuple[str | None, str | None]]:
  """(quant, role) pairs the supplied search space authorizes. An empty set means 'unconstrained'."""
  pairs:set[tuple[str | None, str | None]] = set()
  for role in search.get("roles", []) if isinstance(search, dict) else []:
    pairs.add((role.get("quant"), role.get("role")))
  return pairs


def candidate_in_search_space(candidate:Candidate, search:dict[str, Any]) -> bool:
  """Whether a candidate is authorized by the supplied search space (a stale/wrong search file must be able to
  block a promotion — the profile/search/evidence contract). A candidate applies to a search (quant, role) pair
  if its quant/role constraints (empty = any) overlap that pair."""
  pairs = search_space_pairs(search)
  if not pairs: return True   # no roles listed -> no constraint expressed
  for q, r in pairs:
    if (not candidate.quant or q in candidate.quant) and (not candidate.roles or r in candidate.roles):
      return True
  return False


def candidates_in_search_space(candidates:list[Candidate], search:dict[str, Any]) -> list[Candidate]:
  return [c for c in candidates if candidate_in_search_space(c, search)]


def evaluate_all(evidences:list[NormalizedEvidence], model_id:str, target_id:str, workload:str,
                 candidates:list[Candidate] | None = None) -> list[CandidateDecision]:
  """Match evidence to applicable candidates (via Candidate.applies_to) and evaluate each. Candidates with no
  matching evidence are skipped — a candidate only earns a decision once there is evidence to judge it."""
  cands = candidates if candidates is not None else load_candidates()
  out:list[CandidateDecision] = []
  for c in cands:
    if c.workload != workload: continue
    matched = [e for e in evidences if _matches(c, e)]
    if not matched: continue
    out.append(evaluate(c, matched, model_id=model_id, target_id=target_id, workload=workload))
  return out


# preference order when picking one primary decision out of many: most actionable first
_VERDICT_PRIORITY = {
  Verdict.PROMOTE.value: 0, Verdict.REFUTE.value: 1, Verdict.DEFER.value: 2,
  Verdict.SEARCH_SPACE_INCOMPLETE.value: 3, Verdict.CANDIDATE.value: 4,
  Verdict.DIAGNOSTIC.value: 5, Verdict.INCONCLUSIVE.value: 6, Verdict.ADAPTER_INCOMPLETE.value: 7,
}


def primary_decision(decisions:list[CandidateDecision]) -> CandidateDecision:
  """Pick one decision to report as "the" answer when several candidates disagree, by verdict priority
  (most actionable first: PROMOTE > REFUTE > DEFER > SEARCH_SPACE_INCOMPLETE > CANDIDATE > DIAGNOSTIC >
  INCONCLUSIVE > ADAPTER_INCOMPLETE)."""
  return min(decisions, key=lambda d: _VERDICT_PRIORITY.get(d.verdict, 99))
