"""Reachability audit (audit-brain-build-scope BB10) — the codegen-directive signal.

This module answers ONE question per candidate/knob: can the search legally get here right now, and if
not, why? It does NOT decide promotion (that is eval/) and it does NOT emit or run code. It classifies
each candidate and each route-grammar knob into a `vocab.Reachability` class:

  REACHABLE          the candidate applies to a present role/quant and is not walled or ledger-refuted;
  REFUTED_BY_LEDGER  a ledger row refuted it with do_not_retry and its reopen condition is not met;
  EMITTER_BLOCKED    a knob is present in the grammar (or desired) but the emitter cannot lower it —
                     this also covers knobs that are not represented at all (search-space-incomplete),
                     which are walled, NOT refuted;
  PRIMITIVE_MISSING  the emitter has no base primitive for the role/quant;
  TARGET_INCOMPLETE  the TargetProfile lacks capability data needed to reason about the route.

Emitter capability is DATA (an `EmitterCapabilities` set), not evaluator branches, so widening what the
emitter can express is a data edit. The legal route grammar itself is reused from `search.emit`, keeping a
single source of truth for route families/knobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from boltbeam.vocab import Reachability, LedgerStatus, SCHEMA_REACHABILITY_REPORT
from boltbeam.profile.ir import ModelProfile, TargetProfile
from boltbeam.manifest import Candidate
from boltbeam.ledger.model import LedgerEntry
from boltbeam.ledger.store import REOPEN_MET_KEY
from boltbeam.search.emit import emit_search_space


@dataclass(frozen=True)
class EmitterCapabilities:
  """What the code emitter/grammar can actually express, as DATA.

  `blocked_knobs` holds two token shapes:
    "words_per_group!=8"  the knob exists in the grammar but only the value after '!=' is emittable;
                          every other value is a non-exposed topology knob -> EMITTER_BLOCKED.
    "split_k"             a bare token: the knob is not representable at all (not in the route grammar),
                          a search-space-incomplete hole surfaced as EMITTER_BLOCKED (never refuted).
  `missing_primitives` holds `quant` or `role:quant` tokens whose base primitive the emitter cannot build.
  """
  blocked_knobs: frozenset[str] = frozenset()
  missing_primitives: frozenset[str] = frozenset()

  def allowed_value(self, knob:str) -> str | None:
    """The single emittable value for `knob` if it is conditionally blocked ("name!=val"), else None."""
    for tok in self.blocked_knobs:
      name, sep, val = tok.partition("!=")
      if sep and name == knob:
        return val
    return None

  def fully_blocked_knobs(self) -> tuple[str, ...]:
    """Bare tokens (no '!=') — knobs the grammar does not represent at all."""
    return tuple(sorted(t for t in self.blocked_knobs if "!=" not in t))


# Default emitter reality for the shipped gfx1100 Q4_K LaneMap grammar: only words_per_group==8 lowers, and
# split-K reduction is not in the grammar at all. v_dot2 and vec_store_to_reg are ISA-present but the renderer
# cannot lower to them yet (v_dot2_f32_f16 IS in the target dot_primitives — the gap is lowering, not hardware),
# so they are EMITTER_BLOCKED (walled, not refuted): the named reasons the 14B decode route search closes out
# (see docs/14b-aggregate-fusion-closeout). Cross-workgroup LSE atomics / grid-sync is a MISSING target
# capability, recorded on the TargetProfile (amd_gfx1100 capabilities.cross_workgroup_atomic_float=false), not
# an emitter-lowering gap.
DEFAULT_EMITTER_CAPABILITIES = EmitterCapabilities(
  # Emitter/impl lowering gaps (buildable, not hardware-missing): words_per_group!=8 and split_k are Q4_K
  # topology holes; vec_store_to_reg is ISA-present but the renderer cannot lower to it; fused_reduce_scale is
  # the single-load reduce+scale generated kernel the RMSNorm/reduce candidates need ("not a scheduler tweak");
  # kv_quant_attention_dequant is the q8/q4 KV-cache dequant path in the flash decode route (absent). All are
  # EMITTER_BLOCKED = walled, not refuted; reopen = build them.
  # NOTE: v_dot2 was REMOVED from this set 2026-07-02 — it is NOT emitter-blocked. The renderer lowers it
  # (__builtin_amdgcn_fdot2 / v_dot2_f32_f16) and it is now wired into the Q4_K decode GEMV
  # (tinygrad DECODE_Q4K_GEMV_VDOT2, dNLL +0.00013, +0.3-0.4 tok/s). The stale "renderer can't lower v_dot2"
  # belief was wrong; see candidate decode_q4k_gemv_vdot2.
  # int-dot v_dot4 GEMV is FASTER than G3 on the large FFN shape (-10%) but walled by SUBSTRATE gaps, not a
  # physical refutation. Substrate-accurate blockers (per no-kernel-writes: the fix is vocab/substrate, never a
  # hand-written fused kernel): v_dot4_lowering (renderer/matcher emits v_dot4_i8 from an integer-dot idiom, so
  # the int-dot GEMV can be a GENERATED kernel like v_dot2's pm_fdot2 — today it is an opaque custom_kernel);
  # scheduler_fuse_elementwise_into_gemv (fuse the lazy x->q8_1 quantize into the generated GEMV — impossible
  # today because the GEMV is a custom_kernel, so quantize/pack/dot are 3 separate kernels ~cancelling the win);
  # intdot_shape_selective_routing (route-manifest selects int-dot per-linear only where it wins). See candidate
  # decode_q4k_gemv_intdot_vdot4.
  # v_dot4_lowering REMOVED 2026-07-02 — it is now LOWERED (pm_intdot / __builtin_amdgcn_sudot4, tinygrad
  # d4cfc7f24; bit-exact, v_dot4 in ISA). The int-dot win now needs coalesced_int8x4_load (operands arrive packed
  # so v_dot4 has no pack-from-scalar ALU) instead.
  # scheduler_fuse_elementwise_into_gemv REMOVED 2026-07-02 — DEMONSTRATED lowered (native scheduler fusion when
  # the GEMV is a Tensor-op expression, not a custom_kernel; tinygrad c96c5235c). The int-dot's remaining gap
  # narrowed to pm_intdot_tiled_match (v_dot4 on the composed/derived-operand reduce).
  blocked_knobs=frozenset({"words_per_group!=8", "split_k", "vec_store_to_reg",
                           "fused_reduce_scale", "kv_quant_attention_dequant",
                           "intdot_shape_selective_routing", "coalesced_int8x4_load",
                           "pm_intdot_tiled_match"}),
)


@dataclass(frozen=True)
class ReachabilityRow:
  candidate_id: str            # a candidate id, or a knob token (e.g. "words_per_group!=8", "split_k")
  role: str                    # the role this classification is about ("" = roleless / global)
  cls: str                     # a Reachability value (serialized as "class")
  reason: str
  reopen: str = ""             # what would make it reachable / re-openable

  def __post_init__(self):
    if self.cls not in {r.value for r in Reachability}:
      raise ValueError(f"unknown reachability class {self.cls!r}")

  def to_json(self) -> dict[str, Any]:
    return {"candidate_id": self.candidate_id, "role": self.role, "class": self.cls,
            "reason": self.reason, "reopen": self.reopen}


@dataclass(frozen=True)
class ReachabilityReport:
  model_id: str
  target_id: str
  rows: tuple[ReachabilityRow, ...]

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA_REACHABILITY_REPORT, "model_id": self.model_id, "target_id": self.target_id,
            "rows": [r.to_json() for r in self.rows]}

  def by_class(self, cls) -> tuple[ReachabilityRow, ...]:
    v = cls.value if isinstance(cls, Reachability) else cls
    return tuple(r for r in self.rows if r.cls == v)

  def rows_for(self, candidate_id:str) -> tuple[ReachabilityRow, ...]:
    return tuple(r for r in self.rows if r.candidate_id == candidate_id)

  def reachable_ids(self) -> set[str]:
    return {r.candidate_id for r in self.rows if r.cls == Reachability.REACHABLE.value}


def _ledger_index(ledger:Iterable[LedgerEntry] | None) -> dict[str, LedgerEntry]:
  return {e.candidate_id: e for e in (ledger or ())}


def _applicable_role_quants(cand:Candidate, profile:ModelProfile) -> list[tuple[str, str]]:
  """The (role, quant) pairs present in the profile that a candidate applies to.

  Roleless candidates (diagnostics / global passes) produce a single roleless row so they stay visible
  without exploding across every role in the profile.
  """
  # gate on architecture only when it is KNOWN (dense/moe/enc-dec): a dense-only template is not offered to a
  # moe profile and vice versa (A7). An unknown_transformer profile is not filtered here — that is emit/policy's
  # fail-closed job, not template matching's.
  arch = profile.architecture_class if profile.architecture_class not in ("", None, "unknown_transformer") else None
  if not cand.roles:
    if not cand.applies_to(workload=cand.workload, architecture=arch):
      return []
    return [("", cand.quant[0] if cand.quant else "")]
  seen: list[tuple[str, str]] = []
  for r in profile.roles:
    if (cand.applies_to(workload=cand.workload, quant=r.quant, role=r.role, architecture=arch)
        and (r.role, r.quant) not in seen):
      seen.append((r.role, r.quant))
  return seen


def _classify_candidate(cand:Candidate, role:str, quant:str, target:TargetProfile,
                        caps:EmitterCapabilities, ledger_idx:dict[str, LedgerEntry],
                        reopened:set[str]) -> tuple[str, str, str]:
  # 1) the target must carry enough capability data to reason about any route
  if not target.wave_size or target.wave_size <= 0:
    return (Reachability.TARGET_INCOMPLETE.value,
            "TargetProfile is missing wave_size; cannot reason about lane/reduction topology",
            "provide wave_size (and LDS/VRAM) in the TargetProfile")
  # 1b) a descriptor-only / unsupported backend has capability data but no evaluator: routes on it are
  #     target-backend-incomplete, never promotable (audit A6).
  if getattr(target, "backend_status", "complete") != "complete":
    return (Reachability.TARGET_INCOMPLETE.value,
            f"target backend {target.target_id!r} is {getattr(target, 'backend_status', '?')} "
            f"(capability descriptor only, no evaluator yet)",
            f"build the {target.backend} backend evaluator before routing on {target.target_id!r}")
  # 2) the emitter must have a base primitive for this role/quant
  for tok in (quant, f"{role}:{quant}"):
    if tok and tok in caps.missing_primitives:
      return (Reachability.PRIMITIVE_MISSING.value,
              f"emitter has no base primitive for {tok!r}",
              f"add a {tok!r} primitive to the emitter")
  # 2b) manifest-declared capability walls (blocked_on): the candidate names the capability tokens it needs.
  #     A fully-blocked emitter knob is a lowering gap (EMITTER_BLOCKED, walled not refuted); a target capability
  #     recorded false is a missing coordination/hardware primitive (PRIMITIVE_MISSING). This makes a prose
  #     closeout a queryable edge — reachability names the exact blocker. A token that resolves to no wall means
  #     the capability is now present, so we fall through (REACHABLE = the reopen signal).
  tcaps = getattr(target, "capabilities", None) or {}
  for tok in cand.blocked_on:
    if tok in caps.fully_blocked_knobs():
      return (Reachability.EMITTER_BLOCKED.value,
              f"candidate requires emitter capability {tok!r}, which the emitter cannot lower "
              f"(search-space-incomplete) — walled, not refuted",
              f"teach the emitter to lower {tok!r}, then re-search")
    if tcaps.get(tok) is False:
      return (Reachability.PRIMITIVE_MISSING.value,
              f"candidate requires target capability {tok!r}, which {target.target_id!r} does not have",
              f"add a {tok!r} primitive/capability to {target.target_id!r}")
  # a candidate is reopened either explicitly (reopened_ids) or by a truthy REOPEN_MET_KEY in the ledger
  # entry's scope (the same key ledger.store uses to mark a do_not_retry entry's reopen_condition met).
  entry = ledger_idx.get(cand.candidate_id)
  reopen = entry.reopen_condition if entry else ""
  is_reopened = (cand.candidate_id in reopened
                 or bool(entry and entry.scope.get(REOPEN_MET_KEY)))
  # 3a) a MANIFEST-level refuted axis walls the candidate as refuted-by-ledger even with no ledger row —
  #     the manifest itself records the refutation (refuted_axis_tags), so it is not freshly reachable.
  if cand.refuted_axis_tags and not is_reopened:
    tag = cand.refuted_axis_tags[0]
    return (Reachability.REFUTED_BY_LEDGER.value,
            f"manifest marks candidate as refuted (axis {tag!r}); not reachable without new evidence",
            reopen or f"reopen when a new primitive supersedes refuted axis {tag!r}")
  # 3b) a ledger refutation with do_not_retry walls it until the reopen condition is met
  if (entry and entry.status == LedgerStatus.REFUTED.value and entry.do_not_retry and not is_reopened):
    return (Reachability.REFUTED_BY_LEDGER.value,
            f"ledger-refuted (do_not_retry) and reopen condition not met: {entry.reopen_condition!r}",
            reopen)
  return (Reachability.REACHABLE.value,
          "candidate applies to a present role/quant and is not emitter/target-blocked or ledger-refuted",
          reopen)


def _knob_rows(profile:ModelProfile, target:TargetProfile, caps:EmitterCapabilities) -> list[ReachabilityRow]:
  """Classify the knobs of the legal route grammar (reused from search.emit) against the emitter."""
  rows: list[ReachabilityRow] = []
  seen: set[tuple[str, str]] = set()
  space = emit_search_space(profile, target)
  for role_blk in space["roles"]:
    role = role_blk["role"]
    for fam in role_blk["route_families"]:
      family = fam["family"]
      for knob, values in fam["knobs"].items():
        allowed = caps.allowed_value(knob)
        if allowed is None:
          continue  # emitter can express every value of this knob
        blocked_vals = [v for v in values if str(v) != str(allowed)]
        if not blocked_vals or (role, knob) in seen:
          continue
        seen.add((role, knob))
        rows.append(ReachabilityRow(
          candidate_id=f"{knob}!={allowed}", role=role, cls=Reachability.EMITTER_BLOCKED.value,
          reason=(f"route family {family!r} exposes {knob}={values} but the emitter can only lower "
                  f"{knob}={allowed}; values {blocked_vals} are a non-exposed topology knob"),
          reopen=f"teach the emitter to lower {knob} != {allowed}"))
  # knobs the grammar does not represent at all (e.g. split-K): search-space-incomplete, walled as
  # EMITTER_BLOCKED — explicitly NOT a refutation.
  for knob in caps.fully_blocked_knobs():
    rows.append(ReachabilityRow(
      candidate_id=knob, role="", cls=Reachability.EMITTER_BLOCKED.value,
      reason=(f"{knob!r} is not represented in the route grammar (search-space-incomplete): the emitter "
              f"cannot express it, so it is walled — this is NOT a refutation"),
      reopen=f"add {knob} to the route grammar and emitter, then re-search"))
  return rows


def classify_reachability(profile:ModelProfile, target:TargetProfile, candidates:Iterable[Candidate], *,
                          ledger:Iterable[LedgerEntry] | None = None,
                          emitter_capabilities:EmitterCapabilities | None = None,
                          reopened_ids:Iterable[str] | None = None) -> ReachabilityReport:
  """Build a ReachabilityReport: one row per applicable (candidate, role) plus per-knob grammar rows.

  `emitter_capabilities` defaults to DEFAULT_EMITTER_CAPABILITIES (words_per_group!=8 and split_k blocked).
  `ledger` supplies durable refutations; `reopened_ids` marks candidates whose reopen condition is met.
  """
  caps = emitter_capabilities or DEFAULT_EMITTER_CAPABILITIES
  ledger_idx = _ledger_index(ledger)
  reopened = set(reopened_ids or ())
  rows: list[ReachabilityRow] = []
  for cand in candidates:
    for role, quant in _applicable_role_quants(cand, profile):
      cls, reason, reopen = _classify_candidate(cand, role, quant, target, caps, ledger_idx, reopened)
      rows.append(ReachabilityRow(cand.candidate_id, role, cls, reason, reopen))
  rows.extend(_knob_rows(profile, target, caps))
  return ReachabilityReport(profile.model_id, target.target_id, tuple(rows))
