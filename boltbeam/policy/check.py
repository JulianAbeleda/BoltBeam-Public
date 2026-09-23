"""Route-policy guard (audit-brain-build-scope BB7).

Reads the ledger's collapsed current state (a `LedgerStore` or its `summary()` dict) plus the candidate
manifest and returns the list of policy violations. It never mutates the ledger. `assert_policy` is the
strict wrapper that raises. Rules enforced:

  1. every promoted candidate has a rollback (looked up in the manifest);
  2. every refuted candidate carries evidence AND a reopen_condition;
  3. a search-space-incomplete candidate is not also listed as refuted (in its history or the manifest);
  4. a route policy must not select a refuted candidate (optional selected_candidate_ids).
"""
from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Union

from boltbeam.vocab import LedgerStatus
from boltbeam.ledger.model import LedgerEntry
from boltbeam.manifest import Candidate, load_candidates

# rule ids (one authority for the guard's vocabulary)
RULE_PROMOTED_WITHOUT_ROLLBACK = "promoted_without_rollback"
RULE_REFUTED_MISSING_EVIDENCE = "refuted_missing_evidence"
RULE_SSI_ALSO_REFUTED = "search_space_incomplete_also_refuted"
RULE_ROUTE_SELECTS_REFUTED = "route_selects_refuted"


@dataclass(frozen=True)
class Violation:
  rule: str
  candidate_id: str
  detail: str

  def to_json(self) -> dict[str, Any]:
    return asdict(self)


class PolicyViolation(AssertionError):
  """Raised by assert_policy when the ledger/route policy is not clean."""

  def __init__(self, violations:list[Violation]):
    self.violations = violations
    super().__init__("; ".join(f"[{v.rule}] {v.candidate_id}: {v.detail}" for v in violations))


def _latest_entries(source:Union["object", dict[str, Any]]) -> dict[str, LedgerEntry]:
  if hasattr(source, "latest_entries"):
    return source.latest_entries()
  if isinstance(source, dict) and "candidates" in source:
    return {cid: LedgerEntry.from_json(v) for cid, v in source["candidates"].items()}
  raise TypeError("check_policy expects a LedgerStore or a summary() dict with a 'candidates' map")


def _manifest_index(manifest:Union[None, str, pathlib.Path, Iterable[Candidate]]) -> dict[str, Candidate]:
  if manifest is None:
    cands:Iterable[Candidate] = load_candidates()
  elif isinstance(manifest, (str, pathlib.Path)):
    cands = load_candidates(manifest)
  else:
    cands = manifest
  return {c.candidate_id: c for c in cands}


def check_policy(ledger_summary_or_store:Union["object", dict[str, Any]],
                 manifest:Union[None, str, pathlib.Path, Iterable[Candidate]] = None,
                 selected_candidate_ids:Iterable[str] | None = None) -> list[Violation]:
  latest = _latest_entries(ledger_summary_or_store)
  cand_by_id = _manifest_index(manifest)
  # history is only available on a store; used to catch a candidate refuted earlier then re-listed as SSI
  history = ledger_summary_or_store.history() if hasattr(ledger_summary_or_store, "history") else list(latest.values())
  violations:list[Violation] = []

  for cid, entry in latest.items():
    cand = cand_by_id.get(cid)

    if entry.status == LedgerStatus.PROMOTED.value:
      if cand is None:
        violations.append(Violation(RULE_PROMOTED_WITHOUT_ROLLBACK, cid, "promoted candidate is absent from the manifest"))
      elif not cand.rollback:
        violations.append(Violation(RULE_PROMOTED_WITHOUT_ROLLBACK, cid, "promoted candidate has no rollback in the manifest"))

    if entry.status == LedgerStatus.REFUTED.value:
      if not entry.evidence:
        violations.append(Violation(RULE_REFUTED_MISSING_EVIDENCE, cid, "refuted candidate carries no evidence"))
      if not entry.reopen_condition:
        violations.append(Violation(RULE_REFUTED_MISSING_EVIDENCE, cid, "refuted candidate has no reopen_condition"))

    if entry.status == LedgerStatus.SEARCH_SPACE_INCOMPLETE.value:
      refuted_in_history = any(h.candidate_id == cid and h.status == LedgerStatus.REFUTED.value for h in history)
      refuted_in_manifest = bool(cand and cand.refuted_axis_tags)
      if refuted_in_history or refuted_in_manifest:
        where = "ledger history" if refuted_in_history else "manifest"
        violations.append(Violation(RULE_SSI_ALSO_REFUTED, cid,
                                    f"search-space-incomplete candidate is also listed as refuted ({where})"))

  for cid in (selected_candidate_ids or ()):
    entry = latest.get(cid)
    cand = cand_by_id.get(cid)
    if (entry is not None and entry.status == LedgerStatus.REFUTED.value) or (cand is not None and cand.refuted_axis_tags):
      violations.append(Violation(RULE_ROUTE_SELECTS_REFUTED, cid, "route policy selects a refuted candidate"))

  return violations


def assert_policy(ledger_summary_or_store:Union["object", dict[str, Any]],
                  manifest:Union[None, str, pathlib.Path, Iterable[Candidate]] = None,
                  selected_candidate_ids:Iterable[str] | None = None) -> None:
  violations = check_policy(ledger_summary_or_store, manifest=manifest, selected_candidate_ids=selected_candidate_ids)
  if violations:
    raise PolicyViolation(violations)
