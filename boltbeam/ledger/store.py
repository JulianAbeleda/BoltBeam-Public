"""Append-only route/candidate ledger store (audit-brain-build-scope BB6).

History is durable: every add appends one JSONL line and never rewrites the past. The `summary()` is a
derived view that collapses to the LATEST entry per candidate_id, so a duplicate candidate_id updates the
summary while the JSONL keeps the full trail. Both `CandidateDecision` (converted via
`LedgerEntry.from_decision`) and a ready-made `LedgerEntry` are accepted.

Reopen convention: a do_not_retry entry stays suppressed until its reopen condition is marked met. Because
`LedgerEntry` is a locked shape, "met" is signalled inside the free-form `scope` dict with the key
`reopen_met` (truthy) rather than by inventing a new field.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Union

from boltbeam.vocab import SCHEMA_ROUTE_LEDGER_SUMMARY
from boltbeam.ledger.model import CandidateDecision, LedgerEntry

# scope key that marks a do_not_retry entry's reopen_condition as satisfied. Exported so other modules
# (e.g. search.reachability) reference the SAME key rather than re-inventing the "reopen_met" literal.
REOPEN_MET_KEY = "reopen_met"


def _reopen_met(entry:LedgerEntry) -> bool:
  return bool(entry.scope.get(REOPEN_MET_KEY))


def _scope_for_decision(dec:CandidateDecision) -> dict[str, Any]:
  """Durable scope derived from the decision's own facts (model / target / workload)."""
  return {"model": dec.model_id, "target": dec.target_id, "workload": dec.workload}


class LedgerStore:
  """Append-only JSONL of LedgerEntry rows with a collapsed current-state summary."""

  def __init__(self, jsonl_path:Union[str, pathlib.Path]):
    self.path = pathlib.Path(jsonl_path)

  # ---- write -------------------------------------------------------------------------------------------
  def add(self, entry_or_decision:Union[LedgerEntry, CandidateDecision], *, scope:dict[str, Any] | None = None,
          reopen_condition:str = "", do_not_retry:bool = False, imported:bool = False) -> LedgerEntry:
    """Append one entry. A CandidateDecision is converted; a LedgerEntry is stored as-is (extra kwargs ignored)."""
    if isinstance(entry_or_decision, CandidateDecision):
      dec = entry_or_decision
      entry = LedgerEntry.from_decision(dec, scope=scope if scope is not None else _scope_for_decision(dec),
                                        reopen_condition=reopen_condition, do_not_retry=do_not_retry,
                                        imported=imported)
    elif isinstance(entry_or_decision, LedgerEntry):
      entry = entry_or_decision
    else:
      raise TypeError(f"add() expects LedgerEntry or CandidateDecision, got {type(entry_or_decision).__name__}")
    self._append(entry)
    return entry

  def _append(self, entry:LedgerEntry) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with self.path.open("a", encoding="utf-8") as fh:
      fh.write(json.dumps(entry.to_json(), sort_keys=True) + "\n")

  # ---- read --------------------------------------------------------------------------------------------
  def history(self) -> list[LedgerEntry]:
    """Every appended entry, oldest first. Empty if the file does not exist yet."""
    if not self.path.exists(): return []
    out:list[LedgerEntry] = []
    for line in self.path.read_text(encoding="utf-8").splitlines():
      line = line.strip()
      if not line: continue
      out.append(LedgerEntry.from_json(json.loads(line)))
    return out

  def latest_entries(self) -> dict[str, LedgerEntry]:
    """Collapse history to the latest entry per candidate_id (later appends win)."""
    latest:dict[str, LedgerEntry] = {}
    for e in self.history():
      latest[e.candidate_id] = e
    return latest

  def summary(self) -> dict[str, Any]:
    latest = self.latest_entries()
    return {"schema": SCHEMA_ROUTE_LEDGER_SUMMARY, "count": len(latest),
            "candidates": {cid: e.to_json() for cid, e in latest.items()}}

  def recommendations_suppressed(self) -> set[str]:
    """candidate_ids whose latest entry is do_not_retry and whose reopen_condition is not marked met."""
    return {cid for cid, e in self.latest_entries().items() if e.do_not_retry and not _reopen_met(e)}


def add_decision_cli(decision_path:Union[str, pathlib.Path], jsonl_path:Union[str, pathlib.Path], *,
                     scope:dict[str, Any] | None = None, reopen_condition:str = "",
                     do_not_retry:bool = False) -> dict[str, Any]:
  """CLI helper: read a CandidateDecision JSON, append it to the ledger, return the stored entry's JSON."""
  dec = CandidateDecision.from_json(json.loads(pathlib.Path(decision_path).read_text(encoding="utf-8")))
  entry = LedgerStore(jsonl_path).add(dec, scope=scope, reopen_condition=reopen_condition, do_not_retry=do_not_retry)
  return entry.to_json()
