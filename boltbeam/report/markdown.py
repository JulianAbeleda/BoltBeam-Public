"""Deterministic human report (audit-brain-build-scope BB11).

This module turns durable *decisions* and *ledger* state into a markdown report a human can read. It reads
ONLY the typed CandidateDecision / LedgerEntry objects (and an optional ceiling dict) — it never parses raw
tinygrad artifacts. That keeps report generation orthogonal to artifact ingestion: the same report shape holds
regardless of where the evidence came from.

Two hard properties:
  * every claim line cites an evidence id / path (so no line asserts a fact without a fingerprint behind it);
  * the output is deterministic — stable ordering, no timestamps — so two calls on the same inputs are byte
    identical and the report can be committed / diffed.
"""
from __future__ import annotations

from typing import Any, Iterable

from boltbeam.vocab import Verdict, LedgerStatus
from boltbeam.ledger.model import CandidateDecision, LedgerEntry, EvidenceRef


# status -> display title, in the fixed order the routes table renders them. search-space-incomplete is
# deliberately excluded here because it has its own dedicated "missing knobs" section below.
_STATUS_TITLES: tuple[tuple[str, str], ...] = (
  (LedgerStatus.PROMOTED.value, "Promoted"),
  (LedgerStatus.REFUTED.value, "Refuted"),
  (LedgerStatus.DEFERRED.value, "Deferred"),
  (LedgerStatus.CANDIDATE.value, "Candidate"),
)
_STATUS_RANK = {status: i for i, (status, _title) in enumerate(_STATUS_TITLES)}


def _as_entries(ledger_summary:Any) -> tuple[LedgerEntry, ...]:
  """Normalize the ledger_summary argument to a tuple of LedgerEntry.

  Accepts: None, a single LedgerEntry, any object exposing `.entries`, a dict with an "entries" key, or any
  iterable of LedgerEntry. This keeps the public surface boring — callers pass whatever ledger view they hold.
  """
  if ledger_summary is None: return ()
  if isinstance(ledger_summary, LedgerEntry): return (ledger_summary,)
  if hasattr(ledger_summary, "entries"): return tuple(ledger_summary.entries)
  if isinstance(ledger_summary, dict): return tuple(ledger_summary.get("entries", ()))
  return tuple(ledger_summary)


def _cite(evidence:tuple[EvidenceRef, ...]) -> str:
  """Render an evidence citation (id + path) for a claim line. Every claim line ends with one of these."""
  if not evidence: return "_(no evidence)_"
  return "; ".join(f"`{e.evidence_id}` (`{e.path}`)" for e in evidence)


def _scope_str(scope:dict[str, Any]) -> str:
  if not scope: return "_(no scope)_"
  return ", ".join(f"{k}={scope[k]}" for k in sorted(scope))


def _rollback_str(rollback:dict[str, str]) -> str:
  return " ".join(f"{k}={rollback[k]}" for k in sorted(rollback))


def _routes_section(entries:tuple[LedgerEntry, ...]) -> list[str]:
  lines = ["## Current Routes", ""]
  for status, title in _STATUS_TITLES:
    rows = sorted((e for e in entries if e.status == status), key=lambda e: e.candidate_id)
    lines += [f"### {title}", ""]
    if not rows:
      lines += ["- _none_", ""]
      continue
    for e in rows:
      tail = f" — reopen when {e.reopen_condition}" if e.reopen_condition else ""
      lines.append(f"- `{e.candidate_id}` — scope: {_scope_str(e.scope)}{tail} — evidence {_cite(e.evidence)}")
    lines.append("")
  return lines


def _hottest_buckets_section(decisions:tuple[CandidateDecision, ...]) -> list[str]:
  # Collect each evidence claim carried by a decision. These are the measured buckets the decisions rest on.
  seen: set[tuple[str, str, str, str]] = set()
  rows: list[tuple[str, str, str, str, str]] = []
  for dec in decisions:
    for ref in dec.evidence:
      key = (ref.kind, ref.evidence_id, ref.claim, dec.candidate_id)
      if key in seen: continue
      seen.add(key)
      rows.append((ref.kind, ref.evidence_id, ref.claim, ref.path, dec.candidate_id))
  rows.sort()
  lines = ["## Hottest Measured Buckets", ""]
  if not rows:
    lines += ["- _none_", ""]
    return lines
  for kind, ev_id, claim, path, cid in rows:
    lines.append(f"- {claim} — kind `{kind}`, candidate `{cid}` — evidence `{ev_id}` (`{path}`)")
  lines.append("")
  return lines


def _next_actions_section(decisions:tuple[CandidateDecision, ...], entries:tuple[LedgerEntry, ...]) -> list[str]:
  lines = ["## Next Actions", ""]
  items: list[str] = []
  for dec in sorted(decisions, key=lambda d: d.candidate_id):
    if dec.next_action:
      items.append(f"- `{dec.candidate_id}`: {dec.next_action} — evidence {_cite(dec.evidence)}")
  for e in sorted(entries, key=lambda e: e.candidate_id):
    if e.reopen_condition:
      items.append(f"- `{e.candidate_id}` ({e.status}): reopen when {e.reopen_condition} — evidence {_cite(e.evidence)}")
  items = sorted(dict.fromkeys(items))
  lines += (items if items else ["- _none_"]) + [""]
  return lines


def _missing_evidence_section(decisions:tuple[CandidateDecision, ...]) -> list[str]:
  lines = ["## Missing Evidence", ""]
  rows = sorted((d for d in decisions if d.verdict == Verdict.INCONCLUSIVE.value), key=lambda d: d.candidate_id)
  if not rows:
    lines += ["- _none_", ""]
    return lines
  for dec in rows:
    reason = dec.reason or "evidence insufficient or noisy"
    lines.append(f"- `{dec.candidate_id}`: {reason} — evidence {_cite(dec.evidence)}")
  lines.append("")
  return lines


def _search_space_section(decisions:tuple[CandidateDecision, ...], entries:tuple[LedgerEntry, ...]) -> list[str]:
  lines = ["## Search-Space-Incomplete Knobs", ""]
  items: list[str] = []
  for dec in sorted(decisions, key=lambda d: d.candidate_id):
    if dec.verdict == Verdict.SEARCH_SPACE_INCOMPLETE.value:
      knob = dec.next_action or dec.reason or "unnamed knob"
      items.append(f"- `{dec.candidate_id}`: needs {knob} — evidence {_cite(dec.evidence)}")
  for e in sorted(entries, key=lambda e: e.candidate_id):
    if e.status == LedgerStatus.SEARCH_SPACE_INCOMPLETE.value:
      knob = e.reopen_condition or "unnamed knob"
      items.append(f"- `{e.candidate_id}`: needs {knob} — evidence {_cite(e.evidence)}")
  items = sorted(dict.fromkeys(items))
  lines += (items if items else ["- _none_"]) + [""]
  return lines


def _rollback_section(decisions:tuple[CandidateDecision, ...]) -> list[str]:
  lines = ["## Rollback Commands", ""]
  rows = sorted((d for d in decisions if d.rollback), key=lambda d: d.candidate_id)
  if not rows:
    lines += ["- _none_", ""]
    return lines
  for dec in rows:
    lines += [f"- `{dec.candidate_id}` (verdict `{dec.verdict}`) — evidence {_cite(dec.evidence)}", "",
              "  ```bash", f"  {_rollback_str(dec.rollback)}", "  ```", ""]
  return lines


def _ceiling_section(ceiling:Any) -> list[str]:
  lines = ["## Ceiling", ""]
  if isinstance(ceiling, dict):
    for k in sorted(ceiling):
      v = ceiling[k]
      if isinstance(v, (str, int, float, bool)) or v is None:
        lines.append(f"- {k}: {v}")
  if len(lines) == 2:
    lines.append("- _no scalar ceiling fields_")
  lines.append("")
  return lines


def render_report(ledger_summary:Any, decisions:Iterable[CandidateDecision] | None=None,
                  ceiling:Any=None) -> str:
  """Render a deterministic markdown audit report from ledger + decisions only.

  Args:
    ledger_summary: durable ledger view — a LedgerSummary-like object, a dict with "entries", or an iterable
                    of LedgerEntry.
    decisions: the CandidateDecision objects behind this cycle (for buckets, next actions, missing evidence,
               search-space knobs, and rollback commands). Optional.
    ceiling: an optional ceiling-report dict; when present a small ceiling section is appended.

  The output is stable-ordered and timestamp-free, so two calls on identical inputs are byte identical.
  """
  entries = _as_entries(ledger_summary)
  decs = tuple(decisions or ())
  lines: list[str] = [
    "# BoltBeam Audit Report",
    "",
    "_Deterministic report from ledger + decisions. No timestamps; ordering is stable. Every claim cites an evidence id/path._",
    "",
  ]
  lines += _routes_section(entries)
  lines += _hottest_buckets_section(decs)
  lines += _next_actions_section(decs, entries)
  lines += _missing_evidence_section(decs)
  lines += _search_space_section(decs, entries)
  lines += _rollback_section(decs)
  if ceiling is not None:
    lines += _ceiling_section(ceiling)
  return "\n".join(lines).rstrip() + "\n"
