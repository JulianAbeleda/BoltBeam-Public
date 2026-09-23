"""Typed decision + ledger objects (audit-brain-build-scope 'Candidate Decision' / 'Ledger Entry').

CandidateDecision is what the evaluator emits. LedgerEntry is the durable record. Both use the centralized
vocab enums — no free-form verdict/status strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from boltbeam.vocab import (Verdict, Tier, Workload, LedgerStatus, Guardrail, GuardrailState,
                            SCHEMA_CANDIDATE_DECISION, SCHEMA_ROUTE_LEDGER_ENTRY)


@dataclass(frozen=True)
class EvidenceRef:
  """A citation: evidence id + the byte fingerprint + a one-sentence claim of what it proves."""
  evidence_id: str
  path: str
  kind: str
  fingerprint: str
  claim: str

  def to_json(self) -> dict[str, Any]:
    return asdict(self)

  @staticmethod
  def from_json(d:dict[str, Any]) -> "EvidenceRef":
    return EvidenceRef(d["evidence_id"], d["path"], d["kind"], d["fingerprint"], d["claim"])


def _valid_guardrails(g:dict[str, str]) -> dict[str, str]:
  allowed_keys = {x.value for x in Guardrail}
  allowed_vals = {x.value for x in GuardrailState}
  for k, v in g.items():
    if k not in allowed_keys: raise ValueError(f"unknown guardrail {k!r}")
    if v not in allowed_vals: raise ValueError(f"unknown guardrail state {v!r} for {k!r}")
  return g


@dataclass(frozen=True)
class CandidateDecision:
  candidate_id: str
  model_id: str
  target_id: str
  workload: str
  verdict: str                                  # a Verdict value
  tier: str = Tier.NONE.value
  reason: str = ""
  evidence: tuple[EvidenceRef, ...] = ()
  rollback: dict[str, str] = field(default_factory=dict)
  next_action: str = ""
  guardrails: dict[str, str] = field(default_factory=dict)

  def __post_init__(self):
    if self.verdict not in {v.value for v in Verdict}: raise ValueError(f"unknown verdict {self.verdict!r}")
    if self.tier not in {t.value for t in Tier}: raise ValueError(f"unknown tier {self.tier!r}")
    if self.workload not in {w.value for w in Workload}: raise ValueError(f"unknown workload {self.workload!r}")
    _valid_guardrails(self.guardrails)
    # invariant: a promotion must carry a rollback (audit-brain-build-scope BB5 "no candidate can promote without rollback")
    if self.verdict == Verdict.PROMOTE.value and not self.rollback:
      raise ValueError(f"candidate {self.candidate_id!r} promotes without a rollback")

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA_CANDIDATE_DECISION, "candidate_id": self.candidate_id, "model_id": self.model_id,
            "target_id": self.target_id, "workload": self.workload, "verdict": self.verdict, "tier": self.tier,
            "reason": self.reason, "evidence": [e.to_json() for e in self.evidence], "rollback": self.rollback,
            "next_action": self.next_action, "guardrails": self.guardrails}

  @staticmethod
  def from_json(d:dict[str, Any]) -> "CandidateDecision":
    if d.get("schema") != SCHEMA_CANDIDATE_DECISION:
      raise ValueError(f"expected schema {SCHEMA_CANDIDATE_DECISION}, got {d.get('schema')!r}")
    return CandidateDecision(candidate_id=d["candidate_id"], model_id=d["model_id"], target_id=d["target_id"],
                             workload=d["workload"], verdict=d["verdict"], tier=d.get("tier", Tier.NONE.value),
                             reason=d.get("reason", ""), evidence=tuple(EvidenceRef.from_json(e) for e in d.get("evidence", [])),
                             rollback=d.get("rollback", {}), next_action=d.get("next_action", ""),
                             guardrails=d.get("guardrails", {}))


# how a decision verdict maps onto a durable ledger status
_VERDICT_TO_STATUS = {
  Verdict.PROMOTE.value: LedgerStatus.PROMOTED.value,
  Verdict.REFUTE.value: LedgerStatus.REFUTED.value,
  Verdict.DEFER.value: LedgerStatus.DEFERRED.value,
  Verdict.SEARCH_SPACE_INCOMPLETE.value: LedgerStatus.SEARCH_SPACE_INCOMPLETE.value,
  Verdict.CANDIDATE.value: LedgerStatus.CANDIDATE.value,
}


def status_for_verdict(verdict:str) -> str | None:
  """The ledger status a verdict implies, or None for verdicts that are not durable (diagnostic/inconclusive/adapter)."""
  return _VERDICT_TO_STATUS.get(verdict)


@dataclass(frozen=True)
class LedgerEntry:
  candidate_id: str
  status: str                                   # a LedgerStatus value
  scope: dict[str, Any]                         # e.g. {"model_family","quant","target"}
  evidence: tuple[EvidenceRef, ...] = ()
  reopen_condition: str = ""
  do_not_retry: bool = False
  imported: bool = False                        # True = historical seed, not freshly measured by BoltBeam

  def __post_init__(self):
    if self.status not in {s.value for s in LedgerStatus}: raise ValueError(f"unknown ledger status {self.status!r}")
    # a refuted/search-space-incomplete row must be reopenable (no permanent dead ends without a condition)
    if self.status in (LedgerStatus.REFUTED.value, LedgerStatus.SEARCH_SPACE_INCOMPLETE.value) and not self.reopen_condition:
      raise ValueError(f"{self.status} entry {self.candidate_id!r} needs a reopen_condition")

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA_ROUTE_LEDGER_ENTRY, "candidate_id": self.candidate_id, "status": self.status,
            "scope": self.scope, "evidence": [e.to_json() for e in self.evidence],
            "reopen_condition": self.reopen_condition, "do_not_retry": self.do_not_retry, "imported": self.imported}

  @staticmethod
  def from_json(d:dict[str, Any]) -> "LedgerEntry":
    if d.get("schema") != SCHEMA_ROUTE_LEDGER_ENTRY:
      raise ValueError(f"expected schema {SCHEMA_ROUTE_LEDGER_ENTRY}, got {d.get('schema')!r}")
    return LedgerEntry(candidate_id=d["candidate_id"], status=d["status"], scope=d.get("scope", {}),
                       evidence=tuple(EvidenceRef.from_json(e) for e in d.get("evidence", [])),
                       reopen_condition=d.get("reopen_condition", ""), do_not_retry=d.get("do_not_retry", False),
                       imported=d.get("imported", False))

  @staticmethod
  def from_decision(dec:CandidateDecision, *, scope:dict[str, Any], reopen_condition:str = "",
                    do_not_retry:bool = False, imported:bool = False) -> "LedgerEntry":
    status = status_for_verdict(dec.verdict)
    if status is None:
      raise ValueError(f"verdict {dec.verdict!r} is not a durable ledger status")
    return LedgerEntry(candidate_id=dec.candidate_id, status=status, scope=scope, evidence=dec.evidence,
                       reopen_condition=reopen_condition or (dec.next_action if status != LedgerStatus.PROMOTED.value else ""),
                       do_not_retry=do_not_retry, imported=imported)
