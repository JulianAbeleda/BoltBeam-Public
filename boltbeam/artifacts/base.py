"""Normalized evidence: the ONE data shape every evaluator consumes (audit-brain-build-scope BB1).

Adapters (boltbeam/artifacts/tinygrad.py) turn raw producer artifacts into NormalizedEvidence. Evaluators read
normalized rows only — never raw tinygrad-specific fields. Missing required fields raise AdapterIncomplete
(caught by the ingest layer and turned into a verdict), never a bare crash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from boltbeam.core.canonical import sha256_hex, sha256_json as _sha256_json, prefixed, pretty_json
from boltbeam.vocab import RowKind, Workload, Confidence, SCHEMA_NORMALIZED_EVIDENCE


class AdapterIncomplete(Exception):
  """Raised by an adapter when a recognized artifact is missing fields needed to normalize it safely.

  The ingest layer converts this into an `adapter-incomplete` outcome with the reason preserved — it must never
  be a silent crash or a guessed value.
  """
  def __init__(self, reason:str, *, path:str | None = None):
    self.reason, self.path = reason, path
    super().__init__(reason)


class UnsupportedArtifact(Exception):
  """Raised when no adapter recognizes the artifact kind at all."""
  def __init__(self, reason:str, *, path:str | None = None):
    self.reason, self.path = reason, path
    super().__init__(reason)


def sha256_bytes(data:bytes) -> str:
  return prefixed(sha256_hex(data))


def sha256_json(obj:Any) -> str:
  return prefixed(_sha256_json(obj))


@dataclass(frozen=True)
class EvidenceSource:
  """Where a piece of evidence came from. `path` is repo-relative for BoltBeam artifacts, absolute only for
  external tinygrad provenance during migration; `fingerprint` is the sha256 of the raw artifact bytes."""
  producer: str                       # e.g. "tinygrad"
  tool: str                           # e.g. "qk_decode_role_attribution_modular.py"
  path: str
  fingerprint: str                    # sha256: of raw bytes
  raw_summary: dict[str, Any] = field(default_factory=dict)   # only kept when the evaluator genuinely needs it

  def to_json(self) -> dict[str, Any]:
    d = {"producer": self.producer, "tool": self.tool, "path": self.path, "fingerprint": self.fingerprint}
    if self.raw_summary: d["raw_summary"] = self.raw_summary
    return d


@dataclass(frozen=True)
class EvidenceFlags:
  """Tri-state gate flags: True=proven, False=proven-absent, None=unknown (do NOT infer from missing)."""
  route_bound: bool | None = None
  token_match: bool | None = None
  deterministic: bool | None = None
  hidden_fallback: bool | None = None

  def to_json(self) -> dict[str, Any]:
    return {k: v for k, v in asdict(self).items() if v is not None}

  @staticmethod
  def from_json(d:dict[str, Any] | None) -> "EvidenceFlags":
    d = d or {}
    return EvidenceFlags(d.get("route_bound"), d.get("token_match"), d.get("deterministic"), d.get("hidden_fallback"))


@dataclass(frozen=True)
class EvidenceRow:
  """One normalized measurement. Absent structural fields stay None; they are never fabricated."""
  kind: str                           # a RowKind value
  metric: str                         # e.g. "wall_share", "tok_s", "host_sync_pct", "lag_pct"
  value: float
  unit: str                           # e.g. "fraction", "tok/s", "%", "GB/s"
  role: str | None = None
  quant: str | None = None
  shape: list[int] | None = None
  context: int | None = None
  confidence: str = Confidence.MEASURED.value
  extra: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    if self.kind not in {k.value for k in RowKind}:
      raise AdapterIncomplete(f"unknown row kind {self.kind!r}")

  def to_json(self) -> dict[str, Any]:
    d: dict[str, Any] = {"kind": self.kind, "metric": self.metric, "value": self.value, "unit": self.unit,
                         "confidence": self.confidence}
    for k in ("role", "quant", "shape", "context"):
      v = getattr(self, k)
      if v is not None: d[k] = v
    if self.extra: d["extra"] = self.extra
    return d

  @staticmethod
  def from_json(d:dict[str, Any]) -> "EvidenceRow":
    return EvidenceRow(kind=d["kind"], metric=d["metric"], value=d["value"], unit=d["unit"],
                       role=d.get("role"), quant=d.get("quant"), shape=d.get("shape"), context=d.get("context"),
                       confidence=d.get("confidence", Confidence.MEASURED.value), extra=d.get("extra", {}))


# required top-level fields; absence => AdapterIncomplete, not a crash
_REQUIRED = ("model_id", "target_id", "workload")


@dataclass(frozen=True)
class NormalizedEvidence:
  model_id: str
  target_id: str
  workload: str                       # a Workload value
  source: EvidenceSource
  rows: tuple[EvidenceRow, ...]
  contexts: tuple[int, ...] = ()
  flags: EvidenceFlags = field(default_factory=EvidenceFlags)

  def __post_init__(self):
    for name in _REQUIRED:
      if not getattr(self, name):
        raise AdapterIncomplete(f"missing required field {name!r}")
    if self.workload not in {w.value for w in Workload}:
      raise AdapterIncomplete(f"unknown workload {self.workload!r}")

  @property
  def evidence_id(self) -> str:
    """Path-independent semantic id: stable across checkouts / relocations."""
    return sha256_json({"model_id": self.model_id, "target_id": self.target_id, "workload": self.workload,
                        "contexts": list(self.contexts), "rows": [r.to_json() for r in self.rows],
                        "flags": self.flags.to_json(), "source_fingerprint": self.source.fingerprint})

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA_NORMALIZED_EVIDENCE, "evidence_id": self.evidence_id, "source": self.source.to_json(),
            "model_id": self.model_id, "target_id": self.target_id, "workload": self.workload,
            "contexts": list(self.contexts), "rows": [r.to_json() for r in self.rows], "flags": self.flags.to_json()}

  @staticmethod
  def from_json(d:dict[str, Any]) -> "NormalizedEvidence":
    if d.get("schema") != SCHEMA_NORMALIZED_EVIDENCE:
      raise AdapterIncomplete(f"expected schema {SCHEMA_NORMALIZED_EVIDENCE}, got {d.get('schema')!r}")
    s = d["source"]
    src = EvidenceSource(s["producer"], s["tool"], s["path"], s["fingerprint"], s.get("raw_summary", {}))
    return NormalizedEvidence(model_id=d["model_id"], target_id=d["target_id"], workload=d["workload"], source=src,
                              rows=tuple(EvidenceRow.from_json(r) for r in d["rows"]),
                              contexts=tuple(d.get("contexts", ())), flags=EvidenceFlags.from_json(d.get("flags")))


def write_evidence(ev:NormalizedEvidence, path:str) -> None:
  import pathlib
  p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(pretty_json(ev.to_json()))


def read_evidence(path:str) -> NormalizedEvidence:
  import pathlib
  return NormalizedEvidence.from_json(json.loads(pathlib.Path(path).read_text()))
