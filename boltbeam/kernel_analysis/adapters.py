"""KA-2: schema-driven ingestion.

Providers either emit canonical `boltbeam.kernel_evidence.v1` or register an adapter that normalizes
their artifact into one. Detection is by explicit schema first; structural detection is reserved for
documented legacy formats. Paths are provenance, never identity — fragments are joined by identity
(experiment/candidate/binary/system/session), never by filename or newest timestamp. Contradictory
fragments are retained as blockers on the affected decisions rather than silently merged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from boltbeam.artifacts.base import EvidenceSource, sha256_json
from boltbeam.kernel_analysis.model import (
  CandidateModel, FactBlocker, IdentityModel, KernelEvidence, KernelStages, WorkloadModel,
)
from boltbeam.vocab import SCHEMA_KERNEL_EVIDENCE

OK, UNSUPPORTED, INVALID = "ok", "unsupported", "invalid"


@dataclass(frozen=True)
class EvidenceFragment:
  """A partial contribution toward one KernelEvidence. `payload` mirrors KernelEvidence JSON sections
  (candidate/identity/workload/stages/correctness/resources/structure/timing/health)."""
  payload: dict[str, Any]
  provenance: tuple[EvidenceSource, ...] = ()
  blockers: tuple[FactBlocker, ...] = ()

  def identity_keys(self) -> dict[str, Any]:
    ident = self.payload.get("identity", {})
    cand = self.payload.get("candidate", {})
    return {
      "experiment_id": ident.get("experiment_id"),
      "candidate_id": cand.get("candidate_id"),
      "candidate_hash": cand.get("candidate_hash"),
      "binary_hash": ident.get("binary_hash"),
      "system_snapshot_id": ident.get("system_snapshot_id"),
      "session_id": ident.get("session_id"),
      "workload_sig": sha256_json(self.payload["workload"]) if self.payload.get("workload") else None,
    }


@dataclass(frozen=True)
class AdaptResult:
  status: str
  evidence: KernelEvidence | None = None
  fragment: EvidenceFragment | None = None
  fragments: tuple[EvidenceFragment, ...] = ()   # for multi-row artifacts (e.g. a per-kernel trace)
  schema: str | None = None
  fields: tuple[str, ...] = ()
  message: str = ""

  def to_json(self) -> dict[str, Any]:
    out = {"status": self.status, "schema": self.schema, "message": self.message}
    if self.fields:
      out["fields"] = list(self.fields)
    return out


Adapter = Callable[[Mapping[str, Any], EvidenceSource], AdaptResult]
StructuralDetector = Callable[[Mapping[str, Any]], bool]

_ADAPTERS: dict[str, Adapter] = {}
_STRUCTURAL: list[tuple[str, StructuralDetector]] = []


def register_kernel_adapter(schema: str, adapter: Adapter, *,
                            structural: StructuralDetector | None = None) -> None:
  if not schema:
    raise ValueError("adapter schema is required")
  if schema in _ADAPTERS:
    raise ValueError(f"adapter already registered for schema {schema!r}")
  _ADAPTERS[schema] = adapter
  if structural is not None:
    _STRUCTURAL.append((schema, structural))


def _reset_adapters_for_test() -> None:
  _ADAPTERS.clear()
  _STRUCTURAL.clear()
  _register_builtin()


def detect_kernel_artifact(mapping: Mapping[str, Any]) -> str | None:
  """Explicit `schema` wins; otherwise fall back to documented structural detectors."""
  if not isinstance(mapping, Mapping):
    return None
  schema = mapping.get("schema")
  if isinstance(schema, str) and schema:
    return schema
  for schema_name, detector in _STRUCTURAL:
    try:
      if detector(mapping):
        return schema_name
    except (KeyError, TypeError, ValueError):
      continue
  return None


def adapt_kernel_artifact(mapping: Mapping[str, Any], source: EvidenceSource) -> AdaptResult:
  schema = detect_kernel_artifact(mapping)
  if schema is None:
    return AdaptResult(UNSUPPORTED, message="no schema and no structural match")
  adapter = _ADAPTERS.get(schema)
  if adapter is None:
    return AdaptResult(UNSUPPORTED, schema=schema, message=f"no adapter registered for {schema!r}")
  try:
    return adapter(mapping, source)
  except _AdapterInvalid as exc:
    return AdaptResult(INVALID, schema=schema, fields=tuple(exc.fields), message=str(exc))
  except (KeyError, TypeError, ValueError) as exc:
    return AdaptResult(INVALID, schema=schema, message=str(exc))


class _AdapterInvalid(ValueError):
  def __init__(self, message: str, fields: tuple[str, ...] = ()):
    super().__init__(message)
    self.fields = fields


# --- fragment join -----------------------------------------------------------

def _merge_section(into: dict, key: str, value: Any, path: str, blockers: list[FactBlocker]) -> None:
  if value is None:
    return
  if key not in into:
    into[key] = value
    return
  existing = into[key]
  if isinstance(existing, dict) and isinstance(value, dict):
    for k, v in value.items():
      _merge_section(existing, k, v, f"{path}.{k}", blockers)
  elif existing != value:
    blockers.append(FactBlocker("fragment_conflict", "identity",
                                f"conflicting values for {path}: {existing!r} vs {value!r}", (path,)))


def _merge_payloads(fragments: list[EvidenceFragment]) -> tuple[dict, list[FactBlocker]]:
  merged: dict[str, Any] = {}
  blockers: list[FactBlocker] = []
  for frag in fragments:
    for key, value in frag.payload.items():
      _merge_section(merged, key, value, key, blockers)
    blockers.extend(frag.blockers)
  return merged, blockers


def _join_key(keys: dict[str, Any]) -> tuple | None:
  """Fragment join priority. Never joins on newest timestamp; returns None when no rule applies."""
  if keys["experiment_id"] and keys["candidate_id"] and keys["binary_hash"]:
    return ("exp_cand_bin", keys["experiment_id"], keys["candidate_id"], keys["binary_hash"])
  if keys["candidate_hash"] and keys["binary_hash"] and keys["system_snapshot_id"]:
    return ("hash_bin_sys", keys["candidate_hash"], keys["binary_hash"], keys["system_snapshot_id"])
  if keys["candidate_id"] and keys["workload_sig"] and keys["session_id"]:
    return ("cand_wl_session", keys["candidate_id"], keys["workload_sig"], keys["session_id"])
  return None


@dataclass(frozen=True)
class IngestResult:
  evidence: tuple[KernelEvidence, ...]
  orphans: tuple[EvidenceFragment, ...]
  unsupported: tuple[AdaptResult, ...]
  invalid: tuple[AdaptResult, ...]


def join_fragments(fragments: list[EvidenceFragment]) -> tuple[tuple[KernelEvidence, ...],
                                                                tuple[EvidenceFragment, ...]]:
  """Group fragments by identity and assemble each group into a KernelEvidence. Fragments that cannot
  be keyed or that lack enough to build (no workload) are returned as orphans."""
  groups: dict[tuple, list[EvidenceFragment]] = {}
  unkeyed: list[EvidenceFragment] = []
  for frag in fragments:
    key = _join_key(frag.identity_keys())
    if key is None:
      unkeyed.append(frag)
    else:
      groups.setdefault(key, []).append(frag)

  evidence: list[KernelEvidence] = []
  orphans: list[EvidenceFragment] = []
  # An unkeyed fragment that is self-sufficient (candidate + workload) becomes its own evidence;
  # otherwise it cannot join anything and is reported as an orphan.
  for frag in unkeyed:
    built = _assemble(dict(frag.payload), list(frag.provenance), list(frag.blockers))
    (evidence.append(built) if built is not None else orphans.append(frag))
  for group in groups.values():
    payload, blockers = _merge_payloads(group)
    provenance: list[EvidenceSource] = []
    for frag in group:
      provenance.extend(frag.provenance)
    built = _assemble(payload, provenance, blockers)
    if built is None:
      orphans.extend(group)
    else:
      evidence.append(built)
  return tuple(evidence), tuple(orphans)


def _assemble(payload: dict, provenance: list[EvidenceSource],
              blockers: list[FactBlocker]) -> KernelEvidence | None:
  if not payload.get("candidate", {}).get("candidate_id"):
    return None
  if not payload.get("workload"):
    return None  # cannot build a candidate without a workload; caller keeps it as an orphan fragment
  full = dict(payload)
  full["schema"] = SCHEMA_KERNEL_EVIDENCE
  full.setdefault("stages", {})
  # de-dup provenance and blockers while preserving order
  seen_fp = set()
  prov = []
  for s in provenance:
    if s.fingerprint not in seen_fp:
      seen_fp.add(s.fingerprint)
      prov.append(s.to_json())
  full["provenance"] = prov
  full["blockers"] = [b.to_json() for b in _dedup_blockers(blockers)]
  full.pop("evidence_id", None)
  return KernelEvidence.from_json(full)


def _dedup_blockers(blockers: list[FactBlocker]) -> list[FactBlocker]:
  seen = set()
  out = []
  for b in blockers:
    key = (b.code, b.scope, b.message)
    if key not in seen:
      seen.add(key)
      out.append(b)
  return out


# --- ingestion from paths ----------------------------------------------------

def ingest_kernel_artifacts(paths: list[str | Path]) -> IngestResult:
  fragments: list[EvidenceFragment] = []
  evidence: list[KernelEvidence] = []
  unsupported: list[AdaptResult] = []
  invalid: list[AdaptResult] = []
  for raw in paths:
    path = Path(raw)
    data = path.read_bytes()
    try:
      mapping = json.loads(data)
    except json.JSONDecodeError as exc:
      invalid.append(AdaptResult(INVALID, message=f"{path}: not valid JSON: {exc}"))
      continue
    source = EvidenceSource(
      producer=mapping.get("producer", "unknown") if isinstance(mapping, Mapping) else "unknown",
      tool=mapping.get("tool", path.name) if isinstance(mapping, Mapping) else path.name,
      path=str(path), fingerprint=sha256_json_bytes(data),
    )
    result = adapt_kernel_artifact(mapping, source)
    if result.status == OK and result.evidence is not None:
      # Route full evidence through the join path too, so identity-sharing provider fragments
      # (resource traces, ISA manifests) can enrich it rather than being stranded as orphans.
      fragments.append(_evidence_to_fragment(result.evidence))
    elif result.status == OK and result.fragment is not None:
      fragments.append(result.fragment)
    elif result.status == OK and result.fragments:
      fragments.extend(result.fragments)
    elif result.status == UNSUPPORTED:
      unsupported.append(result)
    else:
      invalid.append(result)
  joined, orphans = join_fragments(fragments)
  return IngestResult(evidence=tuple(evidence) + joined, orphans=orphans,
                      unsupported=tuple(unsupported), invalid=tuple(invalid))


def sha256_json_bytes(data: bytes) -> str:
  from boltbeam.core.canonical import sha256_hex
  return "sha256:" + sha256_hex(data)


def _evidence_to_fragment(evidence: KernelEvidence) -> EvidenceFragment:
  payload = evidence.to_json()
  payload.pop("schema", None)
  payload.pop("evidence_id", None)
  payload.pop("provenance", None)
  blockers = evidence.blockers
  payload.pop("blockers", None)
  return EvidenceFragment(payload=payload, provenance=evidence.provenance, blockers=blockers)


# --- native adapter ----------------------------------------------------------

def _native_adapter(mapping: Mapping[str, Any], source: EvidenceSource) -> AdaptResult:
  try:
    payload = dict(mapping)
    payload.setdefault("provenance", [source.to_json()])
    evidence = KernelEvidence.from_json(payload)
  except (KeyError, TypeError, ValueError) as exc:
    raise _AdapterInvalid(f"native kernel_evidence invalid: {exc}")
  return AdaptResult(OK, evidence=evidence, schema=SCHEMA_KERNEL_EVIDENCE)


_BUILTINS_REGISTERED = False


def _register_builtin() -> None:
  register_kernel_adapter(SCHEMA_KERNEL_EVIDENCE, _native_adapter)
  # Real provider adapters live in a sibling module to keep this registry provider-neutral.
  from boltbeam.kernel_analysis.provider_adapters import register_provider_adapters
  register_provider_adapters()


def _ensure_builtins() -> None:
  global _BUILTINS_REGISTERED
  if not _BUILTINS_REGISTERED:
    _register_builtin()
    _BUILTINS_REGISTERED = True


_ensure_builtins()
