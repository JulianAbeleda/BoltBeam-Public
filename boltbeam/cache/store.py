"""Local cache index (audit-brain-build-scope BB3).

A CacheStore is a JSON file (schema boltbeam.artifact_cache.v1) mapping each ingested evidence's path-independent
`evidence_id` to a small record: producer, model_id, the artifact path, and whether that file was present when
added. The key is semantic, so relocating the artifact does not create a duplicate entry. `status()` re-checks
the disk live so it can flag artifacts that have gone missing or whose bytes drifted from the recorded fingerprint.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.artifacts.base import sha256_bytes
from boltbeam.vocab import SCHEMA_ARTIFACT_CACHE
from boltbeam.core.canonical import pretty_json


class CacheStore:
  def __init__(self, path:str | pathlib.Path, *, root:str | pathlib.Path | None = None):
    self.path = pathlib.Path(path)
    self.root = pathlib.Path(root) if root else None
    self.entries: dict[str, dict[str, Any]] = {}

  def _resolve(self, p:str) -> pathlib.Path:
    pp = pathlib.Path(p)
    if self.root and not pp.is_absolute():
      pp = self.root / pp
    return pp

  def add(self, evidence:Any, *, present:bool | None = None) -> str:
    """Index one NormalizedEvidence. `present` defaults to a live existence check of its artifact path."""
    src = evidence.source
    if present is None:
      present = self._resolve(src.path).exists()
    self.entries[evidence.evidence_id] = {
      "fingerprint": src.fingerprint,
      "producer": src.producer,
      "model_id": evidence.model_id,
      "path": src.path,
      "present": bool(present),
    }
    return evidence.evidence_id

  def _live_present(self, entry:dict[str, Any]) -> bool:
    return self._resolve(entry["path"]).exists()

  def _is_stale(self, entry:dict[str, Any]) -> bool:
    """A present file whose current bytes no longer hash to the recorded fingerprint."""
    p = self._resolve(entry["path"])
    try:
      data = p.read_bytes()
    except OSError:
      return False
    return sha256_bytes(data) != entry["fingerprint"]

  def status(self) -> dict[str, Any]:
    """Summary of the index, with disk re-checked live: counts, producers, model_ids, missing + stale paths."""
    values = list(self.entries.values())
    missing = sorted(e["path"] for e in values if not self._live_present(e))
    stale = sorted(e["path"] for e in values if self._live_present(e) and self._is_stale(e))
    return {
      "entries": len(values),
      "producers": sorted({e["producer"] for e in values}),
      "model_ids": sorted({e["model_id"] for e in values}),
      "missing": missing,
      "stale": stale,
    }

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA_ARTIFACT_CACHE, "entries": self.entries}

  def save(self) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.path.write_text(pretty_json(self.to_json()))

  def load(self) -> "CacheStore":
    if not self.path.exists():
      self.entries = {}
      return self
    data = json.loads(self.path.read_text())
    if data.get("schema") != SCHEMA_ARTIFACT_CACHE:
      raise ValueError(f"{self.path} is not a {SCHEMA_ARTIFACT_CACHE} cache index")
    self.entries = dict(data.get("entries", {}))
    return self
