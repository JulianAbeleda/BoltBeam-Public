"""Path-independent fingerprints for cacheable inputs (audit-brain-build-scope BB3).

Every id here is derived from CONTENT, never from where a file happens to live on disk. That is what lets the
cache stay valid across a fresh checkout, a relocated bench artifact, or a tinygrad tree at a different path.
The byte/JSON hashing itself is reused from artifacts.base so there is one hashing authority in the package.
"""
from __future__ import annotations

from typing import Any

from boltbeam.artifacts.base import sha256_bytes, sha256_json

# top-level model-profile keys that carry an absolute filesystem path; excluded so the fingerprint is stable
# across checkouts and relocation. Adding a new path-bearing key = editing this tuple, not scattering strips.
PATH_KEYS = ("source",)

# metadata keys that may carry the producing tinygrad commit, in priority order
_COMMIT_KEYS = ("tinygrad_commit", "tinygrad_sha", "git_commit", "commit")


def fingerprint_bytes(data:bytes) -> str:
  """Fingerprint raw artifact bytes (a)."""
  return sha256_bytes(data)


def _path_stripped(profile:dict[str, Any]) -> dict[str, Any]:
  return {k: v for k, v in profile.items() if k not in PATH_KEYS}


def fingerprint_profile(profile:dict[str, Any]) -> str:
  """Fingerprint a model-profile dict (b), excluding any absolute 'source' path so it is relocation-stable."""
  return sha256_json(_path_stripped(profile))


def fingerprint_search_space(space:dict[str, Any]) -> str:
  """Fingerprint a search-space dict (c)."""
  return sha256_json(space)


def fingerprint_target(target:dict[str, Any]) -> str:
  """Fingerprint a target-profile dict (d)."""
  return sha256_json(target)


def tinygrad_commit(metadata:dict[str, Any] | None) -> str | None:
  """Extract the producing tinygrad commit string from metadata if present (e); otherwise None."""
  if not metadata: return None
  for k in _COMMIT_KEYS:
    v = metadata.get(k)
    if v: return str(v)
  return None


def semantic_fingerprint(*, profile:dict[str, Any] | None = None, search_space:dict[str, Any] | None = None,
                         target:dict[str, Any] | None = None, artifact_bytes:bytes | None = None,
                         commit:str | None = None) -> str:
  """Combined content id, STABLE across checkouts and path relocation.

  Only the provided components contribute. No component may inject an absolute path: model profiles go through
  `fingerprint_profile` (which strips PATH_KEYS) and artifacts are hashed by bytes, not by location.
  """
  parts: dict[str, str] = {}
  if profile is not None: parts["profile"] = fingerprint_profile(profile)
  if search_space is not None: parts["search_space"] = fingerprint_search_space(search_space)
  if target is not None: parts["target"] = fingerprint_target(target)
  if artifact_bytes is not None: parts["artifact"] = fingerprint_bytes(artifact_bytes)
  if commit: parts["commit"] = commit
  if not parts:
    raise ValueError("semantic_fingerprint needs at least one component")
  return sha256_json(parts)
