"""Local artifact cache (audit-brain-build-scope BB3).

Two concerns, kept orthogonal:
  fingerprint.py  path-independent semantic ids for artifacts / profiles / search spaces / targets
  store.py        a small JSON index of what evidence has been ingested and whether its file still exists

Nothing here reads a model or launches a tool; it only hashes bytes/dicts and tracks paths.
"""
from __future__ import annotations

from boltbeam.cache.fingerprint import (fingerprint_bytes, fingerprint_profile, fingerprint_search_space,
                                        fingerprint_target, tinygrad_commit, semantic_fingerprint)
from boltbeam.cache.store import CacheStore

__all__ = ["fingerprint_bytes", "fingerprint_profile", "fingerprint_search_space", "fingerprint_target",
           "tinygrad_commit", "semantic_fingerprint", "CacheStore"]
