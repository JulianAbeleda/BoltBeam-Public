"""Single canonical-JSON + sha256 implementation for BoltBeam (LN-010).

All artifact hashing and byte-reproducible serialization goes through this module. There is
exactly one hash-string convention: bare-hex sha256 digests (`sha256_hex`/`sha256_json`); the rare
call sites that need the `sha256:`-tagged form use `prefixed()`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any


def canonical_json(value: Any) -> str:
  """Strict canonical JSON: sorted keys, compact separators, ASCII-only, no NaN/Infinity."""
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_bytes(value: Any) -> bytes:
  return canonical_json(value).encode("ascii")


def sha256_hex(data: bytes) -> str:
  """Bare-hex sha256 digest of raw bytes."""
  return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
  """Bare-hex sha256 digest of a value's canonical JSON encoding."""
  return sha256_hex(canonical_bytes(value))


def prefixed(digest: str) -> str:
  """Add the `sha256:` tag for the call sites whose convention requires it."""
  return f"sha256:{digest}"


def sha256_file(path: "pathlib.Path | str", chunk_size: int = 8 * 1024 * 1024) -> str:
  """Bare-hex sha256 digest of a file's bytes, read in chunks (large-file safe)."""
  digest = hashlib.sha256()
  with open(path, "rb") as stream:
    for chunk in iter(lambda: stream.read(chunk_size), b""):
      digest.update(chunk)
  return digest.hexdigest()


def pretty_json(value: Any) -> str:
  """Sorted, human-readable JSON for artifact files; reproducible bytes for equal values (LN-020)."""
  return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def write_artifact(path: "pathlib.Path | str", value: Any) -> None:
  """Write an artifact file with reproducible bytes: sorted keys, ASCII, trailing newline."""
  p = pathlib.Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(pretty_json(value))
