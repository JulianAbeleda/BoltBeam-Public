"""Deterministic data-only expansion of full-kernel candidate dimensions."""
from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from typing import Any

from boltbeam.search.spec import FullKernelCandidate


FieldPath = str | tuple[str, ...]


def instantiate_candidates(seed: FullKernelCandidate, dimensions: Mapping[FieldPath, Sequence[Any]], *,
                           max_candidates: int = 256) -> tuple[FullKernelCandidate, ...]:
  """Apply a deterministic Cartesian product of field substitutions to ``seed``.

  This intentionally performs no compiler-legality filtering. Tinygrad admission
  is the authority for whether a schema-valid generated candidate can lower.
  """
  if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates <= 0:
    raise ValueError("max_candidates must be a positive int")
  normalized = []
  for raw_path, values in dimensions.items():
    path = _field_path(raw_path)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
      raise ValueError(f"candidate dimension {'.'.join(path)!r} must be a non-empty sequence")
    normalized.append((path, tuple(values)))
  normalized.sort(key=lambda item: item[0])
  combination_count = 1
  for _path, values in normalized: combination_count *= len(values)
  if combination_count > max_candidates:
    raise ValueError(f"candidate Cartesian product {combination_count} exceeds max_candidates={max_candidates}")

  products = itertools.product(*(values for _path, values in normalized)) if normalized else ((),)
  out: list[FullKernelCandidate] = []
  seen: set[str] = set()
  for values in products:
    payload = json.loads(seed.canonical_json())
    for (path, _choices), value in zip(normalized, values): _set_existing(payload, path, value)
    candidate = FullKernelCandidate(payload)
    if candidate.candidate_hash in seen: continue
    seen.add(candidate.candidate_hash)
    out.append(candidate)
  return tuple(out)


def instantiate_candidate_rows(seed:FullKernelCandidate, rows:Sequence[Mapping[FieldPath, Any]], *,
                               max_candidates:int=256) -> tuple[FullKernelCandidate, ...]:
  """Apply finite coupled substitutions without duplicating the canonical seed.

  Cartesian dimensions remain the compact form for independent axes. Rows are
  the corresponding compact form when fields must move together, such as an
  explicit LOCAL transform and its declared launch width.
  """
  if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates <= 0:
    raise ValueError("max_candidates must be a positive int")
  if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
    raise ValueError("candidate rows must be a non-empty sequence")
  if len(rows) > max_candidates: raise ValueError(f"candidate rows exceed max_candidates={max_candidates}")
  out: list[FullKernelCandidate] = []
  seen: set[str] = set()
  for row in rows:
    if not isinstance(row, Mapping) or not row: raise ValueError("each candidate row must be a non-empty mapping")
    payload = json.loads(seed.canonical_json())
    normalized = sorted((_field_path(path), value) for path, value in row.items())
    if len({path for path, _ in normalized}) != len(normalized): raise ValueError("candidate row contains duplicate paths")
    for path, value in normalized: _set_existing(payload, path, value)
    candidate = FullKernelCandidate(payload)
    if candidate.candidate_hash in seen: continue
    seen.add(candidate.candidate_hash)
    out.append(candidate)
  return tuple(out)


def _field_path(path: FieldPath) -> tuple[str, ...]:
  parts = tuple(path.split(".")) if isinstance(path, str) else path
  if not isinstance(parts, tuple) or not parts or any(not isinstance(x, str) or not x for x in parts):
    raise ValueError(f"invalid candidate field path {path!r}")
  return parts


def _set_existing(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
  parent: Any = payload
  for part in path[:-1]:
    if not isinstance(parent, dict) or part not in parent: raise ValueError(f"candidate field path {'.'.join(path)!r} does not exist")
    parent = parent[part]
  if not isinstance(parent, dict) or path[-1] not in parent: raise ValueError(f"candidate field path {'.'.join(path)!r} does not exist")
  # JSON normalization prevents callers from retaining a mutable alias.
  parent[path[-1]] = json.loads(json.dumps(value, allow_nan=False))


__all__ = ["FieldPath", "instantiate_candidate_rows", "instantiate_candidates"]
