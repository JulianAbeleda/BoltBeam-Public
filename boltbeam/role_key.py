"""One identity for a weight role across a run's files: (role, quant, rows, k).

A model can hold one role and format at two shapes (a hybrid's two ssm_projection Q8_0 weights), so role and quant
alone collide. Every file spells the shape its own way: the ceiling {"m", "n", "k"}, the role-time trace [1, n, k],
the route policy [n, k], the model profile rows/cols. shape_nk reads all of them; a row without a shape keys by
(role, quant) alone.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def shape_nk(row:Mapping[str, Any] | None) -> tuple[int, int] | None:
  """(rows, k) of a role row in any of the run's spellings, or None."""
  if not row: return None
  shape = row.get("shape")
  if isinstance(shape, Mapping) and shape.get("n") is not None and shape.get("k") is not None:
    return int(shape["n"]), int(shape["k"])
  if isinstance(shape, (list, tuple)) and len(shape) >= 2 and shape[-2] is not None and shape[-1] is not None:
    return int(shape[-2]), int(shape[-1])
  if row.get("rows") is not None and row.get("cols") is not None:
    return int(row["rows"]), int(row["cols"])
  return None


def role_key(row:Mapping[str, Any] | None) -> tuple[Any, ...]:
  row = row or {}
  nk = shape_nk(row)
  return (row.get("role"), row.get("quant")) + (nk if nk else ())


def role_stem(role:str, quant:str, shape:Any = None) -> str:
  """The file stem of one role's search files: role, format and shape."""
  nk = shape_nk({"shape": shape})
  return f"{role}-{quant.lower()}" + (f"-{nk[0]}x{nk[1]}" if nk else "")


def role_label(role:str, quant:str, shape:Any = None) -> str:
  nk = shape_nk({"shape": shape})
  return f"{role} {quant}" + (f" {nk[0]}x{nk[1]}" if nk else "")


__all__ = ["role_key", "role_label", "role_stem", "shape_nk"]
