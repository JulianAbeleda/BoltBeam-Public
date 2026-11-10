"""What a captured kernel is, by its name: the block (attention, recurrent), a weight GEMV no role took, or a helper.
The capture tools report the names; the pairing walk, the attribution and the tie-out read this one classification.
The words are data (data/kernel_kinds.json, first match wins, in the file's order), loaded once here."""
from __future__ import annotations

import json
import pathlib
from typing import Any


def _load() -> dict[str, Any]:
  """The table, checked: every row names a kind the table lists (fail closed; a typo is no silent `other`)."""
  t = json.loads((pathlib.Path(__file__).resolve().parent.parent / "data" / "kernel_kinds.json").read_text())
  bad = [r for r in t["words"] + t["prefixes"] if r["kind"] not in t["kinds"]] + ([] if t["unattributed"] in t["kinds"] else [t])
  if bad:
    raise ValueError(f"data/kernel_kinds.json: a row names a kind the table does not list: {bad[0]}")
  return t


_TABLE = _load()
UNATTRIBUTED = _TABLE["unattributed"]  # a GEMV no role took: weight time outside the limit; the capture has no bytes for it
KINDS = tuple(_TABLE["kinds"])
KIND_WORDS = tuple((r["word"], r["kind"]) for r in _TABLE["words"])  # in order: the first match wins
KIND_PREFIX = tuple((r["prefix"], r["kind"]) for r in _TABLE["prefixes"])
GEMV_WORDS = tuple(w for w, k in KIND_WORDS if k == UNATTRIBUTED)  # the GEMV names the adapters know


def kind_of(row:dict[str, Any]) -> str:
  if row.get("logical_parts"):
    return "weight kernels"
  if row.get("kind") in KINDS:
    return row["kind"]
  name = str(row.get("kernel") or "")
  if row.get("role") in KINDS:
    return row["role"]
  for word, kind in KIND_WORDS:
    if word in name:
      return kind
  for prefix, kind in KIND_PREFIX:
    if name.startswith(prefix):
      return kind
  return "other"
