"""The least memory each role needs resident on the device, from the model's shapes and the run's settings alone.

    exact      the formula is the need, under the condition its row states (dense weights, the KV cells, pooled indexer
               keys, one recurrent state with what a rollback needs)
    bounded    the formula is a floor and the row says what sets the rest (experts: none must be resident; compute
               buffers: the engine's graph), so held - floor is the most a change could free, never a claim
    unknown    nothing in the shapes gives it (the device's context, what it reserves, a tensor the classifier does not
               name): the row names the measurement

Formulas are data (data/min_memory.json, a block per shape the header shows) read by tune's formula reader over facts from a
tensor table cut from the model file's header ({meta, tensors: [name, dims, type id]}), the metadata and the settings.
Tensors are sorted by the one classifier (profile/roles.py) and sized by decode_roles._physical_bytes. The audit's
memory ledger sets each held row beside its minimum (gaps, checks). Engine-blind: an engine's sizes are the held rows."""
from __future__ import annotations

import json, math, pathlib, re
from typing import Any

from boltbeam import formats
from boltbeam.core.refused import Refused
from boltbeam.profile.decode_roles import _physical_bytes
from boltbeam.profile.roles import is_expert_role, role_from_tensor_name
from boltbeam.workflow.context_audit import derived, unknown
from boltbeam.workflow.tune import Missing, evaluate, output_of

TAB = json.loads((pathlib.Path(__file__).resolve().parents[1] / "data" / "min_memory.json").read_text())
COPIES, UNCLASSIFIED, NO_BLOCK = TAB["roles"]["copies"], TAB["roles"]["unclassified"], TAB["roles"]["no_block"]


def kind(name:str, dims:list) -> str:
  role = role_from_tensor_name(name, tuple(dims))
  return "experts" if len(dims) == 3 and is_expert_role(role) else {"lm_head": "head", "embedding": "lookup", "other": "unclassified"}.get(role, "dense")


def nbytes(name:str, dims:list, typ:int) -> int: return _physical_bytes(math.prod(dims), typ, tensor_name=name)


def one(v:Any) -> Any:
  """A per-layer list as one value: its nonzero entries when they agree, else None (the rows that read it are unknown)."""
  return v if not isinstance(v, list) else (s.pop() if len(s := {x for x in v if x}) == 1 else None)


def ev(formula:str, f:dict[str, Any]) -> Any:
  """A row's formula over the facts; a name the facts lack (a setting not given, a typo) is Missing, so its row is unknown, naming it."""
  return evaluate(formula, f, set(f) | set(re.findall(r"[a-z_]+", formula)))


def block(meta:dict[str, Any] | None, tensors:list) -> dict[str, Any] | None:
  """The block of formulas for a header, by what it shows (a block's `has` tensor-name patterns), never by its
  architecture's name; None when it shows none of any block's."""
  return next((b for b in TAB["blocks"] if any(re.search(p, t[0]) for p in b["has"] for t in tensors or [])), None)


def facts(meta:dict[str, Any], tensors:list, settings:dict[str, Any], arch:dict[str, Any], skip:Any = ()) -> dict[str, Any]:
  """What the formulas read: the metadata under the architecture's names, a tensor's dim, layers counted by their
  tensors, bytes per kind (less the tensors in skip), each <x>_type setting as <x>_elem_bytes, the settings, the terms."""
  pre = f"{meta['general.architecture']}."
  out = {k: one(meta.get(pre + key)) for k, key in arch["meta"].items()}
  out |= {k: next((d[axis] for n, d, _ in tensors if re.search(pat, n)), None) for k, (pat, axis) in arch["dims"].items()}
  out |= {f"n_{k}": sum(1 for n, _, _ in tensors if re.search(pat, n)) for k, pat in arch["counts"].items()}
  out |= {f"bytes_{k}": sum(nbytes(n, d, t) for n, d, t in tensors if kind(n, d) == k and n not in skip) for k in ("dense", "head", "lookup", "experts")}
  out |= {k[:-5] + "_elem_bytes": formats.elem_bytes(v) for k, v in settings.items() if k.endswith("_type")}
  out |= {k: v for k, v in settings.items() if not k.endswith("_type")}
  for name, formula in arch.get("terms", {}).items():
    try:
      out[name] = ev(formula, out)
    except Missing:
      out[name] = None  # a row that reads it is unknown, naming it
  return out


def minimum(meta:dict[str, Any], tensors:list, settings:dict[str, Any], *, other:list = (), same:Any = (), prefix:str = "",
            unit:float = 1.0) -> dict[str, dict[str, Any]]:
  """{role: figure with its class}, in bytes / unit. other: the table of a file loaded beside this one; same: citations
  [{tensor, from, output?, output_why?}] that a tensor it shares is one matrix with the other's (none, or no `from`:
  unknown, with the test). A header that shows no block's shape gives the `every` rows and one unknown row naming what it leaves out."""
  out = {} if prefix else {r["role"]: unknown(r["test"]) for r in TAB["every"]}
  if (arch := block(meta, tensors)) is None:  # said: the rows no block sizes for this header
    return out if prefix else {**out, NO_BLOCK: unknown(TAB["roles"]["no_block_test"].format(rows=", ".join(r["role"] for b in TAB["blocks"] for r in b["rows"])))}
  theirs = {n: (tuple(d), t) for n, d, t in other}
  twice = {n: t != theirs[n][1] for n, d, t in tensors if theirs.get(n, ((),))[0] == tuple(d) and kind(n, d) != "lookup"}
  f = facts(meta, tensors, settings, arch, skip=twice)
  for row in arch["rows"]:
    try:
      value = ev(row["formula"], f) / unit
    except Missing as m:
      out[prefix + row["role"]] = unknown(f"a model whose metadata gives {m} (as one value for every layer)")
      continue
    reads = {n: f[n] for n in dict.fromkeys(re.findall(r"[a-z_]+", row["formula"])) if n in f}
    out[prefix + row["role"]] = {**derived(value, row["formula"], reads), "class": row["class"], **{k: row[k] for k in ("why", "rest") if k in row}}
  if odd := sorted(n for n, d, _ in tensors if kind(n, d) == "unclassified"):
    out[prefix + UNCLASSIFIED] = unknown("a classifier row (profile/roles.py) for: " + ", ".join(odd))
  cite = {c["tensor"]: c for c in same if c.get("from")}
  if twice and all(n in cite for n in twice):
    label = [output_of(cite[n]) for n, differ in twice.items() if differ]
    out[prefix + COPIES] = {**derived(0.0, "one matrix with the other file's: needed once", {"tensors": sorted(twice), "from": [cite[n]["from"] for n in twice]}),
                            "class": "exact", **({"output": label[0]["output"], "output_why": label[0]["why"]} if label else {})}
  elif twice:
    out[prefix + COPIES] = unknown("each tensor's rows against the other file's, beside two quantizations of one matrix, cited as `from`: "
                                   + ", ".join(n for n in sorted(twice) if n not in cite))
  return out


def of_record(ledger:dict[str, Any], base:pathlib.Path) -> dict[str, dict[str, Any]]:
  """A ledger record's `shapes` ([{table, settings, prefix?, beside?, same?}]: tables cut from the headers, paths relative
  to base; beside: the index of the file loaded beside it) as one {role: minimum} in the ledger's unit_bytes."""
  if not ledger.get("unit_bytes"):
    raise Refused("the ledger has shapes but no unit_bytes: the bytes of its rows' unit")
  tabs = [json.loads((base / s["table"]).read_text()) for s in ledger["shapes"]]
  return {k: v for s, t in zip(ledger["shapes"], tabs) for k, v in minimum(t["meta"], t["tensors"], s["settings"], prefix=s.get("prefix", ""),
          unit=ledger["unit_bytes"], other=tabs[s["beside"]]["tensors"] if "beside" in s else (), same=s.get("same", ())).items()}


def gaps(rows:list[dict[str, Any]], least:dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  """Held rows, each with its minimum, class (exact, bounded, unknown) and gap = held - minimum: at most that for a
  bounded row; an estimate, with the held row's test, when the held figure is one. No minimum: unknown, with the test."""
  out = []
  for r in rows:
    m = least.get(r["role"]) or unknown(f"a row for {r['role']} in data/min_memory.json, or the measurement that gives its least")
    gap = unknown(m["test"]) if m["value"] is None else derived(r["value"] - m["value"], "held - minimum", [r["role"]])
    if m["value"] is not None and r["label"] == "estimate":
      gap = {"value": gap["value"], "label": "estimate", "formula": "held - minimum", "test": r["test"]}
    out.append({**r, "minimum": m, "class": m.get("class", "unknown"), "gap": gap})
  return out


def checks(rows:list[dict[str, Any]], least:dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  """No exact gap below zero (held under what the math needs: a formula or a held figure is wrong), and every role the
  shapes need (a minimum above zero) has a held row."""
  low = [r["role"] for r in rows if r["class"] == "exact" and r["gap"]["value"] < 0]
  bare = [k for k, m in least.items() if (m["value"] or 0) > 0 and k not in {r["role"] for r in rows}]
  return [{"check": "no exact gap below zero", "ok": not low, "why": "held is at least the minimum on every exact row" if not low else "held below the minimum: " + ", ".join(low)},
          {"check": "every needed role is held", "ok": not bare, "why": "each role with a minimum has a held row" if not bare else "no held row for: " + ", ".join(bare)}]
