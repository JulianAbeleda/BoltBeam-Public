"""A session run's results: the session table (one row per depth), the warm-cache row, the compaction comparison, the
KV cache and the context that fits, and where the turn's time goes at the focus depth.

One session run measures one engine at its depths (`--workload session --ctxs ...`, collectors/session.py). Every
number is read from the run's files; nothing here measures. The table is built once, as cells, and every renderer
(the text summary, the HTML report, the TUI) prints those cells.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.workflow import tie_out as tie

PER = "per turn"
TITLE = "The session: one turn at each depth"
COLUMNS = ("depth", "turn ms", "turn tok/s", "tokens computed", "decode tok/s", "KV cache", "next tokens", "clock")


def _read(path:pathlib.Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    return {}


def _mib(b:float | None) -> str:
  return "-" if b is None else (f"{b / 2**30:.2f} GiB" if b >= 2**30 else f"{b / 2**20:.0f} MiB" if b >= 2**20
                                else f"{b / 2**10:.0f} KiB")


def check_cell(c:dict[str, Any] | None) -> str:
  if not c:
    return "not checked"
  if c["pass"]:
    return f"pass (|dp| {c['max_dp']:.3f})"
  why = (f"token {c['first_mismatch'] + 1} differs" if c.get("first_mismatch") is not None else
         f"top sets differ at {len(c['top_sets_differ_at'])}" if c.get("top_sets_differ_at") else
         f"|dp| {c['max_dp']:.3f}" if c.get("with_probabilities") else "no probabilities")
  return f"FAIL ({why})"


def table(trace:dict[str, Any]) -> dict[str, Any] | None:
  """The session table from timing_trace.json: a row per depth, the warm row, the compaction, the KV cache."""
  rows = [r for r in trace.get("rows") or [] if r.get("workload") == "session"]
  if not rows:
    return None
  kv = trace.get("kv") or {}
  out_rows = []
  for r in sorted((x for x in rows if x["scope"] == "whole_step"), key=lambda x: x["context"]):
    if r.get("failed"):
      out_rows.append({"depth": r["context"], "failed": r["failed"],
                       "cells": [f"{r['context']:,}", "not run", "", "", "", _mib(r.get("kv_bytes")), "", ""]})
      continue
    computed = sorted(set(r.get("computed") or []))
    out_rows.append({"depth": r["context"], "turn_ms": r["turn_ms"], "decode_tok_s": r["decode_tok_s"], "check": r.get("check"),
                     "throttled": r.get("throttled"),
                     "cells": [f"{r['context']:,}", f"{r['turn_ms']:.1f}", f"{r['tok_s']:,.0f}",
                               "/".join(str(c) for c in computed) or "?", f"{r['decode_tok_s']:.1f}", _mib(r.get("kv_bytes")),
                               check_cell(r.get("check")), "THROTTLED" if r.get("throttled") else "held"]})
  done = [r for r in out_rows if "turn_ms" in r]
  warm = next((r for r in rows if r["scope"] == "warm"), None)
  comp = next((r for r in rows if r["scope"] == "compaction"), None)
  words = [f"A turn is {trace.get('turn_tokens')} new tokens appended to a cache already holding the depth's tokens: its "
           "untraced time to the first token. Decode tok/s is greedy decode at that depth. KV cache is what the depth's "
           f"tokens hold ({_mib(kv.get('bytes_per_token'))} per token over {kv.get('attention_layers')} attention layers, "
           f"{kv.get('element')})"
           + (f"; the recurrent layers hold a fixed {_mib(kv['fixed_state_bytes'])} whatever the depth" if kv.get("fixed_state_bytes") else "")
           + f". Next tokens: {trace.get('agreement')}."]
  failed = [r for r in out_rows if r.get("failed")]
  if failed:
    words.append(" ".join(f"Depth {r['depth']:,} not run: {r['failed'][-200:]}" for r in failed))
  out: dict[str, Any] = {"title": TITLE, "columns": list(COLUMNS), "rows": out_rows, "words": " ".join(words),
                         "focus": trace.get("focus"), "kv": kv}
  if warm:
    out["warm"] = (f"Warm cache, depth {warm['context']:,}: the same context and turn sent again with the prefix cache on "
                   f"computed {warm['computed']} token(s) in {warm['ms']:.1f} ms ({warm['source']}).")
  if kv.get("caps"):
    out["max_context"] = caps_words(kv["caps"])
  out["compaction"] = compaction(comp, done)
  return out


def caps_words(c:dict[str, Any]) -> str:
  """The max context before compaction per cache type: the binding one of the model's cap and the memory cap."""
  m = c.get("model")
  head = f"The model declares {m['tokens']:,} tokens ({m['key']})" if m else "The model declares no context length"
  parts = [f"{name} {t['binding']:,} ({'the model' if t['binds'] == 'model' else 'memory'} binds; memory alone {t['memory']:,})"
           for name, t in c["types"].items() if name != "engine"]
  eng = c["types"].get("engine")
  text = f"{head}. Max context before compaction, per KV cache type: {'; '.join(parts)}"
  if eng:
    text += f". This engine's own cache: {eng['binding']:,}"
  return text + (f". Memory cap: the device less {100 * c['reserve']:.0f}% and {c['fixed_bytes'] / 2**30:.1f} GiB fixed "
                 f"({c['fixed_from']}), over the KV bytes per token.")


def compaction(comp:dict[str, Any] | None, done:list[dict[str, Any]]) -> str | None:
  """Re-prefilling a summary on an empty cache against the turn at the deepest depth measured, and how many turns
  at the shallowest depth instead of the deepest pay for it."""
  if not comp:
    return None
  if comp.get("failed"):
    return f"Compaction: the {comp['prompt_tokens']:,}-token summary on an empty cache was not measured: {comp['failed'][-200:]}"
  text = f"Compaction: re-prefilling a {comp['prompt_tokens']:,}-token summary on an empty cache takes {comp['ms']:.1f} ms"
  if not done:
    return text + "."
  deep, shallow = max(done, key=lambda r: r["depth"]), min(done, key=lambda r: r["depth"])
  text += f"; a turn at depth {deep['depth']:,} takes {deep['turn_ms']:.1f} ms"
  saved = deep["turn_ms"] - shallow["turn_ms"]
  if saved > 0 and deep is not shallow:
    text += (f", at depth {shallow['depth']:,} {shallow['turn_ms']:.1f} ms: the summary costs {comp['ms'] / deep['turn_ms']:.1f} deep "
             f"turns and pays for itself after {comp['ms'] / saved:.1f} turns at the shallow depth")
  return text + "."


def where(run:pathlib.Path, trace:dict[str, Any]) -> dict[str, Any] | None:
  """The where table of the turn captured at the focus depth (session_trace.json), tied out to the untraced turn."""
  from boltbeam.collectors.session import TRACE
  st = _read(run / TRACE)
  depth = st.get("depth")
  attr = next(iter((st.get("lengths") or {}).values()), None)
  row = next((r for r in trace.get("rows") or [] if r.get("scope") == "whole_step" and r.get("context") == depth
              and r.get("turn_ms")), None)
  if not (attr and row):
    return None
  from boltbeam.workflow.repeat_bands import find
  attr = {**attr, "end_to_end": tie.end_to_end(attr, row["turn_ms"], "turn", band=find(run, f"prefill {trace['turn_tokens']}"))}
  return tie.where_prefill_goes(attr, untraced_ms=row["turn_ms"], length=int(trace["turn_tokens"]), depth=int(depth))


def loss_block(run:pathlib.Path, manifest:dict[str, Any], base:dict[str, Any]) -> dict[str, Any]:
  """The results' loss block for a session run, on the decode block's keys (base) so every renderer reads it: the
  turn at the focus depth against the sum of its kernels' ceilings, the session table beside it."""
  trace = _read(run / "timing_trace.json")
  out = {**base, "workload": "session", "per": PER, "runtimes": [], "roles": [], "tie_out": None, "where_token_goes": None,
         "session": table(trace)}
  head = next((r for r in trace.get("rows") or [] if r.get("scope") == "whole_step" and r.get("context") == trace.get("focus")
               and r.get("turn_ms")), None)
  if not head:
    out.update(status="absent", reason="no turn measured at the focus depth in this run")
    return out
  w = where(run, trace)
  limit = w["limit_ms"] if w else None
  n = int(trace["turn_tokens"])
  out.update(status="modeled" if w else "measured", limit_ms=limit, limit_tok_s=n / limit * 1e3 if limit else None,
             where_token_goes=w, step=None,
             runtimes=[{"provider": out.get("provider"), "tok_s": head["tok_s"], "ms": head["turn_ms"],
                        "lost_ms": head["turn_ms"] - limit if limit else None, "per_role": False,
                        "note": f"a {n}-token turn at depth {head['context']:,}, measured untraced; {head['chunk']['words']}"
                                + (f"; {(head.get('telemetry') or {}).get('words')}" if head.get("throttled") else "")}])
  if not w:
    out["missing"] = "no turn captured in the model: step 5 (role time) has not run for this run"
  return out


def lines(s:dict[str, Any] | None) -> list[str]:
  """The session table as plain text: one line per depth, each column as wide as its widest cell, then the warm
  row, the context that fits and the compaction."""
  if not s:
    return []
  cells = [list(s["columns"])] + [r["cells"] for r in s["rows"]]
  width = [max(len(row[i]) for row in cells) for i in range(len(cells[0]))]
  out = [s["title"] + ":", f"  {s['words']}"]
  out += [("  " + " ".join(c.rjust(width[i]) for i, c in enumerate(row))).rstrip() for row in cells]
  out += [f"  {s[k]}" for k in ("warm", "max_context", "compaction") if s.get(k)]
  return out
