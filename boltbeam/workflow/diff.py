"""Two runs of one model on one target, compared: the one authority for what may be compared and how two where tables
differ. replay.diff (two replays of one trace) and the engine-upgrade audit (two builds of one engine, prefill or
decode) both refuse on the same rule and diff rows the same way.

  refuse        the runs must hold the same workload, model and target (and whatever else the caller names); the
                first field that differs is said, and nothing is compared
  rows          two where tables by row name: each row's time change, against a band of BAND_PCT of its before time
                and at least BAND_FLOOR_MS (a tiny row's noise); its work (calls, operations, bytes, ceiling) equal or
                not; rows only before or only after
  tie-out       each side's own: kernels plus gaps against the captured wall, and the captured wall against the
                untraced time (tie_out.end_to_end, its own band)

The engine-upgrade audit is the check to run before any engine upgrade (docs/engine-upgrade-audit.md).
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.workflow import tie_out as tie

AUDIT_SCHEMA = "boltbeam.engine_diff.v1"
BAND_PCT = 0.03  # a row stays put when it moves by at most this share of its before time (the loop test's band)
BAND_FLOOR_MS = 0.05  # or by at most this, for rows too small for a share to mean anything
WORK = ("calls", "flops", "bytes", "limit_ms")  # what a row did; equal on both sides when only the engine's speed changed
TIE_RESIDUAL_MS = 1e-3  # kernels plus gaps equal the captured wall within this


class DiffRefused(ValueError):
  """Two runs that cannot be compared: the field that differs."""


def refuse_unless_same(fields:list[tuple[str, Any, Any]]) -> None:
  """Raise DiffRefused naming the first field the runs differ in (or that one of them lacks)."""
  for name, x, y in fields:
    if x != y or x is None:
      raise DiffRefused(f"the runs differ in {name}: {x!r} and {y!r}")


def row_changes(rows_a:dict[Any, dict[str, Any]], rows_b:dict[Any, dict[str, Any]], *, band:bool = False) -> list[dict[str, Any]]:
  """Rows keyed alike on both sides, after less before: now_ms and lost_ms, or only_in. With band: each row's before
  and after time, its band (BAND_PCT of before, at least BAND_FLOOR_MS), whether it moved beyond it, and the work
  fields that changed."""
  out = []
  for k in sorted(set(rows_a) | set(rows_b), key=lambda k: tuple(str(x) for x in (k if isinstance(k, tuple) else (k,)))):
    a, b = rows_a.get(k), rows_b.get(k)
    both = a is not None and b is not None
    r = {"key": k, "now_ms": (b["now_ms"] - a["now_ms"]) if both else None,
         "lost_ms": (b.get("lost_ms", 0) - a.get("lost_ms", 0)) if both and "lost_ms" in a else None,
         "only_in": None if both else ("before" if a is not None else "after")}
    if band:
      before, after = (a or {}).get("now_ms"), (b or {}).get("now_ms")
      width = max(BAND_PCT * (before or 0.0), BAND_FLOOR_MS)
      r.update(before_ms=before, after_ms=after, band_ms=width, moved=both and abs(r["now_ms"]) > width,
               work_changed=[f for f in WORK if both and _differs(a.get(f), b.get(f))])
    out.append(r)
  return out


def _differs(x:Any, y:Any) -> bool:
  if isinstance(x, (int, float)) and isinstance(y, (int, float)):
    return abs(x - y) > 1e-9 * max(abs(x), abs(y), 1.0)
  return x != y


# --- the engine-upgrade audit: two builds of one engine ---------------------------------------------------------------

def _read(path:pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text()) if path.exists() else {}


def _prefill_sides(run:pathlib.Path) -> dict[str, dict[str, Any]]:
  """Per captured length: the rows (now_ms and work) and the tie-out."""
  from boltbeam.collectors.prefill import TRACE
  truth = {int(r["context"]): r["prefill_ms"] for r in _read(run / "timing_trace.json").get("rows") or [] if r.get("prefill_ms")}
  out = {}
  for length, attr in (_read(run / TRACE).get("lengths") or {}).items():
    if not attr.get("rows"):
      continue
    rows = {r["name"]: {"now_ms": r["ms"], **{f: r.get(f) for f in ("calls", "flops", "bytes")}} for r in attr["rows"]}
    e2e = attr.get("end_to_end") or tie.end_to_end(attr, truth.get(int(length)))
    out[f"prefill {length}"] = {"rows": rows, "tie": {
      "captured_ms": attr.get("profiled_wall_ms"), "busy_ms": attr.get("busy_ms"), "gaps_ms": attr.get("gaps_ms"),
      "residual_ms": float(attr["wall_ms"]) - float(attr["busy_ms"]) - float(attr["gaps_ms"]),
      "untraced_ms": truth.get(int(length)), "end_to_end": e2e}}
  return out


def _decode_side(run:pathlib.Path) -> dict[str, dict[str, Any]]:
  """The decode where table (per token) and its tie-out: the captured token against the untraced one."""
  from boltbeam.workflow.screen import results
  loss = results(run).get("loss") or {}
  w, t = loss.get("where_token_goes"), loss.get("tie_out") or {}
  if not w:
    return {}
  rows = {r["name"]: {"now_ms": r["now_ms"], **{f: r.get(f) for f in ("calls", "limit_ms")}} for r in w["rows"]}
  token, busy, untraced = t.get("token_ms"), t.get("busy_ms"), t.get("untraced_ms")
  attr = {"profiled_wall_ms": token, "wall_ms": token, "busy_ms": busy, "gaps_ms": (token - busy) if token and busy else None}
  e2e = tie.end_to_end(attr, untraced) if token and busy is not None else {"confirms": None, "words": "no captured token"}
  return {f"decode {t.get('context')}": {"rows": rows, "tie": {
    "captured_ms": token, "busy_ms": busy, "gaps_ms": attr["gaps_ms"], "residual_ms": 0.0 if token and busy else None,
    "untraced_ms": untraced, "end_to_end": e2e}}}


def audit(a:pathlib.Path, b:pathlib.Path) -> dict[str, Any]:
  """Two audits of one model, target and workload on two engine builds, row by row: boltbeam.engine_diff.v1. Refused,
  naming the field, when the model, the target, the workload or the provider differ, or when either run's engine build
  is unknown. Names both pins (data/engine_pins.json) and says plainly when a build is not its pin."""
  from boltbeam.collectors import providers
  from boltbeam.workflow.common import load_manifest
  from boltbeam.workflow.screen import run_provider
  ma, mb = load_manifest(a), load_manifest(b)
  refuse_unless_same([("workload", ma.get("workload"), mb.get("workload")), ("model", _model(a, ma), _model(b, mb)),
                      ("target", ma.get("target_id"), mb.get("target_id")), ("provider", run_provider(a), run_provider(b))])
  builds = [providers.run_build(a), providers.run_build(b)]
  for run, got in zip((a, b), builds):
    if got is None:
      raise DiffRefused(f"{run.name} recorded no engine build: an audit of an unknown build is not compared")
  pins = [providers.pin_check(run_provider(a), x) for x in builds]
  side = _prefill_sides if ma.get("workload") == "prefill" else _decode_side
  A, B = side(a), side(b)
  if not (A and B):
    raise DiffRefused("a run holds no captured where table")
  parts = []
  for key in sorted(set(A) & set(B), key=lambda k: (k.split()[0], float(k.split()[1]))):
    rows = row_changes(A[key]["rows"], B[key]["rows"], band=True)
    ta, tb = A[key]["tie"], B[key]["tie"]
    gaps = row_changes({"gaps": {"now_ms": ta["gaps_ms"] or 0.0}}, {"gaps": {"now_ms": tb["gaps_ms"] or 0.0}}, band=True)[0]
    kernels = sum(r["now_ms"] for r in rows if r["now_ms"] is not None) + sum(
      (r["after_ms"] or 0.0) - (r["before_ms"] or 0.0) for r in rows if r["only_in"])
    wall = (tb["captured_ms"] - ta["captured_ms"]) if ta["captured_ms"] and tb["captured_ms"] else None
    held = (not any(r["moved"] or r["work_changed"] or r["only_in"] for r in rows) and not gaps["moved"]
            and all(s["end_to_end"].get("confirms") is not False for s in (ta, tb)))
    parts.append({"part": key, "rows": rows, "gaps": gaps, "before": ta, "after": tb, "kernels_ms": kernels,
                  "captured_ms": wall, "reconciles": wall is not None and abs(wall - kernels - gaps["now_ms"]) <= TIE_RESIDUAL_MS,
                  "untraced_ms": (tb["untraced_ms"] - ta["untraced_ms"]) if ta["untraced_ms"] and tb["untraced_ms"] else None,
                  "unchanged": held})
  return {"schema": AUDIT_SCHEMA, "before": a.name, "after": b.name, "builds": builds, "provider": run_provider(a), "pins": pins,
          "model": _model(a, ma), "target": ma.get("target_id"), "workload": ma.get("workload"), "parts": parts,
          "band": f"max({100 * BAND_PCT:.0f}% of the before time, {BAND_FLOOR_MS} ms)",
          "end_to_end_band": f"±{100 * tie.END_TO_END_BAND:.0f}%"}


def _model(run:pathlib.Path, manifest:dict[str, Any]) -> str | None:
  """The model file the run read: the same model under two run ids is the same model."""
  p = manifest.get("model_path")
  return pathlib.Path(p).name if p else manifest.get("model_id")


def audit_lines(d:dict[str, Any]) -> list[str]:
  from boltbeam.collectors import providers
  f = lambda v, s="{:+.3f}": "-" if v is None else s.format(v)  # noqa: E731
  out = [f"Engine upgrade audit: {d['after']} ({d['builds'][1]}) against {d['before']} ({d['builds'][0]}), "
         f"{d['workload']}; band {d['band']}, end to end {d['end_to_end_band']}.",
         f"  pinned: {providers.pin_label(d['pins'][0]['pin'])}. Before: {d['pins'][0]['words']}. After: {d['pins'][1]['words']}."]
  for p in d["parts"]:
    out.append(f"  {p['part']}: {'unchanged outside moved rows' if p['unchanged'] else 'changed'}")
    for r in sorted(p["rows"], key=lambda r: -max(r["before_ms"] or 0, r["after_ms"] or 0)):
      verdict = (f"only {r['only_in']}" if r["only_in"] else ("MOVED" if r["moved"] else "within band")
                 + (f"; work changed: {', '.join(r['work_changed'])}" if r["work_changed"] else ""))
      out.append(f"    {str(r['key'])[:40]:40} {f(r['before_ms'], '{:.3f}'):>9} -> {f(r['after_ms'], '{:.3f}'):>9} "
                 f"({f(r['now_ms'])}, band {r['band_ms']:.3f})  {verdict}")
    g = p["gaps"]
    out.append(f"    gaps {g['before_ms']:.3f} -> {g['after_ms']:.3f} ({g['now_ms']:+.3f}, band {g['band_ms']:.3f})"
               f"{'  MOVED' if g['moved'] else ''}; kernels {p['kernels_ms']:+.3f} + gaps = captured "
               f"{f(p['captured_ms'])} ms: {'reconciles' if p['reconciles'] else 'DOES NOT reconcile'}; untraced {f(p['untraced_ms'])} ms")
    for name in ("before", "after"):
      out.append(f"    {name}: {p[name]['end_to_end']['words']}")
  return out
