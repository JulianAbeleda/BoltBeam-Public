"""Two runs of one model on one target, compared: the one authority for what may be compared and how two where tables
differ. replay.diff (two replays of one trace) and the engine-upgrade audit (two builds of one engine, prefill or
decode) both refuse on the same rule and diff rows the same way.

  refuse        the runs must hold the same workload, model and target (and whatever else the caller names); the
                first field that differs is said, and nothing is compared
  rows          two where tables by row name: each row's time change, against the row's measured repeat spread; missing repeat data is not checked; its work (calls, operations, bytes, ceiling) equal or
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
WORK = ("calls", "flops", "bytes", "limit_ms")  # what a row did; equal on both sides when only the engine's speed changed
TIE_RESIDUAL_MS = 1e-3  # kernels plus gaps equal the captured wall within this


class DiffRefused(ValueError):
  """Two runs that cannot be compared: the field that differs."""


def refuse_unless_same(fields:list[tuple[str, Any, Any]]) -> None:
  """Raise DiffRefused naming the first field the runs differ in (or that one of them lacks)."""
  for name, x, y in fields:
    if x is None and y is None:
      raise DiffRefused(f"neither run recorded its {name}")
    if x != y or x is None:
      raise DiffRefused(f"the runs differ in {name}: {x!r} and {y!r}")


def row_changes(rows_a:dict[Any, dict[str, Any]], rows_b:dict[Any, dict[str, Any]], *, band:bool = False,
                widths:dict[str, dict | None] | None = None, n:int | None = None) -> list[dict[str, Any]]:
  """Rows keyed alike on both sides, after less before: now_ms and lost_ms, or only_in. With band: each row's before
  and after time, its before band (the range of n before runs, repeat_bands.RULE), whether the after time lies outside
  it (None without a band), n and the false-move rate, and the work fields that changed."""
  from boltbeam.workflow.repeat_bands import strength
  out = []
  for k in sorted(set(rows_a) | set(rows_b), key=lambda k: tuple(str(x) for x in (k if isinstance(k, tuple) else (k,)))):
    a, b = rows_a.get(k), rows_b.get(k)
    both = a is not None and b is not None
    r = {"key": k, "now_ms": (b["now_ms"] - a["now_ms"]) if both else None,
         "lost_ms": (b.get("lost_ms", 0) - a.get("lost_ms", 0)) if both and "lost_ms" in a else None,
         "only_in": None if both else ("before" if a is not None else "after")}
    if band:
      before, after = (a or {}).get("now_ms"), (b or {}).get("now_ms")
      rng = (widths or {}).get(k)
      r.update(before_ms=before, after_ms=after, band_ms=rng and rng["width_ms"],
               moved=(not rng["low_ms"] <= after <= rng["high_ms"]) if both and rng else None,
               checked=both and rng is not None, **(strength(n) if n and rng else {}),
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


def _prefill_sides(run:pathlib.Path, record:dict | None = None) -> dict[str, dict[str, Any]]:
  """Per captured length: the rows (now_ms and work) and the tie-out."""
  from boltbeam.collectors.prefill import TRACE
  native = {int(r["context"]): r for r in _read(run / "timing_trace.json").get("rows") or [] if r.get("prefill_ms")}
  out = {}
  for length, attr in (_read(run / TRACE).get("lengths") or {}).items():
    if not attr.get("rows"):
      continue
    rows = {r["name"]: {"now_ms": r["ms"], **{f: r.get(f) for f in ("calls", "flops", "bytes")}} for r in attr["rows"]}
    from boltbeam.workflow.repeat_bands import find
    row = native.get(int(length)) or {}
    truth = row.get("qualification_ms") or row.get("prefill_ms")
    e2e = tie.end_to_end(attr, truth, band=find(run, f"prefill {length}", record=record),
                       reference="untraced final repetition", reference_window=(row.get("engine") or {}).get("timing_window"))
    out[f"prefill {length}"] = {"rows": rows, "tie": {
      "captured_ms": attr.get("profiled_wall_ms"), "engine_elapsed_ms": attr.get("measured_ms"), "busy_ms": attr.get("busy_ms"), "gaps_ms": attr.get("gaps_ms"),
      "residual_ms": float(attr["wall_ms"]) - float(attr["busy_ms"]) - float(attr["gaps_ms"]),
      "untraced_ms": truth, "end_to_end": e2e}}
  return out


def _decode_side(run:pathlib.Path, record:dict | None = None) -> dict[str, dict[str, Any]]:
  """The decode where table (per token) and its tie-out: the captured token against the untraced one."""
  from boltbeam.workflow.screen import results
  # Repeat evidence must not depend on the band it is used to derive. The normal report still applies
  # calibrated floor refusals; this reader compares observations, not the validity of the roofline.
  loss = results(run, repeat_evidence=True).get("loss") or {}
  w, t = loss.get("where_token_goes"), loss.get("tie_out") or {}
  if not w:
    return {}
  rows = {r["name"]: {"now_ms": r["now_ms"], **{f: r.get(f) for f in ("calls", "limit_ms")}}
          for r in w["rows"] if r.get("what") != "gaps"}  # the gaps are the tie-out's own row, compared once
  if loss.get("roles"):
    from boltbeam.role_key import role_label
    rows = {r["name"]: {"now_ms": r["now_ms"], **{f: r.get(f) for f in ("calls", "limit_ms")}}
            for r in w["rows"] if r.get("what") not in ("gaps", "role")}
    for role in loss["roles"]:
      if role.get("actual_ms") is not None:
        key = role_label(role["role"], role["quant"], role.get("shape"))
        rows[key] = {"now_ms": role["actual_ms"], "calls": role.get("calls_per_token"), "limit_ms": role.get("ideal_ms")}
  token, busy, untraced = t.get("token_ms"), t.get("busy_ms"), t.get("untraced_ms")
  if t.get("isolated"):  # kernels timed alone: no captured token, so no gaps and no end-to-end check; the rows repeat
    return {f"decode {t.get('context')}": {"rows": rows, "tie": {
      "captured_ms": None, "busy_ms": busy, "gaps_ms": None, "residual_ms": None, "untraced_ms": untraced,
      "end_to_end": {"confirms": None, "words": "the kernels were timed alone: there is no captured token to check end to end"}}}}
  attr = {"measured_ms": token, "wall_ms": token, "busy_ms": busy, "gaps_ms": (token - busy) if token and busy else None}
  match = t.get("timing_match") or {}
  attr["timing_window"] = match.get("captured_window")
  from boltbeam.workflow.repeat_bands import find
  e2e = tie.end_to_end(attr, untraced, "token", band=find(run, f"decode {t.get('context')}", record=record),
                     reference="untraced reference", reference_window=match.get("native_window")) if token and busy is not None else {"confirms": None, "words": "no captured token"}
  if match and not match["matches"]:
    e2e = {"confirms": None, "words": match["words"], "timing_match": match}
  return {f"decode {t.get('context')}": {"rows": rows, "tie": {
    "captured_ms": token, "engine_elapsed_ms": token, "busy_ms": busy, "gaps_ms": attr["gaps_ms"], "residual_ms": 0.0 if token and busy else None,
    "untraced_ms": untraced, "end_to_end": e2e}}}


def audit(a:pathlib.Path, b:pathlib.Path, *, bands:dict | None = None) -> dict[str, Any]:
  """Two audits of one model, target and workload on two engine builds, row by row: boltbeam.engine_diff.v1. Refused,
  naming the field, when the model, the target, the workload or the provider differ, or when either run's engine build
  is unknown. Names both pins (data/engine_pins.json) and says plainly when a build is not its pin."""
  from boltbeam.collectors import providers
  from boltbeam.workflow.common import load_manifest
  from boltbeam.workflow.screen import run_provider
  from boltbeam.collectors.toolchain import changes
  ma, mb = load_manifest(a), load_manifest(b)
  refuse_unless_same([("workload", ma.get("workload"), mb.get("workload")), ("model", _model(a, ma), _model(b, mb)),
                      ("target", ma.get("target_id"), mb.get("target_id")), ("provider", run_provider(a), run_provider(b))])
  builds = [providers.run_build(a), providers.run_build(b)]
  for run, got in zip((a, b), builds):
    if got is None:
      raise DiffRefused(f"{run.name} recorded no engine build: an audit of an unknown build is not compared")
  pins = [providers.pin_check(run_provider(r), x, providers.run_build_number(r), providers.run_build_installed(r)) for r, x in zip((a, b), builds)]
  side = _prefill_sides if ma.get("workload") == "prefill" else _decode_side
  A, B = side(a, bands), side(b, bands)
  if not (A and B):
    raise DiffRefused("a run holds no captured where table")
  parts = []
  from boltbeam.workflow import repeat_bands as rb
  for key in sorted(set(A) & set(B), key=lambda k: (k.split()[0], float(k.split()[1]))):
    ba = rb.find(a, key, record=bands)  # the before build's band: the after value is judged against its n runs
    if ba and _timing_sha(b) in {s.get("timing_sha256") for s in ba["sources"]}:
      raise DiffRefused(f"{b.name} is one of its own band's runs: a holdout is never judged by a band it made")
    n = (ba or {}).get("n")
    widths = {name: (ba["rows"][name] if rb.width(ba, name, work=row) is not None else None)
              for name, row in A[key]["rows"].items()}
    rows = row_changes(A[key]["rows"], B[key]["rows"], band=True, widths=widths, n=n)
    ta, tb = A[key]["tie"], B[key]["tie"]
    gaps = row_changes({"gaps": {"now_ms": ta["gaps_ms"] or 0.0}}, {"gaps": {"now_ms": tb["gaps_ms"] or 0.0}},
                       band=True, widths={"gaps": (ba or {}).get("timing", {}).get("gaps_ms")}, n=n)[0]
    kernels = sum(r["now_ms"] for r in rows if r["now_ms"] is not None) + sum(
      (r["after_ms"] or 0.0) - (r["before_ms"] or 0.0) for r in rows if r["only_in"])
    wall = (tb["captured_ms"] - ta["captured_ms"]) if ta["captured_ms"] and tb["captured_ms"] else None
    checked = all(r["checked"] for r in [*rows, gaps]) and all(s["end_to_end"].get("confirms") is not None for s in (ta, tb))
    held = (not any(r["moved"] or r["work_changed"] or r["only_in"] for r in rows) and not gaps["moved"]
            and all(s["end_to_end"].get("confirms") is not False for s in (ta, tb)))
    parts.append({"part": key, "rows": rows, "gaps": gaps, "before": ta, "after": tb, "kernels_ms": kernels,
                  "captured_ms": wall, "reconciles": wall is not None and abs(wall - kernels - gaps["now_ms"]) <= TIE_RESIDUAL_MS,
                  "untraced_ms": (tb["untraced_ms"] - ta["untraced_ms"]) if ta["untraced_ms"] and tb["untraced_ms"] else None,
                  "unchanged": held if checked else None, "status": "checked" if checked else "not checked",
                  **(rb.strength(n) if n else {}), "pin_refused": rb.pin_words(ba)})
  return {"schema": AUDIT_SCHEMA, "before": a.name, "after": b.name, "builds": builds, "provider": run_provider(a), "pins": pins,
          "toolchain": changes(ma, mb),
          "model": _model(a, ma), "target": ma.get("target_id"), "workload": ma.get("workload"), "parts": parts,
          "band": f"the range of n before runs per row ({rb.RULE}), or not checked",
          "justifies_pin": bool(parts) and not any(p["pin_refused"] for p in parts),
          "end_to_end_band": "measured repeat spread, or not checked"}


def _timing_sha(run:pathlib.Path) -> str:
  from boltbeam.artifacts.base import sha256_json
  return sha256_json(_read(run / "timing_trace.json"))


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
  refused = sorted({p["pin_refused"] for p in d["parts"] if p.get("pin_refused")})
  out.append("  pin change: " + ("this audit can justify it" if d.get("justifies_pin") else "NOT justified: " + "; ".join(refused)))
  for t in d.get("toolchain", []):
    out.append(f"  {t['name']}: {t['before']} -> {t['after']} "
               f"({'CHANGED' if t['changed'] else 'unchanged' if t['known'] else 'unknown'}; pin {t['pin']})")
  for p in d["parts"]:
    how = (f"; n = {p['n']} before runs, false-move rate {p['false_move_rate']:.0%} per row{' (WEAK)' if p['weak'] else ''}"
           if p.get("n") else "")
    out.append(f"  {p['part']}: {'not checked' if p['unchanged'] is None else 'unchanged outside moved rows' if p['unchanged'] else 'changed'}{how}")
    for r in sorted(p["rows"], key=lambda r: -max(r["before_ms"] or 0, r["after_ms"] or 0)):
      verdict = (f"only {r['only_in']}" if r["only_in"] else ("not checked" if not r["checked"] else ("MOVED" if r["moved"] else "within band")
                 + (f" (n {r['n']}, false-move {r['false_move_rate']:.0%}{', weak' if r['weak'] else ''})" if r.get("n") else ""))
                 + (f"; work changed: {', '.join(r['work_changed'])}" if r["work_changed"] else ""))
      out.append(f"    {str(r['key'])[:40]:40} {f(r['before_ms'], '{:.3f}'):>9} -> {f(r['after_ms'], '{:.3f}'):>9} "
                 f"({f(r['now_ms'])}, band {f(r['band_ms'], '{:.6f}')})  {verdict}")
    g = p["gaps"]
    out.append(f"    gaps {g['before_ms']:.3f} -> {g['after_ms']:.3f} ({g['now_ms']:+.3f}, band {f(g['band_ms'], '{:.6f}')})"
               f"{'  not checked' if not g['checked'] else '  MOVED' if g['moved'] else ''}; kernels {p['kernels_ms']:+.3f} + gaps = captured "
               f"{f(p['captured_ms'])} ms: {'reconciles' if p['reconciles'] else 'DOES NOT reconcile'}; untraced {f(p['untraced_ms'])} ms")
    for name in ("before", "after"):
      out.append(f"    {name}: {p[name]['end_to_end']['words']}")
  return out
