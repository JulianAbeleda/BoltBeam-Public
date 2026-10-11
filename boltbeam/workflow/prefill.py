"""A prefill run's results: the untraced prefill, the where table at the run's prompt length, and the curve across
prompt lengths.

One prefill run measures one engine at its prompt lengths (`--workload prefill --ctxs ...`). Its results read:

    the truth     timing_trace.json: the untraced prefill per length, the engine's chunk, the clock and any throttle
    the table     prefill_trace.json at the run's first length, through tie_out.where_prefill_goes: the one where
                  table every renderer prints
    the curve     every length of this run and of the other prefill runs of the same model, chip and engine in the
                  same runs folder: prefill ms, MFU, and ms and MFU per row, against prompt length

Nothing here measures. Every number is read from the run's files.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.workflow import tie_out as tie

PER = "per prefill"


def _read(path:pathlib.Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    return {}


def truths(run:pathlib.Path) -> dict[int, dict[str, Any]]:
  trace = _read(run / "timing_trace.json")
  return {int(r["context"]): r for r in trace.get("rows") or [] if r.get("scope") == "whole_step" and r.get("workload") == "prefill"}


def tables(run:pathlib.Path) -> dict[int, dict[str, Any]]:
  """The where table at every captured length of the run."""
  from boltbeam.collectors.prefill import TRACE
  trace, truth = _read(run / TRACE), truths(run)
  out = {}
  for length, attr in (trace.get("lengths") or {}).items():
    t = truth.get(int(length)) or {}
    w = tie.where_prefill_goes(attr, untraced_ms=t.get("prefill_ms"), length=int(length))
    if w:
      out[int(length)] = w
  return out


def loss_block(run:pathlib.Path, manifest:dict[str, Any], base:dict[str, Any]) -> dict[str, Any]:
  """The results' loss block for a prefill run, on the decode block's keys (base) so every renderer reads it: the
  untraced prefill against the sum of its kernels' ceilings, and the where table at the run's first length."""
  truth = truths(run)
  out = {**base, "workload": "prefill", "per": PER, "runtimes": [], "roles": [], "tie_out": None, "where_token_goes": None,
         "prefill": {"lengths": {}}, "prefill_curve": curve(run, manifest)}
  if not truth:
    out.update(status="absent", reason="no prefill measured in this run")
    return out
  first = next(iter(truth))
  where = tables(run)
  trace = _read(run / "timing_trace.json")
  for length, row in truth.items():
    w = where.get(length)
    out["prefill"]["lengths"][str(length)] = {"ms": row["prefill_ms"], "tok_s": row["tok_s"], "chunk": row["chunk"],
                                              "batched": row["batched"], "telemetry": row.get("telemetry"),
                                              "throttled": row.get("throttled"), "engine": row.get("engine"),
                                              "ceiling_ms": w["limit_ms"] if w else None, "mfu_pct": w["mfu_pct"] if w else None}
  out["prefill"]["clock_policy"] = trace.get("clock_policy")
  head = truth[first]
  w = where.get(first)
  limit = w["limit_ms"] if w else None
  out.update(status="modeled" if w else "measured", limit_ms=limit, limit_tok_s=first / limit * 1e3 if limit else None,
             where_token_goes=w, step=None,
             runtimes=[{"provider": out.get("provider"), "tok_s": head["tok_s"], "ms": head["prefill_ms"],
                        "lost_ms": head["prefill_ms"] - limit if limit else None, "per_role": False,
                        "note": f"whole prefill of {first} tokens, measured untraced; {head['chunk']['words']}"
                                + (f"; {tie_words(head)}" if head.get("throttled") else "")}])
  if not w:
    out["missing"] = "no prefill captured in the model: step 5 (role time) has not run for this run"
  return out


def tie_words(row:dict[str, Any]) -> str:
  return (row.get("telemetry") or {}).get("words") or ""


def curve(run:pathlib.Path, manifest:dict[str, Any]) -> dict[str, Any] | None:
  """Prefill against prompt length for this run's engine: every length of this run and of its sibling prefill runs
  (same model, chip and engine, same runs folder). Per length: prefill ms, tokens/s, MFU, and per row ms and MFU."""
  from boltbeam.workflow.screen import is_run, load_manifest, run_provider
  provider = run_provider(run)
  runs = [run] + sorted(p for p in run.parent.iterdir() if p != run and is_run(p)) if run.parent.is_dir() else [run]
  points: dict[int, dict[str, Any]] = {}
  for r in runs:
    m = load_manifest(r)
    if m.get("workload") != "prefill" or m.get("model_id") != manifest.get("model_id") or \
       m.get("target_id") != manifest.get("target_id") or run_provider(r) != provider:
      continue
    where = tables(r)
    for length, t in truths(r).items():
      if length in points:
        continue
      w = where.get(length)
      points[length] = {"length": length, "run": r.name, "ms": t["prefill_ms"], "tok_s": t["tok_s"],
                        "batched": t["batched"], "chunk_size": t["chunk"].get("size"), "throttled": t.get("throttled"),
                        "mfu_pct": w["mfu_pct"] if w else None, "ceiling_ms": w["limit_ms"] if w else None,
                        "rows": {x["name"]: {"ms": x["now_ms"], "mfu_pct": x.get("mfu_pct")} for x in (w or {}).get("rows", [])}}
  if not points:
    return None
  lengths = sorted(points)
  names = sorted({n for p in points.values() for n in p["rows"]},
                 key=lambda n: -max(p["rows"].get(n, {}).get("ms") or 0.0 for p in points.values()))
  return {"provider": provider, "lengths": lengths, "points": [points[l] for l in lengths], "rows": names,
          "words": (f"{provider}: prefill against prompt length, every length measured for this model on this chip "
                    "(this run and its sibling prefill runs). MFU is operations over each row's path peak for its time; "
                    "the whole-prefill MFU is the time the matrix units would need at their peaks over the untraced prefill.")}
