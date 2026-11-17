"""Host time per step by engine function: the records of the host-function timer (tools/host_functions: exact entry
and exit times of the functions an engine's row lists), folded into exclusive time per function per decode step at
each context depth, and, with a capture of the same run, the part of each function's time that lies in GPU-idle gaps.

    rows       data/host_functions.json engines.<E>.<release>: the functions (symbol, label), the label whose calls cut
               the steps and tell the contexts apart (scope), the label a turn's decode starts at (start), the labels
               whose own time is a wait for the device (device_wait). Nothing here names an engine or a function.
    read       the records by thread. Refused: no end line (the engine did not exit cleanly), dropped records (a
               thread's buffer filled), a count that differs from the end line's, a torn line.
    capture    the GPU's busy intervals (kernels and device-side copies; a host copy is the calling function's own work)
               and the span each capture covers (its own start and stop), on the timer's clock, and its launch calls.
    fold       per turn: exclusive ns per function (its own time less the timed calls inside it), split by context, the
               steps, and `unattributed` (time on the decode thread inside no timed function). A turn whose window lies
               inside a capture's span and holds its launch calls also gets each function's exclusive time in GPU-idle gaps, by the same
               subtraction, and the count of its launch calls that lie inside a timed record (the clocks agree).
    points     per depth bucket, one point for the turns a capture covers (traced, with their own depth) and one for
               the rest (untraced): ms per step by function, for workflow/context_audit.host_role; the covered point
               ranks the functions by their idle part. No covered turn: the idle part is unknown, with its test.
    checks     steps (the timer's against the engine's step rule, per turn), clock (launch calls inside timed records),
               closes (bookkeeping: the rows sum to the window and the idle rows to the idle time, by construction).
    overhead   a capture's or the timer's cost on the step, with replay_overhead's words and verdict."""
from __future__ import annotations

import bisect
import collections
import json
import pathlib
import sqlite3
from typing import Any, Iterable

from boltbeam.collectors import time_budget as tb
from boltbeam.collectors.replay_overhead import overhead_words, within_ceiling  # the ceiling's one verdict
from boltbeam.core.refused import Refused  # the one refusal every command catches

SCHEMA = "boltbeam.host_functions.v1"
DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "host_functions.json"
UNATTRIBUTED = "unattributed"
DEVICE_COPIES = ("DtoD", "memset")  # copies the GPU does on its own; a host copy (HtoD, DtoH) is the host function's work
IDLE_TEST = "idle share unknown, test: one traced window (the timer under the GPU capture) at this depth and configuration"


def row(engine:str, release:str) -> dict[str, Any]:
  """The engine's rows for one release (data/host_functions.json); none is refused by name."""
  rows = json.loads(DATA.read_text())["engines"].get(engine) or {}
  if release not in rows:
    raise Refused(f"no host-function rows for {engine} {release} (data/host_functions.json has {', '.join(rows) or 'none'})")
  return rows[release]


def read(lines:Iterable[str], name:str) -> dict[int, list[tuple[int, int, str, int]]]:
  """The records of one file (its lines; name, for the words) by thread: (start ns, -end ns, label, first argument)."""
  threads, end, n = collections.defaultdict(list), None, 0
  for i, ln in enumerate(lines):
    if ln.startswith("#end"):
      end = dict(kv.split("=") for kv in ln.split()[1:])
    elif not ln.startswith("#"):
      try:
        tid, label, t0, t1, arg, _ = ln.split("\t")
        threads[int(tid)].append((int(t0), -int(t1), label, int(arg)))
      except ValueError:
        raise Refused(f"{name}: line {i + 1} is torn ({ln[:60]!r}): the file is cut short") from None
      n += 1
  if end is None or int(end["dropped"]) or int(end["records"]) != n:
    raise Refused(f"{name}: " + ("no end line: the engine did not exit cleanly" if end is None else
                                f"the record buffer filled ({end['dropped']} records dropped)" if int(end["dropped"]) else
                                f"{n} records against the {end['records']} its end line counts") + ", so the run is cut short")
  return dict(threads)


def capture(dbs:list[str | pathlib.Path], *, table:dict[str, Any] | None = None) -> dict[str, Any]:
  """One or more capture exports of the timer's run on its clock: busy (kernels and device-side copies, every stream),
  spans (each capture's own start and stop; without them its first and last event, kernel or API call), launches."""
  pattern = tb.causes(table, True)[0]
  busy, spans, launches = [], [], []
  for db in dbs:
    tl = tb.timeline_from_nsys(db)
    if (o := tl.monotonic_origin_ns) is None:
      raise Refused(f"{db}: the capture does not say its clock's origin, so it cannot be put on the timer's clock")
    with sqlite3.connect(db) as con:
      own = con.execute("select startTime, stopTime from ANALYSIS_DETAILS").fetchone() if con.execute(
        "select 1 from sqlite_master where name='ANALYSIS_DETAILS'").fetchone() else None
    events = [x for k in tl.kernels for x in k[:2]] + [x for c in tl.calls for x in c[:2]]
    spans.append(list(own) if own else [min(events) + o, max(events) + o] if events else [0, 0])
    busy += [(a + o, e + o) for a, e, *_ in tl.kernels] + [(c["start_ns"] + o, c["end_ns"] + o) for c in tl.copies if c["kind"] in DEVICE_COPIES]
    launches += [(a + o, e + o) for a, e, n in tl.calls if "launch" in (next((c for p, c in pattern if p.search(n)), None) or "")]
  return {"busy": tb.union(busy), "spans": spans, "launches": sorted(launches)}


def idle_in(busy:list[list[int]]):
  """[a, b) -> its ns with the GPU idle, from the sorted, disjoint busy intervals."""
  st, cum = [s for s, _ in busy], [0]
  for s, e in busy:
    cum.append(cum[-1] + e - s)
  before = lambda x: (k := bisect.bisect_right(st, x)) and cum[k - 1] + min(x, busy[k - 1][1]) - st[k - 1]  # noqa: E731
  return lambda a, b: b - a - (before(b) - before(a))


def fold(threads:dict[int, list], turns:list[dict[str, Any]], row:dict[str, Any], capture:dict[str, Any] | None = None) -> dict[str, Any]:
  """Each turn's decode window on the decode thread (the one with most scope calls): from the first `start` call
  inside the turn's span (span_ns, the driver's CLOCK_MONOTONIC) to the span's end, or the turn's own `window`. A step
  is one scope call on the first-named context; contexts are told apart by the call's first argument and named rarest
  first (a tie is refused). capture: from capture(), for the turns whose window lies inside one of its spans."""
  idle = idle_in(capture["busy"]) if capture else None
  scope, names = row["scope"]["label"], row["scope"]["names"]
  recs = sorted(max(threads.values(), key=lambda v: sum(r[2] == scope for r in v), default=[]))
  calls = sorted(collections.Counter(r[3] for r in recs if r[2] == scope).items(), key=lambda kv: kv[1])
  if not calls or len(calls) > len(names) or len({n for _, n in calls}) < len(calls):
    raise Refused(f"the contexts calling {scope} cannot be named: {len(calls)} context(s) with {[n for _, n in calls]} calls; the row names {len(names)}, rarest first")
  ctx, parent, key, open_ = dict(zip([a for a, _ in calls], names)), [], [], []
  for i, (t0, nt1, label, arg) in enumerate(recs):
    while open_ and -recs[open_[-1]][1] <= t0:
      open_.pop()
    if open_ and -nt1 > -recs[open_[-1]][1]:
      raise Refused(f"records do not nest: {label} at {t0} ends after the {recs[open_[-1]][2]} it starts inside")
    parent.append(open_[-1] if open_ else None)
    key.append((label, ctx[arg] if label == scope else key[parent[i]][1] if open_ else ""))
    open_.append(i)
  key = [f"{label}@{c}" if c else label for label, c in key]
  starts, out, skipped = [r[0] for r in recs], [], []
  for t in turns:
    lo, hi = (bisect.bisect_left(starts, x) for x in t.get("window") or t["span_ns"])
    first = lo if "window" in t else next((i for i in range(lo, hi) if recs[i][2] == row["start"]), None)
    if first is None:
      skipped.append(f"turn {t.get('i')}: no {row['start']} call in its span")
      continue
    (w0, w1), ids, p = t.get("window") or (starts[first], t["span_ns"][1]), list(range(first, hi)), first - 1 if first else None
    while p is not None:  # the calls already open when the window starts: the one before it and what holds it
      ids[:0] = [p] if -recs[p][1] > w0 else []
      p = parent[p]
    ls = capture["launches"][bisect.bisect_left(capture["launches"], (w0,)):bisect.bisect_left(capture["launches"], (w1,))] if capture else []
    seen = bool(ls) and any(a <= w0 and w1 <= b for a, b in capture["spans"])  # inside a span, and captured (a decode launches work)
    on = idle if seen else lambda a, b: 0
    ex, ix, covered, reach = collections.Counter(), collections.Counter(), [0, 0], w0
    for i in ids:
      a, b = max(recs[i][0], w0), min(-recs[i][1], w1)
      for c, v in ((ex, b - a), (ix, on(a, b))):
        c[key[i]] += v
        if parent[i] is not None:
          c[key[parent[i]]] -= v
      if b > reach:  # the union, counted apart from the nesting
        covered, reach = [covered[0] + b - max(a, reach), covered[1] + on(max(a, reach), b)], b
    # each launch call in the window must lie inside a timed record: the two clocks agree
    clock = [len(ls), sum(any(recs[j][0] <= a and e <= -recs[j][1] for j in _chain(bisect.bisect_right(starts, a) - 1, parent)) for a, e in ls)] if seen else None
    out.append({**{k: t[k] for k in ("i", "depth", "expect_steps") if k in t}, "steps": sum(key[i] == f"{scope}@{names[0]}" for i in range(first, hi)),
                "window_ns": w1 - w0, "ns": dict(ex), UNATTRIBUTED: w1 - w0 - covered[0], "clock": clock,
                "idle": {"window_ns": on(w0, w1), "ns": dict(ix), UNATTRIBUTED: on(w0, w1) - covered[1]} if seen else None})
  fired = {r[2] for v in threads.values() for r in v}
  return {"schema": SCHEMA, "turns": out, "skipped": skipped, "never_fired": sorted({f["label"] for f in row.get("functions", []) if "off" not in f} - fired)}


def _chain(j:int, parent:list[int | None]):
  while j is not None and j >= 0:
    yield j
    j = parent[j]


def _sum(ts:list[dict[str, Any]]) -> tuple[int, int, dict[str, int]]:
  ns = collections.Counter({UNATTRIBUTED: sum(t[UNATTRIBUTED] for t in ts)})
  for t in ts:
    ns.update(t["ns"])
  return sum(t["steps"] for t in ts), sum(t["window_ns"] for t in ts), dict(ns)


def points(folded:dict[str, Any], edges:list[int], *, trace:str, configuration:Any = None, qualified=lambda ids: False) -> list[dict[str, Any]]:
  """Per depth bucket, a point for the turns a capture covers (traced: their CPU times carry the capture's cost; the
  functions ranked by their part in GPU-idle gaps) and one for the rest (untraced; idle unknown), each at its own
  step-weighted depth. qualified(turn ids): the capture's overhead on those turns is measured and within the ceiling;
  otherwise the rank is an estimate. The idle part is an upper bound on what removing a function saves (the GPU may wait on other work)."""
  cuts, out = [0, *edges, None], []
  for lo, hi in zip(cuts, cuts[1:]):
    inside = [t for t in folded["turns"] if lo <= t["depth"] and (hi is None or t["depth"] < hi) and t["steps"]]
    for ts in ([t for t in inside if not t["idle"]], [t for t in inside if t["idle"]]):
      if not ts:
        continue
      n, w, ns = _sum(ts)
      p = {"lo": lo, "hi": hi, "depth": sum(t["depth"] * t["steps"] for t in ts) / n, "turns": len(ts), "steps": n, "trace": trace,
           "traced": bool(ts[0]["idle"]), "ids": [t.get("i") for t in ts], "configuration": configuration, "window_ns": w, "parts_ns": ns, "step_ms": w / n / 1e6,
           "parts_ms": {k: v / n / 1e6 for k, v in ns.items()}, "idle": {"value": None, "label": "unknown", "test": IDLE_TEST}}
      if p["traced"]:
        _, iw, ins = _sum([{**t["idle"], "steps": t["steps"]} for t in ts])
        idle = {k: v / n / 1e6 for k, v in ins.items()}
        how = {"label": "measured", "source": trace} if qualified(p["ids"]) else \
          {"label": "estimate", "formula": "exclusive time in GPU-idle gaps, under a capture whose overhead is unknown or over the ceiling",
           "test": "the same capture with its overhead measured within the ceiling (traced against untraced steps of the same turns)"}
        p["idle"] = {**how, "traced": True, "steps": n, "window_ns": iw, "parts_ns": ins, "gpu_idle_ms": iw / n / 1e6, "parts_ms": idle,
                     "rank": [{"part": k, "idle_ms": idle[k], "cpu_ms": p["parts_ms"][k], **how} for k in sorted(idle, key=lambda k: -idle[k])]}
      out.append(p)
  return out


def closes(pts:list[dict[str, Any]]) -> dict[str, Any]:
  """Bookkeeping: the rows and `unattributed` sum to the decode thread's window, and the idle rows to the GPU's idle time
  in it, in integer ns. It holds by construction; the steps and clock checks carry the weight."""
  off = [f"{p['lo']:,}: {what} rows {sum(q['parts_ns'].values()):,} ns against {q['window_ns']:,} ns"
         for p in pts for what, q in (("CPU", p), ("idle", p["idle"])) if "parts_ns" in q and sum(q["parts_ns"].values()) != q["window_ns"]]
  return {"check": "function rows sum to host time", "ok": bool(pts) and not off,
          "why": "; ".join(off) if off else "no depth to check" if not pts else f"rows + {UNATTRIBUTED} = the decode thread's step at {len(pts)} point(s), to the ns"}


def steps_check(folded:dict[str, Any]) -> dict[str, Any]:
  """The timer's steps against the engine's step rule (expect_steps, from time_budget.decode_steps) on every turn."""
  off = [f"turn {t.get('i')}: {t['steps']} against {t['expect_steps']}" for t in folded["turns"] if t["steps"] != t.get("expect_steps")]
  return {"check": "the timer's steps are the engine's", "ok": bool(folded["turns"]) and not off,
          "why": "; ".join(off[:5]) if off else "no turn" if not folded["turns"] else f"equal on {len(folded['turns'])} turn(s)"}


def clock_check(folded:dict[str, Any]) -> dict[str, Any]:
  """For a run given a capture: its launch calls in the covered windows all lie inside timed records (the clocks
  agree), and it covers at least one turn."""
  got = [t["clock"] for t in folded["turns"] if t["clock"]]
  n, inside = sum(c[0] for c in got), sum(c[1] for c in got)
  return {"check": "the capture's clock is the timer's", "ok": n > 0 and inside == n,
          "why": f"{inside:,} of {n:,} launch calls in the covered windows lie inside a timed record" if n else
                 "the capture covers no turn's decode window: its clock or its windows do not match the timer's"}


def overhead(step_ms:float | None, without:list[float], *, source:str, reference:str) -> dict[str, Any]:
  """A capture's or the timer's cost on the step against the mean step of runs without it (same flags, same turns),
  and their spread; the words and the verdict are collectors/replay_overhead's."""
  if not without:
    return {"value": None, "label": "unknown", "test": f"the same turns at the same flags without it, against {reference}, at least two of each"}
  if step_ms is None:
    raise Refused(f"the overhead against {reference} names runs without it but not the step with it (with_ms)")
  base = sum(without) / len(without)
  pct = 100 * (step_ms / base - 1)
  return {"value": pct / 100, "label": "measured", "source": source, "with_ms": step_ms, "without_ms": without,
          "spread": (max(without) - min(without)) / base if len(without) > 1 else None, "qualifies": within_ceiling(pct), "words": overhead_words(pct, reference)}
