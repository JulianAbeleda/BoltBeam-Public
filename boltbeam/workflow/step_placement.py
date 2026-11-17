"""Step placement: for every host segment of a captured decode step, whether it can leave the critical path and where its
operations run, and what that saves, from a replay of the step (docs/step-placement-design-20261115.md).

    steps      the decode thread's steps of one run (the host-function timer's records, collectors/host_functions.read):
               a step runs from one call of the row's scope label on its first-named context to the next, inside a
               turn's decode window. Its atoms: the time between two consecutive record edges (an entry or an exit),
               each with the labels open in it.
    nodes      one traced step as the decode thread's order (collectors/time_budget.Timeline, its call_threads and
               correlations): host pieces and API calls, each call with the device work it released (its kernels and
               copies by correlation id: one block, its span, shifted as a unit). Host pieces take their durations
               from an untraced timer run of the same step, aligned edge by edge (a step whose edge sequence differs is
               refused): an atom's untraced time less the traced calls in it, spread over its pieces as traced. An
               atom that holds a blocking call (a sync, a pageable copy) keeps its traced pieces: its untraced time
               holds a wait the replay recomputes. No parameter is fitted.
    replay     host clock h, device clock g; a host piece or call advances h; a launch releases its block at h plus
               the call's traced latency, the block starts at max(release, g) and g moves to its end; a blocking call
               sets h = max(h, g) plus its traced tail. The step is h at its end. An edit scales or drops the host
               pieces and calls under a set of labels (dropped calls take their blocks with them).
    gates      G1: replay(today) equals the untraced step of the same steps within the record's repeat band, per
               window. G2: an edit's predicted saving lands in the measured interval of a fix that was built and timed.
               Until both pass the replay decides nothing: the saving is host_functions' idle part, an upper bound.

Engine-blind: labels, the scope and the blocking-call rule are data (data/host_functions.json, data/gap_causes.json)."""
from __future__ import annotations

import bisect
import pathlib
import re
import collections
import json
from typing import Any

from boltbeam.collectors import time_budget as tb
from boltbeam.core.refused import Refused

SCHEMA = "boltbeam.step_placement.v1"


def decode_thread(threads:dict[int, list], scope:str) -> list[tuple[int, int, str, int]]:
  """The thread with the most scope calls, its records sorted (start, -end, label, first argument)."""
  return sorted(max(threads.values(), key=lambda v: sum(r[2] == scope for r in v), default=[]))


def steps(recs:list[tuple[int, int, str, int]], row:dict[str, Any], turns:list[dict[str, Any]]) -> dict[Any, list[tuple[int, int]]]:
  """Per turn (its `i`), the steps [start, next start) of the scope label on the first-named (rarest) context inside the
  turn's decode window (from its first `start` call to its span's end). The last call of a turn opens no full step."""
  scope = row["scope"]["label"]
  calls = collections.Counter(r[3] for r in recs if r[2] == scope)
  if not calls:
    raise Refused(f"no {scope} call in the records: nothing cuts the steps")
  first = min(calls, key=calls.get)
  starts, out = [r[0] for r in recs], {}
  for t in turns:
    lo, hi = (bisect.bisect_left(starts, x) for x in t["span_ns"])
    w0 = next((recs[i][0] for i in range(lo, hi) if recs[i][2] == row["start"]), None)
    if w0 is None:
      continue
    b = [recs[i][0] for i in range(lo, hi) if recs[i][2] == scope and recs[i][3] == first and recs[i][0] >= w0]
    out[t["i"]] = list(zip(b, b[1:]))
  return out


def parents(recs:list[tuple[int, int, str, int]]) -> list[int | None]:
  """Each record's enclosing record (index), by one pass over the nested records."""
  out, open_ = [], []
  for t0, nt1, *_ in recs:
    while open_ and -recs[open_[-1]][1] <= t0:
      open_.pop()
    out.append(open_[-1] if open_ else None)
    open_.append(len(out) - 1)
  return out


def edges(recs:list[tuple[int, int, str, int]], up:list[int | None], starts:list[int], a:int, b:int) -> list[tuple[int, str, tuple[str, ...]]]:
  """The record edges in [a, b), a being a record's start: (time, '+label' or '-label', the labels open after it),
  headed by a and closed by b. The records open at a are the chain enclosing the record that starts there."""
  lo, hi = bisect.bisect_left(starts, a), bisect.bisect_left(starts, b)
  chain, p = [], up[lo] if lo < len(recs) else None
  while p is not None:
    chain.insert(0, p)
    p = up[p]
  ev = [(r[0], 1, r[2]) for r in recs[lo:hi]] + [(-r[1], 0, r[2]) for r in (*(recs[c] for c in chain), *recs[lo:hi]) if a <= -r[1] < b]
  stack = [recs[c][2] for c in chain]
  out = [(a, "", tuple(stack))]
  for t, kind, label in sorted(ev, key=lambda e: (e[0], e[1])):
    if kind:
      stack.append(label)
    else:
      del stack[len(stack) - 1 - stack[::-1].index(label)]
    out.append((t, ("+" if kind else "-") + label, tuple(stack)))
  return out + [(b, "", tuple(stack))]


def blocks(tl:tb.Timeline, origin:int) -> dict[int, tuple[int, int, Any]]:
  """Device work by launching call (correlation id): (first start, last end) on the host clock, and its stream (one
  launch's work runs on one stream; a copy engine runs beside the compute queue, so each stream is its own queue)."""
  out: dict[int, list] = {}
  work = [(a, e, st, c) for (a, e, _, st), c in zip(tl.kernels, tl.kernel_correlations)]
  work += [(x["start_ns"], x["end_ns"], x["stream"], c) for x, c in zip(tl.copies, tl.copy_correlations)]
  for a, e, st, c in work:
    if c is None:
      continue
    w = out.setdefault(c, [a + origin, e + origin, st])
    w[0], w[1] = min(w[0], a + origin), max(w[1], e + origin)
  return {c: tuple(w) for c, w in out.items()}


def call_kinds(tl:tb.Timeline, table:dict[str, Any] | None = None) -> list[str]:
  """Each call's cause (data/gap_causes.json, decode rows first), `wait` for a call that holds the host until device
  work is done: a synchronize (stream, event, device), or a copy call into pageable host memory."""
  pattern, cpu, _ = tb.causes(table, True)
  holds = re.compile((table or tb._table("gap_causes.json"))["host_waits"]["call"])  # the data row: which sync calls hold the host
  memo: dict[str, str] = {}
  kind = lambda n: memo.setdefault(n, next((c for p, c in pattern if p.search(n)), cpu))  # noqa: E731
  pageable = {c for x, c in zip(tl.copies, tl.copy_correlations) if x.get("dst_memory") == "pageable"}
  return ["wait" if (k == "sync" and holds.search(n)) or (k == "copy_call" and c in pageable) else k
          for (_, _, n), c in zip(tl.calls, tl.call_correlations) for k in (kind(n),)]


def thread_calls(tl:tb.Timeline, thread:int, kinds:list[str], work:dict[int, tuple[int, int]]) -> dict[str, Any]:
  """The thread's calls on the host clock in start order: start, end, cause, block; for a wait, the call whose block it
  waited for (the latest-ending block that ended by the time it returned: an event or one stream's synchronize waits
  for less than the whole queue); and each block's shape (latency, overlap, length, stream): a block whose stream was
  idle starts its traced latency after its call; one queued behind its stream's work starts when that work ends, less
  the overlap it had with it (a dependent launch may start before the previous one ends). Nothing is fitted."""
  o, rows = tl.monotonic_origin_ns, []
  for (s, e, n), k, t, c in zip(tl.calls, kinds, tl.call_threads, tl.call_correlations):
    if t == thread:
      rows.append((s + o, e + o, n, k, work.get(c)))
  rows.sort(key=lambda r: r[0])
  g, ends, waits, shape, clocks = collections.defaultdict(int), [], [], [], []
  for i, (s, e, n, k, blk) in enumerate(rows):
    clocks.append(dict(g))  # each stream's end of released work as this call starts (a step's g0)
    if blk:  # its stream idle: it starts its traced latency after the call; else when the stream's last work ends, less
      busy = g[blk[2]]  # the overlap it had with that work (a launch that may start before the previous one ends)
      shape.append((blk[0] - s, 0, blk[1] - blk[0], blk[2]) if blk[0] >= busy else (0, busy - blk[0], blk[1] - blk[0], blk[2]))
      g[blk[2]] = max(busy, blk[1])
      bisect.insort(ends, (blk[1], i))
    else:
      shape.append(None)
    j = bisect.bisect_right(ends, (e, len(rows))) - 1 if k == "wait" else -1
    waits.append(ends[j] if j >= 0 else None)
  return {"start": [r[0] for r in rows], "end": [r[1] for r in rows], "name": [r[2] for r in rows], "kind": [r[3] for r in rows],
          "block": [r[4] for r in rows], "shape": shape, "waits": waits, "clocks": clocks}


def nodes(calls:dict[str, Any], traced:list, untraced:list | None = None) -> dict[str, Any]:
  """One step's nodes from its traced edges (edges()) and, when given, the untraced run's edges of the same step.
  A node: ('host', ns, labels) or ('call', ns, labels, block shape (thread_calls) or None, kind, tail, awaited): a wait's
  tail is its traced time after the work it waited for ended (its own cost), awaited ('node', index) of that work's
  call in this step or ('at', its traced end from the step's start) for work released before the step."""
  if untraced is not None and [e[1] for e in traced] != [e[1] for e in untraced]:
    raise Refused("the step's record edges differ between the traced and the untraced run")
  a, b = traced[0][0], traced[-1][0]
  st = calls["start"]
  ci, hi = bisect.bisect_left(st, a), bisect.bisect_left(st, b)
  first, at = ci, {}
  g0 = {st: e - a for st, e in (calls["clocks"][ci] if ci < len(st) else {}).items() if e > a}  # work still running from before the step
  out, kept = [], [0, 0]
  for j in range(len(traced) - 1):
    t0, t1, labels = traced[j][0], traced[j + 1][0], traced[j][2]
    k0 = ci
    while ci < hi and st[ci] < t1:
      ci += 1
    inside = range(k0, ci)
    gaps, x = [], t0
    for i in inside:
      gaps.append(max(0, st[i] - x))
      x = max(x, calls["end"][i])
    gaps.append(max(0, t1 - x))
    if untraced is not None and not any(calls["kind"][i] == "wait" for i in inside):  # a wait's untraced time is not host time
      host = max(0, untraced[j + 1][0] - untraced[j][0] - sum(calls["end"][i] - st[i] for i in inside))
      tot = sum(gaps)
      share = [host * q // tot for q in gaps] if tot else [host] + [0] * (len(gaps) - 1)
      share[max(range(len(gaps)), key=gaps.__getitem__)] += host - sum(share)  # the floor's remainder: the sum is the untraced time
      gaps = share
      kept[0] += sum(gaps)
    else:
      kept[1] += sum(gaps)
    for q, i in zip(gaps, [*inside, None]):
      if q:
        out.append(("host", q, labels))
      if i is not None:
        s, e, blk, w = st[i], calls["end"][i], calls["block"][i], calls["waits"][i]
        at[i] = len(out)
        awaited = None if w is None else ("node", at[w[1]]) if w[1] >= first else ("at", max(0, w[0] - a))
        out.append(("call", e - s, labels, calls["shape"][i], calls["kind"][i],
                    max(0, e - max(s, w[0])) if w else e - s, awaited))
  return {"nodes": out, "g0": g0, "host_untraced_ns": kept[0], "host_traced_ns": kept[1], "traced_ns": b - a,
          "untraced_ns": untraced[-1][0] - untraced[0][0] if untraced is not None else None}


def replay(ns:list[tuple], g0:dict[Any, int] | None = None, edit:dict[str, float] | None = None, drop:set[int] = frozenset(), own:bool = False) -> int:
  """The step's length under an edit. edit {label: scale}: host pieces and calls under a label are scaled (0 drops
  them, and a dropped call's device work with it); own: only host pieces whose innermost label is named (a function's
  own work: its callees and the calls it makes stay). drop (node indexes, a cut): its host pieces and waits go, and
  its launches release their work with no host time (one captured graph). Each stream is a queue with its own clock,
  starting at g0's (work still running from before the step)."""
  edit, h, g, done = edit or {}, 0, collections.defaultdict(int, g0 or {}), {}
  for k, n in enumerate(ns):
    f = (edit.get(n[2][-1], 1.0) if n[0] == "host" and n[2] else 1.0) if own else min((edit[x] for x in n[2] if x in edit), default=1.0)
    done[k] = g[n[3][3]] if n[0] == "call" and n[3] else 0  # a dropped launch: a wait on it waits for its stream's earlier work
    if n[0] == "host" or f == 0:
      h += 0 if k in drop else n[1] * f
      continue
    _, dur, _, blk, kind, tail, awaited = n
    if blk:
      lat, ov, length, st = blk
      done[k] = max(h + (0 if k in drop else lat), g[st] - ov) + length
      g[st] = max(g[st], done[k])
    if k in drop:
      continue
    if kind != "wait":
      h += dur * f
    elif awaited is None:
      h += tail * f
    else:
      h = max(h, done[awaited[1]] if awaited[0] == "node" else awaited[1]) + tail * f
  return max([h, *g.values()]) if drop else h  # a cut removed: the host no longer waits, so the device's work bounds the step


# --- the rule ------------------------------------------------------------------------------------------------------

def rates(recs:list[tuple[int, int, str, int]], starts:list[int], per_turn:dict[Any, list[tuple[int, int]]], scope:str) -> dict[str, str]:
  """Each label's change rate (R3) from its calls per step: per_forward when it runs at least once per forward (the
  scope's calls in the step), per_step when less but in steps after a turn's first, per_turn when only in first steps."""
  later, first, fwd, n = collections.Counter(), collections.Counter(), 0, 0
  for ss in per_turn.values():
    for k, (a, b) in enumerate(ss):
      seen = collections.Counter(r[2] for r in recs[bisect.bisect_left(starts, a):bisect.bisect_left(starts, b)])
      (later if k else first).update(seen)
      if k:
        fwd, n = fwd + seen[scope], n + 1
  per = fwd / n if n else 0
  return {lb: "per_forward" if n and later[lb] / n >= per else "per_step" if later[lb] else "per_turn" for lb in {*later, *first}}


def side(cls:dict[str, Any], rate:str) -> str:
  """Where a class's work goes (R1-R4): device, host_after_step (lag is harmless, or slower than per step),
  host_after_launch (host-only work, off the path once the next step is launched first), home (it stays)."""
  if cls.get("boundary"):
    return "boundary"
  if cls.get("host_only"):
    return "host_after_launch"
  if cls["device_ok"] and rate in ("per_forward", "per_step"):
    return "device"
  return "host_after_step" if cls.get("lag_ok") or rate == "per_turn" else "home"


def cuts(ns:list[tuple], op_of) -> list[dict[str, Any]]:
  """The host segments between two GPU stages: from a wait to the next launch that releases device work (both
  included). op_of(labels) -> the class of a host piece's innermost labelled function (None: no timed function)."""
  out, cur = [], None
  for k, n in enumerate(ns):
    if n[0] == "call" and n[4] == "wait":
      if cur is None:
        cur = {"nodes": [], "ops": set(), "host_ns": 0}
      cur["nodes"].append(k)  # a second wait before any launch stays in the same cut
      continue
    if cur is None:
      continue
    cur["nodes"].append(k)
    if n[0] == "host":
      cur["ops"].add(op_of(n[2]))
      cur["host_ns"] += n[1]
    elif n[3]:
      out.append(cur)
      cur = None
  return out


def removable(c:dict[str, Any], sides:dict[Any, str], can:dict[Any, str]) -> str:
  """A cut leaves the path only if every operation in it can leave: yes, no, or unknown (an untested precondition)."""
  ops = {o for o in c["ops"] if o is None or sides.get(o) != "boundary"}  # the read that bounds a cut is not one of its operations
  if any(o is not None and sides.get(o, "home") == "home" for o in ops):
    return "no"
  return "unknown" if None in ops or any(can.get(o) != "yes" for o in ops) else "yes"  # no timed function: unclassified


# --- one run's steps as replay inputs, the gates, the table -------------------------------------------------------

def _records(path:pathlib.Path, scope:str) -> tuple[list, list, list[int], int]:
  from boltbeam.collectors import host_functions as hf
  with open(path) as f:
    th = hf.read(f, str(path))
  tid = max(th, key=lambda t: sum(r[2] == scope for r in th[t]), default=None)
  recs = sorted(th.get(tid, []))
  return recs, parents(recs), [r[0] for r in recs], tid


def window(traced:dict[str, Any], untraced:dict[str, Any], capture:pathlib.Path, row:dict[str, Any], turns:list[int]) -> dict[str, Any]:
  """The window's steps as nodes, aligned run to run. traced / untraced: {records (path), turns [{i, span_ns}]}: the
  timer under the capture, and the timer alone, on the same tokens. A turn whose step counts differ, or a step whose
  edges differ, is refused by name, never guessed."""
  scope = row["scope"]["label"]
  rt, ru = _records(traced["records"], scope), _records(untraced["records"], scope)
  pick = lambda run: [t for t in run["turns"] if t["i"] in turns]  # noqa: E731
  st, su = steps(rt[0], row, pick(traced)), steps(ru[0], row, pick(untraced))
  tl = tb.timeline_from_nsys(capture)
  if not tl.call_correlations or tl.call_correlations[0] is None or (tl.kernels and tl.kernel_correlations[0] is None):
    raise Refused(f"{capture}: the export carries no correlation ids, so no launch can be tied to its work")
  thread = next((t for t in set(tl.call_threads) if t is not None and t & 0xFFFFFF == rt[3]), None)  # nsys globalTid: the tid in its low 24 bits
  if thread is None:
    raise Refused(f"{capture}: no traced call is on the timer's decode thread {rt[3]}: the two runs are not one process")
  calls = thread_calls(tl, thread, call_kinds(tl), blocks(tl, tl.monotonic_origin_ns))
  out, refused = [], []
  for i in turns:
    if len(st.get(i, [])) != len(su.get(i, [])) or not st.get(i):
      refused.append(f"turn {i}: {len(st.get(i, []))} traced steps against {len(su.get(i, []))} untraced")
      continue
    for k, ((a, b), (c, d)) in enumerate(zip(st[i], su[i])):
      try:
        out.append({"turn": i, "k": k, **nodes(calls, edges(*rt[:3], a, b), edges(*ru[:3], c, d))})
      except Refused as exc:
        refused.append(f"turn {i} step {k}: {exc}")
  return {"steps": out, "refused": refused, "rates": rates(ru[0], ru[2], su, scope)}


def saving(win:dict[str, Any], edit:dict[str, float] | None = None, drop=lambda s: frozenset(), own:bool = False) -> dict[str, float]:
  """replay(today) - replay(edited), ms per step over the window's steps: mean and standard deviation."""
  d = [(replay(s["nodes"], s["g0"]) - replay(s["nodes"], s["g0"], edit, drop(s), own)) / 1e6 for s in win["steps"]]
  if not d:
    raise Refused("no aligned step in the window: nothing to replay")
  m = sum(d) / len(d)
  return {"mean": m, "sd": (sum((x - m) ** 2 for x in d) / max(1, len(d) - 1)) ** 0.5, "steps": len(d)}


def fit(win:dict[str, Any], tol:dict[str, Any], name:str) -> dict[str, Any]:
  """G1: the replay of today's steps against the untraced steps. No parameter is fitted, but it is not wholly out of
  sample: atoms with no wait take the untraced run's own time, so only the waits and the atoms that hold one (their
  traced host pieces, call times, latencies and tails) can move the replay off it; the record says that share."""
  n = len(win["steps"])
  share = sum(s["host_untraced_ns"] for s in win["steps"]) / max(1, sum(s["untraced_ns"] for s in win["steps"]))
  rep = sum(replay(s["nodes"], s["g0"]) for s in win["steps"]) / n / 1e6 if n else None
  un = sum(s["untraced_ns"] for s in win["steps"]) / n / 1e6 if n else None
  ok = n > 0 and tol["value"] is not None and abs(rep - un) <= tol["value"]
  return {"gate": f"G1 {name}", "ok": ok, "label": "derived", "formula": "mean replay of the traced steps with untraced host times, against the "
          "untraced steps; within the record's repeat band", "inputs": "the window's aligned steps, the band", "replay_ms": rep, "untraced_ms": un,
          "by_construction": share,
          "steps": n, "band": tol, "refused": win["refused"],
          "why": (f"replay {rep:.3f} against untraced {un:.3f} ms a step over {n} steps (host pieces from the untraced run: {share:.0%} of it); " if n else "no step aligned; ")
          + (f"tolerance unknown: {tol['test']}" if tol["value"] is None else f"{'within' if ok else 'beyond'} the band {tol['value']:.3f} ({tol['source']})")}


def retro(win:dict[str, Any], fix:dict[str, Any]) -> dict[str, Any]:
  """G2: an edit's predicted saving against the paired interval of the fix it stands for; it discriminates only when
  the naive prediction (the fix's host time in GPU-idle gaps, or its CPU time) lies outside that interval."""
  iv, got = fix["interval"], saving(win, fix["edit"])
  inside = lambda x: iv["lo"] <= x <= iv["hi"]  # noqa: E731
  naive = fix.get("naive_ms")
  ok = inside(got["mean"]) and naive is not None and not inside(naive)
  return {"gate": f"G2 {fix['name']}", "ok": ok, "label": "derived", "formula": "replay(today) - replay(edit), against the fix's paired interval",
          "inputs": f"the fix-off capture and timer runs; the interval: {iv['source']}", "predicted_ms": got, "interval": iv, "naive_ms": naive,
          "why": f"predicted {got['mean']:.3f} ms against {iv['mean']:.3f} [{iv['lo']:.3f}, {iv['hi']:.3f}] ({iv['source']}); "
                 + ("naive unknown: the gate cannot discriminate" if naive is None else f"naive {naive:.3f} {'inside' if inside(naive) else 'outside'}")}


def place(win:dict[str, Any], ops:dict[str, str], classes:dict[str, Any], *, decides:bool, idle:dict[str, float] | None, source:str) -> dict[str, Any]:
  """Per operation class: its labels and their rates, where its work goes, whether it can (yes / no / unknown with the
  tests), and what removing its host work saves: the replay's prediction when the gates passed (derived), else its
  host time in GPU-idle gaps (an upper bound, estimate). Then the cuts: how many a step, and what removing every cut
  that is not held by a home operation saves (replay; only when the gates passed)."""
  from boltbeam.workflow.context_audit import derived, unknown
  by: dict[str, list[str]] = collections.defaultdict(list)
  for lb in win["rates"]:
    if lb in ops:
      by[ops[lb]].append(lb)
  rows, sides, can = [], {}, {}
  for cls, labels in sorted(by.items()):
    c = classes[cls]
    rate = max((win["rates"][lb] for lb in labels), key=("per_turn", "per_step", "per_forward").index)
    sides[cls] = sd = side(c, rate)
    tests = [p["test"] for p in c.get("preconditions") or []]
    if sd == "host_after_step" and not c.get("lag_ok"):
      tests.append("a step of lag is harmless for this class (it changes per turn here): the identity check with it moved after the step")
    can[cls] = "no" if sd in ("home", "boundary") else "unknown" if tests else "yes"
    edit = {lb: 0.0 for lb in labels}
    if sd == "boundary":
      save = None
    elif decides:
      g = saving(win, edit, own=True)
      save = {**derived(g["mean"], "replay(today) - replay(the class's own host work removed; its calls and callees stay), mean over the window's steps", source), "sd": g["sd"]}
    elif idle is not None:
      save = {"value": sum(idle.get(lb, 0.0) for lb in labels), "label": "estimate",
              "formula": "the class's host time in GPU-idle gaps (host_functions), an upper bound", "test": "the replay's gates G1 and G2"}
    else:
      save = unknown("a capture of the window with the timer underneath (host_functions idle part) or the replay's gates")
    rows.append({"class": cls, "labels": sorted(labels), "rate": rate, "side": sd, "can": can[cls], "tests": tests, "why": c["why"], "saving_ms": save})
  op_of = lambda labels: ops.get(labels[-1]) if labels else None  # noqa: E731  (the innermost function's own class; none: unclassified)
  per = [cuts(s["nodes"], op_of) for s in win["steps"]]
  count = collections.Counter(removable(c, sides, can) for cs in per for c in cs)
  n = len(per) or 1
  out = {"rows": rows, "cuts": {k: derived(v / n, "cuts a step by whether they can leave the path", source) for k, v in count.items()}}
  if decides:
    keep = {id(s): {k for c in cs[:-1] if removable(c, sides, can) != "no" for k in c["nodes"]} for s, cs in zip(win["steps"], per)}  # the last: the one read
    g = saving(win, drop=lambda s: keep[id(s)])
    out["all_cuts_ms"] = {**derived(g["mean"], "replay(today) - replay(every cut not held by a home operation removed, the step's last read kept), if every unknown precondition holds", source), "sd": g["sd"]}
  return out


def idle_parts(traced:dict[str, Any], capture:pathlib.Path, row:dict[str, Any], turns:list[int]) -> dict[str, float]:
  """Each label's host time in GPU-idle gaps, ms per step over the window's turns (collectors/host_functions)."""
  from boltbeam.collectors import host_functions as hf
  with open(traced["records"]) as f:
    th = hf.read(f, str(traced["records"]))
  folded = hf.fold(th, [t for t in traced["turns"] if t["i"] in turns], row, hf.capture([capture]))
  got = [t for t in folded["turns"] if t["idle"]]
  n = sum(t["steps"] for t in got)
  ns = collections.Counter()
  for t in got:
    ns.update({k.split("@")[0]: v for k, v in t["idle"]["ns"].items()})
  return {k: v / n / 1e6 for k, v in ns.items()} if n else {}


def section(spec:dict[str, Any], rec:dict[str, Any], *, engine:str, base:pathlib.Path) -> dict[str, Any]:
  """The context audit's placement section. spec: release; traced and untraced {records, turns [{i, span_ns}]} (the
  timer under the capture, the timer alone, same tokens); graph (the timer under a graph-level capture, optional);
  windows [{name, turns, capture, graph_capture?}]; fixes [{name, window,
  edit {label: scale}, interval {mean, lo, hi, source}, naive_ms, traced, untraced, capture}] (G2: fixes built and
  timed, captured with the fix off). The record's repeats of `step_ms <window>` give G1's band."""
  from boltbeam.collectors import host_functions as hf
  from boltbeam.workflow.context_audit import band
  row = hf.row(engine, spec["release"])
  ops = {f["label"]: f["op"] for f in row["functions"] if "op" in f}
  classes = json.loads((pathlib.Path(__file__).resolve().parents[1] / "data" / "step_ops.json").read_text())["classes"]
  at = lambda run: {**run, "records": base / run["records"]}  # noqa: E731
  wins = {w["name"]: (w, window(at(spec["traced"]), at(spec["untraced"]), base / w["capture"], row, w["turns"])) for w in spec["windows"]}
  g1 = [fit(win, band(rec, f"step_ms {w['name']}"), w["name"]) for w, win in wins.values()]
  g1 += [fit(window(at(spec["graph"]), at(spec["untraced"]), base / w["graph_capture"], row, w["turns"]), band(rec, f"step_ms {w['name']}"), f"{w['name']} (graph-level capture)")
         for w, _ in wins.values() if w.get("graph_capture") and spec.get("graph")]
  if not any("graph-level" in g["gate"] for g in g1):
    g1.append({"gate": "G1 graph-level", "ok": False, "label": "unknown", "value": None, "why": "no graph-level capture of the same turns",
               "test": "the window captured again with graph-level tracing (node tracing inflates launches and in-graph gaps)"})
  g2 = [retro(window(at(f["traced"]), at(f["untraced"]), base / f["capture"], row, wins[f["window"]][0]["turns"]), f) for f in spec.get("fixes") or []]
  decides = bool(g2) and all(g["ok"] for g in [*g1, *g2])
  out = {"schema": SCHEMA, "gates": [*g1, *g2], "decides": decides, "windows": {}}
  for name, (w, win) in wins.items():
    idle = idle_parts(at(spec["traced"]), base / w["capture"], row, w["turns"])
    out["windows"][name] = {"label": "measured", "source": f"{spec['traced']['records']}, {w['capture']}, {spec['untraced']['records']}",
                            "turns": w["turns"], "steps": len(win["steps"]), "refused": win["refused"],
                            **place(win, ops, classes, decides=decides, idle=idle, source=f"{spec['traced']['records']}, {w['capture']}, {spec['untraced']['records']}")}
  return out
