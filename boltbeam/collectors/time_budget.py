"""Where one request's wall time went, from a GPU trace: the time budget.

    wall = kernels (their union) + exposed copy + idle + host time outside the GPU window

The window runs from the first GPU operation to the last (kernel, copy or memset) inside the span asked for. Kernels are
the union of every kernel's span, split into families by a data table (data/kernel_families.json, rows keyed by the
engine and the GPU vendor). Exposed copy is copy time with no kernel running; idle is window time with neither a kernel
nor a copy in flight. Each idle gap above a threshold is attributed: its neighbouring kernels and, per instant, the GPU
API call in flight on the host (data/gap_causes.json; no call in flight is CPU work). Host time outside the window is
the client's wall time for the request less the window.

The four parts are measured each on its own (idle is the sum of the gaps, not the window less the rest), and the record
says how far their sum is from the wall. It never forces the sum: what does not add up is `unattributed_ms`.

Engine-neutral by rule (policy/guards.py, budget core): nothing in this module names an engine or a kernel; an engine
and a vendor are keys into the data tables. The trace is a Timeline (kernels, copies, API calls, in ns); nsys's sqlite
export is one reader of it (timeline_from_nsys)."""
from __future__ import annotations

import bisect
import heapq
import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, NamedTuple

SCHEMA = "boltbeam.time_budget.v1"
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
GAP_MS = 1.0  # an idle gap at least this long is attributed one by one; shorter gaps are summed as small gaps
TOLERANCE_PCT = 1.0  # the parts must sum to the wall within this share of it (or TOLERANCE_MIN_MS, the larger)
TOLERANCE_MIN_MS = 2.0
LIST_GAPS = 12  # the largest gaps listed one by one
OTHER_NAMES = 8  # the largest kernels no family row took, listed so a missing row is seen


@dataclass
class Timeline:
  """One GPU trace, session-relative ns. kernels: (start, end, name, stream); copies: {start_ns, end_ns, bytes, kind,
  memory, stream}; calls: (start, end, name) host API calls; clock: how the trace's ns map to the host's monotonic clock."""
  kernels: list[tuple[int, int, str, Any]] = field(default_factory=list)
  copies: list[dict[str, Any]] = field(default_factory=list)
  calls: list[tuple[int, int, str]] = field(default_factory=list)
  monotonic_origin_ns: int | None = None  # host CLOCK_MONOTONIC at trace ns 0, when the reader knows it
  source: str = ""


_COPY_KINDS = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
_MEM_KINDS = {0: "pageable", 1: "pinned", 2: "device"}
_SHORT = {"Host-to-Device": "HtoD", "Device-to-Host": "DtoH", "Device-to-Device": "DtoD", "Peer-to-Peer": "PtoP"}


def timeline_from_nsys(db:str | pathlib.Path) -> Timeline:
  """An nsys capture's sqlite export as a Timeline: CUPTI kernel, memcpy, memset and runtime/driver API rows."""
  import sqlite3
  with sqlite3.connect(db) as con:
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    has = tables.__contains__
    strings = dict(con.execute("select id, value from StringIds")) if has("StringIds") else {}
    enum = lambda t, d: {i: n for i, _, n in con.execute(f"select * from {t}")} if has(t) else d  # noqa: E731
    oper, mem = enum("ENUM_CUDA_MEMCPY_OPER", _COPY_KINDS), enum("ENUM_CUDA_MEM_KIND", _MEM_KINDS)
    mk = lambda m: str(mem.get(m, m)).lower().replace("cuda_memopr_memory_kind_", "")  # noqa: E731
    kernels = [(a, e, str(strings.get(n, n)), st) for a, e, n, st in
               con.execute("select start, end, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL")] if has("CUPTI_ACTIVITY_KIND_KERNEL") else []
    copies = [{"start_ns": a, "end_ns": e, "bytes": b, "kind": _SHORT.get(oper.get(k), oper.get(k, str(k))), "memory": mk(src), "stream": st}
              for a, e, b, k, src, st in con.execute("select start, end, bytes, copyKind, srcKind, streamId from CUPTI_ACTIVITY_KIND_MEMCPY")] \
             if has("CUPTI_ACTIVITY_KIND_MEMCPY") else []
    copies += [{"start_ns": a, "end_ns": e, "bytes": b, "kind": "memset", "memory": "device", "stream": st}
               for a, e, b, st in con.execute("select start, end, bytes, streamId from CUPTI_ACTIVITY_KIND_MEMSET")] \
              if has("CUPTI_ACTIVITY_KIND_MEMSET") else []
    calls = [(a, e, re.sub(r"_v\d+$", "", str(strings.get(n, n)))) for a, e, n in
             con.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")] if has("CUPTI_ACTIVITY_KIND_RUNTIME") else []
    origin = None
    if has("TARGET_INFO_SESSION_START_TIME"):
      cols = [d[0] for d in con.execute("select * from TARGET_INFO_SESSION_START_TIME").description]
      row = con.execute("select * from TARGET_INFO_SESSION_START_TIME").fetchone()
      if row and "systemClockNs" in cols:
        origin = int(row[cols.index("systemClockNs")])  # the session's start on the host's CLOCK_MONOTONIC
  return Timeline(kernels, copies, sorted(calls), origin, str(db))


# --- the data tables ----------------------------------------------------------------------------------------------

def _table(name:str) -> dict[str, Any]:
  return json.loads((DATA / name).read_text())


class Rule(NamedTuple):
  """A family row: a launch whose name matches `pattern` goes to `family`. A row with `after` matches a launch only when,
  on the same stream, a kernel whose name matches `after` ran since this row last matched there: the name alone does not
  tell the launch's role, the call before it does (one helper launch, then the launch it prepares)."""
  pattern: re.Pattern
  family: str
  after: re.Pattern | None = None


def families_for(engine:str | list[str] | None, vendor:str | None, table:dict[str, Any] | None = None) -> list[Rule]:
  """The family rows for a trace of this engine on this vendor's GPU, in the order they are tried: the engine's (or each
  key's, given a list: a fork, then its root), the engines it lists in `also`, the vendor's, then `any`. Fails closed on
  a row naming a family the table does not list."""
  t = table or _table("kernel_families.json")
  rows: list[dict[str, Any]] = []
  seen: set[str] = set()

  def add(key:str | None) -> None:
    e = (t.get("engines") or {}).get(key or "")
    if not e or key in seen:
      return
    seen.add(key)
    rows.extend(e.get("rows") or [])
    for other in e.get("also") or []:
      add(other)

  for key in engine_keys(engine):
    add(key)
  rows += ((t.get("vendors") or {}).get(vendor or "") or {}).get("rows") or []
  rows += t.get("any") or []
  bad = [r for r in rows if r["family"] not in t["families"]]
  if bad:
    raise ValueError(f"kernel_families.json: a row names a family the table does not list: {bad[0]}")
  return [Rule(re.compile(r["pattern"]), r["family"], re.compile(r["after"]) if r.get("after") else None) for r in rows]


def engine_keys(engine:str | list[str] | None) -> list[str]:
  return [k for k in ([engine] if isinstance(engine, str) or engine is None else engine) if k]


def unknown_engines(engine:str | list[str] | None, table:dict[str, Any] | None = None) -> list[str]:
  """The engine keys the family table has no rows for: their kernels fall to the vendor rows or `other`."""
  t = table or _table("kernel_families.json")
  return [k for k in engine_keys(engine) if k not in (t.get("engines") or {})]


def engine_names(table:dict[str, Any] | None = None) -> list[str]:
  """The engine keys the family table has rows for (directly or through `also`)."""
  return sorted((table or _table("kernel_families.json")).get("engines") or {})


def family_of(name:str, rules:list[Rule]) -> str:
  """The family of a name on its own: rows that need the call before a launch (`after`) do not apply."""
  return next((r.family for r in rules if r.after is None and r.pattern.search(name)), "other")


def launch_families(kernels:list[tuple[int, int, str, Any]], rules:list[Rule]) -> list[str]:
  """Each launch's family, in the order given: launches are walked per stream in start order, so a row with `after`
  sees the kernels before the launch on its stream."""
  armed: dict[tuple[int, Any], bool] = {}  # (row, stream): its `after` kernel ran since the row last matched there
  ctx = [i for i, r in enumerate(rules) if r.after is not None]
  out = [""] * len(kernels)
  for j in sorted(range(len(kernels)), key=lambda j: (kernels[j][0], kernels[j][1])):
    _, _, name, stream = kernels[j]
    fam = None
    for i, r in enumerate(rules):
      if not r.pattern.search(name):
        continue
      if r.after is None:
        fam = r.family
        break
      if armed.get((i, stream)):
        armed[(i, stream)] = False
        fam = r.family
        break
    for i in ctx:
      if rules[i].after.search(name):
        armed[(i, stream)] = True
    out[j] = fam or "other"
  return out


def causes(table:dict[str, Any] | None = None) -> tuple[list[tuple[re.Pattern, str]], str, dict[str, str]]:
  t = table or _table("gap_causes.json")
  bad = [r for r in t["rows"] if r["cause"] not in t["causes"]] + ([] if t["cpu"] in t["causes"] else [t])
  if bad:
    raise ValueError(f"gap_causes.json: a row names a cause the table does not list: {bad[0]}")
  return [(re.compile(r["call"]), r["cause"]) for r in t["rows"]], t["cpu"], dict(t["causes"])


# --- intervals ----------------------------------------------------------------------------------------------------

def union(spans) -> list[list[int]]:
  out: list[list[int]] = []
  for a, b in sorted(spans):
    if b <= a:
      continue
    if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
    else: out.append([a, b])
  return out


def meet(u:list[list[int]], v:list[list[int]]) -> int:
  i = j = t = 0
  while i < len(u) and j < len(v):
    t += max(0, min(u[i][1], v[j][1]) - max(u[i][0], v[j][0]))
    if u[i][1] < v[j][1]: i += 1
    else: j += 1
  return t


def length(u:list[list[int]]) -> int:
  return sum(b - a for a, b in u)


def clip(tl:Timeline, span:tuple[int, int] | None) -> Timeline:
  """The timeline cut to [a, b) ns: events outside dropped, events across an edge cut at it."""
  if span is None:
    return tl
  a, b = span
  c = lambda s, e: (max(s, a), min(e, b))  # noqa: E731
  kernels = [(*c(s, e), n, st) for s, e, n, st in tl.kernels if e > a and s < b]
  copies = [{**x, "start_ns": c(x["start_ns"], x["end_ns"])[0], "end_ns": c(x["start_ns"], x["end_ns"])[1]}
            for x in tl.copies if x["end_ns"] > a and x["start_ns"] < b]
  calls = [(*c(s, e), n) for s, e, n in tl.calls if e > a and s < b]
  return Timeline(kernels, copies, calls, tl.monotonic_origin_ns, tl.source)


def span_from_monotonic(tl:Timeline, start_ns:int, end_ns:int) -> tuple[int, int] | None:
  """A request's span on the host's monotonic clock (the driver's request_spans_ns) as trace ns, when the reader knows
  the trace's origin on that clock (nsys: TARGET_INFO_SESSION_START_TIME.systemClockNs)."""
  if tl.monotonic_origin_ns is None:
    return None
  return start_ns - tl.monotonic_origin_ns, end_ns - tl.monotonic_origin_ns


# --- the parts ----------------------------------------------------------------------------------------------------

def kernel_families(kernels:list[tuple[int, int, str, Any]], rules:list[Rule]) -> tuple[dict[str, int], dict[str, int]]:
  """The kernels' union split into families: each instant goes to the family of the kernel that started last among
  those running (exact on one stream; on several streams the overlap goes to the newer kernel). Also each `other`
  kernel name's busy ns, so the names no row took are listed."""
  fams = launch_families(kernels, rules)
  out: dict[str, int] = {}
  other: dict[str, int] = {}
  for (s, e, n, _), f in zip(kernels, fams):
    if f == "other":
      other[n] = other.get(n, 0) + max(0, e - s)
  ks = sorted((s, e, f) for (s, e, _, _), f in zip(kernels, fams) if e > s)
  points = sorted({p for s, e, _ in ks for p in (s, e)})
  heap: list[tuple[int, int, str]] = []  # (-start, end, family): the newest running kernel on top
  i = 0
  for a, b in zip(points, points[1:]):
    while i < len(ks) and ks[i][0] <= a:
      heapq.heappush(heap, (-ks[i][0], ks[i][1], ks[i][2]))
      i += 1
    while heap and heap[0][1] <= a:
      heapq.heappop(heap)
    if heap:
      f = heap[0][2]
      out[f] = out.get(f, 0) + (b - a)
  return out, other


def copy_rows(copies:list[dict[str, Any]], ku:list[list[int]]) -> list[dict[str, Any]]:
  """Per stream, kind and source memory: rows, bytes, busy ms (their union), GB/s over it, and exposed ms (busy time
  with no kernel running anywhere). Two streams' exposed copies can overlap, so the rows' exposed ms may sum to more
  than the budget's exposed copy."""
  groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
  for c in copies:
    groups.setdefault((str(c.get("stream")), c["kind"], c["memory"]), []).append(c)
  rows = []
  for (stream, kind, memory), cs in groups.items():
    u = union([(c["start_ns"], c["end_ns"]) for c in cs])
    busy, total = length(u), sum(int(c.get("bytes") or 0) for c in cs)
    rows.append({"stream": stream, "kind": kind, "memory": memory, "rows": len(cs), "bytes": total, "busy_ms": busy / 1e6,
                 "gbs": total / busy if busy else None, "exposed_ms": (busy - meet(u, ku)) / 1e6})
  return sorted(rows, key=lambda r: (-r["busy_ms"], r["stream"], r["kind"]))


def idle_gaps(au:list[list[int]]) -> list[tuple[int, int]]:
  return [(au[i][1], au[i + 1][0]) for i in range(len(au) - 1) if au[i + 1][0] > au[i][1]]


def attribute_gaps(gaps:list[tuple[int, int]], calls:list[tuple[int, int, str]], kernels:list[tuple[int, int, str, Any]],
                   cause_rows:list[tuple[re.Pattern, str]], cpu:str) -> list[dict[str, Any]]:
  """Each gap: its neighbouring kernels (the last to end before it, the first to start after it), the ns of each
  cause (per instant, the first cause row whose call is in flight; none in flight is cpu) and of each call name."""
  rank = {c: i for i, c in enumerate(dict.fromkeys(c for _, c in cause_rows))}
  cause_of: dict[str, str] = {}
  cause = lambda n: cause_of.setdefault(n, next((c for p, c in cause_rows if p.search(n)), None) or cpu)  # noqa: E731
  inflight: list[list[tuple[int, int, str]]] = [[] for _ in gaps]
  ends = [b for _, b in gaps]
  for s, e, n in calls:
    j = bisect.bisect_right(ends, s)
    while j < len(gaps) and gaps[j][0] < e:
      a, b = gaps[j]
      if min(e, b) > max(s, a):
        inflight[j].append((max(s, a), min(e, b), n))
      j += 1
  by_end = sorted((e, n) for _, e, n, _ in kernels)
  by_start = sorted((s, n) for s, _, n, _ in kernels)
  end_keys, start_keys = [e for e, _ in by_end], [s for s, _ in by_start]
  out = []
  for (a, b), cs in zip(gaps, inflight):
    pts = sorted({a, b, *(p for s, e, _ in cs for p in (s, e))})
    split: dict[str, int] = {}
    for x, y in zip(pts, pts[1:]):
      live = [cause(n) for s, e, n in cs if s <= x and e >= y]
      c = min(live, key=lambda k: rank.get(k, len(rank))) if live else cpu
      split[c] = split.get(c, 0) + (y - x)
    names: dict[str, int] = {}
    for s, e, n in cs:
      names[n] = names.get(n, 0) + (e - s)
    i, k = bisect.bisect_right(end_keys, a) - 1, bisect.bisect_left(start_keys, b)
    out.append({"start_ns": a, "end_ns": b, "causes": split, "calls": names,
                "before": by_end[i][1] if i >= 0 else None, "after": by_start[k][1] if k < len(by_start) else None})
  return out


def budget(tl:Timeline, *, wall_ms:float | None = None, engine:str | list[str] | None = None, vendor:str | None = None,
           span:tuple[int, int] | None = None, gap_ms:float = GAP_MS, tolerance_ms:float | None = None,
           families_table:dict[str, Any] | None = None, causes_table:dict[str, Any] | None = None,
           what:str | None = None) -> dict[str, Any]:
  """The time budget of one request (boltbeam.time_budget.v1, schemas/time_budget.schema.json). span: the request's
  [start, end) in trace ns (None: the whole trace). wall_ms: the client's wall time for it (None: host time outside the
  window is not known and the sum is checked over the window alone)."""
  tl = clip(tl, span)
  kspans = [(s, e) for s, e, _, _ in tl.kernels]
  cspans = [(c["start_ns"], c["end_ns"]) for c in tl.copies]
  everything = [x for x in kspans + cspans if x[1] > x[0]]
  base = {"schema": SCHEMA, "kind": "time_budget", "what": what, "engine": engine, "vendor": vendor, "source": tl.source,
          "span_ns": list(span) if span else None, "wall_ms": wall_ms}
  if not everything:
    return {**base, "refused": "the trace holds no GPU operation in this span"}
  t0, t1 = min(a for a, _ in everything), max(b for _, b in everything)
  window = t1 - t0
  ku, cu = union(kspans), union(cspans)
  au = union(everything)
  exposed = length(cu) - meet(cu, ku)
  gaps = idle_gaps(au)
  idle = sum(b - a for a, b in gaps)
  fams, other = kernel_families(tl.kernels, families_for(engine, vendor, families_table))
  rows, cpu, cause_words = causes(causes_table)
  big = [g for g in gaps if g[1] - g[0] >= gap_ms * 1e6]
  attributed = attribute_gaps(big, tl.calls, tl.kernels, rows, cpu)
  by_cause: dict[str, dict[str, Any]] = {}
  for g in attributed:
    for c, ns in g["causes"].items():
      row = by_cause.setdefault(c, {"cause": c, "words": cause_words.get(c, c), "ms": 0.0, "gaps": 0, "calls": {}})
      row["ms"] += ns / 1e6
      row["gaps"] += 1
    for n, ns in g["calls"].items():
      if r := by_cause.get(next((c for p, c in rows if p.search(n)), cpu)):
        r["calls"][n] = r["calls"].get(n, 0.0) + ns / 1e6
  for r in by_cause.values():
    r["calls"] = dict(sorted(((n, v) for n, v in r["calls"].items() if v > 0), key=lambda x: -x[1])[:4])
  kernels_ms, exposed_ms, idle_ms, window_ms = length(ku) / 1e6, exposed / 1e6, idle / 1e6, window / 1e6
  total = wall_ms if wall_ms is not None else window_ms
  tol = tolerance_ms if tolerance_ms is not None else max(TOLERANCE_MIN_MS, TOLERANCE_PCT / 100.0 * total)
  # a residual (see "host_how"); unknown (None), never 0, when the trace's GPU work is longer than the wall
  host_ms = None if wall_ms is None or window_ms > wall_ms + tol else max(0.0, wall_ms - window_ms)
  parts = kernels_ms + exposed_ms + idle_ms + (host_ms or 0.0)
  unattributed = total - parts
  ms = lambda ns: ns / 1e6  # noqa: E731
  return {**base,
          "window_ms": window_ms, "kernels_ms": kernels_ms, "exposed_copy_ms": exposed_ms, "idle_ms": idle_ms,
          "host_outside_ms": host_ms, "parts_ms": parts, "unattributed_ms": unattributed, "tolerance_ms": tol,
          "sums": abs(unattributed) <= tol,
          "host_how": None if wall_ms is None else "wall_less_window",  # with it the parts meet the wall by construction
          "window_fits_wall": None if wall_ms is None else window_ms <= wall_ms + tol,  # the check that is not an identity
          "engine_rows_missing": unknown_engines(engine, families_table),
          "trace_longer_than_wall_ms": (window_ms - wall_ms) if wall_ms is not None and window_ms > wall_ms else None,
          "duty": length(ku) / window if window else None,
          "families": [{"family": f, "ms": ms(v)} for f, v in sorted(fams.items(), key=lambda x: -x[1])],
          "other_kernels": [{"name": n, "busy_ms": ms(v)} for n, v in sorted(other.items(), key=lambda x: -x[1])[:OTHER_NAMES]],
          "copies": copy_rows(tl.copies, ku),
          "gaps": {"threshold_ms": gap_ms, "count": len(big), "ms": sum(ms(b - a) for a, b in big),
                   "small_count": len(gaps) - len(big), "small_ms": idle_ms - sum(ms(b - a) for a, b in big),
                   "by_cause": sorted(by_cause.values(), key=lambda r: -r["ms"]),
                   "largest": [{"at_ms": ms(g["start_ns"] - t0), "ms": ms(g["end_ns"] - g["start_ns"]), "before": g["before"],
                                "after": g["after"], "causes": {c: ms(v) for c, v in sorted(g["causes"].items(), key=lambda x: -x[1])},
                                "calls": {n: ms(v) for n, v in sorted(g["calls"].items(), key=lambda x: -x[1])[:3]}}
                               for g in sorted(attributed, key=lambda g: g["start_ns"] - g["end_ns"])[:LIST_GAPS]]}}


def from_nsys(db:str | pathlib.Path, **kw) -> dict[str, Any]:
  return budget(timeline_from_nsys(db), **kw)


def request_budget(db:str | pathlib.Path, driver:dict[str, Any], index:int, **kw) -> dict[str, Any] | None:
  """The budget of one timed request of a prefill driver's capture: its span from the driver's request_spans_ns (the
  host's monotonic clock) cut out of the trace, its wall from wall_samples_ms. None when the driver or the trace lacks
  what places the request (no spans, no clock origin)."""
  spans, walls = driver.get("request_spans_ns") or [], driver.get("wall_samples_ms") or []
  if not (0 <= index < len(spans) and index < len(walls)):
    return None
  tl = timeline_from_nsys(db)
  span = span_from_monotonic(tl, *spans[index])
  if span is None:
    return None
  return budget(tl, wall_ms=float(walls[index]), span=span, **kw)
