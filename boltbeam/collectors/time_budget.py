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
export is one reader of it (timeline_from_nsys).

A decode request (decode_budget) is the same budget over the request, then one per step: the request cut into forwards at
the reads of their results, the prompt's those read before the first generated token reached the client, the rest told
the model's or the draft head's by the engine's own counts (its decode row and the request's spec-decode counters), a
step from one verify's result read to the next. Its span and first-token mark are the driver's (runtime/vllm_mtp_drive,
llama_server_decode_drive --mtp-prompts --first-token-mark: on the host's CLOCK_MONOTONIC, placed by the trace's
systemClockNs as a prefill request's is); a cut that does not give the engine's counts is refused.
The capture is the driver under nsys with the flags vendor_capture._nsys_argv uses (CUDA and NVTX traced, graph nodes
traced, sampling and context switches off, a flush timer for a server stopped by signal), then `nsys export --type sqlite`;
the same driver command untraced gives the capture overhead (the CLI's --untraced)."""
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
GAP_MS = 1.0  # a named constant, not derived: the prefill record's gap split (gaps at least this long attributed one by one, shorter
# ones summed as small gaps), kept as the records before it were made so they compare; a decode attributes every gap (decode_budget)
# Named constants, not derived, each the prefill record's as it was made (changing one changes that record): the
# allowance of the sums check (`sums`, recorded as tolerance_ms beside it; a decode's parts meet its wall by construction,
# host time being the residual), and how many names a reader is shown, display cuts only (every call, gap and kernel stays
# in the sums; the lists are the largest first).
TOLERANCE_PCT = 1.0  # the parts must sum to the wall within this share of it (or TOLERANCE_MIN_MS, the larger)
TOLERANCE_MIN_MS = 2.0
LIST_GAPS = 12  # the largest gaps listed one by one
OTHER_NAMES = 8  # the largest kernels no family row took, listed so a missing row is seen
CALLS_PER_CAUSE = 4  # the calls named under each cause
CALLS_PER_GAP = 3  # the calls named under each listed gap


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


def families_for(engine:str | list[str] | None, vendor:str | None, table:dict[str, Any] | None = None, decode:bool = False) -> list[Rule]:
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
    rows.extend([*((e.get("decode_rows") or []) if decode else []), *(e.get("rows") or [])])
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


def causes(table:dict[str, Any] | None = None, decode:bool = False) -> tuple[list[tuple[re.Pattern, str]], str, dict[str, str]]:
  """The cause rows in the order tried (a decode's own rows first), the cause of no call in flight, and the words."""
  t = table or _table("gap_causes.json")
  rows = [*((t.get("decode_rows") or []) if decode else []), *t["rows"]]
  bad = [r for r in rows if r["cause"] not in t["causes"]] + ([] if t["cpu"] in t["causes"] else [t])
  if bad:
    raise ValueError(f"gap_causes.json: a row names a cause the table does not list: {bad[0]}")
  return [(re.compile(r["call"]), r["cause"]) for r in rows], t["cpu"], dict(t["causes"])


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


def inside(kernels:list[tuple[int, int, str, Any]], span:tuple[int, int] | None) -> list[int]:
  """The indexes of the kernels that overlap [a, b) (all of them without a span), in order: clip keeps these."""
  return [i for i, (s, e, _, _) in enumerate(kernels) if span is None or (e > span[0] and s < span[1])]


def clip(tl:Timeline, span:tuple[int, int] | None) -> Timeline:
  """The timeline cut to [a, b) ns: events outside dropped, events across an edge cut at it."""
  if span is None:
    return tl
  a, b = span
  c = lambda s, e: (max(s, a), min(e, b))  # noqa: E731
  kernels = [(*c(s, e), n, st) for s, e, n, st in (tl.kernels[i] for i in inside(tl.kernels, span))]
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

def kernel_families(kernels:list[tuple[int, int, str, Any]], rules:list[Rule],
                    fams:list[str] | None = None) -> tuple[dict[str, int], dict[str, int]]:
  """The kernels' union split into families: each instant goes to the family of the kernel that started last among
  those running (exact on one stream; on several streams the overlap goes to the newer kernel). Also each `other`
  kernel name's busy ns, so the names no row took are listed. fams: each launch's family when the caller labelled
  them (a decode's forwards: decode_budget), else launch_families."""
  fams = fams if fams is not None else launch_families(kernels, rules)
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
           what:str | None = None, labels:list[str] | None = None, span_is_window:bool = False,
           decode:bool = False) -> dict[str, Any]:
  """The time budget of one request (boltbeam.time_budget.v1, schemas/time_budget.schema.json). span: the request's
  [start, end) in trace ns (None: the whole trace). wall_ms: the client's wall time for it (None: host time outside the
  window is not known and the sum is checked over the window alone). decode: the tables' decode rows are read first
  (decode_budget; the prefill record reads the rows it always read). labels: each kernel's family, aligned with tl.kernels (decode_budget's forwards). span_is_window: the window is the
  span itself, so the time before the first operation and after the last is GPU idle, attributed like any gap (a
  decode step inside a request: its edges are the host between two steps, not outside the request)."""
  if labels is not None:
    labels = [labels[i] for i in inside(tl.kernels, span)]
  tl = clip(tl, span)
  kspans = [(s, e) for s, e, _, _ in tl.kernels]
  cspans = [(c["start_ns"], c["end_ns"]) for c in tl.copies]
  everything = [x for x in kspans + cspans if x[1] > x[0]]
  base = {"schema": SCHEMA, "kind": "time_budget", "what": what, "engine": engine, "vendor": vendor, "source": tl.source,
          "span_ns": list(span) if span else None, "wall_ms": wall_ms}
  if not everything:
    return {**base, "refused": "the trace holds no GPU operation in this span"}
  t0, t1 = (span if span_is_window and span else (min(a for a, _ in everything), max(b for _, b in everything)))
  window = t1 - t0
  ku, cu = union(kspans), union(cspans)
  au = union(everything)
  exposed = length(cu) - meet(cu, ku)
  gaps = ([(t0, au[0][0])] if au[0][0] > t0 else []) + idle_gaps(au) + ([(au[-1][1], t1)] if t1 > au[-1][1] else [])
  idle = sum(b - a for a, b in gaps)
  fams, other = kernel_families(tl.kernels, families_for(engine, vendor, families_table, decode), labels)
  rows, cpu, cause_words = causes(causes_table, decode)
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
    r["calls"] = dict(sorted(((n, v) for n, v in r["calls"].items() if v > 0), key=lambda x: -x[1])[:CALLS_PER_CAUSE])
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
                                "calls": {n: ms(v) for n, v in sorted(g["calls"].items(), key=lambda x: -x[1])[:CALLS_PER_GAP]}}
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


# --- a decode request: its forwards, and a step per token the target model verifies ------------------------------

def forward_rule(table:dict[str, Any] | None = None) -> dict[str, Any]:
  """How a trace is cut into forwards (kernel_families.json `forward`): the copy kind that reads a forward's result back
  (the host needs it to pick the next token), the copy kind a step's input arrives by, the families that run once per
  layer, the tail rows (the last launch of a family in a forward becomes another: the output layer), and the family a
  draft forward's kernels go to. Fails closed on a family the table does not list."""
  t = table or _table("kernel_families.json")
  f = t["forward"]
  named = [*f["layer_families"], f["draft"], *(x for r in f["tail"] for x in (r["family"], r["becomes"]))]
  if bad := [x for x in named if x not in t["families"]]:
    raise ValueError(f"kernel_families.json: forward names a family the table does not list: {bad[0]}")
  return f


def decode_row(engine:str | list[str] | None, table:dict[str, Any] | None = None) -> dict[str, Any] | None:
  """The engine's decode row (kernel_families.json engines.<key>.decode, the first along the chain a fork's rows are read
  in): how many tokens its prompt forward gives, how many forwards a prompt takes, and how many draft forwards it runs
  per drafted token and per forward of the model (null: its draft passes run inside a step with no result read between)."""
  t = (table or _table("kernel_families.json")).get("engines") or {}
  todo, seen = list(engine_keys(engine)), set()
  while todo:
    key = todo.pop(0)
    if key in seen or key not in t:
      continue
    seen.add(key)
    if "decode" in t[key]:
      return t[key]["decode"]
    todo += t[key].get("also") or []
  return None


def forwards(tl:Timeline, fams:list[str], rule:dict[str, Any]) -> list[dict[str, Any]]:
  """The trace's forwards in order: the kernels between two reads of a result (copies of rule["result_read"]; no other
  copy cuts one); a run of kernels with no per-layer launch (a sampler) joins the forward before it, or the one after it
  when it comes first. Each: its kernel indexes, start, end (the last result read before the next forward, else its last
  kernel) and its per-layer launches."""
  reads = sorted(c["end_ns"] for c in tl.copies if c["kind"] == rule["result_read"])
  layer = set(rule["layer_families"])
  groups: dict[int, list[int]] = {}
  for j in sorted(range(len(tl.kernels)), key=lambda j: tl.kernels[j][:2]):
    groups.setdefault(bisect.bisect_right(reads, tl.kernels[j][0]), []).append(j)
  segs: list[dict[str, Any]] = []
  for _, ks in sorted(groups.items()):
    n = sum(fams[j] in layer for j in ks)
    if segs and not n:
      segs[-1]["kernels"] += ks
    elif segs and not segs[-1]["layers"]:  # the first run had no per-layer launch: it joins this forward
      segs[-1].update(kernels=segs[-1]["kernels"] + ks, layers=n)
    else:
      segs.append({"kernels": ks, "layers": n})
  for i, s in enumerate(segs):
    s["start"] = min(tl.kernels[j][0] for j in s["kernels"])
    last = max(tl.kernels[j][1] for j in s["kernels"])
    nxt = segs[i + 1]["kernels"] if i + 1 < len(segs) else None
    limit = min(tl.kernels[j][0] for j in nxt) if nxt else None
    s["end"] = max([last, *(r for r in reads if r >= last and (limit is None or r <= limit))])
  return segs


def classify(segs:list[dict[str, Any]], mains:int, catchups:int = 0, drafted:tuple[int, int] = (0, 0)) -> bool:
  """Mark each forward main (the whole model) or draft from the engine's counts: `mains` forwards of the model,
  `catchups` catch-up passes of the draft head (one per model forward, when the engine runs them) and between drafted[0]
  and drafted[1] draft passes (the drafted tokens, and at most the passes the engine runs without counting them). The
  split is the per-layer launch count at and above which exactly `mains` forwards fall; below it each forward is draft
  passes and at most one catch-up, by their launch counts (shapes read from the trace, never written: a draft pass's x
  launches, a catch-up's y; a build that runs a catch-up and a draft pass with no result read between them shows one
  forward of y + x, counted as both; when catch-up and draft are one shape they are counted together). Each draft
  forward gets "passes" (the head passes it holds) and "catchups". No catch-up and no draft: every forward is the
  model's. False when no split and shapes give the counts: the cut is refused, never guessed."""
  lo, hi = drafted
  if not (catchups or hi):
    for x in segs:
      x["kind"] = "main"
    return len(segs) == mains
  freq: dict[int, int] = {}
  for x in segs:
    freq[x["layers"]] = freq.get(x["layers"], 0) + 1
  for c in sorted(freq):
    low = [x for x in segs if x["layers"] < c]
    if not low or len(segs) - len(low) != mains:
      continue
    shapes = sorted({x["layers"] for x in low}, key=lambda n: -freq[n])
    for d in shapes:
      for u in (shapes if catchups else [None]):
        got = _passes(low, d, u)
        if got is None:
          continue
        a, b = sum(g[0] for g in got), sum(g[1] for g in got)
        if (catchups + lo <= a + b <= catchups + hi) if u in (None, d) else (a == catchups and lo <= b <= hi):
          for x in segs:
            x["kind"], x["passes"], x["catchups"] = ("main", 0, 0)
          for x, (ca, dr) in zip(low, got):
            x["kind"], x["passes"], x["catchups"] = "draft", ca + dr, ca
          return True
  return False


def _passes(low:list[dict[str, Any]], d:int, u:int | None) -> list[tuple[int, int]] | None:
  """Each draft forward as (catch-ups, draft passes): its launches as b x d + a x u with a 0 or 1, the one way they
  divide (u None or equal to d: whole passes of d, not told apart). None when a forward divides no way or two ways."""
  out = []
  for x in low:
    n = x["layers"]
    if u is None or u == d:
      if n % d:
        return None
      out.append((0, n // d))
      continue
    ways = [(a, (n - a * u) // d) for a in (0, 1) if n - a * u >= 0 and (n - a * u) % d == 0 and (a or n)]
    if len(ways) != 1:
      return None
    out.append(ways[0])
  return out


def forward_labels(tl:Timeline, fams:list[str], segs:list[dict[str, Any]], rule:dict[str, Any]) -> list[str]:
  """Each launch's family with the forwards applied: a draft forward's kernels go to the draft family; in a main forward
  the last launch of a tail row's family becomes that row's (the output layer: the last matmul before the result read)."""
  out = list(fams)
  for s in segs:
    if s["kind"] == "draft":
      for j in s["kernels"]:
        out[j] = rule["draft"]
      continue
    for r in rule["tail"]:
      if hit := [j for j in s["kernels"] if fams[j] == r["family"]]:
        out[max(hit, key=lambda j: tl.kernels[j][0])] = r["becomes"]
  return out


def input_copies(tl:Timeline, kind:str, a:int, b:int, ku:list[list[int]], rows:dict[str, int]) -> dict[str, dict[str, Any]]:
  """A step's input copies (kind: the forward rule's input_copy) by what they carry: a copy whose bytes are a whole
  number of one input's rows (rows: {input: bytes a token's row takes on the way in}, from the model file and the
  engine's row: its host tables and any other per-token upload of known size) is that input's; one that is a whole
  number of two inputs' rows is "ambiguous" (never named as either); any other is "unidentified" (token ids, positions
  and the like). Per group: count, bytes, busy and exposed ms, and for a named input the GPU idle right before each copy
  (0 when other work runs as it starts): the host gathering what it sends; none for an ambiguous or unidentified copy."""
  groups: dict[str, list[dict[str, Any]]] = {}
  for c in (c for c in tl.copies if c["kind"] == kind and a <= c["start_ns"] < b):
    n = int(c.get("bytes") or 0)
    hits = [name for name, per in rows.items() if per and n and n % per == 0]
    groups.setdefault(hits[0] if len(hits) == 1 else "ambiguous" if hits else "unidentified", []).append(c)
  au = union([(s0, e0) for s0, e0, _, _ in tl.kernels] + [(c["start_ns"], c["end_ns"]) for c in tl.copies])
  starts = [x for x, _ in au]

  def idle_before(c:dict[str, Any]) -> int:  # 0 when other work runs as it starts; else back to the last op's end (or a)
    i = bisect.bisect_right(starts, c["start_ns"]) - 1
    if au[i][0] < c["start_ns"]:
      return 0
    return c["start_ns"] - max(a, au[i - 1][1] if i > 0 else a)
  out = {}
  for name, cs in groups.items():
    u = union((c["start_ns"], c["end_ns"]) for c in cs)
    before = None if name in ("ambiguous", "unidentified") else sum(idle_before(c) for c in cs) / 1e6
    out[name] = {"copies": len(cs), "bytes": sum(int(c.get("bytes") or 0) for c in cs), "busy_ms": length(u) / 1e6,
                 "exposed_ms": (length(u) - meet(u, ku)) / 1e6, "host_before_ms": before}
  return out


def decode_budget(tl:Timeline, *, tokens:int, accepted:int = 0, drafted:int = 0, k:int = 0, first_token:int | None = None,
                  span:tuple[int, int] | None = None, wall_ms:float | None = None, engine:str | list[str] | None = None,
                  vendor:str | None = None, read:dict[str, Any] | None = None, overhead:dict[str, Any] | None = None,
                  input_rows:dict[str, int] | None = None, streaming:dict[str, Any] | None = None, prompt_from_counts:bool = False,
                  families_table:dict[str, Any] | None = None,
                  causes_table:dict[str, Any] | None = None, what:str | None = None) -> dict[str, Any]:
  """The time budget of one decode request (budget, over the whole request) and its `decode` part. The request is cut
  into forwards (forwards). The prompt's are every forward whose result was read before the first generated token
  reached the client (first_token, trace ns: the driver's mark), however many the server ran (batches, checkpoints,
  a draft head's catch-up). The rest are told the model's or the draft head's by the engine's own counts (classify,
  decode_row): tokens, accepted and drafted are the driver's generated tokens and the engine's spec-decode counters, k
  the draft count asked; the model ran tokens - tokens_from_prompt - accepted steps, the head the draft forwards its
  row says (none when nothing could be drafted, k 0). A cut that does not give those counts is refused: no shares. Each
  step runs from the previous main forward's result read to this one's and is budgeted on its own, its edges idle
  attributed by the call in flight; every gap is attributed (no threshold). read: {"gbs", "source", "verify_bytes",
  "draft_bytes", "how", "residency"} for what bounds a step, or {"refused": key}. input_rows: input_copies' inputs.
  overhead: the capture's measured overhead. streaming: whether the request was streamed and its unstreamed control.
  prompt_from_counts: with no first-token mark, the prompt's forwards are the fewest leading forwards after which the
  rest give the engine's counts (said in the record: the count check is then weaker, it chose the prompt)."""
  tl = clip(tl, span)
  t = families_table or _table("kernel_families.json")
  rule, row = forward_rule(t), decode_row(engine, t)
  fams = launch_families(tl.kernels, families_for(engine, vendor, t, True))
  segs = forwards(tl, fams, rule)
  kw = dict(engine=engine, vendor=vendor, families_table=t, causes_table=causes_table, gap_ms=0.0, decode=True)
  steps_expected = tokens - (row or {}).get("tokens_from_prompt", 0) - accepted
  per = (row or {}).get("draft_forwards") if k else None
  catchups = per["per_main_forward"] * steps_expected if per else 0
  lo = per["per_draft_token"] * drafted if per else 0
  span_drafts = (lo, lo + per.get("uncounted_per_step", 0) * steps_expected) if per else (0, 0)
  fits = lambda rest: classify(rest, steps_expected, catchups, span_drafts)  # noqa: E731
  if first_token is not None:
    prompt = [x for x in segs if x["end"] <= first_token]
    ok = bool(prompt) and fits(segs[len(prompt):])
  elif prompt_from_counts:
    cut = next((n for n in range(1, len(segs)) if fits(segs[n:])), None)
    prompt, ok = segs[:cut or 0], cut is not None
    if ok:
      fits(segs[cut:])  # the marks of the fit
  else:
    prompt, ok = [], False
  rest = segs[len(prompt):]
  check = {"steps_expected": steps_expected, "forwards_found": len(segs), "prompt_forwards": len(prompt), "mains_expected": steps_expected,
           "catchups_expected": catchups, "drafts_expected": list(span_drafts), "drafted": drafted, "layers_found": sorted({x["layers"] for x in rest}),
           "prompt_from": "first_token_mark" if first_token is not None else "counts" if prompt_from_counts else None}
  refused = ("no_decode_row" if row is None else "no_steps" if steps_expected < 1 else
             "no_first_token" if first_token is None and not prompt_from_counts else
             "prompt_unmatched" if first_token is not None and not prompt else None if ok else "forwards_unmatched")
  if refused:
    return {**budget(tl, wall_ms=wall_ms, span=span, what=what, **kw), "decode": {"refused": refused, "check": check}}
  for x in prompt:
    x["kind"] = "prompt"
  labels = forward_labels(tl, fams, segs, rule)
  whole = budget(tl, wall_ms=wall_ms, span=span, what=what, labels=labels, **kw)
  ms = lambda ns: ns / 1e6  # noqa: E731
  main = [prompt[-1]] + [x for x in rest if x["kind"] == "main"]
  begin, finish = (span if span else (segs[0]["start"], segs[-1]["end"]))
  ku = union((s0, e0) for s0, e0, _, _ in tl.kernels)
  w = lambda ss: ms(length(union((tl.kernels[j][0], tl.kernels[j][1]) for x in ss for j in x["kernels"])))  # noqa: E731
  steps = []
  for prev, cur in zip(main, main[1:]):
    a, b = prev["end"], cur["end"]
    got = budget(tl, wall_ms=ms(b - a), span=(a, b), labels=labels, span_is_window=True, **kw)
    passes = [x for x in rest if x["kind"] == "draft" and a < x["start"] <= b]
    weight = None
    if read and not read.get("refused") and read.get("gbs") and read.get("verify_bytes") and (not passes or read.get("draft_bytes")):
      weight = (read["verify_bytes"] + sum(x["passes"] for x in passes) * (read.get("draft_bytes") or 0)) / read["gbs"] / 1e6
    steps.append({"at_ms": ms(a - begin), "wall_ms": ms(b - a), "draft_passes": sum(x["passes"] for x in passes), "draft_forwards": len(passes), "verify_ms": w([cur]), "draft_ms": w(passes),
                  "families": {f["family"]: f["ms"] for f in got["families"]}, "exposed_copy_ms": got["exposed_copy_ms"],
                  "idle_ms": got["idle_ms"], "idle_by_cause": {c["cause"]: c["ms"] for c in got["gaps"]["by_cause"]},
                  "input_copy": input_copies(tl, rule["input_copy"], a, b, ku, input_rows or {}), "weight_read_ms": weight})
  decode_ms = sum(x["wall_ms"] for x in steps)
  out_tokens = tokens - row["tokens_from_prompt"]
  add = lambda key: {n: sum(x[key].get(n, 0.0) for x in steps) for n in dict.fromkeys(n for x in steps for n in x[key])}  # noqa: E731
  pct = lambda d: {n: 100.0 * v / decode_ms for n, v in sorted(d.items(), key=lambda x: -x[1])}  # noqa: E731
  one = lambda v: 100.0 * v / decode_ms  # noqa: E731
  weights = [x["weight_read_ms"] for x in steps]
  weight_ms = sum(weights) if None not in weights else None
  import statistics
  walls = [x["wall_ms"] for x in steps]
  inputs: dict[str, dict[str, Any]] = {}
  for x in steps:
    for name, g in x["input_copy"].items():
      acc = inputs.setdefault(name, {"copies": 0, "bytes": 0, "busy_ms": 0.0, "exposed_ms": 0.0, "host_before_ms": None if g["host_before_ms"] is None else 0.0})
      for key in ("copies", "bytes", "busy_ms", "exposed_ms"):
        acc[key] += g[key]
      if g["host_before_ms"] is not None:
        acc["host_before_ms"] += g["host_before_ms"]
  for g in inputs.values():
    g.update(per_token_bytes=g["bytes"] / out_tokens, per_token_ms=(g["exposed_ms"] + (g["host_before_ms"] or 0.0)) / out_tokens,
             pct=one(g["exposed_ms"] + (g["host_before_ms"] or 0.0)))
  decode = {
    "prompt_ms": ms(prompt[-1]["end"] - begin), "decode_ms": decode_ms, "tail_ms": ms(finish - main[-1]["end"]),
    "tokens": tokens, "out_tokens": out_tokens, "steps": len(steps), "draft_passes": sum(x["draft_passes"] for x in steps),
    "check": check, "gaps": "every gap attributed (no threshold)",
    "layers_per_forward": {"main": sorted({x["layers"] for x in main[1:]}), "draft": sorted({x["layers"] for x in rest if x["kind"] == "draft"})},
    "drafts_apart": bool(row.get("draft_forwards")),
    "draft_forwards": sum(x["draft_forwards"] for x in steps), "draft_forwards_holding_more": sum(x.get("passes", 0) > 1 for x in rest),
    "catchups": sum(x.get("catchups", 0) for x in rest),
    "input_rows": input_rows or {}, "input_element_bytes": row.get("host_rows_element_bytes"), "streaming": streaming or {"streamed": False},
    "constants": {"gap_ms": "none: every gap attributed", "tolerance": f"{TOLERANCE_PCT:g}% or {TOLERANCE_MIN_MS:g} ms (named, the sums check)",
                  "display_cuts": f"gaps {LIST_GAPS}, other kernels {OTHER_NAMES}, calls {CALLS_PER_CAUSE} a cause and {CALLS_PER_GAP} a gap (named, lists only)"},
    "per_token_ms": decode_ms / out_tokens,
    "step_ms": {"median": statistics.median(walls), "min": min(walls), "max": max(walls)},
    "input_copy": inputs,
    "pct": {"families": pct(add("families")), "idle_by_cause": pct(add("idle_by_cause")),
            "exposed_copy": one(sum(x["exposed_copy_ms"] for x in steps)), "verify": one(sum(x["verify_ms"] for x in steps)),
            "draft": one(sum(x["draft_ms"] for x in steps))},
    "bound": None if weight_ms is None else {
      "weight_read_ms": weight_ms, "weight_share": weight_ms / decode_ms, "rest_ms": decode_ms - weight_ms,
      "per_token_weight_ms": weight_ms / out_tokens, "read": read},
    "bound_missing": None if weight_ms is not None else (read or {}).get("refused") or ("no_read" if not read else "no_draft_bytes"),
    "capture_overhead": overhead or {"pct": None, "words": None, "why": "no_untraced"},
    "steps_list": steps,
  }
  return {**whole, "decode": decode}


def spans_from_trace(tl:Timeline, walls_ms:list[float]) -> list[tuple[int, int]]:
  """Request spans for a driver that recorded none, from the trace itself: the requests are the trace's last ones, in
  order; the last ends where the trace's last GPU operation ends, each starts its own wall time (the driver's client
  wall) before its end, and the one before it ends where the last GPU operation before that start ends. The host time
  after a request's last GPU work (its reply's way back) is not in its span: a span from the trace is the GPU's end
  of the request, said by the caller."""
  ops = sorted([(s0, e0) for s0, e0, _, _ in tl.kernels] + [(c["start_ns"], c["end_ns"]) for c in tl.copies])
  starts, ends, m = [s0 for s0, _ in ops], [], 0
  for _, e0 in ops:
    m = max(m, e0)
    ends.append(m)
  out, end = [], ends[-1] if ends else 0
  for wall in reversed(walls_ms):
    a = end - int(wall * 1e6)
    out.append((a, end))
    i = bisect.bisect_left(starts, a) - 1
    end = ends[i] if i >= 0 else a
  return out[::-1]


def request_decode_budget(db:str | pathlib.Path, row:dict[str, Any], *, tl:Timeline | None = None,
                          trace_span:tuple[int, int] | None = None, **kw) -> dict[str, Any] | None:
  """The decode budget of one driver row (runtime/llama_server_decode_drive.py --mtp-prompts --first-token-mark,
  runtime/vllm_mtp_drive.py): its span_ns cut out of the trace on the host's monotonic clock (as a prefill request's),
  its first_token_ns placed the same way, its wall the span's length, its tokens, k and spec counters from the row.
  None without the trace's clock origin. trace_span: the span in trace ns when the driver recorded none (spans_from_trace);
  tl: the trace, read once for several rows."""
  tl = tl or timeline_from_nsys(db)
  span = trace_span or span_from_monotonic(tl, *row["span_ns"])
  if span is None:
    return None
  spec, mark, ctl = row.get("spec") or {}, row.get("first_token_ns"), row.get("unstreamed")
  streaming = {"streamed": bool(row.get("streamed")), "unstreamed_tok_s": (ctl or {}).get("decode_tok_s"),
               "pct": 100.0 * (ctl["decode_tok_s"] / row["decode_tok_s"] - 1) if ctl and row.get("decode_tok_s") else None}
  return decode_budget(tl, span=span, wall_ms=(span[1] - span[0]) / 1e6, tokens=int(row["tokens"]),
                       accepted=int(spec.get("accepted_tokens") or 0), drafted=int(spec.get("draft_tokens") or 0), k=int(row.get("k") or 0),
                       first_token=None if mark is None or trace_span else mark - tl.monotonic_origin_ns, streaming=streaming, **kw)
