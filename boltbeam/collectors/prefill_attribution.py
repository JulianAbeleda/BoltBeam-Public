"""Which role each kernel of a captured prefill ran for, from launch counts and operations, and each kernel's ceiling.

A capture from outside the engine (nsys) sees kernels, not roles, and a prefill GEMM's launch grid need not show its
shape (a stream-k kernel launches one block per SM whatever the matrix). So the pairing reads what does not depend on
the engine's launch geometry:

  the window     a capture that holds the engine's warm-up prefill and the measured one (the adapter says which) is
                 cut to its second half, when the two halves launch the same kernels the same number of times (an engine may idle on the
                 host inside a prefill, before its last token's tail, so the longest idle stretch is not the edge)
  templates      launches are grouped by kernel template (instruction_path.template), and a template's matrix path
                 is read from its code (instruction_path.kernel_path); a template the kind words name as attention
                 (tie_out.kind_of, the decode table's own rule for the attention row) does attention's work, its own
                 row, and is never paired with a weight role
  counts         a role runs `count` times per chunk (model_profile.json), so a template that serves a set of roles
                 launches the sum of their counts per chunk, times the chunks; the engine may skip a role in the last
                 layer of a chunk (its rows are not needed), so up to one layer's worth fewer is allowed and said
  one format     a template reads one weight format: the roles it serves share a quant
  operations     more operations per launch is more time per launch: within a template the launches fall into duration
                 clusters, one per role row, in the order of the rows' operations, each cluster holding that row's
                 launches; roles whose operations are within GAP_RATIO cannot be told apart by time and are one row
                 ("ffn_down + ffn_gate_up": the same N x K turned around)
  the peak       a pairing that needs more operations per second than the measured peak of the template's path is
                 impossible and is refused (PEAK_SLACK)
  the bytes      so is one that reads its roles' weights faster than the memory's measured bandwidth: a kernel whose
                 code could not be read (a library that hides it) may still take a role by count and time, its work
                 counted and its ceiling refused, but never a kernel too short to have read the weights

The assignment that places the most roles wins; two such assignments are ambiguous and nothing is paired. Kernels no
role takes keep their time, named by kind; a GEMV over one token that matches a role by count and bytes (the output
head on each engine call's last token: once per call, `outputs` calls) is paired the way decode pairs
(attribution.attribute, a call standing for a decode token).
"""
from __future__ import annotations

import collections
import itertools
import math
import statistics
from typing import Any

from boltbeam.collectors.attribution import attribute, share_overlap
from boltbeam.collectors.instruction_path import FROM_NAME, kernel_path, template
from boltbeam.roofline import prefill_work as work
from boltbeam.workflow.tie_out import kind_of

PEAK_SLACK = 1.05
GAP_RATIO = 1.3  # two neighbouring duration clusters differ by at least this ratio, or they are one cluster
MAX_ASSIGN = 2_000_000
STEM_CHARS = 40  # a row named by its template is cut to this many characters; the trace keeps the whole name
MAX_OWN = 2  # a role assembled from at most this many templates that serve it alone
PAIRING_RULE = ("a template named as attention or recurrent (tie_out.kind_of) does that block's work and serves no role; a role is "
                "assembled from up to two templates that serve it alone, or shares one template with roles of its format (whose launches may be a whole multiple of theirs: each product cut along M into that many launches, each reading the weights); "
                "roles of one format that templates serve alone keep the order of their operations per launch in their time per launch; "
                "a shared template with a matrix instruction serves roles of one weight format whose launches per chunk sum to its own, less at most one "
                "layer's launches per chunk and never all of a role's (the engine skipping rows it does not need), at no more operations per "
                "second than its path's measured peak and no faster read of its roles' weights than the memory's bandwidth (a kernel whose code "
                "was not read may take roles so, its ceiling refused); within it, and within each chunk width, duration clusters follow the roles' operations per "
                "launch, a smaller launch never faster per operation than a larger one beyond the cluster gap (it may be slower: it fills fewer units); roles with equal operations are one row; the assignment placing the most roles wins, then the one leaving the fewest matrix kernels unpaired, then the one pairing the most kernel time; among tied ones, only the roles every one places the same way are paired")


REPEAT = "repeat"  # measured_window: the capture ends in two identical turns (a session's turn, captured twice)


def measured_window(launches:list[dict[str, Any]], warmup:bool | str = True) -> tuple[list[dict[str, Any]], str]:
  """The measured prefill. With warmup (the engine ran a warm-up prefill inside the capture): the second half of the
  launches, when the first half launched the same kernels the same number of times, else refused. Without: every
  launch (a prefill of identical chunks has identical halves too, so halves alone cannot say there was a warm-up).
  REPEAT (the capture holds other work, then the measured prefill twice): the shortest tail of launches that the
  launches just before it repeat kernel for kernel, in order (equal counts alone would also match a window that
  straddles the two turns); a once-per-call kernel (the output head) makes that tail one whole call."""
  timed = sorted((l for l in launches if "start_us" in l), key=lambda l: l["start_us"])
  if not warmup:
    return timed, f"one prefill of {len(timed)} launches (the capture holds the measured prefill only)"
  count = lambda ls: sorted(collections.Counter(template(l["key"]) for l in ls).items())  # noqa: E731
  if warmup == REPEAT:
    keys = [template(l["key"]) for l in timed]
    for n in range(1, len(timed) // 2 + 1):
      if keys[-n:] == keys[-2 * n:-n]:
        return timed[-n:], (f"the last of two identical turns at the end of the capture ({n} launches each); the "
                            f"{len(timed) - n} launches before it fill the cache and repeat the turn")
    raise ValueError(f"the capture's {len(timed)} launches do not end in two identical turns")
  half = len(timed) // 2
  if len(timed) % 2 == 0 and half and count(timed[:half]) == count(timed[half:]):
    gap = timed[half]["start_us"] - (timed[half - 1]["start_us"] + timed[half - 1]["wall_us"])
    return timed[half:], (f"the second of two identical prefills ({half} launches each, {gap / 1000:.1f} ms apart); the "
                          "first is the engine's warm-up")
  raise ValueError(f"the capture's {len(timed)} launches are not a warm-up prefill and a measured one of the same kernels")


def _clusters(durations:list[float], k:int) -> list[list[float]] | None:
  """k duration clusters cut at the k - 1 widest gaps in log duration, each gap at least GAP_RATIO; None if not."""
  if k == 1:
    return [sorted(durations)]
  d = sorted(durations)
  gaps = sorted(((math.log(d[i + 1] / d[i]) if d[i] > 0 else 0.0, i) for i in range(len(d) - 1)), reverse=True)[:k - 1]
  if len(gaps) < k - 1 or min(g for g, _ in gaps) < math.log(GAP_RATIO):
    return None
  cuts = sorted(i for _, i in gaps)
  out, start = [], 0
  for c in cuts:
    out.append(d[start:c + 1])
    start = c + 1
  out.append(d[start:])
  return out


def chunk_of(keys:list[str], chunks:int) -> list[int] | None:
  """Each launch's chunk, for a window of `chunks` chunks in stream order that each run the model's graph once:
  boundary k is the launch where every kernel stem has run closest to k / chunks of its launches (the sum over stems
  of the distance is least). Stems, not templates, because an engine may launch another variant of a kernel for a
  chunk of another width, and the count need not match exactly (a narrow chunk's stream-k fixups, the rows a last
  layer skips). None when a chunk would be empty or a boundary is off by more than a quarter of a chunk's launches."""
  if chunks <= 1:
    return [0] * len(keys)
  stems = [k.split("<", 1)[0] for k in keys]
  total = collections.Counter(stems)
  seen: collections.Counter = collections.Counter()
  best = [(math.inf, 0)] * (chunks - 1)
  for i in range(len(stems) + 1):  # i launches before the boundary
    for k in range(1, chunks):
      d = sum(abs(seen[s] - k * n / chunks) for s, n in total.items())
      if d <= best[k - 1][0]:  # among equal distances, the latest: a launch that ends a chunk belongs to it
        best[k - 1] = (d, i)
    if i < len(stems):
      seen[stems[i]] += 1
  edges = [i for _, i in best]
  if any(d > 0.25 * len(stems) / chunks for d, _ in best) or any(b <= a for a, b in zip([0, *edges], [*edges, len(stems)])):
    return None
  return [sum(1 for e in edges if i >= e) for i in range(len(stems))]


def role_groups(roles:list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Roles of one format whose operations per launch are within GAP_RATIO of each other, joined: time cannot tell
  them apart (ffn_gate_up and ffn_down are the same N x K turned around)."""
  out: list[dict[str, Any]] = []
  for quant in sorted({r["quant"] for r in roles}):
    for r in sorted((r for r in roles if r["quant"] == quant), key=lambda r: (r["n"] * r["k"], r["role"])):
      last = out[-1] if out and out[-1]["quant"] == quant else None
      if last and r["n"] * r["k"] < last["nk"] * GAP_RATIO:
        last["roles"].append(r)
        last["count"] += r["count"]
        last["skip"] += r["skip"]
      else:
        out.append({"quant": quant, "nk": r["n"] * r["k"], "roles": [r], "count": r["count"], "skip": r["skip"]})
  for g in out:
    g["name"] = " + ".join(sorted(r["role"] for r in g["roles"])) + f" {g['quant']}"
  return out


def split(groups:list[dict[str, Any]], tmpl:dict[str, Any], chunks:int) -> list[list[dict[str, Any]]] | None:
  """The template's launches cut by duration into one cluster per role row, in the order of their operations, each
  holding that row's launches (its count per chunk times the chunks, less at most its skip), and one kernel's rate:
  a cluster's operations per launch over its median time is never over GAP_RATIO times that of a cluster with more
  operations per launch (a smaller launch fills fewer of the GPU's units, so it may run slower, never faster); None
  if they do not."""
  ordered = sorted(groups, key=lambda g: g["nk"])
  parts = _clusters([l["raw_wall_us"] for l in tmpl["launches"]], len(groups))
  if parts is None:
    return None
  for g, p in zip(ordered, parts):
    if not (g["count"] - g["skip"]) * chunks <= len(p) <= g["count"] * chunks:
      return None
  rates = [sum(r["n"] * r["k"] * r["count"] for r in g["roles"]) / g["count"] / max(statistics.median(p), 1e-9)
           for g, p in zip(ordered, parts)]
  if any(rates[i] > GAP_RATIO * rates[j] for i in range(len(rates)) for j in range(i + 1, len(rates))):
    return None
  edges = [max(p) for p in parts]
  out: list[list[dict[str, Any]]] = [[] for _ in ordered]
  for l in tmpl["launches"]:
    out[next(i for i, e in enumerate(edges) if l["raw_wall_us"] <= e)].append(l)
  return out


def m_splits(groups:list[dict[str, Any]], tmpl:dict[str, Any], chunks:int, mean_m:float) -> int:
  """The launches a shared template runs per product of its roles: a library that caps the rows of one launch runs
  a chunk's product as p launches of M / p rows, each reading the weights again. p is its launches over its roles'
  launches, a whole number dividing the chunk's width; 1 when it is not one."""
  want = sum(g["count"] for g in groups) * chunks
  p = round(tmpl["calls"] / want) if want else 1
  return p if p > 1 and (mean_m / p).is_integer() and want * p - sum(g["skip"] for g in groups) * chunks * p <= tmpl["calls"] else 1


def _reads_in_time(weight_bytes:float, wall_us:float, bandwidth_gbs:float | None) -> bool:
  """The launches had time to read the weights once each at the memory's measured bandwidth (none known: not checked)."""
  return not bandwidth_gbs or weight_bytes / (wall_us * 1e-6) <= bandwidth_gbs * 1e9 * PEAK_SLACK


def _fits(groups:list[dict[str, Any]], tmpl:dict[str, Any], chunks:int, mean_m:float, peak:float | None,
          bandwidth_gbs:float | None = None) -> bool:
  if len({g["quant"] for g in groups}) > 1:
    return False
  want = sum(g["count"] for g in groups) * chunks
  skip = sum(g["skip"] for g in groups) * chunks
  if not (want - skip <= tmpl["calls"] <= want):
    return False
  flops = sum(g["count"] * chunks * 2.0 * mean_m * g["nk"] for g in groups)
  if peak and flops / (tmpl["wall_us"] * 1e-6) > peak * 1e12 * PEAK_SLACK:
    return False
  wb = sum(r["weight_bytes"] * (r["count"] - r["skip"]) * chunks for g in groups for r in g["roles"])
  return _reads_in_time(wb, tmpl["wall_us"], bandwidth_gbs) and split(groups, tmpl, chunks) is not None


def _own_fits(r:dict[str, Any], ts:list[dict[str, Any]], chunks:int, mean_m:float, peaks:dict[str, float],
              bandwidth_gbs:float | None = None) -> bool:
  """A role assembled from templates that serve it alone: their launches add up to its count, each within its peak
  and with time to read its weight."""
  calls = sum(t["calls"] for t in ts)
  if not (r["count"] - r["skip"]) * chunks <= calls <= r["count"] * chunks:
    return False
  if not all(_reads_in_time(r["weight_bytes"] * t["calls"], t["wall_us"], bandwidth_gbs) for t in ts):
    return False
  per_call = 2.0 * mean_m * r["n"] * r["k"]
  return all(not _peak(t, peaks) or t["calls"] * per_call / (t["wall_us"] * 1e-6) <= _peak(t, peaks) * 1e12 * PEAK_SLACK
             for t in ts)


def pair(templates:list[dict[str, Any]], roles:list[dict[str, Any]], *, chunks:int, mean_m:float,
         peaks:dict[str, float], bandwidth_gbs:float | None = None) -> tuple[dict[int, tuple[str, tuple[int, ...]]], dict[str, Any] | None]:
  """role index -> ("shared", (t,)) or ("own", (t1, t2)) by PAIRING_RULE, or ({}, why) when two assignments place
  as many roles. A shared template may serve several roles of one format (split by duration); an own template serves
  one role alone, and a role may be assembled from up to MAX_OWN of them."""
  # a kernel whose code was not read may hold a matrix instruction: it may take roles, under the bytes rule
  matrix = [i for i, t in enumerate(templates) if t["path"]["status"] in ("one", "mixed", "not_found")
            and not t.get("attention") and not t.get("recurrent")]
  options: list[list[tuple[str, tuple[int, ...]] | None]] = []
  for r in roles:
    lo = (r["count"] - r["skip"]) * chunks
    opts: list[tuple[str, tuple[int, ...]] | None] = [None]
    opts += [("shared", (t,)) for t in matrix if templates[t]["calls"] >= lo]
    opts += [("own", ts) for ts in itertools.combinations(matrix, MAX_OWN)
             if lo <= sum(templates[t]["calls"] for t in ts) <= r["count"] * chunks]
    # an option the role breaks on its own (its weights read faster than the bandwidth, its operations faster than
    # the peak) fails with any other roles added too: dropped before the search, so many unread kernels stay cheap
    options.append([c for c in opts if c is None or _alone_fits(r, c, templates, chunks, mean_m, peaks, bandwidth_gbs)])
  best: dict[str, Any] = {"score": (0, 0, 0, 0.0), "ties": []}
  nodes = 0

  def leaf(choice):
    shared: dict[int, list[dict[str, Any]]] = {}
    for r, c in zip(roles, choice):
      if c and c[0] == "shared":
        shared.setdefault(c[1][0], []).append(r)
    for t, rs in shared.items():
      gs = role_groups(rs)
      p = m_splits(gs, templates[t], chunks, mean_m)
      if not _fits(gs, templates[t], chunks * p, mean_m / p, _peak(templates[t], peaks), bandwidth_gbs):
        return False
    if not all(_own_fits(r, [templates[t] for t in c[1]], chunks, mean_m, peaks, bandwidth_gbs)
               for r, c in zip(roles, choice) if c and c[0] == "own"):
      return False
    # more operations per launch, more time per launch: roles of one format that a template serves alone, compared
    alone = [(r, sum(templates[t]["wall_us"] for t in c[1]) / sum(templates[t]["calls"] for t in c[1]))
             for r, c in zip(roles, choice) if c and (c[0] == "own" or len(shared.get(c[1][0], [])) == 1)]
    for (a, ta), (b, tb) in itertools.combinations(alone, 2):
      if a["quant"] == b["quant"] and max(a["n"] * a["k"], b["n"] * b["k"]) >= GAP_RATIO * min(a["n"] * a["k"], b["n"] * b["k"]):
        if (a["n"] * a["k"] > b["n"] * b["k"]) != (ta > tb):
          return False
    return True

  def walk(i, choice, load, own_used):
    nonlocal nodes
    nodes += 1
    if nodes > MAX_ASSIGN:
      raise _TooMany
    placed = sum(1 for c in choice if c)
    if placed + len(roles) - i < best["score"][0]:
      return
    if i == len(roles):
      if not leaf(choice):
        return
      used = {t for c in choice if c for t in c[1]}
      score = (placed, len(used), -sum(1 for c in choice if c and c[0] == "own"),
               round(sum(templates[t]["wall_us"] for t in used), 3))
      if score > best["score"]:
        best.update(score=score, ties=[list(choice)])
      elif score == best["score"] and placed:
        best["ties"].append(list(choice))
      return
    r = roles[i]
    for c in options[i]:
      if c is None:
        walk(i + 1, choice + [None], load, own_used)
        continue
      kind, ts = c
      if any(t in own_used for t in ts) or (kind == "own" and any(load.get(t) for t in ts)):
        continue
      if kind == "shared":
        t = ts[0]
        need = load.get(t, 0) + (r["count"] - r["skip"]) * chunks
        if need > templates[t]["calls"]:
          continue
        walk(i + 1, choice + [c], {**load, t: need}, own_used)
      else:
        walk(i + 1, choice + [c], load, own_used | set(ts))

  try:
    walk(0, [], {}, set())
  except _TooMany:
    return {}, {"reason": f"more than {MAX_ASSIGN} ways to pair {len(roles)} roles with {len(matrix)} templates"}
  if not best["ties"]:
    return {}, None
  if len(best["ties"]) > 1:  # keep what every tied assignment does the same way, when that alone still fits
    ties = best["ties"]
    common = [c if c and all(t[i] == c for t in ties) else None for i, c in enumerate(ties[0])]
    kept = {i: c for i, c in enumerate(common) if c} if any(common) and leaf(common) else {}
    differ = sorted({f"{roles[i]['role']} {roles[i]['quant']}" for i in range(len(roles)) if len({repr(t[i]) for t in ties}) > 1})
    return kept, {"reason": f"{len(ties)} ways to pair the roles with the kernel templates place {best['score'][0]} roles each; "
                            f"{', '.join(differ)} differ between them and are not paired"
                            + (f"; the {len(kept)} placed the same way in every one are kept" if kept else "")}
  return {i: c for i, c in enumerate(best["ties"][0]) if c}, None


def _alone_fits(r:dict[str, Any], c:tuple[str, tuple[int, ...]], templates:list[dict[str, Any]], chunks:int,
                mean_m:float, peaks:dict[str, float], bandwidth_gbs:float | None) -> bool:
  kind, ts = c
  if kind == "own":
    return _own_fits(r, [templates[t] for t in ts], chunks, mean_m, peaks, bandwidth_gbs)
  t, n = templates[ts[0]], (r["count"] - r["skip"]) * chunks
  peak = _peak(t, peaks)
  return _reads_in_time(r["weight_bytes"] * n, t["wall_us"], bandwidth_gbs) and \
    (not peak or n * 2.0 * mean_m * r["n"] * r["k"] / (t["wall_us"] * 1e-6) <= peak * 1e12 * PEAK_SLACK)


class _TooMany(Exception):
  pass


def _peak(t:dict[str, Any], peaks:dict[str, float]) -> float | None:
  """The template's path peak; for a mixed template the peak its paths make together for its instruction mix
  (prefill_work.effective_peak: each path's share of the operations at its own peak)."""
  p = t["path"]
  if p["status"] == "one":
    return peaks.get(p["path"])
  if p["status"] == "mixed":
    return work.effective_peak(p["paths"], peaks)
  return None


def attribute_prefill(launches:list[dict[str, Any]], roles:list[dict[str, Any]], *, chunk_widths:list[int], outputs:int = 1,
                      layers:int, kernel_paths:dict[str, dict[str, int]], peaks:dict[str, float],
                      bandwidth_gbs:float | None, profile:dict[str, Any] | None = None, length:int | None = None,
                      program_roles:dict[str, dict[str, Any]] | None = None, untraced_ms:float | None = None,
                      warmup:bool | str = True, formats:dict[str, Any] | None = None,
                      cache:dict[str, float] | None = None, depth:int = 0,
                      observed:dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
  """The measured prefill split by role, attention and other kernels, each with operations, bytes, path and ceiling.
  roles: {role, quant, n, k, count (per chunk), weight_bytes (per launch)}. kernel_paths: template -> {path: sites}.
  outputs: the engine calls that each ended in an output (its last token's work, the output head, once per call).
  program_roles: template -> {role, quant, shape [M, N, K]} where the engine names each program's role and logical
  shape itself (the fork's own metadata): those launches are that role at that M, N, K, and the role is not paired
  again. untraced_ms: for an engine's own timing, whose run is slowed by profiling, the prefill the rows sum to: the
  gaps are it less the busy time, never the profiled run's idle. formats: the engine adapter's {activations: {in, out},
  kv, stream_k: template -> layout or None} (prefill.ENGINES); cache: the chip's last-level cache {bytes, gbs}. depth: tokens
  already in the cache (a session's turn), so attention's work counts every key the turn's tokens read. observed:
  template -> {path: instructions} ncu saw a closed kernel execute (instruction_path.ncu_observe)."""
  formats = formats or {}
  acts = formats.get("activations")
  gemm = lambda m, n, k, wb: work.gemm(m, n, k, wb, acts)["bytes"]  # noqa: E731
  window, window_words = measured_window(launches, warmup)
  shared, overlap = share_overlap(window)
  ordered = sorted(shared, key=lambda l: l["start_us"])  # the launches in stream order: a fixup follows its GEMM
  begin = min(l["start_us"] for l in window)
  end = max(l["start_us"] + l["wall_us"] for l in window)
  chunks, mean_m = len(chunk_widths), sum(chunk_widths) / max(len(chunk_widths), 1)
  for r in roles:
    # up to one layer's launches fewer per chunk, never all of them: a role with no launch is not on a template
    r.setdefault("skip", min(math.ceil(r["count"] / layers), r["count"] - 1) if layers else 0)
  by_t: dict[str, dict[str, Any]] = {}
  for l in shared:
    t = by_t.setdefault(template(l["key"]), {"template": template(l["key"]), "name": l["key"].split(" grid=")[0], "launches": [],
                                              "calls": 0, "wall_us": 0.0})
    t["launches"].append(l)
    t["calls"] += 1
    t["wall_us"] += l["wall_us"]
  templates = sorted(by_t.values(), key=lambda t: -t["wall_us"])
  for t in templates:
    t["path"] = kernel_path(t["name"], kernel_paths, observed)
    t["attention"] = kind_of({"kernel": t["name"]}) == "attention"  # attention's work is its own row, never a role's
    t["recurrent"] = kind_of({"kernel": t["name"]}) == "recurrent"  # so is a recurrent block's scan and convolution
  # chunks of different widths: each launch's chunk from the sequence every chunk repeats, and each template cut
  # into one per width, so each width pairs as a prefill of equal chunks (a narrow chunk runs every launch shorter,
  # and an engine may launch another variant of a kernel for it)
  ids = chunk_of([template(l["key"]) for l in ordered], chunks) if len(set(chunk_widths)) > 1 else None
  widths = sorted(set(chunk_widths), reverse=True) if ids is not None else [None]
  if ids is not None:
    where = {id(l): chunk_widths[c] for l, c in zip(ordered, ids)}
    templates = [{**t, "width": w, "launches": ls, "calls": len(ls), "wall_us": sum(l["wall_us"] for l in ls)}
                 for t in templates for w in widths if (ls := [l for l in t["launches"] if where[id(l)] == w])]
  rows: list[dict[str, Any]] = []
  taken: set[int] = set()
  shapes: dict[int, list[tuple[dict[str, Any], float]]] = {}  # a paired GEMM launch -> its roles, weighted by count
  named: dict[tuple[str, str], dict[str, Any]] = {}
  for ti, t in enumerate(templates):  # the engine's own role and (M, N, K) per program
    info = (program_roles or {}).get(t["template"])
    if not info or t.get("attention"):
      continue
    m, n, k = info["shape"]
    r = next((x for x in roles if (x["role"], x["quant"]) == (info["role"], info.get("quant"))), None)
    wb = r["weight_bytes"] if r else 0.0
    slot = named.setdefault((info["role"], str(info.get("quant"))), {"calls": 0, "us": 0.0, "flops": 0.0, "bytes": 0.0, "t": []})
    slot["calls"] += t["calls"]
    slot["us"] += t["wall_us"]
    slot["flops"] += t["calls"] * 2.0 * m * n * k
    slot["bytes"] += t["calls"] * gemm(m, n, k, wb)
    slot["t"].append(t)
    taken.add(ti)
  for (role, quant), slot in named.items():
    ts = slot["t"]
    t = {"template": " + ".join(x["template"] for x in ts), "path": _merge_paths([x["path"] for x in ts])}
    rows.append(_row(f"{role} {quant}", "role", t, slot["calls"], slot["us"], slot["flops"], slot["bytes"], peaks,
                     bandwidth_gbs, roles=[(role, quant)], role_source="the engine's own program metadata (role, M, N, K)"))
  roles = [r for r in roles if (r["role"], str(r["quant"])) not in named]
  ambiguous = None
  for w in widths:
    here = [ti for ti, t in enumerate(templates) if t.get("width") == w]
    n_w, m_w = (chunk_widths.count(w), float(w)) if w else (chunks, mean_m)
    tag = f" ({w}-token chunks)" if w else ""
    left = [templates[ti] if ti not in taken else {**templates[ti], "path": {"status": "taken"}} for ti in here]
    choice, why = pair(left, roles, chunks=n_w, mean_m=m_w, peaks=peaks, bandwidth_gbs=bandwidth_gbs)
    ambiguous = ambiguous or why
    shared: dict[int, list[dict[str, Any]]] = {}
    for i, (kind, ts) in choice.items():
      if kind == "shared":
        shared.setdefault(here[ts[0]], []).append(roles[i])
      else:  # one role from templates that serve it alone
        r, mine = roles[i], [templates[here[t]] for t in ts]
        t = {"template": " + ".join(x["template"] for x in mine), "path": _merge_paths([x["path"] for x in mine])}
        calls = sum(x["calls"] for x in mine)
        for x in mine:
          for l in x["launches"]:
            shapes[id(l)] = [(r, 1.0)]
        rows.append(_row(f"{r['role']} {r['quant']}{tag}", "role", t, calls, sum(x["wall_us"] for x in mine),
                         calls * 2.0 * m_w * r["n"] * r["k"], calls * gemm(m_w, r["n"], r["k"], r["weight_bytes"]),
                         peaks, bandwidth_gbs, roles=[(r["role"], r["quant"])], skipped=r["count"] * n_w - calls,
                         tokens_per_call=m_w))
        taken.update(here[t] for t in ts)
    for ti, rs in sorted(shared.items()):
      t = templates[ti]
      gs = sorted(role_groups(rs), key=lambda g: g["nk"])
      p = m_splits(gs, t, n_w, m_w)
      parts, m = split(gs, t, n_w * p), m_w / p
      for gi, g in enumerate(gs):
        mine = parts[gi]
        calls = len(mine)
        for l in mine:
          shapes[id(l)] = [(r, r["count"] / g["count"]) for r in g["roles"]]
        per_call_flops = sum(2.0 * m * r["n"] * r["k"] * r["count"] for r in g["roles"]) / g["count"]
        per_call_bytes = sum(gemm(m, r["n"], r["k"], r["weight_bytes"]) * r["count"] for r in g["roles"]) / g["count"]
        rows.append(_row(g["name"] + tag, "role", t, calls, sum(l["wall_us"] for l in mine), calls * per_call_flops,
                         calls * per_call_bytes, peaks, bandwidth_gbs, roles=[(r["role"], r["quant"]) for r in g["roles"]],
                         skipped=g["count"] * n_w * p - calls, tokens_per_call=m))
      taken.add(ti)
  placed = {rr for row in rows for rr in row.get("roles") or []}
  # a GEMV over one token (the output head on the last token): paired by count and bytes, as decode pairs
  # work only a call's last token does, once per call: a role run once per token (the output head)
  left_roles = [r for r in roles if (r["role"], r["quant"]) not in placed and r["count"] == 1]
  # only kernels with no matrix instruction: a GEMV over one token streams its weight, it does not tile it
  loose = [l for ti, t in enumerate(templates) if ti not in taken and t["path"]["status"] == "none" for l in t["launches"]]
  gemv_rows = []
  if left_roles:
    from boltbeam.collectors.attribution import groups as key_groups
    kg = key_groups([{k: v for k, v in l.items() if k != "start_us"} for l in loose])
    ceil_roles = [{"role": r["role"], "quant": r["quant"], "bytes_moved": r["weight_bytes"] * r["count"]} for r in left_roles]
    res = attribute(kg, ceil_roles, {(r["role"], r["quant"]): r["count"] for r in left_roles}, outputs, bandwidth_gbs)
    for p in res["pairs"]:
      g = next(x for x in res["groups"] if x["key"] == p["key"])
      r = next(x for x in left_roles if (x["role"], x["quant"]) == (p["role"], p["quant"]))
      t = by_t[template(g["key"])]
      gemv_rows.append(_row(f"{r['role']} {r['quant']}", "role", t, g["calls"], g["wall_us"], g["calls"] * 2.0 * r["n"] * r["k"],
                            g["calls"] * gemm(1, r["n"], r["k"], r["weight_bytes"]), peaks, bandwidth_gbs,
                            roles=[(r["role"], r["quant"])], tokens_per_call=1, key=g["key"]))
  rows += gemv_rows
  gemv_keys = {r["key"] for r in gemv_rows if r.get("key")}
  others: dict[str, dict[str, Any]] = {}
  att = work.attention(profile or {}, length or int(sum(chunk_widths)), chunk_widths, formats.get("kv") or "F16",
                       formats.get("attention_activations") or acts, depth=depth) if profile else None
  rec = work.recurrent(profile or {}, length or int(sum(chunk_widths)), chunk_widths) if profile else None
  for ti, t in enumerate(templates):
    if ti in taken:
      continue
    for l in t["launches"]:
      if l["key"] in gemv_keys:
        continue
      # attention's kernels are one row with attention's work, except one with no matrix instruction (its fixup):
      # that, like every other kernel, is its own row, named by its template's stem
      kind = ("attention" if t.get("attention") and t["path"]["status"] != "none" else
              "recurrent" if t.get("recurrent") and rec else stem(t["template"]))
      o = others.setdefault(kind, {"calls": 0, "wall_us": 0.0, "templates": {}})
      o["calls"] += 1
      o["wall_us"] += l["wall_us"]
      o["templates"][t["template"]] = t["path"]
  for kind, o in others.items():
    paths = {p["words"] for p in o["templates"].values()}
    if kind == "attention" and att:
      t = {"path": _merge_paths(list(o["templates"].values())), "template": " + ".join(sorted(o["templates"]))}
      rows.append(_row("attention", "attention", t, o["calls"], o["wall_us"], att["flops"], att["bytes"], peaks, bandwidth_gbs,
                       note=att["words"]))
    elif kind == "recurrent" and rec:
      rows.append(_recurrent_row(o, rec, peaks, bandwidth_gbs, work.recurrent_executed(
        profile or {}, length or int(sum(chunk_widths)), formats.get("recurrent_form"))))
    elif (fix := _fixup_row(kind, o, ordered, shapes, formats, mean_m, bandwidth_gbs, cache)) is not None:
      rows.append(fix)
    else:
      matrix = any(p.get("status") in ("one", "mixed") for p in o["templates"].values())
      rows.append({"name": kind, "what": "kernels", "calls": o["calls"], "ms": o["wall_us"] / 1000.0, "flops": None,
                   "bytes": None, "path": " | ".join(sorted(paths)), "ceiling_ms": None, "mfu_pct": None,
                   "templates": sorted(o["templates"]),
                   "refused": "no ceiling: a matrix kernel no role could be paired with, so its work is not known"
                              if matrix else None})
  busy = sum(r["ms"] for r in rows)
  wall = (end - begin) / 1000.0 if untraced_ms is None else float(untraced_ms)
  return {"window": window_words, "launches": len(window), "wall_ms": wall, "profiled_wall_ms": (end - begin) / 1000.0,
          "wall_source": "the untraced prefill (the engine's own timing slows its run)" if untraced_ms is not None else
                         "the captured prefill, first kernel to last", "busy_ms": busy, "gaps_ms": wall - busy,
          "overlap": overlap, "chunks": chunk_widths, "rows": sorted(rows, key=lambda r: -r["ms"]),
          "ambiguous": ambiguous, "rule": PAIRING_RULE,
          "templates": [{"template": t["template"], "calls": t["calls"], "ms": t["wall_us"] / 1000.0, "path": t["path"]}
                        for t in templates]}


def stem(tmpl:str) -> str:
  """A template's name without its arguments or a trailing content hash: `mul_mat_q_stream_k_fixup<...>` is
  `mul_mat_q_stream_k_fixup`, `E_64_64_8_16_4_292d...16a14` is `E_64_64_8_16_4`."""
  import re
  name = re.sub(r"_[0-9a-f]{32,}$", "", tmpl.split("<", 1)[0])
  return name if len(name) <= STEM_CHARS else name[:STEM_CHARS - 1] + "…"


def _merge_paths(paths:list[dict[str, Any]]) -> dict[str, Any]:
  known: dict[str, int] = {}
  for p in paths:
    for k, n in (p.get("paths") or {}).items():
      known[k] = known.get(k, 0) + n
  if not known:
    return paths[0] if paths else {"status": "not_found", "paths": {}, "path": None, "words": "no code"}
  named = [p for p in paths if p.get("from_name") and p.get("paths")]
  tag = {"from_name": True, "name_source": named[0]["name_source"]} if named else {}
  label = f" ({FROM_NAME})" if named else ""
  if len(known) == 1:
    k = next(iter(known))
    return {"status": "one", "paths": known, "path": k, "words": k + label, **tag}
  return {"status": "mixed", "paths": known, "path": None, "words": " + ".join(sorted(known)) + label, **tag}


def _row(name:str, what:str, t:dict[str, Any], calls:int, wall_us:float, flops:float, nbytes:float,
         peaks:dict[str, float], bandwidth_gbs:float | None, **extra:Any) -> dict[str, Any]:
  """One row: time, operations, bytes, the path its instructions use, and its ceiling. A row whose kernel code was
  not found has no compute ceiling and says why (refused); a kernel with no matrix instruction is bound by bytes."""
  path = t["path"]
  peak = _peak(t, peaks)
  ms = wall_us / 1000.0
  if path["status"] == "not_found" or (path["status"] in ("one", "mixed") and peak is None):
    c = {"ms": None, "compute_ms": None, "memory_ms": None, "bound": None}
    refused = f"no ceiling: {path['words'] if path['status'] == 'not_found' else 'no measured peak for ' + path['words']}"
  else:
    c = work.ceiling(flops if peak else 0.0, nbytes, peak, bandwidth_gbs,
                     work.compute_ms(flops, path["paths"], peaks) if path["status"] == "mixed" else None)
    refused = None
    if path.get("from_name") and c["ms"] and c["ms"] > ms * PEAK_SLACK:  # a name's path may not beat the guards
      refused = (f"no ceiling: the path {path['words']} would put the kernel {c['ms']:.2f} ms over its {ms:.2f} ms "
                 f"({'above the peak' if c['bound'] == 'compute' else 'faster than the bandwidth'})")
      c = {"ms": None, "compute_ms": None, "memory_ms": None, "bound": None}
  return {"name": name, "what": what, "calls": calls, "ms": ms, "flops": flops, "bytes": nbytes, "template": t["template"],
          "path": path["words"], "path_status": path["status"], "peak_tflops": peak,
          "ceiling_ms": c["ms"], "compute_ms": c["compute_ms"], "memory_ms": c["memory_ms"], "bound": c["bound"],
          "mfu_pct": work.mfu(flops, ms, peak) if path["status"] in ("one", "mixed") and not refused else None,
          "refused": refused, "mixed": path["status"] == "mixed", **({"name_source": path["name_source"]} if path.get("from_name") else {}),
          **({"path_shares": work.path_shares(path["paths"])} if path["status"] == "mixed" else {}), **extra}


def _recurrent_row(o:dict[str, Any], rec:dict[str, Any], peaks:dict[str, float], bandwidth_gbs:float | None,
                   executed:dict[str, Any] | None = None) -> dict[str, Any]:
  """The recurrent kernels as one row with the recurrent block's work (prefill_work.recurrent). Their code issues no
  matrix instruction, so the compute side is the measured fp32 vector rate; a kernel whose code was found to use a
  matrix path keeps that path. With no measured fp32 rate the row is bound by its bytes alone and says so."""
  merged = _merge_paths(list(o["templates"].values()))
  if merged["status"] in ("one", "mixed"):
    path = merged
  elif merged["status"] == "not_found":
    path = merged  # the code was not read: no compute side is claimed
  elif rec["path"] in peaks:
    path = {"status": "one", "paths": {rec["path"]: 1}, "path": rec["path"], "words": "fp32 vector (no matrix instruction)"}
  else:
    path = {"status": "none", "paths": {}, "path": None, "words": "no matrix instruction; no measured fp32 vector rate"}
  t = {"path": path, "template": " + ".join(sorted(o["templates"]))}
  row = _row("recurrent", "recurrent", t, o["calls"], o["wall_us"], rec["flops"], rec["bytes"], peaks, bandwidth_gbs,
             note=rec["words"])
  if executed:  # what the engine's algorithm executes, beside the model's work; the ceiling is not changed by it
    row.update(executed_flops=executed["flops"], executed_words=executed["words"])
  return row


FIXUP_WORD = "stream_k_fixup"  # a kernel that combines the partial tiles a stream-k GEMM left (kind word, in its name)


def _fixup_row(kind:str, o:dict[str, Any], window:list[dict[str, Any]], shapes:dict[int, list[tuple[dict[str, Any], float]]],
               formats:dict[str, Any], mean_m:float, bandwidth_gbs:float | None,
               cache:dict[str, float] | None) -> dict[str, Any] | None:
  """A stream-k fixup's row with its bytes ceiling: each launch follows its GEMM launch on the stream; that GEMM's
  roles (N, K, weight block) and launch grid give the partial tiles it left (prefill_work.stream_k_partials), read
  once, and the tiles fixed, read and written, in the engine's partial format. Partials that fit the last-level cache,
  written by the GEMM just before, are read at its bandwidth; else at the memory's. None when this is not a fixup,
  or the adapter names no stream-k layout for it (its row then says so)."""
  if FIXUP_WORD not in kind:
    return None
  layout_of = formats.get("stream_k")
  tmpls = sorted(o["templates"])
  base = {"name": kind, "what": "kernels", "calls": o["calls"], "ms": o["wall_us"] / 1000.0, "flops": None, "path": "no matrix instruction",
          "templates": tmpls, "mfu_pct": None, "compute_ms": None, "bound": "memory"}
  layouts = {t: layout_of(t) if layout_of else None for t in tmpls}
  if not all(layouts.values()):
    return {**base, "bytes": None, "ceiling_ms": None,
            "refused": f"no ceiling: the engine adapter names no stream-k layout for {', '.join(t for t, v in layouts.items() if not v)}"}
  from boltbeam.quantization.quant import load_quant_registry
  quants = load_quant_registry()
  per_elem = work.elem_bytes(formats.get("partials") or "F32")
  nbytes, cached, unshaped, launches = 0.0, 0.0, 0, 0
  for i, l in enumerate(window):
    t = template(l["key"])
    if t not in layouts:
      continue
    launches += 1
    prev = window[i - 1] if i else None
    roles = shapes.get(id(prev)) if prev is not None else None
    if not roles:
      unshaped += 1
      continue
    lay = layouts[t]
    blocks = int(str(prev["key"]).split(" grid=", 1)[1].split()[0].split("x")[0]) if " grid=" in str(prev["key"]) else 0
    for r, wt in roles:
      q = quants.get(str(r["quant"]).upper())
      kb = q.block_elems if q else 1
      got = work.stream_k_partials(int(round(mean_m)), r["n"], r["k"], blocks=blocks, tile_m=lay["tile_m"], tile_n=lay["tile_n"],
                                   k_block=kb, k_iter=max(lay["k_iter"] // kb, 1))
      tile = lay["tile_m"] * lay["tile_n"] * per_elem
      b = (got["partials"] + 2 * got["tiles"]) * tile * wt
      nbytes += b
      if cache and blocks * tile <= cache.get("bytes", 0):
        cached += b
  bw_c, bw = (cache or {}).get("gbs"), bandwidth_gbs
  ms = (cached / (bw_c * 1e9) if bw_c else 0.0) * 1e3 + ((nbytes - cached) / (bw * 1e9) * 1e3 if bw else 0.0)
  return {**base, "bytes": nbytes, "ceiling_ms": ms if unshaped == 0 else None, "memory_ms": ms,
          "cached_bytes": cached,
          "note": (f"{launches} launches, each after its GEMM: partial tiles read once and tiles fixed read and written "
                   f"({formats.get('partials') or 'F32'}); {cached / max(nbytes, 1) * 100:.0f}% of the bytes fit the "
                   "last-level cache, read at its bandwidth"),
          "refused": None if unshaped == 0 else f"no ceiling: {unshaped} of its launches follow a GEMM no role was paired with"}
