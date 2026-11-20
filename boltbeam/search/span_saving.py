"""search/span-saving: what one candidate kernel that replaces a run of stock launches saves in the model, from the trace.

The saving is the run's measured stream wall time minus the candidate's in-model time, never the run's summed kernel
times (realization: a projection from summed times misses by the overlap the engine already has).

    wall      per occurrence: the run's last end minus the later of its first start and the end of all work started before
              it (the union, first-started wins, as the kernel ledger counts exclusive time). PDL overlap with the launch
              before the run and between its own launches is counted once; idle gaps between its launches are in it. Idle
              before the run (a host gap, a launch) is not: the candidate waits for it too. Nor is an idle gap inside the run
              of HOST_GAP_US or more: that is a host wait or a copy (decode-floor's idle classes: under 2 us is a launch),
              which one fused launch still waits for; `held` counts the occurrences that had one.
    summed    the run's kernel durations added: what a projection from per-kernel trace times counts. summed - wall is
              overlap a fused kernel cannot take out again.
    gaps      each later launch's start minus the union end before it: idle (> 0) or overlap (< 0) inside the run. Their
              median is the per-launch gap a fusion removes (below 0: the overlap it gives up).
    saving    wall - candidate (derived); per removed launch = saving / (launches - 1).

Medians over every occurrence, p10 / p90 beside them. Engine-blind: the run is a list of regular expressions over kernel
names (NAME or NAME@gridX), given by the caller."""
from __future__ import annotations

import collections, re, statistics
from typing import Any

from boltbeam.workflow.context_audit import derived, measured

SCHEMA = "boltbeam.span_saving.v1"
HOST_GAP_US = 2.0


def _q(xs:list[float], p:float) -> float:
  xs = sorted(xs)
  return xs[min(len(xs) - 1, max(0, round(p * (len(xs) - 1))))]


def _pattern(spec:str) -> tuple[re.Pattern, int | None]:
  name, _, grid = spec.rpartition("@") if re.fullmatch(r".*@\d+", spec) else (spec, "", "")
  return re.compile(name), int(grid) if grid else None


def spans(kernels:list[tuple], grids:list[tuple] | None, run:list[str], windows:list[tuple[int, int]] | None = None) -> list[dict[str, float]]:
  """Every occurrence of `run` (consecutive launches in start order on one stream) in the kernels (start, end, name,
  stream; ns), each {wall, summed, entry, gaps, held} in us. Each stream is read alone (PDL overlap is per stream), so
  another stream's kernels neither break a run nor count in its wall. Names match by re.search. windows: only
  occurrences that start inside one of them."""
  by = collections.defaultdict(list)
  for j, k in enumerate(kernels):
    by[k[3] if len(k) > 3 else None].append(j)
  return [o for js in by.values() for o in _stream_spans([kernels[j] for j in js], [grids[j] if grids else None for j in js], run, windows)]


def _stream_spans(kernels:list[tuple], grids:list, run:list[str], windows:list[tuple[int, int]] | None) -> list[dict[str, float]]:
  order = sorted(range(len(kernels)), key=lambda j: kernels[j][:2])
  ks = [kernels[j] for j in order]
  gs = [grids[j] for j in order]
  pats, n = [_pattern(r) for r in run], len(run)
  ok = lambda i, p: p[0].search(ks[i][2]) and (p[1] is None or (gs[i] or (None,))[0] == p[1])  # noqa: E731
  seen, m = [], 0  # union end of everything started up to and including i
  for k in ks:
    m = max(m, k[1]); seen.append(m)
  out, i = [], 0
  while i + n <= len(ks):
    if not all(ok(i + t, pats[t]) for t in range(n)):
      i += 1; continue
    if windows and not any(a <= ks[i][0] < b for a, b in windows):
      i += 1; continue
    run_k, prev = ks[i:i + n], seen[i - 1] if i else ks[0][0]
    gaps, u = [], max(prev, run_k[0][1])
    for k in run_k[1:]:
      gaps.append((k[0] - u) / 1e3); u = max(u, k[1])
    host = sum(g for g in gaps if g >= HOST_GAP_US)
    out.append({"wall": (seen[i + n - 1] - max(prev, run_k[0][0])) / 1e3 - host, "held": host > 0,
                "summed": sum(e - s for s, e, *_ in run_k) / 1e3, "entry": (run_k[0][0] - prev) / 1e3, "gaps": gaps})
    i += n
  return out


def summary(occ:list[dict[str, Any]], source:str) -> dict[str, Any]:
  """Medians (p10, p90) of an occurrence list: wall, summed, overlap (summed - wall), entry, per-launch gap."""
  if not occ:
    return {"occurrences": 0}
  stat = lambda xs: {"median": round(statistics.median(xs), 3), "p10": round(_q(xs, 0.1), 3), "p90": round(_q(xs, 0.9), 3)}  # noqa: E731
  src = f"{source}: {len(occ)} occurrences"
  return {"occurrences": len(occ), "launches": len(occ[0]["gaps"]) + 1, "held": sum(o["held"] for o in occ),
          "wall_us": measured(stat([o["wall"] for o in occ]), src + ", last end - union end before the first start"),
          "summed_us": measured(stat([o["summed"] for o in occ]), src + ", kernel durations added"),
          "overlap_us": measured(stat([o["summed"] - o["wall"] for o in occ]), src + ", summed - wall: overlap net of short idle"),
          "entry_us": measured(stat([o["entry"] for o in occ]), src + ", first start - union end before it (idle > 0 is outside the wall)"),
          "gap_us": measured(stat([g for o in occ for g in o["gaps"] if g < HOST_GAP_US] or [0.0]), src + ", each later launch's start - the union end before it, "
                             "host waits (>= HOST_GAP_US) left out (0 for one launch)")}


def predict(s:dict[str, Any], candidate_us:float, candidate_source:str, per_step:float | None = None, label:str = "measured") -> dict[str, Any]:
  """The candidate's saving a copy, overlap-aware (wall - candidate), beside the summed-time projection it replaces.
  label: the candidate time's (measured: a bench file; estimate: a model's figure)."""
  wall, summed, n = s["wall_us"]["value"]["median"], s["summed_us"]["value"]["median"], s["launches"]
  out = {"candidate_us": {"value": candidate_us, "label": label, "source": candidate_source},
         "saving_us": derived(round(wall - candidate_us, 3), "wall median - candidate", {"wall": wall, "candidate": candidate_us}),
         "summed_projection_us": derived(round(summed - candidate_us, 3), "summed median - candidate (the projection this replaces)", {"summed": summed}),
         "per_removed_launch_us": derived(round((wall - candidate_us) / (n - 1), 3) if n > 1 else None, "saving / (launches - 1)", {"launches": n}),
         "gap_removed_us": derived(round(max(0.0, s["gap_us"]["value"]["median"]) * (n - 1), 3), "max(0, gap median) x (launches - 1): idle a fusion takes out",
                                   {"gap_median": s["gap_us"]["value"]["median"]})}
  if per_step:
    out["saving_ms_step"] = derived(round((wall - candidate_us) * per_step / 1e3, 4), "saving x occurrences a step / 1000", {"per_step": per_step})
  return out


def lines(rec:dict[str, Any]) -> list[str]:
  s, p = rec["spans"], rec.get("prediction") or {}
  if not s.get("occurrences"):
    return [f"span-saving: {' > '.join(rec['run'])}: no occurrence in {rec['source']}"]
  v = lambda k: s[k]["value"]  # noqa: E731
  out = [f"span-saving: {' > '.join(rec['run'])} in {rec['source']}: {s['occurrences']} occurrences, {s['launches']} launches",
         f"  wall {v('wall_us')['median']} us (p10 {v('wall_us')['p10']}, p90 {v('wall_us')['p90']}); summed {v('summed_us')['median']}; "
         f"overlap {v('overlap_us')['median']}; entry {v('entry_us')['median']}; gap a launch {v('gap_us')['median']} (measured)"]
  if p:
    out.append(f"  candidate {p['candidate_us']['value']} us: saving {p['saving_us']['value']} us a copy (derived; the summed projection says "
               f"{p['summed_projection_us']['value']}), {p['per_removed_launch_us']['value']} a removed launch"
               + (f", {p['saving_ms_step']['value']} ms a step" if "saving_ms_step" in p else ""))
  return out
