"""Which model role each captured GPU kernel ran for, from bytes and launch counts, for any provider.

A capture from outside the provider (nsys, rocprofv3, metal-system-trace) sees kernels, not roles. Kernel names
differ per provider and per build, so this module never reads them: a name is only an opaque key that groups
launches of one program (with its launch geometry, when the capture has it).

Two facts tie a kernel group to a role:
  calls per token   each role runs `count` times per token (model_profile.json roles). A group's launches over
                    the measured tokens must equal that count.
  bytes per call    each role moves `bytes_moved / count` bytes per call (the roofline's own figure). Within one
                    count, a role that moves more bytes takes the group that runs longer per call (decode GEMVs
                    stream their weights). A pairing that would need more bandwidth than the chip's peak is
                    physically impossible and is refused.

Kernels on several streams can run at once. Each launch then gets only its share of the GPU busy time
(share_overlap: shared time split equally), so a token's kernel time is the union of its intervals, never the
sum. A fused kernel (llama.cpp on CUDA runs ffn gate and up as one) makes 1/k of a role's calls and moves k
calls' bytes each.

A MoE expert stack is read k of n experts per token (the limit counts k matrices per stack), but the engine launches
it once per layer: llama.cpp's mul_mat_id at one decode token is one kernel (CUDA: mul_mat_vec_q with grid.y = k, each
y-block reading the expert ids[y] names; Metal: one mul_mv_id dispatch over k), so that launch moves k experts' bytes.
The profile's count for an expert role is its stacks, the launches per token, and bytes per call is the role's limit
bytes over that count: k experts' bytes. CUDA also fuses an expert gate and up with their GLU into one launch that
reads 2k matrices; that is the fused case below with k = 2 (the gate_up role's count is 2 per layer). The router and
the shared experts are plain matrices: one launch per layer each.

The pairing is one assignment over every role at once (PAIRING_RULE), not a walk that places one role and moves
on: on the 27B hybrid, ffn_down and the fused gate+up kernel both run 64 times per token, and placing ffn_down
first gave it the 100 MB launch (755 GB/s, plausible alone) and left the 50 MB launch to no one. Paired by bytes,
more bytes per launch is more time per launch within one launch count, and both fit at 88 to 89% of peak.

Time in groups no role takes is "not attributed". Nothing is guessed: a role with no matching group stays absent,
and loss() (tinygrad_role_time.py) then applies the same floor rule as for every provider.
"""
from __future__ import annotations

import pathlib
from typing import Any, Iterable

ROLE_SOURCE = "attributed_by_bytes_and_count"
PEAK_SLACK = 1.05  # the registry's peak is a measured figure; a kernel may beat it by measurement noise only
COUNT_TOLERANCE = 0.02  # launches per token must be this close to a whole number
MAX_FUSED = 3  # a kernel may serve at most this many of one role's calls (gate and up fused: 2)


def groups(launches:Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
  """Sum launches by their identity key: {"key", "calls", "wall_us"}. A launch row may already carry calls."""
  out: dict[str, dict[str, Any]] = {}
  for l in launches:
    g = out.setdefault(str(l["key"]), {"key": str(l["key"]), "calls": 0, "wall_us": 0.0, "launches": []})
    g["calls"] += int(l.get("calls", 1))
    g["wall_us"] += float(l["wall_us"])
    if "calls" not in l:  # one launch: its own duration (for clusters) and its share of the busy time
      g["launches"].append((float(l.get("raw_wall_us", l["wall_us"])), float(l["wall_us"])))
  return list(out.values())


MAX_NODES = 200_000  # pairings the walk may visit before the table is refused as too many to tell apart
PAIRING_RULE = ("a role takes a kernel group that runs its count of times per token, or 1/k of them for a fused kernel "
                f"moving k calls' bytes (k up to {MAX_FUSED}), at no more than the chip's peak bandwidth; among groups with "
                "one launch count, more bytes per launch is more time per launch; the assignment that places the most "
                "roles wins, then the one with the fewest fused kernels, then the walk's order (more bytes first, an exact "
                "group before a fused one, a slower group before a faster one); a group no role's bytes fit stays unattributed")


def attribute(kernel_groups:list[dict[str, Any]], roles:list[dict[str, Any]], counts:dict[tuple[str, str], int],
              tokens:int, peak_gbs:float | None) -> dict[str, Any]:
  """Pair kernel groups with roles by PAIRING_RULE. roles: the ceiling's decode roles (role, quant, bytes_moved,
  floor_ms). counts: calls per token per (role, quant). Returns {"pairs": [...], "groups": every group with its role
  or None, "unpaired", "assembled", "ambiguous", "rule"}. Roles the one assignment leaves out are then tried by
  assemble() from the groups left and their duration clusters."""
  known = sorted([{**r, "count": int(counts[(r["role"], r["quant"])]), "bytes_per_call": r["bytes_moved"] / counts[(r["role"], r["quant"])]}
                  for r in roles if counts.get((r["role"], r["quant"]))], key=lambda r: -r["bytes_per_call"])
  rows = []
  for g in kernel_groups:
    cpt = g["calls"] / tokens
    whole = round(cpt)
    rows.append({**g, "calls_per_token": cpt, "count": whole if whole and abs(cpt - whole) <= COUNT_TOLERANCE * whole
                 else None, "us_per_call": g["wall_us"] / g["calls"], "role": None, "quant": None})
  choice, ambiguous = _pairing(known, [_candidates(r, rows, peak_gbs) for r in known], rows)
  pairs = []
  for i, (gi, k) in sorted(choice.items()):
    r, g = known[i], rows[gi]
    gbs = k * r["bytes_per_call"] / (g["us_per_call"] * 1e3)
    g.update(role=r["role"], quant=r["quant"], gbs=gbs, fused=k)
    pairs.append({"role": r["role"], "quant": r["quant"], "key": g["key"], "count": r["count"], "fused": k, "gbs": gbs})
  left = [r for i, r in enumerate(known) if i not in choice]
  assembled, ambiguous_pieces = assemble(rows, left, tokens, peak_gbs) if left else ([], None)
  built = {(a["role"], a["quant"]) for a in assembled}
  unpaired = [{"role": r["role"], "quant": r["quant"], "count": r["count"]} for r in left if (r["role"], r["quant"]) not in built]
  return {"pairs": pairs, "groups": rows, "unpaired": unpaired, "assembled": assembled,
          "ambiguous": ambiguous or ambiguous_pieces, "rule": PAIRING_RULE}


def _candidates(r:dict[str, Any], rows:list[dict[str, Any]], peak_gbs:float | None) -> list[tuple[int, int]]:
  """The (group index, k) a role may take, in the walk's order: exact (k = 1) before fused, the slower group first.
  A group is a candidate when its launches per token are the role's count over k and k calls' bytes over its time
  per launch is no more than the peak (a faster pairing is physically impossible)."""
  out = []
  for k in range(1, MAX_FUSED + 1):
    if r["count"] % k:
      continue
    fit = [(i, g) for i, g in enumerate(rows) if g["count"] == r["count"] // k]
    for i, g in sorted(fit, key=lambda ig: -ig[1]["us_per_call"]):
      if not peak_gbs or k * r["bytes_per_call"] / (g["us_per_call"] * 1e3) <= peak_gbs * PEAK_SLACK:
        out.append((i, k))
  return out


class _TooMany(Exception):
  pass


def _pairing(known:list[dict[str, Any]], cands:list[list[tuple[int, int]]],
             rows:list[dict[str, Any]]) -> tuple[dict[int, tuple[int, int]], dict[str, Any] | None]:
  """The one assignment role index -> (group index, k) by PAIRING_RULE: a depth-first walk over the roles (more
  bytes per call first), each trying its candidates in order and then going unpaired, keeping the first assignment
  at the best score (roles placed, then fewest fused). Within one launch count the bytes per launch and the time
  per launch must order the same way. A walk past MAX_NODES refuses the whole pairing, named."""
  best:dict[str, Any] = {"score": (-1, 0), "choice": {}}
  nodes = 0

  def monotone(choice:dict[int, tuple[int, int]], i:int, gi:int, k:int) -> bool:
    g, mine = rows[gi], k * known[i]["bytes_per_call"]
    for j, (gj, kj) in choice.items():
      h, theirs = rows[gj], kj * known[j]["bytes_per_call"]
      if h["count"] == g["count"] and ((mine > theirs and g["us_per_call"] <= h["us_per_call"])
                                       or (mine < theirs and g["us_per_call"] >= h["us_per_call"])):
        return False
    return True

  def walk(i:int, choice:dict[int, tuple[int, int]], used:set[int], fused:int) -> None:
    nonlocal nodes
    nodes += 1
    if nodes > MAX_NODES:
      raise _TooMany
    if len(choice) + (len(known) - i) < best["score"][0]:
      return  # cannot place as many roles as the best found
    if i == len(known):
      if (len(choice), -fused) > best["score"]:
        best.update(score=(len(choice), -fused), choice=dict(choice))
      return
    for gi, k in cands[i]:
      if gi in used or not monotone(choice, i, gi, k):
        continue
      choice[i], _ = (gi, k), used.add(gi)
      walk(i + 1, choice, used, fused + (k > 1))
      del choice[i]
      used.discard(gi)
    walk(i + 1, choice, used, fused)

  try:
    walk(0, {}, set(), 0)
  except _TooMany:
    return {}, {"reason": f"more than {MAX_NODES} ways to pair {len(known)} roles with the kernel groups: too many to tell apart"}
  return best["choice"], None


# --- roles assembled from several groups or duration clusters -------------------------------------------------

SPLIT_RATIO = 1.5  # two duration clusters count as different work only when their means differ this much
SPLIT_SHARE = 0.05  # and each holds at least this share of the calls
PIECE_SPREAD = 1.5  # the pieces of one role move the same bytes per call: their times per call agree within this
COUNT_SLACK = 1.0  # a role's pieces may miss its count by this many calls per token (window edges)
MAX_PIECES = 8  # more candidate pieces than this is too many to tell apart: refused as ambiguous


def split_clusters(g:dict[str, Any], tokens:int) -> list[dict[str, Any]] | None:
  """Two pieces of one kernel group when its per-call durations are clearly bimodal, else None. 2-means on log
  duration over the long capture's launches; the short capture's launches are subtracted per cluster."""
  import math
  durs = sorted(d for d, _ in g.get("launches") or [])
  if len(durs) < 20:
    return None
  logs = [math.log(max(d, 1e-6)) for d in durs]
  lo, hi = logs[len(logs) // 10], logs[len(logs) * 9 // 10]
  a = b = []
  for _ in range(50):
    cut = (lo + hi) / 2
    a = [x for x in logs if x <= cut]
    b = [x for x in logs if x > cut]
    if not a or not b:
      return None
    lo, hi = sum(a) / len(a), sum(b) / len(b)
  if math.exp(hi - lo) < SPLIT_RATIO or min(len(a), len(b)) < SPLIT_SHARE * len(logs):
    return None
  edge = math.exp((lo + hi) / 2)
  out = []
  for name, test in (("fast", lambda d: d <= edge), ("slow", lambda d: d > edge)):
    calls = sum(1 for d, _ in g["launches"] if test(d)) - sum(1 for d, _ in g.get("short_launches") or [] if test(d))
    us = sum(s for d, s in g["launches"] if test(d)) - sum(s for d, s in g.get("short_launches") or [] if test(d))
    if calls <= 0 or us <= 0:
      return None
    out.append({"key": f"{g['key']} cluster={name}", "parent": g["key"], "cluster": name, "edge_us": edge,
                "calls": calls, "wall_us": us, "calls_per_token": calls / tokens, "us_per_call": us / calls})
  return out


def assemble(rows:list[dict[str, Any]], left:list[dict[str, Any]], tokens:int,
             peak_gbs:float | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
  """Roles exact matching could not place, each built from one or more unpaired groups or duration clusters whose
  calls per token add up to the role's count. A piece must be slow enough to move the role's bytes per call
  (no more than the peak), the pieces of one role must agree in time per call (same bytes), and a role that
  moves more bytes per call must take longer per call than one that moves fewer. Exactly one assignment must
  satisfy this; none or several leave the roles unpaired, and several are recorded as ambiguous."""
  import itertools
  smallest = min(r["bytes_per_call"] for r in left)
  pieces = []
  for g in [g for g in rows if g["role"] is None]:
    parts = split_clusters(g, tokens) or [{**g, "parent": g["key"], "cluster": None}]
    for part in parts:  # a piece too fast to move even the smallest role's bytes is not weight work
      if not peak_gbs or smallest / (part["us_per_call"] * 1e3) <= peak_gbs * PEAK_SLACK:
        pieces.append(part)
  if not pieces:
    return [], None
  if len(pieces) > MAX_PIECES:
    return [], {"reason": f"{len(pieces)} candidate kernel groups for {len(left)} roles: too many to tell apart"}
  valid = []
  for choice in itertools.product(range(len(left) + 1), repeat=len(pieces)):  # len(left) = no role
    per = {i: [pieces[p] for p, c in enumerate(choice) if c == i] for i in range(len(left))}
    ok = True
    for i, r in enumerate(left):
      ps = per[i]
      if not ps or abs(sum(p["calls_per_token"] for p in ps) - r["count"]) > COUNT_SLACK:
        ok = False
        break
      times = [p["us_per_call"] for p in ps]
      if max(times) / min(times) > PIECE_SPREAD or (peak_gbs and r["bytes_per_call"] / (min(times) * 1e3) > peak_gbs * PEAK_SLACK):
        ok = False
        break
    if ok:  # more bytes per call, more time per call, across roles
      for i, j in itertools.permutations(range(len(left)), 2):
        if left[i]["bytes_per_call"] > left[j]["bytes_per_call"] and            min(p["us_per_call"] for p in per[i]) <= max(p["us_per_call"] for p in per[j]):
          ok = False
          break
    if ok:
      valid.append(per)
  if len(valid) != 1:
    return [], ({"reason": f"{len(valid)} ways to build the roles from the kernels left; refused as ambiguous"}
                if len(valid) > 1 else None)
  out = []
  for i, r in enumerate(left):
    ps = valid[0][i]
    for p in ps:
      if p["cluster"] is None:
        g = next(g for g in rows if g["key"] == p["key"])
        g["role"], g["quant"] = r["role"], r["quant"]
      else:
        rows.append({**p, "role": r["role"], "quant": r["quant"], "count": None})
    out.append({"role": r["role"], "quant": r["quant"], "count": r["count"],
                "pieces": [{"key": p["parent"], "cluster": p["cluster"], "edge_us": p.get("edge_us"),
                            "calls_per_token": p["calls_per_token"], "us_per_call": p["us_per_call"],
                            "gbs": r["bytes_per_call"] / (p["us_per_call"] * 1e3)} for p in ps]})
  # a split group's rows are its clusters now: drop the whole group so its time is not counted twice
  for key in {p["key"] for a in out for p in a["pieces"] if p["cluster"]}:
    whole = next(g for g in rows if g["key"] == key)
    taken = {g["cluster"] for g in rows if g.get("parent") == key and g.get("cluster")}
    rows.remove(whole)
    for part in split_clusters(whole, tokens) or []:  # a cluster no role took stays as an unattributed row
      if part["cluster"] not in taken:
        rows.append({**part, "role": None, "quant": None, "count": None})
  return out, None


def trace_rows(result:dict[str, Any], *, context:int, time_source:str) -> list[dict[str, Any]]:
  """Kernel rows of a boltbeam.timing_trace.v1 for the attributed groups, the shape loss() reads."""
  return [{"scope": "kernel", "context": context, "kernel": g["key"], "calls": g["calls"], "wall_us": g["wall_us"],
           "role": g["role"], "quant": g["quant"], "role_source": ROLE_SOURCE if g["role"] else None,
           "time_source": time_source} for g in result["groups"]]


def agreement(by_bytes:dict[str, Any], by_metadata:dict[str, tuple[str | None, str | None]]) -> dict[str, Any]:
  """Compare this module's pairing with another attribution (tinygrad's model metadata), weighted by time."""
  total = same = 0.0
  diffs = []
  for g in by_bytes["groups"]:
    total += g["wall_us"]
    want = by_metadata.get(g["key"], (None, None))
    got = (g["role"], g["quant"])
    if got == want or (got[0] is None and want[0] is None):
      same += g["wall_us"]
    else:
      diffs.append({"key": g["key"], "bytes": got, "metadata": want, "wall_us": g["wall_us"]})
  return {"time_agreement": same / total if total else None, "differences": diffs}


OVERLAP_RULE = "time when kernels ran at once is split equally among them, so the shares add up to the GPU busy time"


def share_overlap(launches:list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
  """Each launch's share of the GPU busy time. Kernels on different streams can run at once; summing their
  durations then counts that time twice and can exceed the token. A sweep over the start and end times splits
  every stretch equally among the kernels running in it, so the shares sum to the union of the intervals.
  Launches with no start time are kept as they are (they cannot overlap anything known)."""
  timed = [l for l in launches if "start_us" in l]
  events = sorted([(l["start_us"], 1, i) for i, l in enumerate(timed)] + [(l["start_us"] + l["wall_us"], -1, i) for i, l in enumerate(timed)],
                  key=lambda e: (e[0], e[1]))  # ends before starts at the same instant: touching is not overlap
  share = [0.0] * len(timed)
  active: set[int] = set()
  last = None
  for t, kind, i in events:
    if active and last is not None and t > last:
      part = (t - last) / len(active)
      for j in active:
        share[j] += part
    last = t
    if kind == 1:
      active.add(i)
    else:
      active.discard(i)
  total = sum(l["wall_us"] for l in launches)
  out = [{**l, "wall_us": share[i], "raw_wall_us": l["wall_us"]} for i, l in enumerate(timed)]
  out += [l for l in launches if "start_us" not in l]
  union = sum(share) + sum(l["wall_us"] for l in launches if "start_us" not in l)
  return out, {"sum_us": total, "union_us": union}


def shared_window(long:list[dict[str, Any]], short:list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """The decode window from two captures, each first reduced to its share of the GPU busy time."""
  long_s, lo = share_overlap(long)
  short_s, so = share_overlap(short)
  window = difference(groups(long_s), groups(short_s))
  return window, {"rule": OVERLAP_RULE, "long_sum_us": lo["sum_us"], "long_union_us": lo["union_us"],
                  "short_sum_us": so["sum_us"], "short_union_us": so["union_us"]}


def difference(long:list[dict[str, Any]], short:list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Kernel groups of the long capture minus the short one, per key: the work only the extra tokens did."""
  base = {g["key"]: g for g in short}
  out = []
  for g in long:
    b = base.get(g["key"], {"calls": 0, "wall_us": 0.0})
    calls, us = g["calls"] - b["calls"], g["wall_us"] - b["wall_us"]
    if calls > 0 and us > 0:
      out.append({"key": g["key"], "calls": calls, "wall_us": us, "launches": g.get("launches") or [],
                  "short_launches": b.get("launches") or []})
  return out


class CaptureRefused(RuntimeError):
  """The capture cannot be a measurement. The message says why."""


def check_window(window:list[dict[str, Any]], *, method:str, tokens:int, token_ms:float | None,
                 long_kernels:int, band:float = 0.01) -> None:
  """Refuse a window with no kernels, or with more GPU time per token than the captured run took per token by more
  than the chip's plausibility band (screen.plausibility_band)."""
  if not window:
    raise CaptureRefused(f"{method} captured {long_kernels} kernels, but none ran only in the decode tokens: the "
                         "decode's kernels were not visible to it (for llama.cpp on CUDA, its CUDA graphs)")
  per_token = sum(g["wall_us"] for g in window) / tokens / 1000.0
  if token_ms and per_token > token_ms * (1 + band):
    raise CaptureRefused(f"{method} counted {per_token:.2f} ms of GPU time per token, more than the {token_ms:.2f} ms "
                         f"the captured run took per token by more than ±{100 * band:.1f}%, so it counted work "
                         "outside the decode tokens")


def provider_trace(run, window:list[dict[str, Any]], *, provider:str, method:str, model_id:str, target,
                   context:int, tokens:int, out:str, token_ms:float | None = None,
                   long_kernels:int = 0, overlap:dict[str, Any] | None = None, batch:int = 1) -> dict[str, Any]:
  """A boltbeam.timing_trace.v1 of one provider's captured decode window, attributed, written to run/out.
  token_ms is the captured run's own time per token, as the runtime printed it. With batch B, a token is one decode
  step of B streams: the weights are read once per step, so a role's calls per step are its count at any batch. Refused (CaptureRefused, no file
  written) when the window is empty or holds more GPU time per token than that."""
  (pathlib.Path(run) / out).unlink(missing_ok=True)  # a refused capture leaves no earlier result standing
  import json
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  from boltbeam.workflow.common import load_manifest
  from boltbeam.workflow.screen import _measured_vs_ceiling, _optional
  manifest = load_manifest(run)
  profile = _optional(run, "model_profile.json")
  ceil = _measured_vs_ceiling(manifest, profile)
  check_window(window, method=method, tokens=tokens, token_ms=token_ms, long_kernels=long_kernels,
               band=max(float(ceil.get("band") or 0.0), 0.01))
  counts = {(r["role"], r["quant"]): int(r["count"]) for r in profile.get("roles", []) if r.get("count")}
  result = attribute(window, ceil.get("_roles") or [], counts, tokens, target.memory_bandwidth_gbs)
  wall = sum(g["wall_us"] for g in window)
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": model_id, "target_id": target.target_id, "workload": "decode",
           "provider_id": provider, "capture": {"method": method, "reason": None},
           "timing_source": f"{method} kernel timeline, attributed by bytes and count",
           "rows": [{"scope": "whole_step", "context": context, "decode_tokens": tokens, "wall_us": wall,
                     "tok_s": tokens / (wall / 1e6) if wall else None, "launch_count": sum(g["calls"] for g in window),
                     "measurement_scope": "summed_kernel_intervals", "time_source": method, "token_ms": token_ms,
                     "batch": batch}]
                   + trace_rows(result, context=context, time_source=method),
           "pairs": result["pairs"], "unpaired_roles": result["unpaired"], "assembled_roles": result["assembled"],
           "ambiguous": result["ambiguous"], "pairing_rule": result["rule"], "overlap": overlap}
  (pathlib.Path(run) / out).write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
  return trace
