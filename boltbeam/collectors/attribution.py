"""Which model role each captured GPU kernel ran for in a decode, for any provider.

A capture from outside the provider (nsys, rocprofv3, metal-system-trace) sees kernels, not roles. Kernel names
differ per provider and per build, so this module reads them for two things only: the kind words the tie-out already
uses for the attention and recurrent rows (tie_out.kind_of), whose kernels serve no weight role, and a GGUF weight
format a name states (pairing.stated_format: q4k_, mul_mv_q6_K), which that kernel alone may serve. Otherwise a name
is only an opaque key that groups launches of one program (with its launch geometry, when the capture has it).

The pairing is the layer walk (collectors/pairing.py, docs/pairing-layer-walk-20261011.md): the block kernels mark
the layers, the launches around them are the layer's projections, the profile's counts and fine classes say which;
a token is a unit of M = 1, so a role's launches read its weights and a launch cannot do that faster than the chip's
measured bandwidth. A capture without launch times pairs the flat way: counts per token, bytes and time.

Kernels on several streams can run at once. Each launch then gets only its share of the GPU busy time
(share_overlap: shared time split equally), so a token's kernel time is the union of its intervals, never the
sum. A fused kernel (llama.cpp on CUDA runs ffn gate and up as one) makes 1/k of a role's calls and moves k
calls' bytes each.

A MoE expert stack is read k of n experts per token (the limit counts k matrices per stack), but the engine launches
it once per layer: llama.cpp's mul_mat_id at one decode token is one kernel (CUDA: mul_mat_vec_q with grid.y = k, each
y-block reading the expert ids[y] names; Metal: one mul_mv_id dispatch over k), so that launch moves k experts' bytes.
The profile's count for an expert role is its stacks, the launches per token, and bytes per call is the role's limit
bytes over that count: k experts' bytes. CUDA also fuses an expert gate and up with their GLU into one launch that
reads 2k matrices; that is the fused case with k = 2 (the gate_up role's count is 2 per layer). The router and
the shared experts are plain matrices: one launch per layer each.

Time in groups no role takes is "not attributed". Nothing is guessed: a role with no matching group stays absent,
and loss() (tinygrad_role_time.py) then applies the same floor rule as for every provider.
"""
from __future__ import annotations

import pathlib
from typing import Any, Iterable

from boltbeam.collectors import pairing
from boltbeam.collectors.pairing import BLOCKS  # noqa: F401  (the decode readers import it from here)
from boltbeam.role_key import role_key
from boltbeam.target import vendors

ROLE_SOURCE = "attributed_by_bytes_and_count"


def groups(launches:Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
  """Sum launches by their identity key: {"key", "calls", "wall_us"}. A launch row may already carry calls."""
  out: dict[str, dict[str, Any]] = {}
  for l in launches:
    g = out.setdefault(str(l["key"]), {"key": str(l["key"]), "calls": 0, "wall_us": 0.0, "launches": [], "starts": [],
                                       "streams": []})
    g["calls"] += int(l.get("calls", 1))
    g["wall_us"] += float(l["wall_us"])
    if "calls" not in l:  # one launch: its own duration, its share of the busy time, its start
      g["launches"].append((float(l.get("raw_wall_us", l["wall_us"])), float(l["wall_us"])))
      g["starts"].append(l.get("start_us"))
      g["streams"].append(l.get("stream"))
  return list(out.values())


def attribute(kernel_groups:list[dict[str, Any]], roles:list[dict[str, Any]], counts:dict[tuple, int],
              tokens:int, peak_gbs:float | None, *, kinds:dict[str, int] | None = None,
              read:dict[tuple, dict[str, Any]] | None = None) -> dict[str, Any]:
  """Pair kernel groups with roles by the layer walk. roles: the ceiling's decode roles (role, quant, bytes_moved,
  floor_ms, and fine or part_roles for an engine's packed matrix). counts: calls per token per role_key (or role, quant);
  kinds: layers per block kind (pairing.layer_kinds); read: the profile row of each role, keyed alike (its fine class,
  a hybrid's layer_kinds). peak_gbs: the bandwidth at the top of its measured uncertainty (targets.upper), None not checked. Returns {"pairs": [...], "groups":
  every group with its role or None (a group several roles share is one row per role), "unpaired", "ambiguous",
  "rule", "structure"}."""
  at = lambda d, r: (d or {}).get(role_key(r), (d or {}).get((r["role"], r["quant"])))  # noqa: E731
  rows = [{"role": r["role"], "quant": r["quant"], "count": n, "weight_bytes": r["bytes_moved"] / n, "nk": None,
           "role_class": (p := at(read, r) or {}).get("role_class"), "role_key": role_key(r), **({"fine": r["fine"]} if r.get("fine") else {}),
           **({"layer_kinds": p["layer_kinds"]} if sum(c for k in (p.get("layer_kinds") or {}).values() for c in k.values()) == n else {}),
           **({"per_layer": p["per_layer"]} if p.get("per_layer") else {})}
          for r in roles if (n := int(at(counts, r) or 0))]
  timed = all(g.get("launches") and len(g["launches"]) == g["calls"] and None not in (g.get("starts") or [None]) for g in kernel_groups)
  launches = [{"key": g["key"], "raw_wall_us": d, "wall_us": sh, "start_us": g["starts"][i], "stream": (g.get("streams") or [None] * g["calls"])[i]}
              for g in kernel_groups for i, (d, sh) in enumerate(g["launches"])] if timed and kernel_groups else []
  got = pairing.pair(launches, rows, kinds=kinds or {}, widths=[1] * tokens, bandwidth_gbs=peak_gbs,
                     flat=None if timed and kernel_groups else kernel_groups)
  by_key: dict[str, dict[str, Any]] = {g["key"]: g for g in kernel_groups}
  rows_out: list[dict[str, Any]] = []
  pairs: dict[tuple[str, str, str], dict[str, Any]] = {}
  taken: dict[str, float] = {}
  short = {x["role_key"] for u in got["unplaced"] for x in u["item"]["parts"]}  # a row placed in some layers only: not timed
  for p in got["placed"]:
    item, g = p["item"], p["group"]
    if short & {x["role_key"] for x in item["parts"]}: continue  # its launches would stand for all its calls
    key = g.get("key") or p["launches"][0][0]["key"]
    us = sum(l["wall_us"] for l, _, _, _ in p["launches"])
    for part in item["parts"]:
      share = part["weight_bytes"] * part["count"] / sum(x["weight_bytes"] * x["count"] for x in item["parts"])
      row = pairs.setdefault((key, part["role_key"]), {"key": key, "calls": 0, "wall_us": 0.0, "role": part["role"], "quant": part["quant"],
                                                  "fused": p["fused"], "count": part["count"], "weight_bytes": part["weight_bytes"]})
      row["calls"] += len(p["launches"])
      row["wall_us"] += us * share
    taken[key] = taken.get(key, 0.0) + us
  for (key, _), row in pairs.items():
    rows_out.append({**row, "calls_per_token": row["calls"] / tokens, "us_per_call": row["wall_us"] / row["calls"],
                     "gbs": row.pop("weight_bytes") * row["fused"] / (row["wall_us"] / row["calls"] * 1e3),
                     **({"shared_key": True} if sum(1 for k in pairs if k[0] == key) > 1 else {})})
  for key, g in by_key.items():
    left_us = g["wall_us"] - taken.get(key, 0.0)
    left_calls = g["calls"] - sum(r["calls"] for r in rows_out if r["key"] == key)
    if left_calls > 0:
      rows_out.append({"key": key, "calls": left_calls, "wall_us": max(left_us, 0.0), "role": None, "quant": None,
                       "calls_per_token": left_calls / tokens, "us_per_call": max(left_us, 0.0) / left_calls})
  pair_rows = [{"role": r["role"], "quant": r["quant"], "key": r["key"], "count": r.pop("count"), "fused": r["fused"],
                "gbs": r["gbs"]} for r in rows_out if r["role"]]
  unpaired = [{"role": u["item"]["role"], "quant": u["item"]["quant"], "count": u["item"]["count"], "reason": u["reason"]} for u in got["unplaced"]]
  return {"pairs": pair_rows, "groups": rows_out, "unpaired": unpaired, "ambiguous": got.get("ambiguous"), "rule": got["words"],
          "structure": got["structure"]}


def trace_rows(result:dict[str, Any], *, context:int, time_source:str) -> list[dict[str, Any]]:
  """Kernel rows of a boltbeam.timing_trace.v1 for the attributed groups, the shape loss() reads. A kernel several
  roles share is one row per role, keyed with the role so the rows stay distinct."""
  return [{"scope": "kernel", "context": context, "kernel": g["key"] + (f" role={g['role']}" if g.get("shared_key") else ""),
           "calls": g["calls"], "wall_us": g["wall_us"], "role": g["role"], "quant": g["quant"],
           "role_source": ROLE_SOURCE if g["role"] else None, "time_source": time_source,
           **{k: g[k] for k in ("shape", "logical_parts", "logical_flops") if g.get(k) is not None}}
          for g in result["groups"]]


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
  origin = min((l["start_us"] for l in timed), default=0.)
  events = sorted([(l["start_us"]-origin, 1, i) for i, l in enumerate(timed)] + [(l["start_us"]-origin + l["wall_us"], -1, i) for i, l in enumerate(timed)],
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


BLOCK_CUT_RULE = ("the block kernels cut the window: they run only in prefill and decode, so the long capture's block "
                  "launches past the short one's count are the extra tokens; every launch from the first of them on is "
                  "decode, and before it only the launches past the short capture's last block (the shared token's "
                  "output side and tail, the first extra layer's input side) are differenced, per kernel; a model "
                  "load that runs a decode kernel a varying number of times never enters the window")


def block_cut(long:list[dict[str, Any]], short:list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float] | None:
  """The decode window's launches, cut by time at the block kernels (BLOCK_CUT_RULE), and the GPU time left out before
  the cut. None when either capture lacks launch times or block kernels, or the long one has no extra block launch:
  the per-kernel difference then decides."""
  from boltbeam.workflow.tie_out import kind_of
  if any(l.get("start_us") is None for l in [*long, *short]):
    return None
  is_block = lambda l: kind_of({"kernel": str(l["key"])}) in BLOCKS  # noqa: E731
  bl = sorted(l["start_us"] for l in long if is_block(l))
  bs = sorted(l["start_us"] for l in short if is_block(l))
  if not bs or len(bl) <= len(bs):
    return None
  # the count cuts in the right place only when both captures ran the same block launches up to it. A start-up that
  # runs block kernels more often in one capture (vLLM's Triton autotune on a cold cache in the first run: 367 extra
  # launches on the R9700, cutting halfway into the timed prefill) moves the cut into work both captures did
  before: dict[str, int] = {}
  for l in long:
    if is_block(l) and l["start_us"] <= bl[len(bs) - 1]:
      before[str(l["key"])] = before.get(str(l["key"]), 0) + 1
  for l in short:
    if is_block(l):
      before[str(l["key"])] = before.get(str(l["key"]), 0) - 1
  if (off := {k: n for k, n in before.items() if n}):
    k, n = max(off.items(), key=lambda kv: abs(kv[1]))
    raise CaptureRefused(f"the two captures' start-ups ran block kernels a different number of times ({len(off)} "
                         f"kernels differ; {k[:80]} by {n:+d} in the long one), so the extra tokens cannot be found by "
                         "count: a JIT or autotune cache filled in one capture and not the other")
  shared_last, first_extra, short_last = bl[len(bs) - 1], bl[len(bs)], bs[-1]
  after = [l for l in long if l["start_us"] >= first_extra]
  region = sorted((l for l in long if shared_last < l["start_us"] < first_extra), key=lambda l: l["start_us"])
  in_short: dict[str, int] = {}
  for l in short:
    if l["start_us"] > short_last:
      in_short[str(l["key"])] = in_short.get(str(l["key"]), 0) + 1
  by_key: dict[str, list[dict[str, Any]]] = {}
  for l in region:
    by_key.setdefault(str(l["key"]), []).append(l)
  extra = [l for k, ls in by_key.items() for l in ls[len(ls) - min(len(ls), max(0, len(ls) - in_short.get(k, 0))):]]
  kept = after + extra
  return kept, sum(l["wall_us"] for l in long) - sum(l["wall_us"] for l in kept)


def shared_window(long:list[dict[str, Any]], short:list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """The decode window from two captures, each first reduced to its share of the GPU busy time."""
  long_s, lo = share_overlap(long)
  short_s, so = share_overlap(short)
  facts = {"rule": OVERLAP_RULE, "long_sum_us": lo["sum_us"], "long_union_us": lo["union_us"],
           "short_sum_us": so["sum_us"], "short_union_us": so["union_us"],
           "overlap_us": lo["sum_us"] - lo["union_us"]}
  if (cut := block_cut(long_s, short_s)) is not None:
    kept, outside = cut
    facts.update(outside_us=outside, window_rule=BLOCK_CUT_RULE)
    return groups(kept), facts
  window = difference(groups(long_s), groups(short_s))
  # the extra tokens run last. A kernel whose extra launches all ran before the step both captures made (the
  # most-launched kernel's last launch before its extra ones) differs only in the engine's start-up (a warm-up sized
  # by the token count, graph capture, a model load whose launch count varies run to run): not decode work, though a
  # per-kernel difference would count it. The edge comes from the kernel with the most extra launches, a decode
  # kernel that runs many times in every token, never from the heaviest: a load kernel can be the heaviest (tinygrad
  # on the M5: 5 extra launches of a ~200 ms load kernel were half the "token")
  heavy = max((g for g in window if g.get("before_us") is not None), key=lambda g: (g["calls"], g["wall_us"]), default=None)
  edge = heavy.get("before_us") if heavy else None
  if edge is not None:
    kept = [g for g in window if g.get("from_us") is None or g["from_us"] > edge]
    facts.update(startup_us=sum(g["wall_us"] for g in window) - sum(g["wall_us"] for g in kept),
                 window_rule="a kernel's extra launches are its latest; those that all ran before the step both "
                             "captures made are the engine's start-up, not decode")
    return kept, facts
  return window, facts


def difference(long:list[dict[str, Any]], short:list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Kernel groups of the long capture minus the short one, per key: the work only the extra tokens did. The extra
  tokens run last, so with launch times a key's extra calls are its latest launches in the long capture, timed as
  they ran: an engine's start-up may run the same kernel many times at start-up clocks, and subtracting the short
  capture's total would carry that start-up's run-to-run noise into the window. Without times, the totals are
  subtracted."""
  base = {g["key"]: g for g in short}
  out = []
  for g in long:
    b = base.get(g["key"], {"calls": 0, "wall_us": 0.0})
    calls = g["calls"] - b["calls"]
    timed = g.get("launches") or []
    if calls > 0 and len(timed) == g["calls"] and None not in (g.get("starts") or [None]):
      every = sorted(range(len(timed)), key=lambda i: g["starts"][i])
      order = every[-calls:]
      latest = [timed[i] for i in order]
      out.append({"key": g["key"], "calls": calls, "wall_us": sum(sh for _, sh in latest), "launches": latest,
                  "short_launches": [], "from_us": g["starts"][order[0]], "starts": [g["starts"][i] for i in order],
                  "streams": [(g.get("streams") or [None] * len(timed))[i] for i in order],
                  "before_us": g["starts"][every[-calls - 1]] if len(every) > calls else None})
      continue
    us = g["wall_us"] - b["wall_us"]
    if calls > 0 and us > 0:
      out.append({"key": g["key"], "calls": calls, "wall_us": us, "launches": timed,
                  "short_launches": b.get("launches") or []})
  return out


class CaptureRefused(RuntimeError):
  """The capture cannot be a measurement. The message says why."""


class UnnamedCapture(CaptureRefused):
  """The capture holds the token, but only in command buffers the engine did not name: no kernel to pair. The
  engine's kernels timed alone (the generic measurement) can still split it (providers.role_time)."""


def check_window(window:list[dict[str, Any]], *, method:str, tokens:int, token_ms:float | None,
                 long_kernels:int, band:float | None) -> str:
  """Refuse a window with no kernels, or with more GPU time per token than the captured run took per token by more
  than the chip's measured uncertainty (screen.bandwidth_uncertainty). Returns what was checked: with no measured
  uncertainty the token comparison is not checked."""
  if not window:
    row = vendors.BY_CAPTURE.get(method)  # the capture's vendor: its graphs hid the kernels (target/vendors.py)
    graphs = f" (for llama.cpp on {row.backend}, its {row.term('graphs')})" if row and row.term("graphs") else ""
    raise CaptureRefused(f"{method} captured {long_kernels} kernels, but none ran only in the decode tokens: the "
                         f"decode's kernels were not visible to it{graphs}")
  per_token = sum(g["wall_us"] for g in window) / tokens / 1000.0
  if not token_ms or band is None:
    return "not checked: " + ("no captured token time" if not token_ms else "the chip has no measured uncertainty")
  if per_token > token_ms * (1 + band):
    raise CaptureRefused(f"{method} counted {per_token:.2f} ms of GPU time per token, more than the {token_ms:.2f} ms "
                         f"the captured run took per token by more than its measured ±{100 * band:.2f}%, so it counted work "
                         "outside the decode tokens")
  return f"checked: kernels within the captured token by the measured ±{100 * band:.2f}%"


def provider_trace(run, window:list[dict[str, Any]], *, provider:str, method:str, model_id:str, target,
                   context:int, tokens:int, out:str, token_ms:float | None = None,
                   long_kernels:int = 0, overlap:dict[str, Any] | None = None, batch:int = 1,
                   matrices:Any = None, engine:dict[str, Any] | None = None) -> dict[str, Any]:
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
  # a decode capture is paired with the decode ceiling's bytes whatever the run's workload (a replay, a session):
  # any other workload reads the prefill block, whose role bytes are not one decode token's
  ceil = _measured_vs_ceiling({**manifest, "workload": "decode"}, profile, pathlib.Path(run))
  window_check = check_window(window, method=method, tokens=tokens, token_ms=token_ms, long_kernels=long_kernels,
                              band=ceil.get("uncertainty"))
  from boltbeam.collectors.vendor_capture import unnamed
  if window and all(unnamed(g.get("key", "")) for g in window):  # nothing to pair: no group is a kernel
    raise UnnamedCapture(f"{method} saw only command buffers {provider} did not name, not its kernels, so it cannot "
                         "split the token by role")
  read = {k: r for r in profile.get("roles", []) for k in ((r["role"], r["quant"]), role_key(r))}  # by shape where read
  counts = {k: int(r["count"]) for k, r in read.items() if r.get("count")}
  roles, pairing_roles, packed = ceil.get("_roles") or [], "the profile's roles", {}
  if matrices:  # an engine that packs checkpoint tensors into one GEMM: pair with its own matrices (prefill's authority)
    from boltbeam.collectors.prefill import engine_roles, prefill_roles
    pr = prefill_roles(profile, roles)
    got, pairing_roles = engine_roles(pr, matrices)
    if got is not pr:
      roles = [{"role": m["role"], "quant": m["quant"], "bytes_moved": m["weight_bytes"] * m["count"], "floor_ms": None, "fine": m["fine"]}
               for m in got]
      counts = {(m["role"], m["quant"]): m["count"] for m in got}
      packed = {(m["role"], m["quant"]): m["part_roles"] for m in got}
  from boltbeam.target.targets import upper
  result = attribute(window, roles, counts, tokens, upper(ceil.get("peak_bandwidth_gbs"), ceil.get("uncertainty")), kinds=pairing.layer_kinds(profile), read=read)
  if packed:  # one measured duration belongs to the packed launch, not separate byte-weighted part timings
    from collections import Counter
    for g in result["groups"]:
      parts = packed.get((g["role"], g["quant"]))
      if not parts:
        continue
      counts = Counter((p["role"], p["quant"], *p["shape"]) for p in parts)
      logical = [{"role": r, "quant": q, "shape": [batch, n, k], "count": c} for (r, q, n, k), c in counts.items()]
      joint = len({role_key(p) for p in logical}) > 1
      g.update(logical_parts=logical, logical_flops=sum(2 * batch * n * k * c for (_, _, n, k), c in counts.items()) * g["calls"],
               role=None if joint else logical[0]["role"], quant=None if joint else logical[0]["quant"])
      if not joint: g["shape"] = logical[0]["shape"]
  wall = sum(g["wall_us"] for g in window)
  trace = {"schema": SCHEMA_TIMING_TRACE, "model_id": model_id, "target_id": target.target_id, "workload": "decode",
           "provider_id": provider, "capture": {"method": method, "reason": None},
           "timing_source": f"{method} kernel timeline, attributed by the layer walk",
           "rows": [{"scope": "whole_step", "context": context, "decode_tokens": tokens, "wall_us": wall,
                     "tok_s": tokens / (wall / 1e6) if wall else None, "launch_count": sum(g["calls"] for g in window),
                     "measurement_scope": "summed_kernel_intervals", "time_source": method, "token_ms": token_ms,
                     "batch": batch}]
                   + trace_rows(result, context=context, time_source=method),
           "pairs": result["pairs"], "unpaired_roles": result["unpaired"], "ambiguous": result["ambiguous"],
           "pairing_rule": result["rule"], "structure": result["structure"], "overlap": overlap, "pairing_roles": pairing_roles,
           "window_check": window_check}
  if packed:
    from boltbeam.artifacts.tinygrad_profile_events import matrix_work_coverage
    work = matrix_work_coverage(trace["rows"], profile, tokens, batch=batch)
    unresolved = [{"kernel": r["kernel"], "calls": r["calls"]} for r in trace["rows"] if r.get("scope") == "kernel"
                  and not r.get("logical_parts") and any(w in r["kernel"].lower() for w in ("gemm", "gemv", "matmul"))]
    trace["aux_sources"] = {"attribution": {"complete": work["complete"] and not result["unpaired"] and not unresolved,
                                            "matrix_work": work, "unresolved_launches": unresolved,
                                            "source": "the engine's loaded matrix inventory"}}
  if engine:
    for row in trace.get('rows', []):
      if row.get('scope') == 'whole_step':
        row['engine'] = {**row.get('engine', {}), **engine}
  (pathlib.Path(run) / out).write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
  return trace
