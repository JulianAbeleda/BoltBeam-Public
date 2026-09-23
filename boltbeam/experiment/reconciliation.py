"""Reconcile HCQGraph dispatch records without confusing device and host time."""
from __future__ import annotations
from collections import defaultdict
from typing import Any

def busy_time_union(intervals:list[tuple[float, float]]) -> float:
  """Sum the length of the union of [start, end] intervals, without double-counting overlaps.

  Intervals with end <= start contribute nothing. Input order does not matter."""
  if not intervals: return 0.0
  spans = sorted((float(s), float(e)) for s, e in intervals if e > s)
  if not spans: return 0.0
  total = 0.0
  cur_start, cur_end = spans[0]
  for s, e in spans[1:]:
    if s > cur_end:
      total += cur_end - cur_start
      cur_start, cur_end = s, e
    else:
      cur_end = max(cur_end, e)
  total += cur_end - cur_start
  return total

def reconcile_dispatches(records:list[dict[str, Any]], *, host_wall_us:float|None=None) -> dict[str, Any]:
  categories:dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "device_us": 0.0})
  unknown:list[dict[str, Any]] = []
  device_total = 0.0
  for i, rec in enumerate(records):
    label = rec.get("category") or rec.get("semantic_label") or rec.get("role")
    dur = rec.get("device_us", rec.get("duration_us"))
    try: dur = float(dur)
    except (TypeError, ValueError): dur = None
    if not label or dur is None or dur < 0:
      unknown.append({"index": i, "reason": "missing_category_or_device_duration", "record": rec})
      continue
    categories[str(label)]["count"] += 1
    categories[str(label)]["device_us"] += dur
    device_total += dur
  out = {
    "schema": "boltbeam.dispatch_reconciliation.v1",
    "categories": dict(sorted(categories.items())),
    "unknown": unknown,
    "device_timeline_us": device_total,
    "host_wall_us": host_wall_us,
    "time_domains": {"device_timeline": "sum of dispatch durations; overlapping dispatches may double-count",
                      "host_wall": "end-to-end wall interval; includes overlap and host overhead",
                      "comparable_by_sum": False},
  }
  if host_wall_us is not None and host_wall_us > 0:
    out["device_to_host_ratio"] = device_total / float(host_wall_us)
  return out
