"""Static depth-slope analysis for normalized tinygrad decode authority traces."""
from __future__ import annotations

from typing import Any, Iterable


def decode_depth_slope(traces: Iterable[dict[str, Any]]) -> dict[str, Any]:
  """Report matched-context W/D slopes without treating unavailable counters as measurements."""
  grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
  for trace in traces:
    model = trace["model_id"]
    for row in trace["rows"]:
      grouped.setdefault(model, {}).setdefault(row["measurement"], {})[int(row["context"])] = row
  models = []
  for model_id in sorted(grouped):
    measurements = []
    for measurement in ("W", "D"):
      points = grouped[model_id].get(measurement, {})
      matched = [ctx for ctx in (512, 4096) if ctx in points]
      result: dict[str, Any] = {"measurement": measurement, "matched_contexts": matched}
      if len(matched) == 2:
        low, high = (points[ctx] for ctx in matched)
        slope = (high["wall_us"] - low["wall_us"]) / (high["context"] - low["context"])
        result.update({"status": "proven_for_measured_pair", "wall_us_per_context_token": slope,
                       "tok_s_low": low["tok_s"], "tok_s_high": high["tok_s"],
                       "route_identity_matched": low["route_sequence"] == high["route_sequence"],
                       "programs_per_token_matched": low["programs_per_token"] == high["programs_per_token"]})
      else:
        result.update({"status": "unavailable", "reason": "requires matched ctx512 and ctx4096"})
      measurements.append(result)
    models.append({"model_id": model_id, "measurements": measurements,
                   "ctx128_present": 128 in grouped[model_id].get("W", {}),
                   "generation_control": {
                     "status": "unavailable",
                     "reason": "authority JSON has no explicit G-control, route variant, build setting, or program identity",
                   },
                   "counter_conclusion": "unavailable: authority artifacts contain no hardware counter stream"})
  return {"analysis": "tinygrad_decode_depth_slope.v1", "pair": [512, 4096], "models": models,
          "limitations": ["W is production generate-item/token wall time.",
                          "D is diagnostic same-model JIT plus final synchronization, not an upper bound.",
                          "One repetition per supplied artifact; no uncertainty interval is proven."]}
