"""What a matrix-path probe measured (cuda_compute, hip_compute: the vendor row's compute module), as a chip
profile keeps it, for any vendor: each probe row names its path, its instruction (PTX or a HIP builtin), its rate
and the clock it ran at."""
from __future__ import annotations

from typing import Any


def matrix_tflops(result:dict[str, Any]) -> dict[str, float]:
  """The chip profile's matrix_tflops from a probe result: the plateau rate of every available path."""
  return {r["path"]: r["tops"] for r in result["paths"] if r.get("available")}


def fact_source(result:dict[str, Any], *, scope:str, today:str) -> dict[str, Any]:
  """The chip profile's fact source for matrix_tflops: per path its rate, the clock it was measured at, its
  instruction and the rate per multiprocessor per clock; the policy and whether anything throttled (unknown where the
  driver reports no throttle reasons)."""
  return {"kind": "measurement", "bound": "achievable", "scope": scope, "observed_at": today, "method": result["method"],
          "clock_policy": result["clock_policy"].get("policy"), "throttled": result["telemetry"].get("throttled"),
          "paths": {r["path"]: ({"tops": r["tops"], "sm_mhz": r["sm_mhz"], "ops_per_sm_per_clock": r["ops_per_sm_per_clock"],
                                 "instruction": r["instruction"], "words": r["words"], "throttled": r["clock"].get("throttled"),
                                 **{k: r[k] for k in ("uncertainty", "shape_spread") if k in r},
                                 "power_w_mean": (r["clock"].get("power_w") or {}).get("mean")}
                                if r.get("available") else {"available": False, "reason": r["reason"], "instruction": r["instruction"]})
                    for r in result["paths"]}}
