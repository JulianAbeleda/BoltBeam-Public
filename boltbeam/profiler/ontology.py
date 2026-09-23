"""Provider-neutral profiler concepts and typed observation outcomes."""
from __future__ import annotations

from typing import Any

CONCEPTS = ("dispatch", "program", "execution_unit", "matrix_unit", "vector_unit", "memory_level",
            "occupancy", "bytes_moved", "work_done", "baseline_efficiency", "missing_evidence")
MEMORY_TIERS = ("register", "scratch", "lds", "l0", "l1", "l2", "last_level_cache", "dram", "host", "unknown")
OBSERVATION_SCOPES = ("global", "per_kernel", "per_operand", "controlled_proxy")
MEASURED, DERIVED, MODELED = "measured", "derived", "modeled"
PROXY, MISSING, UNSUPPORTED, INVALID = "proxy", "missing", "unsupported", "invalid"


def metric(value: Any, unit: str, source: str, quality: str = MEASURED, **metadata: Any) -> dict[str, Any]:
  if value is None:
    raise ValueError("present metrics require a value; use missing() or unsupported()")
  return {"value": value, "unit": unit, "source": source, "quality": quality, **metadata}


def missing(reason: str) -> dict[str, Any]:
  return {"value": None, "status": MISSING, "reason": reason}


def unsupported(reason: str, *, counter: str | None = None, tier: str | None = None) -> dict[str, Any]:
  out = {"value": None, "status": UNSUPPORTED, "reason": reason}
  if counter is not None: out["counter"] = counter
  if tier is not None: out["tier"] = tier
  return out


def tier_observation(tier: str, *, scope: str, source: str, status: str = MEASURED,
                     bytes: float | None = None, requests: float | None = None,
                     hits: float | None = None, misses: float | None = None,
                     hit_rate: float | None = None, uncertainty: dict | None = None,
                     reason: str | None = None, **metadata: Any) -> dict[str, Any]:
  """Create an orthogonal serving-tier fact; unsupported and absent are never zero."""
  if tier not in MEMORY_TIERS: raise ValueError(f"unknown memory tier {tier!r}")
  if scope not in OBSERVATION_SCOPES: raise ValueError(f"unknown observation scope {scope!r}")
  if status == UNSUPPORTED:
    if not reason: raise ValueError("unsupported observations require a reason")
    return {"tier": tier, "scope": scope, "source": source, "status": status, "reason": reason,
            "bytes": None, "requests": None, "hits": None, "misses": None, "hit_rate": None, **metadata}
  if status not in {MEASURED, DERIVED, MODELED, PROXY, MISSING}:
    raise ValueError(f"invalid observation status {status!r}")
  if status == MEASURED and scope == "controlled_proxy":
    raise ValueError("controlled proxies cannot be labeled measured")
  if hit_rate is not None and not 0 <= hit_rate <= 1: raise ValueError("hit_rate must be in [0, 1]")
  return {"tier": tier, "scope": scope, "source": source, "status": status, "bytes": bytes,
          "requests": requests, "hits": hits, "misses": misses, "hit_rate": hit_rate,
          "uncertainty": uncertainty or {}, **metadata}
