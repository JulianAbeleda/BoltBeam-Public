"""Provider-neutral, hardware-free memory hierarchy sweep requests."""
from __future__ import annotations

from typing import Iterable

from boltbeam.target.targets import TARGETS
from boltbeam.vocab import SCHEMA_WS_SWEEP_REQUEST

_MIN_BYTES = 4096
_FALLBACK_MAX_BYTES = 268435456
_MODES = {"bandwidth", "pointer_chase"}
_TRAFFIC = {"read", "write", "mixed"}


def _default_max_bytes(target_id: str) -> int:
  target = TARGETS.get(target_id)
  tiers = ((target.capabilities if target else {}).get("memory_hierarchy") or {}).get("tiers", {})
  llc = tiers.get("last_level_cache") or tiers.get("mall") or {}
  capacity = llc.get("bytes")
  return max(_FALLBACK_MAX_BYTES, 2 * int(capacity)) if capacity else _FALLBACK_MAX_BYTES


def _default_sizes(max_bytes: int) -> list[int]:
  sizes, n = [], float(_MIN_BYTES)
  while n <= max_bytes:
    sizes.append(int(round(n)))
    n *= 2 ** 0.5
  if not sizes or sizes[-1] < max_bytes:
    sizes.append(max_bytes)
  return sizes


def _sizes(values: Iterable[int | float] | None, target_id: str) -> list[int]:
  points = _default_sizes(_default_max_bytes(target_id)) if values is None else [int(round(v)) for v in values]
  if not points or any(v < _MIN_BYTES for v in points):
    raise ValueError("sweep sizes must be non-empty and at least 4096 bytes")
  if points != sorted(set(points)):
    raise ValueError("sweep sizes must be unique and strictly increasing")
  return points


def build_ws_sweep_request(target_id: str, sizes: list[int] | None = None, *,
                           modes: tuple[str, ...] = ("bandwidth",),
                           temperatures: tuple[str, ...] = ("warm", "cold"),
                           traffic: tuple[str, ...] = ("read",), repeats: int = 5,
                           isolated: bool = True) -> dict:
  """Build a calibrated sweep request without dispatching hardware.

  Legacy fields (``kind=copy``, ``sustained``, and ``cold``) remain so existing
  runners continue to consume the default request.
  """
  if not modes or any(v not in _MODES for v in modes):
    raise ValueError(f"modes must be drawn from {sorted(_MODES)}")
  if not temperatures or any(v not in {"warm", "cold"} for v in temperatures):
    raise ValueError("temperatures must contain warm and/or cold")
  if not traffic or any(v not in _TRAFFIC for v in traffic):
    raise ValueError(f"traffic must be drawn from {sorted(_TRAFFIC)}")
  if repeats < 2:
    raise ValueError("at least two repetitions are required to establish a band")
  points_bytes = _sizes(sizes, target_id)
  points = []
  for n in points_bytes:
    for mode in modes:
      for pattern in traffic:
        points.append({"bytes": n, "kind": "copy" if mode == "bandwidth" else "pointer_chase",
                       "mode": mode, "traffic": pattern, "temperatures": list(temperatures),
                       "repeats": repeats, "sustained": "warm" in temperatures, "cold": "cold" in temperatures})
  return {
    "schema": SCHEMA_WS_SWEEP_REQUEST, "target_id": target_id,
    "purpose": "hierarchy_calibration", "candidate_observation": False,
    "note": "capacity/effective-bandwidth calibration only; never a candidate hit-rate measurement",
    "execution": {"isolated": isolated, "health_preflight": True, "health_postflight": True,
                  "record_system_clocks_compiler": True},
    "points": points,
    "lds_probe": {"kind": "lds_resident", "mode": "bandwidth", "traffic": "mixed",
                  "bytes": 32768, "reuse": 256, "repeats": repeats, "temperatures": ["warm"]},
  }
