"""Reproducible static occupancy bounds from exact compiler resources."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from boltbeam.perf.calibration import CalibrationProfile


@dataclass(frozen=True)
class OccupancyBound:
  waves_per_workgroup: int | None
  resident_workgroups_per_cu: int | None
  resident_waves_per_cu: int | None
  limiting_resource: str | None
  limits: Mapping[str, int]
  blockers: tuple[str, ...] = ()

  @property
  def complete(self) -> bool: return not self.blockers


def derive_occupancy(resources: Mapping[str, Any], calibration: CalibrationProfile) -> OccupancyBound:
  required_resources = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "workgroup_threads")
  missing = [name for name in required_resources if not isinstance(resources.get(name), int)]
  if isinstance(resources.get("scratch_bytes"), int) and resources["scratch_bytes"] != 0:
    missing.append("zero scratch_bytes")
  params = {name: _integer(calibration, "gpu." + name) for name in (
    "wave_size", "max_waves_per_cu", "max_workgroups_per_cu", "vgpr_per_cu", "sgpr_per_cu", "lds_bytes_per_cu",
    "vgpr_alloc_granule", "sgpr_alloc_granule", "lds_alloc_granule")}
  missing += [name for name, value in params.items() if value is None]
  if missing: return OccupancyBound(None, None, None, None, {}, tuple(f"missing {x}" for x in missing))
  waves = math.ceil(resources["workgroup_threads"] / params["wave_size"])
  vgpr_wave = _round(resources["vgpr"], params["vgpr_alloc_granule"])
  sgpr_wave = _round(resources["sgpr"], params["sgpr_alloc_granule"])
  lds_wg = _round(resources["lds_bytes"], params["lds_alloc_granule"])
  limits = {
    "architectural": params["max_workgroups_per_cu"],
    "waves": params["max_waves_per_cu"] // waves,
    "vgpr": params["vgpr_per_cu"] // max(1, vgpr_wave * waves),
    "sgpr": params["sgpr_per_cu"] // max(1, sgpr_wave * waves),
    "lds": params["max_workgroups_per_cu"] if lds_wg == 0 else params["lds_bytes_per_cu"] // lds_wg,
  }
  resident = max(0, min(limits.values()))
  limiter = min(limits, key=limits.get)
  blockers = ("resources permit zero resident workgroups",) if resident == 0 else ()
  return OccupancyBound(waves, resident, resident * waves, limiter, limits, blockers)


def _integer(profile: CalibrationProfile, name: str) -> int | None:
  interval = profile.interval(name)
  return int(interval.median) if interval is not None and interval.median > 0 else None


def _round(value: int, granule: int) -> int: return math.ceil(value / granule) * granule


__all__ = ["OccupancyBound", "derive_occupancy"]
