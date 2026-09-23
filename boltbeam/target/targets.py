from __future__ import annotations

import json
import pathlib

from boltbeam.vocab import BackendStatus, SCHEMA_TARGET_REGISTRY
from boltbeam.profile.ir import TargetProfile

_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "targets.json"


def _from_row(d:dict) -> TargetProfile:
  return TargetProfile(target_id=d["target_id"], backend=d["backend"], wave_size=int(d["wave_size"]),
                       lds_bytes_per_cu=d.get("lds_bytes_per_cu"), vram_bytes=d.get("vram_bytes"),
                       subgroup_size=d.get("subgroup_size"), vector_load_bits=d.get("vector_load_bits"),
                       dot_primitives=tuple(d.get("dot_primitives", ())),
                       dot_primitives_lowered=tuple(d.get("dot_primitives_lowered", ())),
                       dequant_primitives=tuple(d.get("dequant_primitives", ())),
                       compute_units=d.get("compute_units"),
                       memory_bandwidth_gbs=d.get("memory_bandwidth_gbs"),
                       peak_tflops=dict(d.get("peak_tflops", {})),
                       backend_status=d.get("backend_status", "complete"),
                       capabilities=dict(d.get("capabilities", {})))


def load_target_registry(path:str | pathlib.Path | None = None) -> dict[str, TargetProfile]:
  p = pathlib.Path(path) if path else _DEFAULT_PATH
  data = json.loads(p.read_text())
  if data.get("schema") != SCHEMA_TARGET_REGISTRY:
    raise ValueError(f"{p} is not a {SCHEMA_TARGET_REGISTRY} registry")
  return {row["target_id"]: _from_row(row) for row in data["targets"]}


TARGETS = load_target_registry()

# Vendor prefixes are read off the registry rather than listed, so a new vendor row needs no edit here.
_VENDORS = frozenset(t.split("_", 1)[0] for t in TARGETS if "_" in t)


def get_target(name:str) -> TargetProfile:
  try: return TARGETS[name]
  except KeyError as exc:
    known = ", ".join(sorted(TARGETS))
    raise SystemExit(f"unknown target {name!r}. Known targets: {known}") from exc


def target_backend_status(name:str) -> str:
  """Backend build status for a target id: complete / descriptor_only / unsupported (unknown -> unsupported)."""
  t = TARGETS.get(name)
  return t.backend_status if t else BackendStatus.UNSUPPORTED.value


def target_capability(name:str, key:str, default=None):
  """One target capability value. Unknown targets/capabilities return `default`."""
  t = TARGETS.get(name)
  caps = t.capabilities if t else None
  return (caps or {}).get(key, default)


def is_exact_target(name:str | None) -> bool:
  """Whether a registry row is an observed-hardware target rather than a family descriptor."""
  return target_kind(name) == "exact"


def target_aliases(name:str | None) -> frozenset[str]:
  """Every spelling that denotes one registry target.

  Producers write a target either as the registry id (``amd_gfx1100``) or as the bare
  architecture the vendor toolchain reports (``gfx1100``, ``sm120``). Both name the same
  hardware, so the registry — which owns target identity — owns the equivalence rather
  than each consumer re-deriving it from a hand-typed pair.
  """
  if not name: return frozenset()
  spellings = {name}
  vendor, sep, arch = name.partition("_")
  if sep and vendor in _VENDORS: spellings.add(arch)
  else: spellings |= {f"{v}_{name}" for v in _VENDORS if f"{v}_{name}" in TARGETS}
  return frozenset(spellings)


def target_matches(declared:str | None, target_id:str | None) -> bool:
  """Whether a producer's declared target denotes `target_id`, allowing registry aliases."""
  if not declared or not target_id: return False
  return declared in target_aliases(target_id) or target_id in target_aliases(declared)


def target_kind(name:str | None) -> str | None:
  """Registry-owned target granularity (for example ``exact`` or ``family``)."""
  if not name or name not in TARGETS: return None
  # Older concrete target rows predate the explicit field; retain their established behavior.
  return target_capability(name, "target_kind", "exact")


# The default-target bandwidth fact, derived from the registry so there is exactly one source
# (LN-130). amd_gfx1100 is the historical default target every peak_gbs parameter assumed.
DEFAULT_PEAK_MEM_GBS: float = float(next(
  row["memory_bandwidth_gbs"] for row in json.loads(_DEFAULT_PATH.read_text())["targets"]
  if row["target_id"] == "amd_gfx1100"))
