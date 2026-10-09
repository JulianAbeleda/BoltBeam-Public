from __future__ import annotations

import json
import os
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
                       matrix_tflops=dict(d.get("matrix_tflops", {})),
                       backend_status=d.get("backend_status", "complete"),
                       capabilities=dict(d.get("capabilities", {})))


def load_target_registry(path:str | pathlib.Path | None = None) -> dict[str, TargetProfile]:
  p = pathlib.Path(path) if path else _DEFAULT_PATH
  data = json.loads(p.read_text())
  if data.get("schema") != SCHEMA_TARGET_REGISTRY:
    raise ValueError(f"{p} is not a {SCHEMA_TARGET_REGISTRY} registry")
  return {row["target_id"]: _from_row(row) for row in data["targets"]}


# --- chip profiles made on this machine -------------------------------------------------------------------------
# A chip no registry row claims gets a profile measured on the machine itself (workflow/chips.py). It is a registry
# row in the same shape, saved per machine, never in the repo: the registry is reviewed and shared, a local profile is
# one machine's measurement. One shape means get_target, the ceiling and the tie-out read both the same way.

SCHEMA_CHIP_PROFILE = "boltbeam.chip_profile.v1"
CHIPS_DIR_ENV = "BOLTBEAM_CHIPS_DIR"
IGNORE_REGISTRY_ENV = "BOLTBEAM_IGNORE_REGISTRY_TARGET"


def chips_dir() -> pathlib.Path:
  """Where this machine keeps its chip profiles: $BOLTBEAM_CHIPS_DIR, else ~/.boltbeam/chips."""
  return pathlib.Path(os.environ.get(CHIPS_DIR_ENV) or pathlib.Path.home() / ".boltbeam" / "chips").expanduser()


def ignore_registry() -> bool:
  """BOLTBEAM_IGNORE_REGISTRY_TARGET=1: the built-in rows never claim this machine, so it gets its own profile.
  The test switch for the new-chip path; the rows still exist for what-if limits."""
  return os.environ.get(IGNORE_REGISTRY_ENV, "") not in ("", "0")


def local_profiles(folder:pathlib.Path | None = None) -> dict[str, dict]:
  """The chip profiles saved on this machine, by id. A file that is not a profile is skipped, never guessed at."""
  out = {}
  d = folder or chips_dir()
  for p in sorted(d.glob("*.json")) if d.is_dir() else ():
    try:
      doc = json.loads(p.read_text())
    except (OSError, ValueError):
      continue
    row = doc.get("target") if doc.get("schema") == SCHEMA_CHIP_PROFILE else None
    if isinstance(row, dict) and row.get("target_id") == p.stem:
      out[p.stem] = row
  return out


def _load_all() -> dict[str, TargetProfile]:
  rows = load_target_registry()
  for tid, row in local_profiles().items():
    rows.setdefault(tid, _from_row(row))  # a registry row with the same id wins: reviewed beats local
  return rows


def is_local(name:str | None) -> bool:
  """Whether a target is a chip profile made on this machine rather than a registry row."""
  return target_capability(name, "profile_source") == "generated" if name else False


def save_local_profile(row:dict, *, measured_at:str) -> pathlib.Path:
  """Write one chip profile to this machine's store and make it a target now."""
  d = chips_dir()
  d.mkdir(parents=True, exist_ok=True)
  path = d / f"{row['target_id']}.json"
  path.write_text(json.dumps({"schema": SCHEMA_CHIP_PROFILE, "measured_at": measured_at, "target": row},
                             indent=1, sort_keys=True) + "\n")
  TARGETS[row["target_id"]] = _from_row(row)
  return path


TARGETS = _load_all()

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


def match_target(observed:dict) -> str | None:
  """The registry row whose declared facts the scanned device satisfies, or None.

  A row states what it is in ``capabilities.match`` (for example ``{"apple_soc": "M3", "gpu_cores": 10}``).
  A scanner reports what the machine said and asks here; it never carries its own table of chips, so a new
  row is a data edit and needs no scanner change. Only ``exact`` rows match: a family descriptor names a
  backend, not a machine. Two rows claiming the same device is a registry fault, so it matches nothing
  rather than picking a winner.

  Registry rows are asked first; a profile made on this machine only when no registry row claims the device
  (BOLTBEAM_IGNORE_REGISTRY_TARGET=1 skips the registry rows).
  """
  def hits(local:bool) -> list[str]:
    return [name for name, t in TARGETS.items() if is_exact_target(name) and is_local(name) == local
            and (rule := t.capabilities.get("match")) and all(observed.get(k) == v for k, v in rule.items())]
  tiers = (True,) if ignore_registry() else (False, True)
  for local in tiers:
    found = hits(local)
    if found:
      return found[0] if len(found) == 1 else None
  return None


def family_target(backend:str) -> str | None:
  """The backend's family descriptor row, for a device no exact row claims."""
  hits = [name for name, t in TARGETS.items()
          if target_kind(name) == "family" and t.backend.lower() == backend.lower()]
  return hits[0] if len(hits) == 1 else None


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
