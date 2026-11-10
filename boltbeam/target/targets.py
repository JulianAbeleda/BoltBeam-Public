from __future__ import annotations

import json
import math
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
# Every GPU gets a profile measured on the machine itself (workflow/chips.py), and that profile comes before any
# registry row that claims the same GPU (match_target): a chip's figures are derived where they are used. It is a
# registry row in the same shape, saved per machine, never in the repo. One shape means get_target, the ceiling and the tie-out read both the same way.

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


# A chip profile another machine made, brought home with a run by `boltbeam fetch` (workflow/fetch.py), lives apart
# from this machine's own: <chips_dir>/fetched/<host>/<id>.json, marked profile_source "fetched". It is read only for
# what that machine's runs need (their limit and results): a registry row or a profile made here with the same id wins,
# it never claims this machine's GPU (match_target), and it is never written over this machine's own profile.
FETCHED = "fetched"


def fetched_profile_path(host:str, target_id:str) -> pathlib.Path:
  if not host or host != pathlib.Path(host).name or host.startswith(".") or target_id != pathlib.Path(target_id).name:
    raise ValueError(f"not a plain host and chip id: {host!r}, {target_id!r}")
  return chips_dir() / FETCHED / host / f"{target_id}.json"


def _fetched_row(row:dict) -> TargetProfile:
  return _from_row({**row, "capabilities": {**(row.get("capabilities") or {}), "profile_source": FETCHED}})


def _load_all() -> dict[str, TargetProfile]:
  rows = load_target_registry()
  for tid, row in local_profiles().items():
    rows.setdefault(tid, _from_row(row))  # ids never collide in practice: a local id is the hardware's (chip_profile_id)
  hosts = chips_dir() / FETCHED
  for host in sorted(hosts.iterdir()) if hosts.is_dir() else ():
    for tid, row in local_profiles(host).items():
      rows.setdefault(tid, _fetched_row(row))  # reviewed, then made here, then fetched (first host by name)
  return rows


# A run keeps the chip profile it was measured against (RUN_CHIP), byte for byte, so the run opens on any machine:
# reading a run is reading its data, and a Mac that never saw an H100 still has the H100's measured row in the run.
RUN_CHIP = "chip_profile.json"


def keep_run_chip(run:pathlib.Path, target_id:str) -> None:
  """Copy this machine's (or a fetched) profile of target_id into the run; a registry row needs none."""
  hosts = chips_dir() / FETCHED
  for src in [chips_dir() / f"{target_id}.json", *(sorted(hosts.glob(f"*/{target_id}.json")) if hosts.is_dir() else ())]:
    if src.is_file():
      (pathlib.Path(run) / RUN_CHIP).write_bytes(src.read_bytes())
      return


def adopt_run_chip(run:pathlib.Path) -> None:
  """A run's own chip profile becomes a target when this machine knows none by its id (read only, never saved)."""
  try:
    doc = json.loads((pathlib.Path(run) / RUN_CHIP).read_text())
  except (OSError, ValueError):
    return
  row = doc.get("target") if isinstance(doc, dict) and doc.get("schema") == SCHEMA_CHIP_PROFILE else None
  if isinstance(row, dict) and row.get("target_id"):
    TARGETS.setdefault(row["target_id"], _fetched_row(row))


def is_fetched(name:str | None) -> bool:
  """Whether a target is a chip profile another machine made, brought home by `boltbeam fetch`."""
  return target_capability(name, "profile_source") == FETCHED if name else False


def save_fetched_profile(raw:bytes, host:str) -> pathlib.Path:
  """Keep one fetched chip profile, byte for byte as the other machine wrote it (its sha256 stays checkable), and make
  it a target now, unless an id this machine already knows has it. An existing file is never replaced."""
  doc = json.loads(raw)
  row = doc.get("target") if isinstance(doc, dict) and doc.get("schema") == SCHEMA_CHIP_PROFILE else None
  if not isinstance(row, dict) or not row.get("target_id"):
    raise ValueError("not a chip profile")
  path = fetched_profile_path(host, row["target_id"])
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
  TARGETS.setdefault(row["target_id"], _fetched_row(row))
  return path


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


def stored_path_scan(arch:str | None) -> dict | None:
  """The matrix-path scan a chip profile keeps for one compiler target (profile_row's matrix_path_scan), or None."""
  return next((rec for t in TARGETS.values() if arch and (rec := (t.capabilities or {}).get("matrix_path_scan"))
               and rec.get("arch") == arch and any(p.get("kind") == "mma" for p in rec.get("paths") or ())), None)

# Vendor prefixes are read off the registry rather than listed, so a new vendor row needs no edit here.
_VENDORS = frozenset(t.split("_", 1)[0] for t in TARGETS if "_" in t)


def get_target(name:str) -> TargetProfile:
  try: return TARGETS[name]
  except KeyError as exc:
    known = ", ".join(sorted(TARGETS))
    raise SystemExit(f"unknown target {name!r}. Known targets: {known}") from exc


def uncertainty(src:dict | None) -> float | None:
  """A measured figure's own relative uncertainty, kept beside it (a fact source, a path of one, or a probe's read):
  the spread of the probe's repeated runs at its best launch shape. None when it has none, so a guard says not checked
  and never borrows a constant. A shape spread (how much the launch shape matters) is not an uncertainty and is never
  read here."""
  u = (src or {}).get("uncertainty")
  return float(u) if isinstance(u, (int, float)) and math.isfinite(u) and u >= 0 else None


def fact_uncertainty(target, fact:str = "memory_bandwidth_gbs", path:str | None = None) -> float | None:
  """uncertainty() of one registry figure: the bandwidth, or one matrix path's peak."""
  src = ((getattr(target, "capabilities", None) or {}).get("fact_sources") or {}).get(fact) or {}
  return uncertainty((src.get("paths") or {}).get(path) if path else src)


def upper(value:float | None, u:float | None) -> float | None:
  """The most a measured figure can be within its uncertainty; None (not checked) when either is unknown."""
  return value * (1 + u) if value and u is not None else None


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

  A profile measured on this machine is asked first: a chip's figures are derived on the GPU in front of it, and a
  registry row is what a GPU uses only until it has been measured (chips.autoscan measures it) or when its
  measurement fails (BOLTBEAM_IGNORE_REGISTRY_TARGET=1 skips the registry rows altogether).
  """
  def hits(local:bool) -> list[str]:
    return [name for name, t in TARGETS.items() if is_exact_target(name) and is_local(name) == local and not is_fetched(name)
            and (rule := t.capabilities.get("match")) and all(observed.get(k) == v for k, v in rule.items())]
  tiers = (True,) if ignore_registry() else (True, False)
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


def target_peak_gbs(target_id:str | None) -> float | None:
  """A registry target's memory bandwidth, the one source of a peak_gbs (LN-130); None when no target is named or
  it records none. No chip's figure stands in for another's: every peak_gbs comes from the run's target."""
  t = TARGETS.get(str(target_id)) if target_id else None
  return float(t.memory_bandwidth_gbs) if t and t.memory_bandwidth_gbs else None


def need_target_id(target_id:str | None, what:str) -> str:
  """target_id, or ValueError: a trace names the chip it ran on, never a default one."""
  if not target_id:
    raise ValueError(f"{what}: --target-id is required unless --run supplies one; no chip is assumed")
  return target_id


def need_peak_gbs(peak_gbs:float | None, what:str) -> float:
  """peak_gbs, or ValueError naming what needed it: a bandwidth limit with no bandwidth is not computed."""
  if peak_gbs is None:
    raise ValueError(f"{what} needs the target's peak bandwidth (peak_gbs): none was given, and no chip's figure "
                     "is assumed in its place")
  return float(peak_gbs)
