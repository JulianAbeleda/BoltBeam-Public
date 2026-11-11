"""The speed limit: the chip's facts, the roofline for one model on one chip, and the one memory speed every number
of a run uses (run_bandwidth). The ceiling is `model_roofline` exactly as `roofline-theoretical` calls it, with
tokens/s read off the floor (one token at context 1 for decode; the context's tokens for prefill). Below workflow
and collectors: the collectors that write a trace read the run's bandwidth and ceiling here."""
from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.artifacts.run_io import DECODE_CAPTURE, MACHINE, MEASURE_STATUS, optional
from boltbeam.core.refused import Refused
from boltbeam.kernel_analysis.theoretical_roofline import model_roofline
from boltbeam.profile import weight_ledger
from boltbeam.target import targets as reg
from boltbeam.target import vendors
from boltbeam.target.targets import fact_uncertainty, get_target, uncertainty
from boltbeam.vocab import SCHEMA_TUI as SCHEMA


def peak_flops_fact(target, dtype:str) -> tuple[float | None, str | None]:
  """The registry's compute figure in TFLOP/s and the fact_status key that says where it came from.

  The one place the precedence lives: the matrix unit's measured rate for the dtype, then the ALU rate for the dtype,
  then fp16, then the largest. A screen reads the key to say where the ceiling's compute came from.
  """
  measured = target.matrix_tflops_for(dtype)
  if measured:
    return measured, "matrix_tflops"
  peaks = target.peak_tflops or {}
  for key in (dtype, "fp16"):
    if peaks.get(key):
      return float(peaks[key]), key
  if peaks:
    key = max(peaks, key=lambda k: peaks[k])
    return float(peaks[key]), key
  return None, None


def peak_flops_status(target, dtype:str) -> str:
  """fact_status of the compute figure peak_flops_fact picks: the row's own word for it, or unknown."""
  _, key = peak_flops_fact(target, dtype)
  status = ((target.capabilities or {}).get("fact_status") or {})
  if key is None: return "unknown"
  if key == "matrix_tflops": return status.get("matrix_tflops", "unknown")
  # rows name an ALU rate either by dtype (fp16_tflops) or as one peak_tflops entry
  return status.get(f"{key}_tflops") or status.get("peak_tflops") or "unknown"


def resolve_peak_flops(target, dtype:str, override_tflops:float | None) -> float:
  """Compute ceiling in FLOP/s, or a stated refusal.

  A target whose registry row carries no `peak_tflops` has no compute ceiling to place a
  model against. Descriptor-only rows are exactly that case, and the registry already says
  so via `backend_status`. Reporting the absence beats both crashing on an empty `max()`
  and inventing a vendor figure — "a capability a backend lacks is stated explicitly,
  never faked".

  Precedence: what the caller measured, then what the row measured on its matrix unit, then the
  row's ALU rates. A measured figure outranks a sheet figure for the same silicon.
  """
  if override_tflops is not None:
    if override_tflops <= 0: raise SystemExit("--peak-tflops must be positive")
    return override_tflops * 1e12
  value, _ = peak_flops_fact(target, dtype)
  if value:
    return value * 1e12
  raise SystemExit(
    f"target {target.target_id!r} carries no peak_tflops, so there is no compute ceiling to "
    f"place this model against (backend_status={target.backend_status!r}).\n"
    f"On the machine with this GPU, `python -m boltbeam.workflow.screen autoscan` measures its bandwidth and compute "
    f"peak once and saves a chip profile. Or supply a measured ceiling with --peak-tflops <TFLOP/s>, or add one to "
    f"boltbeam/data/targets.json once it has been measured on the hardware.")


# --- targets and the ceiling ------------------------------------------------------------------------------

def target_facts(target) -> dict[str, Any]:
  caps = target.capabilities or {}
  status = caps.get("fact_status", {}) or {}
  sources = caps.get("fact_sources", {}) or {}
  return {
    "id": target.target_id,
    # registry: the repo's reviewed rows; generated: a profile measured on this machine (workflow/chips.py);
    # fetched: another machine's profile brought home with its runs (workflow/fetch.py)
    "source": "generated" if reg.is_local(target.target_id) else "fetched" if reg.is_fetched(target.target_id) else "registry",
    "backend": target.backend,
    "backend_status": target.backend_status,
    "scope": (sources.get("memory_bandwidth_gbs") or {}).get("scope"),
    "memory_bandwidth_gbs": target.memory_bandwidth_gbs,
    "peak_tflops": dict(target.peak_tflops or {}),
    "matrix_tflops": dict(target.matrix_tflops or {}),
    # peak_tflops: the status of the compute figure the ceiling uses (peak_flops_fact), so the chip
    # and the speed limit never disagree about where compute came from
    "fact_status": {"memory_bandwidth_gbs": status.get("memory_bandwidth_gbs", "unknown"),
                    "peak_tflops": peak_flops_status(target, "fp16")},
    # the scan and the date behind the memory figure; a recorded fact, not this machine read live
    "scope_observed_at": (sources.get("memory_bandwidth_gbs") or {}).get("observed_at"),
    "has_ceiling": bool(target.memory_bandwidth_gbs) and bool(target.matrix_tflops or target.peak_tflops),
  }


def targets() -> dict[str, Any]:
  return {"schema": SCHEMA, "kind": "targets", "targets": [target_facts(t) for t in reg.TARGETS.values()]}


def _roles(report:dict[str, Any]) -> list[dict[str, Any]]:
  return [{"role": r["role"], "quant": r["quant"], "shape": r["shape"], "bytes_moved": r["bytes_moved"],
           "floor_ms": r["floor_ms"], "share": r["whole_share"], "regime": r["regime"]} for r in report["roles"]]


def _block(report:dict[str, Any], context:int) -> dict[str, Any]:
  whole = report["whole"]
  floor_ms = whole["floor_ms"]
  return {"context": context, "bytes_moved": whole["bytes_moved"], "floor_ms": floor_ms,
          "tok_s": (context * 1000.0 / floor_ms) if floor_ms > 0 else None, "regime": whole["regime"],
          "roles": _roles(report)}


def bandwidth_uncertainty(target, read:dict[str, Any] | None = None) -> float | None:
  """The memory speed's own measured uncertainty: this machine's read probe's when the speed is its read, else the
  registry figure's (targets.uncertainty). None: the guards that need it say not checked."""
  if read and read.get("source") == "explicit --peak-gbs":
    return None
  return uncertainty(read) if read and "uncertainty" in read else fact_uncertainty(target)


def ceiling(profile:dict[str, Any], target, *, context:int = 512, dtype:str = "fp16",
            peak_gbs:float | None = None, peak_tflops:float | None = None,
            read:dict[str, Any] | None = None) -> dict[str, Any]:
  """The roofline for one model on one chip: decode (one token) and prefill (the context), with tokens/s.
  read is this machine's one-GPU read bandwidth (read_bandwidth): when given it is the memory speed, so the
  limit, the tie-out and the per-role rule use the same number, and bandwidth_source says where it came from."""
  try:
    peak_flops = resolve_peak_flops(target, dtype, peak_tflops)
  except SystemExit as exc:
    raise Refused(str(exc)) from exc
  if read is None and (known := profile_read({"target_id": target.target_id})):
    when = ((target.capabilities.get("fact_sources") or {}).get("memory_bandwidth_gbs") or {}).get("observed_at")
    read = {"gbs": known[0], "source": known[1], "short": short_source(known[1], when) + ", chip profile"}
  if peak_gbs is not None:
    read = {"gbs": peak_gbs, "short": "explicit --peak-gbs", "source": "explicit --peak-gbs"}
  bw = (read or {}).get("gbs") or target.memory_bandwidth_gbs
  if not bw:
    raise Refused(f"target {target.target_id!r} carries no memory_bandwidth_gbs, so there is no memory ceiling; "
                  "on the machine with this GPU, `python -m boltbeam.workflow.screen autoscan` measures it once and "
                  "saves a chip profile; or pass --peak-gbs, or add a measured figure to boltbeam/data/targets.json")
  decode = model_roofline(profile, peak_flops=peak_flops, peak_bw_bytes_s=bw * 1e9, context=1)
  prefill = model_roofline(profile, peak_flops=peak_flops, peak_bw_bytes_s=bw * 1e9, context=context)
  return {
    "schema": SCHEMA, "kind": "ceiling", "model_id": profile["model_id"], "target": target_facts(target),
    "peak_bandwidth_gbs": bw, "bandwidth_source": (read or {}).get("short"), "uncertainty": bandwidth_uncertainty(target, read), "peak_tflops": peak_flops / 1e12, "truth_status": decode["truth_status"],
    "ridge_intensity": decode["ridge_intensity"], "assumptions": decode["assumptions"],
    "decode": _block(decode, 1), "prefill": _block(prefill, context),
    # what the limit counts and what it excludes, with reasons (profile/weight_ledger.py)
    "weights": weight_ledger.facts((profile.get("metadata") or {}).get("weights")),
  }


def ceilings(profile:dict[str, Any]) -> dict[str, Any]:
  """The decode limit of this model on every registered chip that carries a ceiling: read-only what-if facts."""
  chips = []
  for t in reg.TARGETS.values():
    if not target_facts(t)["has_ceiling"]:
      continue
    try:
      c = ceiling(profile, t)
    except (Refused, SystemExit):
      continue
    chips.append({"id": t.target_id, "tok_s": c["decode"]["tok_s"], "floor_ms": c["decode"]["floor_ms"],
                  "peak_bandwidth_gbs": c["peak_bandwidth_gbs"]})
  return {"schema": SCHEMA, "kind": "ceilings", "model_id": profile["model_id"], "chips": chips}

def run_read(run:pathlib.Path, target_id:str | None) -> dict[str, Any] | None:
  """The run's one-GPU read bandwidth from its machine facts (read_bandwidth), when its GPU is the run's chip."""
  got = read_bandwidth(optional(run, MACHINE), optional(run, MEASURE_STATUS).get("layout") or "one")
  return got if got and got.get("target_id") in (None, target_id) else None


def run_bandwidth(run:pathlib.Path, target) -> tuple[float | None, str]:
  """The one read bandwidth every number of a run uses (the limit, the tie-out, the per-role rule, every trace's
  peak_gbs and the probe's target): this GPU's measured read from the run's machine facts when it has them, else
  the chip's own figure, labelled. Collectors call this instead of reading the registry themselves."""
  got = run_read(run, target.target_id)
  if got:
    return got["gbs"], got["short"]
  known = profile_read({"target_id": target.target_id})
  return target.memory_bandwidth_gbs, (known[1] if known else f"the chip's registry figure ({target.target_id}), not measured on this GPU")


def measured_vs_ceiling(manifest:dict[str, Any], profile:dict[str, Any], run:pathlib.Path | None = None) -> dict[str, Any]:
  """The modeled ceiling for this run's model and chip, or the reason there is none. Never a guess. With machine
  facts in the run, the memory speed is the one measured on this GPU (or the registry's, labelled)."""
  if not profile:
    return {"status": "absent", "reason": "no model_profile.json in the run"}
  read = run_read(run, manifest.get("target_id")) if run is not None else None
  try:
    c = ceiling(profile, get_target(manifest.get("target_id")), read=read)
  except (Refused, SystemExit) as exc:
    return {"status": "absent", "reason": str(exc)}
  block = c["decode"] if manifest.get("workload") == "decode" else c["prefill"]
  return {"status": "modeled", "context": block["context"], "tok_s": block["tok_s"], "floor_ms": block["floor_ms"],
          "bytes_moved": block["bytes_moved"],
          "peak_bandwidth_gbs": c["peak_bandwidth_gbs"], "bandwidth_source": c["bandwidth_source"],
          # each floor carries the memory speed's measured uncertainty beside it (None: its floor rule is not checked)
          "uncertainty": c.get("uncertainty"), "_roles": [{**r, "floor_uncertainty": c.get("uncertainty")} for r in block["roles"]]}

def decode_refused(profile:dict[str, Any] | None, run:pathlib.Path) -> str | None:
  """Why the run has no decode limit: the model formula cannot limit the profile (roofline/mamba2, a Mamba2 hybrid)
  and the run holds no captured decode work to limit it from (DECODE_CAPTURE). None: the limit stands. The one
  rule the decode report and the replay's decode limit follow. measured_vs_ceiling keeps the modeled ceiling, since its
  roles are also the role ledger the pairing reads (attribution, prefill): refusing there would unpair the roles."""
  from boltbeam.roofline.mamba2 import FORMULA_REFUSED, formula_cannot_limit
  return FORMULA_REFUSED if formula_cannot_limit(profile) and not (run / DECODE_CAPTURE).is_file() else None


def profile_read(gpu:dict[str, Any]) -> tuple[float, str] | None:
  """A GPU's read bandwidth from the chip profile made on this machine, with its date; None for a registry chip.
  One measurement: the profile's number is the limit's number."""
  tid = gpu.get("target_id")
  if not reg.is_local(tid):
    return None
  t = get_target(tid)
  src = ((t.capabilities or {}).get("fact_sources") or {}).get("memory_bandwidth_gbs") or {}
  if not t.memory_bandwidth_gbs:
    return None
  row = vendors.BY_BACKEND.get(t.backend)
  return t.memory_bandwidth_gbs, (f"measured on this GPU with {row.term('native_read_probe') if row else 'BoltBeam'}, "
                                  f"{src.get('observed_at', 'date unknown')} (chip profile {tid})")


# --- the one read bandwidth a single-GPU limit uses -----------------------------------------------------------------

def short_source(source:str | None, measured_at:str | None = None) -> str:
  """The read bandwidth's source in a few words for a screen line: which probe, and when; or that it is the
  registry's figure. The full source stays in the limit's inputs."""
  from boltbeam.target import vendors
  s = source or ""
  when = f", {measured_at}" if measured_at else ""
  probes = [(t, t.removeprefix("BoltBeam's ")) for v in vendors.VENDORS.values() if (t := v.term("native_read_probe"))]
  for needle, words in (*probes, ("tinygrad read-sum", "tinygrad read-sum, a lower bound")):
    if needle in s:
      return f"measured on this GPU ({words}{when})"
  if s.startswith("registry figure"):
    return "registry figure, not measured on this GPU"
  return s or "unknown"


def read_bandwidth(facts:dict[str, Any] | None, layout:str = "one") -> dict[str, Any] | None:
  """The read bandwidth the one-GPU limit uses (the layout's first GPU), with its source in full and in short;
  None for no facts, a multi-GPU layout or a GPU with no read bandwidth. Measured or registry, it is the one number
  the speed limit, the tie-out and the per-role rule all use."""
  if not facts or not facts.get("gpus") or (layout != "one" and len(facts["gpus"]) > 1):
    return None
  g = facts["gpus"][0]
  if not g.get("read_gbs"):
    return None
  return {"gbs": g["read_gbs"], "source": g.get("read_source"), "target_id": g.get("target_id"),
          **{k: g[k] for k in ("shape_spread", "drift", "uncertainty") if g.get(k) is not None},
          "measured": str(g.get("read_source") or "").startswith("measured"),
          "short": short_source(g.get("read_source"), facts.get("measured_at"))}
