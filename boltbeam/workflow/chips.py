"""Chip autoscan: use the profile that fits this machine's GPU, or measure a new one and keep it.

    known   a registry row or a profile made earlier on this machine claims the GPU (targets.match_target): use it
    new     nothing claims it: measure read bandwidth and an fp16 matrix rate here, save the profile, use it

The match rule is the exact GPU name, the architecture, the memory size in GiB, and the core count where the driver
tool reports one (autoscan.match_facts). A profile is a registry row in the registry's own shape, saved in this
machine's store (targets.chips_dir: $BOLTBEAM_CHIPS_DIR or ~/.boltbeam/chips), never in the repo. Its id comes from
the hardware (autoscan.chip_profile_id). Every number in it says its source and date. "Measure again" refreshes it.

chip_list sorts every chip for Setup: this machine, measured chips (what-if limits), chips not measured yet, and
families (not a run target). Python decides the groups; the screen draws them.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
from typing import Any, Callable

from boltbeam.target import targets as reg
from boltbeam.workflow.autoscan import _hardware_profile, chip_profile_id

Measure = Callable[[], dict[str, Any]]
_SMI_FACTS = "driver_version,clocks.max.sm,clocks.max.memory,power.limit"


def _today() -> str:
  return _dt.date.today().isoformat()


def _nvidia_smi_facts(index:int = 0) -> dict[str, Any]:
  """Driver facts nvidia-smi reports for one GPU; an empty dict when it cannot answer."""
  smi = shutil.which("nvidia-smi")
  if not smi:
    return {}
  try:
    out = subprocess.run([smi, "-i", str(index), f"--query-gpu={_SMI_FACTS}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10, check=False).stdout.strip()
  except (OSError, subprocess.TimeoutExpired):
    return {}
  cells = [c.strip() for c in out.split(",")]
  if len(cells) != 4:
    return {}
  num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None  # noqa: E731
  return {"driver_version": cells[0], "max_sm_clock_mhz": num(cells[1]), "max_memory_clock_mhz": num(cells[2]),
          "power_limit_w": num(cells[3])}


def _probes(vendor:str) -> tuple[Measure, Measure]:
  """The read-bandwidth and matrix-rate probes for this vendor, both BoltBeam's own."""
  if vendor == "apple":
    from boltbeam.collectors import metal_bandwidth as m
    return m.measure_read_gbs, m.measure_matrix_tflops
  if vendor == "nvidia":
    from boltbeam.collectors import cuda_bandwidth as c
    return c.measure_read_gbs, c.measure_matrix_tflops
  raise RuntimeError(f"no BoltBeam probe for {vendor or 'this'} GPUs yet")


def profile_row(device:dict[str, Any], read:dict[str, Any], matrix:dict[str, Any], driver:dict[str, Any],
                *, today:str) -> dict[str, Any]:
  """A registry-shaped row for one GPU from what was measured on it. Pure: the probes ran before."""
  pid = chip_profile_id(device)
  if not pid:
    raise RuntimeError(f"the scan of {device.get('name')!r} lacks the name, memory or core count a profile id needs")
  apple = device.get("vendor") == "apple"
  scope = device.get("name") + (f", {device.get('gpu_cores')}-core GPU" if apple else f", driver {driver.get('driver_version') or device.get('driver_version')}")
  scan = {"kind": "hardware_scan", "scope": scope, "observed_at": today,
          "method": "system_profiler SPDisplaysDataType and the Metal device" if apple else
                    "nvidia-smi and cudaDeviceGetAttribute (BoltBeam's CUDA matrix probe prints them)",
          **({"working_set_bytes": matrix.get("working_set_bytes"), "unified_memory_bytes": device.get("unified_memory_bytes")} if apple else
             {k: matrix.get(k) for k in ("sm_count", "shared_mem_per_sm_bytes", "l2_cache_bytes", "clock_khz",
                                          "memory_clock_khz", "memory_bus_bits", "total_global_mem_bytes")}),
          **driver}
  match = dict(device.get("match") or {})
  caps = {
    "target_kind": "exact",
    "profile_source": "generated",
    "tinygrad_device": "METAL" if apple else "NV",
    **({} if apple else {"compiler_arch": device.get("architecture")}),
    "match": match,
    "fact_status": {"memory_bandwidth_gbs": "measurement", "matrix_tflops": "measurement_lower_bound",
                    "wave_size": "hardware_family",
                    "vram_bytes": "not_applicable_unified_memory" if apple else "hardware_scan",
                    "compute_units": "unknown" if apple else "hardware_scan"},
    "fact_sources": {
      "hardware_scan": scan,
      "memory_bandwidth_gbs": {"kind": "measurement", "scope": scope, "method": read["method"],
                               "value_gbs": read["read_gbs"], "observed_at": today},
      "matrix_tflops": {"kind": "measurement", "bound": "lower", "scope": scope, "method": matrix["method"],
                        "value_tflops": matrix["tflops"], "dtype": "fp16", "note": matrix["note"], "observed_at": today},
    },
  }
  return {
    "target_id": pid, "backend": "Metal" if apple else "CUDA", "wave_size": 32, "subgroup_size": 32,
    "lds_bytes_per_cu": None if apple else matrix.get("shared_mem_per_sm_bytes"),
    "vram_bytes": None if apple else device.get("memory_bytes"), "vector_load_bits": 128,
    "compute_units": None if apple else matrix.get("sm_count"),
    "memory_bandwidth_gbs": read["read_gbs"], "peak_tflops": {}, "matrix_tflops": {"fp16": matrix["tflops"]},
    "dot_primitives": ["simdgroup_matrix"] if apple else ["mma_m16n8k16", "dp4a"], "dequant_primitives": [],
    "backend_status": "descriptor_only", "capabilities": caps,
  }


def _first_gpu(hardware:dict[str, Any]) -> dict[str, Any] | None:
  gpu = hardware.get("gpu") or {}
  devs = gpu.get("devices") or []
  return devs[0] if devs else None


def _view(device:dict[str, Any], tid:str | None, status:str, action:str, path:str | None = None) -> dict[str, Any]:
  return {"kind": "chip_autoscan", "status": status, "action": action, "name": device.get("name"), "target_id": tid,
          "source": "generated" if reg.is_local(tid) else ("registry" if tid else None),
          "match": device.get("match"), "path": path}


def autoscan(*, remeasure:bool = False, hardware:dict[str, Any] | None = None,
             probes:Callable[[str], tuple[Measure, Measure]] = _probes,
             driver:Callable[[], dict[str, Any]] = _nvidia_smi_facts) -> dict[str, Any]:
  """Match this machine's first GPU to a profile, or measure one. remeasure refreshes a profile made here (a
  registry row is reviewed data and is never rewritten from one machine)."""
  device = _first_gpu(hardware or _hardware_profile())
  if device is None:
    return {"kind": "chip_autoscan", "status": "no_gpu", "action": "none", "name": None, "target_id": None,
            "source": None, "match": None, "path": None}
  profile = device.get("profile") or {}
  if profile.get("status") == "known" and not (remeasure and profile.get("source") == "generated"):
    action = "kept" if not remeasure else "kept: a built-in profile is not measured again from one machine"
    return _view(device, profile.get("id"), "known", action)
  today = _today()
  try:
    read_probe, matrix_probe = probes(device.get("vendor") or "")
    row = profile_row(device, read_probe(), matrix_probe(), driver() if device.get("vendor") == "nvidia" else {},
                      today=today)
  except (RuntimeError, OSError, subprocess.SubprocessError, KeyError) as exc:  # the probe's own words, not a guess
    return {**_view(device, profile.get("id") if profile.get("status") == "known" else None, profile.get("status") or "new",
                    "failed"), "reason": str(exc)[-300:]}
  path = reg.save_local_profile(row, measured_at=today)
  return _view(device, row["target_id"], "known", "measured again" if remeasure else "generated", str(path))


# --- the chip list Setup draws ------------------------------------------------------------------------------------

def _measured_words(t) -> str:
  caps = t.capabilities or {}
  src = (caps.get("fact_sources") or {}).get("memory_bandwidth_gbs") or {}
  status = (caps.get("fact_status") or {}).get("memory_bandwidth_gbs")
  if status == "measurement":
    when = f", measured {src['observed_at']}" if src.get("observed_at") else ", measured"
  else:
    when = f", {status.replace('_', ' ')} figure, not measured" if status else ", source not recorded"
  return f"{t.memory_bandwidth_gbs:.1f} GB/s{when}"


def _vendor_words(t) -> str | None:
  """A vendor figure the registry records for a chip it has not measured, labelled so; None when there is none."""
  src = ((t.capabilities or {}).get("fact_sources") or {}).get("memory_bandwidth_gbs") or {}
  gbs = src.get("vendor_spec_gbs")
  return f"vendor figure {gbs:.0f} GB/s, not measured" if gbs else None


def chip_list(detected:dict[str, Any] | None = None) -> dict[str, Any]:
  """Every chip in the four groups Setup shows, in order. `detected` is this machine's first GPU from the scan."""
  from boltbeam.workflow.screen import target_facts
  device = detected if detected is not None else (_first_gpu(_hardware_profile()) or {})
  profile = device.get("profile") or {}
  here = profile.get("id") if profile.get("status") == "known" else None
  this = {"name": device.get("name"), "target_id": here, "status": profile.get("status") or "no_gpu",
          "source": profile.get("source"), "new_id": profile.get("id") if profile.get("status") == "new" else None}
  if here and reg.is_local(here):
    src = (reg.TARGETS[here].capabilities.get("fact_sources") or {}).get("memory_bandwidth_gbs") or {}
    this["words"] = f"profile made on {here_words()}, {src.get('observed_at', 'date unknown')}"
  elif here:
    this["words"] = "built-in profile"
  elif this["status"] == "new":
    this["words"] = "new chip: no profile yet. Autoscan measures it (a minute at most)."
  else:
    this["words"] = "no GPU found"
  rows = list(reg.TARGETS.values())
  facts = [target_facts(t) for t in rows]
  groups = {"this": [], "measured": [], "not_measured": [], "families": []}
  words = {}
  for t, f in zip(rows, facts):
    tid = t.target_id
    if reg.target_kind(tid) == "family":
      groups["families"].append(tid)
      words[tid] = "a family, not one chip: not a run target"
    elif tid == here:
      groups["this"].append(tid)
      words[tid] = this["words"]
    elif f["has_ceiling"]:
      groups["measured"].append(tid)
      words[tid] = "what if: " + _measured_words(t) + (", made on this machine" if reg.is_local(tid) else "")
    else:
      groups["not_measured"].append(tid)
      words[tid] = "run BoltBeam on one to measure it" + (f" ({v})" if (v := _vendor_words(t)) else "")
  titles = {"this": "This machine", "measured": "Measured chips", "not_measured": "Not measured yet", "families": "Families"}
  return {"kind": "chips", "targets": facts, "this_machine": this,
          "groups": [{"key": k, "title": titles[k], "selectable": k in ("this", "measured"), "folded": k == "families",
                      "chips": [{"id": tid, "words": words[tid]} for tid in ids]} for k, ids in groups.items()]}


def here_words() -> str:
  import sys
  return "this Mac" if sys.platform == "darwin" else "this machine"
