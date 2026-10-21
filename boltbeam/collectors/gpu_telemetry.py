"""The GPU's clock, power and temperature while something runs, sampled from the vendor's management library, and
the clock policy a measurement ran under.

    with Telemetry(device) as t:      a background thread samples every INTERVAL_S
        run()
    t.summary()                       SM clock (mean, min, max), power, temperature, the throttle reasons seen
    clock_policy(device)              fixed clock if this user may set it, else fixed power, else the default

A compute rate follows the SM clock, so a peak or a prefill time without the clock it ran at is half a fact. The
sampler reads the library the driver ships (NVML, libnvidia-ml) through ctypes; no Python package is needed. A
machine without it gets an empty summary with its reason, and the measurement still runs.

Throttling is the library's own word: the clock-event reasons it reports (THROTTLE_REASONS) while the work ran. A
reason that only says the GPU was idle or that an application set the clock is not a throttle.
"""
from __future__ import annotations

import ctypes
import pathlib
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any

INTERVAL_S = 0.05
LIBRARY = "libnvidia-ml.so.1"
_CLOCK_SM, _CLOCK_MEM, _TEMP_GPU = 1, 2, 0  # nvmlClockType_t NVML_CLOCK_SM, NVML_CLOCK_MEM; NVML_TEMPERATURE_GPU
# nvmlClocksEventReasons bits: the ones that slow the clock below what the work asks for
THROTTLE_REASONS = {0x4: "software power cap", 0x8: "hardware slowdown", 0x20: "software thermal slowdown",
                    0x40: "hardware thermal slowdown", 0x80: "hardware power brake"}
IDLE = 0x1


class _Nvml:
  """The few NVML calls the sampler makes, one reviewed boundary."""

  def __init__(self, index:int):
    self.lib = ctypes.CDLL(LIBRARY)
    if self.lib.nvmlInit_v2() != 0:
      raise OSError("nvmlInit failed")
    self.handle = ctypes.c_void_p()
    if self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self.handle)) != 0:
      raise OSError(f"no NVML device {index}")

  def _uint(self, fn:str, *args) -> int | None:
    v = ctypes.c_uint()
    return v.value if getattr(self.lib, fn)(self.handle, *args, ctypes.byref(v)) == 0 else None

  def memory_mib(self) -> tuple[float | None, float | None]:
    """The device memory in use and in all, MiB (nvmlMemory_t: total, free, used)."""
    m = (ctypes.c_ulonglong * 3)()
    if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(m)) != 0:
      return None, None
    return m[2] / 2**20, m[0] / 2**20

  def sample(self) -> dict[str, Any]:
    reasons = ctypes.c_ulonglong()
    ok = self.lib.nvmlDeviceGetCurrentClocksEventReasons(self.handle, ctypes.byref(reasons)) == 0
    mw = self._uint("nvmlDeviceGetPowerUsage")
    return {"t": time.monotonic(), "sm_mhz": self._uint("nvmlDeviceGetClockInfo", ctypes.c_int(_CLOCK_SM)),
            "mem_mhz": self._uint("nvmlDeviceGetClockInfo", ctypes.c_int(_CLOCK_MEM)),
            "power_w": mw / 1000.0 if mw is not None else None,
            "temp_c": self._uint("nvmlDeviceGetTemperature", ctypes.c_int(_TEMP_GPU)),
            "reasons": reasons.value if ok else None, "mem_mib": self.memory_mib()[0]}

  def power_limit_w(self) -> float | None:
    mw = self._uint("nvmlDeviceGetEnforcedPowerLimit")
    return mw / 1000.0 if mw is not None else None

  def close(self) -> None:
    self.lib.nvmlShutdown()


class AmdSysfs:
  """The same samples as _Nvml for an AMD GPU, from the amdgpu driver's sysfs and hwmon files, which need no
  library. The card is found by its PCI address (HIP's device order and the kernel's card numbers differ), so the
  caller names the device as the driver does. A file the driver does not provide reads as unknown. amdgpu reports
  no throttle reasons, so `reasons` is None: unknown, not "not throttled"."""

  ROOT = "/sys/class/drm"

  def __init__(self, pci_bus_id:str, root:str | None = None):
    want = pci_bus_id.lower()
    base = pathlib.Path(root or self.ROOT)
    found = [d / "device" for d in sorted(base.glob("card[0-9]*")) if (d / "device" / "uevent").is_file()
             and f"pci_slot_name={want}" in (d / "device" / "uevent").read_text().lower()]
    if not found:
      raise OSError(f"no amdgpu card at PCI {pci_bus_id} under {base}")
    self.device = found[0]
    self.hwmon = next(iter(sorted((self.device / "hwmon").glob("hwmon*"))), None)

  def _read(self, path:pathlib.Path | None) -> int | None:
    try:
      return int(path.read_text().split()[0]) if path else None
    except (OSError, ValueError, IndexError):
      return None

  def _hw(self, name:str) -> int | None:
    return self._read(self.hwmon / name if self.hwmon else None)

  def memory_mib(self) -> tuple[float | None, float | None]:
    used, total = (self._read(self.device / f) for f in ("mem_info_vram_used", "mem_info_vram_total"))
    return (used / 2**20 if used is not None else None, total / 2**20 if total is not None else None)

  def sample(self) -> dict[str, Any]:
    sclk, mclk, uw, mc = self._hw("freq1_input"), self._hw("freq2_input"), self._hw("power1_average"), self._hw("temp1_input")
    return {"t": time.monotonic(), "sm_mhz": sclk / 1e6 if sclk is not None else None,
            "mem_mhz": mclk / 1e6 if mclk is not None else None, "power_w": uw / 1e6 if uw is not None else None,
            "temp_c": mc / 1000 if mc is not None else None, "reasons": None, "mem_mib": self.memory_mib()[0]}

  def power_limit_w(self) -> float | None:
    uw = self._hw("power1_cap")
    return uw / 1e6 if uw is not None else None

  def performance_level(self) -> str | None:
    try:
      return (self.device / "power_dpm_force_performance_level").read_text().strip()
    except OSError:
      return None

  def close(self) -> None:
    pass


def amd_clock_policy(source:AmdSysfs) -> dict[str, Any]:
  """AMD's clock policy, read without changing anything: setting the clock is a root write to sysfs, which a
  measurement does not make, so the board runs at its own power limit (fixed power) and the clock it chose is logged.
  The driver's performance level is recorded as found. A card whose sysfs is not here has an unknown policy."""
  if isinstance(source, Unsampled):
    return {"policy": None, "clock_settable": None, "words": f"the clock policy is unknown: {source.reason}"}
  level = source.performance_level()
  return {"policy": "fixed_power", "clock_settable": False, "words": POLICY_WORDS["fixed_power"],
          "why": f"amdgpu performance level {level!r}; BoltBeam does not write it" if level else
                 "amdgpu performance level unreadable; BoltBeam does not write it"}


def reason_words(bits:int) -> list[str]:
  return [w for b, w in THROTTLE_REASONS.items() if bits & b]


HELD = 0.98  # a throttle reason with the SM clock at or above this share of the run's highest clock held the clock


def summarize(samples:list[dict[str, Any]], *, power_limit_w:float | None = None, top_mhz:float | None = None,
              memory_total:float | None = None) -> dict[str, Any]:
  """The samples as one record: SM clock mean, min and max (MHz), power mean and max, temperature max, and the
  throttle reasons seen with the share of samples each was set in. The driver sets a reason (a software power cap)
  while the clock stays where it was, so a sample counts as throttled only when its clock is also below HELD of the
  run's highest clock (top_mhz, else these samples' own): throttled is True when any sample was; held lists the
  reasons that were set with the clock held. Samples the driver marks idle (a program loading its model) are left
  out of every figure when the GPU worked at all; all_samples counts them."""
  every = samples
  samples = [s for s in every if not (s.get("reasons") or 0) & IDLE] or every  # the GPU at work, not loading a model
  clocks = [s["sm_mhz"] for s in samples if s.get("sm_mhz")]
  memory = [s["mem_mhz"] for s in samples if s.get("mem_mhz")]  # a weight read's rate follows the memory clock
  power = [s["power_w"] for s in samples if s.get("power_w") is not None]
  temps = [s["temp_c"] for s in samples if s.get("temp_c") is not None]
  top = top_mhz or (max(clocks) if clocks else None)
  seen: dict[str, int] = {}
  held: dict[str, int] = {}
  for s in samples:
    slow = top is not None and s.get("sm_mhz") is not None and s["sm_mhz"] < HELD * top
    for w in reason_words(s.get("reasons") or 0):
      (seen if slow else held)[w] = (seen if slow else held).get(w, 0) + 1
  out = {"samples": len(samples), "all_samples": len(every), "interval_s": INTERVAL_S,
         "sm_mhz": {"mean": round(statistics.fmean(clocks), 1), "min": min(clocks), "max": max(clocks)} if clocks else None,
         "mem_mhz": {"mean": round(statistics.fmean(memory), 1), "min": min(memory), "max": max(memory)} if memory else None,
         "power_w": {"mean": round(statistics.fmean(power), 1), "max": round(max(power), 1)} if power else None,
         "temp_c_max": max(temps) if temps else None, "power_limit_w": power_limit_w,
         "throttle": {w: round(n / len(samples), 3) for w, n in sorted(seen.items())},
         "held": {w: round(n / len(samples), 3) for w, n in sorted(held.items())}, "top_mhz": top,
         # a source that reports no throttle reasons (amdgpu) leaves throttling unknown, not false
         "throttled": bool(seen) if any(s.get("reasons") is not None for s in samples) else None}
  used = [s["mem_mib"] for s in every if s.get("mem_mib") is not None]
  if used:  # what the device held at most while the work ran (a model loading counts: it is held memory)
    out["memory_mib"] = {"used_max": round(max(used), 1), "total": memory_total}
  out["words"] = words(out)
  return out


def compact(samples:list[dict[str, Any]]) -> list[list[Any]]:
  """The raw samples kept with a measurement, so its summary can be read again: [seconds from the first, SM MHz,
  W, C, reason bits, memory MHz]."""
  t0 = samples[0]["t"] if samples else 0.0
  return [[round(s["t"] - t0, 3), s.get("sm_mhz"), round(s["power_w"], 1) if s.get("power_w") is not None else None,
           s.get("temp_c"), s.get("reasons"), s.get("mem_mhz")] for s in samples]


def words(s:dict[str, Any]) -> str:
  if not s.get("sm_mhz"):
    return s.get("reason") or "no clock samples"
  c = s["sm_mhz"]
  text = f"SM clock {c['mean']:.0f} MHz mean ({c['min']}-{c['max']}) over {s['samples']} busy samples"
  if s.get("power_w"):
    text += f", {s['power_w']['mean']:.0f} W mean, {s['power_w']['max']:.0f} W max"
    if s.get("power_limit_w"):
      text += f" of {s['power_limit_w']:.0f} W"
  if s.get("temp_c_max") is not None:
    text += f", {s['temp_c_max']} C max"
  if s["throttled"]:
    text += "; THROTTLED: " + ", ".join(f"{w} in {100 * f:.0f}% of samples" for w, f in s["throttle"].items())
  elif s["throttled"] is None:
    text += "; throttling unknown (the driver reports no throttle reasons)"
  else:
    text += "; not throttled"
  if s.get("held"):
    text += (" (" + ", ".join(f"{w} set in {100 * f:.0f}% of samples" for w, f in s["held"].items())
             + f" with the clock held within {100 * (1 - HELD):.0f}% of {s['top_mhz']:.0f} MHz)")
  return text


class Unsampled:
  """A GPU with no telemetry source here (no vendor row, or its card not found): Telemetry samples nothing and its
  summary says why. Never the NVIDIA library in its place."""

  def __init__(self, reason:str):
    self.reason = reason


def source_for(device:dict[str, Any] | None):
  """The telemetry source for one hardware-profile GPU, from its vendor's row (chips._vendors() "telemetry"): the
  amdgpu sysfs for an AMD card, found by its PCI address; None for NVIDIA, whose library Telemetry opens by index
  (the scan's order is NVML's). A vendor with no source, or a card not found, is Unsampled with the reason. No
  device (a run with no hardware profile) is None, as before the profile named the GPU."""
  if not device:
    return None
  from boltbeam.workflow.chips import NoVendorRow, vendor_row
  try:
    return vendor_row(device.get("vendor"), "telemetry")(device)
  except (NoVendorRow, OSError, KeyError) as exc:
    return Unsampled(f"no clock samples: {exc}")


def opened(source, *args, make=None):
  """Telemetry(*args) on a source; with no source (NVIDIA's library, by index) called exactly as before there was one."""
  make = make or Telemetry
  return make(*args) if source is None else make(*args, source=source)


def run_telemetry(run:pathlib.Path, *args, make=None):
  """Telemetry(*args) on the run's GPU (common.run_gpu), by its vendor's source (source_for)."""
  from boltbeam.workflow.common import run_gpu
  return opened(source_for(run_gpu(run)), *args, make=make)


def run_clock_policy(run:pathlib.Path) -> dict[str, Any]:
  """The clock policy the run's GPU holds, decided by its vendor's row (chips._vendors() "clock_policy"); unknown,
  with the reason, for a vendor with none. A run with no hardware profile asks nvidia-smi, as before."""
  from boltbeam.workflow.chips import NoVendorRow, vendor_row
  from boltbeam.workflow.common import run_gpu
  if (device := run_gpu(run)) is None:
    return clock_policy(0)
  try:
    return vendor_row(device.get("vendor"), "clock_policy")(source_for(device))
  except NoVendorRow as exc:
    return {"policy": None, "clock_settable": None, "words": f"the clock policy is unknown: {exc}"}


class Telemetry:
  """Samples the GPU in a background thread while the block runs. summary() after the block. On a machine without
  the library the summary says so and nothing is sampled."""

  def __init__(self, device:int = 0, interval_s:float = INTERVAL_S, *, source=None):
    self.device, self.interval_s = device, interval_s
    self.samples: list[dict[str, Any]] = []
    self.reason: str | None = None
    self._source, self._stop, self._thread, self._limit, self._total = source, threading.Event(), None, None, None

  def __enter__(self) -> "Telemetry":
    if isinstance(self._source, Unsampled):
      self._nvml, self.reason = None, self._source.reason
      return self
    try:
      self._nvml = self._source or _Nvml(self.device)
    except OSError as exc:
      self._nvml, self.reason = None, f"no clock samples: {LIBRARY} not usable here ({exc})"
      return self
    self._limit = self._nvml.power_limit_w()
    self._total = getattr(self._nvml, "memory_mib", lambda: (None, None))()[1]
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()
    return self

  def _loop(self) -> None:
    while not self._stop.is_set():
      self.samples.append(self._nvml.sample())
      self._stop.wait(self.interval_s)

  def __exit__(self, *exc) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join()
      self.samples.append(self._nvml.sample())
    if self._nvml is not None and self._source is None:
      self._nvml.close()

  def window(self, t0:float, t1:float) -> list[dict[str, Any]]:
    """The samples taken between two time.monotonic() readings."""
    return [s for s in self.samples if t0 <= s["t"] <= t1]

  def summary(self, t0:float | None = None, t1:float | None = None) -> dict[str, Any]:
    if self._nvml is None:
      return {"samples": 0, "reason": self.reason, "throttled": None, "sm_mhz": None, "words": self.reason}
    rows = self.window(t0, t1) if t0 is not None and t1 is not None else self.samples
    clocks = [s["sm_mhz"] for s in self.samples if s.get("sm_mhz") and not (s.get("reasons") or 0) & IDLE]
    return summarize(rows, power_limit_w=self._limit, top_mhz=max(clocks) if clocks else None, memory_total=self._total)


# --- the clock policy ----------------------------------------------------------------------------------------------

POLICY_WORDS = {
  "fixed_clock": "fixed clock: the SM clock is locked to the chip's maximum for the run (code and architecture comparison)",
  "fixed_power": ("fixed power: the clock cannot be set by this user, so the board runs at its own power limit, "
                  "unchanged, and the clock it chose is logged (realism; a rate is quoted with its clock)"),
}


def clock_policy(device:int = 0, *, run=subprocess.run) -> dict[str, Any]:
  """Which policy a measurement here can hold, decided by asking the driver tool for each without changing anything:
  a clock reset (-rgc) is allowed only to a user who may set clocks. Fixed clock when it is; else fixed power (the
  board's own limit, which no run changes); the record says which and why."""
  smi = shutil.which("nvidia-smi")
  if not smi:
    return {"policy": None, "words": "no nvidia-smi: the clock policy is unknown", "clock_settable": None}
  proc = run([smi, "-i", str(device), "-rgc"], capture_output=True, text=True, timeout=30, check=False)
  settable = proc.returncode == 0
  why = (proc.stdout + proc.stderr).strip().splitlines()[0] if not settable and (proc.stdout + proc.stderr).strip() else None
  policy = "fixed_clock" if settable else "fixed_power"
  return {"policy": policy, "clock_settable": settable, "words": POLICY_WORDS[policy],
          "why": why or ("the clock can be set here" if settable else "nvidia-smi refused the clock reset")}


class hold:
  """Hold the policy for a block: under fixed_clock the SM clock is locked to the chip's own maximum (nvidia-smi
  -lgc) and reset after; under fixed_power nothing is changed."""

  def __init__(self, policy:dict[str, Any], device:int = 0, *, run=subprocess.run):
    self.policy, self.device, self.run = policy, device, run

  def __enter__(self) -> dict[str, Any]:
    if self.policy.get("policy") == "fixed_clock":
      smi = shutil.which("nvidia-smi") or "nvidia-smi"
      top = self.run([smi, "-i", str(self.device), "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=30, check=False).stdout.strip()
      self.run([smi, "-i", str(self.device), "-lgc", f"{top},{top}"], capture_output=True, text=True, timeout=30, check=True)
      self.policy = {**self.policy, "locked_mhz": float(top)}
    return self.policy

  def __exit__(self, *exc) -> None:
    if self.policy.get("policy") == "fixed_clock":
      self.run([shutil.which("nvidia-smi") or "nvidia-smi", "-i", str(self.device), "-rgc"], capture_output=True,
               text=True, timeout=30, check=False)
