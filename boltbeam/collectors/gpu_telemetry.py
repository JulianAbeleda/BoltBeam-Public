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
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any

INTERVAL_S = 0.05
LIBRARY = "libnvidia-ml.so.1"
_CLOCK_SM, _TEMP_GPU = 1, 0  # nvmlClockType_t NVML_CLOCK_SM; nvmlTemperatureSensors_t NVML_TEMPERATURE_GPU
# nvmlClocksEventReasons bits: the ones that slow the clock below what the work asks for
THROTTLE_REASONS = {0x4: "software power cap", 0x8: "hardware slowdown", 0x20: "software thermal slowdown",
                    0x40: "hardware thermal slowdown", 0x80: "hardware power brake"}
IDLE = 0x1
OTHER_REASONS = {0x1: "idle", 0x2: "application clock setting", 0x10: "sync boost", 0x100: "display clock setting"}


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

  def sample(self) -> dict[str, Any]:
    reasons = ctypes.c_ulonglong()
    ok = self.lib.nvmlDeviceGetCurrentClocksEventReasons(self.handle, ctypes.byref(reasons)) == 0
    mw = self._uint("nvmlDeviceGetPowerUsage")
    return {"t": time.monotonic(), "sm_mhz": self._uint("nvmlDeviceGetClockInfo", ctypes.c_int(_CLOCK_SM)),
            "power_w": mw / 1000.0 if mw is not None else None,
            "temp_c": self._uint("nvmlDeviceGetTemperature", ctypes.c_int(_TEMP_GPU)),
            "reasons": reasons.value if ok else None}

  def power_limit_w(self) -> float | None:
    mw = self._uint("nvmlDeviceGetEnforcedPowerLimit")
    return mw / 1000.0 if mw is not None else None

  def close(self) -> None:
    self.lib.nvmlShutdown()


def reason_words(bits:int) -> list[str]:
  return [w for b, w in THROTTLE_REASONS.items() if bits & b]


HELD = 0.98  # a throttle reason with the SM clock at or above this share of the run's highest clock held the clock


def summarize(samples:list[dict[str, Any]], *, power_limit_w:float | None = None, top_mhz:float | None = None) -> dict[str, Any]:
  """The samples as one record: SM clock mean, min and max (MHz), power mean and max, temperature max, and the
  throttle reasons seen with the share of samples each was set in. The driver sets a reason (a software power cap)
  while the clock stays where it was, so a sample counts as throttled only when its clock is also below HELD of the
  run's highest clock (top_mhz, else these samples' own): throttled is True when any sample was; held lists the
  reasons that were set with the clock held. Samples the driver marks idle (a program loading its model) are left
  out of every figure when the GPU worked at all; all_samples counts them."""
  every = samples
  samples = [s for s in every if not (s.get("reasons") or 0) & IDLE] or every  # the GPU at work, not loading a model
  clocks = [s["sm_mhz"] for s in samples if s.get("sm_mhz")]
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
         "power_w": {"mean": round(statistics.fmean(power), 1), "max": round(max(power), 1)} if power else None,
         "temp_c_max": max(temps) if temps else None, "power_limit_w": power_limit_w,
         "throttle": {w: round(n / len(samples), 3) for w, n in sorted(seen.items())},
         "held": {w: round(n / len(samples), 3) for w, n in sorted(held.items())}, "top_mhz": top,
         "throttled": bool(seen)}
  out["words"] = words(out)
  return out


def compact(samples:list[dict[str, Any]]) -> list[list[Any]]:
  """The raw samples kept with a measurement, so its summary can be read again: [seconds from the first, SM MHz,
  W, C, reason bits]."""
  t0 = samples[0]["t"] if samples else 0.0
  return [[round(s["t"] - t0, 3), s.get("sm_mhz"), round(s["power_w"], 1) if s.get("power_w") is not None else None,
           s.get("temp_c"), s.get("reasons")] for s in samples]


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
  else:
    text += "; not throttled"
  if s.get("held"):
    text += (" (" + ", ".join(f"{w} set in {100 * f:.0f}% of samples" for w, f in s["held"].items())
             + f" with the clock held within {100 * (1 - HELD):.0f}% of {s['top_mhz']:.0f} MHz)")
  return text


class Telemetry:
  """Samples the GPU in a background thread while the block runs. summary() after the block. On a machine without
  the library the summary says so and nothing is sampled."""

  def __init__(self, device:int = 0, interval_s:float = INTERVAL_S, *, source=None):
    self.device, self.interval_s = device, interval_s
    self.samples: list[dict[str, Any]] = []
    self.reason: str | None = None
    self._source, self._stop, self._thread, self._limit = source, threading.Event(), None, None

  def __enter__(self) -> "Telemetry":
    try:
      self._nvml = self._source or _Nvml(self.device)
    except OSError as exc:
      self._nvml, self.reason = None, f"no clock samples: {LIBRARY} not usable here ({exc})"
      return self
    self._limit = self._nvml.power_limit_w()
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
    return summarize(rows, power_limit_w=self._limit, top_mhz=max(clocks) if clocks else None)


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
