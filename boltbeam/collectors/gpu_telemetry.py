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

import contextlib
import ctypes
import math
import os
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

  def __init__(self, device:int | str):
    """By NVML index, or by PCI address (a hardware-profile GPU's pci_bus_id), as AmdSysfs finds its card."""
    self.lib = ctypes.CDLL(LIBRARY)
    if self.lib.nvmlInit_v2() != 0:
      raise OSError("nvmlInit failed")
    self.handle, self.device = ctypes.c_void_p(), device
    got = (self.lib.nvmlDeviceGetHandleByPciBusId_v2(device.encode(), ctypes.byref(self.handle)) if isinstance(device, str)
           else self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(device), ctypes.byref(self.handle)))
    if got != 0:
      raise OSError(f"no NVML device {device}")

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


class AppleState:
  """What macOS lets a user process read about the machine while work runs, with no root and no subprocess: the
  thermal state (NSProcessInfo.thermalState, through the Objective-C runtime), the memory pressure level and the
  swap in use (sysctl). macOS gives a user process no GPU clock, power or temperature; where this user has
  passwordless sudo, start() and stop() run the shared Powermetrics witness for the GPU clock and power, else they stay
  unknown. No NVIDIA field (sm_mhz, throttle reasons) is filled. pmset -g therm tells the same thermal story only once
  a warning is recorded, at a process per read."""

  THERMAL = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}  # NSProcessInfoThermalState; serious: macOS slows the chip
  PRESSURE = {1: "normal", 2: "warn", 4: "critical"}  # kern.memorystatus_vm_pressure_level

  def __init__(self, device:dict[str, Any] | None = None):
    self.libc, objc = ctypes.CDLL(None), ctypes.CDLL("/usr/lib/libobjc.A.dylib")  # OSError off macOS: Unsampled
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")  # NSProcessInfo's class
    objc.objc_getClass.restype = objc.sel_registerName.restype = ctypes.c_void_p
    send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    self._long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(send)
    self._info = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)(
      objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
    self._thermal = objc.sel_registerName(b"thermalState")

  def _sysctl(self, name:bytes, buf):
    size = ctypes.c_size_t(ctypes.sizeof(buf))
    return buf if self.libc.sysctlbyname(name, ctypes.byref(buf), ctypes.byref(size), None, ctypes.c_size_t(0)) == 0 else None

  def sample(self) -> dict[str, Any]:
    level = self._sysctl(b"kern.memorystatus_vm_pressure_level", ctypes.c_int())
    swap = self._sysctl(b"vm.swapusage", (ctypes.c_uint64 * 4)())  # xsw_usage: total, avail, used, page size + flag
    return {"t": time.monotonic(), "thermal_state": self.THERMAL.get(t := self._long(self._info, self._thermal), str(t)),
            "memory_pressure": self.PRESSURE.get(level.value, str(level.value)) if level else None,
            "swap_used_mib": round(swap[2] / 2**20, 1) if swap else None}

  def start(self) -> None:
    self.__dict__.pop("_rows", None)  # a source opened again starts a new window
    self._t0, self._used = time.monotonic(), WITNESS.acquire()

  def stop(self) -> None:
    WITNESS.release()  # the window's samples, kept past the witness's next start
    self._rows, self._reason = [r for r in list(WITNESS.rows) if self._t0 < r[0] <= time.monotonic()], WITNESS.reason

  def witness(self, t0:float | None = None, t1:float | None = None) -> dict[str, Any] | None:
    """The witness's samples in the window as [s from start, MHz, active share, W, s covered]; None without sudo."""
    if not getattr(self, "_used", False):
      return None
    got, reason = (self._rows, self._reason) if hasattr(self, "_rows") else (  # a summary while the block still runs
      [r for r in list(WITNESS.rows) if self._t0 < r[0]], WITNESS.reason)
    rows = [[round(r[0] - self._t0, 3), *r[1:]] for r in got if t0 is None or t1 is None or t0 <= r[0] <= t1]
    out = {"source": "powermetrics (sudo -n)", "interval_ms": WITNESS_MS, "samples": rows}
    return {**out, "reason": reason} if rows and reason else out if rows else {**out, "reason": reason or "powermetrics gave no sample in the window"}


WITNESS_MS, WITNESS_CAP, BUSY = 200, 108_000, 0.5  # 6 h of samples: an orphan stops by itself; BUSY: share active


_SUDO: dict[Any, bool] = {}


def sudo_n(run=None) -> bool:
  """Whether this user has passwordless sudo, asked once per process in its own session as the witness runs (a tty's
  cached sudo ticket does not count); never prompts."""
  if run not in _SUDO:
    try:
      _SUDO[run] = (run or subprocess.run)(("sudo", "-n", "true"), capture_output=True, timeout=10, check=False,
                                           start_new_session=True).returncode == 0
    except (OSError, subprocess.SubprocessError):
      _SUDO[run] = False
  return _SUDO[run]


class Powermetrics:
  """The GPU clock and power macOS gives only root, from powermetrics under sudo -n (never a password prompt; no
  passwordless sudo, no witness). One process for every open AppleState, so a capture inside a measurement samples
  once. Its own session (no tty, so sudo makes no pty), it writes into our pipe: if we die, its next write fails and
  it exits; WITNESS_CAP bounds it besides. A failure is a reason, never raised. A sample: gpu.freq_hz is MHz, the
  frequency weighted by the time at each DVFS state while active; 1 - idle_ratio the share active; gpu_energy mJ."""

  ARGV = ("sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-i", str(WITNESS_MS), "-f", "plist", "-n", str(WITNESS_CAP))

  def __init__(self, run=None, popen=subprocess.Popen):
    self.run, self.popen, self.lock, self.users, self.proc, self.rows, self.reason, self.tried = run, popen, threading.Lock(), 0, None, [], None, False

  @staticmethod
  def sample(raw:bytes, t:float) -> list[float]:
    import plistlib
    d = plistlib.loads(raw.strip())
    gpu, s = d["gpu"], d["elapsed_ns"] / 1e9
    return [t, round(float(gpu["freq_hz"]), 1), round(1 - float(gpu["idle_ratio"]), 4), round(gpu["gpu_energy"] / 1e3 / s, 3), round(s, 4)]

  def acquire(self) -> bool:
    """Start the witness for the first user; True when this user has passwordless sudo (it was tried)."""
    with self.lock:
      self.users += 1
      if self.users == 1:
        self.rows, self.reason, self.tried = [], None, sudo_n(self.run)
        try:
          if self.tried:
            self.proc = self.popen(self.ARGV, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            (thread := threading.Thread(target=self._read, args=(self.proc.stdout, self.rows), daemon=True)).start()
            self.thread = thread
        except (OSError, subprocess.SubprocessError) as exc:
          self.reason = f"powermetrics did not start: {exc}"
      return self.tried

  def _read(self, out, rows:list) -> None:  # its own process's rows: a stale reader never writes the next one's
    buf = b""  # a NUL comes before each sample after the first; a sample is whole at its </plist>
    while chunk := self._chunk(out):
      parts = (buf + chunk).split(b"\0")
      buf = b"" if parts[-1].rstrip().endswith(b"</plist>") else parts.pop()
      for part in filter(bytes.strip, parts):
        try:
          rows.append(self.sample(part, time.monotonic()))
        except Exception as exc:  # plistlib raises several kinds; a bad sample is a reason, not a crash
          self.reason = f"a powermetrics sample did not parse: {type(exc).__name__}: {exc}"

  @staticmethod
  def _chunk(out) -> bytes:
    try:
      return out.read1(1 << 16)
    except (OSError, ValueError):  # the pipe closed under a reader still waiting
      return b""

  def release(self) -> None:
    with self.lock:
      self.users = max(self.users - 1, 0)
      if self.users or self.proc is None:
        return
      proc, self.proc = self.proc, None
    proc.terminate()  # sudo, ours to signal, passes SIGTERM on; powermetrics stops sampling and exits
    try:
      proc.wait(5)
    except subprocess.TimeoutExpired:
      proc.kill(), proc.wait()  # sudo only: powermetrics dies at its next write into the pipe closed below
    self.thread.join(2)
    err = proc.stderr.read().decode(errors="replace").strip()
    proc.stdout.close(), proc.stderr.close()
    if proc.returncode not in (0, -15) and not self.reason:
      self.reason = f"powermetrics exited {proc.returncode}" + (f": {err.splitlines()[-1]}" if err else "")


WITNESS = Powermetrics()


def apple_clock_policy(source) -> dict[str, Any]:
  """macOS chooses the GPU clock and gives no process a way to set it, nor a user process a way to read it: the OS's
  policy, the clock said to be unwitnessed, or read through sudo (Powermetrics); never NVIDIA's words."""
  read = sudo_n()
  return {"policy": "os_managed", "clock_settable": False, "words": POLICY_WORDS["os_managed_read" if read else "os_managed"],
          "why": "macOS exposes no GPU clock control; " + ("BoltBeam reads the clock through powermetrics under sudo -n" if read
                 else "powermetrics, the only clock reader, needs root")}


UNWITNESSED = "GPU clock, power and temperature not witnessed (macOS reads them only through powermetrics, which needs root)"


def witness_clock(w:dict[str, Any] | None) -> tuple[float, float, float] | None:
  """A witness's GPU clock (mean weighted by active time, min, max) over its samples with the GPU at least BUSY
  active (all of them when none is); None with no samples."""
  if not (rows := [r for r in (w or {}).get("samples") or [] if r[4] > 0]):
    return None
  use = [r for r in rows if r[2] >= BUSY] or rows
  active = sum(r[2] * r[4] for r in use)
  mhz = sum(r[1] * r[2] * r[4] for r in use) / active if active else statistics.fmean(r[1] for r in use)
  return mhz, min(r[1] for r in use), max(r[1] for r in use)


def settle(run:pathlib.Path, say=lambda _: None, sleep=time.sleep) -> float:
  """Idle the run's GPU for its vendor row's settle_s before a measured arm (an untraced prefill or turn, or its
  capture), so every arm starts from the same rested clock; the seconds idled, for the arm's record (0: none)."""
  from boltbeam.target.vendors import NoVendorRow, vendor
  from boltbeam.workflow.common import run_gpu
  try:
    s = float(vendor((run_gpu(run) or {}).get("vendor")).settle_s)
  except NoVendorRow:
    return 0.0
  if s > 0:
    say(f"the GPU idles {s:g} s first, as before every measured arm")
    sleep(s)
  return s


def witness_words(w:dict[str, Any] | None) -> str:
  """The GPU clock and power a Powermetrics witness saw, or that none was: disclosure, not a gate."""
  rows = [r for r in (w or {}).get("samples") or [] if r[4] > 0]
  if not rows:
    return UNWITNESSED + (f"; powermetrics under sudo -n gave none: {w['reason']}" if (w or {}).get("reason") else "")
  of = f" in {w['witnessed']} of {w['of']} captures" if w.get("of", 0) > w.get("witnessed", 0) else ""
  use = (busy := [r for r in rows if r[2] >= BUSY]) or rows
  mhz, lo, hi = witness_clock(w)
  watts = sum(r[3] * r[4] for r in use) / sum(r[4] for r in use)
  which = (f"the {len(busy)} of {len(rows)} samples with the GPU at least {BUSY:.0%} active" if busy
           else f"all {len(rows)} samples (none had the GPU {BUSY:.0%} active)")
  return (f"GPU clock {mhz:.0f} MHz mean ({lo:.0f}-{hi:.0f}), GPU power "
          f"{watts:.1f} W mean ({max(r[3] for r in use):.1f} W max) over {which}, GPU active "
          f"{100 * sum(r[2] * r[4] for r in rows) / sum(r[4] for r in rows):.0f}% of the window, from powermetrics under "
          f"sudo -n every {w.get('interval_ms', WITNESS_MS)} ms{of}" + (f" ({w['reason']})" if w.get("reason") else "") + "; the clock is the frequency weighted by the GPU's "
          "active time, which macOS chooses, not a fixed clock; GPU temperature not witnessed (powermetrics gives none)")


def state_words(samples:list[dict[str, Any]], witness:dict[str, Any] | None = None) -> str:
  """A state source's samples (AppleState) as words, each state with its share of the samples, and what its witness
  saw: disclosure, not a gate."""
  share = lambda key: ", ".join(f"{k} {100 * sum(s.get(key) == k for s in samples) / len(samples):.0f}%"
                                for k in sorted({s[key] for s in samples if s.get(key)})) or "unread"
  swap = [s["swap_used_mib"] for s in samples if s.get("swap_used_mib") is not None]
  return (f"device state over {len(samples)} samples: thermal state {share('thermal_state')}, memory pressure "
          f"{share('memory_pressure')}" + (f", swap {min(swap):.0f}-{max(swap):.0f} MiB used" if swap else "")
          + "; " + witness_words(witness))


def compact_state_words(rows:list[list[Any]], witness:dict[str, Any] | None) -> str:
  """state_words from compact rows (the state's columns from the seventh on) and a summary's gpu_witness."""
  return state_words([dict(zip(("thermal_state", "memory_pressure", "swap_used_mib"), r[6:])) for r in rows], witness)


def run_state_words(run:pathlib.Path) -> str | None:
  """The device state through every capture of a run (the compact samples and the witness in the telemetry.json
  beside each), as words; None when no capture's source reports a state (a clock source's words stay with each)."""
  import json
  def state(p):  # compact rows with a state's columns only; another writer's raw samples or a bad file say nothing
    try:
      d = json.loads(p.read_text())
      return [r for r in d.get("samples") or [] if isinstance(r, list) and len(r) > 6], (d.get("summary") or {}).get("gpu_witness")
    except (ValueError, AttributeError, OSError): return [], None
  files = [state(p) for p in sorted(pathlib.Path(run).rglob("telemetry.json"))]
  seen = [w for f, w in files if f and isinstance(w, dict)]
  witness = {"samples": [r for w in seen for r in w.get("samples") or []], "interval_ms": WITNESS_MS, "of": sum(bool(f) for f, _ in files),
             "witnessed": sum(bool(w.get("samples")) for w in seen), "reason": next((w["reason"] for w in seen if w.get("reason")), None)} if seen else None
  rows = [r for f, _ in files for r in f]
  return f"during its {sum(bool(f) for f, _ in files)} captures, {compact_state_words(rows, witness)}" if rows else None


def derived_state_words(run:pathlib.Path) -> list[str]:
  """Every device-state line the run's own samples derive (its captures', and each untraced row's in a top-level
  trace), so a report's line can be checked against them."""
  import json
  out = [w] if (w := run_state_words(run)) else []
  for p in sorted(pathlib.Path(run).glob("*.json")):
    try: rows = json.loads(p.read_text()).get("rows") or []
    except (ValueError, AttributeError, OSError): continue
    out += [compact_state_words([x for x in r.get("telemetry_samples") or [] if isinstance(x, list)], r["telemetry"].get("gpu_witness")) for r in rows
            if isinstance(r, dict) and isinstance(r.get("telemetry"), dict) and r["telemetry"].get("state")]
  return out


def reason_words(bits:int) -> list[str]:
  return [w for b, w in THROTTLE_REASONS.items() if bits & b]


HELD = 0.98  # a throttle reason with the SM clock at or above this share of the run's highest clock held the clock


def summarize(samples:list[dict[str, Any]], *, power_limit_w:float | None = None, top_mhz:float | None = None,
              memory_total:float | None = None, witness:dict[str, Any] | None = None,
              clock_term:str | None = None) -> dict[str, Any]:
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
  if not clocks and any("thermal_state" in s for s in every):  # AppleState: the machine's state, not a clock
    out["state"] = True
    if witness is not None:
      out["gpu_witness"] = witness
  if clock_term:  # the vendor's word for the clock sampled (its row's core_clock term)
    out["clock_term"] = clock_term
  out["words"] = state_words(every, witness) if out.get("state") else words(out)
  return out


def compact(samples:list[dict[str, Any]]) -> list[list[Any]]:
  """The raw samples kept with a measurement, so its summary can be read again: [seconds from the first, SM MHz,
  W, C, reason bits, memory MHz], then a state source's thermal state, memory pressure and swap MiB."""
  t0 = samples[0]["t"] if samples else 0.0
  return [[round(s["t"] - t0, 3), s.get("sm_mhz"), round(s["power_w"], 1) if s.get("power_w") is not None else None,
           s.get("temp_c"), s.get("reasons"), s.get("mem_mhz"),
           *([s["thermal_state"], s.get("memory_pressure"), s.get("swap_used_mib")] if "thermal_state" in s else [])]
          for s in samples]


def words(s:dict[str, Any]) -> str:
  if not s.get("sm_mhz"):
    return s.get("reason") or "no clock samples"
  c = s["sm_mhz"]
  text = f"{s.get('clock_term') or 'GPU clock'} {c['mean']:.0f} MHz mean ({c['min']}-{c['max']}) over {s['samples']} busy samples"
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
  """The telemetry source for one hardware-profile GPU, from its vendor's row (chips._vendors() "telemetry"), found
  by its PCI address: NVML for an NVIDIA card, the amdgpu sysfs for an AMD one. A vendor with no source, or a card not found, is Unsampled with the reason. No
  device (a run with no hardware profile) is None, as before the profile named the GPU."""
  if not device:
    return None
  from boltbeam.target.vendors import vendor
  from boltbeam.workflow.chips import NoVendorRow, vendor_row
  try:
    source = vendor_row(device.get("vendor"), "telemetry")(device)
    source.clock_term = vendor(device.get("vendor")).term("core_clock")
    return source
  except (NoVendorRow, OSError, KeyError, AttributeError, ctypes.ArgumentError) as exc:
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
    self._limit = getattr(self._nvml, "power_limit_w", lambda: None)()
    self._total = getattr(self._nvml, "memory_mib", lambda: (None, None))()[1]
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()
    getattr(self._nvml, "start", lambda: None)()  # a source's own witness for the block (AppleState's Powermetrics)
    return self

  def _loop(self) -> None:
    while not self._stop.is_set():
      self.samples.append(self._nvml.sample())
      self._stop.wait(self.interval_s)

  def __exit__(self, *exc) -> None:
    self._stop.set()
    if self._thread is not None:
      try:
        self._thread.join()
        self.samples.append(self._nvml.sample())
      finally:
        getattr(self._nvml, "stop", lambda: None)()  # a root witness never outlives its block
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
    return summarize(rows, power_limit_w=self._limit, top_mhz=max(clocks) if clocks else None, memory_total=self._total,
                     witness=self.witness(t0, t1), clock_term=getattr(self._nvml, "clock_term", None))

  def witness(self, t0:float | None = None, t1:float | None = None) -> dict[str, Any] | None:
    """The source's own witness in the window (AppleState's Powermetrics); None for a source with none."""
    return getattr(self._nvml, "witness", lambda *w: None)(t0, t1)


# --- the clock policy ----------------------------------------------------------------------------------------------

POLICY_WORDS = {
  "fixed_clock": "fixed clock: the SM clock is locked to the chip's maximum for the run (code and architecture comparison)",
  "fixed_power": ("fixed power: the clock cannot be set by this user, so the board runs at its own power limit, "
                  "unchanged, and the clock it chose is logged (realism; a rate is quoted with its clock)"),
  "os_managed": ("OS-managed clock: the OS chooses the GPU clock and a user process can neither set nor read it, so "
                 "the clock a rate ran at is not witnessed"),
  "os_managed_read": ("OS-managed clock: the OS chooses the GPU clock and no process can set it; BoltBeam reads the "
                      "clock it ran at through powermetrics under sudo -n, so a rate is quoted with it"),
}


def clock_policy(device:int | str = 0, *, run=subprocess.run) -> dict[str, Any]:
  """Which policy a measurement here can hold, decided by asking the driver tool for each without changing anything:
  a clock reset (-rgc) is allowed only to a user who may set clocks. Fixed clock when it is; else fixed power (the
  board's own limit, which no run changes); the record says which and why."""
  if requested := os.environ.get("BOLTBEAM_TIMING_SM_MHZ"):
    mhz = int(requested)
    if mhz <= 0:
      raise ValueError("BOLTBEAM_TIMING_SM_MHZ needs a positive SM clock")
    return {"policy": "external_request", "requested_sm_mhz": mhz, "clock_settable": None,
            "words": f"externally requested SM clock {mhz} MHz; observations recorded separately"}
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


@contextlib.contextmanager
def measurement_clock(run:pathlib.Path, phase:str, *, backend:str):
  """Witness an externally controlled clock in both pipeline arms; never change device settings."""
  if phase not in ("measure_timing", "role_time"):
    yield
    return
  from boltbeam.workflow.common import read_json, write_json
  path = run / "measurement_clock.json"
  requested = os.environ.get("BOLTBEAM_TIMING_SM_MHZ")
  if not requested:
    if path.exists():
      if phase == "measure_timing":
        path.unlink()  # a new native arm starts a fresh uncontrolled measurement
      else:
        write_json(path, {"schema": "boltbeam.measurement_clock.v1", "phases": {}})
    yield
    return
  mhz = int(requested)
  if backend != "CUDA" or mhz <= 0:
    raise ValueError("BOLTBEAM_TIMING_SM_MHZ needs a positive CUDA SM clock")
  record = read_json(path) if path.exists() else {}
  if record.get("requested_sm_mhz") != mhz or phase == "measure_timing":
    record = {"schema": "boltbeam.measurement_clock.v1", "requested_sm_mhz": mhz, "phases": {}}
  key = {"measure_timing": "native", "role_time": "captured"}[phase]
  record["phases"][key] = {"complete": False}
  write_json(path, record)
  complete = False
  telemetry = run_telemetry(run)
  try:
    with telemetry:
      yield
    complete = True
  finally:
    record["phases"][key] = {"complete": complete, "summary": telemetry.summary(),
                             "samples": compact(telemetry.samples)}
    write_json(path, record)


def measurement_clock_scope(run:pathlib.Path) -> dict | None:
  """A frequency regime after both observed peaks agree within 2%; not proof of a driver lock."""
  from boltbeam.workflow.common import read_json
  path = run / "measurement_clock.json"
  if not path.exists():
    return None
  try:
    record = read_json(path)
    requested = record["requested_sm_mhz"]
    if record["schema"] != "boltbeam.measurement_clock.v1":
      raise ValueError("unknown clock witness schema")
    if not isinstance(requested, (int, float)) or isinstance(requested, bool) or not math.isfinite(requested) or requested <= 0:
      raise ValueError("invalid requested clock")
    peaks = []
    for key in ("native", "captured"):
      phase = record["phases"][key]
      if phase["complete"] is not True:
        raise ValueError("incomplete clock phase")
      clocks = [s[1] for s in phase["samples"] if isinstance(s[1], (int, float)) and not isinstance(s[1], bool)
                and math.isfinite(s[1]) and s[1] > 0]
      if not clocks:
        raise ValueError("missing clock samples")
      peaks.append(max(clocks))
    if not all(abs(p / requested - 1) <= .02 for p in peaks) or abs(peaks[0] / peaks[1] - 1) > .02:
      raise ValueError("observed clocks differ from the declared regime")
  except (KeyError, TypeError, ValueError, IndexError) as exc:
    raise ValueError("invalid or incomplete measurement clock witness") from exc
  return {"requested_sm_mhz": requested, "observed_peak_tolerance_pct": 2}


def measurement_clock_words(run:pathlib.Path) -> str | None:
  """Disclose requested and observed frequency separately wherever timing qualification is presented."""
  if not (run / "measurement_clock.json").exists():
    return None
  from boltbeam.workflow.common import read_json
  try:
    scope = measurement_clock_scope(run)
  except (ValueError, TypeError, KeyError):
    return "Timing scope: unqualified; clock observations are incomplete or differ from the requested setup."
  record = read_json(run / "measurement_clock.json")
  peaks = {key: max(s[1] for s in record["phases"][key]["samples"] if isinstance(s[1], (int, float))
                   and not isinstance(s[1], bool) and math.isfinite(s[1]) and s[1] > 0)
           for key in ("native", "captured")}
  return (f"Timing scope: SM clock requested externally at {scope['requested_sm_mhz']:g} MHz; "
          f"observed peaks native {peaks['native']:g} MHz, captured {peaks['captured']:g} MHz. "
          "Timing bands apply to this clock setup.")
