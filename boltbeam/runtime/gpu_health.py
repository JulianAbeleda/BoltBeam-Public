"""GPU health, two reads. gpu_health_snapshot: processes stuck on the GPU (a hung AMD reset, a dead holder of
/dev/kfd). check: an NVIDIA machine's memory faults and contexts, no DRAM row remap waiting for a reset, no failed
remap, no uncorrectable DRAM error since boot, and a CUDA context on every GPU. A RunPod H200 with remaps pending
passed every other check and then ran slow and crashed (2026-11-08, twice). The doctor reads check; tools/pod-setup.sh
runs the same checks in shell before BoltBeam is installed, and tests/test_gpu_health.py keeps both on one verdict."""
from __future__ import annotations

import ctypes
import os
import pathlib
import re
import shutil
import subprocess
from typing import Any


GPU_FD_MARKERS = ("/dev/kfd", "/dev/dri/", "/sys/devices/", "/sys/bus/pci/devices/")
GPU_CMD_MARKERS = ("amdgpu", "/dev/kfd", "/dev/dri", "renderD", "/sys/bus/pci/devices")
GPU_KERNEL_COMMS = ("ttm", "amdgpu", "kfd")


def _read_text(path:pathlib.Path) -> str:
  try:
    return path.read_text(encoding="utf-8", errors="replace").strip()
  except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
    return ""


def _proc_state(stat:str) -> str:
  # /proc/<pid>/stat format has comm in parentheses; state follows the closing paren.
  try:
    return stat.rsplit(")", 1)[1].strip().split()[0]
  except IndexError:
    return ""


def _cmdline(proc:pathlib.Path) -> str:
  raw = _read_text(proc / "cmdline")
  return raw.replace("\x00", " ").strip()


def _fd_targets(proc:pathlib.Path, limit:int = 256) -> list[str]:
  out:list[str] = []
  try:
    fds = list((proc / "fd").iterdir())[:limit]
  except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
    return out
  for fd in fds:
    try:
      out.append(os.readlink(fd))
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
      continue
  return out


def gpu_health_snapshot(*, proc_root:str | pathlib.Path = "/proc", dev_root:str | pathlib.Path = "/dev") -> dict[str, Any]:
  proc_root, dev_root = pathlib.Path(proc_root), pathlib.Path(dev_root)
  kfd_present = (dev_root / "kfd").exists()
  dri = dev_root / "dri"
  render_nodes = sorted(p.name for p in dri.glob("renderD*")) if dri.exists() else []

  d_state_tasks:list[dict[str, Any]] = []
  gpu_fd_holders:list[dict[str, Any]] = []
  for proc in proc_root.iterdir() if proc_root.exists() else []:
    if not proc.name.isdigit():
      continue
    stat = _read_text(proc / "stat")
    state = _proc_state(stat)
    comm = _read_text(proc / "comm")
    cmd = _cmdline(proc)
    fds = [target for target in _fd_targets(proc) if any(marker in target for marker in GPU_FD_MARKERS)]
    if fds:
      gpu_fd_holders.append({"pid": int(proc.name), "state": state, "comm": comm, "cmd": cmd, "fds": fds[:16]})
    if state == "D":
      text = f"{comm} {cmd} {' '.join(fds)}"
      if any(marker in text for marker in GPU_CMD_MARKERS) or any(comm.startswith(prefix) for prefix in GPU_KERNEL_COMMS):
        d_state_tasks.append({"pid": int(proc.name), "comm": comm, "cmd": cmd, "fds": fds[:16]})

  problems:list[str] = []
  if kfd_present and not render_nodes:
    problems.append("kfd_present_but_no_dri_render_node")
  if d_state_tasks:
    problems.append("gpu_related_d_state_tasks")
  if any(row.get("state") in ("D", "X", "Z") for row in gpu_fd_holders):
    problems.append("dead_or_blocked_gpu_fd_holders")

  return {
    "schema": "boltbeam.gpu_health.v1",
    "ok": not problems,
    "problems": problems,
    "devices": {"kfd_present": kfd_present, "render_nodes": render_nodes},
    "d_state_tasks": d_state_tasks,
    "gpu_fd_holders": gpu_fd_holders,
  }


def assert_gpu_health(mode:str = "warn", *, stage:str = "preflight") -> dict[str, Any]:
  if mode == "off":
    return {"schema": "boltbeam.gpu_health.v1", "ok": True, "problems": [], "stage": stage, "skipped": True}
  snap = gpu_health_snapshot()
  snap["stage"] = stage
  if not snap["ok"] and mode == "fail":
    raise RuntimeError(f"GPU health check failed at {stage}: {', '.join(snap['problems'])}")
  return snap


# --- NVIDIA memory faults ------------------------------------------------------------------------------------

QUERY = ["nvidia-smi", "-q", "-d", "ROW_REMAPPER,ECC"]


def faults(smi:str) -> list[str]:
  """The memory faults in `nvidia-smi -q -d ROW_REMAPPER,ECC` text, one phrase each; [] when none."""
  out = []
  remap = smi[smi.find("Remapped Rows"):] if "Remapped Rows" in smi else ""
  if re.search(r"Pending\s*:\s*Yes", remap):
    out.append("row remaps pending a GPU reset")
  if re.search(r"Remapping Failure Occurred\s*:\s*Yes", smi):
    out.append("a row remap failed")
  volatile = 0
  for block in re.findall(r"Volatile(.*?)(?:Aggregate|\Z)", smi, re.S):
    volatile += sum(int(n) for n in re.findall(r"DRAM Uncorrectable\s*:\s*(\d+)", block))
  if volatile:
    out.append(f"{volatile} uncorrectable DRAM errors since boot")
  return out


def contexts(lib:str = "libcuda.so.1") -> str | None:
  """Why a CUDA context does not open on some GPU, or None when every GPU opens one."""
  try:
    cu = ctypes.CDLL(lib)
  except OSError as exc:
    return f"the CUDA driver library does not load: {exc}"
  n = ctypes.c_int()
  if r := cu.cuInit(0):
    return f"cuInit failed: CUresult {r}"
  cu.cuDeviceGetCount(ctypes.byref(n))
  for i in range(n.value):
    d, ctx = ctypes.c_int(), ctypes.c_void_p()
    cu.cuDeviceGet(ctypes.byref(d), i)
    if r := cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), d):
      return f"GPU {i}: a CUDA context does not open (CUresult {r})"
    cu.cuDevicePrimaryCtxRelease(d)
  return None


def check() -> dict[str, Any]:
  """{"healthy": True/False/None, "problems": [...]}; None when nvidia-smi is absent (not checked, not passed)."""
  if not shutil.which(QUERY[0]):
    return {"healthy": None, "problems": ["nvidia-smi is not on PATH"]}
  try:
    smi = subprocess.run(QUERY, capture_output=True, text=True, timeout=60).stdout
  except (OSError, subprocess.SubprocessError) as exc:
    return {"healthy": None, "problems": [f"nvidia-smi did not answer: {exc}"]}
  problems = faults(smi)
  if why := contexts():
    problems.append(why)
  return {"healthy": not problems, "problems": problems}
