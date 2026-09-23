from __future__ import annotations

import os
import pathlib
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
