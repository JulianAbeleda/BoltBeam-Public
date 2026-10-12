"""Nsight Compute as the kernel timer's oracle: the cache check on every timed kernel, when the counters are there.

The timer (kernel_timer.py) never needs ncu. When ncu is on PATH, the GPU's counters are open to this user and
BOLTBEAM_NCU=1 is set, the generic run (engine_kernels.collect) runs the same timer once more under ncu and records,
per role, what the timed launches read from DRAM: `cache_check` on the row. Otherwise the row says "not checked" and
why.

    the timer, under ncu          one pass of METRICS per launch, --cache-control none (the timer's own sweep is the
                                  cache state measured), the free clock (profiler/ncu_counters.ncu_command); the
                                  adapted kernel, the sweep and the empty kernel, so a launch after the floor's
                                  sweep is not taken for a timed one
    the launches that follow a    the timed samples
    sweep kernel
    per role                      ncu's own duration, the DRAM bytes read, the weight bytes: checked when the DRAM
                                  bytes are the weight bytes (CACHE_SLACK)

The metrics are kept to one pass: with more, ncu replays the kernel and the later passes find the L2 the first pass
filled, so no L2 counter is asked for. The L2 hit share is derived: 1 less the DRAM bytes over the bytes the kernel
reads (weights and warm activations).
"""
from __future__ import annotations

import csv
import io
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
from typing import Any

from boltbeam.collectors import kernel_timer

METRICS = ("gpu__time_duration.sum", "dram__bytes_op_read.sum", "launch__grid_size")  # one pass on an RTX 5090
CHILD_ENV = "BOLTBEAM_ORACLE_CHILD"  # set in the timer run under ncu, so it does not start a third
CACHE_SLACK = 0.02  # the weights count as read from DRAM when the DRAM bytes are within 2% of them
CHECKED = "checked by Nsight Compute: the timed launches' DRAM bytes against the weight bytes"
_SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}  # ncu's byte prefixes are decimal


def unavailable(backend:str) -> str | None:
  """Why the check cannot run here, or None when it can."""
  from boltbeam.collectors.instruction_path import NCU_ENV, counters_open
  if os.environ.get(CHILD_ENV):
    return "this is the run under ncu"
  if backend != "CUDA":
    return f"Nsight Compute reads NVIDIA counters; this is {backend}"
  if os.environ.get(NCU_ENV) != "1":
    return f"not asked: set {NCU_ENV}=1 to check the cache state with Nsight Compute"
  if not shutil.which("ncu"):
    return "ncu is not on PATH"
  if not counters_open():
    return "the GPU's performance counters are closed to this user (RmProfilingAdminOnly is not 0)"
  return None


def read_launches(text:str) -> list[dict[str, Any]]:
  """`ncu --page raw --csv` with METRICS: one {name, grid, us, dram_bytes} per profiled launch, in launch order."""
  rows = list(csv.DictReader(io.StringIO(text)))
  if not rows:
    return []
  units, out = rows[0], []
  for r in rows[1:]:
    num = lambda k: float(str(r[k]).replace(",", "") or 0)  # noqa: E731
    out.append({"name": r["Kernel Name"], "grid": int(num("launch__grid_size")), "us": num("gpu__time_duration.sum"),
                "dram_bytes": num("dram__bytes_op_read.sum") * _SCALE.get(units["dram__bytes_op_read.sum"], 1)})
  return out


def _is(name:str, kernel:str) -> bool:
  return name.split("(")[0].split("<")[0].split()[-1] == kernel


def timed_groups(launches:list[dict[str, Any]], adapted:str, sweep:str) -> list[list[dict[str, Any]]]:
  """The timer's launches of the adapted kernel, one list per role in order, of the launches that came straight after
  a sweep (the timed samples). A role starts at a launch that did not follow a sweep, after a timed one."""
  groups:list[list[dict[str, Any]]] = []
  prev, timed_last = None, False
  for x in launches:
    if adapted in x["name"]:
      timed = prev == "sweep"
      if not groups or (not timed and timed_last):
        groups.append([])
      if timed:
        groups[-1].append(x)
      timed_last = timed
    prev = "sweep" if _is(x["name"], sweep) else x["name"]
  return [g for g in groups if g]


def check(group:list[dict[str, Any]], row:dict[str, Any]) -> dict[str, Any]:
  """The cache check of one role from its timed launches under ncu."""
  dram = statistics.median(x["dram_bytes"] for x in group)
  weights = float(row["bytes"])
  warm = float(((row.get("timing") or {}).get("warm") or [None, 0])[1])
  ok = abs(dram - weights) <= CACHE_SLACK * weights
  return {"status": "checked", "words": CHECKED, "launches": len(group), "ncu_us": statistics.median(x["us"] for x in group),
          "dram_bytes": dram, "weight_bytes": weights, "dram_over_weights": dram / weights if weights else None,
          "l2_hit_pct": max(0.0, 100.0 * (1 - dram / (weights + warm))) if weights + warm else None, "passed": ok,
          "reason": None if ok else f"the timed launches read {dram / 1e6:.2f} MB from DRAM against {weights / 1e6:.2f} MB of weights"}


def not_checked(reason:str) -> dict[str, Any]:
  return {"status": "not checked", "reason": reason}


def attach(rows:list[dict[str, Any]], launches:list[dict[str, Any]], adapted:str, sweep:str) -> None:
  """Each measured row's cache_check from the launches of the same timer run under ncu, matched in order and by grid."""
  measured = [r for r in rows if r.get("status") == "measured"]
  groups = timed_groups(launches, adapted, sweep)
  if len(groups) != len(measured):
    for r in measured:
      r["cache_check"] = not_checked(f"the run under ncu timed {len(groups)} roles, this run {len(measured)}")
    return
  for r, g in zip(measured, groups):
    grid = ((r.get("geometry") or {}).get("threadgroups"))
    r["cache_check"] = check(g, r) if grid in (None, g[0]["grid"]) else not_checked(
      f"the launch under ncu has {g[0]['grid']} blocks, this role {grid}")


def cache_check(run:pathlib.Path, provider:str, adapted:str, backend:str, rows:list[dict[str, Any]]) -> None:
  """Run the timer once more under ncu and attach a cache_check to every measured row; "not checked" and why when
  the check cannot run or fails to."""
  from boltbeam.profiler.ncu_counters import export_raw_csv, ncu_command
  why = unavailable(backend)
  if why is None:
    out = run / "ncu-cache-check"
    out.mkdir(exist_ok=True)
    sweep, tool = kernel_timer.FLUSH[backend]["kernel"], shutil.which("ncu") or "ncu"
    argv = ncu_command([sys.executable, "-m", "boltbeam.collectors.kernel_oracle", str(run), provider], out=str(out / "timer"),
                       ncu=tool, sections=(), metrics=METRICS, cache_control="none",
                       profile_from_start="on", kernel_name=f"regex:{adapted}|{sweep}|{kernel_timer.FLUSH[backend]['empty']}")
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=7200, env={**os.environ, CHILD_ENV: "1"})
    (out / "ncu.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode == 0 and (out / "timer.ncu-rep").exists():
      attach(rows, read_launches(export_raw_csv(out / "timer.ncu-rep", ncu=tool)), adapted, sweep)
      return
    why = f"ncu exited {proc.returncode}; see {out / 'ncu.log'}"
  for r in rows:
    if r.get("status") == "measured":
      r["cache_check"] = not_checked(why)


if __name__ == "__main__":  # the timer run under ncu: the same roles, its trace kept apart from the run's own
  from boltbeam.collectors import engine_kernels
  engine_kernels.collect(pathlib.Path(sys.argv[1]), sys.argv[2], out="ncu-cache-check/timer_trace.json")
