"""The GPU kernel timeline of any provider's decode, captured from outside by the GPU vendor's own tool.

One row per GPU vendor (by the registry target's backend). Each row knows how to find its tool, how to wrap a
provider's command, and how to read the launches back as {"key", "wall_us"}. The key is the kernel name plus its
launch geometry when the tool reports it; attribution.py treats it as opaque. Nothing here knows a provider.

    CUDA   nsys (Nsight Systems)  `nsys profile -t cuda`, then `nsys stats --report cuda_gpu_trace` as CSV
    AMD    rocprofv3              `rocprofv3 --kernel-trace --output-format csv`
    Metal  xctrace                Metal System Trace; needs Xcode (found with `xcrun --find xctrace` at run
                                  time). Its export is read with metal_system_trace.py. It reports command
                                  buffers, not dispatches, so it splits a runtime's time per kernel only when
                                  the runtime puts one kernel in each command buffer.

`plan(backend)` says which method applies here, or why none does. A provider whose GPU work the vendor tool
cannot see (tinygrad's NV and AMD backends talk to the driver directly, so CUPTI and rocprofv3 see 0 kernels:
collectors/hw_trace.py) uses its own timing instead; that choice is the provider adapter's, not this module's.
"""
from __future__ import annotations

import csv
import io
import os
import re
import pathlib
import shutil
import subprocess
from typing import Any, Callable

from boltbeam.target import toolkits, vendors


TELEMETRY = "telemetry.json"  # the GPU's clocks, power and temperature through one capture, beside it


class NoCapture(RuntimeError):
  """No outside view of the GPU here. The message says why, in one sentence."""


def _nsys_argv(tool:str, argv:list[str], out:pathlib.Path, env:dict[str, str], flush_ms:int | None = None,
               profiler_range:bool = False) -> list[str]:
  # llama.cpp runs its decode as CUDA graphs. By default nsys reports a graph launch as one item with no
  # kernels; node tracing reports every kernel inside the graph, with its grid.
  # flush_ms: a GPU process that its parent stops with a signal (a server's runner) never flushes CUPTI's buffers
  # at exit; flushing on a timer keeps its kernels.
  # profiler_range: the program marks its measured window itself (cudaProfilerStart/Stop); only that is captured
  flush = [f"--cuda-flush-interval={flush_ms}"] if flush_ms else []
  window = ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"] if profiler_range else []
  return [tool, "profile", "-t", "cuda,nvtx", "--cuda-graph-trace=node", *flush, *window, "--force-overwrite", "true",
          "-o", str(out / "capture"), *argv]


def _nsys_read(tool:str, out:pathlib.Path) -> list[dict[str, Any]]:
  # --force-export: a capture again in the same folder must not be read through the last capture's sqlite export
  proc = subprocess.run([tool, "stats", "--force-export=true", "--report", "cuda_gpu_trace", "--format", "csv", "--output", "-",
                         str(out / "capture.nsys-rep")], capture_output=True, text=True, timeout=3600)
  if proc.returncode != 0:
    raise RuntimeError(f"nsys stats exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
  (out / "cuda_gpu_trace.csv").write_text(proc.stdout)  # the evidence the launches were read from
  return read_nsys_csv(proc.stdout)


def busy_window(out:pathlib.Path, launches:list[dict[str, Any]]) -> list[float] | None:
  """The time.monotonic() span from the first captured kernel's start to the last one's end: the GPU work, without
  the program's load. nsys keeps its session start on the system's monotonic clock (systemClockNs) and the launch
  starts relative to it; None when the export does not hold it (another tool, an older nsys)."""
  import sqlite3
  db = pathlib.Path(out) / "capture.sqlite"
  if not launches or not db.is_file():
    return None
  try:
    with sqlite3.connect(db) as con:
      row = con.execute("SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()
  except sqlite3.Error:
    return None
  if not row or not row[0]:
    return None
  zero = float(row[0]) / 1e9
  return [zero + min(l["start_us"] for l in launches) / 1e6, zero + max(l["start_us"] + l["wall_us"] for l in launches) / 1e6]


def read_nsys_csv(text:str) -> list[dict[str, Any]]:
  """Kernel launches from `nsys stats --report cuda_gpu_trace` CSV. Memory copies have no grid and are skipped."""
  body = text[text.find("Start"):] if "Start" in text else text
  out = []
  for row in csv.DictReader(io.StringIO(body)):
    grid = [row.get(k) for k in ("GrdX", "GrdY", "GrdZ")]
    if not all(grid) or not row.get("Duration (ns)"):
      continue
    block = [row.get(k) for k in ("BlkX", "BlkY", "BlkZ")]
    out.append({"key": f"{row['Name']} grid={'x'.join(grid)} block={'x'.join(b or '' for b in block)}",
                "wall_us": float(row["Duration (ns)"]) / 1000.0, "start_us": float(row["Start (ns)"]) / 1000.0,
                **({"stream": row["Strm"]} if row.get("Strm") else {})})
  return out


def sequence(launches:list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  """The launches in the order they ran, per stream (one stream when the capture does not name them): the one
  ordered view of a capture. A launch without a start time has no place in it and is left out."""
  out: dict[str, list[dict[str, Any]]] = {}
  for l in sorted((l for l in launches if l.get("start_us") is not None), key=lambda l: l["start_us"]):
    out.setdefault(str(l.get("stream", "")), []).append(l)
  return out


def _rocprof_argv(tool:str, argv:list[str], out:pathlib.Path, env:dict[str, str], flush_ms:int | None = None,
                  profiler_range:bool = False) -> list[str]:
  # profiler_range: the program opens and closes its window with roctx (runtime/vllm_semantics.hip_profiler_window);
  # its markers, HIP calls and kernels go to a rocpd database, read as Nsight's tables (rocpd_trace.py)
  if profiler_range:
    return [tool, "--kernel-trace", "--marker-trace", "--hip-runtime-trace", "--selected-regions", "--output-format",
            "rocpd", "-d", str(out), "-o", "capture_%pid%", "--", *argv]  # one database per traced process
  return [tool, "--kernel-trace", "--output-format", "csv", "-d", str(out), "-o", "capture", "--", *argv]


def _rocprof_read(tool:str, out:pathlib.Path) -> list[dict[str, Any]]:
  if dbs := sorted(out.rglob("*_results.db")):
    from boltbeam.collectors import rocpd_trace, semantic_capture
    return semantic_capture.read(rocpd_trace.export(dbs, out / "capture.sqlite"))
  files = sorted(out.rglob("*kernel_trace.csv"))
  if not files:
    raise RuntimeError(f"rocprofv3 wrote no kernel_trace.csv under {out}")
  return [l for f in files for l in read_rocprof_csv(f.read_text())]


def read_rocprof_csv(text:str) -> list[dict[str, Any]]:
  """Kernel launches from rocprofv3's kernel_trace.csv (nanosecond timestamps)."""
  out = []
  for row in csv.DictReader(io.StringIO(text)):
    grid = "x".join(row.get(k, "") for k in ("Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z"))
    block = "x".join(row.get(k, "") for k in ("Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z"))
    out.append({"key": f"{row['Kernel_Name']} grid={grid} block={block}",
                "wall_us": (float(row["End_Timestamp"]) - float(row["Start_Timestamp"])) / 1000.0,
                "start_us": float(row["Start_Timestamp"]) / 1000.0})
  return out


def _xctrace_read(tool:str, out:pathlib.Path) -> list[dict[str, Any]]:
  """GPU intervals of a Metal System Trace, exported with metal_system_trace.py's own export and XML reader."""
  from boltbeam.collectors import metal_system_trace as mst
  gpu_xml, sub_xml = out / "gpu.xml", out / "submissions.xml"
  mst._export_table("xcrun", out / "capture.trace", mst.GPU_INTERVALS, gpu_xml)
  mst._export_table("xcrun", out / "capture.trace", mst.COMMAND_BUFFERS, sub_xml)
  return read_metal_rows(mst._xml_rows(gpu_xml), mst._xml_rows(sub_xml))


# xctrace formats a label with the process and the command buffer's address: "q4k_gemv:Compute Command 0
# ( python (2691) )  0xe7e9df46". Both change per process and per buffer, so they are cut from the key.
_METAL_SUFFIX = re.compile(r"\s*\(\s*[^()]*\(\d+\)\s*\)\s*0x[0-9a-fA-F]+\s*$")


# A command buffer the runtime did not name: Metal's own "Command Buffer 0:Compute Command 0". It may hold a whole token's
# kernels (llama.cpp encodes its graph so), and nothing in it says which kernel ran or for how long.
_UNNAMED = re.compile(r"^Command Buffer \d+:\w+ Command \d+ encoders=\d+$")


def unnamed(key:str) -> bool:
  return bool(_UNNAMED.match(key))


def metal_label(formatted:str) -> str:
  return re.sub(r"\s+", " ", _METAL_SUFFIX.sub("", formatted)).strip()


def read_metal_rows(gpu:list[dict[str, Any]], submissions:list[dict[str, Any]]) -> list[dict[str, Any]]:
  """One launch per compute interval of the traced program. Metal System Trace reports command buffers, not
  dispatches: the key is the interval's label (its process and buffer address cut) and its encoder count, so a
  runtime that puts one kernel in each command buffer (tinygrad with JIT=2) is seen per kernel, and one that
  batches the graph (llama.cpp) is seen per batch. Rows of other programs (WindowServer), vertex and fragment
  rows, and nested rows (event-depth above 0, already inside a depth-0 row) are not counted. The GPU can pause a
  command buffer mid-run (another program's work, the display): the trace then shows it as back-to-back rows of one
  buffer and one label, which are one launch, its time the rows' sum (an M5 capture split 48 of 20,751 buffers so)."""
  from boltbeam.collectors import metal_system_trace as mst
  encoders = {mst._text(r.get("cmdbuffer-id")): mst._number(r.get("num-encoders")) for r in submissions}
  out = []
  last = None  # (buffer, label) of the row kept before
  for row in sorted(gpu, key=lambda r: mst._number(r.get("start")) or 0):
    cb = mst._text(row.get("cmdbuffer-id"))
    ns = mst._number(row.get("duration"))
    if cb not in encoders or ns is None:
      continue
    if (mst._text(row.get("channel-name")) or "Compute") != "Compute" or (mst._number(row.get("event-depth")) or 0) != 0:
      continue
    label = metal_label(mst._text(row.get("event-label")) or "") or "command buffer"
    start = mst._number(row.get("start"))
    if last == (cb, label):
      out[-1]["wall_us"] += ns / 1000.0
      continue
    last = (cb, label)
    out.append({"key": f"{label} encoders={encoders[cb]}", "wall_us": ns / 1000.0,
                **({"start_us": start / 1000.0} if start is not None else {})})
  return out


def _find_xctrace() -> str | None:
  try:
    proc = subprocess.run(["xcrun", "--find", "xctrace"], capture_output=True, text=True, timeout=30)
  except (OSError, subprocess.SubprocessError):
    return None
  return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


# each capture's code, by its method (target/vendors.py names a vendor's method and what to install): find the tool,
# wrap argv, read launches
_TOOLS: dict[str, tuple[Callable[[], str | None], Callable[..., list[str]], Callable[..., list[dict[str, Any]]]]] = {
  "nsys": (lambda: shutil.which("nsys"), _nsys_argv, _nsys_read),
  # ROCm installs rocprofv3 in its own bin/, which is often not on PATH: target/toolkits.py knows where it is
  "rocprofv3": (lambda: toolkits.tool("rocprofv3", shutil.which), _rocprof_argv, _rocprof_read),
  "metal-system-trace": (_find_xctrace,
                         lambda tool, argv, out, env: [tool, "record", "--template", "Metal System Trace", "--output",
                                                       str(out / "capture.trace"), "--no-prompt", "--target-stdout",
                                                       str(out / "target.stdout.txt"),
                                                       # xctrace starts the program with its own environment: pass ours
                                                       *[a for k, v in sorted(env.items()) for a in ("--env", f"{k}={v}")],
                                                       "--launch", "--", *argv],
                         _xctrace_read),
}
# backend -> (method, find the tool, wrap argv, read launches, what to install): the registry's rows with their code
VENDORS: dict[str, tuple[str, Callable[[], str | None], Callable[..., list[str]], Callable[..., list[dict[str, Any]]], str]] = {
  v.backend: (v.capture.method, *_TOOLS[v.capture.method], v.capture.install) for v in vendors.VENDORS.values()}


def plan(backend:str, find:Callable[[str], str | None] | None = None) -> dict[str, Any]:
  """The outside capture this machine has for a backend: {"method", "tool", "reason"}. tool None means none."""
  row = VENDORS.get(backend)
  if row is None:
    return {"method": None, "tool": None, "reason": f"BoltBeam knows no vendor capture for {backend}"}
  method, finder, _, _, install = row
  tool = find(method) if find else finder()
  if not tool:
    return {"method": method, "tool": None, "reason": f"{method} is not installed. To get it, {install}"}
  return {"method": method, "tool": tool, "reason": None}


def capture(backend:str, argv:list[str], out:pathlib.Path, *, timeout_s:float = 1800.0, cwd:pathlib.Path | None = None,
            env:dict[str, str] | None = None, flush_ms:int | None = None,
            profiler_range:bool = False, device:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """Run argv under the vendor tool and return its kernel launches. NoCapture when there is no tool here.
  flush_ms asks the tool to flush its buffers on a timer, where it can (nsys); profiler_range captures only the
  window the program marks with the CUDA profiler API (nsys) or roctx (rocprofv3). The program runs in `cwd`, by default the capture
  folder `out`: whatever it drops into its working directory (an old llama.cpp build writes a 421 KB
  llama_decode.dot graph dump on every decode) lands beside its capture, not in the caller's folder. Callers pass
  absolute paths in argv. device: the run's GPU (common.run_gpu), whose clocks are logged beside the capture."""
  p = plan(backend)
  if p["tool"] is None:
    raise NoCapture(p["reason"])
  _, _, wrap, read, _ = VENDORS[backend]
  out = pathlib.Path(out).absolute()  # the tool's own output paths must survive the cwd below
  out.mkdir(parents=True, exist_ok=True)
  cwd = cwd or out
  for stale in out.rglob("*_results.db"):  # rocprofv3's databases are read whole: a capture again starts clean
    stale.unlink()
  log = out / "capture.log"
  import contextlib
  from boltbeam.collectors import gpu_telemetry
  # the clocks the captured kernels ran at (the run's GPU, by its vendor's source): a repeat's row moves with them, so
  # a capture without them cannot say whether a difference between repeats was the clock
  tele = gpu_telemetry.opened(gpu_telemetry.source_for(device), 0) if device or backend == "CUDA" else None
  with open(log, "w") as fh, (tele or contextlib.nullcontext()):
    wrapped = (wrap(p["tool"], argv, out, env or {}, flush_ms, *((True,) if profiler_range else ())) if backend in ("CUDA", "AMD")
               else wrap(p["tool"], argv, out, env or {}))
    proc = subprocess.run(wrapped, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s,
                          cwd=cwd, env={**os.environ, **(env or {})})
  import json
  if tele is not None:
    (out / TELEMETRY).write_text(json.dumps({"summary": tele.summary(), "samples": gpu_telemetry.compact(tele.samples)}) + "\n")
  if proc.returncode != 0:
    raise RuntimeError(f"{p['method']} exited {proc.returncode}: {log.read_text()[-300:]}")
  launches = read(p["tool"], out)
  if not launches:
    raise RuntimeError(f"{p['method']} captured 0 kernels; the provider's GPU work was not visible to it")
  return launches


# rocprofv3's own log lines (glog: severity, date, time, thread, file:line]) share the program's stream and can land
# inside its output, even mid-line (one split llama-bench's JSON array); no engine prints this prefix
_TOOL_LOG = re.compile(r"[IWEF]\d{4} \d\d:\d\d:\d\d\.\d+ +\d+ \S+:\d+\] .*$")  # to the line's end


def program_output(out:pathlib.Path) -> str:
  """What the captured program printed: xctrace keeps it in target.stdout.txt, nsys and rocprofv3 in the log."""
  text = "\n".join(f.read_text(errors="replace") for f in (out / "target.stdout.txt", out / "capture.log") if f.is_file())
  return "\n".join(kept for line in text.splitlines() if (kept := _TOOL_LOG.sub("", line)) or not line)
