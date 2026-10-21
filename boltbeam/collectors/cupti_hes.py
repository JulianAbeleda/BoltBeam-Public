"""Opt-in process-lifetime CUPTI hardware event timing, with no CUDA API tracing.

BOLTBEAM_CUDA_CAPTURE=cupti-hes selects this collector; BOLTBEAM_CUPTI_ROOT names
a CUDA 13 CUPTI SDK containing include/ and lib/. Unsupported hardware fails closed.
The engine retains its own CUDA libraries: only CUPTI's profiler dependencies enter
the loader path. Profiler-range/server captures still require their existing tools.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess

from boltbeam.runtime.cuda_device import demangle, find_cuda_tool

FLAGS = ["-shared", "-fPIC", "-O2", "-std=c++17", "-Wl,-rpath,$ORIGIN/profiler-libraries"]


def find() -> str | None:
  root = Path(os.environ.get("BOLTBEAM_CUPTI_ROOT", "/nonexistent"))
  return str(root.resolve()) if all((root / p).is_file() for p in
    ("include/cupti.h", "lib/libcupti.so.13", "lib/libnvperf_host.so", "lib/libnvperf_target.so")) and shutil.which("g++") and find_cuda_tool("cu++filt") else None


def argv(tool:str, command:list[str], out:Path, env:dict[str, str], flush_ms:int | None = None,
         profiler_range:bool = False) -> list[str]:
  if flush_ms or profiler_range:
    raise ValueError("cupti-hes supports complete process captures only; use nsys for profiler ranges or servers")
  sdk = Path(tool)
  cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
  libraries = out / "profiler-libraries"
  libraries.mkdir(exist_ok=True)
  for name in ("libcupti.so.13", "libnvperf_host.so", "libnvperf_target.so"):
    link = libraries / name
    if not link.exists():
      link.symlink_to((sdk / "lib" / name).resolve())
  helper = out / "cupti-hes.so"
  source = shutil.copyfile(Path(__file__).with_suffix(".cpp"), out / "cupti-hes.cpp")
  subprocess.run(["g++", *FLAGS, str(source),
                  "-I" + str(sdk / "include"), "-I" + str(cuda / "include"), "-L" + str(libraries),
                  "-l:libcupti.so.13", "-l:libcuda.so.1", "-o", str(helper)],
                 check=True, capture_output=True, text=True, timeout=120)
  files = {"helper_source": Path(source), "helper_binary": helper, "collector_source": Path(__file__),
           "cupti_library": libraries / "libcupti.so.13", "nvperf_host": libraries / "libnvperf_host.so",
           "nvperf_target": libraries / "libnvperf_target.so", "symbol_decoder": Path(find_cuda_tool("cu++filt"))}
  identity = {"method": "cupti-hes", "compiler_flags": FLAGS,
              **{key + "_sha256": hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}}
  (out / "capture_identity.json").write_text(json.dumps(identity, sort_keys=True) + "\n")
  env["LD_PRELOAD"] = ":".join(filter(None, (str(helper), env.get("LD_PRELOAD", os.environ.get("LD_PRELOAD", "")))))
  env["LD_LIBRARY_PATH"] = ":".join(filter(None, (str(libraries), env.get("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", "")))))
  env["BOLTBEAM_HES_OUTPUT"] = str(out / "cuda_gpu_trace.csv")
  env["BOLTBEAM_HES_CLOCK"] = str(out / "cupti-clock.json")
  return command


def complete(out:Path) -> None:
  identity = json.loads((out / "capture_identity.json").read_text())
  identity["cupti_version"] = json.loads((out / "cupti-clock.json").read_text())["cupti_version"]
  (out / "capture_identity.json").write_text(json.dumps(identity, sort_keys=True) + "\n")


def read(tool:str, out:Path) -> list[dict]:
  from boltbeam.collectors.vendor_capture import read_nsys_csv
  complete = re.findall(r"^BOLTBEAM_HES kernels=(\d+) dropped=(\d+)$", (out / "capture.log").read_text(), re.M)
  if len(complete) != 1 or complete[0][1] != "0":
    raise ValueError("cupti-hes capture lacks one complete, zero-drop process")
  launches = read_nsys_csv((out / "cuda_gpu_trace.csv").read_text(),
                          origin_ns=json.loads((out / "cupti-clock.json").read_text())["cupti_ns"])
  if len(launches) != int(complete[0][0]):
    raise ValueError("cupti-hes kernel count disagrees with its completed capture")
  decoder = find_cuda_tool("cu++filt")
  if not decoder:
    raise ValueError("cupti-hes needs cu++filt to decode its retained kernel symbols")
  names = demangle(sorted({l["key"].split(" grid=", 1)[0] for l in launches}), filt=decoder)
  for launch in launches:
    name, geometry = launch["key"].split(" grid=", 1)
    launch["key"] = names[name] + " grid=" + geometry
  return launches


def window(out:Path, launches:list[dict]) -> list[float]:
  anchor = json.loads((out / "cupti-clock.json").read_text())
  zero = anchor["monotonic_ns"] / 1e9
  return [zero + min(l["start_us"] for l in launches) / 1e6,
          zero + max(l["start_us"] + l["wall_us"] for l in launches) / 1e6]
