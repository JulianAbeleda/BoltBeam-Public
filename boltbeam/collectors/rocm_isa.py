"""Which matrix path an AMD kernel's instructions use, read from the code object the capture saw it loaded from:
the AMD counterpart of instruction_path's PTX and SASS tables, in the same shape ({template: {path: count}}).

    where   rocprofv3's database names each kernel's code object. One loaded from a file (a library's device code,
            a hipBLASLt .co) is read at the recorded offset and size, so the code is the code that ran; one loaded
            from memory (Triton's JIT) is found by its symbol among the engine's JIT cache's .hsaco files
    what    llvm-objdump's disassembly, function by function. A matrix instruction is a v_wmma_* or v_swmmac_*; its
            path is the measured hip_compute path whose builtin lowers to that mnemonic, so the path a kernel is
            judged on is the path that was measured. A matrix instruction of no measured path is unknown:<mnemonic>.
            Vector dot and fused multiply-add instructions are the vector ALU, as FFMA is on CUDA: no matrix path.

A kernel whose code object is neither is left out, and instruction_path.kernel_path refuses it by name, unless
rocprofv3's thread trace saw it execute (observe_closed, below; also for code holding several matrix paths).
"""
from __future__ import annotations

import collections
import csv
import hashlib
import pathlib
import re
import shutil
import sqlite3
import subprocess
from typing import Any

from boltbeam.collectors.hip_compute import FAMILIES, paths_for
from boltbeam.collectors.instruction_path import template
from boltbeam.target import toolkits
from boltbeam.target.targets import stored_path_scan

MNEMONICS = {f: {p.mnemonic: p.name for p in FAMILIES[f] if p.kind == "mma"} for f in FAMILIES}  # the verified tables


def mnemonics(arch:str) -> dict[str, str]:
  """{mnemonic: path} of one gfx target: the paths hip_compute measures there (a profile's scan, else its family)."""
  return {p.mnemonic: p.name for p in paths_for(arch) if p.kind == "mma"}
_MATRIX = re.compile(r"\bv_s?wmmac?_\w+")
_FUNC = re.compile(r"^[0-9a-f]+ <(.+)>:\s*$")  # a demangled name holds its own <...>
_FILE = re.compile(r"file://(.+)#offset=(\d+)&size=(\d+)")


def entries(text:str, arch:str | None = None) -> dict[str, dict[str, int]]:
  """Disassembly text -> {symbol: {path: count}}; a function with no matrix instruction maps to {}."""
  # with no target, every family's map: a mnemonic names the same path wherever it exists
  known = {m: p for f in MNEMONICS.values() for m, p in f.items()} if arch is None else mnemonics(arch)
  out: dict[str, dict[str, int]] = {}
  current = None
  for line in text.splitlines():
    if m := _FUNC.match(line):
      current = out.setdefault(m[1], {})
    elif current is not None and (m := _MATRIX.search(line)):
      path = known.get(m[0], "unknown:" + m[0])
      current[path] = current.get(path, 0) + 1
  return out


def disassemble(code:bytes, tool:str, cache:pathlib.Path, *, demangle:bool = False, arch:str | None = None) -> dict[str, dict[str, int]]:
  digest = hashlib.sha256(code).hexdigest()
  co = cache / f"{digest}.co"
  if not co.is_file():
    cache.mkdir(parents=True, exist_ok=True)
    co.write_bytes(code)
  proc = subprocess.run([tool, "-d", *(["-C"] if demangle else []), str(co)], capture_output=True, text=True, timeout=600)
  if proc.returncode != 0:
    raise RuntimeError(f"llvm-objdump exited {proc.returncode}: {proc.stderr.strip()[-200:]}")
  return entries(proc.stdout, arch)


BUNDLES = (b"CCOB", b"__CLANG_OFFLOAD_BUNDLE__")  # a .hip_fatbin is a chain of (compressed) offload bundles


def bundled(library:pathlib.Path, arch:str, tools:pathlib.Path) -> list[pathlib.Path]:
  """The code objects for `arch` in a library's .hip_fatbin, unbundled once into BoltBeam's HIP cache (torch's 229
  gfx1201 objects take about 10 s), keyed by the library's path, size and time."""
  from boltbeam.runtime.hip_device import default_cache
  st = library.stat()
  folder = default_cache() / f"bundles-{hashlib.sha256(f'{library.resolve()}:{st.st_size}:{st.st_mtime_ns}'.encode()).hexdigest()[:16]}-{arch}"
  if not folder.is_dir():
    (tmp := folder.with_suffix(".tmp")).mkdir(parents=True, exist_ok=True)
    subprocess.run([str(tools / "llvm-objcopy"), f"--dump-section=.hip_fatbin={tmp / 'fat'}", str(library), "/dev/null"],
                   capture_output=True, timeout=600)
    data = (tmp / "fat").read_bytes() if (tmp / "fat").is_file() else b""
    starts = sorted(m.start() for magic in BUNDLES for m in re.finditer(re.escape(magic), data))
    for k, (a, b) in enumerate(zip(starts, [*starts[1:], len(data)])):  # a false magic fails to unbundle: skipped
      (tmp / "part").write_bytes(data[a:b])
      subprocess.run([str(tools / "clang-offload-bundler"), "--unbundle", "--type=o", f"--input={tmp / 'part'}",
                      f"--targets=hipv4-amdgcn-amd-amdhsa--{arch}", f"--output={tmp / f'{k}.co'}"], capture_output=True)
    for f in ("fat", "part"):
      (tmp / f).unlink(missing_ok=True)
    tmp.rename(folder)
  return [f for f in sorted(folder.glob("*.co")) if f.stat().st_size]


def table(dbs:list[pathlib.Path], *, jit_caches:list[str] | None = None, cache:pathlib.Path | None = None,
          libraries:list[pathlib.Path] | None = None) -> dict[str, Any]:
  """The matrix paths of every kernel a rocpd capture launched, in library_paths' shape: {kernels, libraries, how}."""
  tool = shutil.which("llvm-objdump") or toolkits.file("rocm", "llvm/bin/llvm-objdump")
  if tool is None:
    raise RuntimeError("llvm-objdump is not here (it comes with ROCm, under llvm/bin): AMD kernel code cannot be read")
  cache = cache or pathlib.Path(dbs[0]).parent / "code_objects"
  launched = [r for d in dbs for r in sqlite3.connect(d).execute(  # one database per traced process
    "select distinct s.code_object_id, s.kernel_name, s.display_name, c.uri from kernels k "
    "join kernel_symbols s on s.id = k.kernel_id join code_objects c on c.id = s.code_object_id")]
  by_object: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
  for _, symbol, display, uri in launched:
    by_object[uri].append((symbol.removesuffix(".kd"), display))
  arch = next((r[0] for d in dbs for r in sqlite3.connect(d).execute("select name from rocpd_info_agent where type='GPU'")), None)
  kernels: dict[str, dict[str, int]] = {}
  read, jit = [], {}
  for uri, symbols in sorted(by_object.items()):
    if not (m := _FILE.match(uri)):
      jit.update(symbols)
      continue
    with open(m[1], "rb") as fh:
      fh.seek(int(m[2]))
      found = disassemble(fh.read(int(m[3])), tool, cache, arch=arch)
    kernels.update({template(d): found[s] for s, d in symbols if s in found})
    read.append({"library": m[1], "offset": int(m[2]), "kernels": sum(s in found for s, _ in symbols), "code": "amdgcn"})
  for d in jit_caches or []:
    for f in sorted(pathlib.Path(d).rglob("*.hsaco")) if jit and pathlib.Path(d).is_dir() else []:
      found = disassemble(f.read_bytes(), tool, cache, arch=arch)
      hits = {s: jit.pop(s) for s in list(jit) if s in found}
      kernels.update({template(display): found[s] for s, display in hits.items()})
      if hits:
        read.append({"library": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "kernels": len(hits),
                     "code": "amdgcn (the engine's JIT cache)"})
  # the rest were loaded from memory out of a library's fat binary (torch's own kernels): found there by symbol
  for lib in (libraries or []) if jit and arch else []:
    for co in bundled(pathlib.Path(lib), arch, pathlib.Path(tool).parent) if pathlib.Path(lib).is_file() else []:
      syms = subprocess.run([tool, "--syms", str(co)], capture_output=True, text=True, timeout=600).stdout
      if wanted := [s for s in jit if s in syms]:
        found = disassemble(co.read_bytes(), tool, cache, arch=arch)
        kernels.update({template(jit.pop(s)): found[s] for s in wanted if s in found})
        read.append({"library": str(lib), "bundle": co.name, "kernels": len(wanted), "code": "amdgcn (the library's fat binary)"})
      if not jit:
        break
  return {"kernels": kernels, "libraries": read, "unread": sorted(template(d) for d in jit.values()),
          "how": "the code objects rocprofv3 saw load (a library's at its recorded offset, Triton's JIT cache), llvm-objdump"}


def library_table(libraries:list[pathlib.Path], arch:str | None) -> dict[str, Any]:
  """The matrix paths of the kernels in the libraries' fat binaries for one gfx target (the hardware profile's), by
  demangled template: a capture with no rocpd database (llama.cpp's libggml-hip). A library with no code object for
  that target is read as none; no tool or no target refuses, as table does."""
  tool = shutil.which("llvm-objdump") or toolkits.file("rocm", "llvm/bin/llvm-objdump")
  if tool is None or not arch:
    raise RuntimeError("llvm-objdump is not here (it comes with ROCm, under llvm/bin): AMD kernel code cannot be read"
                       if tool is None else "the hardware profile names no gfx target: AMD kernel code cannot be chosen")
  from boltbeam.runtime.hip_device import default_cache
  kernels: dict[str, dict[str, int]] = {}
  read = []
  for lib in dict.fromkeys(pathlib.Path(x).resolve() for x in libraries):
    for co in bundled(lib, arch, pathlib.Path(tool).parent) if lib.is_file() else []:
      found = disassemble(co.read_bytes(), tool, default_cache() / "code_objects", demangle=True, arch=arch)
      kernels.update({template(name): paths for name, paths in found.items()})
      read.append({"library": str(lib), "bundle": co.name, "kernels": len(found), "code": f"amdgcn for {arch}"})
  return {"kernels": kernels, "libraries": read, "how": f"the libraries' fat binaries for {arch}, llvm-objdump"}


# --- rocprofv3 thread trace: the instructions a closed kernel executed (optional, ATT_ENV=1) -----------------------
# AMD's closed_observer (target/vendors.py), the counterpart of instruction_path.observe_closed (ncu). Thread trace
# (rocprofv3 --att) is trace-only on RDNA3 and RDNA4 (gfx1100-1102, gfx1200-1201) and needs the ROCprof Trace Decoder.
# Each traced dispatch's stats_*.csv holds Codeobj, Vaddr, Instruction, Hitcount, Latency, Stall, Idle, Source;
# Hitcount is the times an instruction executed, summed over the traced waves.

FROM_ATT = "observed by rocprofv3"
ATT_ENV = "BOLTBEAM_ATT"  # set to 1 to thread-trace a capture's closed kernels
ATT_ARGS = "BOLTBEAM_ATT_ARGS"  # extra rocprofv3 flags (--att-target-cu, --att-shader-engine-mask, ...), shell words
_DISPATCH = re.compile(r"dispatch_(\d+)")


def att_command(tool:str, argv:list[str], out:pathlib.Path, templates:set[str], arch:str, *,
                profiler_range:bool = False, extra:list[str] | None = None) -> list[str]:
  """rocprofv3 thread-tracing the named templates' dispatches, with a kernel trace beside it to name each dispatch.
  One CU (a WGP on Navi) of shader engine 0 is traced, rocprofv3's defaults stated; Navi's SIMD select is 0x0."""
  names = "|".join(re.escape(t.split("<", 1)[0].split("::")[-1]) for t in sorted(templates))
  return [tool, "--att", "--kernel-trace", "--output-format", "csv", "-d", str(out), "-o", "att",
          "--kernel-include-regex", names, "--att-target-cu", "1", "--att-shader-engine-mask", "0x1",
          *(["--att-simd-select", "0x0"] if arch.startswith("gfx1") else []), "--att-buffer-size", "0x6000000",
          *(["--selected-regions"] if profiler_range else []), *(extra or []), "--", *argv]


def att_stats(out:pathlib.Path, arch:str) -> dict[str, Any]:
  """{"kernels": {template: {path: executed}}, "unmapped": stats files naming no traced dispatch}. A dispatch's
  stats file is found by the dispatch id in its path (ui_output_agent_{agent}_dispatch_{id}): ASSUMED, the docs do
  not pin the stats file's name; one whose path holds no id, or an id the kernel trace lacks, is unmapped, not
  guessed. A kernel whose stats hold no executed instruction (no wave landed on the traced CU) is left out: absent
  is not observed, never "no matrix instruction"."""
  known, names, got, unmapped = mnemonics(arch), {}, {}, 0
  for f in out.rglob("*kernel_trace.csv"):
    with open(f, newline="") as fh:
      names.update({r["Dispatch_Id"]: template(r["Kernel_Name"]) for r in csv.DictReader(fh)})
  for f in sorted(out.rglob("stats_*.csv")):
    ids = _DISPATCH.findall(str(f.relative_to(out)))
    if (t := names.get(ids[-1]) if ids else None) is None:
      unmapped += 1
      continue
    with open(f, newline="") as fh:
      rows = [(r.get("Instruction") or "", int(float(r.get("Hitcount") or 0))) for r in csv.DictReader(fh)]
    if not any(n for _, n in rows):
      continue
    paths = got.setdefault(t, {})
    for ins, n in rows:  # an instruction of no matrix mnemonic is the vector ALU: no matrix path
      if n and (m := _MATRIX.search(ins)):
        p = known.get(m[0], "unknown:" + m[0])
        paths[p] = paths.get(p, 0) + n
  return {"kernels": got, "unmapped": unmapped}


def observe_closed(program:dict[str, Any], closed:set[str], ran:list[str]) -> dict[str, Any]:
  """AMD's closed-kernel observer: rocprofv3's thread trace of the closed templates, when asked for (ATT_ENV), the
  program run as captured. {} when not asked; observed_how alone when it cannot run; every closed kernel the trace
  did not see is named in observed_how. `ran` (launch order) is unused: the regex selects the dispatches."""
  import os, shlex
  if os.environ.get(ATT_ENV) != "1":
    return {}
  tool, raw = toolkits.tool("rocprofv3", shutil.which), pathlib.Path(program["raw"])
  arch = program.get("arch") or next((r[0] for d in sorted(raw.glob("*_results.db")) for r in sqlite3.connect(d).execute(
    "select name from rocpd_info_agent where type='GPU'")), None)
  if not tool or not (stored_path_scan(arch) or (arch or "")[:5] in MNEMONICS):
    return {"observed_how": "rocprofv3 thread trace not run: " + (
      "rocprofv3 not found on PATH or under ROCM_PATH, HIP_PATH or /opt/rocm" if not tool else
      f"no matrix-path table for the capture's GPU target {arch!r}")}
  out = raw / "att"
  shutil.rmtree(out, ignore_errors=True)
  out.mkdir(parents=True)
  cmd = att_command(tool, program["argv"], out, closed, arch, profiler_range=bool(program.get("profiler_range")),
                    extra=shlex.split(os.environ.get(ATT_ARGS, "")))
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=out, env={**os.environ, **(program.get("env") or {})})
  (out / "att.log").write_text(proc.stdout + proc.stderr)
  if proc.returncode != 0:
    return {"observed_how": f"rocprofv3 --att exited {proc.returncode} (thread trace needs the ROCprof Trace Decoder "
                            f"and an RDNA3/RDNA4 or CDNA target): {(proc.stdout + proc.stderr).strip()[-300:]}"}
  got = att_stats(out, arch)
  seen = {t: p for t, p in got["kernels"].items() if t in closed}
  missing = sorted(closed - set(seen))
  how = ("rocprofv3 thread trace (--att, one CU of shader engine 0): each closed kernel's executed instructions "
         "(stats Hitcount), its WMMA mnemonics mapped to the measured paths")
  if missing:
    how += f"; not observed (no wave on the traced CU, or no stats file named its dispatch): {', '.join(t[:60] for t in missing)}"
  if got["unmapped"]:
    how += f"; {got['unmapped']} stats file(s) named no traced dispatch"
  return {"observed": seen, "observed_how": how, "observed_by": FROM_ATT}
