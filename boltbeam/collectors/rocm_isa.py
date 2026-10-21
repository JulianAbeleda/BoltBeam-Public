"""Which matrix path an AMD kernel's instructions use, read from the code object the capture saw it loaded from:
the AMD counterpart of instruction_path's PTX and SASS tables, in the same shape ({template: {path: count}}).

    where   rocprofv3's database names each kernel's code object. One loaded from a file (a library's device code,
            a hipBLASLt .co) is read at the recorded offset and size, so the code is the code that ran; one loaded
            from memory (Triton's JIT) is found by its symbol among the engine's JIT cache's .hsaco files
    what    llvm-objdump's disassembly, function by function. A matrix instruction is a v_wmma_* or v_swmmac_*; its
            path is the measured hip_compute path whose builtin lowers to that mnemonic, so the path a kernel is
            judged on is the path that was measured. A matrix instruction of no measured path is unknown:<mnemonic>.
            Vector dot and fused multiply-add instructions are the vector ALU, as FFMA is on CUDA: no matrix path.

A kernel whose code object is neither is left out, and instruction_path.kernel_path refuses it by name.
"""
from __future__ import annotations

import collections
import hashlib
import pathlib
import re
import shutil
import sqlite3
import subprocess
from typing import Any

from boltbeam.collectors.hip_compute import PATHS
from boltbeam.collectors.instruction_path import template
from boltbeam.target import toolkits

MNEMONICS = {"v_" + p.builtin.removeprefix("__builtin_amdgcn_").removesuffix("_w32_gfx12"): p.name
             for p in PATHS if p.kind == "mma"}
_MATRIX = re.compile(r"\bv_s?wmmac?_\w+")
_FUNC = re.compile(r"^[0-9a-f]+ <(.+)>:\s*$")  # a demangled name holds its own <...>
_FILE = re.compile(r"file://(.+)#offset=(\d+)&size=(\d+)")


def entries(text:str) -> dict[str, dict[str, int]]:
  """Disassembly text -> {symbol: {path: count}}; a function with no matrix instruction maps to {}."""
  out: dict[str, dict[str, int]] = {}
  current = None
  for line in text.splitlines():
    if m := _FUNC.match(line):
      current = out.setdefault(m[1], {})
    elif current is not None and (m := _MATRIX.search(line)):
      path = MNEMONICS.get(m[0], "unknown:" + m[0])
      current[path] = current.get(path, 0) + 1
  return out


def disassemble(code:bytes, tool:str, cache:pathlib.Path, *, demangle:bool = False) -> dict[str, dict[str, int]]:
  digest = hashlib.sha256(code).hexdigest()
  co = cache / f"{digest}.co"
  if not co.is_file():
    cache.mkdir(parents=True, exist_ok=True)
    co.write_bytes(code)
  proc = subprocess.run([tool, "-d", *(["-C"] if demangle else []), str(co)], capture_output=True, text=True, timeout=600)
  if proc.returncode != 0:
    raise RuntimeError(f"llvm-objdump exited {proc.returncode}: {proc.stderr.strip()[-200:]}")
  return entries(proc.stdout)


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
  kernels: dict[str, dict[str, int]] = {}
  read, jit = [], {}
  for uri, symbols in sorted(by_object.items()):
    if not (m := _FILE.match(uri)):
      jit.update(symbols)
      continue
    with open(m[1], "rb") as fh:
      fh.seek(int(m[2]))
      found = disassemble(fh.read(int(m[3])), tool, cache)
    kernels.update({template(d): found[s] for s, d in symbols if s in found})
    read.append({"library": m[1], "offset": int(m[2]), "kernels": sum(s in found for s, _ in symbols), "code": "amdgcn"})
  for d in jit_caches or []:
    for f in sorted(pathlib.Path(d).rglob("*.hsaco")) if jit and pathlib.Path(d).is_dir() else []:
      found = disassemble(f.read_bytes(), tool, cache)
      hits = {s: jit.pop(s) for s in list(jit) if s in found}
      kernels.update({template(display): found[s] for s, display in hits.items()})
      if hits:
        read.append({"library": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "kernels": len(hits),
                     "code": "amdgcn (the engine's JIT cache)"})
  # the rest were loaded from memory out of a library's fat binary (torch's own kernels): found there by symbol
  arch = next((r[0] for d in dbs for r in sqlite3.connect(d).execute("select name from rocpd_info_agent where type='GPU'")), None)
  for lib in (libraries or []) if jit and arch else []:
    for co in bundled(pathlib.Path(lib), arch, pathlib.Path(tool).parent) if pathlib.Path(lib).is_file() else []:
      syms = subprocess.run([tool, "--syms", str(co)], capture_output=True, text=True, timeout=600).stdout
      if wanted := [s for s in jit if s in syms]:
        found = disassemble(co.read_bytes(), tool, cache)
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
      found = disassemble(co.read_bytes(), tool, default_cache() / "code_objects", demangle=True)
      kernels.update({template(name): paths for name, paths in found.items()})
      read.append({"library": str(lib), "bundle": co.name, "kernels": len(found), "code": f"amdgcn for {arch}"})
  return {"kernels": kernels, "libraries": read, "how": f"the libraries' fat binaries for {arch}, llvm-objdump"}
