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
_FUNC = re.compile(r"^[0-9a-f]+ <([^>]+)>:")
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


def disassemble(code:bytes, tool:str, cache:pathlib.Path) -> dict[str, dict[str, int]]:
  digest = hashlib.sha256(code).hexdigest()
  co = cache / f"{digest}.co"
  if not co.is_file():
    cache.mkdir(parents=True, exist_ok=True)
    co.write_bytes(code)
  proc = subprocess.run([tool, "-d", str(co)], capture_output=True, text=True, timeout=600)
  if proc.returncode != 0:
    raise RuntimeError(f"llvm-objdump exited {proc.returncode}: {proc.stderr.strip()[-200:]}")
  return entries(proc.stdout)


def table(db_path:pathlib.Path, *, jit_caches:list[str] | None = None, cache:pathlib.Path | None = None) -> dict[str, Any]:
  """The matrix paths of every kernel a rocpd capture launched: {"kernels": {template: {path: count}}, "read": [...]}."""
  tool = shutil.which("llvm-objdump") or toolkits.file("rocm", "llvm/bin/llvm-objdump")
  if tool is None:
    raise RuntimeError("llvm-objdump is not here (it comes with ROCm, under llvm/bin): AMD kernel code cannot be read")
  cache = cache or pathlib.Path(db_path).parent / "code_objects"
  with sqlite3.connect(db_path) as db:
    launched = db.execute("select distinct s.code_object_id, s.kernel_name, s.display_name, c.uri from kernels k "
                          "join kernel_symbols s on s.id = k.kernel_id join code_objects c on c.id = s.code_object_id").fetchall()
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
  return {"kernels": kernels, "read": read, "unread": sorted(template(d) for d in jit.values())}
