"""Which matrix path a GPU kernel's instructions use, read from the kernel's own code, never from its name.

A kernel's compute ceiling is its work over the peak of the path its matrix instructions run on (fp16, int8, fp8,
...: collectors/cuda_compute.py PATHS). The path is read from the code the GPU runs:

    PTX     the library's embedded PTX (`cuobjdump --dump-ptx`), one `.entry` per kernel, or a kernel's own source
            text when an engine keeps it (inline `mma.sync...` in CUDA C, or PTX)
    match   the capture names a launch by its demangled name; the entry is demangled (cu++filt) and both are cut to
            the template (runtime/cuda_device.normalise_symbol), so `mul_mat_q<(ggml_type)12, (int)128, (bool)0>`
            meets `mul_mat_q<(ggml_type)12, 128, false>`

A matrix instruction's path is its operand types: the input type and the accumulator type (`signature`). The table
from signature to path is built from the probe's own instructions (PATHS), so the path a kernel is judged on is the
path that was measured. A kernel with no matrix instruction has no matrix path (its ceiling is its bytes); a kernel
whose instructions are on several paths is `mixed` and the row says which; a kernel whose code was not found has no
path, and its compute ceiling is refused, named.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
from typing import Any, Iterable

from boltbeam.collectors.cuda_compute import PATHS

_MMA = re.compile(r"\b(?:w?mma)\.sync\.aligned\.[\w.:]+")
_ENTRY = re.compile(r"\.entry\s+([^\s(]+)\s*\(")
_SHAPE = re.compile(r"^m\d+n\d+k\d+$")
_SKIP = {"sync", "aligned", "row", "col", "satfinite", "block_scale", "scale_vec::2X", "scale_vec::4X", "scale_vec::1X"}
NO_MATRIX = "no matrix instruction"
NOT_FOUND = "kernel code not found"


def signature(instr:str) -> tuple[str, str] | None:
  """(input type, accumulator type) of a PTX mma instruction: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` is
  ("s8", "s32"). The types follow the shape and layouts as D, A, B, C (a block-scaled one ends with its scale type)."""
  parts = instr.split(".")
  try:
    at = next(i for i, p in enumerate(parts) if _SHAPE.match(p))
  except StopIteration:
    return None
  types = [p for p in parts[at + 1:] if p not in _SKIP and not p.startswith("kind::")]
  if len(types) < 4:
    return None
  return types[1], types[3]


SIGNATURES = {signature(p.ptx): p.name for p in PATHS}
_INPUT_ALIASES = {"u8": "s8", "e5m2": "e4m3"}  # same unit, same rate as the measured sibling


def path_of_instruction(instr:str) -> str | None:
  sig = signature(instr)
  if sig is None:
    return None
  return SIGNATURES.get(sig) or SIGNATURES.get((_INPUT_ALIASES.get(sig[0], sig[0]), sig[1]))


def paths_in_text(text:str) -> dict[str, int]:
  """The matrix paths named by the mma instructions in a piece of code, with how many sites each."""
  out: dict[str, int] = collections.Counter()
  for m in _MMA.finditer(text):
    out[path_of_instruction(m.group(0)) or f"unknown:{m.group(0)}"] += 1
  return dict(out)


def ptx_entries(lines:Iterable[str]) -> dict[str, dict[str, int]]:
  """{mangled entry name: {path: sites}} over a PTX dump, every entry listed (an entry with no mma has {})."""
  out: dict[str, dict[str, int]] = {}
  cur = None
  for line in lines:
    m = _ENTRY.search(line)
    if m:
      cur = m.group(1)
      out.setdefault(cur, {})
      continue
    if cur and "mma." in line:
      for path, n in paths_in_text(line).items():
        out[cur][path] = out[cur].get(path, 0) + n
  return out


def template(name:str) -> str:
  """A kernel name cut to its template, as the matcher compares it: no return type, no argument list, no launch
  grid, casts and spaces out (cuda_device.normalise_symbol)."""
  from boltbeam.runtime.cuda_device import normalise_symbol
  s = name.split(" grid=", 1)[0].strip()
  s = re.sub(r"^void\s+", "", s)
  depth, cut = 0, len(s)
  for i, ch in enumerate(s):
    if ch == "<":
      depth += 1
    elif ch == ">":
      depth -= 1
    elif ch == "(" and depth == 0:
      cut = i
      break
  return normalise_symbol(s[:cut])


def library_table(library:str | pathlib.Path, *, cache:pathlib.Path | None = None,
                  cuobjdump:str | None = None) -> dict[str, Any]:
  """{template: paths} for every kernel in a CUDA library, from its embedded PTX; cached by the library's sha256.
  The record says which tool read it and the library's hash."""
  from boltbeam.runtime.cuda_device import default_cache, demangle, find_nvcc
  lib = pathlib.Path(library)
  sha = hashlib.sha256(lib.read_bytes()).hexdigest()
  cache = cache or default_cache()
  cached = cache / f"paths-{sha[:16]}.json"
  if cached.exists():
    return json.loads(cached.read_text())
  tool = cuobjdump or shutil.which("cuobjdump") or (str(pathlib.Path(find_nvcc() or "").with_name("cuobjdump")) if find_nvcc() else None)
  if not tool or not pathlib.Path(tool).exists() and not shutil.which(tool):
    raise RuntimeError("cuobjdump is not installed: a kernel's matrix path cannot be read from its library")
  proc = subprocess.Popen([tool, "--dump-ptx", str(lib)], stdout=subprocess.PIPE, text=True)
  entries = ptx_entries(proc.stdout)
  if proc.wait() != 0:
    raise RuntimeError(f"cuobjdump --dump-ptx {lib} exited {proc.returncode}")
  names = demangle(sorted(entries))
  table: dict[str, dict[str, int]] = {}
  for mangled, paths in entries.items():
    key = template(names.get(mangled, mangled))
    merged = table.setdefault(key, {})
    for p, n in paths.items():
      merged[p] = max(merged.get(p, 0), n)  # instantiations of one template on its launch keys: the most sites seen
  out = {"schema": "boltbeam.kernel_paths.v1", "library": str(lib), "sha256": sha, "how": "PTX embedded in the library, cuobjdump --dump-ptx",
         "kernels": table}
  cache.mkdir(parents=True, exist_ok=True)
  cached.write_text(json.dumps(out, sort_keys=True))
  return out


def kernel_path(name:str, table:dict[str, dict[str, int]]) -> dict[str, Any]:
  """The path of one captured kernel: {"path", "paths", "status", "words"}. status: one (one matrix path), mixed
  (several: the slowest is the ceiling's, said), none (no matrix instruction: bytes only), not_found (refused)."""
  key = template(name)
  paths = table.get(key)
  if paths is None:
    return {"path": None, "paths": {}, "status": "not_found", "words": f"{NOT_FOUND} for {key[:80]}"}
  known = {p: n for p, n in paths.items() if not p.startswith("unknown:")}
  if not paths:
    return {"path": None, "paths": {}, "status": "none", "words": NO_MATRIX}
  if not known:
    return {"path": None, "paths": paths, "status": "not_found", "words": f"matrix instruction of no measured path: {next(iter(paths))[8:]}"}
  if len(known) == 1:
    p = next(iter(known))
    return {"path": p, "paths": known, "status": "one", "words": p}
  return {"path": None, "paths": known, "status": "mixed", "words": " + ".join(sorted(known))}
