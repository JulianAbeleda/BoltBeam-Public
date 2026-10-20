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
whose instructions are on several paths is `mixed` and the row says which.

A kernel whose code is not readable (a closed library such as cuBLAS, or a JIT-compiled DSL kernel with no cached
code) is read from its name when NAME_PATHS holds a documented pattern for it: the row says "from the kernel name",
and a path read from code always wins. A kernel with neither has no path, and its compute ceiling is refused, named.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re
from functools import lru_cache
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
  # Block-scale format/vector length identifies a distinct instruction path. MXFP4
  # and NVFP4 have the same operands but must not borrow one another's probe.
  scale = next((p for p in parts if p.startswith("scale_vec::")), None)
  operand = types[1] + (f"/{types[-1]}/{scale}" if scale else "")
  return operand, types[3]


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


@lru_cache(maxsize=4096)
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
  from boltbeam.runtime.cuda_device import default_cache, demangle, find_cuda_tool
  lib = pathlib.Path(library)
  sha = hashlib.sha256(lib.read_bytes()).hexdigest()
  cache = cache or default_cache()
  decoder = hashlib.sha256(repr(sorted(SIGNATURES.items(), key=str)).encode()).hexdigest()[:12]
  cached = cache / f"paths-{sha[:16]}-{decoder}.json"
  if cached.exists():
    return json.loads(cached.read_text())
  tool = cuobjdump or find_cuda_tool('cuobjdump')
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


# --- SASS: a library that ships machine code only (cuBLAS ships no PTX for most of its kernels) ------------------
# The machine-code matrix opcode of each path is read from the probe's own kernel compiled for this GPU, so a kernel
# is judged on the same instruction the probe measured: `HMMA.16816.F32.BF16` is whatever the bf16 probe compiles to.

_SASS_MMA = re.compile(r"\b[A-Z]*MMA(?:\.[A-Z0-9_]+)+")
_SASS_FMA = re.compile(r"\bFFMA(?:\.[A-Z0-9_]+)*\b")
_SASS_FUNC = re.compile(r"^\s*\.text\.(\S+):")


def sass_entries(lines:Iterable[str], *, include_scalar:bool = False) -> dict[str, dict[str, int]]:
  """{mangled function: {matrix opcode: sites}} over nvdisasm output; a function with no matrix opcode has {}."""
  out: dict[str, dict[str, int]] = {}
  cur = None
  for line in lines:
    m = _SASS_FUNC.match(line)
    if m:
      cur = m.group(1)
      out.setdefault(cur, {})
      continue
    if cur:
      for op in _SASS_MMA.findall(line) + (_SASS_FMA.findall(line) if include_scalar else []):
        out[cur][op] = out[cur].get(op, 0) + 1
  return out


def sass_signatures(arch:str, *, cache:pathlib.Path | None = None, nvcc:str | None = None,
                    nvdisasm:str | None = None, include_scalar:bool = False) -> dict[str, str]:
  """{matrix opcode: path} for this GPU: each probe path (cuda_compute.PATHS) compiled for `arch` (sm_XY), its matrix
  opcode read back. A path that does not compile here is left out (its opcodes then read as unknown)."""
  from boltbeam.collectors.cuda_compute import source
  from boltbeam.runtime.cuda_device import build_cubin, find_nvcc
  nvcc = nvcc or find_nvcc()
  out: dict[str, str] = {}
  for p in PATHS:
    if (p.kind != "mma" and not (include_scalar and p.kind == "fma")) or not nvcc:
      continue
    try:
      cubin = build_cubin(nvcc, source(p), cache=cache, arch=arch + p.arch_suffix)
    except RuntimeError:
      continue
    for ops in _disassemble(cubin, nvdisasm, cache, include_scalar=p.kind == "fma").values():
      for op in ops:
        out.setdefault(op, p.name)
  return out


def _disassemble(cubin:bytes, nvdisasm:str | None, cache:pathlib.Path | None, *, include_scalar:bool = False) -> dict[str, dict[str, int]]:
  from boltbeam.runtime.cuda_device import default_cache
  tool = nvdisasm or shutil.which("nvdisasm")
  if not tool:
    raise RuntimeError("nvdisasm is not installed: a kernel's machine code cannot be read")
  d = cache or default_cache()
  d.mkdir(parents=True, exist_ok=True)
  f = d / f"c-{hashlib.sha256(cubin).hexdigest()[:16]}.cubin"
  if not f.exists():
    f.write_bytes(cubin)
  proc = subprocess.run([tool, "-c", str(f)], capture_output=True, text=True, timeout=1800)
  if proc.returncode != 0:
    raise RuntimeError(f"nvdisasm {f} exited {proc.returncode}: {proc.stderr.strip()[-200:]}")
  return sass_entries(proc.stdout.splitlines(), include_scalar=include_scalar)


def sass_table(library:str | pathlib.Path, wanted:set[str], arch:str, signatures:dict[str, str], *,
               cache:pathlib.Path | None = None, cuobjdump:str | None = None, nvdisasm:str | None = None) -> dict[str, Any]:
  """{template: paths} for the wanted kernel templates found in a library's machine code for `arch`: its cubins
  extracted (cuobjdump -xelf), their function symbols read, and only the cubins that hold a wanted kernel
  disassembled. Each opcode is mapped to a path by `signatures` (sass_signatures); one it does not know is
  `unknown:OPCODE`. The library's symbol index and every disassembled cubin's table are cached by hash."""
  import tempfile
  from boltbeam.runtime.cuda_device import default_cache, demangle, elf_symbols, find_cuda_tool
  lib = pathlib.Path(library)
  sha = hashlib.sha256(lib.read_bytes()).hexdigest()
  cache = cache or default_cache()
  cache.mkdir(parents=True, exist_ok=True)
  index_file = cache / f"sass-{sha[:16]}-{arch}.json"
  try:  # an index cut short (a run stopped while writing it) is rebuilt, never trusted
    index = json.loads(index_file.read_text()) if index_file.exists() else None
  except ValueError:
    index = None
  if index is not None:
    renamed={s:template(d) for s,d in demangle([s for s in index['symbols'] if s.startswith('_Z')]).items() if s!=d}
    if renamed:
      for field in ('symbols','paths'):
        index[field]={renamed.get(s,s):v for s,v in index[field].items()}
      index_file.write_text(json.dumps(index,sort_keys=True))
    # Saved unknown opcodes retain enough evidence to resolve newly measured
    # paths without rereading multi-GB libraries. Do not guess an opcode alias.
    for paths in index['paths'].values():
      for old in list(paths):
        if old.startswith('unknown:') and old[8:] in signatures:
          new = signatures[old[8:]]
          paths[new] = paths.get(new, 0) + paths.pop(old)
  tool = cuobjdump or find_cuda_tool('cuobjdump')
  with tempfile.TemporaryDirectory(dir=cache) as tmp:
    if index is None or any(t in index["symbols"] and t not in index["paths"] for t in wanted):
      if not tool:
        raise RuntimeError("cuobjdump is not installed: a library's machine code cannot be read")
      proc = subprocess.run([tool, "-xelf", "all", str(lib.absolute())], cwd=tmp, capture_output=True, text=True, timeout=3600)
      cubins = sorted(pathlib.Path(tmp).glob(f"*.{arch}.cubin")) if proc.returncode == 0 else []
      if index is None:
        names = {c.name: elf_symbols(c.read_bytes()) for c in cubins}
        plain = demangle(sorted({s for v in names.values() for s in v}))
        index = {"symbols": {template(plain.get(s, s)): c for c, syms in names.items() for s in syms}, "paths": {}}
      for t in sorted(wanted):
        c = index["symbols"].get(t)
        if c is None or t in index["paths"] or not (pathlib.Path(tmp) / c).exists():
          continue
        entries = _disassemble((pathlib.Path(tmp) / c).read_bytes(), nvdisasm, cache)
        plain = demangle(sorted(entries))
        for mangled, ops in entries.items():
          key = template(plain.get(mangled, mangled))
          if key in wanted or key not in index["paths"]:
            merged: dict[str, int] = {}
            for op, n in ops.items():
              p = signatures.get(op) or f"unknown:{op}"
              merged[p] = merged.get(p, 0) + n
            index["paths"][key] = merged
      index_file.write_text(json.dumps(index, sort_keys=True))
  return {"library": str(lib), "sha256": sha, "how": f"machine code for {arch} (cuobjdump -xelf, nvdisasm)",
          "kernels": {t: index["paths"][t] for t in wanted if t in index["paths"]}}


FROM_NAME = "from the kernel name"
# A closed kernel's matrix path from its name: (pattern, path or None for no matrix instruction, the source that
# documents it). First match wins. Only for kernels whose code was not read; code always wins over a name.
NAME_PATHS: tuple[tuple[str, str | None, str], ...] = (
  (r"delta_rule_cp_sm\d+CPDeltaRule(?:TPrecompute|MNPrecompute|Prefill)Sm\d+", "bf16",
   "FlashInfer 0.7.0.post1 gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py lines 436, 872 and 2074: "
   "warp.MmaF16BF16Op(dtype, Float32, (16, 8, 16)) in the model's dtype; ncu on a bf16 model (2026-10-11) saw "
   "HMMA.16816.F32.BF16 (TPrecompute also about 16% fp16 m16n8k16/k8); fp16 runs at the same measured rate"),
  (r"delta_rule_cp_sm\d+CPDeltaRuleFixupHmmaSm\d+", "tf32",
   "ncu (2026-10-11): the HMMA fixup executes HMMA.1688.F32.TF32 only, tf32 in with fp32 accumulate"),
  (r"delta_rule_cp_sm\d+CPDeltaRuleFixupSimtSm\d+", None,
   "FlashInfer 0.7.0.post1 delta_rule_cp_sm120.py line 2463: the SIMT fixup, no MMA atom"),
  (r"(?:tensorop|xmma|gemm)\w*?(?:e4m3|e5m2)", "fp8", "CUTLASS and cuBLASLt names carry an fp8 element type (e4m3, e5m2)"),
  (r"_bf16_s\d{4,6}gemm", "bf16", "CUTLASS and cuBLAS names: bf16 elements, s16816 = fp32 accumulate on m16n8k16"),
  (r"_(?:fp16_|f16_)?h\d{4,6}gemm", "fp16_f16acc", "CUTLASS and cuBLAS names: fp16 elements, h = fp16 accumulate"),
  (r"_(?:fp16_|f16_)?s\d{4,6}gemm", "fp16", "CUTLASS and cuBLAS names: fp16 elements, s = fp32 accumulate"),
  (r"_(?:int8_|s8_)?i\d{4,6}gemm", "int8", "CUTLASS and cuBLAS names: int8 elements, i = int32 accumulate"),
  (r"gemm_bf16bf16_\w*?f32", "bf16", "cuBLASLt xmma names: A and B bf16, fp32 accumulate"),
  (r"gemm_f16f16_f16f16", "fp16_f16acc", "cuBLASLt xmma names: A and B fp16, fp16 accumulate"),
  (r"gemm_f16f16_\w*?f32", "fp16", "cuBLASLt xmma names: A and B fp16, fp32 accumulate"),
  (r"gemm_(?:s8|i8)(?:s8|i8)_", "int8", "cuBLASLt xmma names: A and B int8"),
)


def name_path(name:str) -> dict[str, Any] | None:
  """A kernel's path from NAME_PATHS, labelled FROM_NAME with its source; None when no pattern matches."""
  for pat, path, source in NAME_PATHS:
    if re.search(pat, name, re.I):
      if path is None:
        return {"path": None, "paths": {}, "status": "none", "words": f"{NO_MATRIX} ({FROM_NAME})", "from_name": True,
                "name_source": source}
      return {"path": path, "paths": {path: 1}, "status": "one", "words": f"{path} ({FROM_NAME})", "from_name": True,
              "name_source": source}
  return None


def kernel_path(name:str, table:dict[str, dict[str, int]], observed:dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
  """The path of one captured kernel: {"path", "paths", "status", "words"}. status: one (one matrix path), mixed
  (several: each path's share of the operations at its own peak, prefill_work.compute_ms), none (no matrix instruction: bytes only), not_found (refused).
  Code read from the kernel decides; a kernel whose code was not read takes the path ncu saw it execute (observed,
  ncu_observe), else a path from its name (name_path)."""
  key = template(name)
  paths = table.get(key)
  if paths is None and observed and key in observed:
    got = kernel_path(name, {key: observed[key]})
    return {**got, "words": f"{got['words']} ({FROM_NCU})", "from_ncu": True}
  if paths is None:
    return name_path(name) or {"path": None, "paths": {}, "status": "not_found", "words": f"{NOT_FOUND} for {key[:80]}"}
  known = {p: n for p, n in paths.items() if not p.startswith("unknown:")}
  if not paths:
    return {"path": None, "paths": {}, "status": "none", "words": NO_MATRIX}
  unknown = [p for p in paths if p.startswith("unknown:")]
  if unknown:
    return {"path": None, "paths": paths, "status": "not_found", "words": "matrix instruction of no measured path: " + ", ".join(p[8:] for p in unknown)}
  if len(known) == 1:
    p = next(iter(known))
    return {"path": p, "paths": known, "status": "one", "words": p}
  return {"path": None, "paths": known, "status": "mixed", "words": " + ".join(sorted(known))}


# --- ncu: the instructions a closed kernel executed (optional; needs the GPU's counters open to this user) ----------
# BoltBeam works without it: code, then names, decide. ncu only answers for kernels whose code was not read, and what
# it saw is folded back into NAME_PATHS by hand (docs/ncu-instruction-mix-*.md), so the name route learns from it.

FROM_NCU = "observed by ncu"
NCU_ENV = "BOLTBEAM_NCU"  # set to 1 to run ncu on a capture's closed kernels


def ncu_entries(lines:Iterable[str], *, include_scalar:bool = False) -> dict[str, dict[str, int]]:
  """{kernel name: {matrix opcode: warp instructions executed}} from `ncu -i R --page source --print-source sass
  --csv`: one "Kernel Name" line per profiled launch, then its SASS lines; launches of one name are summed."""
  import csv
  out: dict[str, dict[str, int]] = {}
  cur, col = None, None
  for row in csv.reader(lines):
    if len(row) >= 2 and row[0] == "Kernel Name":
      cur, col = row[1], None
      out.setdefault(cur, {})
    elif cur is not None and col is None and "Source" in row and "Instructions Executed" in row:
      col = (row.index("Source"), row.index("Instructions Executed"))
    elif cur is not None and col is not None and len(row) > max(col):
      ops = _SASS_MMA.findall(row[col[0]]) + (_SASS_FMA.findall(row[col[0]]) if include_scalar else [])
      if ops:
        n = int(float(row[col[1]] or 0))
        for op in ops:out[cur][op] = out[cur].get(op, 0) + n
  return out


def ncu_table(entries:dict[str, dict[str, int]], signatures:dict[str, str]) -> dict[str, dict[str, int]]:
  """{template: {path: instructions executed}}: each executed opcode mapped by the probe's own (sass_signatures),
  one it does not know as unknown:OPCODE. Optional scalar evidence applies only when no matrix instruction ran;
  scalar epilogue instructions do not change a GEMM's matrix-work path."""
  out: dict[str, dict[str, int]] = {}
  for name, ops in entries.items():
    got = out.setdefault(template(name), {})
    matrix = any(_SASS_MMA.fullmatch(op) and n for op,n in ops.items())
    for op, n in ops.items():
      if n and (not matrix or not _SASS_FMA.fullmatch(op)):
        p = signatures.get(op) or f"unknown:{op}"
        got[p] = got.get(p, 0) + n
  return out


def counters_open(params:str = "/proc/driver/nvidia/params") -> bool:
  """Whether this user may read the GPU's performance counters (the driver's RmProfilingAdminOnly is 0)."""
  try:
    return "RmProfilingAdminOnly: 0" in pathlib.Path(params).read_text()
  except OSError:
    return False


def ncu_observe(argv:list[str], env:dict[str, str] | None, out:pathlib.Path, templates:set[str],
                signatures:dict[str, str], *, ncu:str | None = None, launches:int = 200) -> dict[str, Any]:
  """The program run once under ncu (its profiler range only), the launches of the named templates profiled for
  their executed SASS (SourceCounters), at most `launches` of them: {"kernels": ncu_table, "how"}."""
  import os
  tool = ncu or shutil.which("ncu")
  if not tool or not templates:
    return {"kernels": {}, "how": "ncu not run"}
  out.mkdir(parents=True, exist_ok=True)
  names = "|".join(re.escape(t.split("<", 1)[0].split("::")[-1]) for t in sorted(templates))
  rep = out / "ncu"
  from boltbeam.profiler.ncu_counters import ncu_command
  proc = subprocess.run(ncu_command(argv, out=str(rep), ncu=tool, sections=("SourceCounters",), kernel_name=f"regex:{names}",
                                    launch_count=launches),
                        capture_output=True, text=True, timeout=7200, env={**os.environ, **(env or {})})
  (out / "ncu.log").write_text(proc.stdout + proc.stderr)
  if proc.returncode != 0 or not rep.with_suffix(".ncu-rep").exists():
    raise RuntimeError(f"ncu exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
  csv_text = subprocess.run([tool, "-i", str(rep.with_suffix(".ncu-rep")), "--page", "source", "--print-source", "sass",
                             "--csv"], capture_output=True, text=True, timeout=3600).stdout
  table = ncu_table(ncu_entries(csv_text.splitlines()), signatures)
  return {"kernels": {t: p for t, p in table.items() if t in templates},
          "how": "ncu SourceCounters: the SASS each closed kernel executed, its matrix opcodes mapped to the probe's paths"}

