"""The matrix paths of one NVIDIA sm target, found by asking the installed toolkit rather than written per chip.

    python -m boltbeam.collectors.cuda_path_scan sm_120 [--nvcc PATH]

    candidates  the warp-level `mma.sync.aligned` forms of the PTX ISA's m16n8 family: every shape x operand type x
                accumulator x kind (none, kind::f8f6f4, and the block_scale kinds with each scale vector and scale
                type), the A/B fragment sizes each element width (and the 8-bit container) implies. The toolkit
                ships no grammar, so GRAMMAR is the ISA's own dimensions, unfiltered: ptxas decides what is valid.
                Not covered: m8n8 shapes, f64, b1, sparse (mma.sp) and .satfinite
    target      one PTX module of every candidate through ptxas for `sm_XY`, then for `sm_XYa` (arch-specific
                instructions): a line ptxas names is not on this target
    instruction cuda_compute's own probe compiled (nvcc, BoltBeam's CUDA cache, in parallel) and disassembled
                (nvdisasm; with none the scan says so and finds nothing): kept when it issues exactly one distinct matrix opcode; forms
                lowering to one opcode keep the named one, else the shortest
    name        NAMES, the human rule (which weight format a path serves, roofline/prefill_work): two of one name keep
                it for the largest K; any other is named by its PTX form and listed in `unmapped` (cuda_compute measures the
                named ones; an unmapped one when passed in `paths`)

The record is cached per toolkit and target and kept in the chip profile (matrix_path_scan); cuda_compute.PATHS is
the verified reference it reproduces and the table used where there is no toolkit.
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import hashlib
import itertools
import os
import pathlib
import re
import subprocess
from typing import Any

from boltbeam.collectors import cuda_compute as cc
from boltbeam.collectors import path_scan

SCAN_VERSION = 1
SHAPES = ("m16n8k4", "m16n8k8", "m16n8k16", "m16n8k32", "m16n8k64")
BITS = {"f16": 16, "bf16": 16, "tf32": 32, "s8": 8, "u8": 8, "s4": 4, "u4": 4, "e4m3": 8, "e5m2": 8, "e3m2": 6,
        "e2m3": 6, "e2m1": 4}
FP = ("e4m3", "e5m2", "e3m2", "e2m3", "e2m1")
ACCS = ("f32", "f16", "s32")
# (kind, operand types, scale vectors, scale types): the PTX ISA's mma.sync dimensions
GRAMMAR = (("", tuple(t for t in BITS if t not in ("e3m2", "e2m3", "e2m1")), ("",), ("",)), ("kind::f8f6f4", FP, ("",), ("",)),
           *((f"kind::{k}.block_scale", FP, ("1X", "2X", "4X"), ("ue8m0", "ue4m3")) for k in ("mxf8f6f4", "mxf4", "mxf4nvf4")))
SCALE_ONE = {"ue8m0": 0x7F, "ue4m3": 0x38}  # a unit scale's byte, replicated over the scale register
WORDS = {"f16": "fp16", "bf16": "bf16", "tf32": "tf32", "s8": "int8", "u8": "uint8", "s4": "int4", "u4": "uint4",
         "e4m3": "fp8 e4m3", "e5m2": "fp8 e5m2", "e3m2": "fp6 e3m2", "e2m3": "fp6 e2m3", "e2m1": "fp4 e2m1",
         "f32": "fp32", "s32": "int32"}
# (kind, accumulator, input, scale type) -> (path name, its words): the names prefill_work's weight formats key on
NAMES = {("", "f32", "f16", ""): ("fp16", "fp16 in, fp32 accumulate"),
         ("", "f16", "f16", ""): ("fp16_f16acc", "fp16 in, fp16 accumulate"),
         ("", "f32", "bf16", ""): ("bf16", "bf16 in, fp32 accumulate"),
         ("", "f32", "tf32", ""): ("tf32", "tf32 in, fp32 accumulate"),
         ("", "s32", "s8", ""): ("int8", "int8 in, int32 accumulate"),
         ("", "f32", "e4m3", ""): ("fp8", "fp8 e4m3 in, fp32 accumulate"),
         ("kind::mxf4.block_scale", "f32", "e2m1", "ue8m0"): ("fp4", "fp4 e2m1 in with ue8m0 block scales, fp32 accumulate"),
         ("kind::mxf4nvf4.block_scale", "f32", "e2m1", "ue4m3"): ("nvfp4", "fp4 e2m1 in with ue4m3 block-16 scales, fp32 accumulate")}
_PTX = re.compile(r"mma\.sync\.aligned\.m(\d+)n(\d+)k(\d+)\.row\.col\.(?:(kind::\w+(?:\.block_scale)?)\.)?(?:scale_vec::\w+\.)?"
                  r"(\w+)\.(\w+)\.(\w+)\.\w+(?:\.(\w+))?$")
_LINE = re.compile(r"line (\d+); error")


def fields(ptx:str) -> tuple[str, str, str, str, int] | None:
  """(kind, accumulator, input, scale type, K) of a PTX form; the input is its A type, or A.B when they differ."""
  if not (m := _PTX.match(ptx)):
    return None
  return m[4] or "", m[5], m[6] if m[6] == m[7] else f"{m[6]}.{m[7]}", m[8] or "", int(m[3])


def candidates() -> list[cc.Path]:
  """Every form GRAMMAR spans, with the fragment sizes its element widths imply (a sub-byte type also as 8-bit)."""
  out = []
  for (kind, types, vecs, stypes), shape in itertools.product(GRAMMAR, SHAPES):
    m, n, k = map(int, re.findall(r"\d+", shape))
    for a, b, acc, vec, st in itertools.product(types, types, ACCS, vecs, stypes):
      scale = f"scale_vec::{vec}." if vec else ""
      ptx = f"mma.sync.aligned.{shape}.row.col.{kind + '.' if kind else ''}{scale}{acc}.{a}.{b}.{acc}{'.' + st if st else ''}"
      for bits in sorted({BITS[a], 8} if BITS[a] < 8 else {BITS[a]}):
        if (m * k * bits) % 1024 or (n * k * bits) % 1024:
          continue
        out.append(cc.Path("", ptx, m, n, k, a_regs=m * k * bits // 1024, b_regs=n * k * bits // 1024, acc=acc,
                           extra=", {sa}, {{0, 0}}, {sb}, {{0, 0}}" if st else "",
                           scale_bits=int.from_bytes(bytes([SCALE_ONE.get(st, 0x7F)]) * 4, "little")))
  return out


def _ptx_line(p:cc.Path) -> str:
  regs = itertools.count()
  frag = lambda n: "{" + ", ".join(f"%r{next(regs)}" for _ in range(n)) + "}"  # noqa: E731
  d = frag({"f32": 4, "s32": 4, "f16": 2}[p.acc])
  a, b = frag(p.a_regs), frag(p.b_regs)
  extra = p.extra.format(sa="%r60", sb="%r61") if p.extra else ""
  return f"{p.ptx} {d}, {a}, {b}, {d}{extra};"


def valid(ptxas:str, version:str, sm:str, paths:list[cc.Path], scratch:pathlib.Path) -> set[int]:
  """The indices of the candidates ptxas accepts for `sm`: one module, a function per candidate, run again without
  the lines it named until it passes (ptxas stops reporting after some errors)."""
  live = list(range(len(paths)))
  while live:
    body = []
    for i in live:  # the k-th live candidate is lines 4 + 5k to 8 + 5k
      body += [f".visible .entry c{i}() {{", "  .reg .b32 %r<64>;", "  " + _ptx_line(paths[i]), "  ret;", "}"]
    f = scratch / f"scan-{sm}-{hashlib.sha256(''.join(body).encode()).hexdigest()[:12]}.ptx"
    f.write_text(f".version {version}\n.target {sm}\n.address_size 64\n" + "\n".join(body) + "\n")
    proc = subprocess.run([ptxas, f"-arch={sm}", "-o", os.devnull, str(f)], capture_output=True, text=True, timeout=600)
    f.unlink()
    bad = {live[(int(n) - 4) // 5] for n in _LINE.findall(proc.stdout + proc.stderr) if 4 <= int(n) < 4 + 5 * len(live)}
    if not proc.returncode:
      return set(live)
    if not bad:
      raise RuntimeError(f"ptxas cannot build for {sm}: {(proc.stderr or proc.stdout).strip()[-200:]}")
    live = [i for i in live if i not in bad]
  return set()


def lower(nvcc:str, p:cc.Path, sm:str, nvdisasm:str, cache:pathlib.Path) -> set[str]:
  """The matrix opcodes the probe for one path issues on `sm`."""
  from boltbeam.collectors.instruction_path import _SASS_MMA
  from boltbeam.runtime.cuda_device import build_cubin
  cubin = build_cubin(nvcc, cc.source(p), cache=cache, arch=sm + p.arch_suffix)
  f = path_scan.keep(cache / f"c-{hashlib.sha256(cubin).hexdigest()[:16]}.cubin", cubin)
  proc = subprocess.run([nvdisasm, "-c", str(f)], capture_output=True, text=True, timeout=300)
  if proc.returncode:
    raise RuntimeError(f"nvdisasm exited {proc.returncode}: {proc.stderr.strip()[-200:]}")
  return set(_SASS_MMA.findall(proc.stdout))


def table(kept:list[cc.Path]) -> tuple[tuple[cc.Path, ...], list[str]]:
  """NAMES order, then the unmapped forms by PTX, then the vector FMA."""
  named: dict[str, cc.Path] = {}
  rest = []
  for p in sorted(kept, key=lambda p: (-p.k, p.ptx)):
    kind, acc, ins, st, _ = fields(p.ptx)
    hit = NAMES.get((kind, acc, ins, st))
    if hit and hit[0] not in named:
      named[hit[0]] = dataclasses.replace(p, name=hit[0], words=hit[1])
    else:
      words = f"{' x '.join(WORDS.get(t, t) for t in ins.split('.'))} in{f' with {st} block scales' if st else ''}, {WORDS.get(acc, acc)} accumulate"
      rest.append(dataclasses.replace(p, name=p.ptx.removeprefix("mma.sync.aligned."), words=words + " (unmapped: found, not measured by default)"))
  fma = next(p for p in cc.PATHS if p.kind == "fma")
  return (*[named[n] for n, _ in NAMES.values() if n in named], *sorted(rest, key=lambda p: p.ptx), fma), [p.name for p in rest]


def scan(sm:str, *, nvcc:str | None = None, nvdisasm:str | None = None, cache:pathlib.Path | None = None,
         refresh:bool = False) -> dict[str, Any] | None:
  """The scan record for one sm target (sm_120), or None when there is no toolkit."""
  from boltbeam.runtime.cuda_device import default_cache, find_cuda_tool, find_nvcc
  nvcc = nvcc or find_nvcc()
  ptxas = nvcc and str(pathlib.Path(nvcc).with_name("ptxas"))
  if not nvcc or not pathlib.Path(ptxas).is_file():
    return None
  nvdisasm = nvdisasm or find_cuda_tool("nvdisasm", pathlib.Path(nvcc).parent)
  cache = cache or default_cache()

  def build() -> dict[str, Any]:
    version = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    rec: dict[str, Any] = {"vendor": "nvidia", "arch": sm, "nvcc": nvcc, "compiler": ptxas, "compiler_version": version}
    probe = subprocess.run([nvcc, "-ptx", f"-arch=compute_{sm.removeprefix('sm_')}", "-x", "cu", "-o", "-", os.devnull],
                           capture_output=True, text=True, timeout=120)
    if not (m := re.search(r"^\.version (\S+)", probe.stdout, re.M)) or not nvdisasm:
      return {**rec, "reason": f"nvcc cannot target {sm}: {probe.stderr.strip()[-200:]}" if not m else
              "nvdisasm is not installed: the probes' machine code cannot be read"}
    forms = candidates()
    base = valid(ptxas, m[1], sm, forms, cache)
    rest = [p for i, p in enumerate(forms) if i not in base]
    ok = [forms[i] for i in sorted(base)] + [dataclasses.replace(rest[i], arch_suffix="a") for i in sorted(valid(ptxas, m[1], sm + "a", rest, cache))]
    with cf.ThreadPoolExecutor(os.cpu_count() or 4) as pool:
      futs = {p.ptx: (p, pool.submit(lower, nvcc, p, sm, nvdisasm, cache)) for p in ok}
    by_op: dict[str, cc.Path] = {}
    rank = lambda q: (fields(q.ptx)[:4] in NAMES, -len(q.ptx))  # noqa: E731
    for ptx, (p, fut) in futs.items():
      try:
        ops = fut.result()
      except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        rec.setdefault("rejected", {})[ptx] = ((m := re.search(r"error\s*: (.*)", str(exc))) and m[1] or str(exc).strip()[-200:])[:200]
        continue
      if len(ops) != 1:
        rec.setdefault("rejected", {})[ptx] = f"issues {sorted(ops) or 'no matrix opcode'}, not exactly one"
      elif (op := ops.pop()) not in by_op or rank(p) > rank(by_op[op]):
        by_op[op] = p
    found, rec["unmapped"] = table(list(by_op.values()))
    return {**rec, "candidates": len({p.ptx for p in forms}), "sass": {p.ptx: op for op, p in by_op.items()},
            "paths": [dataclasses.asdict(p) for p in found]}
  st = pathlib.Path(os.path.realpath(ptxas)).stat()
  probe_id = hashlib.sha256(cc.source(cc.PATHS[0]).encode()).hexdigest()[:8]  # a changed probe template rescans
  return path_scan.cached(f"{SCAN_VERSION}:{probe_id}:{ptxas}:{st.st_size}:{st.st_mtime_ns}:{nvcc}:{sm}", cache, refresh, build)


def paths(rec:dict[str, Any] | None) -> tuple[cc.Path, ...]:
  """A scan record's table; () when it found none."""
  return path_scan.rows(rec, cc.Path)


if __name__ == "__main__":
  raise SystemExit(path_scan.main(None, "boltbeam.collectors.cuda_path_scan", scan, ("nvcc", "nvdisasm")))
