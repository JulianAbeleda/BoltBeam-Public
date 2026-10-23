"""The matrix paths of one AMD gfx target, found by scanning the installed compiler rather than written per chip.

    python -m boltbeam.collectors.hip_path_scan gfx1100 [--hipcc PATH]

    candidates  the matrix builtins the compiler itself knows: the __builtin_amdgcn_*wmma* names in the clang binary
                hipcc runs (HIPCC_VERBOSE=1 names it) and the clang libraries it links
    signature   each builtin's parameters, from clang's own diagnostics: a call with too few arguments states how
                many it takes, a struct passed in one position states that parameter's type, one assigned from the
                call states the result type. A bool is a flag: before the accumulator a signed-operand flag (true),
                after it OPSEL or clamp (false). A builtin of other than two equal inputs and an accumulator of the
                result's type is not probed: sparse (swmmac, whose index operand needs a 2:4 pattern the probe does
                not build) and modifier-taking builtins are said so in `excluded`
    target      hip_compute's own probe for that builtin compiled for the target (hipcc, BoltBeam's HIP cache, in
                parallel): rejected means not on this chip, with the compiler's words in `rejected`
    instruction llvm-objdump of the probe: kept only when it holds exactly one distinct v_wmma_*/v_swmmac_* mnemonic,
                of the builtin's own fields; several builtins lowering to one mnemonic keep the shortest name
    name        from the mnemonic's fields (v_wmma_<acc>_<MxNxK>_<in>) by NAMES, the one rule a human owns (which
                weight format a path serves, roofline/prefill_work); two of one name keep it for the largest K. Any
                other path is measured under its mnemonic and listed in `unmapped`: no weight format maps to it

The record is cached per compiler and target and kept in the chip profile (profile_row's matrix_path_scan), so a
target is scanned once. hip_compute's FAMILIES is the verified reference the scan reproduces and the table used
where there is no compiler.
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

from boltbeam.collectors import hip_compute as hc
from boltbeam.collectors import path_scan

SCAN_VERSION = 1  # bump when the rules below change: cached records of an older rule are not reused
CANDIDATE = re.compile(rb"__builtin_amdgcn_s?wmmac?_[a-z0-9_]+")
MNEMONIC = re.compile(r"\bv_s?wmmac?_([a-z0-9]+)_(\d+)x(\d+)x(\d+)_(\w+)")
_NAME = re.compile(r"wmmac?_([a-z0-9]+)_(\d+)x(\d+)x(\d+)_(\w+)")
_DIAG = re.compile(r":(\d+):\d+: error: (.*)")
_VECTOR = re.compile(r"\(vector of (\d+) '([\w ]+)' values\)")
# (accumulator, input) of the mnemonic -> the path name prefill_work's weight formats are keyed on
NAMES = {("f32", "f16"): "fp16", ("f16", "f16"): "fp16_f16acc", ("f32", "bf16"): "bf16", ("i32", "iu8"): "int8",
         ("i32", "iu4"): "int4", ("f32", "fp8_fp8"): "fp8", ("f32", "bf8_bf8"): "bf8"}
# a mnemonic field's words, and an input format's bits and one small element's bit pattern (fills an integer operand)
WORDS = {"f32": "fp32", "f16": "fp16", "bf16": "bf16", "i32": "int32", "iu8": "int8", "iu4": "int4",
         "fp8": "fp8 e4m3", "bf8": "fp8 e5m2"}
ENCODING = {"f16": (16, 0x1418), "bf16": (16, 0x3a83), "iu8": (8, 0x01), "iu4": (4, 0x1), "fp8": (8, 0x38), "bf8": (8, 0x3c)}
ELEMENTS = {"__fp16": "_Float16", "_Float16": "_Float16", "__bf16": "__bf16", "float": "float", "int": "int", "short": "short"}
LETTER = {"_Float16": "h", "__bf16": "b", "float": "f", "int": "i", "short": "s"}
BITS = {"int": 32, "short": 16}
BASE = tuple(re.findall(r"typedef (\S+) (\w+) __attribute__\(\(ext_vector_type\((\d+)\)\)\);", hc.TYPES))


def clang_of(hipcc:str) -> str | None:
  """The clang hipcc runs, from hipcc's own verbose line."""
  out = subprocess.run([hipcc, "--version"], capture_output=True, text=True, timeout=60, env={**os.environ, "HIPCC_VERBOSE": "1"})
  m = re.search(r"hipcc-cmd:\s*(\S+)", out.stdout + out.stderr)
  return m[1] if m else None


def candidates(clang:str) -> list[str]:
  """Every matrix builtin name in the clang binary and the clang libraries it links (a distro clang keeps its
  builtins in libclang-cpp)."""
  real = os.path.realpath(clang)
  libs = re.findall(r"=> (\S*clang\S*) ", subprocess.run(["ldd", real], capture_output=True, text=True).stdout)
  return sorted({m.decode() for f in [real, *libs] for m in CANDIDATE.findall(pathlib.Path(f).read_bytes())})


def _ctype(text:str) -> tuple[str, int] | None:
  """A diagnostic's parameter type -> (element, count); count 0 is a scalar, ('bool', 0) a flag."""
  if m := _VECTOR.search(text):
    return (ELEMENTS[m[2]], int(m[1])) if m[2] in ELEMENTS else None
  t = text.strip("'")
  return (ELEMENTS.get(t, t), 0) if t in ELEMENTS or t == "bool" else None


def _tname(t:tuple[str, int]) -> str:
  return f"{LETTER[t[0]]}{t[1]}" if t[1] else t[0]


def _typedefs(types) -> str:
  return "".join(f"typedef {e} {_tname((e, n))} __attribute__((ext_vector_type({n})));\n" for e, n in types if n)


def _diagnose(hipcc:str, arch:str, calls:dict[str, str], scratch:pathlib.Path, types) -> dict[str, str]:
  """One -fsyntax-only compile of every call, a line each: {builtin: its first error}."""
  names = list(calls)
  src = ("struct Q {};\n" + _typedefs(types) + "__attribute__((global)) void k() {\n  Q q;\n" +
         "".join(f"  {{ {calls[b]}; }}\n" for b in names) + "}\n")
  f = scratch / f"sig-{hashlib.sha256(src.encode()).hexdigest()[:16]}.hip"
  f.write_text(src)
  proc = subprocess.run([hipcc, "-fsyntax-only", "-fno-crash-diagnostics", "-ferror-limit=0", "--cuda-device-only", "-nogpuinc", f"--offload-arch={arch}", "-x", "hip", str(f)],
                        capture_output=True, text=True, timeout=600)
  first = 4 + len(_typedefs(types).splitlines())
  out: dict[str, str] = {}
  for line, msg in _DIAG.findall(proc.stderr):
    if 0 <= (i := int(line) - first) < len(names):
      out.setdefault(names[i], msg)
  if not out and proc.returncode and "error" in proc.stderr:
    raise RuntimeError(f"the compiler cannot target {arch}: {proc.stderr.strip().splitlines()[-1][:200]}")
  return out


def signatures(hipcc:str, arch:str, builtins:list[str], scratch:pathlib.Path) -> tuple[dict[str, Any], dict[str, str]]:
  """{builtin: (params, result)} from the compiler's diagnostics, and {builtin: why not read}."""
  types = dict.fromkeys((e, int(n)) for e, _, n in BASE)
  arity: dict[str, int] = {}
  for b, msg in _diagnose(hipcc, arch, {b: f"{b}()" for b in builtins}, scratch, types).items():
    if m := re.search(r"expected (\d+), have 0", msg):
      arity[b] = int(m[1])
  params: dict[str, list] = {b: [] for b in arity}
  bad: dict[str, str] = {b: "no arity in the compiler's diagnostic" for b in builtins if b not in arity}
  value = lambda t: "true" if t == ("bool", 0) else f"({_tname(t)})0"  # noqa: E731
  for i in range(max(arity.values(), default=0) + 1):
    live = [b for b in params if b not in bad and len(params[b]) == min(i, arity[b])]
    calls = {b: (f"{b}({', '.join([*map(value, params[b]), *['q'] * (arity[b] - i)])})" if i < arity[b] else
                 f"Q z = {b}({', '.join(map(value, params[b]))})") for b in live}
    for b, msg in _diagnose(hipcc, arch, calls, scratch, types).items():
      m = re.match(r"no viable conversion from (.+) to (.+)", msg)
      t = m and _ctype(m[2] if i < arity[b] else m[1])
      if t is None:
        bad[b] = f"type not read: {msg[:120]}"
      else:
        params[b].append(t)
        types[t] = None
  return {b: (p[:-1], p[-1]) for b, p in params.items() if b not in bad}, bad


def _fields(builtin:str) -> tuple[str, str, int, int, int] | None:
  """(acc, input, m, n, k) from a builtin's name: the input is its leading tokens that are formats."""
  if not (m := _NAME.search(builtin)):
    return None
  ins = list(itertools.takewhile(lambda t: t in ENCODING, m[5].split("_")))
  return (m[1], "_".join(ins), int(m[2]), int(m[3]), int(m[4])) if ins else None


def _init(t:tuple[str, int], fmt:str) -> str:
  if t[0] in BITS:  # an integer operand carries packed elements of the input format
    bits, pattern = ENCODING[fmt.split("_")[0]]
    word = sum(pattern << s for s in range(0, BITS[t[0]], bits))
    return f"({_tname(t)})0x{word:0{BITS[t[0]] // 4}x}"
  return f"({_tname(t)})({t[0]})0.001f"


def probe(builtin:str, sig) -> hc.Path | str:
  """The probe path of one builtin, or why it cannot be driven."""
  ps, ret = sig
  ops = [i for i, t in enumerate(ps) if t != ("bool", 0)]
  if len(ops) != 3 or ps[ops[0]] != ps[ops[1]] or ps[ops[2]] != ret:
    return "sparse or modifier-taking: not two equal inputs and an accumulator of the result's type"
  if (f := _fields(builtin)) is None:
    return "no operand encoding known for its input format"
  acc, fmt, m, n, k = f
  arg = {ops[0]: "{a}", ops[1]: "{a}", ops[2]: "{c}"}
  call = f"{builtin}({', '.join(arg.get(i, 'true' if i < ops[2] else 'false') for i in range(len(ps)))})"
  words = f"{' x '.join(dict.fromkeys(WORDS.get(x, x) for x in fmt.split('_')))} in, {WORDS.get(acc, acc)} accumulate"
  return hc.Path("", builtin, m, n, k, _tname(ps[ops[0]]), _init(ps[ops[0]], fmt), _tname(ret), call, words,
                 wave=64 if "_w64" in builtin else 32, lanes=ret[1] or 1, mnemonic=f"v_wmma_{acc}_{m}x{n}x{k}_{fmt}",
                 types=hc.TYPES + "".join(l for l in _typedefs(dict.fromkeys([ps[ops[0]], ret])).splitlines(True) if l not in hc.TYPES))


def lower(hipcc:str, arch:str, p:hc.Path, objdump:str, cache:pathlib.Path) -> str:
  """The one matrix mnemonic the probe compiles to, or RuntimeError with the compiler's words."""
  from boltbeam.runtime.hip_device import build_code_object
  code = build_code_object(hipcc, hc.source(p), arch=arch)
  co = path_scan.keep(cache / f"scan-{hashlib.sha256(code).hexdigest()[:16]}.co", code)
  found = {m[0] for m in MNEMONIC.finditer(subprocess.run([objdump, "-d", str(co)], capture_output=True, text=True, timeout=120).stdout)}
  if len(found) != 1:
    raise RuntimeError(f"lowers to {sorted(found) or 'no matrix instruction'}, not exactly one")
  return found.pop()


def table(kept:list[hc.Path]) -> tuple[tuple[hc.Path, ...], list[str]]:
  """The named table: NAMES order, then unmapped by mnemonic, then the vector FMA. One typedef preamble for the
  mapped paths (hip_compute.TYPES and what they add, in order); an unmapped path adds its own after it."""
  def key(p):
    m = MNEMONIC.search(p.mnemonic)
    return NAMES.get((m[1], m[5])), -p.k
  named: dict[str, hc.Path] = {}
  rest = []
  for p in sorted(kept, key=lambda p: (key(p)[0] or "", key(p)[1], p.mnemonic)):
    if (name := key(p)[0]) and name not in named:
      named[name] = dataclasses.replace(p, name=name)
    else:
      rest.append(dataclasses.replace(p, name=p.mnemonic.removeprefix("v_"), mapped=False, words=p.words + " (measured, unmapped)"))
  mapped = [named[n] for n in NAMES.values() if n in named]
  pre = "".join(dict.fromkeys(l for t in [hc.TYPES, *(p.types for p in mapped)] for l in t.splitlines(True)))
  out = [dataclasses.replace(p, types=pre) for p in mapped]
  out += [dataclasses.replace(p, types=pre + "".join(l for l in p.types.splitlines(True) if l not in pre))
          for p in sorted(rest, key=lambda p: p.mnemonic)]
  return (*out, hc.FP32_FMA), [p.name for p in rest]


def scan(arch:str, *, hipcc:str | None = None, cache:pathlib.Path | None = None, refresh:bool = False) -> dict[str, Any] | None:
  """The scan record for one gfx target, or None when there is no hipcc. A compiler that cannot target `arch` gives
  a record with no paths and its reason."""
  from boltbeam.runtime.hip_device import default_cache, find_hipcc
  hipcc = hipcc or find_hipcc()
  if not hipcc or not (clang := clang_of(hipcc)):
    return None
  real = pathlib.Path(os.path.realpath(clang))
  cache = cache or default_cache()

  def build() -> dict[str, Any]:
    names = candidates(clang)
    rec: dict[str, Any] = {"vendor": "amd", "arch": arch, "hipcc": hipcc, "compiler": str(real), "candidates": len(names),
                           "compiler_version": subprocess.run([str(real), "--version"], capture_output=True, text=True).stdout.splitlines()[0]}
    try:
      sigs, rec["excluded"] = signatures(hipcc, arch, names, cache)
    except RuntimeError as exc:
      return {**rec, "reason": str(exc)}
    probes = {b: probe(b, s) for b, s in sigs.items()}
    rec["excluded"].update({b: p for b, p in probes.items() if isinstance(p, str)})
    with cf.ThreadPoolExecutor(os.cpu_count() or 4) as pool:
      futs = {b: pool.submit(lower, hipcc, arch, p, str(real.parent / "llvm-objdump"), cache) for b, p in probes.items()
              if not isinstance(p, str)}
    by_mnemonic: dict[str, hc.Path] = {}
    for b, fut in futs.items():
      try:
        got = fut.result()
      except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        rec.setdefault("rejected", {})[b] = ((m := re.search(r"error: (.*)", str(exc))) and m[1][:200] or
                                             ("the compiler crashed: no lowering for this target" if "PLEASE ATTACH" in str(exc) else str(exc)[-200:]))
        continue
      if got != probes[b].mnemonic:
        rec.setdefault("rejected", {})[b] = f"lowers to {got}, not its own {probes[b].mnemonic}"
      elif got not in by_mnemonic or len(b) < len(by_mnemonic[got].builtin):
        by_mnemonic[got] = probes[b]
    found, rec["unmapped"] = table(list(by_mnemonic.values()))
    return {**rec, "paths": [dataclasses.asdict(p) for p in found]}
  st = real.stat()
  probe_id = hashlib.sha256(hc.source(hc.GFX11[0]).encode()).hexdigest()[:8]  # a changed probe template rescans
  return path_scan.cached(f"{SCAN_VERSION}:{probe_id}:{real}:{st.st_size}:{st.st_mtime_ns}:{hipcc}:{arch}", cache, refresh, build)


def paths(rec:dict[str, Any] | None) -> tuple[hc.Path, ...]:
  """A scan record's table; () when it found none."""
  return path_scan.rows(rec, hc.Path)


if __name__ == "__main__":
  raise SystemExit(path_scan.main(None, "boltbeam.collectors.hip_path_scan", scan, ("hipcc",)))
