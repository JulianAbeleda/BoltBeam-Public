"""The machine code of the CUDA modules a process loads, for kernels no library file holds.

cuBLASLt builds its nvjet_* kernels in the process, so no file has their code. The driver loads the library named by
CUDA_INJECTION64_PATH at cuInit and calls its InitializeInjection; ours (cuda_module_load.cpp) subscribes CUPTI's
module-loaded callback and writes every module as the driver holds it. Measured on an RTX 5090 (driver 595, CUDA 13,
2026-11-09): the callback hands over a finished ELF cubin whichever way the image arrived (torch's and cuBLAS's fat
binaries, NVRTC's PTX, NVRTC's cubin), where interposing cuModuleLoadData* or cuLibraryLoadData saw nothing (their
callers reach the driver through dlsym or its export table, a different way each). It opens no counter, so it needs no
counter permission. It is a pass of its own: nsys replaces CUDA_INJECTION64_PATH, and CUPTI takes one subscriber a
process (a second made nsys abort).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

FLAGS = ["-shared", "-fPIC", "-O2", "-std=c++17"]
HEADER = "file,cubin_size,context_handle,module_id,pid\n"  # cuda_counter_stream.catalog reads the first three
METHOD = "CUPTI module-loaded callback, injected by the driver (CUDA_INJECTION64_PATH)"
HOW = ("modules the driver loaded in a pass of the same program outside the timed captures (CUPTI module-loaded "
       "callback): each kernel found by its entry symbol, its machine code read with nvdisasm")
KINDS = {b"\x7fELF": "cubin", b"\x50\xed\x55\xba": "fat binary", b"BC\xc0\xde": "NVVM bitcode"}


class NoModuleCapture(RuntimeError):
  """The module-load capture cannot run here; the message says why."""


def sdk(python:str | None) -> tuple[Path, Path] | None:
  """(include folder, libcupti) of a CUPTI SDK: BOLTBEAM_CUPTI_ROOT, an NVIDIA wheel of the engine's venv (vLLM's
  ships nvidia/cu13 with cupti.h and libcupti.so.13), or the toolkit's extras/CUPTI."""
  roots = [Path(os.environ["BOLTBEAM_CUPTI_ROOT"])] if os.environ.get("BOLTBEAM_CUPTI_ROOT") else []
  roots += sorted(Path(python or "").parent.parent.glob("lib/python*/site-packages/nvidia/*"))
  roots += [Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "extras/CUPTI"]
  for root in roots:
    libraries = sorted(p for d in ("lib", "lib64") for p in root.glob(d + "/libcupti.so.*"))
    if (root / "include/cupti.h").is_file() and libraries:
      return root / "include", libraries[0].resolve()
  return None


def capture(argv:list[str], env:dict[str, str], out:Path, *, python:str | None = None, timeout_s:float = 3600.0) -> list[dict]:
  """argv run once with the shim injected, no profiler around it: the program's JSON lines, and in `out` every module
  the driver loaded (<pid>-<n>.cubin, modules.csv) with capture_identity.json. NoModuleCapture when it cannot run."""
  from boltbeam.collectors.engine_common import run_driver
  found, compiler = sdk(python), shutil.which("g++")
  if not found or not compiler:
    raise NoModuleCapture("g++ is not installed" if found else "no CUPTI SDK (include/cupti.h and lib/libcupti.so.*) in "
                          "the engine's NVIDIA wheels, BOLTBEAM_CUPTI_ROOT or the CUDA toolkit")
  out.mkdir(parents=True, exist_ok=True)
  for stale in out.glob("*.cubin"):  # modules are read whole: a pass again starts clean
    stale.unlink()
  shim, source = out / "module-load.so", Path(shutil.copyfile(Path(__file__).with_suffix(".cpp"), out / "module-load.cpp"))
  subprocess.run([compiler, *FLAGS, str(source), "-I" + str(found[0]), str(found[1]), "-Wl,-rpath," + str(found[1].parent),
                  "-o", str(shim)], check=True, capture_output=True, text=True, timeout=120)
  (out / "modules.csv").write_text(HEADER)
  files = {"helper_source": source, "helper_binary": shim, "collector_source": Path(__file__), "cupti_library": found[1]}
  (out / "capture_identity.json").write_text(json.dumps({"method": METHOD, "compiler_flags": FLAGS, "command": argv,
    **{key + "_sha256": hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}}, sort_keys=True) + "\n")
  log = out / "capture.log"
  try:
    rows = run_driver(argv, env={**env, "CUDA_INJECTION64_PATH": str(shim), "BOLTBEAM_MODULES": str(out)}, log=log, timeout_s=timeout_s)
  except RuntimeError:
    refused = [l for l in log.read_text().splitlines() if l.startswith("BOLTBEAM_MODULES failed")] if log.exists() else []
    if refused:  # the shim ended the process: CUPTI refused the subscription, or a module could not be written
      raise NoModuleCapture(refused[-1][len("BOLTBEAM_MODULES failed "):]) from None
    raise
  if "BOLTBEAM_MODULES subscribed" not in log.read_text():
    raise NoModuleCapture("the driver did not load the shim (CUDA_INJECTION64_PATH was not honoured)")
  return rows


def table(folder:Path, wanted:set[str], signatures:dict[str, str], *, nvdisasm:str | None = None,
          cache:Path | None = None) -> dict:
  """{"kernels": {template: paths}, "evidence", "modules", "how"} for the wanted kernel templates the captured modules
  hold as an entry symbol: never a similar name. Every module is checked against the size the driver reported and
  recorded with its sha256 and kind; only a cubin holding a wanted entry is disassembled, its opcodes mapped by
  `signatures` (instruction_path.sass_signatures). A captured PTX module is read as a library's embedded PTX is
  (instruction_path.ptx_entries); any other kind is recorded and not read. One kernel whose captured copies hold
  different paths is refused: the capture cannot say which one a launch ran."""
  from boltbeam.collectors import instruction_path as ip
  from boltbeam.runtime.cuda_device import demangle, elf_symbols
  with open(folder / "modules.csv", newline="") as f:
    rows = list(csv.DictReader(f))
  if not rows:
    raise ValueError("module-load capture recorded no module: the program loaded none while the shim was subscribed")
  kernels: dict[str, dict[str, int]] = {}
  evidence: dict[str, list[dict]] = {}
  modules = []
  for row in rows:
    if Path(row["file"]).name != row["file"]:
      raise ValueError("module-load capture names a module outside its folder")
    raw = (folder / row["file"]).read_bytes()
    if len(raw) != int(row["cubin_size"]):
      raise ValueError(f"captured module {row['file']} holds {len(raw)} bytes; the driver loaded {row['cubin_size']}")
    kind = KINDS.get(raw[:4]) or ("ptx" if b".entry" in raw else "unknown")
    module = {"file": row["file"], "sha256": hashlib.sha256(raw).hexdigest(), "cubin_size": len(raw), "kind": kind,
              "module_id": int(row["module_id"]), "pid": int(row["pid"]), "context_handle": row["context_handle"]}
    modules.append(module)
    ptx = ip.ptx_entries(raw.decode(errors="replace").splitlines()) if kind == "ptx" else {}
    entries = elf_symbols(raw, entry=True) if kind == "cubin" else sorted(ptx)
    plain = demangle(entries)
    held = [s for s in entries if ip.template(plain[s]) in wanted]
    code = ip._disassemble(raw, nvdisasm, cache) if held and kind == "cubin" else ptx
    for symbol in held:
      if symbol not in code:
        raise ValueError(f"captured module {row['file']} has entry {symbol} and nvdisasm printed no code for it")
      got: dict[str, int] = {}
      for op, n in code[symbol].items():
        path = op if kind == "ptx" else signatures.get(op) or f"unknown:{op}"
        got[path] = got.get(path, 0) + n
      key = ip.template(plain[symbol])
      if kernels.setdefault(key, got) != got:
        raise ValueError(f"captured modules hold different matrix paths for {key}: {evidence[key][0]['file']} and {row['file']}")
      evidence.setdefault(key, []).append({**module, "symbol": symbol, "code": "ptx" if kind == "ptx" else "machine code (nvdisasm)"})
  return {"kernels": kernels, "evidence": evidence, "modules": modules, "how": HOW}
