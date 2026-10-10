"""Where the engines are on this machine, looked for once and remembered: ~/.boltbeam/engines.json.

This mirrors the chip autoscan (workflow/chips.py): the machine is read once, the answer is saved per machine and
never in the repo, and every reader sees the same engines with no env vars set. The order of authority is fixed:
an env var always wins; then PATH; then the saved file. The finders in engine_common, llama_bench_decode,
engine_kernels and role_compare read the saved file as their last fallback, so `providers.available()`, the
pipeline, `boltbeam doctor` and the screen agree.

    item (its env var)             looked for, in order; the first hit wins
    BOLTBEAM_LLAMA_BENCH           $PATH; <prefix>/opt/llama.cpp/bin; <llama.cpp checkout>/build*/bin, the builds that
                                   carry this machine's backend library (libggml-cuda.*, libggml-metal.*) first, then by
                                   name; /usr/local/bin
    BOLTBEAM_LLAMA_BATCHED_BENCH   the same places, llama-batched-bench
    BOLTBEAM_GGML_CUDA_SRC         <llama.cpp checkout>/ggml/src/ggml-cuda (the folder that holds mmvq.cu)
    BOLTBEAM_GGML_METAL            engine_kernels.ggml_library's candidates under the Homebrew prefixes
    BOLTBEAM_TINYGRAD_ROOT         the fork next to the checkout, ~/tinygrad-arkey*, ~/env/tinygrad*,
                                   ~/storage/*/tinygrad*; the first that role_compare.readiness calls ready, else
                                   the first that is a fork at all
    BOLTBEAM_VLLM_PYTHON           a venv python that can import vllm: */bin/python and */*/bin/python under
                                   ~/env, ~/storage, ~/venvs, ~/.venvs and ~/storage/*; NVIDIA machines only
    BOLTBEAM_TRTLLM_PYTHON         the same venvs, tensorrt_llm
    BOLTBEAM_OLLAMA                $PATH; /usr/local/bin; ~/.ollama; ~/storage/*/ollama/bin; Ollama.app

    llama.cpp checkouts            ~/env/llama.cpp, ~/llama.cpp, ~/src/llama.cpp, ~/storage/*/llama.cpp
    Homebrew prefixes              /opt/homebrew, /usr/local

Bounded: os.scandir to the depth stated, never the whole home; one python started per venv found (5 s each, at
most VENV_CAP of them); no network; never the GPU. A saved path that is gone from the disk is dropped by the
next scan, and `saved()` never returns a path that is not there.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping

from boltbeam.vocab import SCHEMA_ENGINES as SCHEMA

FILE_ENV = "BOLTBEAM_ENGINES_FILE"
VENV_CAP = 40  # pythons started in one scan, at most
PYTHON_TIMEOUT_S = 5.0
_PREFIXES = ("/opt/homebrew", "/usr/local")

# env var -> (short name, the engine it belongs to). The order is the order rows are shown.
ITEMS: tuple[tuple[str, str, str], ...] = (
  ("BOLTBEAM_LLAMA_BENCH", "llama-bench", "llama.cpp"),
  ("BOLTBEAM_LLAMA_BATCHED_BENCH", "llama-batched-bench", "llama.cpp"),
  ("BOLTBEAM_GGML_METAL", "ggml Metal library", "llama.cpp"),
  ("BOLTBEAM_GGML_CUDA_SRC", "ggml-cuda source", "llama.cpp"),
  ("BOLTBEAM_TINYGRAD_ROOT", "tinygrad fork", "tinygrad"),
  ("BOLTBEAM_VLLM_PYTHON", "vLLM python", "vllm"),
  ("BOLTBEAM_OLLAMA", "ollama", "ollama"),
  ("BOLTBEAM_TRTLLM_PYTHON", "TensorRT-LLM python", "tensorrt-llm"),
)
PRIMARY = {engine: env for env, _, engine in reversed(ITEMS)}  # the one item that says the engine is here
CUDA_ONLY = ("BOLTBEAM_VLLM_PYTHON", "BOLTBEAM_TRTLLM_PYTHON")  # not looked for on a Mac
_MODULES = {"BOLTBEAM_VLLM_PYTHON": "vllm", "BOLTBEAM_TRTLLM_PYTHON": "tensorrt_llm"}


# --- the saved file ---------------------------------------------------------------------------------------------

def engines_file() -> pathlib.Path:
  """Where this machine keeps its scan: $BOLTBEAM_ENGINES_FILE, else ~/.boltbeam/engines.json. A machine setting, read
  from the process environment (the finders' `env` argument is the engine settings a caller fakes, not this)."""
  return pathlib.Path(os.environ.get(FILE_ENV) or pathlib.Path.home() / ".boltbeam" / "engines.json").expanduser()


def load() -> dict[str, Any]:
  """The saved scan, or {} when there is none or it is not a scan."""
  try:
    data = json.loads(engines_file().read_text())
  except (OSError, ValueError):
    return {}
  return data if isinstance(data, dict) and data.get("schema") == SCHEMA else {}


def saved(env_var:str) -> str | None:
  """The path the last scan found for this env var, when it is still on the disk. The finders' last fallback."""
  row = (load().get("found") or {}).get(env_var) or {}
  path = row.get("path")
  return path if path and (_is_file(pathlib.Path(path)) or _is_dir(pathlib.Path(path))) else None


def save(result:dict[str, Any]) -> pathlib.Path:
  path = engines_file()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return path


# --- looking -----------------------------------------------------------------------------------------------------

# Every look at the disk tolerates a folder that cannot be read (a root-owned lost+found under ~/storage): it is
# simply not there for the scan. A permission error must never end the scan.

def _is_dir(p:pathlib.Path) -> bool:
  try:
    return p.is_dir()
  except OSError:
    return False


def _is_file(p:pathlib.Path) -> bool:
  try:
    return p.is_file()
  except OSError:
    return False


def _is_exe(p:pathlib.Path) -> bool:
  return _is_file(p) and os.access(p, os.X_OK)


def _glob(folder:pathlib.Path, pattern:str) -> list[pathlib.Path]:
  try:
    return sorted(folder.glob(pattern))
  except OSError:
    return []


def _subdirs(folder:pathlib.Path) -> list[pathlib.Path]:
  """The folders directly under folder, by name; [] when it is not a folder."""
  try:
    with os.scandir(folder) as it:
      return sorted((pathlib.Path(e.path) for e in it if e.is_dir(follow_symlinks=True)), key=lambda p: p.name)
  except OSError:
    return []


def _hit(path:pathlib.Path, how:str) -> dict[str, str]:
  return {"path": str(path), "how": how}


def _from_env(env_var:str, env:Mapping[str, str]) -> dict[str, str] | None:
  p = pathlib.Path(env[env_var]).expanduser() if env.get(env_var) else None
  if p is not None and (_is_file(p) or _is_dir(p)):
    return _hit(p, "env")
  return None


def _on_path(name:str, env:Mapping[str, str]) -> dict[str, str] | None:
  found = shutil.which(name, path=env.get("PATH"))
  return _hit(pathlib.Path(found), "path") if found else None


def _scan_hit(path:pathlib.Path) -> dict[str, str]:
  return _hit(path, f"scan: {path.parent}")


def llama_checkouts(home:pathlib.Path) -> list[pathlib.Path]:
  roots = [home / "env" / "llama.cpp", home / "llama.cpp", home / "src" / "llama.cpp"]
  roots += [d / "llama.cpp" for d in _subdirs(home / "storage")]
  return [r for r in roots if _is_dir(r)]


def backend_tag(env:Mapping[str, str] | None = None) -> str | None:
  """The ggml backend this machine runs: cuda where nvidia-smi is, metal on a Mac, else unknown."""
  env = os.environ if env is None else env
  if shutil.which("nvidia-smi", path=env.get("PATH")):
    return "cuda"
  return "metal" if sys.platform == "darwin" else None


def build_bins(checkout:pathlib.Path, tag:str | None) -> list[pathlib.Path]:
  """The bin folders of a llama.cpp checkout's build* folders, by name. With a backend tag, the builds whose bin
  carries libggml-<tag>.* come first (a `build` beside a `build-cuda` is often the CPU build)."""
  bins = [d / "bin" for d in _subdirs(checkout) if d.name.startswith("build")]
  if not tag:
    return bins
  has = lambda b: bool(_glob(b, f"libggml-{tag}.*"))  # noqa: E731
  return [b for b in bins if has(b)] + [b for b in bins if not has(b)]


def _program(name:str, env:Mapping[str, str], home:pathlib.Path, prefixes:Iterable[str], tag:str | None = None) -> dict[str, str] | None:
  """A llama.cpp program: PATH, the Homebrew keg, a checkout's build folders (the backend's builds first), /usr/local/bin."""
  if hit := _on_path(name, env):
    return hit
  places = [pathlib.Path(p) / "opt" / "llama.cpp" / "bin" / name for p in prefixes]
  for co in llama_checkouts(home):
    places += [b / name for b in build_bins(co, tag)]
  places.append(pathlib.Path("/usr/local/bin") / name)
  return next((_scan_hit(p) for p in places if _is_exe(p)), None)


def _cuda_source(home:pathlib.Path) -> dict[str, str] | None:
  for co in llama_checkouts(home):
    src = co / "ggml" / "src" / "ggml-cuda"
    if _is_file(src / "mmvq.cu"):
      return _scan_hit(src)
  return None


def _metal_library(bench:str | None) -> dict[str, str] | None:
  """engine_kernels.ggml_library's own candidates (the keg that holds llama-bench first); None when it raises."""
  from boltbeam.collectors import engine_kernels as ek
  try:
    rec = ek.ggml_library(bench)
  except ek.NoEngineSource:
    return None
  return _scan_hit(pathlib.Path(rec["path"]))


def default_checkout() -> pathlib.Path | None:
  """The BoltBeam checkout the fork would sit next to: walk up from the working folder to boltbeam/__init__.py, else
  the folder this package was imported from (an editable install), else none (a plain pip install)."""
  here = pathlib.Path.cwd().resolve()
  for d in (here, *here.parents):
    if (d / "boltbeam" / "__init__.py").is_file():
      return d
  pkg = pathlib.Path(__file__).resolve().parents[2]
  return pkg if (pkg / "boltbeam" / "__init__.py").is_file() and pkg.name != "site-packages" else None


def fork_candidates(home:pathlib.Path, checkout:pathlib.Path | None) -> list[pathlib.Path]:
  """Where a tinygrad fork may be: next to the checkout, then the home folders, each once."""
  from boltbeam.target.tinygrad_root import _SIBLING_NAMES
  out: list[pathlib.Path] = []
  if checkout:
    out += [checkout.parent / name for name in _SIBLING_NAMES]
  out += _glob(home, "tinygrad-arkey*") + _glob(home / "env", "tinygrad*")
  for d in _subdirs(home / "storage"):
    out += _glob(d, "tinygrad*")
  seen, uniq = set(), []
  for p in out:
    if _is_dir(p) and str(p) not in seen:
      seen.add(str(p))
      uniq.append(p)
  return uniq


def _fork(home:pathlib.Path, checkout:pathlib.Path | None) -> dict[str, str] | None:
  from boltbeam.search import role_compare
  forks = [p for p in fork_candidates(home, checkout) if _is_file(p / role_compare.PROVIDER)]
  ready = next((p for p in forks if role_compare.readiness(p)["ready"]), None)
  pick = ready or (forks[0] if forks else None)
  return _scan_hit(pick) if pick else None


def venv_pythons(home:pathlib.Path) -> list[pathlib.Path]:
  """Every */bin/python and */*/bin/python under the venv roots, each venv once (two venvs that link to the same
  interpreter are two venvs: the key is the venv folder, not the python it points at), capped at VENV_CAP."""
  roots = [home / "env", home / "storage", home / "venvs", home / ".venvs", *_subdirs(home / "storage")]
  out, seen = [], set()
  for root in roots:
    for one in _subdirs(root):
      for folder in (one, *_subdirs(one)):
        py = folder / "bin" / "python"
        try:
          key = str(folder.resolve())
        except OSError:
          continue
        if key not in seen and _is_exe(py):
          seen.add(key)
          out.append(py)
          if len(out) >= VENV_CAP:
            return out
  return out


def imports(python:pathlib.Path, modules:Iterable[str], timeout_s:float = PYTHON_TIMEOUT_S) -> set[str]:
  """Which of the modules that python can find (by spec only; importing vLLM takes seconds)."""
  code = ("import importlib.util as u, sys\n"
          "print(' '.join(m for m in sys.argv[1:] if u.find_spec(m) is not None))")
  try:
    proc = subprocess.run([str(python), "-I", "-c", code, *modules], capture_output=True, text=True, timeout=timeout_s)
  except (OSError, subprocess.SubprocessError):
    return set()
  return set(proc.stdout.split()) if proc.returncode == 0 else set()


def _venv_engines(env_vars:list[str], env:Mapping[str, str], home:pathlib.Path,
                  check:Callable[[pathlib.Path, Iterable[str]], set[str]]) -> dict[str, dict[str, str]]:
  """The engines installed in venvs: PATH's entry point first (the python beside it), then the venv folders."""
  from boltbeam.collectors import engine_common as ec
  found: dict[str, dict[str, str]] = {}
  entry = {"BOLTBEAM_VLLM_PYTHON": "vllm", "BOLTBEAM_TRTLLM_PYTHON": "trtllm-bench"}
  for var in env_vars:
    py = ec.python_with(var, entry[var], {"PATH": env.get("PATH", "")})
    if py and _MODULES[var] in check(pathlib.Path(py), [_MODULES[var]]):
      found[var] = _hit(pathlib.Path(py), "path")
  want = [v for v in env_vars if v not in found]
  for py in venv_pythons(home) if want else []:
    got = check(py, [_MODULES[v] for v in want])
    for var in list(want):
      if _MODULES[var] in got:
        found[var] = _hit(py, f"scan: {py.parent.parent}")
        want.remove(var)
    if not want:
      break
  return found


def _ollama(env:Mapping[str, str], home:pathlib.Path) -> dict[str, str] | None:
  if hit := _on_path("ollama", env):
    return hit
  places = [pathlib.Path("/usr/local/bin/ollama"), home / ".ollama" / "ollama", home / ".ollama" / "bin" / "ollama"]
  places += [d / "ollama" / "bin" / "ollama" for d in _subdirs(home / "storage")]
  places.append(pathlib.Path("/Applications/Ollama.app/Contents/Resources/ollama"))
  return next((_scan_hit(p) for p in places if _is_exe(p)), None)


def scan(*, home:pathlib.Path | None = None, env:Mapping[str, str] | None = None, prefixes:Iterable[str] = _PREFIXES,
         checkout:pathlib.Path | None = None, cuda_engines:bool | None = None, tag:str | None = None,
         check:Callable[[pathlib.Path, Iterable[str]], set[str]] = imports) -> dict[str, Any]:
  """Look for every engine once. The result is the saved file's content: found rows by env var, with how each was
  found and when, and the items looked for but not found. checkout: the BoltBeam checkout the fork would sit next
  to (default: default_checkout). cuda_engines: look for vLLM and TensorRT-LLM (default: not on a Mac). tag: the
  ggml backend whose llama.cpp builds come first (default: backend_tag)."""
  t0 = time.monotonic()
  env = os.environ if env is None else env
  home = pathlib.Path(home) if home else pathlib.Path.home()
  cuda = cuda_engines if cuda_engines is not None else sys.platform != "darwin"
  checkout = checkout if checkout is not None else default_checkout()
  tag = tag if tag is not None else backend_tag(env)
  found: dict[str, dict[str, str]] = {}
  skipped: list[str] = []
  for var, _, _ in ITEMS:
    if hit := _from_env(var, env):
      found[var] = hit
  if "BOLTBEAM_LLAMA_BENCH" not in found and (hit := _program("llama-bench", env, home, prefixes, tag)):
    found["BOLTBEAM_LLAMA_BENCH"] = hit
  if "BOLTBEAM_LLAMA_BATCHED_BENCH" not in found and (hit := _program("llama-batched-bench", env, home, prefixes, tag)):
    found["BOLTBEAM_LLAMA_BATCHED_BENCH"] = hit
  if "BOLTBEAM_GGML_METAL" not in found and sys.platform == "darwin":
    if hit := _metal_library((found.get("BOLTBEAM_LLAMA_BENCH") or {}).get("path")):
      found["BOLTBEAM_GGML_METAL"] = hit
  if "BOLTBEAM_GGML_CUDA_SRC" not in found and (hit := _cuda_source(home)):
    found["BOLTBEAM_GGML_CUDA_SRC"] = hit
  if "BOLTBEAM_TINYGRAD_ROOT" not in found and (hit := _fork(home, checkout)):
    found["BOLTBEAM_TINYGRAD_ROOT"] = hit
  if cuda:
    found.update(_venv_engines([v for v in CUDA_ONLY if v not in found], env, home, check))
  else:
    skipped += [v for v in CUDA_ONLY if v not in found]
  if "BOLTBEAM_OLLAMA" not in found and (hit := _ollama(env, home)):
    found["BOLTBEAM_OLLAMA"] = hit
  now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  for row in found.values():
    row["found_at"] = now
  return {"schema": SCHEMA, "scanned_at": now, "seconds": round(time.monotonic() - t0, 3),
          "found": {var: found[var] for var, _, _ in ITEMS if var in found},
          "missing": [var for var, _, _ in ITEMS if var not in found and var not in skipped],
          "skipped": {var: "NVIDIA machines only" for var in skipped}}


def scan_and_save(env:Mapping[str, str] | None = None, **kw) -> dict[str, Any]:
  """Scan, write the file, and return the result with its path under "file"."""
  result = scan(env=env, **kw)
  return {**result, "file": str(save(result))}


def rows(result:dict[str, Any]) -> list[dict[str, Any]]:
  """One row per item, in ITEMS order, for a screen or doctor: the path and how it was found, or neither."""
  found, skipped = result.get("found") or {}, result.get("skipped") or {}
  return [{"item": name, "env": var, "engine": engine, "path": (found.get(var) or {}).get("path"),
           "how": (found.get(var) or {}).get("how"), "found_at": (found.get(var) or {}).get("found_at"),
           "skipped": skipped.get(var)} for var, name, engine in ITEMS]
