"""The engine scan's saved file (~/.boltbeam/engines.json) and its items, read without scanning (engine_scan.py looks and
writes): the finders' last fallback, apart so a finder that reads one saved path does not import the scan."""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any

from boltbeam.vocab import SCHEMA_ENGINES as SCHEMA

FILE_ENV = "BOLTBEAM_ENGINES_FILE"

# env var -> (short name, the engine it belongs to). The order is the order rows are shown.
ITEMS: tuple[tuple[str, str, str], ...] = (
  ("BOLTBEAM_LLAMA_BENCH", "llama-bench", "llama.cpp"),
  ("BOLTBEAM_LLAMA_BATCHED_BENCH", "llama-batched-bench", "llama.cpp"),
  ("BOLTBEAM_GGML_METAL", "ggml Metal library", "llama.cpp"),
  ("BOLTBEAM_GGML_CUDA_SRC", "ggml-cuda source", "llama.cpp"),
  ("BOLTBEAM_TINYGRAD_ROOT", "tinygrad fork", "arkey"),
  ("BOLTBEAM_VLLM_PYTHON", "vLLM python", "vllm"),
  ("BOLTBEAM_OLLAMA", "ollama", "ollama"),
  ("BOLTBEAM_TRTLLM_PYTHON", "TensorRT-LLM python", "tensorrt-llm"),
)
PRIMARY = {engine: env for env, _, engine in reversed(ITEMS)}  # the one item that says the engine is here


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


# Every look at the disk tolerates a folder that cannot be read (a root-owned lost+found under ~/storage): it is
# simply not there for the scan. A permission error must never end the scan.

def _scan_error(exc:OSError, path:pathlib.Path) -> None:
  """Absent paths are ordinary; inaccessible paths are reported; I/O faults remain errors."""
  import warnings
  if isinstance(exc, (FileNotFoundError, NotADirectoryError)): return
  if isinstance(exc, PermissionError):
    warnings.warn(f"engine scan skipped {path}: {exc}", RuntimeWarning)
    return
  raise exc


def _is_dir(p:pathlib.Path) -> bool:
  try:
    return p.is_dir()
  except OSError as exc:
    _scan_error(exc, p)
    return False


def _is_file(p:pathlib.Path) -> bool:
  try:
    return p.is_file()
  except OSError as exc:
    _scan_error(exc, p)
    return False
