"""Where llama.cpp's ggml-cuda source (mmvq.cu) is on this machine, found without the kernel adapter (engine_kernels.py)
that builds from it: gemv_reference.iq3s_grid reads ggml-common.h beside it."""
from __future__ import annotations

import hashlib
import os
import pathlib
from typing import Any

from boltbeam.collectors.engine_saved import saved

GGML_CUDA_ENV = "BOLTBEAM_GGML_CUDA_SRC"  # the ggml-cuda source folder (llama.cpp/ggml/src/ggml-cuda)
_MMVQ_CANDIDATES = ("~/env/llama.cpp/ggml/src/ggml-cuda", "~/llama.cpp/ggml/src/ggml-cuda", "/usr/local/src/llama.cpp/ggml/src/ggml-cuda")


class NoEngineSource(RuntimeError):
  """The engine's shader source is not on this machine, so its kernels cannot be timed here."""


def _file_record(path:pathlib.Path) -> dict[str, Any]:
  real = path.resolve()
  version = None
  parts = real.parts
  if "Cellar" in parts:  # /opt/homebrew/Cellar/ggml/0.19.0/libexec/...: the formula and its version
    i = parts.index("Cellar")
    version = f"{parts[i + 1]} {parts[i + 2]}" if len(parts) > i + 2 else None
  return {"path": str(path), "real_path": str(real), "version": version, "bytes": real.stat().st_size,
          "sha256": hashlib.sha256(real.read_bytes()).hexdigest(), "mtime": _mtime(real)}


def _mtime(path:pathlib.Path) -> str:
  import datetime
  return datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mmvq_source_dir() -> dict[str, Any]:
  """The installed llama.cpp's ggml-cuda source folder: $BOLTBEAM_GGML_CUDA_SRC, a known checkout, or what the
  engine scan saved (engine_scan.py)."""
  tried = []
  kept = saved(GGML_CUDA_ENV)
  for cand in ([os.environ[GGML_CUDA_ENV]] if os.environ.get(GGML_CUDA_ENV) else []) + list(_MMVQ_CANDIDATES) + ([kept] if kept else []):
    p = pathlib.Path(cand).expanduser()
    tried.append(str(p))
    if (p / "mmvq.cu").is_file():
      rec = _file_record(p / "mmvq.cu")
      git = p.parents[2] / ".git" if len(p.parents) > 2 else None
      head = None
      if git and (git / "HEAD").is_file():
        ref = (git / "HEAD").read_text().strip()
        head = (git / ref.split(" ", 1)[1]).read_text().strip()[:12] if ref.startswith("ref:") and (git / ref.split(" ", 1)[1]).is_file() else ref[:12]
      return {**rec, "dir": str(p), "commit": head}
  raise NoEngineSource(f"llama.cpp's ggml-cuda source (mmvq.cu) was not found; looked in {', '.join(tried)}. "
                       f"Set {GGML_CUDA_ENV} to llama.cpp/ggml/src/ggml-cuda")
