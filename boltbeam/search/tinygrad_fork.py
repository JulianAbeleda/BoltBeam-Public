"""The tinygrad fork as an engine under test (arkey): where it is, its python, and whether it is ready to time roles.
The fork is $BOLTBEAM_TINYGRAD_ROOT, the one the engine scan saved, or tinygrad-arkey-exp next to this checkout.

It no longer serves a kernel search: its search provider left tinygrad-arkey with the kernel generator, and the per-role
search that asked it was retired 2026-11-21 (search/search_stage.RETIRED). The generator is reached only through its
contract (search/generator_repo).
"""
from __future__ import annotations

import os, pathlib, subprocess
from typing import Any

from boltbeam.collectors import engine_families

FORK_URL = engine_families.repo("tinygrad", "own")  # the fork's repo is a row of data/engine_families.json
ENGINE = "tinygrad/llm/decode_kernels.py"  # a file only the fork (the engine) has: how a checkout is known as the fork


# --- readiness: what the screen says before anything starts ---------------------------------------------------

def default_fork_root() -> pathlib.Path:
  """$BOLTBEAM_TINYGRAD_ROOT (or the older TINYGRAD_ROOT; target/tinygrad_root.py reads both), else the fork the
  engine scan saved (collectors/engine_scan.py), else tinygrad-arkey-exp next to this BoltBeam checkout."""
  from boltbeam.collectors import engine_scan
  from boltbeam.target.tinygrad_root import BOLTBEAM_TINYGRAD_ROOT_ENV, configured_root
  if env := configured_root():
    return pathlib.Path(env).expanduser()
  if kept := engine_scan.saved(BOLTBEAM_TINYGRAD_ROOT_ENV):
    return pathlib.Path(kept)
  return pathlib.Path(__file__).resolve().parents[2].parent / "tinygrad-arkey-exp"


PYTHON_ENV = "BOLTBEAM_TINYGRAD_PYTHON"  # the fork's interpreter, outside the checkout
VENV_ENV = "BOLTBEAM_TINYGRAD_VENV"  # or the venv folder that holds it (bin/python)


def fork_python(root:pathlib.Path) -> pathlib.Path:
  """The fork's interpreter: $BOLTBEAM_TINYGRAD_PYTHON, else $BOLTBEAM_TINYGRAD_VENV/bin/python, else the fork's own
  .venv folder."""
  if env := os.environ.get(PYTHON_ENV):
    return pathlib.Path(env).expanduser()
  if env := os.environ.get(VENV_ENV):
    return pathlib.Path(env).expanduser() / "bin" / "python"
  return root / ".venv" / "bin" / "python"


def _has_numpy(python:pathlib.Path) -> bool:
  venv = python.parent.parent
  if any((venv / "lib").glob("python3*/site-packages/numpy")):
    return True
  try:
    return subprocess.run([str(python), "-c", "import numpy"], capture_output=True, timeout=60).returncode == 0
  except (OSError, subprocess.TimeoutExpired):
    return False


def readiness(root:pathlib.Path | None = None, model:str | None = None) -> dict[str, Any]:
  """Can this machine run the fork now? Each missing piece names the one command that fixes it."""
  root, what = pathlib.Path(root) if root else default_fork_root(), "The tinygrad fork"
  out: dict[str, Any] = {"ready": False, "fork": str(root), "missing": None, "message": None, "fix": None}
  if not (root / ENGINE).is_file():
    return out | {"missing": "fork", "message": f"{what} is not at {root}.", "fix": f"git clone -b exp {FORK_URL} {root}"}
  python = fork_python(root)
  if not python.is_file():
    return out | {"missing": "venv", "message": f"{what}'s python is not at {python}.",
                  "fix": f"cd {root} && python3.12 -m venv .venv && .venv/bin/pip install numpy"}
  if not _has_numpy(python):
    return out | {"missing": "numpy", "message": f"{what}'s python has no numpy.", "fix": f"{python} -m pip install numpy"}
  if model is not None and not pathlib.Path(model).expanduser().is_file():
    return out | {"missing": "model", "message": f"The model file is not at {model}.", "fix": None}
  return out | {"ready": True}
