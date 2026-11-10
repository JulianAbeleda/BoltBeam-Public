"""The tinygrad fork and its search provider: where the fork is, its python, whether it is ready to search, and one
request to the provider (capability, describe) with the device the search runs on.

The search runtime table (boltbeam/data/search_runtime.json) says which tinygrad devices a BoltBeam backend runs on and
the whole-model A/B's environment; it is a table, not code. The fork is $BOLTBEAM_TINYGRAD_ROOT, the one the engine
scan saved, or tinygrad-arkey-exp next to this checkout; its interpreter is never a link inside the checkout.
"""
from __future__ import annotations

import functools, json, os, pathlib, subprocess
from collections.abc import Mapping
from typing import Any

FORK_URL = "https://github.com/JulianAbeleda/tinygrad-arkey"
PROVIDER = "extra/llm_research/search_provider.py"
PROTOCOL = "tinygrad.search_provider.v1"
RUNTIME_TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "search_runtime.json"


@functools.lru_cache(maxsize=1)
def _runtime_table() -> dict[str, Any]:
  return json.loads(RUNTIME_TABLE.read_text())


def runtime_row(backend:str | None) -> dict[str, Any] | None:
  """The search runtime row of a BoltBeam target backend (Metal, CUDA, AMD), or None when the table has none."""
  rows = _runtime_table()["backends"]
  return next((dict(v) for k, v in rows.items() if backend and k.lower() == str(backend).lower()), None)


def decode_env(backend:str | None, device:str) -> dict[str, str]:
  """The whole-model A/B environment: the device the search served on and the backend's recorded switches."""
  return {"DEV": device, **dict((runtime_row(backend) or {}).get("decode_env") or {})}


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
  .venv folder when git ignores it. Never a link placed in the checkout: git counts that as an untracked file, the
  provider then reports a dirty tree, and the search refuses every candidate."""
  if env := os.environ.get(PYTHON_ENV):
    return pathlib.Path(env).expanduser()
  if env := os.environ.get(VENV_ENV):
    return pathlib.Path(env).expanduser() / "bin" / "python"
  return root / ".venv" / "bin" / "python"


def _venv_link(root:pathlib.Path) -> bool:
  """The python would come from a .venv link inside the checkout (no env set)."""
  return not (os.environ.get(PYTHON_ENV) or os.environ.get(VENV_ENV)) and (root / ".venv").is_symlink()


def _has_numpy(python:pathlib.Path) -> bool:
  venv = python.parent.parent
  if any((venv / "lib").glob("python3*/site-packages/numpy")):
    return True
  try:
    return subprocess.run([str(python), "-c", "import numpy"], capture_output=True, timeout=60).returncode == 0
  except (OSError, subprocess.TimeoutExpired):
    return False


def readiness(root:pathlib.Path | None = None, model:str | None = None) -> dict[str, Any]:
  """Can this machine compare kernels now? Each missing piece names the one command that fixes it."""
  root = pathlib.Path(root) if root else default_fork_root()
  venv = f"cd {root} && python3.12 -m venv .venv && .venv/bin/pip install numpy"
  out: dict[str, Any] = {"ready": False, "fork": str(root), "missing": None, "message": None, "fix": None}
  if not (root / PROVIDER).is_file():
    return out | {"missing": "fork", "message": f"The tinygrad fork is not at {root}.",
                  "fix": f"git clone -b exp {FORK_URL} {root}"}
  if _venv_link(root):
    return out | {"missing": "venv", "message": f"{root / '.venv'} is a link. Git counts it as a change, so the "
                  "search refuses every kernel.",
                  "fix": f"rm {root / '.venv'} && export {PYTHON_ENV}={(root / '.venv').resolve() / 'bin' / 'python'}"}
  python = fork_python(root)
  if not python.is_file():
    return out | {"missing": "venv", "message": f"The fork's python is not at {python}.", "fix": venv}
  if not _has_numpy(python):
    return out | {"missing": "numpy", "message": "The fork's python has no numpy.",
                  "fix": f"{python} -m pip install numpy"}
  if model is not None and not pathlib.Path(model).expanduser().is_file():
    return out | {"missing": "model", "message": f"The model file is not at {model}.", "fix": None}
  return out | {"ready": True}


def fork_changes(root:pathlib.Path) -> list[str]:
  """The fork's uncommitted paths, as git status shows them. The provider refuses to search a checkout with any."""
  try:
    text = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                          check=True).stdout
  except (OSError, subprocess.CalledProcessError) as exc:
    return [f"git status failed: {exc}"]
  return [line[3:] for line in text.splitlines() if line.strip()]


def clean_refusal(root:pathlib.Path) -> str | None:
  """Why the search cannot run on this fork checkout, or None. Checked before the search, so one sentence says it
  instead of every candidate being refused with provider_describe:dirty_or_missing_clean_evidence."""
  changes = fork_changes(root)
  if not changes:
    return None
  shown = ", ".join(changes[:5]) + (f" and {len(changes) - 5} more" if len(changes) > 5 else "")
  hint = f" Put the venv outside the checkout and set {PYTHON_ENV}." if any(c.rstrip("/") == ".venv" for c in changes) else ""
  return f"the tinygrad fork at {root} has uncommitted changes ({shown}); the search runs on a clean checkout only.{hint}"


# --- one request to the provider ---------------------------------------------------------------------------

def ask(root:pathlib.Path, action:str, payload:Mapping[str, Any], timeout_s:float = 180.0) -> dict[str, Any]:
  """One request to the fork's provider, in its own process; the result, or RuntimeError with the provider's words."""
  line = json.dumps({"protocol": PROTOCOL, "request_id": action, "action": action, "payload": dict(payload)})
  proc = subprocess.run([str(fork_python(root)), PROVIDER], cwd=root, input=line + "\n", capture_output=True, text=True,
                        timeout=timeout_s, env={**os.environ, "PYTHONPATH": "."})
  reply = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
  if reply.get("status") != "ok":
    why = (reply.get("error") or {}).get("message") or proc.stderr.strip()[-300:] or "no reply"
    raise RuntimeError(f"provider {action} failed: {why}")
  return reply["result"]


def capability(root:pathlib.Path, devices:list[str]) -> dict[str, Any]:
  """The provider's own account of the devices it can run: one row per named device, opened or with its reason."""
  return ask(root, "capability", {"devices": list(devices)})


def describe(root:pathlib.Path, device:str | None = None, *, shapes:list[dict[str, int]] | None = None) -> dict[str, Any]:
  """The provider's describe; with shapes ([{n, k}]), its fused emitters' validate verdicts at each shape too."""
  return ask(root, "describe",({"device": device} if device else {}) | ({"shapes": list(shapes)} if shapes else {}))


def pick_device(backend:str | None, reply:Mapping[str, Any]) -> tuple[str | None, str | None]:
  """(device, None) for the first of the backend's devices the provider opened, else (None, why)."""
  row = runtime_row(backend)
  if row is None:
    return None, f"the search runtime table (data/search_runtime.json) has no row for {backend or 'an unknown backend'}"
  rows = {str(r.get("device")).upper(): r for r in reply.get("devices") or [] if isinstance(r, Mapping)}
  for name in row["tinygrad_devices"]:
    got = rows.get(name.upper())
    if got and got.get("opened"):
      return str(got["device"]), None
  said = "; ".join(f"{name}: {(rows.get(name.upper()) or {}).get('reason') or 'not reported'}" for name in row["tinygrad_devices"])
  return None, f"the provider runs on none of this {backend} machine's devices ({said})"
