"""The kernel comparison's files in a run folder and their provenance: the kernel_compare folder, the status and
fusions file names, how a record is written (sorted JSON, its sha256 returned), the label every record gives its
times, and the stamps a record carries (the time, a file's sha256, BoltBeam's own git state).
"""
from __future__ import annotations

import datetime as _dt
import hashlib, json, pathlib, subprocess
from typing import Any

FOLDER = "kernel_compare"
STATUS_FILE = "search_status.json"
FUSIONS_FILE = "fusions.json"
TIMING_SOURCE = "tinygrad runtime (whole model) and BoltBeam's kernel timer (each kernel alone)"


def sha256_file(path:pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""): digest.update(chunk)
  return digest.hexdigest()


def now() -> str:
  return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_state(repo:pathlib.Path) -> tuple[str | None, bool]:
  try:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
    return rev, dirty
  except (OSError, subprocess.CalledProcessError):
    return None, True


def write(path:pathlib.Path, obj:Any) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
  path.write_text(text)
  return hashlib.sha256(text.encode()).hexdigest()
