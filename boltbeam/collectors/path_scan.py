"""What a matrix-path scan (hip_path_scan, cuda_path_scan) keeps, for any vendor: one record per toolchain and
target, cached by the toolchain's identity, kept in the chip profile (profile_row's matrix_path_scan) and read back
from there (targets.stored_path_scan). A record holds the paths as the vendor probe's own rows, the candidates it
excluded or the target rejected with the toolchain's words, and the paths no weight format maps to (unmapped)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable

SCHEMA = "boltbeam.matrix_path_scan.v1"


def cached(ident:str, cache:pathlib.Path, refresh:bool, build:Callable[[], dict[str, Any]]) -> dict[str, Any]:
  """The record for one toolchain identity: read from the cache, else built, timed and (when it found paths) kept."""
  f = cache / f"path-scan-{hashlib.sha256(ident.encode()).hexdigest()[:16]}.json"
  try:
    if f.is_file() and not refresh:
      return json.loads(f.read_text())
  except ValueError:  # a record cut short is scanned again, never trusted
    pass
  cache.mkdir(parents=True, exist_ok=True)
  t0 = time.monotonic()
  rec = {"schema": SCHEMA, "excluded": {}, "rejected": {}, "paths": [], "unmapped": [], **build()}
  rec["seconds"] = round(time.monotonic() - t0, 2)
  if usable(rec):
    keep(f, json.dumps(rec, indent=1, sort_keys=True).encode())
  return rec


def keep(f:pathlib.Path, data:bytes) -> pathlib.Path:
  """Write a file whole (a temporary name, then renamed), so a parallel reader never sees it half written."""
  (tmp := f.with_name(f"{f.name}.{os.getpid()}.{id(data)}.tmp")).write_bytes(data)
  os.replace(tmp, f)
  return f


def usable(rec:dict[str, Any] | None) -> bool:
  """A record that found at least one matrix path (the vector FMA row alone is not a scan result)."""
  return any(p.get("kind") == "mma" for p in (rec or {}).get("paths") or ())


def rows(rec:dict[str, Any] | None, cls) -> tuple:
  """A record's table as the probe's rows; () when it found none."""
  return tuple(cls(**p) for p in rec["paths"]) if usable(rec) else ()


def main(argv:list[str] | None, prog:str, scan:Callable[..., Any], tools:tuple[str, ...]) -> int:
  """`python -m <prog> TARGET [--tool PATH ...] [--refresh]`: the record as JSON."""
  a = argparse.ArgumentParser(prog=f"python -m {prog}")
  a.add_argument("target")
  for t in tools:
    a.add_argument(f"--{t}")
  a.add_argument("--refresh", action="store_true")
  ns = vars(a.parse_args(argv))
  sys.stdout.write(json.dumps(scan(ns.pop("target"), **ns), indent=1, sort_keys=True) + "\n")
  return 0
