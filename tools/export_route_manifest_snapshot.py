#!/usr/bin/env python3
"""Export BoltBeam's canonical route policy as a byte-identical, hashed snapshot."""
from __future__ import annotations
import argparse
import hashlib
import pathlib

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "boltbeam/policy/assets/route_manifest.v1.json"

def export(destination: pathlib.Path) -> str:
  raw = SOURCE.read_bytes()
  expected = (SOURCE.with_suffix(SOURCE.suffix + ".sha256")).read_text().split()[0]
  digest = hashlib.sha256(raw).hexdigest()
  if digest != expected: raise RuntimeError("BoltBeam authority asset hash mismatch")
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_bytes(raw)
  destination.with_suffix(destination.suffix + ".sha256").write_text(f"{digest}  {destination.name}\n")
  return digest

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("destination", type=pathlib.Path)
  args = parser.parse_args()
  print(export(args.destination))
