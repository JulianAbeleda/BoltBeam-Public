#!/usr/bin/env python3
"""Classify every tracked file under bench/ by content, not by path.

Emits docs/task_workflow/output/bench-classification-20260731.json: one row
per tracked bench/ file with `classification` in {verdict, measurement,
unreadable} and `reason` naming the field or pattern that decided it.

Decision order per file:
  1. Read the bytes. If they do not decode as UTF-8, or do not parse as JSON,
     classify `unreadable` and record why (not a silent bucket).
  2. If the parsed JSON is an object with a top-level `status` or `verdict`
     key, classify `verdict` and cite the key and its value.
  3. Else, if the filename matches the documented verdict-naming convention
     (*refutation*, *blocked*, *promotion*, *validation* — case-insensitive),
     classify `verdict` via that filename pattern. This is the path used by
     scope §1/§3.1 for artifacts that carry their verdict in the name rather
     than a top-level key.
  4. Else classify `measurement`.

This is read-only: it deletes nothing and does not move any file.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "docs/task_workflow/output/bench-classification-20260731.json"

VERDICT_NAME_PATTERN = re.compile(r"refutation|blocked|promotion|validation", re.IGNORECASE)


def tracked_bench_files() -> list[str]:
  out = subprocess.run(
    ["git", "ls-files", "bench"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
  ).stdout
  return [line for line in out.splitlines() if line]


def classify_one(rel_path: str) -> dict:
  abs_path = REPO_ROOT / rel_path
  raw = abs_path.read_bytes()

  try:
    text = raw.decode("utf-8")
  except UnicodeDecodeError as exc:
    return {
      "path": rel_path,
      "classification": "unreadable",
      "reason": f"not UTF-8 text ({exc})",
    }

  try:
    data = json.loads(text)
  except json.JSONDecodeError as exc:
    snippet = text.lstrip()[:40].replace("\n", "\\n")
    return {
      "path": rel_path,
      "classification": "unreadable",
      "reason": (
        f"not valid JSON: {exc} — first non-blank bytes are {snippet!r}, "
        "i.e. this is prose/Markdown, not a JSON artifact"
      ),
    }

  if isinstance(data, dict) and "status" in data:
    return {
      "path": rel_path,
      "classification": "verdict",
      "reason": f"top-level 'status' key = {data['status']!r}",
    }

  if isinstance(data, dict) and "verdict" in data:
    value = data["verdict"]
    value_repr = repr(value)[:80]
    return {
      "path": rel_path,
      "classification": "verdict",
      "reason": f"top-level 'verdict' key = {value_repr}",
    }

  if VERDICT_NAME_PATTERN.search(rel_path):
    return {
      "path": rel_path,
      "classification": "verdict",
      "reason": (
        "filename convention match "
        f"({VERDICT_NAME_PATTERN.search(rel_path).group(0)!r}); "
        "no top-level status/verdict key found in content"
      ),
    }

  top_keys = list(data.keys()) if isinstance(data, dict) else None
  return {
    "path": rel_path,
    "classification": "measurement",
    "reason": (
      "no top-level status/verdict key and no verdict filename pattern; "
      f"top-level keys={top_keys}"
      if top_keys is not None
      else f"parsed JSON is a {type(data).__name__}, not an object; no verdict signal"
    ),
  }


def main(argv: list[str] | None = None) -> int:
  files = tracked_bench_files()
  rows = [classify_one(f) for f in files]
  rows.sort(key=lambda r: r["path"])

  counts = {"verdict": 0, "measurement": 0, "unreadable": 0}
  for row in rows:
    counts[row["classification"]] += 1

  manifest = {
    "schema": "bench_classification.v1",
    "generated_by": "tools/classify_bench.py",
    "total_files": len(rows),
    "counts": counts,
    "rows": rows,
  }

  MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
  MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

  print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
  print(f"total={len(rows)} verdict={counts['verdict']} "
        f"measurement={counts['measurement']} unreadable={counts['unreadable']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
