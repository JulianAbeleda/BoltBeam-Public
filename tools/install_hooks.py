#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import stat
import subprocess


def main() -> int:
  root = pathlib.Path(__file__).resolve().parents[1]
  hook = root / ".githooks" / "commit-msg"
  hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
  subprocess.check_call(["git", "config", "core.hooksPath", ".githooks"], cwd=root)
  print("installed BoltBeam git hooks: core.hooksPath=.githooks")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
