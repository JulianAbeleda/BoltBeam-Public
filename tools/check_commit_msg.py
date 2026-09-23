#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys

ALLOWED_PREFIXES = (
  "profile",
  "search",
  "policy",
  "eval",
  "synth",
  "cli",
  "schema",
  "docs",
  "test",
  "repo",
  "trace",
  "target",
  "bench",
  "path",
  "collectors",
)

PREFIX_RE = re.compile(rf"^\[({'|'.join(ALLOWED_PREFIXES)})\]( NFC -)? .+")


def check_message(message:str) -> str | None:
  first = message.splitlines()[0].strip() if message.splitlines() else ""
  if first.startswith(("Merge ", "Revert ")): return None
  if PREFIX_RE.match(first): return None
  allowed = ", ".join(f"[{p}]" for p in ALLOWED_PREFIXES)
  return (
    "commit-msg: need BoltBeam subsystem prefix. "
    f"Allowed prefixes: {allowed}. "
    "Use optional 'NFC -' after the prefix for behavior-preserving changes."
  )


def main(argv:list[str] | None=None) -> int:
  args = argv if argv is not None else sys.argv[1:]
  if len(args) != 1:
    print("usage: check_commit_msg.py COMMIT_MSG_FILE", file=sys.stderr)
    return 2
  msg = pathlib.Path(args[0]).read_text()
  err = check_message(msg)
  if err:
    print(err, file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
