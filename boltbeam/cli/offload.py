"""`boltbeam offload MODEL --target T [--engine E]`: the offload page, the same command as
`python -m boltbeam.workflow.screen offload` (workflow/screen.py owns the flags and the answer; this only makes it
reachable from an install that has the `boltbeam` command and not the module path in its fingers)."""
from __future__ import annotations

import argparse


def cmd_offload(args) -> int:
  from boltbeam.workflow import screen
  return screen.main(["offload", *args.rest], prog="boltbeam")


def register(sub) -> None:
  p = sub.add_parser("offload", help="the offload page for a model on this machine: what goes to the card and to RAM, each setup's limits, the pinned-copy gap (flags as `boltbeam offload --help`)", add_help=False)
  p.add_argument("rest", nargs=argparse.REMAINDER)
  p.set_defaults(fn=cmd_offload)
