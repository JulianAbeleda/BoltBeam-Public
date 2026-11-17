"""`boltbeam mods [--apply]`: the arkey_ie_mods mods that fit this machine's engine build, and with --apply the modded
build. The same command as `python -m boltbeam.workflow.screen mods` (workflow/mods.py owns the flags, the plan and the build,
screen.mods_page the record); this makes it reachable from an install with only the `boltbeam` command,
in plain lines unless --json."""
from __future__ import annotations

import json
import sys


def cmd_mods(args) -> int:
  from boltbeam.core.refused import Refused
  from boltbeam.workflow import mods, screen
  try:
    res = screen.mods_page(args, say=(lambda line: print(line, file=sys.stderr, flush=True)) if args.apply else (lambda line: None))
  except Refused as exc:  # a bad --target, or no single engine to take
    res = {"kind": "mods", "refused": str(exc)}
  print(json.dumps(res, indent=1, default=str) if args.json else mods.text(res))
  return 1 if res.get("refused") else 0


def register(sub) -> None:
  from boltbeam.workflow import mods
  p = sub.add_parser("mods", help=mods.HELP)
  mods.arguments(p)
  p.add_argument("--json", action="store_true", help="the whole record as JSON")
  p.set_defaults(fn=cmd_mods)
