from __future__ import annotations

import argparse

from boltbeam.cli import analysis, search, profile, roofline, workflow

_MODULES = (analysis, search, profile, roofline, workflow)


def main(argv:list[str] | None=None) -> int:
  ap = argparse.ArgumentParser(prog="boltbeam", description="Translate weights into codegen search.")
  sub = ap.add_subparsers(dest="cmd", required=True)
  for module in _MODULES:
    module.register(sub)
  args = ap.parse_args(argv)
  return args.fn(args)


if __name__ == "__main__":
  raise SystemExit(main())
