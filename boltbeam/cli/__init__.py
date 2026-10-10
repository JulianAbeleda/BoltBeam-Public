from __future__ import annotations

import argparse
import sys
import textwrap

from boltbeam.cli import analysis, doctor, gemm, ncu, search, profile, roofline, selfcheck, tui, workflow, lifecycle

_MODULES = (analysis, search, profile, roofline, workflow, gemm, ncu, lifecycle, selfcheck, doctor, tui)

# The core group `boltbeam --help` shows first, in the order a reader uses them.
# Every other command still runs. Scripts and papers call them by name. --help lists them after the core group.
CORE: tuple[tuple[str, tuple[str, ...]], ...] = (
  ("Check the install", ("selfcheck", "doctor")),
  ("Open the screen", ("tui",)),
  ("Read a model (no GPU)", ("inspect", "roofline-theoretical")),
  ("Run the pipeline", ("load", "autoscan", "analyze", "output")),
  ("Bring in measurements", ("ingest-timing", "ingest-probe", "collect-hw-trace", "import-hw-trace",
                             "metal-measure", "metal-bandwidth")),
  ("Record what you learned", ("evaluate", "ledger", "reopen-check")),
)


class _Recorder:
  """Wraps the subparsers object. It keeps each command's one-line help for the grouped listing,
  and passes it on as the command's own description, so `boltbeam <cmd> --help` still shows it."""

  def __init__(self, sub):
    self.sub, self.help = sub, {}

  def add_parser(self, name, **kw):
    text = kw.pop("help", None)
    self.help[name] = text or ""
    kw.setdefault("description", text)
    return self.sub.add_parser(name, **kw)


def _listing(helps:dict[str, str]) -> str:
  core = [name for _, names in CORE for name in names]
  missing = [name for name in core if name not in helps]
  if missing:  # a renamed core command must fail loudly, not vanish from the listing
    raise RuntimeError(f"core commands not registered: {missing}")
  lines = []
  for title, names in CORE:
    lines.append(f"{title}:")
    for name in names:
      first = helps[name].split(";")[0].strip()
      lines.append(f"  {name:<22}{first}")
  rest = sorted(n for n in helps if n not in core)
  lines.append("")
  lines.append(f"Advanced and campaign commands ({len(rest)}; `boltbeam <command> --help` for each):")
  lines += textwrap.wrap(", ".join(rest), width=100, initial_indent="  ", subsequent_indent="  ",
                         break_on_hyphens=False)
  return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(prog="boltbeam", description="Translate weights into codegen search.",
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>",
                          help="one of the commands listed below")
  rec = _Recorder(sub)
  for module in _MODULES:
    module.register(rec)
  ap.epilog = _listing(rec.help)
  return ap


def main(argv:list[str] | None=None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  if argv[:1] == ["tui"] and argv[1:2] not in (["-h"], ["--help"]):
    return tui.run(argv[1:])  # every flag passes through to boltbeam-tui; argparse would claim the dashes
  args = build_parser().parse_args(argv)
  return args.fn(args)


if __name__ == "__main__":
  raise SystemExit(main())
