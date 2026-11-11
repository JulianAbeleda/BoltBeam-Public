"""`boltbeam fetch DEST`: command wiring for workflow/fetch.py (list a remote's runs, bring them home, verified)."""
from __future__ import annotations

import json
import pathlib
import sys

from boltbeam.workflow import fetch as fx


def _size(n:int) -> str:
  for unit in ("B", "KB", "MB", "GB"):
    if n < 1024 or unit == "GB":
      return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
    n /= 1024
  return ""


STATE_WORDS = {"new": "not here", "here": "already here", "conflict": "a different run here under this id"}


def words(r:dict) -> list[str]:
  lines = [f"remote runs root on {r['dest']}: {r['remote_root']}" +
           (f" (tried {', '.join(r['tried'][:-1])} first)" if len(r["tried"]) > 1 else ""),
           f"local saved runs: {r['saved']}"]
  if not r["runs"]:
    lines.append("no runs there")
  for x in r["runs"]:
    left = f", {_size(x['left_behind_bytes'])} of raw traces left behind" if x["left_behind_bytes"] else ""
    lines.append(f"  {x['id']}  {x['model_id'] or '?'} on {x['target_id'] or '?'} ({x['gpu'] or 'GPU not recorded'})  "
                 f"{x['date'][:10]}  {_size(x['bytes'])}{left}  {STATE_WORDS[x['state']]}")
  if not r["fetched"] and r["runs"]:
    lines.append("fetch one with `boltbeam fetch DEST RUN_ID`, or all with --all")
  return lines + fetched_words(r)


def fetched_words(r:dict) -> list[str]:
  """What each fetch did: where it landed and how it was checked, or why it was skipped or refused (the screen shows
  these, --json's `lines`)."""
  lines = []
  for f in r["fetched"]:
    if f["status"] == "fetched":
      lines.append(f"fetched {f['id']}: {f['files']} files, {_size(f['bytes'])}, checked by sha256 ({f['transport']}) -> {f['dir']}")
      c = f["chip"]
      if c["used"] == "fetched" and c.get("path"):
        lines.append(f"  chip {c['target_id']}: its profile from {c['remote_path']}, checked by sha256 -> {c['path']}")
      if c.get("why"):
        lines.append(f"  chip {c['target_id']}: {c['why']}")
      if not f["results"]:
        lines.append("  results.json not computed here (fetched.json says why)")
      if f["warning"]:
        lines.append(f"  warning: {f['warning']}")
    elif f["status"] == "skipped":
      lines.append(f"skipped {f['id']}: {f['why']} ({f['dir']})")
    else:
      lines.append(f"refused {f['id']}: {f['why']}")
  if any(f["status"] == "fetched" for f in r["fetched"]):
    lines.append("results are home: pause the remote machine now if nothing else needs it")
  return lines


def cmd_fetch(args, out=sys.stdout, ssh:fx.Ssh | None = None) -> int:
  try:
    if args.all and args.run_ids:
      raise fx.Refused("give run ids or --all, not both")
    here, saved = fx.local_roots(args.into)
    if getattr(args, "saved", None):  # the screen's own saved-runs folder (its --saved), wherever that is
      saved = pathlib.Path(args.saved).expanduser().resolve()
    r = fx.fetch(ssh or fx.Ssh(args.dest, args.port), args.run_ids, saved, root=args.root, every=args.all,
                 with_traces=args.with_traces, here=here)
  except fx.Refused as exc:
    print(json.dumps({"schema": fx.SCHEMA_FETCHED, "kind": "error", "error": str(exc)}) if args.json
          else f"fetch refused: {exc}", file=out)
    return 1
  print(json.dumps({**r, "lines": fetched_words(r)}, indent=2) if args.json else "\n".join(words(r)), file=out)
  return 1 if any(f["status"] == "refused" for f in r["fetched"]) else 0


def register(sub) -> None:
  p = sub.add_parser("fetch", help="bring measurement runs home from a remote GPU machine over SSH; without run ids it "
                                  "lists the remote's runs, with ids or --all it copies them into saved runs, checked by sha256")
  p.add_argument("dest", metavar="DEST", help="the remote, as ssh takes it: user@host or a Host alias")
  p.add_argument("run_ids", metavar="RUN_ID", nargs="*", help="the runs to fetch (none: list them)")
  p.add_argument("-p", "--port", type=int, help="the remote's SSH port (a pod's exposed TCP port)")
  p.add_argument("--root", help="the remote runs root (default: the first of %s)" % ", ".join(fx.REMOTE_ROOTS))
  p.add_argument("--all", action="store_true", help="fetch every run the remote holds")
  p.add_argument("--with-traces", action="store_true", help="also copy raw traces (%s)" % ", ".join(fx.RAW_TRACES))
  p.add_argument("--into", help="the local runs folder (default: BOLTBEAM_RUNS, else the checkout's runs/); runs land in its saved/")
  p.add_argument("--saved", help="the saved-runs folder runs land in (default: <runs folder>/saved, or BOLTBEAM_SAVED)")
  p.add_argument("--json", action="store_true", help="print the answer as JSON, with `lines`: what each fetch did, in words")
  p.set_defaults(fn=cmd_fetch)
