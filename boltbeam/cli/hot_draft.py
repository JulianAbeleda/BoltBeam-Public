"""`boltbeam derive hot-draft DRAFT`: a draft copy holding only its hottest experts, sized to a measured memory budget
(workflow/hot_draft.py). --json prints the record, or {"kind": "error", "error": ...}."""
from __future__ import annotations

import json
import sys


def cmd_hot_draft(args, out=sys.stdout) -> int:
  from boltbeam.core.refused import Refused
  from boltbeam.workflow import hot_draft
  try:
    rec = hot_draft.derive(args.draft, engine=args.engine, out=args.out, routes=args.routes, main=args.main, k=args.k, budget_mib=args.budget_mib,
                           measured=args.measured, arm=args.arm, margin_mib=args.margin_mib, build=args.build, acceptance=not args.no_acceptance)
  except (Refused, OSError, ValueError, RuntimeError) as exc:  # RuntimeError: mtp.MtpRefused (no server, no head)
    print(json.dumps({"schema": hot_draft.SCHEMA, "kind": "error", "error": str(exc)}) if args.json else f"hot-draft refused: {exc}", file=out)
    return 1
  print(json.dumps({**rec, "lines": hot_draft.words(rec)}, indent=1) if args.json else "\n".join(hot_draft.words(rec)), file=out)
  return 0


def add(what) -> None:
  q = what.add_parser("hot-draft", help="a draft (MTP head) copy holding only its hottest experts, sized to a measured GPU memory budget")
  q.add_argument("draft", help="the draft's GGUF (llama-server -md)")
  q.add_argument("--engine", default="llama.cpp", help="the engine whose route count and file format it reads (data/hot_draft.json)")
  q.add_argument("--out", help="the copy (default: <draft>-hot.gguf beside it); its record is <out>.json")
  q.add_argument("--routes", help="the draft's route count the engine wrote (llama.cpp: LLAMA_DRAFT_ROUTE_COUNT=path); else captured with --main --k")
  q.add_argument("--main", help="the main model the draft drafts for: capture the route count and measure acceptance (full and trimmed) on real text")
  q.add_argument("--k", type=int, help="the draft count the server runs (--spec-draft-n-max), for the capture and acceptance")
  q.add_argument("--build", help="the engine's server binary (default: found as other commands find it); the capture needs the route count switch in it")
  q.add_argument("--budget-mib", type=float, help="measured MiB of GPU memory the draft's experts may take")
  q.add_argument("--measured", help="a session-ab record: the budget is an arm's lowest free memory less its run-to-run spread, plus its own draft's experts")
  q.add_argument("--arm", help="with --measured: the arm whose setup the copy will run in")
  q.add_argument("--margin-mib", type=float, help="with --measured: the margin kept free instead of the measured spread (an arm run once has none)")
  q.add_argument("--no-acceptance", action="store_true", help="skip measuring the trimmed draft's acceptance")
  q.add_argument("--json", action="store_true")
  q.set_defaults(fn=cmd_hot_draft)
