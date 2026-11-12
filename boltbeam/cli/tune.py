"""`boltbeam tune MODEL --engine E`: this machine's values for an engine's knobs (workflow/tune.py, data/engine_knobs.json),
each with its source and reason. Facts only by default (instant, no model started); --measure runs the table's short
real-text prefill sweeps. The engines are the table's; each is read and run by its adapter."""
from __future__ import annotations

import json
import pathlib


def cmd_tune(args) -> int:
  from boltbeam.core.refused import Refused
  from boltbeam.workflow import tune
  try:
    budget = tune.time_budget(args.time_budget, args.prompt) if args.time_budget else None
    reply = tune.reply_budget(args.replies) if args.replies else None
    res = tune.tune(args.model, args.engine, args.build, prompt=args.prompt, context=args.context, measure=args.measure,
                    knobs_only=args.knob, port=args.port, out=pathlib.Path(args.out).expanduser() if args.out else None,
                    extra=args.flags.split() if args.flags else None, experts_in_ram=args.experts_in_ram,
                    workload=args.workload, reply=reply, **({"time_budget": budget} if budget else {}))
  except Refused as exc:
    print(f"tune: {exc}")
    return 2
  print(json.dumps(res, indent=1, default=str) if args.json else tune.text(res))
  return 0


def register(sub) -> None:
  from boltbeam.workflow import tune
  p = sub.add_parser("tune", help="this machine's values for an engine's knobs (local patch knobs, chunk sizes, memory shares, "
                     "offload), each with its source; facts only unless --measure")
  p.add_argument("model", help="the model as the engine reads it (a file, a split model's first file, or a folder)")
  p.add_argument("--engine", required=True, choices=tune.engines(), help="the engine whose knobs to set")
  p.add_argument("--build", help="the engine build: its server binary or its python (default: found as "
                 "doctor and screen find it)")
  p.add_argument("--experts-in-ram", help="all, or the first N layers' experts kept in host RAM (default: the offload plan's)")
  p.add_argument("--prompt", type=int, default=tune.PROMPT, help="the prompt length the setup is stated and swept at")
  p.add_argument("--context", type=int, help=f"the context the setup must hold (default: the prompt plus {tune.REPLY_TOKENS} tokens for the reply)")
  p.add_argument("--replies", help="measured replies (a JSON line per reply: tokens, finish): the room for the reply is the length that holds 95%% "
                 "of them, instead of the unmeasured default (not with --context)")
  p.add_argument("--workload", choices=tune.WORKLOADS, default="chat",
                 help="what the server will serve: chat (follow-ups reuse earlier turns; the default) or one-shot (each prompt once)")
  p.add_argument("--time-budget", help="a measured time budget (boltbeam.time_budget.v1 JSON) of this setup's prompt: its kernel time "
                 "replaces the prompt limit's in the slot count")
  p.add_argument("--measure", action="store_true", help="run the measured rows' prefill sweeps (starts the engine once per point)")
  p.add_argument("--knob", action="append", help="with --measure: only this knob (repeatable)")
  p.add_argument("--port", type=int, default=0, help="with --measure: the server's port, for an engine that opens one (0: a free one)")
  p.add_argument("--flags", help="extra engine flags for the command and, where the engine's driver takes them, every measured point")
  p.add_argument("--out", help="with --measure: the folder for logs and the record (default ~/.boltbeam/tune/<engine>-<time>)")
  p.add_argument("--json", action="store_true", help="the whole record as JSON")
  p.set_defaults(fn=cmd_tune)
