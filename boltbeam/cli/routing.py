"""`boltbeam routing MODEL --engine E`: which experts the model's router picks per layer on real text, saved as a routing
profile the prompt limit, the offload plan and tune read (workflow/routing.py)."""
from __future__ import annotations

import json
import pathlib


def cmd_routing(args) -> int:
  from boltbeam.core.refused import Refused
  from boltbeam.workflow import routing
  try:
    res = routing.run(args.model, args.engine, tokens=args.tokens, source=args.source, out=pathlib.Path(args.out).expanduser() if args.out else None,
                      build=args.build, port=args.port, corpus=args.corpus)
  except Refused as exc:
    print(f"routing: {exc}")
    return 2
  other = json.loads(pathlib.Path(args.compare).expanduser().read_text()) if args.compare else None
  print(json.dumps(res, indent=1) if args.json else routing.text(res, other))
  return 0


def register(sub) -> None:
  from boltbeam.workflow import routing
  p = sub.add_parser("routing", help="a routing profile: the experts an MoE model's router picks per layer on real text, which the "
                     "prompt limit, offload plan and tune then use instead of uniform routing")
  p.add_argument("model", help="the model file (a split model's first file)")
  p.add_argument("--engine", required=True, choices=routing.engines(), help="whose routing record to read or capture")
  p.add_argument("--tokens", type=int, default=routing.TOKENS, help="tokens of the corpus (data/gate_corpus.json) to read")
  p.add_argument("--from", dest="source", help="the engine's own routing record to read instead of capturing one")
  p.add_argument("--corpus", help="with --from: the text the run that wrote it read (a name, recorded in the profile)")
  p.add_argument("--out", help="where to save it (default ~/.boltbeam/routing/<model id>.json, where other commands look)")
  p.add_argument("--build", help="the engine's server binary for a capture")
  p.add_argument("--port", type=int, default=0, help="the capture server's port (0: a free one)")
  p.add_argument("--compare", help="another saved profile: print the rank correlation of the two hotness orders")
  p.add_argument("--json", action="store_true")
  p.set_defaults(fn=cmd_routing)
