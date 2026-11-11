"""`boltbeam derive lighter-head CHECKPOINT`: a copy of a model with a lighter output layer, kept only if it passes
the accuracy check (quantization/speed_up.py). --json prints the record, or {"kind": "error", "error": ...}."""
from __future__ import annotations

import json
import sys

from boltbeam.core.refused import Refused
from boltbeam.quantization import speed_up


def cmd_lighter_head(args, out=sys.stdout) -> int:
  try:
    named = args.engine in {r["engine"] for r in speed_up.table()["engines"]}  # an engine row, else the engine's own path
    if args.verify:
      rec = speed_up.verify(args.checkpoint, args.engine if named else None, None if named else args.engine)
      print(json.dumps(rec, indent=2) if args.json else f"verified: {rec['name']} matches its record: files, gate text, {rec['engine_version']}, " + ("speed measured on this GPU" if rec["speed"] else "SPEED WAS NOT MEASURED"), file=out)
      return 0
    progress = getattr(args, "progress", False)
    if progress:  # the pipeline's own lines on stderr, for a screen's bar; stdout stays the one JSON object
      for line in speed_up.progress_lines(args.checkpoint, not args.no_speed):
        print(line, file=sys.stderr, flush=True)
    rec = speed_up.derive(args.checkpoint, args.engine if named else None, args.out, None if named else args.engine, args.keep_failed, not args.no_speed,
                          (lambda stage, event: print(f"stage {stage}: {event}", file=sys.stderr, flush=True)) if progress else None)
  except Refused as exc:
    print(json.dumps({"schema": speed_up.SCHEMA, "kind": "error", "error": str(exc)}) if args.json else f"speed up refused: {exc}", file=out)
    return 1
  if getattr(args, "progress", False):
    print("pipeline done: speed up", file=sys.stderr, flush=True)
  print(json.dumps({**rec, "lines": speed_up.words(rec)}, indent=2) if args.json else "\n".join(speed_up.words(rec)), file=out)
  return 0 if rec["kept"] and rec["passed"] and all(e["passed"] for e in rec["speed"]) else 2


def register(sub) -> None:
  p = sub.add_parser("derive", help="make a derived copy of a model; the original stays (lighter-head: a lighter output layer, kept only if it still answers the same)")
  what = p.add_subparsers(dest="what", required=True, metavar="<what>")
  q = what.add_parser("lighter-head", help="a faster copy: the output layer quantized lighter, stock engines only")
  q.add_argument("checkpoint", help="a GGUF file or a Hugging Face checkpoint folder")
  q.add_argument("--engine", default=None, help="an engine row of data/speed_up.json (vllm, llama.cpp), or the engine's python (vLLM) or llama-bench (llama.cpp); default: the row for the format, the engine found as other commands find it")
  q.add_argument("--out", default=None, help="where the copy goes (default: <checkpoint>-boltbeam beside it)")
  q.add_argument("--keep-failed", action="store_true", help="keep a copy that fails the accuracy check")
  q.add_argument("--no-speed", action="store_true", help="skip the decode-speed measurement (original vs copy, short and long context)")
  q.add_argument("--verify", action="store_true", help="CHECKPOINT is a copy: check its files, gate text, engine version and GPU against its speed_up.json; refuses on a mismatch")
  q.add_argument("--progress", action="store_true", help="print the stages to stderr as they start and finish (a screen draws its bar from them)")
  q.add_argument("--json", action="store_true")
  q.set_defaults(fn=cmd_lighter_head)
