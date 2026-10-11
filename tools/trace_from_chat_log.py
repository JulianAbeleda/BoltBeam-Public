#!/usr/bin/env python3
"""Convert a chat log into a replay trace (boltbeam.replay_trace.v1): token counts per turn, never the words.

    python3 tools/trace_from_chat_log.py --log LOG.jsonl [--saved CONVERSATION.json] --system-tokens N \
        --model MODEL.gguf --out trace.json

The chat log is JSON lines, one event each, with `kind`, `request_id`, `sequence`, `at_ms` and `detail`:
`turn_started` holds the user's text (detail.text) and `tool_resolved` a tool's output (detail.output) for that
request. The saved conversation, when given, is a JSON object whose `turns` list holds {author, text}: a `speaker`
turn after a `user` turn is that turn's reply, matched to the log's turn by its text. A turn whose reply is not kept
takes the median of the matched replies, and the trace's words say how many. The system prompt is in neither file,
so its size is given.

Tokens are counted by the engine's own tokenizer (a small llama.cpp server for --model, stopped afterwards). Only
counts and the log file's sha256 leave this script; the trace is checked by boltbeam.trace.replay_trace before it
is written. BoltBeam reads the trace file as it reads any other; nothing in boltbeam/ imports this script.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import statistics
import sys
import tempfile
from typing import Any, Callable

from boltbeam.trace import replay_trace as rt

KIND = "chat_log"


def from_chat_log(log:pathlib.Path, saved:pathlib.Path | None, *, system_tokens:int,
                  count:Callable[[str], int]) -> dict[str, Any]:
  """A chat log as a counts-only replay trace (see the module's words for the formats)."""
  events = []
  for line in log.read_text(errors="replace").splitlines():
    try:
      events.append(json.loads(line))
    except ValueError:
      continue
  events.sort(key=lambda e: e.get("sequence") or 0)
  replies: dict[str, list[int]] = {}
  if saved is not None:
    kept = json.loads(saved.read_text()).get("turns") or []
    for a, b in zip(kept, kept[1:]):
      if a.get("author") == "user" and b.get("author") == "speaker":
        replies.setdefault(hashlib.sha256(str(a.get("text")).encode()).hexdigest(), []).append(count(str(b.get("text"))))
  order: list[str] = []
  by: dict[str, dict[str, Any]] = {}
  for e in events:
    rid, kind, detail = e.get("request_id"), e.get("kind"), e.get("detail") or {}
    if kind == "turn_started" and rid not in by:
      order.append(rid)
      key = hashlib.sha256(str(detail.get("text")).encode()).hexdigest()
      by[rid] = {"at": int(e.get("at_ms") or 0), "new": count(str(detail.get("text") or "")), "tool": 0,
                 "reply": replies[key].pop(0) if replies.get(key) else None}
    elif kind == "tool_resolved" and rid in by:
      by[rid]["tool"] += count(str(detail.get("output") or ""))
  if not order:
    raise rt.TraceRefused(f"{log.name}: no turn_started events")
  matched = [by[r]["reply"] for r in order if by[r]["reply"]]
  if not matched:
    raise rt.TraceRefused("no reply in the saved conversation matches a turn of the log, so no reply length is known")
  fill = int(statistics.median(matched))
  t0 = by[order[0]]["at"]
  turns = [{"i": n, "at_ms": max(by[r]["at"] - t0, 0), "new_tokens": by[r]["new"], "tool_tokens": by[r]["tool"],
            "reply_tokens": by[r]["reply"] or fill, "compaction": False} for n, r in enumerate(order, 1)]
  return rt.check({"schema": rt.SCHEMA, "source": {
    "kind": KIND, "name": f"chat-log-{len(turns)}", "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "words": (f"a chat log of {len(turns)} turns, counted by the engine's tokenizer; replies from the saved conversation "
              f"for {len(matched)}, the median {fill} for the other {len(turns) - len(matched)}; a system prompt of "
              f"{system_tokens} tokens as stated")},
    "system_tokens": int(system_tokens), "turns": turns})


@contextlib.contextmanager
def tokenizer(model:str):
  """The engine's tokenizer for a model: a small llama.cpp server beside the found llama-bench, stopped after the
  block. Yields a count(text) -> tokens function."""
  from boltbeam.adapters.llama_server_control import serve
  from boltbeam.collectors.replay import llama_tokenize
  from boltbeam.collectors.session import _server
  with tempfile.TemporaryDirectory() as tmp:
    with serve(executable=_server(pathlib.Path(tmp)), model=model, context_size=512,
               log_path=pathlib.Path(tmp) / "tokenizer.log", extra=["--parallel", "1"]) as (url, _):
      yield lambda text: len(llama_tokenize(url, text)) if text else 0


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--log", required=True, help="the chat log, JSON lines")
  p.add_argument("--saved", default=None, help="the saved conversation that holds the replies")
  p.add_argument("--system-tokens", type=int, required=True, help="the system prompt's tokens (the log does not hold it)")
  p.add_argument("--model", required=True, help="the model whose tokenizer counts the log")
  p.add_argument("--out", required=True)
  args = p.parse_args(argv)
  with tokenizer(args.model) as count:
    trace = from_chat_log(pathlib.Path(args.log).expanduser(), pathlib.Path(args.saved).expanduser() if args.saved else None,
                          system_tokens=args.system_tokens, count=count)
  rt.write(trace, pathlib.Path(args.out).expanduser())
  print(json.dumps({"out": args.out, "label": rt.label(trace), "source": trace["source"]}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
