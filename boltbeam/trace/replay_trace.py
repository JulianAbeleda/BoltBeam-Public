"""A replay's input: a session as counts, one row per turn (boltbeam.replay_trace.v1). Never chat text.

    {"schema": "boltbeam.replay_trace.v1",
     "source": {"kind": "shape" | "gameterm", "name": "...", "words": "...", "sha256": "<of the source>"},
     "system_tokens": 3000,
     "turns": [{"i": 1, "at_ms": 0, "new_tokens": 41, "tool_tokens": 0, "reply_tokens": 112, "compaction": false}, ...]}

A turn adds new_tokens from the user and tool_tokens of tool output to the conversation, then the model answers
reply_tokens. A compaction turn starts the conversation again from the system prompt, with its new tokens as the
summary. Readers adapt a source into this one schema (READERS):

  shape      a synthesised trace from a table of ranges (data/replay_shapes.json), drawn with the shape's seed
  gameterm   a GameTerm transcript and its save, counted by the engine's tokenizer on the user's own machine; the
             trace holds the counts and the source file's sha256, never its words or its path. The system prompt is
             in neither file, so its size is stated by the caller.

Privacy is enforced, not promised: `check` refuses any string longer than MAX_STRING characters outside a field named
`words` (a chat line would be one), so a trace that holds text cannot be written or read.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import statistics
from typing import Any, Callable

SCHEMA = "boltbeam.replay_trace.v1"
SHAPES = pathlib.Path(__file__).resolve().parents[1] / "data" / "replay_shapes.json"
KINDS = ("shape", "gameterm")
COUNTS = ("new_tokens", "tool_tokens", "reply_tokens")
MAX_STRING = 80  # longer strings are refused outside `words` fields: a trace holds counts, not chat text
FILE = "replay_trace.json"  # the trace as the run read it


class TraceRefused(ValueError):
  """The trace is not a counts-only replay_trace.v1: what is wrong and where."""


def _strings(value:Any, path:str = "") -> list[tuple[str, str]]:
  if isinstance(value, str):
    return [(path, value)]
  if isinstance(value, dict):
    return [s for k, v in value.items() if k != "words" for s in _strings(v, f"{path}.{k}")]
  if isinstance(value, list):
    return [s for i, v in enumerate(value) for s in _strings(v, f"{path}[{i}]")]
  return []


def check(trace:dict[str, Any]) -> dict[str, Any]:
  """The trace, if it is a counts-only replay_trace.v1; TraceRefused naming the field otherwise."""
  if not isinstance(trace, dict) or trace.get("schema") != SCHEMA:
    raise TraceRefused(f"not a {SCHEMA}")
  long = [p for p, s in _strings(trace) if len(s) > MAX_STRING]
  if long:
    raise TraceRefused(f"a replay trace holds counts, not text: {long[0]} is a string over {MAX_STRING} characters")
  src = trace.get("source") or {}
  if src.get("kind") not in KINDS:
    raise TraceRefused(f"source.kind is {src.get('kind')!r}; a trace comes from one of {', '.join(KINDS)}")
  if not isinstance(trace.get("system_tokens"), int) or trace["system_tokens"] < 0:
    raise TraceRefused("system_tokens is not a count")
  turns = trace.get("turns")
  if not isinstance(turns, list) or not turns:
    raise TraceRefused("the trace has no turns")
  for n, t in enumerate(turns, 1):
    if t.get("i") != n:
      raise TraceRefused(f"turn {n} is numbered {t.get('i')!r}: turns are numbered from 1 in order")
    bad = [k for k in (*COUNTS, "at_ms") if not isinstance(t.get(k), int) or t[k] < 0]
    if bad or t["reply_tokens"] < 1 or not isinstance(t.get("compaction"), bool):
      raise TraceRefused(f"turn {n}: {', '.join(bad) or 'reply_tokens or compaction'} is not a valid count or flag")
  return trace


def read(path:pathlib.Path | str) -> dict[str, Any]:
  try:
    return check(json.loads(pathlib.Path(path).read_text()))
  except (OSError, ValueError) as exc:
    raise TraceRefused(f"{path}: {exc}") from exc


def write(trace:dict[str, Any], path:pathlib.Path | str) -> pathlib.Path:
  out = pathlib.Path(path)
  out.write_text(json.dumps(check(trace), indent=1) + "\n")
  return out


def prefix_tokens(trace:dict[str, Any]) -> int:
  """The tokens the conversation holds after its last turn, by the trace's counts (a compaction starts it again)."""
  total = trace["system_tokens"]
  for t in trace["turns"]:
    if t["compaction"]:
      total = trace["system_tokens"]
    total += t["new_tokens"] + t["tool_tokens"] + t["reply_tokens"]
  return total


def label(trace:dict[str, Any]) -> str:
  """One line for a picker or a summary: the name, the turns and the prefix they reach."""
  p = prefix_tokens(trace)
  return f"{trace['source'].get('name') or trace['source']['kind']}, {len(trace['turns'])} turns to {p / 1000:.0f}K"


# --- readers -----------------------------------------------------------------------------------------------------

def shapes() -> dict[str, Any]:
  return json.loads(SHAPES.read_text())


def shape(name:str | None = None) -> dict[str, Any]:
  """A synthesised trace from data/replay_shapes.json: each count drawn uniformly from its range, the shape's seed."""
  table = shapes()
  name = name or table["default"]
  if name not in table["shapes"]:
    raise TraceRefused(f"no shape {name!r}; the shapes are {', '.join(table['shapes'])}")
  s = table["shapes"][name]
  rng = random.Random(s["seed"])
  turns, at = [], 0
  for i in range(1, s["turns"] + 1):
    row = {"i": i, "at_ms": at, **{k: rng.randint(*s[k]) for k in COUNTS}, "compaction": False}
    turns.append(row)
    at += rng.randint(*s["gap_ms"])
  digest = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()
  return check({"schema": SCHEMA, "source": {"kind": "shape", "name": name, "words": s["words"], "sha256": digest},
                "system_tokens": s["system_tokens"], "turns": turns})


def gameterm(transcript:pathlib.Path, save:pathlib.Path | None, *, system_tokens:int,
             count:Callable[[str], int]) -> dict[str, Any]:
  """A GameTerm session as counts. The transcript gives each turn's user text (turn_started) and its tools' output
  (tool_resolved); the save gives the replies it kept (a speaker turn after a user turn, matched to the transcript's
  turn by its text). A turn whose reply the save no longer holds takes the median of the matched replies, and the
  words say how many. `count` is the engine's tokenizer. Only counts leave this function."""
  events = []
  for line in transcript.read_text(errors="replace").splitlines():
    try:
      events.append(json.loads(line))
    except ValueError:
      continue
  events.sort(key=lambda e: e.get("sequence") or 0)
  replies: dict[str, list[int]] = {}
  if save is not None:
    kept = json.loads(save.read_text()).get("turns") or []
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
                 "reply": (replies.get(key) or [None]).pop(0) if replies.get(key) else None}
    elif kind == "tool_resolved" and rid in by:
      by[rid]["tool"] += count(str(detail.get("output") or ""))
  if not order:
    raise TraceRefused(f"{transcript.name}: no turn_started events")
  matched = [by[r]["reply"] for r in order if by[r]["reply"]]
  if not matched:
    raise TraceRefused("no reply in the save matches a transcript turn, so no reply length is known")
  fill = int(statistics.median(matched))
  t0 = by[order[0]]["at"]
  turns = [{"i": n, "at_ms": max(by[r]["at"] - t0, 0), "new_tokens": by[r]["new"], "tool_tokens": by[r]["tool"],
            "reply_tokens": by[r]["reply"] or fill, "compaction": False} for n, r in enumerate(order, 1)]
  digest = hashlib.sha256(transcript.read_bytes()).hexdigest()
  return check({"schema": SCHEMA, "source": {
    "kind": "gameterm", "name": f"gameterm-{len(turns)}", "sha256": digest,
    "words": (f"a GameTerm session of {len(turns)} turns, counted by the engine's tokenizer; replies from the save for "
              f"{len(matched)}, the median {fill} for the other {len(turns) - len(matched)}; a system prompt of "
              f"{system_tokens} tokens as stated (the logs do not hold it)")},
    "system_tokens": int(system_tokens), "turns": turns})
