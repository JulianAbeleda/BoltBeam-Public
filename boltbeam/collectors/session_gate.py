"""Stage 1 of the agreement gate through the server's own chat path: the `--hook` of `boltbeam session-ab`.

    boltbeam session-ab ... --arm BASE=... --arm FLOOR='<BASE's flags> -b 5 -ub 5' --arm X=... \
      --hook 'boltbeam session-gate --reference BASE --floor FLOOR'

Speed up's stage-1 rule (quantization/speed_up: agreement, bar) on what an arm actually changes in a long chat: the
session's own turns (session-ab's --session and --turns, recorded replies) go through /v1/chat/completions with the real
template and the server's prompt cache, so the user-message prefill, its checkpoints, the KV handling and the expert
placement at the session's depth are all on the path. Per turn, the next token is scored teacher-forced at positions
spread over the recorded reply: the turn's messages, then the reply's first k tokens as an assistant prefill (the server
continues the final message), one greedy token (penalties and DRY sent as no-ops). Every arm sends the same requests in
the same order, so positions pair by request. A document = a turn.
- BASE (the first --arm) is the reference: stored on its block-1 run.
- FLOOR is BASE with llama.cpp's declared no-op change appended (speed_up.perturbation: another batch, raised past the
  draft's tail): checked against BASE's command; d_floor = its disagreement with BASE.
- Every other arm: d = its disagreement with BASE; pass iff speed_up.bar passes (d - d_floor <= t SE, every turn holds).
Positions: the threshold's min_positions spread over the turns (shares: a short reply gives all it has, the others the
rest; a cut that does not read back moves to the nearest free one). The span needs speed_up.min_documents turns, each
with a position: fewer positions than asked are said, not refused (the SE and the t at k - 1 df carry them). Files under SESSION_AB_OUT/gate: tops-<arm>-<block>.json, verdict-<arm>-<block>.json (a retried arm
overwrites its own; the session-ab record says which attempt was valid).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
from typing import Any, Callable

DIR = "gate"
NEUTRAL = {"repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0, "xtc_probability": 0.0,
           "samplers": ["temperature"]}  # llama.cpp's sampler chain cut to temperature 0, whatever the server's argv sets: greedy = top-1
REQUEST = {"max_tokens": 1, "temperature": 0.0, "cache_prompt": True, "logprobs": True, "top_logprobs": 1,
           "chat_template_kwargs": {"enable_thinking": False}, **NEUTRAL}  # as session-ab's turns (replay.REQUEST), one token


def cuts(n:int, k:int) -> list[int]:
  """k reply prefixes (token counts) spread evenly over a reply of n tokens, 0 (the bare turn) first; all n if k >= n."""
  return list(range(n)) if k >= n else sorted({round(j * (n - 1) / max(1, k - 1)) for j in range(k)})


def shares(lengths:list[int], total:int) -> list[int]:
  """Positions per reply: total spread evenly, a reply shorter than its share giving all it has and the rest going to the others."""
  got, left, open_ = [0] * len(lengths), total, sorted(range(len(lengths)), key=lambda i: lengths[i])
  while open_ and left > 0:
    i = open_.pop(0)
    got[i] = min(lengths[i], math.ceil(left / (len(open_) + 1)))
    left -= got[i]
  return got


def top(response:dict[str, Any]) -> str:
  """The greedy token of a one-token chat reply: its id when the server gives one, else its text."""
  lp = (((response.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or [{}])[0]
  got = lp.get("id", lp.get("token"))
  if got is None:
    raise ValueError("the server returned no logprobs for the scored token")
  return str(got)


def tops(url:str, session:dict[str, Any], span:list[int], total:int, post:Callable | None = None,
         tokenize:Callable | None = None, detokenize:Callable | None = None) -> dict[str, Any]:
  """Per turn of span, the greedy token at its share() of total positions of the recorded reply; the requests' own sha256, and the
  server's prompt_n, cache_n and context offset per request (prompt tokens - the bare turn's - k: 0 = the scored position
  sees exactly the generation prompt + k reply tokens; a template that renders a continued message otherwise shows here).
  Recorded and said, never judged: every arm sends the same requests, so the pairing holds either way."""
  from boltbeam.adapters.llama_server_control import _json_post
  from boltbeam.collectors import replay
  send, tok, detok = post or _json_post, tokenize or replay.llama_tokenize, detokenize or replay.llama_detokenize
  msgs = [{"role": "system", "content": session.get("system", "")}]
  for t in session["turns"][:span[0] - 1]:
    msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["assistant"]}]
  out, seen, comp = [], hashlib.sha256(), []
  turns = session["turns"][span[0] - 1:span[1]]
  replies = [tok(url, t["assistant"]) for t in turns]
  for t, reply, want in zip(turns, replies, shares([len(r) for r in replies], total)):
    msgs.append({"role": "user", "content": t["user"]})
    got, base, done = [], None, set()
    for c in cuts(len(reply), want):
      # the nearest cut at or after c, else before it, that is not used and the tokenizer reads back as the same tokens (no
      # split character, no merge across it): a cut that collides or runs off the reply's end moves, so the turn still
      # gives its share and the span its total (a 116-turn session fell one short of 2,000 when such a cut was dropped)
      k = next((j for j in [*range(c, len(reply)), *range(c - 1, -1, -1)]
                if j not in done and (not j or tok(url, detok(url, reply[:j])) == reply[:j])), None)
      if k is None:
        continue
      if k:
        text = detok(url, reply[:k])
      done.add(k)
      body = {"messages": msgs + ([{"role": "assistant", "content": text}] if k else []), **REQUEST}
      seen.update(json.dumps(body, sort_keys=True).encode())
      r = send(url.rstrip("/") + "/v1/chat/completions", body, 3600.0)
      got.append(top(r))
      pt = (r.get("usage") or {}).get("prompt_tokens")
      base = pt if not k else base  # the bare turn's prompt: a prefill of k reply tokens should be exactly k longer
      comp.append([(r.get("timings") or {}).get("prompt_n"), (r.get("timings") or {}).get("cache_n"), None if pt is None or base is None or not k else pt - base - k])
    out.append(got)
    msgs.append({"role": "assistant", "content": t["assistant"]})
  return {"tops": out, "requests_sha256": seen.hexdigest(), "computed": comp}


def per_doc(a:list[list[str]], b:list[list[str]]) -> list[dict[str, int]]:
  if [len(x) for x in a] != [len(x) for x in b]:
    raise ValueError("the two arms scored different positions, so they cannot be paired")
  return [{"n": len(x), "hits": sum(p == q for p, q in zip(x, y))} for x, y in zip(a, b)]


def floor_flags(command:list[str]) -> list[str]:
  """The flags the floor arm appends to the reference's command: llama.cpp's declared no-op change for that setup."""
  from boltbeam.quantization import speed_up
  return speed_up.perturbation("llama.cpp", command)["args"]["dst_flags"]


def mine_flags(command:list[str]) -> list[str]:
  """The user's command without the --host/--port pairs session-ab gives each server (llama_server_control.own_argv)."""
  from boltbeam.adapters.llama_server_control import own_argv
  return own_argv(command, 0)[:-4]


def run(*, url:str, arm:str, block:str, out:str, reference:str, floor:str, session:str, span:list[int], command:list[str],
        post:Callable | None = None, tokenize:Callable | None = None, detokenize:Callable | None = None,
        say:Callable[[str], None] = print) -> dict[str, Any]:
  """One hook call: the arm's pass; stored for the reference and the floor, else judged against both."""
  from boltbeam.collectors.session_ab import read_session
  from boltbeam.quantization import speed_up
  sess, _ = read_session(session, "recorded")
  th, n_turns = speed_up.table()["threshold"], span[1] - span[0] + 1
  d = pathlib.Path(out) / DIR
  d.mkdir(parents=True, exist_ok=True)
  mine = {"arm": arm, "block": block, "span": span, "command": command,
          **tops(url, sess, span, th["min_positions"], post, tokenize, detokenize)}
  (d / f"tops-{arm}-{block}.json").write_text(json.dumps(mine))
  if arm == reference and block == "1":
    n, k_min, docs = sum(map(len, mine["tops"])), speed_up.min_documents(), sum(1 for x in mine["tops"] if x)
    if docs < k_min:  # a turn with an empty reply scores no position: it is not a document, the others still are
      raise ValueError(f"the span gives {n} positions over {docs} of its {n_turns} turns; the gate needs at least {k_min} turns (documents) with a "
                       f"scored position each: its SE is a bootstrap over documents, and k give C(2k-1, k) distinct resamples, which "
                       f"must resolve the one-sided tail 1 - {th['confidence']}")
    if n < th["min_positions"]:  # fewer than asked only where the replies hold no more: the t at k - 1 df and the SE carry it
      say(f"session-gate: {n} positions, under the {th['min_positions']} asked: the span's replies hold no more")
    off = [c[2] for c in mine["computed"] if c[2] is not None]
    say(f"session-gate: context offset 0 on {sum(x == 0 for x in off)}/{len(off)} prefilled positions")
    say(f"session-gate: {arm} is the reference: {sum(map(len, mine['tops']))} positions over {n_turns} turns stored")
    return {"verdict": "reference"}
  ref_file = d / f"tops-{reference}-1.json"
  if not ref_file.exists():
    raise ValueError(f"no reference pass in {d}: the reference arm {reference!r} must be session-ab's first --arm")
  ref = json.loads(ref_file.read_text())
  if ref["requests_sha256"] != mine["requests_sha256"]:
    raise ValueError(f"this arm's requests differ from {reference!r}'s (session, span or tokenizer), so the positions do not pair")
  if arm == floor and block == "1":
    want = floor_flags(ref["command"])
    if mine_flags(command) != mine_flags(ref["command"]) + want:
      raise ValueError(f"the floor arm {floor!r} must be {reference!r}'s command with only the declared no-op change {' '.join(want)} appended (speed_up.perturbation)")
    fl = {"per_doc": per_doc(ref["tops"], mine["tops"]), "says": " ".join(want)}
    fl["n"] = sum(x["n"] for x in fl["per_doc"])
    (d / "floor.json").write_text(json.dumps(fl))
    say(f"session-gate: {arm} is the floor ({fl['says']}): agreement with {reference} {sum(x['hits'] for x in fl['per_doc']) / fl['n']:.4f} over {fl['n']} positions")
    return {"verdict": "floor", "floor": fl}
  if not (d / "floor.json").exists():
    raise ValueError(f"no floor in {d}: the floor arm {floor!r} must be session-ab's second --arm")
  fl = json.loads((d / "floor.json").read_text())
  pd = per_doc(mine["tops"], ref["tops"])
  keep = [i for i, x in enumerate(pd) if x["n"]]  # the documents: turns with a scored position (an empty reply has none)
  figures = {"per_doc": [pd[i] for i in keep], "ctx": None, "method": "llama-server chat path, teacher-forced greedy top-1 (session-gate)"}
  figures["n"] = sum(x["n"] for x in figures["per_doc"])
  figures = speed_up.agreement(figures, [str(span[0] + i) for i in keep])
  g = speed_up.bar(figures, {**fl, "per_doc": [fl["per_doc"][i] for i in keep if i < len(fl["per_doc"])]})
  rec = {"arm": arm, "block": block, "reference": reference, "floor": floor, "accuracy": figures, "gate": g}
  (d / f"verdict-{arm}-{block}.json").write_text(json.dumps(rec, indent=1))
  say(f"session-gate: {arm} block {block} vs {reference}: {g['verdict']}; agreement {1 - g['d_copy']:.4f}, floor {1 - g['d_floor']:.4f}, "
      f"excess {g['excess']:+.4f} (allowed {g['allowed']:.4f}), worst turn {g['worst']['excess']:+.4f} (allowed {g['worst']['allowed']:.4f})")
  return rec


def from_env(reference:str, floor:str, environ:dict[str, str] | None = None) -> dict[str, Any]:
  e = os.environ if environ is None else environ
  keys = ("SESSION_AB_URL", "SESSION_AB_ARM", "SESSION_AB_BLOCK", "SESSION_AB_OUT", "SESSION_AB_SESSION", "SESSION_AB_TURNS", "SESSION_AB_COMMAND")
  if missing := [k for k in keys if not e.get(k)]:
    raise ValueError(f"session-gate runs as session-ab's --hook: {', '.join(missing)} not set")
  return run(url=e["SESSION_AB_URL"], arm=e["SESSION_AB_ARM"], block=e["SESSION_AB_BLOCK"], out=e["SESSION_AB_OUT"], reference=reference,
             floor=floor, session=e["SESSION_AB_SESSION"], span=json.loads(e["SESSION_AB_TURNS"]), command=json.loads(e["SESSION_AB_COMMAND"]))
