"""What every engine's MTP driver shares (collectors/mtp.py runs them); the standard library only, imported by path.

The long depth's prompts (long_prompts): each document of the shipped real-text corpus (data/mtp_long_corpus.json) cut
to the long depth by the model's own tokenizer at a paragraph break (cut), then its task: what the model reads is text
no model wrote, the same at every k. Token diversity (diversity): the least diverse stretch of a token sequence, which
collectors/mtp.guard holds every context and reply to, against the corpus's own (reference). The quiet-CPU gate (quiet):
before each timed request no build or test runner may be running (BUSY, the hand-run parity gate of 2026-11-12: a build
or a test suite beside a decode slows it); the load average is recorded beside it, not gated, the engine's own threads
being in it. delta: an engine's cumulative spec-decode counters, one request's worth."""
from __future__ import annotations

import collections, hashlib, json, operator, re, time
from pathlib import Path

BUSY = ("pytest", "make", "go", "nvcc", "cicc", "cc1plus")


def sha(ids:list[int]) -> str:
  return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def cut(text:str, n:int, tokenize, detokenize) -> str:
  """The document's start that its first n tokens hold, ended at the last paragraph break (a blank line; else a line
  break) in it; SystemExit when the document is not longer than n tokens: a long depth cut short is another depth."""
  ids = tokenize(text)
  if len(ids) <= n:
    raise SystemExit(f"a long-depth document is {len(ids):,} tokens on this tokenizer, not more than the {n:,} asked: ask a shorter long depth")
  head = detokenize(ids[:n])
  at = max(head.rfind("\n\n"), 0) or head.rfind("\n")
  if at <= 0:
    raise SystemExit(f"no paragraph break in the first {n:,} tokens of a long-depth document")
  if not text.startswith(head[:at]):
    raise SystemExit("this tokenizer's decode does not give back the long-depth document's text, so it cannot be cut at a paragraph break")
  return text[:at]


def long_prompts(corpus:str, n:int, tokenize, detokenize) -> tuple[dict[str, str], dict[str, list[int]]]:
  """{document id: the user message} (its start cut to n tokens, a blank line, its task; none when n is 0, short only),
  and {document id: the whole stored document's ids}, for the reference."""
  docs = json.loads(Path(corpus).read_text())["documents"]
  return ({d["id"]: cut(d["text"], n, tokenize, detokenize) + "\n\n" + d["task"] for d in docs} if n else {},
          {d["id"]: tokenize(d["text"]) for d in docs})


def diversity(ids:list[int], window:int) -> dict | None:
  """How much of ids repeats itself, by two measures. loop: the longest stretch equal to itself `period` places back
  (each token the same as the one period before it) that is at least one period long, so it is a whole period
  repeated, over every period up to the window (a loop's onset shows as soon as it repeats once; a period past the
  window or a loop shorter than its period is not seen). distinct: the lowest share of distinct tokens in any
  window-token stretch (low variety with no period). None when ids are shorter than one window."""
  n = len(ids)
  if n < window:
    return None
  counts = collections.Counter(ids[:window])
  low = len(counts)
  for i in range(window, n):
    counts[ids[i]] += 1
    counts[ids[i - window]] -= 1
    if not counts[ids[i - window]]:
      del counts[ids[i - window]]
    low = min(low, len(counts))
  run, period = 0, None
  for p in range(1, min(window, n - 1) + 1):
    same = bytes(map(operator.eq, ids[p:], ids))  # 1 where a token equals the one p places back
    longest = max(map(len, re.findall(rb"\x01+", same)), default=0)
    if longest >= p and longest > run:
      run, period = longest, p
  return {"distinct": low / window, "loop": run, "period": period}


def reference(docs:dict[str, list[int]], window:int) -> dict:
  """The corpus's own extremes on this tokenizer: the lowest distinct share and the longest loop over every document
  (diversity), each with the document it is in. Real text reaches them, so a context or reply past them is not text
  the corpus knows."""
  each = {d: diversity(ids, window) for d, ids in docs.items()}
  lo, hi = min(each, key=lambda d: each[d]["distinct"]), max(each, key=lambda d: each[d]["loop"])
  return {"window": window, "distinct": each[lo]["distinct"], "distinct_in": lo, "loop": each[hi]["loop"], "loop_in": hi,
          "period": each[hi]["period"], "documents": len(docs)}


def comms() -> list[str]:
  out = []
  for p in Path("/proc").glob("[0-9]*/comm"):
    try:
      out.append(p.read_text().strip())
    except OSError:  # it exited between the listing and the read
      pass
  return out


def quiet(wait_s:float, comms=comms) -> dict:
  """{"waited_s", "load1"} once no BUSY program runs; SystemExit past wait_s."""
  t0 = time.time()
  while busy := sorted(set(comms()) & set(BUSY)):
    if time.time() - t0 > wait_s:
      raise SystemExit(f"the CPU was not quiet in {wait_s:.0f} s ({', '.join(busy)} running): a decode beside them is slower and looks real")
    time.sleep(5)
  return {"waited_s": time.time() - t0, "load1": float(Path("/proc/loadavg").read_text().split()[0])}


def delta(after:dict, before:dict) -> dict:
  """One request's spec-decode counters: drafts (verify steps), draft tokens, accepted tokens, and accepted per draft position."""
  per = [a - (before["accepted_per_position"][i] if i < len(before["accepted_per_position"]) else 0)
         for i, a in enumerate(after["accepted_per_position"])]
  return {**{k: after[k] - before[k] for k in after if k != "accepted_per_position"}, "accepted_per_position": per}
