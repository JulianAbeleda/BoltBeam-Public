"""Multi-token prediction (MTP) measured in the repo, on any engine: decode speed per draft count k on natural prompts,
the spread over those prompts, and the acceptance the engine counted, against k=0 on the same prompts and engine.

This module is engine-blind (policy/guards.py checks it). The authorities for what is measured are data/mtp_prompts.json
(the short depth's prompts) and data/mtp_long_corpus.json (the long depth's: real documents cut to the depth by the
model's tokenizer, each with a task at the end, runtime/mtp_shared.long_prompts), both versioned and hashed into every
measurement. The long depth was the model's own greedy continuation until 2026-11-12, when those continuations were
found to loop (boltbeam-runs/mtp-layer-27b-20261112/README.md, CORRECTION): records before SCHEMA's v2 keep only their
short depth (workflow/recommend.mtp_of, quantization/speed_up.mtp_depths_valid). The engine's row in
data/engine_knobs.json (engines.<E>.mtp) names the module that measures it: its offer(), what Measure MTP runs for a
checkpoint, and its measure(), which hands sweep() one function that runs one k. sweep() refuses to start while
another program holds the GPU, runs k=0 first and every k after it, checks the GPU is free again after each (no engine
left behind), refuses a k whose contexts or replies repeat themselves (guard), grows the sweep while core/draft_count
says a larger k could still win, and writes boltbeam.mtp_measurement.v2: one shape for every engine, so tune, the
recommended setup and Speed up read either. summarize() and guard() are the arithmetic, kept pure so they are tested alone.

The one call every caller makes (screen mtp, Speed up's arms), stable for every engine: measure(provider, model, ks,
backend=, out=, tokens=, long_tokens=, long_why=, grow=, head=) -> the record; long_tokens is the long depth's document
tokens; each engine module's measure takes the same keywords (an engine's own, such as its python, after them). An
engine whose driver runs a server in a process group of its own writes the group's id to out/SERVER_GROUP: sweep() stops
that group after each k, whatever ended it (a killed driver included), before it checks the GPU is free."""
from __future__ import annotations

import datetime, hashlib, importlib, json, os, pathlib, signal, statistics, subprocess, time
from typing import Any, Callable

from boltbeam.collectors import providers
from boltbeam.core import draft_count

SCHEMA = "boltbeam.mtp_measurement.v3"  # v2: the long depth is real text (LONG); v3: its replies held to the loop statistic (guard). Older long depths are not read
PROMPTS = pathlib.Path(__file__).resolve().parent.parent / "data" / "mtp_prompts.json"
LONG = PROMPTS.parent / "mtp_long_corpus.json"  # the long depth's documents and their tasks
TABLE = PROMPTS.parent / "engine_knobs.json"  # engines.<E>.mtp: the module that measures MTP on E
MEASUREMENT = "mtp_measurement.json"
SERVER_GROUP = "engine-k{k}.pgid"  # the process group a k's server runs in, for sweep() to stop
KS = (1, 2, 3)  # the draft counts a measurement starts from (screen mtp --ks); the sweep goes on (core/draft_count.sweep_next)
TOKENS = 256  # tokens decoded per prompt and depth
TEMPLATE_TOKENS = 256  # the chat template's own tokens, with room: ~50 on the 27B (its 73-token short context less the prompt)
GUARD = ("every row must say how its text repeats (runtime/mtp_shared.diversity) or the k is refused: the long depth's context, and "
         "each reply with the context's last window before it (a reply exactly one window long). One is refused when it holds a loop "
         "longer than the shipped real-text corpus's longest (a stretch equal to itself one period back, at least a period long, "
         "periods up to the window), or a window with fewer distinct tokens than any of the corpus's, both as this model's tokenizer "
         "reads it (reference). Acceptance never refuses (notes: a long reply accepting more than every short one is said). "
         "Not seen: a loop whose period is longer than the window, or a loop that drifts with no exact repeat")


class MtpRefused(RuntimeError):
  """The measurement cannot start or cannot be trusted: the reason says which and why."""


def engine(provider:str):
  """The module that measures MTP on this engine, as its row names it; None when the engine has none."""
  name = (json.loads(TABLE.read_text())["engines"].get(provider) or {}).get("mtp")
  return importlib.import_module(name) if name else None


def offer(provider:str, model:str, draft_bytes:int, *, facts:bool) -> dict[str, Any] | None:
  """What Measure MTP runs for this engine and checkpoint (the engine module's offer), else None."""
  m = engine(provider)
  return m.offer(model, draft_bytes, facts=facts) if m else None


def measure(provider:str, model:str, ks:list[int], *, backend:str, out:pathlib.Path, tokens:int = TOKENS, long_tokens:int = 2048,
            long_why:str = "asked (--long-tokens)", grow:bool = False, head:str | None = None, **engine_kw) -> dict[str, Any]:
  """The measurement on this engine (its row's module); head: a draft head file, where the engine takes one."""
  m = engine(provider)
  if m is None:
    raise MtpRefused(f"{provider} has no MTP measurement in its engine row (data/engine_knobs.json)")
  return m.measure(model, ks, backend=backend, out=out, tokens=tokens, long_tokens=long_tokens, long_why=long_why, grow=grow, head=head, **engine_kw)


def stop_group(pgid_file:pathlib.Path, wait_s:float = 30.0) -> bool:
  """Stop the process group the file names (SIGTERM, then SIGKILL after wait_s) and remove the file; True when one
  was still running."""
  if not pgid_file.is_file():
    return False
  pgid, alive = int(pgid_file.read_text()), False
  for sig in (signal.SIGTERM, signal.SIGKILL):
    try:
      os.killpg(pgid, sig)
      alive = True
    except ProcessLookupError:
      break
    t0 = time.time()
    while time.time() - t0 < wait_s and _group_alive(pgid):
      time.sleep(0.2)
  pgid_file.unlink()
  return alive


def _group_alive(pgid:int) -> bool:
  try:
    os.killpg(pgid, 0)
    return True
  except ProcessLookupError:
    return False


def prompt_set(path:pathlib.Path = PROMPTS, long:pathlib.Path | None = None) -> dict[str, Any]:
  """The short depth's prompts; room: the most tokens a prompt or a long document's task adds around the text (a token
  is at least a byte), with the chat template's."""
  raw = path.read_bytes()
  data = json.loads(raw)
  if data.get("schema") != "boltbeam.mtp_prompts.v1" or not data.get("prompts"):
    raise MtpRefused(f"{path} is not a boltbeam.mtp_prompts.v1 prompt set")
  ids = [p["id"] for p in data["prompts"]]
  if len(set(ids)) != len(ids):
    raise MtpRefused(f"{path}: prompt ids repeat")
  texts = [p["text"] for p in data["prompts"]] + ["\n\n" + t for t in long_corpus(long)["tasks"]]
  return {"path": str(path), "version": data["version"], "sha256": hashlib.sha256(raw).hexdigest(), "ids": ids,
          "room": max(len(t.encode()) for t in texts) + TEMPLATE_TOKENS}


def long_corpus(path:pathlib.Path | None = None) -> dict[str, Any]:
  """The long depth's corpus (default LONG), every document's text checked against its sha256: a changed text is another measurement."""
  path = path or LONG
  raw = path.read_bytes()
  data = json.loads(raw)
  if data.get("schema") != "boltbeam.mtp_long_corpus.v1" or not data.get("documents"):
    raise MtpRefused(f"{path} is not a boltbeam.mtp_long_corpus.v1 corpus")
  bad = [d["id"] for d in data["documents"] if hashlib.sha256(d["text"].encode()).hexdigest() != d["sha256"]]
  if bad:
    raise MtpRefused(f"{path}: the text of {', '.join(bad)} does not match its sha256")
  return {"path": str(path), "version": data["version"], "sha256": hashlib.sha256(raw).hexdigest(),
          "ids": [d["id"] for d in data["documents"]], "tasks": [d["task"] for d in data["documents"]]}


def guard(rows:list[dict[str, Any]], ref:dict[str, Any]) -> list[str]:
  """Why these rows (one k's) cannot be trusted as text, in plain words; empty when all can (GUARD, the rule). Fails
  closed: a row that does not say how its text repeats is refused, never passed."""
  out, w = [], ref["window"]
  for r in rows:
    at, text = f"at the {r['depth']} depth for {r['prompt']} (k={r['k']})", r.get("text") or {}
    if r.get("tokens") != w or not text.get("reply") or (r["depth"] != "short" and not text.get("context")):
      out.append(f"the reply {at} did not say how its text repeats ({r.get('tokens')} tokens, {w} needed), so it cannot be shown to be text")
      continue
    for what in ("context", "reply"):
      d = text.get(what)
      said = [f"the same {d['period']}-token stretch over and over for {d['loop']} tokens, where the real-text corpus never goes past {ref['loop']}"] \
        if d and d["loop"] > ref["loop"] else []
      said += [f"{d['distinct'] * w:.0f} different tokens in {w}, where every stretch of the real-text corpus has at least {ref['distinct'] * w:.0f}"] \
        if d and d["distinct"] < ref["distinct"] else []
      if said:
        out.append(f"the {what} {at} repeated itself ({'; '.join(said)}), so its speed would measure repetition, not text")
  return out


def notes(rows:list[dict[str, Any]]) -> list[str]:
  """Information, never a refusal: a reply past the short depth that accepted more of its drafts than every short-depth
  reply at its k. Acceptance measures how well the head drafts this text (code continuing code drafts well), not
  whether the text repeats; the text checks decide that (guard)."""
  out = []
  for r in rows:
    shorts = [(x["spec"]["accepted_tokens"] / x["spec"]["draft_tokens"], x["prompt"]) for x in rows
              if x["depth"] == "short" and x["k"] == r["k"] and x["spec"]["draft_tokens"]]
    s = r["spec"]
    if r["depth"] != "short" and r["k"] and s["draft_tokens"] and shorts and s["accepted_tokens"] / s["draft_tokens"] > max(shorts)[0]:
      out.append(f"the reply at the {r['depth']} depth for {r['prompt']} (k={r['k']}) accepted {s['accepted_tokens'] / s['draft_tokens']:.0%} of its drafts, "
                 f"more than any short-depth reply at that k (the most, {max(shorts)[1]}, {max(shorts)[0]:.0%})")
  return out


def summarize(rows:list[dict[str, Any]]) -> dict[str, Any]:
  """Per depth and k: each prompt's decode tok/s, their median, range and spread ((max - min) / median), the speedup
  of the median over k=0's, and the acceptance (mean tokens per step, 1 + accepted / drafts; the accepted share of draft
  tokens; per draft position where the engine counts it). Refuses rows that do not cover the same prompts at every k of
  a depth: a speed per k is only comparable over one prompt set."""
  out:dict[str, Any] = {}
  for depth in sorted({r["depth"] for r in rows}):
    at = [r for r in rows if r["depth"] == depth]
    ks = sorted({r["k"] for r in at})
    if 0 not in ks:
      raise MtpRefused(f"{depth}: no k=0 baseline; a speedup needs the same prompts without speculation")
    prompts = {k: sorted(r["prompt"] for r in at if r["k"] == k) for k in ks}
    if any(p != prompts[0] for p in prompts.values()):
      raise MtpRefused(f"{depth}: the k values did not run the same prompts ({prompts})")
    if any(len({r.get("prompt_sha") for r in at if r["prompt"] == p}) > 1 for p in prompts[0]):
      raise MtpRefused(f"{depth}: the k values read different token ids for one prompt; a speed per k is only comparable on one text")
    per_k = {}
    for k in ks:
      rs = sorted((r for r in at if r["k"] == k), key=lambda r: r["prompt"])
      speeds = [r["decode_tok_s"] for r in rs]
      med = statistics.median(speeds)
      drafts = sum(r["spec"]["drafts"] for r in rs)
      draft_tokens = sum(r["spec"]["draft_tokens"] for r in rs)
      accepted = sum(r["spec"]["accepted_tokens"] for r in rs)
      positions = [sum(r["spec"]["accepted_per_position"][i] for r in rs if i < len(r["spec"]["accepted_per_position"]))
                   for i in range(k)] if any(r["spec"]["accepted_per_position"] for r in rs) else None
      if k and not drafts:
        raise MtpRefused(f"{depth} k={k}: the engine counted no drafts; speculation did not run")
      per_k[str(k)] = {
        "per_prompt": {r["prompt"]: round(r["decode_tok_s"], 3) for r in rs},
        "median_tok_s": med, "min_tok_s": min(speeds), "max_tok_s": max(speeds),
        "spread": (max(speeds) - min(speeds)) / med,
        "context": statistics.median(r["context"] for r in rs),
        **({"accept_length": 1 + accepted / drafts, "accept_rate": accepted / draft_tokens,
            **({"accept_rate_per_position": [p / drafts for p in positions]} if positions else {})} if k else {})}
    base = per_k["0"]["median_tok_s"]
    for k, v in per_k.items():
      v["speedup"] = v["median_tok_s"] / base
    out[depth] = per_k
  return out


def by_context(depths:dict[str, Any]) -> dict[float, dict[int, tuple]]:
  """A summary (summarize) as core/draft_count reads it: {context: {k: (per_prompt, median tok/s)}}."""
  return {next(iter(per_k.values()))["context"]: {int(k): (v["per_prompt"], v["median_tok_s"]) for k, v in per_k.items()}
          for per_k in depths.values() if per_k}


def sweep(run_k:Callable[[int], list[dict[str, Any]]], ks:list[int], *, model:str, backend:str, out:pathlib.Path,
          tokens:int, long_tokens:int, long_why:str, grow:bool, decode_speed:str) -> dict[str, Any]:
  """Run k=0 then each k in `ks` through run_k (one engine start: its prompt rows and one {"engine"} row), then (grow)
  one k more while core/draft_count.sweep_next says a larger k could still win; write and return the measurement. The
  GPU must be free before, and free again after each k. decode_speed: how the engine's rows time a token, said."""
  ks = sorted({int(k) for k in ks})
  if not ks or ks[0] < 1:
    raise MtpRefused("ks are the draft counts to measure, 1 or more; k=0, the baseline, always runs")
  free = providers.gpu_free(backend)
  if not free["free"]:
    raise MtpRefused(f"not started: {free['reason']}; a GPU another program holds gives slower numbers that look real")
  out.mkdir(parents=True, exist_ok=True)
  rows, engines, todo, why, ref, said_too = [], [], [0, *ks], [], None, []
  while todo:
    k = todo.pop(0)
    try:
      got = run_k(k)
    except subprocess.TimeoutExpired as exc:
      raise MtpRefused(f"k={k} ran past {exc.timeout:.0f} s and was stopped; its log is in {out}") from exc
    finally:
      stop_group(out / SERVER_GROUP.format(k=k))
    after = providers.gpu_free(backend)
    if not after["free"]:
      raise MtpRefused(f"k={k} left the GPU held: {after['reason']}; the next k would not run alone")
    new, engine_line = [r for r in got if "prompt" in r], [r["engine"] for r in got if "engine" in r]
    refs = [e.get("real_text") for e in engine_line]
    if not refs or None in refs or (ref and refs != [ref]):
      raise MtpRefused(f"k={k}: the driver did not say how diverse the real-text corpus is on this tokenizer, or said it differently from k=0's")
    ref = refs[0]
    if said := guard(new, ref):
      raise MtpRefused(f"k={k}: " + "; ".join(said[:3]) + (f"; and {len(said) - 3} more" if len(said) > 3 else "") + f". Its logs are in {out}")
    rows, engines, said_too = rows + new, engines + engine_line, said_too + notes(new)
    if grow and not todo:
      more, said = draft_count.sweep_next(by_context(summarize(rows)))
      why.append(said)
      todo += [more] if more else []
  long = long_corpus()
  record = {"schema": SCHEMA, "model": model, "prompts": prompt_set(), "tokens": tokens, "long_tokens": long_tokens, "long_why": long_why,
            "long_text": {k: long[k] for k in ("path", "version", "sha256", "ids")} if long_tokens else None, "guard": {"rule": GUARD, "reference": ref, "notes": said_too},
            "ks": sorted({r["k"] for r in rows}), "sweep": why, "engines": engines, "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "decode_speed": decode_speed, "depths": summarize(rows), "rows": rows}
  (out / MEASUREMENT).write_text(json.dumps(record, indent=1))
  return record
