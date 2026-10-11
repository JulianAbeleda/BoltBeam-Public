#!/usr/bin/env python3
"""A chat session's costs at depth in tinygrad's own runtime: the turn (N fresh tokens appended to a KV cache already
holding C tokens), decode at that depth, the next tokens' probabilities, and (with --capture) every kernel launch of
one turn.

The tinygrad fork's python runs this file by path, with the fork as the working directory. It imports only tinygrad,
the standard library and its sibling prefill driver (tinygrad_prefill_profile.py, for the call spy, the program
census and the sources). One process per depth:

    fill     generate() over the first C tokens of --tokens (the fork's own prefill), timed. The fill is setup, not a measurement, and runs
             on the fork's own short-chunk path (prefill_v2 off for the fill: one symbolic graph of up to 32 tokens
             at any position, replayed). The fork holds about two prefill-v2 graphs beside its weights: on its default
             path every chunk position is its own graph, and with one symbolic graph for the positions past the
             first, the second chunk still runs out of GPU memory (Allocation failed, 2026-10-10 and 2026-10-11). The
             check below proves the cache the fill wrote is read right
    check    generate() over the first C + N tokens, on the fork's default path: it keeps the C tokens it holds and
             appends N (its prefix match is how a session's earlier turns stay in the cache); the next --check greedy
             tokens and the top probabilities of each (diagnostic_full_logits). The first token comes from the prefill
             graph, which returns no logits, so it is compared by its id alone
    turns    N new tokens (later text of --tokens) appended to exactly what the cache holds, turn after turn, as a
             chat appends: a recurrent state cannot go back to an earlier position, so each turn starts where the last
             ended and its depth grows by N plus the token it produced (each turn's depth is recorded). On the fork's
             default path with workload reuse admitted: the first turn runs with its compiles, one captures, then
             --samples are timed from the first forward call to the first token; the last one decodes --decode more
             tokens, each timed: decode at that depth. The forward calls' token widths are recorded, so a turn that
             computed other than N tokens shows it
    warm     (--warm) what the cache holds, sent again: the warm-cache case, timed the same way, with what the
             engine computed for it
    capture  (--mode capture, PROFILE=1 JIT=2 set by the caller) after the compile and capture turns, one turn with
             every launch, its GPU timestamps and each program's source, written to --out

Prints one JSON line.
"""
from __future__ import annotations

import argparse, heapq, json, math, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinygrad_prefill_profile as pp  # noqa: E402

DECODE_WARM = 4  # decode tokens left out of the rate: the decode graph compiles and captures in the first ones
TOP = 5


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--model", required=True)
  p.add_argument("--tokens", required=True, help="JSON: token ids (or {ids: [...]}): the context, then text for the turns")
  p.add_argument("--depth", type=int, required=True)
  p.add_argument("--turn", type=int, required=True)
  p.add_argument("--samples", type=int, default=3)
  p.add_argument("--decode", type=int, default=64)
  p.add_argument("--check", type=int, default=16)
  p.add_argument("--warm", action="store_true")
  p.add_argument("--mode", choices=("truth", "capture"), default="truth")
  p.add_argument("--out")
  args = p.parse_args(argv)
  data = json.load(open(args.tokens))
  ids = [int(x) for x in (data["ids"] if isinstance(data, dict) else data)]
  C, N = args.depth, args.turn
  need = C + N * (args.samples + 4)
  if len(ids) < need:
    raise SystemExit(f"--tokens holds {len(ids)} ids; depth {C} with turns of {N} needs {need}")

  from tinygrad.llm.model import Transformer
  # the context the turns reach: the check's C + N and its tokens, then samples + 2 turns of N and the token each
  # produced, the last with its decode (generate() yields nothing once the tokens reach max_context)
  model, _ = Transformer.from_gguf(args.model, C + N * (args.samples + 3) + args.check + args.samples + args.decode + 16)
  import dataclasses
  config = model.config
  reuse_was = getattr(config, "prefill_workload_reuse", None)
  v2_was = getattr(config, "prefill_v2", None)
  calls: list = []
  first: list[float] = []
  cls = type(model)
  real_call = cls.__call__

  def spy(self, tokens, *a, **k):
    if not calls:
      first.append(time.perf_counter())
    calls.append(pp._width(tokens))
    return real_call(self, tokens, *a, **k)
  cls.__call__ = spy

  def run(tokens:list[int], extra:int = 0, logits:bool = False) -> dict:
    """generate() over `tokens`: the time from its first forward call to the first token, the calls' widths, then
    `extra` more tokens, each with its time (and its top probabilities when logits)."""
    calls.clear()
    first.clear()
    gen = model.generate(list(tokens), diagnostic_full_logits=logits)
    out = next(gen)
    done = time.perf_counter()
    widths = list(calls)
    got = {"s": done - first[0], "widths": widths, "tokens": [out[0] if logits else out], "top": [None], "times": [done]}
    for _ in range(extra):
      out = next(gen)
      got["times"].append(time.perf_counter())
      got["tokens"].append(out[0] if logits else out)
      got["top"].append(top_probs(out[1]) if logits else None)
    gen.close()
    return got

  out = {"depth": C, "turn": N, "mode": args.mode, "workload_reuse_default": reuse_was, "prefill_v2_default": v2_was}
  if v2_was:  # the fill on the short-chunk path (the docstring says why)
    model.config = dataclasses.replace(config, prefill_v2=False)
  fill = run(ids[:C])
  model.config = config
  out.update(fill_s=fill["s"], fill_chunks=fill["widths"], fill_path="short chunks (prefill_v2 off)" if v2_was else "default")
  if args.mode == "truth":
    check = run(ids[:C + N], args.check, logits=True)
    out["check"] = {"widths": check["widths"], "tokens": check["tokens"], "top": check["top"]}
  if reuse_was is not None:  # replay the turn, as the prefill driver replays its prefill
    model.config = dataclasses.replace(config, prefill_workload_reuse=True)
  start, turns = C + N, []
  last = args.samples + 1
  for i in range(args.samples + 2):
    history = list(model._cached_tokens)  # what the cache holds: the next turn appends to exactly this
    piece, start = ids[start:start + N], start + N
    capture = i == 2 and args.mode == "capture"
    if capture:
      from tinygrad.device import Compiled, Device
      for name in sorted(Device._opened_devices): Device[name].synchronize()
      Compiled.profile_events.clear()
    got = run(history + piece, args.decode if i == last and args.mode == "truth" else 0)
    turns.append({"s": got["s"], "widths": got["widths"], "depth": len(history)})
    if capture:
      for name in sorted(Device._opened_devices): Device[name].synchronize()
      launches = []
      for e in list(Compiled.profile_events):
        if type(e).__name__ == "ProfileRangeEvent" and getattr(e, "en", None) is not None and not getattr(e, "is_copy", False):
          launches.append({"key": pp._name(getattr(e, "name", "")), "start_us": float(e.st), "wall_us": float(e.en) - float(e.st),
                           "device": str(e.device)})
      with open(args.out, "w") as fh:
        json.dump({"launches": launches, "programs": {}, "sources": pp._sources(model), "chunks": got["widths"],
                   "census": pp._census(model)}, fh)
      out.update(launches=len(launches), wall_s=got["s"], chunks=got["widths"], turn_depth=len(history))
      break
  out.update(compile_s=turns[0]["s"], capture_s=turns[1]["s"] if len(turns) > 1 else None)
  if args.mode == "truth":
    timed = turns[2:]
    steady = got["times"][1 + DECODE_WARM:]
    out["decode"] = {"tokens": len(steady) - 1, "s": steady[-1] - steady[0], "tok_s": (len(steady) - 1) / (steady[-1] - steady[0]),
                     "from": timed[-1]["depth"] + N + 1 + DECODE_WARM, "warm_left_out": DECODE_WARM}
    out.update(samples_s=[t["s"] for t in timed], wall_s=statistics.median(t["s"] for t in timed), chunks=timed[-1]["widths"],
               computed=[sum(w for w in t["widths"] if isinstance(w, int)) for t in timed], turn_depths=[t["depth"] for t in timed])
    if args.warm:  # what the cache holds, sent again
      held = list(model._cached_tokens)
      warm = run(held)
      out["warm"] = {"s": warm["s"], "widths": warm["widths"], "depth": len(held)}
  print(json.dumps(out))
  return 0


def top_probs(logits) -> list[list[float]]:
  """The TOP most probable next tokens as [id, probability], from one position's logits (softmax over the whole
  vocabulary)."""
  row = [float(x) for x in logits.float().reshape(-1).tolist()]
  peak = max(row)
  z = sum(math.exp(v - peak) for v in row)
  best = heapq.nlargest(TOP, range(len(row)), key=row.__getitem__)
  return [[i, math.exp(row[i] - peak) / z] for i in best]


if __name__ == "__main__":
  sys.exit(main())
