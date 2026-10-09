#!/usr/bin/env python3
"""A matched whole-model decode A/B, timed in tinygrad's own Metal runtime, with the model loaded once.

The tinygrad fork's python runs this file by path, with the fork as the working directory. It imports only
tinygrad and the standard library, never boltbeam, so the fork needs no BoltBeam install.

Arms alternate default, plan, default, plan, ... in one process. Between arms every JIT is reset, so the plan is
compiled in or out at capture. The plan is installed in tinygrad's real warm-start table (whose entries are part
of the program cache key), on the decode matvec kernels whose output size is N and reduce size is K. Those keys
and the axis holding the N rows are read off the model's own kernels during the first default arm.

It prints one JSON object: tok/s per block for each arm, the greedy tokens of each arm (the correctness check),
and coverage: how many decode graph calls ran a program the plan produced, out of the calls it replaced.
"""
from __future__ import annotations

import argparse, collections, json, math, resource, statistics, sys, time


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--model", required=True)
  p.add_argument("--opts", required=True, help='JSON list of {"op","axis","arg"}: the plan, axis 0 = the N rows')
  p.add_argument("--shape", required=True, help="N,K of the role")
  p.add_argument("--pairs", type=int, default=3)
  p.add_argument("--context", type=int, default=128, help="prefill depth before decode (the July record used 128)")
  p.add_argument("--warm", type=int, default=3, help="decode tokens dropped while the JIT settles")
  p.add_argument("--blocks", type=int, default=3)
  p.add_argument("--block-tokens", type=int, default=10)
  args = p.parse_args(argv)

  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.codegen.opt import postrange as pr
  from tinygrad.engine.jit import GraphRunner, TinyJit
  from tinygrad.llm.model import Transformer

  n, k = (int(x) for x in args.shape.split(","))
  plan = json.loads(args.opts)

  # every decode graph call, by program name, for the arm in flight
  calls: collections.Counter = collections.Counter()
  graph_init = GraphRunner.__init__
  def counting_init(self, *a, **kw):
    graph_init(self, *a, **kw)
    for call in self.calls:
      prg = call[1]
      info = prg.src[0].arg if prg.src else None
      # the program's content hash tells a plan program from the default one; the name is for reading
      calls[f"{getattr(info, 'name', '?')}#{prg.key.hex()[:12]}"] += 1
  GraphRunner.__init__ = counting_init

  # the role's warm-start keys and the axis of the N rows, read off the model's kernels
  keys: dict = {}
  original_match = pr._warmstart_match
  def recording_match(kernel):
    key = pr._warmstart_key(kernel)
    out, red, _ = key
    if red == k and math.prod(out) == n:
      rows = next((i for i, size in enumerate(kernel.full_shape) if size == n), None)
      keys[key] = rows
    return None

  started = time.perf_counter()
  model, kv = Transformer.from_gguf(args.model, max(512, args.context + args.warm + args.blocks * args.block_tokens + 8))
  load_s = time.perf_counter() - started
  bos = kv.get("tokenizer.ggml.bos_token_id") or 0

  def reset_jits():
    for value in vars(model).values():
      for jit in (value.values() if isinstance(value, dict) else (value,)):
        if isinstance(jit, TinyJit) and jit.fxn is not None: jit.reset()

  def arm():
    reset_jits()
    gen = model.generate([bos] * args.context)
    tokens = [next(gen)]
    calls.clear()  # count the decode graph only
    for _ in range(args.warm): tokens.append(next(gen))
    blocks = []
    for _ in range(args.blocks):
      t0 = time.perf_counter()
      for _ in range(args.block_tokens): tokens.append(next(gen))
      blocks.append(args.block_tokens / (time.perf_counter() - t0))
    return {"tok_s_samples": blocks, "tokens": tokens, "calls": dict(calls)}

  def install(opts_by_key):
    pr._WARMSTART_OPTS = opts_by_key
    pr._warmstart_stats.update(match=0, apply=0, error=0)
    pr._warmstart_stats.pop("errs", None)

  # first default arm also records the keys
  pr._warmstart_match, pr._WARMSTART_OPTS = recording_match, {}
  base, cand = [arm()], []
  pr._warmstart_match = original_match
  forced = {key: tuple(Opt(OptOps[o["op"]], rows + int(o["axis"]) if o["op"] in ("LOCAL", "UPCAST") else int(o["axis"]), o["arg"])
                       for o in plan) for key, rows in keys.items() if rows is not None}
  binding = {"keys": len(keys), "keys_with_rows_axis": len(forced), "applied": 0, "errors": 0, "error_samples": []}
  for i in range(args.pairs):
    if forced:
      install(forced)
      cand.append(arm())
      st = pr._warmstart_stats
      binding["applied"], binding["errors"] = max(binding["applied"], int(st.get("apply", 0))), max(binding["errors"], int(st.get("error", 0)))
      binding["error_samples"] = [str(e)[:200] for e in st.get("errs", [])[:3]] or binding["error_samples"]
    install(None)
    if i < args.pairs - 1 or not forced: base.append(arm())
  if not forced: cand = []
  # coverage: calls that ran a plan program, out of the default calls those programs replaced
  if cand:
    new = {name for name in cand[-1]["calls"] if name not in base[0]["calls"]}
    gone = {name for name in base[0]["calls"] if name not in cand[-1]["calls"]}
    reached = sum(cand[-1]["calls"][x] for x in new)
    replaced = sum(base[0]["calls"][x] for x in gone)
  else:
    reached = replaced = 0
  binding.update(calls_reached=reached, calls_replaced=replaced,
                 decode_graph_calls=sum(base[0]["calls"].values()))
  out = {"load_s": load_s, "context": args.context, "pairs": args.pairs, "binding": binding,
         "baseline": [{"tok_s_samples": a["tok_s_samples"], "tokens": a["tokens"]} for a in base],
         "candidate": [{"tok_s_samples": a["tok_s_samples"], "tokens": a["tokens"]} for a in cand],
         "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  print(json.dumps(out))
  return 0


if __name__ == "__main__":
  sys.exit(main())
