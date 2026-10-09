#!/usr/bin/env python3
"""Per-kernel GPU time of a real decode, in tinygrad's own runtime, with each program's role from the model.

The tinygrad fork's python runs this file by path, with the fork as the working directory, under PROFILE=1. It
imports only tinygrad and the standard library. Run it with JIT=2 (JIT on, graph off): every kernel then runs in
its own command buffer and carries its own GPU timestamps. Graph replay only gives an even split of a batch.

It writes two files for BoltBeam's own normalizer (boltbeam/artifacts/tinygrad_profile_events.py):
  --events  a pickle of the profile events of the measured decode window only (after prefill and warm tokens)
  --census  {"capture": {"program_evidence_by_jit": {...}}}: per program in the decode JIT, its name, binary hash
            and the semantic identities (role, quant, logical shape) the model attached to the call
and prints one JSON line: tokens, wall time, tok/s.
"""
from __future__ import annotations

import argparse, hashlib, json, pickle, sys, time


def _programs(jit) -> list[dict]:
  from tinygrad.engine.metadata import PROGRAM_IDENTITY_FIELDS, resolve_call_metadata
  from tinygrad.helpers import Metadata
  from tinygrad.uop.ops import Ops, ProgramInfo
  def calls(call):
    target = call.src[0]
    if target.op is Ops.PROGRAM: yield call
    elif target.op is Ops.CUSTOM_FUNCTION and target.arg == "graph" and target.src:
      for child in target.src[0].src: yield from calls(child)
  rows = []
  for top in jit.captured.linear.src:
    for call in calls(top):
      program = call.src[0]
      if not isinstance(program.arg, ProgramInfo): continue
      binary = next((x.arg for x in program.src if x.op is Ops.BINARY and isinstance(x.arg, bytes)), None)
      own = getattr(getattr(call, "arg", None), "metadata", ())
      metadata = (*own, *resolve_call_metadata(call)) if isinstance(own, tuple) else resolve_call_metadata(call)
      semantic = [{f: getattr(m, f) for f in PROGRAM_IDENTITY_FIELDS} for m in metadata
                  if isinstance(m, Metadata) and all(hasattr(m, f) for f in PROGRAM_IDENTITY_FIELDS)]
      rows.append({"program_name": program.arg.name, "program_hash": program.key.hex(),
                   "binary_sha256": hashlib.sha256(binary).hexdigest() if binary is not None else None,
                   "semantic_identities": semantic})
  return rows


def _flush() -> None:
  """Wait for every open device, which also moves its pending kernel timestamps into Compiled.profile_events."""
  from tinygrad.device import Device
  for name in sorted(Device._opened_devices): Device[name].synchronize()


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--model", required=True)
  p.add_argument("--context", type=int, default=128)
  p.add_argument("--warm", type=int, default=3)
  p.add_argument("--tokens", type=int, default=10, help="decode tokens in the measured window")
  p.add_argument("--events", required=True)
  p.add_argument("--census", required=True)
  args = p.parse_args(argv)

  from tinygrad.device import Compiled
  from tinygrad.engine.jit import TinyJit
  from tinygrad.llm.model import Transformer

  model, kv = Transformer.from_gguf(args.model, max(512, args.context + args.warm + args.tokens + 8))
  bos = kv.get("tokenizer.ggml.bos_token_id") or 0
  gen = model.generate([bos] * args.context)
  next(gen)
  for _ in range(args.warm): next(gen)
  jits = [j for v in vars(model).values() for j in (v.values() if isinstance(v, dict) else (v,)) if isinstance(j, TinyJit)]
  before = {id(j): j.cnt for j in jits}
  # A device keeps kernel timestamps on the host side until it synchronizes. HCQ devices (NV, AMD) skip that
  # sync on a token copyout (NV_COPYOUT_SKIP_PRESYNC=1 by default), so the warm tokens' records would land
  # inside the window and the window's last tokens would land outside it. Flush before clearing, and again
  # before reading, so the events are exactly the measured tokens.
  _flush()
  Compiled.profile_events.clear()
  t0 = time.perf_counter()
  for _ in range(args.tokens): next(gen)
  _flush()
  wall = time.perf_counter() - t0
  events = list(Compiled.profile_events)
  # the decode JIT is the one whose call count moved in the window
  used = [j for j in jits if j.cnt != before[id(j)] and j.captured is not None]
  census = {"capture": {"program_evidence_by_jit": {f"jit{i}": _programs(j) for i, j in enumerate(used)}}}
  with open(args.events, "wb") as fh: pickle.dump(events, fh)
  with open(args.census, "w") as fh: json.dump(census, fh)
  print(json.dumps({"tokens": args.tokens, "wall_s": wall, "tok_s": args.tokens / wall, "events": len(events),
                    "decode_jits": len(used), "context": args.context}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
