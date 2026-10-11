#!/usr/bin/env python3
"""A real prefill in tinygrad's own runtime: its untraced wall time, what the engine batched, and (with --capture) every
kernel launch of one prefill with its GPU timestamps and the program's code.

The tinygrad fork's python runs this file by path, with the fork as the working directory. It imports only tinygrad
and the standard library. One process per prompt length: the model is loaded for that length, one prefill warms the
JIT (compiles every chunk's graph), then the measured prefills run.

    --mode truth     JIT as a user runs it, no profiling: `--samples` prefills, each timed from the first forward call
                     to the first token (every chunk and the token), the median kept; the host time generate() spends
                     before its first forward call (prompt setup, graph prewarm) is recorded beside it as setup_s
    --mode capture   PROFILE=1 JIT=2 (set by the caller): one kernel per command buffer, each with GPU timestamps; one
                     measured prefill after the warm one, its launches written to --out with each program's source
    --eager          the fork's own default for an independent prompt: no workload reuse, so every prefill schedules
                     its graph again (the caller uses it when the captured graphs do not fit in GPU memory)

Every prefill uses different token ids, so the engine's prefix reuse never skips work. What the engine batched is
recorded from its own forward calls: the token width of each call during the prefill (`chunks`).

Prints one JSON line.
"""
from __future__ import annotations

import argparse, json, statistics, sys, time


def _width(x) -> int | str:
  s = x.shape[1] if len(x.shape) > 1 else x.shape[0]
  if isinstance(s, int):
    return s
  try:
    return int(s.unbind()[1])
  except Exception:  # noqa: BLE001  a symbolic width the fork cannot bind: kept as its text
    return str(s)


def _name(value) -> str:
  for attr in ("name", "display_name", "key"):
    named = getattr(value, attr, None)
    if isinstance(named, str) and named:
      return named
  return str(value)


def main(argv:list[str] | None = None) -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--model", required=True)
  p.add_argument("--length", type=int, required=True)
  p.add_argument("--mode", choices=("truth", "capture"), default="truth")
  p.add_argument("--samples", type=int, default=3)
  p.add_argument("--out")
  p.add_argument("--eager", action="store_true", help="the fork's default: no workload reuse, every prompt scheduled again")
  args = p.parse_args(argv)

  from tinygrad.llm.model import Transformer
  model, kv = Transformer.from_gguf(args.model, args.length + 16)
  # Replay, not re-scheduling: without workload reuse admitted the fork runs every independent prompt eagerly and
  # schedules its whole graph again (13.5 s of host time for a 512-token prompt against 47 ms of GPU
  # time, 2026-10-10). The prefill is measured as the captured graph replays it; the eager first call's time is kept
  # beside it (warm_s) so the scheduling cost is not hidden.
  reuse_was = getattr(model.config, "prefill_workload_reuse", None)
  if reuse_was is not None and not args.eager:
    import dataclasses
    model.config = dataclasses.replace(model.config, prefill_workload_reuse=True)
  calls: list = []
  cls = type(model)
  real_call = cls.__call__

  first: list[float] = []

  def spy(self, tokens, *a, **k):
    if not calls:
      first.append(time.perf_counter())
    calls.append(_width(tokens))
    return real_call(self, tokens, *a, **k)
  cls.__call__ = spy

  def prefill(seed:int) -> tuple[float, list]:
    calls.clear()
    first.clear()
    if hasattr(model, "reset_generation_state"):
      model.reset_generation_state()
    ids = [1000 + (seed * 7919 + i) % 30000 for i in range(args.length)]
    t0 = time.perf_counter()
    gen = model.generate(ids)
    next(gen)
    done = time.perf_counter()
    gen.close()
    setup.append(first[0] - t0 if first else None)
    return done - (first[0] if first else t0), list(calls)

  setup: list = []  # host time in generate() before the first forward call (prompt setup), not prefill
  wall, chunks = prefill(0)  # eager: schedules and compiles every chunk's graph
  capture_s = prefill(1)[0] if not args.eager else None  # the JIT captures
  begin = 1 if args.eager else 2  # the first measured prefill's seed
  out = {"length": args.length, "warm_s": wall, "capture_s": capture_s, "chunks": chunks, "mode": args.mode,
         "workload_reuse_default": reuse_was, "replay": not args.eager}
  if args.mode == "truth":
    walls = [prefill(i + begin)[0] for i in range(args.samples)]
    out.update(samples_s=walls, wall_s=statistics.median(walls), setup_s=setup[begin:])
  else:
    from tinygrad.device import Compiled, Device
    for name in sorted(Device._opened_devices): Device[name].synchronize()
    Compiled.profile_events.clear()
    wall, chunks = prefill(begin)
    for name in sorted(Device._opened_devices): Device[name].synchronize()
    launches, programs = [], {}
    for e in list(Compiled.profile_events):
      kind = type(e).__name__
      if kind == "ProfileRangeEvent" and getattr(e, "en", None) is not None and not getattr(e, "is_copy", False):
        name = _name(getattr(e, "name", ""))
        launches.append({"key": name, "start_us": float(e.st), "wall_us": float(e.en) - float(e.st), "device": str(e.device)})
      elif kind == "ProfileProgramEvent":
        name = _name(getattr(e, "name", ""))
        lib = getattr(e, "lib", None)
        programs[name] = {"lib_bytes": len(lib) if lib else 0}
    out.update(wall_s=wall, chunks=chunks, launches=len(launches))
    srcs = _sources(model)
    with open(args.out, "w") as fh:
      json.dump({"launches": launches, "programs": programs, "sources": srcs, "chunks": chunks, "census": _census(model)}, fh)
  print(json.dumps(out))
  return 0


def _census(model) -> dict:
  """Per program in the model's JITs, the semantic identities the model attached (role, quant, logical M, N, K),
  in the decode driver's census shape (runtime/tinygrad_decode_profile.py, read by its normalizer). A call whose
  metadata the fork itself refuses as invalid is listed with no identity, so its launches are paired by counts."""
  from tinygrad.engine.jit import TinyJit
  from tinygrad.engine.metadata import PROGRAM_IDENTITY_FIELDS, resolve_call_metadata
  from tinygrad.helpers import Metadata
  from tinygrad.uop.ops import Ops, ProgramInfo
  def calls(call):
    target = call.src[0]
    if target.op is Ops.PROGRAM: yield call
    elif target.op is Ops.CUSTOM_FUNCTION and target.arg == "graph" and target.src:
      for child in target.src[0].src: yield from calls(child)
  out = {}
  jits = [j for v in vars(model).values() for j in (v.values() if isinstance(v, dict) else (v,)) if isinstance(j, TinyJit)]
  for i, j in enumerate(jits):
    if getattr(j, "captured", None) is None: continue
    rows = []
    for top in j.captured.linear.src:
      for call in calls(top):
        program = call.src[0]
        if not isinstance(program.arg, ProgramInfo): continue
        try:
          own = getattr(getattr(call, "arg", None), "metadata", ())
          metadata = (*own, *resolve_call_metadata(call)) if isinstance(own, tuple) else resolve_call_metadata(call)
          semantic = [{f: getattr(m, f) for f in PROGRAM_IDENTITY_FIELDS} for m in metadata
                      if isinstance(m, Metadata) and all(hasattr(m, f) for f in PROGRAM_IDENTITY_FIELDS)]
        except ValueError:
          semantic = []
        rows.append({"program_name": program.arg.name, "binary_sha256": None, "semantic_identities": semantic})
    out[f"jit{i}"] = rows
  return {"capture": {"program_evidence_by_jit": out}}


def _sources(model) -> dict[str, str]:
  """Each program the model's JITs captured, by name: its source text (the PROGRAM's SOURCE), so BoltBeam can read
  the matrix instructions it issues."""
  from tinygrad.engine.jit import TinyJit
  from tinygrad.uop.ops import Ops, ProgramInfo
  def calls(call):
    target = call.src[0]
    if target.op is Ops.PROGRAM: yield call
    elif target.op is Ops.CUSTOM_FUNCTION and target.arg == "graph" and target.src:
      for child in target.src[0].src: yield from calls(child)
  jits = [j for v in vars(model).values() for j in (v.values() if isinstance(v, dict) else (v,)) if isinstance(j, TinyJit)]
  out: dict[str, str] = {}
  for j in jits:
    if getattr(j, "captured", None) is None: continue
    for top in j.captured.linear.src:
      for call in calls(top):
        program = call.src[0]
        if not isinstance(program.arg, ProgramInfo): continue
        src = next((x.arg for x in program.src if x.op is Ops.SOURCE and isinstance(x.arg, str)), None)
        if src is not None: out[program.arg.function_name] = src
  return out


if __name__ == "__main__":
  sys.exit(main())
