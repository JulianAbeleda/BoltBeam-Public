#!/usr/bin/env python3
"""Decode time per step in vLLM, at a fixed context and batch, through vLLM's offline LLM API.

vLLM's python runs this file by path. It imports only vllm and the standard library.

How one point is measured. Each of B streams gets its own prompt of exactly C token ids (TokensPrompt), so the
context is fixed and no stream shares a prefix with another. The same batch is generated twice: once with N
new tokens and once with 1 (min_tokens = max_tokens, ignore_eos, greedy). Both calls do the same prefill, so

    step ms = (time of N tokens - time of 1 token) / (N - 1)

is the decode step alone: B tokens per step, one per stream. Prefix caching is off, so the second call prefills
again instead of reusing the first call's cache. vLLM runs as a user runs it: CUDA graphs on (its default).

It prints one JSON line per point: {"context", "batch", "tokens", "long_s", "short_s", "step_ms",
"tok_s_stream", "tok_s_total"}. With --reps R each call is the median of R runs after one warm call.

--once N is the capture mode: the warm calls, then one call of N tokens at the first context and batch, and one
line {"tokens", "seconds"}. Two such runs, N and 1, differ only in the decode steps, so a capture of each can be
subtracted kernel by kernel (collectors/attribution.shared_window).

--prefill L is the prefill mode (collectors/prefill.py): one prompt of L token ids drawn from a fixed seed (each call
its own ids), max_tokens 1, prefix caching off, after --warm calls. Untimed by BoltBeam: vLLM's own request timing,
first token less scheduled (what its server reports as time to first token), the median of --reps calls; the wall
time of each call is kept beside it. The line also says the scheduler's max_num_batched_tokens and chunked prefill,
the engine's chunk. With --capture the one measured call is wrapped in cudaProfilerStart/Stop, so a capture holds it
alone, and the line adds its system-wide monotonic start and end (the clock telemetry is read over that window), the weight matrices the model runs (name, N x K, count, the checkpoint tensors packed into
it) and the shared libraries this process loaded: vLLM's own view of what it ran (run with the engine in-process).
"""
from __future__ import annotations

import argparse, json, random, statistics, time


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--contexts", default="128")
  ap.add_argument("--batches", default="1")
  ap.add_argument("--tokens", type=int, default=64)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--warm", type=int, default=1, help="warm calls of 2 tokens before the timed ones")
  ap.add_argument("--once", type=int, default=None, help="capture mode: one call of this many tokens")
  ap.add_argument("--prefill", type=int, default=None, help="prefill mode: one prompt of this many tokens")
  ap.add_argument("--capture", action="store_true", help="prefill mode: the measured call alone, marked for the profiler")
  ap.add_argument("--seed", type=int, default=0, help="prefill mode: the seed the prompt ids are drawn from")
  ap.add_argument("--dtype", default="auto")
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
  a = ap.parse_args()
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt
  contexts = [int(x) for x in a.contexts.split(",")]
  batches = [int(x) for x in a.batches.split(",")]
  if a.prefill:
    return prefill(a)
  max_len = max(contexts) + max(a.tokens, a.once or 0) + 8
  llm = LLM(model=a.model, dtype=a.dtype, max_model_len=max_len, enable_prefix_caching=False,
            gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=max(max(batches), 1), seed=0)
  vocab = llm.get_tokenizer().vocab_size

  def prompts(context:int, batch:int) -> list:
    # stream s: ids 1000+s, 1001+s, ... so no two streams share a prefix
    return [TokensPrompt(prompt_token_ids=[(1000 + s * 7 + i) % vocab for i in range(context)]) for s in range(batch)]

  def timed(context:int, batch:int, n:int) -> float:
    params = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0)
    t0 = time.perf_counter()
    out = llm.generate(prompts(context, batch), params, use_tqdm=False)
    dt = time.perf_counter() - t0
    got = [len(o.outputs[0].token_ids) for o in out]
    if got != [n] * batch:
      raise SystemExit(f"asked for {n} tokens per stream, got {got}")
    return dt

  if a.once:
    for _ in range(a.warm):
      timed(contexts[0], batches[0], 2)
    print(json.dumps({"tokens": a.once, "seconds": timed(contexts[0], batches[0], a.once)}), flush=True)
    return 0
  for context in contexts:
    for batch in batches:
      for _ in range(a.warm):
        timed(context, batch, 2)
      long_s = statistics.median(timed(context, batch, a.tokens) for _ in range(a.reps))
      short_s = statistics.median(timed(context, batch, 1) for _ in range(a.reps))
      step_ms = (long_s - short_s) * 1000.0 / (a.tokens - 1)
      print(json.dumps({"context": context, "batch": batch, "tokens": a.tokens, "long_s": long_s, "short_s": short_s,
                        "step_ms": step_ms, "tok_s_stream": 1000.0 / step_ms, "tok_s_total": batch * 1000.0 / step_ms}),
            flush=True)
  return 0


def prefill(a) -> int:
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt
  llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.prefill + 8, enable_prefix_caching=False,
            gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=1, seed=0, disable_log_stats=False)
  vocab = llm.get_tokenizer().vocab_size
  rng = random.Random(a.seed)
  params = SamplingParams(max_tokens=1, temperature=0.0)

  def call() -> tuple[float, float | None]:
    ids = [rng.randrange(vocab) for _ in range(a.prefill)]
    t0 = time.perf_counter()
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], params, use_tqdm=False)[0]
    wall = time.perf_counter() - t0
    m = getattr(out, "metrics", None)
    first, sched = getattr(m, "first_token_ts", 0.0), getattr(m, "scheduled_ts", 0.0)
    return wall, (first - sched if first and sched else None)

  for _ in range(a.warm):
    call()
  sc = llm.llm_engine.vllm_config.scheduler_config
  line = {"length": a.prefill, "seed": a.seed, "max_num_batched_tokens": int(sc.max_num_batched_tokens),
          "chunked_prefill": bool(sc.enable_chunked_prefill), "vllm": __import__("vllm").__version__}
  if a.capture:
    import torch
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    t0 = time.monotonic()
    wall, engine = call()
    torch.cuda.synchronize()
    t1 = time.monotonic()
    torch.cuda.profiler.stop()
    line.update(wall_s=wall, engine_s=engine, window_monotonic=[t0, t1], matrices=_matrices(llm), libraries=_libraries())
  else:
    got = [call() for _ in range(a.reps)]
    line.update(wall_samples_s=[w for w, _ in got], engine_samples_s=[e for _, e in got])
  print(json.dumps(line), flush=True)
  return 0


def _matrices(llm) -> list | str:
  """The 2-D weights the model runs, by module name: {name, n, k, dtype, count, parts}; parts are the checkpoint
  tensors vLLM packed into it (packed_modules_mapping), else the module itself."""
  def read(model):
    packed = dict(getattr(model, "packed_modules_mapping", None) or {})
    out: dict = {}
    for name, mod in model.named_modules():
      w = getattr(mod, "weight", None)
      if w is None or not hasattr(w, "dim") or w.dim() != 2:
        continue
      last = name.split(".")[-1]
      key = (last, int(w.shape[0]), int(w.shape[1]), str(w.dtype))
      out[key] = out.get(key, 0) + 1
    return [{"name": k[0], "n": k[1], "k": k[2], "dtype": k[3], "count": c, "parts": list(packed.get(k[0], [k[0]]))}
            for k, c in sorted(out.items())]
  try:
    return llm.apply_model(read)[0]
  except Exception as exc:  # the engine's API differs: said, the adapter then pairs the checkpoint's roles
    return f"not read: {type(exc).__name__}: {str(exc)[:200]}"


def _libraries() -> list:
  with open("/proc/self/maps") as fh:
    return sorted({l.split()[-1] for l in fh if ".so" in l.split()[-1] and l.split()[-1].startswith("/")})


if __name__ == "__main__":
  raise SystemExit(main())
