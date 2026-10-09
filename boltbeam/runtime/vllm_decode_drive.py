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
"""
from __future__ import annotations

import argparse, json, statistics, time


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--contexts", default="128")
  ap.add_argument("--batches", default="1")
  ap.add_argument("--tokens", type=int, default=64)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--warm", type=int, default=1, help="warm calls of 2 tokens before the timed ones")
  ap.add_argument("--once", type=int, default=None, help="capture mode: one call of this many tokens")
  ap.add_argument("--dtype", default="auto")
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
  a = ap.parse_args()
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt
  contexts = [int(x) for x in a.contexts.split(",")]
  batches = [int(x) for x in a.batches.split(",")]
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


if __name__ == "__main__":
  raise SystemExit(main())
