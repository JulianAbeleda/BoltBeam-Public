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

import argparse, json, os, random, statistics, time
from pathlib import Path

import build_jobs

# Direct driver invocation gets the same policy; multiprocessing workers start
# fresh Python interpreters and import sitecustomize through PYTHONPATH.
_runtime = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = os.pathsep.join([_runtime, *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])])
os.environ.setdefault("BOLTBEAM_BUILD_LIMIT", "1")
os.environ.setdefault("BOLTBEAM_REQUIRE_PREBUILT", "1")
build_jobs.install_ninja_limit()


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--model-context-limit", type=int)
  ap.add_argument("--contexts", default="", help="explicit decode depths; prefill uses --prefill")
  ap.add_argument("--batches", default="1")
  ap.add_argument("--tokens", type=int, default=64)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--warm", type=int, default=1, help="warm calls of 2 tokens before the timed ones")
  ap.add_argument("--once", type=int, default=None, help="capture mode: one call of this many tokens")
  ap.add_argument("--prefill", type=int, default=None, help="prefill mode: one prompt of this many tokens")
  ap.add_argument("--capture", action="store_true", help="prefill mode: the measured call alone, marked for the profiler")
  ap.add_argument("--seed", type=int, default=0, help="prefill mode: the seed the prompt ids are drawn from")
  ap.add_argument("--max-num-batched-tokens", type=int)
  ap.add_argument("--dtype", default="auto")
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
  ap.add_argument("--speculative-config", default=os.environ.get("BOLTBEAM_VLLM_SPECULATIVE_CONFIG"),
                  help="explicit MTP configuration; recorded in the measurement")
  ap.add_argument("--quality-suite", help="JSON prompts and expected JSON answers for a bounded smoke check")
  ap.add_argument("--quality-thinking", action="store_true", help="enable thinking for quality prompts")
  ap.add_argument("--quality-max-tokens", type=int, help="answer budget including thinking (default: 4096 on, 128 off)")
  ap.add_argument("--quality-only", action="store_true", help="run quality checks without decode benchmarks")
  a = ap.parse_args()
  if a.quality_max_tokens is None:
    a.quality_max_tokens = 4096 if a.quality_thinking else 128
  if a.quality_max_tokens <= 0:
    ap.error("--quality-max-tokens must be positive")
  if a.quality_only and not a.quality_suite:
    ap.error("--quality-only requires --quality-suite")
  a.speculative_config = json.loads(a.speculative_config) if a.speculative_config else None
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt
  contexts = [int(x) for x in a.contexts.split(",") if x]
  batches = [int(x) for x in a.batches.split(",")]
  if a.prefill:
    return prefill(a)
  if not contexts and not a.quality_only:
    ap.error("decode needs explicit --contexts")
  max_len = a.model_context_limit or max(
    max(contexts, default=0) + max(a.tokens, a.once or 0) + 8,
    a.quality_max_tokens + 512 if a.quality_suite and (a.quality_thinking or a.quality_only) else 0)
  llm = LLM(model=a.model, dtype=a.dtype, max_model_len=max_len, enable_prefix_caching=False,
            language_model_only=True, speculative_config=a.speculative_config,
            gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=max(max(batches), 1), seed=0,
            **({"max_num_batched_tokens": a.max_num_batched_tokens, "enable_chunked_prefill": True} if a.max_num_batched_tokens else {}))
  vocab = llm.get_tokenizer().vocab_size
  if a.quality_suite:
    tokenizer = llm.get_tokenizer()
    suite = json.load(open(a.quality_suite))
    quality_prompts = [tokenizer.apply_chat_template([
      {"role": "system", "content": "Answer with only a JSON object containing the key answer. No explanation."},
      {"role": "user", "content": c["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=a.quality_thinking) for c in suite]
    if any(len(tokenizer.encode(prompt, add_special_tokens=False)) + a.quality_max_tokens > max_len
           for prompt in quality_prompts):
      raise ValueError("Quality prompt and generation budget exceed --model-context-limit")
    quality_started = time.perf_counter()
    responses = llm.generate(quality_prompts, SamplingParams(max_tokens=a.quality_max_tokens, temperature=0.0), use_tqdm=False)
    checks = []
    for case, response in zip(suite, responses):
      text = response.outputs[0].text
      from quality_checks import grade_response
      checks.append(grade_response(case, text, response.outputs[0].token_ids,
                                   thinking=a.quality_thinking, finish_reason=response.outputs[0].finish_reason))
    print(json.dumps({"quality": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks),
                      "strict_passed": sum(c["strict_passed"] for c in checks),
                      "thinking": a.quality_thinking, "max_tokens": a.quality_max_tokens,
                      "temperature": 0.0, "seconds": time.perf_counter() - quality_started,
                      "speculative_config": a.speculative_config}), flush=True)

  if a.quality_only:
    return 0

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
    print(json.dumps({"tokens": a.once, "seconds": timed(contexts[0], batches[0], a.once), "matrices": _matrices(llm)}),
          flush=True)
    return 0
  for context in contexts:
    for batch in batches:
      for _ in range(a.warm):
        timed(context, batch, 2)
      long_s = statistics.median(timed(context, batch, a.tokens) for _ in range(a.reps))
      short_s = statistics.median(timed(context, batch, 1) for _ in range(a.reps))
      step_ms = (long_s - short_s) * 1000.0 / (a.tokens - 1)
      print(json.dumps({"context": context, "batch": batch, "tokens": a.tokens, "long_s": long_s, "short_s": short_s,
                        "engine": {"build": __import__("vllm").__version__, "max_model_len": max_len,
                                   "speculative_config": a.speculative_config,
                                   "max_num_batched_tokens": int(llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens),
                                   "chunked_prefill": bool(llm.llm_engine.vllm_config.scheduler_config.enable_chunked_prefill)},
                        "step_ms": step_ms, "tok_s_stream": 1000.0 / step_ms, "tok_s_total": batch * 1000.0 / step_ms}),
            flush=True)
  return 0


def prefill(a) -> int:
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt
  llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.prefill + 8, enable_prefix_caching=False,
            language_model_only=True, speculative_config=a.speculative_config,
            gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=1, seed=0, disable_log_stats=False,
            **({"max_num_batched_tokens": a.max_num_batched_tokens, "enable_chunked_prefill": True} if a.max_num_batched_tokens else {}))
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
          "speculative_config": a.speculative_config,
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
    line.update(wall_s=wall, engine_s=engine, window_monotonic=[t0, t1], matrices=_matrices(llm), libraries=_libraries(),
                jit_caches=[d for d in (os.environ.get(k) for k in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")) if d])
  else:
    got = [call() for _ in range(a.reps)]
    line.update(wall_samples_s=[w for w, _ in got], engine_samples_s=[e for _, e in got])
  print(json.dumps(line), flush=True)
  return 0


def _matrices(llm) -> list | str:
  """The 2-D weights the model runs, by module name: {name, n, k, dtype, count, parts}; parts are the checkpoint
  tensors vLLM packed into it (packed_modules_mapping), else the module itself, each under its parent module
  (linear_attn.out_proj is not self_attn.o_proj). A quantized linear layer holds no `weight` (its packed integers
  are other tensors): its shape is the layer's own input and output sizes, its dtype its quantization method."""
  def read(model):
    packed = dict(getattr(model, "packed_modules_mapping", None) or {})
    out: dict = {}
    for name, mod in model.named_modules():
      w = getattr(mod, "weight", None)
      if w is not None and hasattr(w, "dim"):
        if w.dim() != 2:
          continue
        n, k, dtype = int(w.shape[0]), int(w.shape[1]), str(w.dtype)
      elif hasattr(mod, "output_size_per_partition") and getattr(mod, "quant_method", None) is not None:
        n, k = int(mod.output_size_per_partition), int(mod.input_size_per_partition)
        dtype = type(mod.quant_method).__name__ + ":" + type(getattr(mod, "scheme", None) or mod.quant_method).__name__
      else:
        continue
      parent, last = (name.split(".")[-2:] if "." in name else ["", name])
      key = (parent, last, n, k, dtype)
      out[key] = out.get(key, 0) + 1
    return [{"name": k[1], "n": k[2], "k": k[3], "dtype": k[4], "count": c,
             "parts": [f"{k[0]}.{p}" if k[0] else p for p in packed.get(k[1], [k[1]])]}
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
