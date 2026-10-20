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
from vllm_semantics import compilation_facts

import build_jobs

# Direct driver invocation gets the same policy; multiprocessing workers start
# fresh Python interpreters and import sitecustomize through PYTHONPATH.
_runtime = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = os.pathsep.join([_runtime, *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])])
os.environ.setdefault("BOLTBEAM_BUILD_LIMIT", "1")
os.environ.setdefault("BOLTBEAM_REQUIRE_PREBUILT", "1")
build_jobs.install_ninja_limit()


def _resolved_formats(llm) -> dict:
  from vllm_semantics import config_facts
  return config_facts(llm.llm_engine.vllm_config)


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
  ap.add_argument("--semantic-capture", action="store_true", help="compiled graphless NVTX labeling pass; never production timing")
  ap.add_argument("--capture-pass", choices=('label','timing','paired'), help="compiled decode labeling or production graph timing")
  ap.add_argument("--semantic-dispatch", action="store_true", help="NVTX operator evidence preserving compiled execution; refuses unobserved graph nodes")
  ap.add_argument("--profile-step", type=int, help="expose one production engine step to GPU counters")
  ap.add_argument("--profile-end", type=int, help="last engine step in the counter window")
  ap.add_argument("--seed", type=int, default=0, help="prefill mode: the seed the prompt ids are drawn from")
  ap.add_argument("--max-num-batched-tokens", type=int)
  ap.add_argument("--dtype", default="auto")
  ap.add_argument("--attention-backend", help="vLLM's attention backend by name (a requested engine setting)")
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
  ap.add_argument("--tensor-parallel-size", type=int, default=1, help="GPUs the model is split across (collectors/multi_gpu.py)")
  ap.add_argument("--speculative-config", default=os.environ.get("BOLTBEAM_VLLM_SPECULATIVE_CONFIG"),
                  help="explicit MTP configuration; recorded in the measurement")
  ap.add_argument("--quality-suite", help="JSON prompts and expected JSON answers for a bounded smoke check")
  ap.add_argument("--quality-thinking", action="store_true", help="enable thinking for quality prompts")
  ap.add_argument("--quality-max-tokens", type=int, help="answer budget including thinking (default: 4096 on, 128 off)")
  ap.add_argument("--quality-only", action="store_true", help="run quality checks without decode benchmarks")
  a = ap.parse_args()
  if a.capture_pass and (not a.once or a.prefill or a.semantic_capture or a.semantic_dispatch):
    ap.error('--capture-pass requires decode --once and excludes other semantic modes')
  if a.profile_step is not None and (a.capture_pass not in ('label','timing') or a.profile_step < 0):
    ap.error('--profile-step requires a single capture pass and a nonnegative step')
  if a.profile_end is not None and (a.profile_step is None or a.profile_end<a.profile_step):
    ap.error('--profile-end requires an ordered profiler range')
  if a.semantic_capture and not (a.capture and a.prefill):
    ap.error('--semantic-capture requires --prefill and --capture')
  if a.semantic_dispatch and (not (a.capture and a.prefill) or a.semantic_capture):
    ap.error('--semantic-dispatch requires --prefill --capture and excludes --semantic-capture')
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
  from contextlib import nullcontext
  llm = LLM(model=a.model, dtype=a.dtype, max_model_len=max_len, enable_prefix_caching=False,
            **({'worker_cls':'boltbeam.runtime.vllm_decode_worker.DecodeCaptureWorker'} if a.capture_pass=='paired' else {}),
            language_model_only=True, speculative_config=a.speculative_config,
            gpu_memory_utilization=a.gpu_memory_utilization, tensor_parallel_size=a.tensor_parallel_size, max_num_seqs=max(max(batches), 1), seed=0,
            **({"attention_backend": a.attention_backend} if a.attention_backend else {}),
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

  token_ids=[]
  def timed(context:int, batch:int, n:int) -> float:
    params = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0)
    t0 = time.perf_counter()
    out = llm.generate(prompts(context, batch), params, use_tqdm=False)
    dt = time.perf_counter() - t0
    token_ids[:]=[list(o.outputs[0].token_ids) for o in out]
    got = [len(ids) for ids in token_ids]
    if got != [n] * batch:
      raise SystemExit(f"asked for {n} tokens per stream, got {got}")
    return dt

  if a.once:
    for capture_pass in (('timing','label') if a.capture_pass=='paired' else (a.capture_pass,)):
      warm_outputs=[]
      for _ in range(a.warm):
        timed(contexts[0], batches[0], 2)
        warm_outputs.append(list(token_ids))
      from vllm_semantics import observe_steps, dispatch_mode, without_graph_replay, module_ranges
      with observe_steps(llm,a.profile_step,profile_end=a.profile_end,capture_pass=capture_pass) if capture_pass else nullcontext() as steps:
        with without_graph_replay(llm) if capture_pass=='label' else nullcontext():
          with module_ranges(llm) if capture_pass=='label' else nullcontext():
            with dispatch_mode(llm) if capture_pass=='label' else nullcontext():
              seconds=timed(contexts[0], batches[0], a.once)
      line={"tokens":a.once,"output_token_ids":token_ids,"warm_output_token_ids":warm_outputs,"seconds":seconds,"matrices":_matrices(llm),"engine":_resolved_formats(llm)}
      if capture_pass:
        config=llm.llm_engine.vllm_config
        line.update(capture_pass=capture_pass,steps=steps,
                    runtime_graph_mode='NONE' if capture_pass=='label' else 'production',
                    compilation_config=compilation_facts(config.compilation_config),
                    enforce_eager=config.model_config.enforce_eager)
      if capture_pass:
        import vllm
        line.update(**line['engine'], execution_mode=('production CUDA graph timing' if capture_pass=='timing' else 'compiled graphless labeling'),
                    steps=steps, profile_step=a.profile_step,decode_step=a.profile_step,
                    context=contexts[0],batch=batches[0],length=1,seed=0,vllm=vllm.__version__,
                    max_model_len=max_len,max_num_batched_tokens=llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens,
                    triton_cache_dir=os.environ.get('TRITON_CACHE_DIR',str(Path.home()/'.triton/cache')),
                    libraries=_libraries())
      if a.profile_end is not None:
        if a.profile_step<1:raise ValueError('decode counter range must exclude prefill')
        line['counter_steps']=[{**line,'context':contexts[0]+step-1,'decode_step':step}
                               for step in range(a.profile_step,a.profile_end+1)]
      print(json.dumps(line),flush=True)
    if a.capture_pass=='paired':
      import torch
      torch.cuda.synchronize();torch.cuda.profiler.stop()
    return 0
  for context in contexts:
    for batch in batches:
      for _ in range(a.warm):
        timed(context, batch, 2)
      pairs=[];output_pairs=[]
      for _ in range(a.reps):
        long=timed(context,batch,a.tokens);long_ids=list(token_ids)
        short=timed(context,batch,1)
        pairs.append((long,short));output_pairs.append([long_ids,list(token_ids)])
      samples=[(long-short)*1000/(a.tokens-1) for long,short in pairs]
      if not samples or min(samples)<=0:raise ValueError('decode repeats have no positive time difference')
      long_s,short_s=(statistics.median(p[i] for p in pairs) for i in (0,1))
      step_ms = statistics.median(samples)
      print(json.dumps({"context": context, "batch": batch, "tokens": a.tokens, "long_s": long_s, "short_s": short_s,
                        "repeat_pairs_s":pairs,"repeat_output_token_ids":output_pairs,
                        "step_samples_ms":samples,"repeat_band_ms":[min(samples),max(samples)],
                        "engine": {**_resolved_formats(llm), "build": __import__("vllm").__version__, "max_model_len": max_len,
                                   "compilation_config":compilation_facts(llm.llm_engine.vllm_config.compilation_config),
                                   "enforce_eager":llm.llm_engine.vllm_config.model_config.enforce_eager,
                                   "speculative_config": a.speculative_config,
                                   "max_num_batched_tokens": int(llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens),
                                   "chunked_prefill": bool(llm.llm_engine.vllm_config.scheduler_config.enable_chunked_prefill),
                                   "attention_backend": _attention_backend(llm),
                                   "tensor_parallel_size": int(llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size)},
                        "step_ms": step_ms, "tok_s_stream": 1000.0 / step_ms, "tok_s_total": batch * 1000.0 / step_ms}),
            flush=True)
  return 0


def prefill(a) -> int:
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt
  llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.model_context_limit or a.prefill + 1, enable_prefix_caching=False,
            **({'compilation_config': {'cudagraph_mode':'NONE'}} if a.semantic_capture else {}),
            language_model_only=True, speculative_config=a.speculative_config,
            gpu_memory_utilization=a.gpu_memory_utilization, tensor_parallel_size=a.tensor_parallel_size, max_num_seqs=1, seed=0, disable_log_stats=False,
            **({"attention_backend": a.attention_backend} if a.attention_backend else {}),
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
  semantic = None
  if a.semantic_capture or a.semantic_dispatch:
    from vllm_semantics import install
    semantic = install(llm)
  if a.semantic_dispatch:
    from vllm_semantics import install_routing
    install_routing(llm)
  sc = llm.llm_engine.vllm_config.scheduler_config
  line = {"length": a.prefill, "max_model_len": llm.llm_engine.vllm_config.model_config.max_model_len, **_resolved_formats(llm), "seed": a.seed, "max_num_batched_tokens": int(sc.max_num_batched_tokens),
          "speculative_config": a.speculative_config,
          "chunked_prefill": bool(sc.enable_chunked_prefill), "vllm": __import__("vllm").__version__}
  if a.capture:
    import torch
    from contextlib import nullcontext
    mode = nullcontext()
    if a.semantic_dispatch:
      from vllm_semantics import dispatch_mode
      mode = dispatch_mode(llm)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    t0 = time.monotonic()
    with mode:
      wall, engine = call()
    torch.cuda.synchronize()
    t1 = time.monotonic()
    torch.cuda.profiler.stop()
    line.update(wall_s=wall, engine_s=engine, window_monotonic=[t0, t1], matrices=_matrices(llm), libraries=_libraries(),
                triton_cache_dir=os.environ.get('TRITON_CACHE_DIR', str(Path.home()/'.triton/cache')),
                jit_caches=[d for d in (os.environ.get('TRITON_CACHE_DIR', str(Path.home()/'.triton/cache')),
                                       os.environ.get('TORCHINDUCTOR_CACHE_DIR')) if d])
    if semantic is not None:
      line.update(execution_mode='compiled graphless NVTX labeling', semantic_modules=semantic)
    if a.semantic_dispatch:
      from vllm_semantics import read_routing
      line['routing'] = read_routing(llm)
      line.update(execution_mode='compiled NVTX dispatch capture')
  else:
    got = [call() for _ in range(a.reps)]
    line.update(wall_samples_s=[w for w, _ in got], engine_samples_s=[e for _, e in got])
  print(json.dumps(line), flush=True)
  return 0


def _attention_backend(llm) -> str | None:
  """The attention backend the engine resolved, read from its own config (None when this vLLM does not say)."""
  backend = getattr(getattr(llm.llm_engine.vllm_config, "attention_config", None), "backend", None)
  return getattr(backend, "name", None) or (str(backend) if backend is not None else None)


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
      method = getattr(mod, "quant_method", None)
      packed_weight = w is not None and str(getattr(w, "dtype", "")) in ("torch.uint8", "torch.int32")
      if hasattr(mod, "output_size_per_partition") and method is not None:
        n, k = int(mod.output_size_per_partition), int(mod.input_size_per_partition)
        dtype = (type(method).__name__ + ":" + type(getattr(mod, "scheme", None) or method).__name__) if w is None or packed_weight else str(w.dtype)
      elif w is not None and hasattr(w, "dim"):
        if w.dim() != 2:
          continue
        n, k, dtype = int(w.shape[0]), int(w.shape[1]), str(w.dtype)
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
  from vllm_semantics import loaded_libraries
  return loaded_libraries()


if __name__ == "__main__":
  raise SystemExit(main())
