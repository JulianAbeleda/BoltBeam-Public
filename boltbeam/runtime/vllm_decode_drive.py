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

--mtp-prompt ID (with --speculative-config, one stream): speculation's speed follows how predictable the text is, so
the context is natural text, not token ids: prompt ID of --prompt-set (collectors/vllm_mtp.PROMPTS, through the model's
chat template) and this engine's own greedy continuation of it, cut to exactly the context. The ids' sha256 is
recorded, so two processes (the untraced timing and the capture) are shown to decode from the same context. Under
speculation the engine keeps its statistics (disable_log_stats off, in every arm alike): vLLM's spec-decode counters are
read around every timed call, and a capture records each scheduler step's drafts and accepted tokens
(vllm_semantics.observe_speculation).

--prefill L is the prefill mode (collectors/prefill.py): one prompt of L token ids drawn from a fixed seed (each call
its own ids), max_tokens 1, prefix caching off, after --warm calls. Untimed by BoltBeam: vLLM's own request timing,
first token less scheduled (what its server reports as time to first token), the median of --reps calls; the wall
time of each call is kept beside it. The line also says the scheduler's max_num_batched_tokens and chunked prefill,
the engine's chunk. With --capture the one measured call is wrapped in cudaProfilerStart/Stop, so a capture holds it
alone, and the line adds its system-wide monotonic start and end (the clock telemetry is read over that window), the weight matrices the model runs (name, N x K, count, the checkpoint tensors packed into
it) and the shared libraries this process loaded: vLLM's own view of what it ran (run with the engine in-process).
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, statistics, time
from pathlib import Path
from vllm_semantics import compilation_facts

import build_jobs

SCRIPT_PROCESSOR = 'boltbeam.runtime.vllm_token_script:TokenScriptProcessor'  # replay's token forcing

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
  ap.add_argument("--corpus", default=None, help="prefill mode: real text (data/gate_corpus.json) instead of random ids")
  ap.add_argument("--prompt-form", default="ids", choices=("ids", "chat"),
                  help="prefill mode with --corpus: the ids as they are, or the text as one user message in the model's chat template")
  ap.add_argument("--semantic-capture", action="store_true", help="compiled graphless NVTX labeling pass; never production timing")
  ap.add_argument("--capture-pass", choices=('label','timing','paired'), help="compiled decode labeling or production graph timing")
  ap.add_argument("--semantic-dispatch", action="store_true", help="NVTX operator evidence preserving compiled execution; refuses unobserved graph nodes")
  ap.add_argument("--profile-step", type=int, help="expose one production engine step to GPU counters")
  ap.add_argument("--profile-end", type=int, help="last engine step in the counter window")
  ap.add_argument("--token-script", help="capture mode: force every pass to the tokens in this file, written by a free "
                  "production-graph generation when absent (vllm_token_script)")
  ap.add_argument("--seed", type=int, default=0, help="prefill mode: the seed the prompt ids are drawn from")
  ap.add_argument("--max-num-batched-tokens", type=int)
  ap.add_argument("--engine-args", type=json.loads, default={}, help="prefill mode: more LLM(...) keyword arguments, one JSON object (workflow/tune_vllm.py)")
  ap.add_argument("--max-num-seqs", type=int, help="the engine's stream slots when they are more than the largest of "
                  "--batches: a capture of one batch on the engine a sweep timed it on")
  ap.add_argument("--dtype", default="auto")
  ap.add_argument("--attention-backend", help="vLLM's attention backend by name (a requested engine setting)")
  ap.add_argument("--gemm-backends", help="Inductor's matrix backends, e.g. TRITON (a requested engine setting): the "
                  "compiled graph's matrix products become Triton kernels instead of cuBLAS calls")
  ap.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("BOLTBEAM_VLLM_GPU_MEMORY", 0.85)))
  ap.add_argument("--tensor-parallel-size", type=int, default=1, help="GPUs the model is split across (collectors/multi_gpu.py)")
  ap.add_argument("--speculative-config", default=os.environ.get("BOLTBEAM_VLLM_SPECULATIVE_CONFIG"),
                  help="explicit MTP configuration; recorded in the measurement")
  ap.add_argument("--mtp-prompt", help="speculative decode: the natural prompt (an id of --prompt-set) the context is built from")
  ap.add_argument("--prompt-set", help="a boltbeam.mtp_prompts.v1 file (collectors/vllm_mtp.PROMPTS)")
  ap.add_argument("--mtp-context", help="the run's natural contexts: read when present, else written (natural_context)")
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
  if a.token_script and not a.capture_pass:
    ap.error('--token-script requires a decode --capture-pass')
  if a.prompt_form == "chat" and not (a.prefill and a.corpus):
    ap.error("--prompt-form chat needs --prefill and --corpus: a chat request is text")
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
  if a.mtp_prompt and not (a.speculative_config and a.prompt_set):
    ap.error("--mtp-prompt needs --speculative-config and --prompt-set")
  if a.speculative_config and a.token_script:
    ap.error("token scripts force one logits row per request; speculative verification scores k+1 rows per request, so "
             "a speculative capture cannot be forced (vllm_token_script)")
  if a.speculative_config and (a.once or a.capture_pass) and not a.mtp_prompt:
    ap.error("a speculative decode capture needs --mtp-prompt: acceptance on token-id prompts is not the engine's")
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
  seqs = a.max_num_seqs or max(max(batches), 1)
  if seqs < max(batches):
    ap.error(f"--max-num-seqs {seqs} is fewer stream slots than the largest batch asked, {max(batches)}")
  from contextlib import nullcontext
  if a.gemm_backends:  # compile_lm_head reaches the engine's worker process as a function, which vLLM sends by pickle
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
  llm = LLM(model=a.model, dtype=a.dtype, max_model_len=max_len, enable_prefix_caching=False,
            **({'worker_cls':'boltbeam.runtime.vllm_decode_worker.DecodeCaptureWorker'} if a.capture_pass=='paired' else {}),
            **({'logits_processors':[SCRIPT_PROCESSOR]} if a.token_script else {}),
            language_model_only=True, speculative_config=a.speculative_config,
            **({"disable_log_stats": False} if a.speculative_config else {}),
            gpu_memory_utilization=a.gpu_memory_utilization, tensor_parallel_size=a.tensor_parallel_size, max_num_seqs=seqs, seed=0,
            **({"attention_backend": a.attention_backend} if a.attention_backend else {}),
            **({"compilation_config": gemm_config(a.gemm_backends)} if a.gemm_backends else {}),
            **({"max_num_batched_tokens": a.max_num_batched_tokens, "enable_chunked_prefill": True} if a.max_num_batched_tokens else {}))
  if a.gemm_backends:
    llm.apply_model(compile_lm_head(a.gemm_backends))
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

  natural = {}
  if a.mtp_prompt:
    if batches != [1]:
      raise SystemExit("--mtp-prompt measures one stream: every stream would need its own natural text")
    natural = {c: natural_context(llm, a.prompt_set, a.mtp_prompt, c, a.mtp_context) for c in contexts}

  def prompts(context:int, batch:int) -> list:
    if natural:
      return [TokensPrompt(prompt_token_ids=natural[context]["ids"])]
    # stream s: ids 1000+s, 1001+s, ... so no two streams share a prefix
    return [TokensPrompt(prompt_token_ids=[(1000 + s * 7 + i) % vocab for i in range(context)]) for s in range(batch)]
  def speculation_at(context:int) -> dict | None:
    """The speculation this point ran under and the natural context it decoded from (its ids left out: the sha)."""
    if not (a.speculative_config and natural):
      return None
    return {"config": a.speculative_config, "prompt": {k: v for k, v in natural[context].items() if k != "ids"}}
  speculation = speculation_at(contexts[0]) if contexts else None
  counters = []  # vLLM's spec-decode counters over each timed call, in call order

  token_ids=[]
  def timed(context:int, batch:int, n:int, scripts:list | None = None) -> float:
    params = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0)
    if scripts is not None:  # one script per stream, its first n tokens (vllm_token_script.TokenScriptProcessor)
      params = [SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0,
                               extra_args={'boltbeam_token_script':list(t[:n])}) for t in scripts]
    if a.speculative_config:
      from vllm_mtp_drive import spec_counters, delta
      before = spec_counters(llm)
    t0 = time.perf_counter()
    out = llm.generate(prompts(context, batch), params, use_tqdm=False)
    dt = time.perf_counter() - t0
    if a.speculative_config:
      counters.append({"tokens": n, **delta(spec_counters(llm), before)})
    token_ids[:]=[list(o.outputs[0].token_ids) for o in out]
    got = [len(ids) for ids in token_ids]
    if got != [n] * batch:
      raise SystemExit(f"asked for {n} tokens per stream, got {got}")
    return dt

  if a.once:
    scripts=None
    if a.token_script:
      # Graphless labelling and graphed timing round differently, and a near tie in a greedy argmax then takes the
      # passes onto different tokens. Every pass is forced to one production-graph generation's tokens, so all of them
      # compute the same steps. The free generation runs twice and must repeat itself; the forcing runs in every pass,
      # so the joined launch sequences stay identical and its cost is in the measured capture.
      path=Path(a.token_script)
      if path.exists():
        scripts=json.loads(path.read_text())['scripts']
      else:
        for _ in range(a.warm):timed(contexts[0], batches[0], 2)
        free=[]
        for _ in range(2):
          timed(contexts[0], batches[0], a.once);free.append([list(t) for t in token_ids])
        where=[[s,next(i for i,(x,y) in enumerate(zip(u,w)) if x!=y)] for s,(u,w) in enumerate(zip(*free)) if u!=w]
        if where and batches[0]!=1:
          raise SystemExit(f'production-graph generation does not repeat itself (stream, first differing token: {where}): '
                           'no token script to force')
        # An engine that does not repeat itself (FlashInfer's fused MoE finalize sums with atomics) forces one stream
        # to the first: its work is the same whichever tokens it takes (dense rows, k of top-k); the split is recorded
        scripts=free[0]
        path.write_text(json.dumps({'scripts':scripts,'context':contexts[0],'batch':batches[0],'tokens':a.once,
                                    'source':'free production-graph generation, '+('the first of two that differ' if where else 'repeated identically'),
                                    **({'engine_nondeterministic':where} if where else {})}))
      if len(scripts)!=batches[0] or any(len(t)!=a.once for t in scripts):
        raise SystemExit('token script does not cover every stream and token')
    for capture_pass in (('timing','label') if a.capture_pass=='paired' else (a.capture_pass,)):
      warm_outputs=[]
      for _ in range(a.warm):
        timed(contexts[0], batches[0], 2)
        warm_outputs.append(list(token_ids))
      from vllm_semantics import (observe_steps, dispatch_mode, without_graph_replay, module_ranges, inductor_ranges,
                                  observe_speculation, draft_scope)
      with observe_steps(llm,a.profile_step,profile_end=a.profile_end,capture_pass=capture_pass) if capture_pass else nullcontext() as steps:
       with observe_speculation(llm) if a.speculative_config else nullcontext() as scheduled:
        with without_graph_replay(llm) if capture_pass=='label' else nullcontext():
          with module_ranges(llm) if capture_pass=='label' else nullcontext():
            with draft_scope(llm) if capture_pass=='label' else nullcontext():
             with dispatch_mode(llm) if capture_pass=='label' else nullcontext():
              with inductor_ranges(llm) if capture_pass=='label' and a.gemm_backends else nullcontext():
                seconds=timed(contexts[0], batches[0], a.once, scripts)
      measured_counters=counters[-1] if a.speculative_config else None
      out_ids=[list(ids) for ids in token_ids];short_seconds=timed(contexts[0], batches[0], 1, scripts) if capture_pass=='timing' else None  # the native arm's order: long, then short; unobserved
      if scripts is not None and out_ids!=scripts:raise SystemExit('forced pass did not emit its token script')
      line={"tokens":a.once,"short_seconds":short_seconds,
            **({"speculation":{**speculation,"steps":list(scheduled),"counters":measured_counters}} if speculation else {}),**({"token_forcing":SCRIPT_PROCESSOR,"token_script_sha256":hashlib.sha256(Path(a.token_script).read_bytes()).hexdigest(),**({"engine_nondeterministic":d} if (d:=json.loads(Path(a.token_script).read_text()).get('engine_nondeterministic')) else {})} if scripts is not None else {}),"warm":a.warm,"samples":1,"call_order":"long_then_short","output_token_ids":out_ids,"warm_output_token_ids":warm_outputs,"seconds":seconds,"matrices":_matrices(llm),"engine":_resolved_formats(llm)}
      if capture_pass:
        import vllm;config=llm.llm_engine.vllm_config
        line.update(line['engine'], capture_pass=capture_pass, runtime_graph_mode='NONE' if capture_pass=='label' else 'production',
                    compilation_config=compilation_facts(config.compilation_config), enforce_eager=config.model_config.enforce_eager,
                    execution_mode=('production CUDA graph timing' if capture_pass=='timing' else 'compiled graphless labeling'),
                    steps=steps, profile_step=a.profile_step,decode_step=a.profile_step,
                    context=contexts[0],batch=batches[0],length=1,seed=config.model_config.seed,vllm=vllm.__version__,
                    max_model_len=max_len,max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
                    triton_cache_dir=os.environ.get('TRITON_CACHE_DIR',str(Path.home()/'.triton/cache')),
                    libraries=_libraries())
      if a.profile_end is not None:
        if a.profile_step<1:raise ValueError('decode counter range must exclude prefill')
        at=speculative_depths(line['speculation']['steps'],contexts[0]) if speculation else {}
        line['counter_steps']=[{**line,'context':at[step][0] if at else contexts[0]+step-1,'decode_step':step,
                                **({'length':at[step][1]} if at else {})}
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
      pairs=[];output_pairs=[];repeat_counters=[]
      for _ in range(a.reps):
        long=timed(context,batch,a.tokens);long_ids=list(token_ids)
        if a.speculative_config:repeat_counters.append(counters[-1])
        short=timed(context,batch,1)
        if a.speculative_config:repeat_counters[-1]={"long":repeat_counters[-1],"short":counters[-1]}
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
                                   "enforce_eager":llm.llm_engine.vllm_config.model_config.enforce_eager,"warm":a.warm,"seed":llm.llm_engine.vllm_config.model_config.seed,"call_order":"long_then_short",
                                   "speculative_config": a.speculative_config,
                                   **({"speculation":{**speculation_at(context),"repeats":repeat_counters}} if natural else {}),
                                   "max_num_batched_tokens": int(llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens),
                                   "max_num_seqs": int(llm.llm_engine.vllm_config.scheduler_config.max_num_seqs),
                                   "chunked_prefill": bool(llm.llm_engine.vllm_config.scheduler_config.enable_chunked_prefill),
                                   "attention_backend": _attention_backend(llm),
                                   "gemm_backends": a.gemm_backends,
                                   "tensor_parallel_size": int(llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size)},
                        "step_ms": step_ms, "tok_s_stream": 1000.0 / step_ms, "tok_s_total": batch * 1000.0 / step_ms}),
            flush=True)
  return 0


def speculative_depths(steps:list, context:int) -> dict:
  """{engine step: (tokens in the cache when it starts, its query tokens)} from observe_speculation's records; the
  arithmetic is collectors/vllm_speculation.ledger's (prompt and emitted tokens, less the step's own first query)."""
  done, out = 0, {}
  for record in steps:
    if not record["requests"]:  # an empty scheduler step (nothing left to run) launches nothing
      continue
    row = next(iter(record["requests"].values()))
    if record["schedule"]:
      out[record["engine_step"]] = (context + done - 1, row["tokens"])
    done += row.get("emitted", 0)
  return out


def natural_context(llm, prompt_set:str, prompt_id:str, context:int, store:str | None = None) -> dict:
  """Exactly `context` token ids of natural text: the prompt's chat-template ids, then the engine's own greedy
  continuation of them, cut to length. With its identity: the prompt set's version and sha256, the prompt id, how many
  ids are the prompt and how many the continuation, and the ids' sha256.

  store: a JSON file every process of one run reads the ids from (the untraced timing, both capture passes, the
  counter runs). The first process to need a context writes it. Generated afresh in each process the continuation is
  not the same text: under speculation the target verifies in windows the acceptance sets, so a near tie in bf16 can
  turn a greedy choice (an ncu counter run continued the prompt differently from its capture, 2026-11-08)."""
  from vllm import SamplingParams
  from vllm.inputs import TokensPrompt
  raw = Path(prompt_set).read_bytes()
  data = json.loads(raw)
  found = [p for p in data.get("prompts") or [] if p.get("id") == prompt_id]
  if data.get("schema") != "boltbeam.mtp_prompts.v1" or len(found) != 1:
    raise SystemExit(f"{prompt_set} has no one prompt {prompt_id!r} (boltbeam.mtp_prompts.v1)")
  identity = {"prompt_set_version": data.get("version"), "prompt_set_sha256": hashlib.sha256(raw).hexdigest(),
              "prompt_id": prompt_id}
  saved = json.loads(Path(store).read_text()) if store and Path(store).exists() else {}
  if (held := saved.get(str(context))) is not None:
    if any(held.get(k) != v for k, v in identity.items()) or len(held.get("ids") or []) != context:
      raise SystemExit(f"{store} holds another context for {context} tokens than prompt {prompt_id} of this prompt set")
    return {**held, "context_sha256": hashlib.sha256(json.dumps(held["ids"]).encode()).hexdigest(),
            "source": held["source"] + f"; read from {Path(store).name}"}
  got = llm.get_tokenizer().apply_chat_template([{"role": "user", "content": found[0]["text"]}], tokenize=True,
                                                add_generation_prompt=True)
  ids = list(got["input_ids"] if hasattr(got, "keys") else got)
  if len(ids) > context:
    raise SystemExit(f"prompt {prompt_id} is {len(ids)} tokens, longer than the context {context}")
  more = context - len(ids)
  if more:
    params = SamplingParams(max_tokens=more, min_tokens=more, ignore_eos=True, temperature=0.0)
    ids += list(llm.generate([TokensPrompt(prompt_token_ids=ids)], params, use_tqdm=False)[0].outputs[0].token_ids)
  if len(ids) != context:
    raise SystemExit(f"natural context is {len(ids)} tokens, not {context}")
  out = {"ids": ids, **identity, "prompt_tokens": context - more, "continuation_tokens": more,
         "source": "the prompt through the model's chat template, then the engine's own greedy continuation"}
  if store:
    Path(store).write_text(json.dumps({**saved, str(context): out}))
  return {**out, "context_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}


def prefill(a) -> int:
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt
  llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.model_context_limit or a.prefill + 1, enable_prefix_caching=False,
            **({'compilation_config': {**({'cudagraph_mode':'NONE'} if a.semantic_capture else {}),
                                       **(gemm_config(a.gemm_backends) if a.gemm_backends else {})}}
               if a.semantic_capture or a.gemm_backends else {}),
            language_model_only=True, speculative_config=a.speculative_config,
            gpu_memory_utilization=a.gpu_memory_utilization, tensor_parallel_size=a.tensor_parallel_size, max_num_seqs=1, seed=0, disable_log_stats=False,
            **({"attention_backend": a.attention_backend} if a.attention_backend else {}),
            **({"max_num_batched_tokens": a.max_num_batched_tokens, "enable_chunked_prefill": True} if a.max_num_batched_tokens else {}),
            **a.engine_args)
  vocab = llm.get_tokenizer().vocab_size
  rng = random.Random(a.seed)
  params = SamplingParams(max_tokens=1, temperature=0.0)
  text_ids = corpus_ids(llm.get_tokenizer(), a.corpus) if a.corpus else None
  chat = chat_ids(llm.get_tokenizer(), window(text_ids, a.prefill), a.prefill) if a.prompt_form == "chat" else None

  def call() -> tuple[float, float | None]:
    ids = chat or (window(text_ids, a.prefill) if text_ids else [rng.randrange(vocab) for _ in range(a.prefill)])
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
  line = {"length": a.prefill, "prompt_kind": "real_text" if a.corpus else "random_tokens", "prompt_form": a.prompt_form,
          "prompt_tokens": len(chat) if chat else a.prefill,
          **({"corpus": {"file": Path(a.corpus).name, "tokens": len(text_ids)}} if a.corpus else {}), "max_model_len": llm.llm_engine.vllm_config.model_config.max_model_len, **_resolved_formats(llm), "seed": a.seed, "max_num_batched_tokens": int(sc.max_num_batched_tokens),
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


def corpus_ids(tokenizer, path:str) -> list[int]:
  """The corpus's documents (data/gate_corpus.json, the llama path's text) as this engine's token ids, in order."""
  text = "\n\n".join(d["text"] for d in json.loads(Path(path).read_text())["documents"])
  ids = tokenizer.encode(text, add_special_tokens=False)
  if not ids:
    raise SystemExit(f"the corpus {path} tokenized to nothing")
  return ids


def chat_ids(tokenizer, ids:list[int], n:int) -> list[int]:
  """The chat form of a prompt: the text of the window's first k ids as one user message, through the model's chat
  template with the generation prompt, as token ids; k chosen so the result is n tokens (a few tries: a detokenized cut
  can tokenize back a token longer or shorter). It is what LLM.chat renders and tokenizes; vLLM has no batch break at the
  user message, so the form changes only the tokens."""
  def render(k:int) -> list[int]:
    got = tokenizer.apply_chat_template([{"role": "user", "content": tokenizer.decode(ids[:k])}], add_generation_prompt=True, tokenize=True)
    return list(got["input_ids"] if hasattr(got, "keys") else got)  # newer transformers: a BatchEncoding
  base, k, best = len(render(0)), n, None
  for _ in range(12):
    got = render(k)
    if best is None or abs(len(got) - n) < abs(len(best) - n):
      best = got
    if len(got) == n:
      break
    step = round((n - len(got)) * k / max(len(got) - base, 1))  # the text's tokens per id so far
    k = max(1, k + (step or (1 if len(got) < n else -1)))
  if len(best) != n:  # a prompt of another length is another measurement: refused, as llama's chat form refuses it
    raise SystemExit(f"could not fit the corpus text to a {n}-token chat prompt (closest {len(best)})")
  return best


def window(ids:list[int], n:int) -> list[int]:
  """Every call's prompt: the corpus's first n ids (repeated past its end), the window the llama.cpp path sends
  (llama_server_decode_drive.corpus_ids). Each call is the same text, as there: prefix caching is off (LLM(...,
  enable_prefix_caching=False)), as llama-server's prompt cache is, so a repeat is read in full."""
  return [ids[i % len(ids)] for i in range(n)]


def gemm_config(backends:str) -> dict:
  """vLLM's compilation settings that hand Inductor's matrix products to `backends` (max_autotune on: Inductor only
  generates its own GEMM templates while autotuning). Products outside the compiled graph keep their own kernels."""
  return {"inductor_compile_config": {"max_autotune": True, "max_autotune_gemm_backends": backends.upper()}}


def compile_lm_head(backends:str):
  """For llm.apply_model: vLLM projects the logits outside its compiled graph, so gemm_config does not reach lm_head,
  and on an H100 its bf16 product stays a closed cuBLAS nvjet kernel. This sends that product through Inductor with the
  same matrix backends, so it too becomes a template whose code is read (runtime.vllm_semantics.inductor_ranges labels
  it by the weight it reads). Returns the module it changed."""
  def patch(model):
    import torch, torch.nn.functional as F
    heads = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] == "lm_head" and hasattr(m, "quant_method")]
    if len(heads) != 1:
      raise ValueError(f"--gemm-backends: expected one lm_head module, found {[n for n, _ in heads]}")
    name, head = heads[0]
    if type(head.quant_method).__name__ not in ("UnquantizedEmbeddingMethod", "UnquantizedLinearMethod"):
      raise ValueError(f"--gemm-backends: lm_head is {type(head.quant_method).__name__}, not a plain weight")
    linear = torch.compile(lambda x, w, b: F.linear(x, w, b), dynamic=True,
                           options={"max_autotune": True, "max_autotune_gemm_backends": backends.upper()})
    from torch.utils._python_dispatch import _disable_current_modes
    def apply(layer, x, bias=None):
      # The labeling pass runs under a dispatch mode, and compiled code falls back to eager inside one, which ran cuBLAS
      # there and the template in the timing pass: the two launch sequences diverged at lm_head (2026-11-08). Outside
      # the mode both passes launch the same template; inductor_ranges labels it at launch, not by dispatch.
      with _disable_current_modes():
        return linear(x, layer.weight, bias)
    head.quant_method.apply = apply
    return name
  return patch


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
