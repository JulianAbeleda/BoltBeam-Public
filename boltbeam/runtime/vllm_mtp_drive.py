"""vLLM decode with multi-token prediction (MTP) at one draft count k, on natural prompts; run with vLLM's python.

One process per k (collectors/vllm_mtp.py runs them in turn): each k is its own engine, and the process ends with it,
so no engine worker holds the GPU after the measurement. k=0 is the same engine with no speculation: the baseline
every k is read against, on the same prompts.

Each prompt, one user turn through the model's chat template, is generated greedily for --tokens tokens, ignoring
EOS, at two depths:
  short  each prompt of data/mtp_prompts.json;
  long   each document of --long-corpus (data/mtp_long_corpus.json, real text no model wrote) cut to --long-tokens
         tokens by this tokenizer at a paragraph break, then its task (runtime/mtp_shared.long_prompts): every k reads
         the same ids, made the same way.
Decode speed is (tokens - 1) / (last token - first token), the engine's own timestamps, so prefill is not in it.
Acceptance is vLLM's spec-decode counters read before and after each prompt: drafts, draft tokens, accepted tokens and
accepted tokens per draft position; span_ns, the generate call's start and end on the host's CLOCK_MONOTONIC (the clock
nsys keeps its session start on: collectors/time_budget.request_decode_budget cuts a trace to it), and first_token_ns, the
engine core's first-token stamp (v1/metrics/stats.py: time.monotonic, the same clock) in ns. Each timed prompt waits for
the quiet-CPU gate (runtime/mtp_shared.quiet). One JSON line per prompt and depth, with the token diversity of its context
and its reply (mtp_shared.diversity), then one line naming the engine with the corpus's own diversity on this tokenizer
(real_text), which collectors/mtp.guard holds them to."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run by path or imported: the MTP drivers' shared module
from mtp_shared import delta, diversity, long_prompts, quiet, reference, sha  # noqa: E402

COUNTERS = {"vllm:spec_decode_num_drafts": "drafts", "vllm:spec_decode_num_draft_tokens": "draft_tokens",
            "vllm:spec_decode_num_accepted_tokens": "accepted_tokens"}
PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos"


def spec_counters(llm) -> dict:
  """vLLM's cumulative spec-decode counters; zeros with no speculation (the counters exist only when it is on)."""
  out = {name: 0 for name in COUNTERS.values()}
  out["accepted_per_position"] = []
  for m in llm.get_metrics():
    if m.name in COUNTERS:
      out[COUNTERS[m.name]] = int(m.value)
    elif m.name == PER_POSITION:
      out["accepted_per_position"] = [int(v) for v in m.values]
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", required=True)
  ap.add_argument("--k", type=int, required=True, help="draft tokens per step; 0 is no speculation")
  ap.add_argument("--prompts", required=True, help="a boltbeam.mtp_prompts.v1 file")
  ap.add_argument("--tokens", type=int, required=True, help="tokens generated per prompt and depth")
  ap.add_argument("--long-tokens", type=int, default=0, help="the long depth's document tokens; 0: short only")
  ap.add_argument("--long-corpus", required=True, help="a boltbeam.mtp_long_corpus.v1 file (data/mtp_long_corpus.json)")
  ap.add_argument("--max-model-len", type=int, required=True)
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
  ap.add_argument("--quiet-s", type=float, default=600.0, help="how long to wait for no build or test runner before a timed prompt")
  a = ap.parse_args()
  if a.k < 0 or a.tokens < 2 or a.long_tokens < 0:
    ap.error("k must be 0 or more, tokens 2 or more, long tokens 0 or more")
  from vllm import LLM, SamplingParams, __version__
  from vllm.inputs import TokensPrompt
  prompts = json.loads(Path(a.prompts).read_text())["prompts"]
  llm = LLM(model=a.model, max_model_len=a.max_model_len, enable_prefix_caching=False, seed=0,
            gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=1, disable_log_stats=False,
            **({"speculative_config": {"method": "mtp", "num_speculative_tokens": a.k}} if a.k else {}))
  tok = llm.get_tokenizer()
  greedy = lambda n: SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=n)  # noqa: E731
  llm.generate([TokensPrompt(prompt_token_ids=list(range(1000, 1064)))], greedy(32), use_tqdm=False)  # warm
  def chat(text:str) -> list[int]:
    got = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    return list(got["input_ids"] if isinstance(got, dict) or hasattr(got, "input_ids") else got)
  ids = {p["id"]: chat(p["text"]) for p in prompts}
  encode = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
  decode = lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False)  # noqa: E731  the text itself, for the cut
  messages, docs = long_prompts(a.long_corpus, a.long_tokens, encode, decode)
  long_ids = {d: chat(m) for d, m in messages.items()}
  real = reference(docs, a.tokens)
  for depth, source in (("short", ids), ("long", long_ids)):
    for pid, prompt in source.items():
      cpu = quiet(a.quiet_s)
      before = spec_counters(llm)
      m0, t0 = time.monotonic_ns(), time.perf_counter()
      out = llm.generate([TokensPrompt(prompt_token_ids=prompt)], greedy(a.tokens), use_tqdm=False)[0]
      wall, m1 = time.perf_counter() - t0, time.monotonic_ns()
      m, n = out.metrics, len(out.outputs[0].token_ids)
      decode_s = (m.last_token_ts - m.first_token_ts) if m and m.first_token_ts and m.last_token_ts else None
      if n != a.tokens or not decode_s or decode_s <= 0:
        raise SystemExit(f"{pid} {depth}: {n} of {a.tokens} tokens, decode window {decode_s}: not a measurement")
      reply = list(out.outputs[0].token_ids)
      print(json.dumps({"k": a.k, "prompt": pid, "depth": depth, "context": len(prompt), "tokens": n,
                        "decode_tok_s": (n - 1) / decode_s, "wall_s": wall, "spec": delta(spec_counters(llm), before), "cpu": cpu,
                        "prompt_sha": sha(prompt), "token_ids_sha": sha(reply), "span_ns": [m0, m1], "first_token_ns": round(m.first_token_ts * 1e9),
                        "text": {"context": diversity(prompt, a.tokens), "reply": diversity(prompt[-a.tokens:] + reply, a.tokens)}, "reply_ids": reply}), flush=True)
  print(json.dumps({"engine": {"vllm": __version__, "k": a.k, "max_model_len": a.max_model_len, "real_text": real,
                               "speculative_config": {"method": "mtp", "num_speculative_tokens": a.k} if a.k else None}}), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
