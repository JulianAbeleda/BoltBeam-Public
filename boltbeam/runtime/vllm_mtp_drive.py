"""vLLM decode with multi-token prediction (MTP) at one draft count k, on natural prompts; run with vLLM's python.

One process per k (collectors/vllm_mtp.py runs them in turn): each k is its own engine, and the process ends with it,
so no engine worker holds the GPU after the measurement. k=0 is the same engine with no speculation: the baseline
every k is read against, on the same prompts.

Each prompt (data/mtp_prompts.json, one user turn through the model's chat template) is generated greedily for
--tokens tokens, ignoring EOS, at two depths:
  short  the prompt alone;
  long   the prompt and the model's own greedy continuation of it, --long-tokens long, so the context is natural text
         at depth. The k=0 process writes the continuations (--continuations); every k>0 process reads the same ids.
Decode speed is (tokens - 1) / (last token - first token), the engine's own timestamps, so prefill is not in it.
Acceptance is vLLM's spec-decode counters read before and after each prompt: drafts, draft tokens, accepted tokens and
accepted tokens per draft position. One JSON line per prompt and depth, then one line naming the engine."""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

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


def delta(after:dict, before:dict) -> dict:
  per = [a - (before["accepted_per_position"][i] if i < len(before["accepted_per_position"]) else 0)
         for i, a in enumerate(after["accepted_per_position"])]
  return {**{k: after[k] - before[k] for k in COUNTERS.values()}, "accepted_per_position": per}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", required=True)
  ap.add_argument("--k", type=int, required=True, help="draft tokens per step; 0 is no speculation")
  ap.add_argument("--prompts", required=True, help="a boltbeam.mtp_prompts.v1 file")
  ap.add_argument("--tokens", type=int, required=True, help="tokens generated per prompt and depth")
  ap.add_argument("--long-tokens", type=int, default=0, help="the long depth's continuation length; 0: short only")
  ap.add_argument("--continuations", help="k=0 writes the long depth's continuation ids here; k>0 reads them")
  ap.add_argument("--max-model-len", type=int, required=True)
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
  a = ap.parse_args()
  if a.k < 0 or a.tokens < 2 or a.long_tokens < 0:
    ap.error("k must be 0 or more, tokens 2 or more, long tokens 0 or more")
  if a.long_tokens and not a.continuations:
    ap.error("--long-tokens needs --continuations: every k must read the same long context")
  from vllm import LLM, SamplingParams, __version__
  from vllm.inputs import TokensPrompt
  prompts = json.loads(Path(a.prompts).read_text())["prompts"]
  llm = LLM(model=a.model, max_model_len=a.max_model_len, enable_prefix_caching=False, seed=0,
            gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=1, disable_log_stats=False,
            **({"speculative_config": {"method": "mtp", "num_speculative_tokens": a.k}} if a.k else {}))
  tok = llm.get_tokenizer()
  greedy = lambda n: SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=n)  # noqa: E731
  llm.generate([TokensPrompt(prompt_token_ids=list(range(1000, 1064)))], greedy(32), use_tqdm=False)  # warm
  ids = {}
  for p in prompts:
    got = tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=True, add_generation_prompt=True)
    ids[p["id"]] = list(got["input_ids"] if isinstance(got, dict) or hasattr(got, "input_ids") else got)
  long_ids = {}
  if a.long_tokens:
    path = Path(a.continuations)
    if a.k == 0:
      cont = {pid: list(llm.generate([TokensPrompt(prompt_token_ids=x)], greedy(a.long_tokens), use_tqdm=False)[0]
                        .outputs[0].token_ids) for pid, x in ids.items()}
      path.write_text(json.dumps({"long_tokens": a.long_tokens, "continuations": cont}))
    saved = json.loads(path.read_text())
    if saved["long_tokens"] != a.long_tokens or set(saved["continuations"]) != set(ids):
      raise SystemExit(f"{path} holds other continuations than this run asks for")
    long_ids = {pid: ids[pid] + saved["continuations"][pid] for pid in ids}
  for depth, source in (("short", ids), ("long", long_ids)):
    for pid, prompt in source.items():
      before = spec_counters(llm)
      t0 = time.perf_counter()
      out = llm.generate([TokensPrompt(prompt_token_ids=prompt)], greedy(a.tokens), use_tqdm=False)[0]
      wall = time.perf_counter() - t0
      m, n = out.metrics, len(out.outputs[0].token_ids)
      decode_s = (m.last_token_ts - m.first_token_ts) if m and m.first_token_ts and m.last_token_ts else None
      if n != a.tokens or not decode_s or decode_s <= 0:
        raise SystemExit(f"{pid} {depth}: {n} of {a.tokens} tokens, decode window {decode_s}: not a measurement")
      print(json.dumps({"k": a.k, "prompt": pid, "depth": depth, "context": len(prompt), "tokens": n,
                        "decode_tok_s": (n - 1) / decode_s, "wall_s": wall, "spec": delta(spec_counters(llm), before),
                        "token_ids_sha": hashlib.sha256(json.dumps(list(out.outputs[0].token_ids)).encode()).hexdigest()}),
            flush=True)
  print(json.dumps({"engine": {"vllm": __version__, "k": a.k, "max_model_len": a.max_model_len,
                               "speculative_config": {"method": "mtp", "num_speculative_tokens": a.k} if a.k else None}}),
        flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
