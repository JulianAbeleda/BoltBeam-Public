#!/usr/bin/env python3
"""Decode time per step in TensorRT-LLM, at a fixed context and batch, through its LLM API.

TensorRT-LLM's python runs this file by path. The measuring rule is vllm_decode_drive.py's: B streams with their
own prompts of exactly C token ids, the same batch generated with N tokens and with 1 (min_tokens = max_tokens,
ignore_eos, greedy), step ms = (time of N - time of 1) / (N - 1).

--probe prints {"reason": null} when TensorRT-LLM imports and its torch runs on GPU 0, else the reason.
--once N is the capture mode, as in the vLLM driver.
"""
from __future__ import annotations

import argparse, json, statistics, time


def probe() -> str | None:
  try:
    import torch
    import tensorrt_llm  # noqa: F401
  except Exception as exc:  # the reason is the fact the reader needs
    text = str(exc).strip()
    if "libmpi" in text or "MPI" in text:
      return ("TensorRT-LLM needs an MPI library: install libopenmpi (sudo apt install libopenmpi-dev) or "
              "pip install openmpi into its venv")
    return f"tensorrt_llm does not import: {text.splitlines()[-1][:300] if text else type(exc).__name__}"
  if not torch.cuda.is_available():
    return "TensorRT-LLM's torch sees no CUDA GPU"
  major, minor = torch.cuda.get_device_capability(0)
  sm = f"sm_{major}{minor}"
  if sm not in torch.cuda.get_arch_list():
    return f"TensorRT-LLM's torch is not built for {sm} ({torch.cuda.get_device_name(0)})"
  return None


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--probe", action="store_true")
  ap.add_argument("--model")
  ap.add_argument("--contexts", default="128")
  ap.add_argument("--batches", default="1")
  ap.add_argument("--tokens", type=int, default=64)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--warm", type=int, default=1)
  ap.add_argument("--once", type=int, default=None)
  ap.add_argument("--backend", default=None, choices=["pytorch", "tensorrt"],
                  help="pytorch (TensorRT-LLM's default) or tensorrt (build a TensorRT engine)")
  a = ap.parse_args()
  if a.probe:
    print(json.dumps({"reason": probe()}), flush=True)
    return 0
  from tensorrt_llm import SamplingParams
  if a.backend == "tensorrt":  # a TensorRT engine is built for this GPU, then run
    from tensorrt_llm._tensorrt_engine import LLM
  else:  # TensorRT-LLM's default LLM: its PyTorch backend
    from tensorrt_llm import LLM
  contexts = [int(x) for x in a.contexts.split(",")]
  batches = [int(x) for x in a.batches.split(",")]
  max_len = max(contexts) + max(a.tokens, a.once or 0) + 8
  llm = LLM(model=a.model, max_seq_len=max_len, max_batch_size=max(batches), max_num_tokens=max(8192, max_len))
  backend = a.backend or "pytorch"
  vocab = 150000

  def prompts(context:int, batch:int) -> list[list[int]]:
    return [[(1000 + s * 7 + i) % vocab for i in range(context)] for s in range(batch)]

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
                        "step_ms": step_ms, "tok_s_stream": 1000.0 / step_ms, "tok_s_total": batch * 1000.0 / step_ms,
                        "backend": backend}), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
