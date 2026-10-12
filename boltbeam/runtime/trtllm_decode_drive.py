#!/usr/bin/env python3
"""Decode time per step in TensorRT-LLM, at a fixed context and batch, through its LLM API.

TensorRT-LLM's python runs this file by path. The measuring rule is vllm_decode_drive.py's: B streams with their
own prompts of exactly C token ids, the same batch generated with N tokens and with 1 (min_tokens = max_tokens,
ignore_eos, greedy), step ms = (time of N - time of 1) / (N - 1).

--probe prints {"reason": null} when TensorRT-LLM imports and its torch runs on GPU 0, else the reason.
--once N is the capture mode, as in the vLLM driver.
"""
from __future__ import annotations

import importlib.metadata

import argparse, json, os, statistics, time


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
  ap.add_argument("--model-context-limit", type=int)
  ap.add_argument("--max-num-tokens", type=int)
  ap.add_argument("--profile-window", action="store_true")
  ap.add_argument("--worker-single-process", choices=("0", "1"), default=os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS", "0"))
  ap.add_argument("--contexts")
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
  if not a.contexts:
    ap.error("--contexts is required for a measurement")
  os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = a.worker_single_process
  from tensorrt_llm import SamplingParams
  if a.backend == "tensorrt":  # a TensorRT engine is built for this GPU, then run
    from tensorrt_llm._tensorrt_engine import LLM
  else:  # TensorRT-LLM's default LLM: its PyTorch backend
    from tensorrt_llm import LLM
  contexts = [int(x) for x in a.contexts.split(",")]
  batches = [int(x) for x in a.batches.split(",")]
  max_len = a.model_context_limit or max(contexts) + max(a.tokens, a.once or 0) + 8
  backend = a.backend or "pytorch"
  # The scheduler needs capacity for this request's prompts, not an unrelated 8192-token build profile.
  max_tokens = a.max_num_tokens or max(1, max(contexts)) * max(batches)
  llm = LLM(model=a.model, max_seq_len=max_len, max_batch_size=max(batches), max_num_tokens=max_tokens,
            **({"enable_build_cache": True} if backend == "tensorrt" else {}))
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
    import torch
    if a.profile_window: torch.cuda.cudart().cudaProfilerStart()
    try:
      seconds = timed(contexts[0], batches[0], a.once)
    finally:
      if a.profile_window: torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps({"tokens": a.once, "seconds": seconds, "matrices": _matrices(llm)}), flush=True)
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
                        "backend": backend, "engine": {"build": importlib.metadata.version("tensorrt_llm"),
                                                        "max_seq_len": max_len, "max_num_tokens": max_tokens, "backend": backend,
                                                        "worker_single_process": a.worker_single_process}}), flush=True)
  return 0


def _matrices(llm) -> list | str:
  """Read loaded, local Linear dimensions and the runtime's weight-packing contract, outside the timed window."""
  try:
    model = llm._executor.engine.model_engine.model
    out = {}
    for name, mod in model.named_modules():
      if not all(hasattr(mod, k) for k in ("in_features", "out_features", "weights_loading_config")):
        continue
      if getattr(mod, "tp_size", 1) != 1:
        return "not read: local tensor-parallel matrices do not establish the full model's work"
      parent, _, last = name.rpartition(".")
      parent = parent.rsplit(".", 1)[-1]
      shards = mod.weights_loading_config.weight_mode.shard_keys
      parts = tuple(f"{parent}.{s}_proj" for s in shards) if shards else (f"{parent}.{last}",)
      key = (last, int(mod.out_features), int(mod.in_features), str(mod.dtype), parts)
      out[key] = out.get(key, 0) + 1
    return [{"name": k[0], "n": k[1], "k": k[2], "dtype": k[3], "parts": list(k[4]), "count": count}
            for k, count in sorted(out.items())]
  except (AttributeError, TypeError, ValueError) as exc:
    return f"not read: {type(exc).__name__}: {str(exc)[:200]}"


if __name__ == "__main__":
  raise SystemExit(main())
