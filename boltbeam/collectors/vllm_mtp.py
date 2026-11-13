"""vLLM's multi-token prediction (MTP), measured as collectors/mtp.py measures any engine (one record shape, one sweep):
runtime/vllm_mtp_drive.py runs one k per process, its own engine, so no engine worker holds the GPU after it. Decode
speed is (tokens - 1) / (last token - first token), the engine's own timestamps; the acceptance is vLLM's spec-decode
counters per prompt. The long depth is real text (data/mtp_long_corpus.json) each process cuts with the model's tokenizer.

Before this, every MTP figure BoltBeam showed came from one-off scripts outside the repo whose prompts are gone
(docs/recommended-setup.md, "No in-repo MTP measurement")."""
from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.collectors import engine_common as ec, mtp, vllm_decode as vd

DRIVER = ec.RUNTIME / "vllm_mtp_drive.py"
GPU_SHARE = 0.85  # the GPU share each engine is given here: below vLLM's 0.92 default, so a desktop's display keeps room


def offer(model:str, draft_bytes:int, *, facts:bool) -> dict[str, Any] | None:
  """What Measure MTP runs for this checkpoint: a draft head in it, else None. The measurement is kept in this
  machine's facts, where the recommended setup reads it; without them it would be measured and not kept."""
  from boltbeam.report.plain import words
  return {"ks": list(mtp.KS), "blocked": None if facts else words()["mtp_pick"]["offer_facts"]} if draft_bytes else None


def measure(model:str, ks:list[int], *, backend:str, out:pathlib.Path, tokens:int = mtp.TOKENS, long_tokens:int = 2048,
            gpu_memory_utilization:float = GPU_SHARE, python:str | None = None, long_why:str = "asked (--long-tokens)",
            grow:bool = False, head:str | None = None) -> dict[str, Any]:
  """collectors/mtp.sweep on vLLM. python: the engine's python (default: the one vd.python() finds)."""
  if head:
    raise mtp.MtpRefused("vLLM's MTP head is the checkpoint's own: a head file (--draft) is llama.cpp's")
  py = python or vd.python()
  if py is None:
    raise mtp.MtpRefused(f"vLLM was not found: set ${vd.ENV}")
  prompts = mtp.prompt_set()

  def run_k(k:int) -> list[dict[str, Any]]:
    argv = [py, str(DRIVER), "--model", model, "--k", str(k), "--prompts", prompts["path"], "--tokens", str(tokens),
            "--long-tokens", str(long_tokens), "--long-corpus", str(mtp.LONG),
            "--max-model-len", str(prompts["room"] + long_tokens + tokens),
            "--gpu-memory-utilization", str(gpu_memory_utilization)]
    return ec.run_driver(argv, env={**ec.venv_env(py), **vd.RUN_ENV}, log=out / f"mtp-k{k}.log")
  return mtp.sweep(run_k, ks, model=model, backend=backend, out=out, tokens=tokens, long_tokens=long_tokens, long_why=long_why, grow=grow,
                   decode_speed="(tokens - 1) / (last token - first token), the engine's own timestamps: no prefill")
