"""vLLM's multi-token prediction (MTP) measured in the repo: decode speed per draft count k on natural prompts, the
spread over those prompts, and the acceptance vLLM counted, against k=0 on the same prompts and engine.

The one authority for what is measured is data/mtp_prompts.json (the prompt set, versioned, hashed into every
measurement); runtime/vllm_mtp_drive.py runs one k per process; this module refuses to start while another program
holds the GPU, runs k=0 first and every k after it, checks the GPU is free again after each (no engine worker left
behind), and writes boltbeam.mtp_measurement.v1. summarize() is the arithmetic, kept pure so it is tested alone.

Before this, every MTP figure BoltBeam showed came from one-off scripts outside the repo whose prompts are gone
(docs/recommended-setup.md, "No in-repo MTP measurement")."""
from __future__ import annotations

import datetime, hashlib, json, pathlib, statistics
from typing import Any

from boltbeam.collectors import engine_common as ec, providers, vllm_decode as vd

SCHEMA = "boltbeam.mtp_measurement.v1"
PROMPTS = pathlib.Path(__file__).resolve().parent.parent / "data" / "mtp_prompts.json"
DRIVER = ec.RUNTIME / "vllm_mtp_drive.py"
MEASUREMENT = "mtp_measurement.json"
KS = (1, 2, 3)  # the draft counts measured unless asked otherwise (screen mtp --ks, the screen's Measure MTP)


def offer(provider:str, draft_bytes:int, *, facts:bool) -> dict[str, Any] | None:
  """What Measure MTP runs for this engine and checkpoint: vLLM and a draft head, else None. The measurement is kept
  in this machine's facts, where the recommended setup reads it; without them it would be measured and not kept."""
  if provider != vd.PROVIDER or not draft_bytes:
    return None
  return {"ks": list(KS), "blocked": None if facts else "needs this machine's facts: run once on this chip first"}


class MtpRefused(RuntimeError):
  """The measurement cannot start or cannot be trusted: the reason says which and why."""


def prompt_set(path:pathlib.Path = PROMPTS) -> dict[str, Any]:
  raw = path.read_bytes()
  data = json.loads(raw)
  if data.get("schema") != "boltbeam.mtp_prompts.v1" or not data.get("prompts"):
    raise MtpRefused(f"{path} is not a boltbeam.mtp_prompts.v1 prompt set")
  ids = [p["id"] for p in data["prompts"]]
  if len(set(ids)) != len(ids):
    raise MtpRefused(f"{path}: prompt ids repeat")
  return {"path": str(path), "version": data["version"], "sha256": hashlib.sha256(raw).hexdigest(), "ids": ids}


def summarize(rows:list[dict[str, Any]]) -> dict[str, Any]:
  """Per depth and k: each prompt's decode tok/s, their median, range and spread ((max - min) / median), the speedup
  of the median over k=0's, and the acceptance (mean tokens per step, 1 + accepted / drafts; the accepted share of draft
  tokens; per draft position). Refuses rows that do not cover the same prompts at every k of a depth: a speed per k is
  only comparable over one prompt set."""
  out:dict[str, Any] = {}
  for depth in sorted({r["depth"] for r in rows}):
    at = [r for r in rows if r["depth"] == depth]
    ks = sorted({r["k"] for r in at})
    if 0 not in ks:
      raise MtpRefused(f"{depth}: no k=0 baseline; a speedup needs the same prompts without speculation")
    prompts = {k: sorted(r["prompt"] for r in at if r["k"] == k) for k in ks}
    if any(p != prompts[0] for p in prompts.values()):
      raise MtpRefused(f"{depth}: the k values did not run the same prompts ({prompts})")
    per_k = {}
    for k in ks:
      rs = sorted((r for r in at if r["k"] == k), key=lambda r: r["prompt"])
      speeds = [r["decode_tok_s"] for r in rs]
      med = statistics.median(speeds)
      drafts = sum(r["spec"]["drafts"] for r in rs)
      draft_tokens = sum(r["spec"]["draft_tokens"] for r in rs)
      accepted = sum(r["spec"]["accepted_tokens"] for r in rs)
      positions = [sum(r["spec"]["accepted_per_position"][i] for r in rs if i < len(r["spec"]["accepted_per_position"]))
                   for i in range(k)]
      if k and not drafts:
        raise MtpRefused(f"{depth} k={k}: vLLM counted no drafts; speculation did not run")
      per_k[str(k)] = {
        "per_prompt": {r["prompt"]: round(r["decode_tok_s"], 3) for r in rs},
        "median_tok_s": med, "min_tok_s": min(speeds), "max_tok_s": max(speeds),
        "spread": (max(speeds) - min(speeds)) / med,
        "context": statistics.median(r["context"] for r in rs),
        **({"accept_length": 1 + accepted / drafts, "accept_rate": accepted / draft_tokens,
            "accept_rate_per_position": [p / drafts for p in positions]} if k else {})}
    base = per_k["0"]["median_tok_s"]
    for k, v in per_k.items():
      v["speedup"] = v["median_tok_s"] / base
    out[depth] = per_k
  return out


def measure(model:str, ks:list[int], *, backend:str, out:pathlib.Path, tokens:int = 256, long_tokens:int = 2048,
            gpu_memory_utilization:float = 0.85) -> dict[str, Any]:
  """Run k=0 then each k in `ks` on the prompt set; write and return the measurement. The GPU must be free before,
  and free again after each k."""
  ks = sorted({int(k) for k in ks})
  if not ks or ks[0] < 1:
    raise MtpRefused("ks are the draft counts to measure, 1 or more; k=0, the baseline, always runs")
  py = vd.python()
  if py is None:
    raise MtpRefused(f"vLLM was not found: set ${vd.ENV}")
  free = providers.gpu_free(backend)
  if not free["free"]:
    raise MtpRefused(f"not started: {free['reason']}; a GPU another program holds gives slower numbers that look real")
  prompts = prompt_set()
  out.mkdir(parents=True, exist_ok=True)
  continuations = out / "mtp_continuations.json"
  rows, engines = [], []
  for k in [0, *ks]:
    argv = [py, str(DRIVER), "--model", model, "--k", str(k), "--prompts", prompts["path"], "--tokens", str(tokens),
            "--long-tokens", str(long_tokens), "--continuations", str(continuations),
            "--max-model-len", str(_context_limit(tokens, long_tokens)),
            "--gpu-memory-utilization", str(gpu_memory_utilization)]
    got = ec.run_driver(argv, env={**ec.venv_env(py), **vd.RUN_ENV}, log=out / f"mtp-k{k}.log")
    rows += [r for r in got if "prompt" in r]
    engines += [r["engine"] for r in got if "engine" in r]
    after = providers.gpu_free(backend)
    if not after["free"]:
      raise MtpRefused(f"k={k} left the GPU held: {after['reason']}; the next k would not run alone")
  record = {"schema": SCHEMA, "model": model, "prompts": prompts, "tokens": tokens, "long_tokens": long_tokens,
            "ks": [0, *ks], "engines": engines, "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "decode_speed": "(tokens - 1) / (last token - first token), the engine's own timestamps: no prefill",
            "depths": summarize(rows), "rows": rows}
  (out / MEASUREMENT).write_text(json.dumps(record, indent=1))
  return record


def _context_limit(tokens:int, long_tokens:int) -> int:
  """Room for the longest prompt, its continuation and the measured tokens; the prompts are a few hundred tokens."""
  return 1024 + long_tokens + tokens
