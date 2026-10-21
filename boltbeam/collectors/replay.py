"""A recorded session replayed on the user's own server: per turn, where the time goes against the roofline.

    pipeline MODEL --run RUN --target T --workload replay --replay-trace TRACE --server-argv 'llama-server ...' --analyze

The input is a replay trace (trace/replay_trace.py): counts per turn, never chat text. Step 4 (`measure`):

  server   the user's own command (`--server-argv`), with only --host and --port replaced; prompt cache on, as the
           server defaults it. The binary's version line and sha256, the load time (start to /health) and the
           settings it logged (context, chunk, KV cache types, flash attention, checkpoints) are recorded
  words    the document set (collectors/session.documents) tokenized by the server's own tokenizer; each turn's new
           tokens are the next slice of it, turned back into text, so every token is real and none repeats
  turns    played through the chat endpoint with the user's template, as a client would: the system prompt, then per
           turn a user message (its new and tool tokens) and the model's own reply (greedy, `ignore_eos`, exactly
           reply_tokens), which goes back into the history. The server's timings (cache_n, prompt_n, prompt_ms,
           predicted_n, predicted_ms) and the client's own clock (send, first token, last token) per turn, and the GPU
           clock over each turn from one sampler at INTERVAL_S. The first turn is cold (weights just loaded, cache
           empty) and flagged. The replay stops before the context fills, with a row that says so

Step 5 (`role_time`) captures PICKS turns with llama-bench at the turn's depth with the server's cache types, chunk and
flash attention (the same build as the server), attributed as a prefill at that depth (prefill.role_time). A capture
is a separate benchmark: it does not reproduce the server's checkpointing and scheduling. Its rows remain
benchmark-only unless both its engine timer and GPU span match the server's prompt_ms within the matching repeat
band. One decode capture at the deepest turn. The ceilings and the table are workflow/replay.py's.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import time
import urllib.request
from typing import Any, Callable

from boltbeam.collectors import gpu_telemetry
from boltbeam.collectors import prefill as pf
from boltbeam.collectors.replay_overhead import ORDER, PAIRS
from boltbeam.trace import replay_trace as rt

SERVER = "replay_server.json"  # the user's server command as given, and what the run read back from it
SERVER_LOG = "replay_server.log"
CAPTURE = "replay_capture.json"
PROFILED = "timing_trace_profiled.json"  # the same server under the profiler: the arms qualification pairs against the native ones
ESTIMATOR_VERSION = 1  # of server_window and the pairing below
PROFILE_FLUSH_MS = 1000  # a server its parent stops with a signal flushes the profiler's buffers on a timer (vendor_capture.capture)
INTERVAL_S = 0.01  # a 150 ms turn at the default 50 ms would hold three samples
HEADROOM = 64  # tokens left free under the server's context: the replay stops before a turn would need them
DECODE_TOKENS = 64  # decode tokens in the deep turn's decode capture
REQUEST = {"temperature": 0.0, "ignore_eos": True, "cache_prompt": True, "stream": True,
           "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
REQUEST_WORDS = ("greedy, ignore_eos, exactly the trace's reply tokens, prompt cache on, streamed, thinking off in "
                 "the template")


# --- the llama.cpp server -----------------------------------------------------------------------------------------

def _post(url:str, body:dict[str, Any], timeout:float = 600.0) -> dict[str, Any]:
  from boltbeam.adapters.llama_server_control import _json_post
  return _json_post(url, body, timeout)


def stream_chat(url:str, messages:list[dict[str, str]], reply:int, **extra:Any) -> dict[str, Any]:
  """One turn streamed through the OpenAI chat endpoint: the reply text, the timings and usage the server put on
  its chunks, and the client's clock (monotonic) at send, first token and last token."""
  body = json.dumps({"messages": messages, "max_tokens": reply, **REQUEST, **extra}).encode()
  req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
  from boltbeam.collectors.replay_semantics import StreamTokens
  tokens = StreamTokens() if extra.get("return_token_ids") else None
  text, timings, usage, first = [], {}, {}, None
  sent = time.monotonic()
  with urllib.request.urlopen(req, timeout=3600) as resp:
    for raw in resp:
      line = raw.decode(errors="replace").strip()
      if not line.startswith("data:") or line == "data: [DONE]":
        continue
      chunk = json.loads(line[5:])
      if tokens is not None:tokens.add(chunk)
      for c in chunk.get("choices") or []:
        piece = (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
        if piece:
          first = first or time.monotonic()
          text.append(piece)
      timings = chunk.get("timings") or timings
      usage = chunk.get("usage") or usage
  last = time.monotonic()
  return {"text": "".join(text), "timings": timings, "usage": usage, "t_send": sent, "t_first": first or last, "t_last": last,
          **({"response":tokens.finish(usage)} if tokens is not None else {})}


llama_chat = stream_chat  # llama.cpp puts its own timings on the last chunk


def llama_tokenize(url:str, text:str) -> list[int]:
  return _post(url + "/tokenize", {"content": text, "add_special": False}, 120.0)["tokens"]


def llama_detokenize(url:str, ids:list[int]) -> str:
  return _post(url + "/detokenize", {"tokens": ids}, 120.0)["content"]


FLAGS = {"n_batch": ("-b", "--batch-size"), "n_ubatch": ("-ub", "--ubatch-size"), "type_k": ("-ctk", "--cache-type-k"),
         "type_v": ("-ctv", "--cache-type-v"), "flash_attn": ("-fa", "--flash-attn")}  # llama.cpp's server and llama-bench share them
DEFAULTS = {"n_batch": 2048, "n_ubatch": 512, "type_k": "f16", "type_v": "f16", "flash_attn": "auto"}  # common/common.h
LOGGED = {"n_ctx": r"(?:new slot, n_ctx|n_ctx_slot)\s*=\s*(\d+)", "ctx_checkpoints": r"context checkpoints enabled, max = (\d+)",
          "cache_ram_mib": r"prompt cache is enabled, size limit: (\d+) MiB"}  # what the server logs at start


def flag_value(argv:list[str], names:tuple[str, ...]) -> str | None:
  """The last value the command gives a flag (`-x V` or `--x=V`)."""
  got = None
  for i, a in enumerate(argv):
    if a in names and i + 1 < len(argv):
      got = argv[i + 1]
    elif a.split("=", 1)[0] in names and "=" in a:
      got = a.split("=", 1)[1]
  return got


def llama_settings(argv:list[str], log:str) -> dict[str, Any]:
  """The configuration the server ran: each FLAGS setting from the user's command, else the engine's default; the
  context, checkpoints and prompt-cache size from its log. `source` says where each came from."""
  out: dict[str, Any] = {"source": {}}
  for key, names in FLAGS.items():
    v = flag_value(argv, names)
    out[key] = int(v) if v and v.isdigit() else (v if v is not None else DEFAULTS[key])
    out["source"][key] = "the command" if v is not None else "llama.cpp's default"
  for key, pat in LOGGED.items():
    m = re.search(pat, log)
    out[key] = int(m.group(1)) if m else None
    out["source"][key] = "the server's log" if m else "not logged"
  checkpoints = flag_value(argv, ("-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"))
  if out["ctx_checkpoints"] is None and checkpoints and checkpoints.isdigit():
    out["ctx_checkpoints"] = int(checkpoints)
    out["source"]["ctx_checkpoints"] = "the command"
  if out["n_ctx"] is None:
    # --ctx-size is shared across slots; prefer the actual per-slot size from the log.
    ctx = flag_value(argv, ("-c", "--ctx-size"))
    parallel = flag_value(argv, ("-np", "--parallel"))
    if ctx and ctx.isdigit() and parallel and parallel.isdigit() and int(parallel) > 0:
      out["n_ctx"] = int(ctx) // int(parallel)
      out["source"]["n_ctx"] = "the command's context divided by its parallel slots"
  return out


def llama_weight_path(quant:str) -> str | None:
  """A weight's matrix path in llama.cpp's prefill: block quants run mul_mat_q on int8 (activations quantized to
  q8_1); float weights run cuBLAS on their own type."""
  from boltbeam.quantization.quant import load_quant_registry
  q = load_quant_registry().get(str(quant).upper())
  if q is None:
    return None
  return {"F16": "fp16", "BF16": "bf16"}.get(str(quant).upper()) if q.dequant_family == "float_passthrough" else "int8"


def llama_attention_paths(binary:pathlib.Path, device:dict[str, Any] | None = None) -> dict[str, int]:
  """The attention kernels' instruction sites per matrix path, summed over the engine's libraries (no GPU);
  device: the run's GPU, whose vendor and target say which code to read."""
  return _attention(pf.library_paths(pf.linked_libraries(binary.resolve()), device=device))


def _attention(found:dict[str, Any]) -> dict[str, int]:
  from boltbeam.workflow.tie_out import kind_of
  out: dict[str, int] = {}
  for name, paths in (found.get("kernels") or {}).items():
    if kind_of({"kernel": name}) == "attention":
      for p, n in paths.items():
        out[p] = out.get(p, 0) + n
  return out


def llama_bench_flags(s:dict[str, Any]) -> list[str]:
  """llama-bench flags that match the server's configuration: cache types, chunk and flash attention (FLAGS)."""
  return [x for key in FLAGS if s.get(key) is not None for x in (FLAGS[key][0], str(s[key]))]


# --- the vLLM OpenAI server ----------------------------------------------------------------------------------------

VLLM_METRICS = {"hits": "vllm:prefix_cache_hits_total", "prompt": "vllm:prompt_tokens_total",
                "generated": "vllm:generation_tokens_total", "prefill_s": "vllm:request_prefill_time_seconds_sum",
                "decode_s": "vllm:request_decode_time_seconds_sum"}  # vLLM's own counters, read before and after a turn
VLLM_SPEC_METRICS = {"draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
                     "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total"}


def vllm_metrics(url:str) -> dict[str, float]:
  text = urllib.request.urlopen(url + "/metrics", timeout=60).read().decode()
  out = {}
  for key, name in (VLLM_METRICS | VLLM_SPEC_METRICS).items():
    out[key] = sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines() if l.startswith(name + "{") or l.startswith(name + " "))
  return out


def vllm_timings(before:dict[str, float], after:dict[str, float]) -> dict[str, Any]:
  """One request's timings on llama.cpp's keys, from vLLM's counters (one request at a time, so the deltas are its):
  cache_n the prefix-cache hits in tokens, prompt_n the prompt less them, prompt_ms vLLM's prefill time (scheduled to
  first token), predicted_n and predicted_ms its generated tokens and decode time (first token to last)."""
  d = {k: after[k] - before[k] for k in VLLM_METRICS}
  spec = {k: after.get(k, 0) - before.get(k, 0) for k in VLLM_SPEC_METRICS}
  return {"cache_n": int(d["hits"]), "prompt_n": int(d["prompt"] - d["hits"]), "prompt_ms": d["prefill_s"] * 1e3,
          "predicted_n": int(d["generated"]), "predicted_ms": d["decode_s"] * 1e3,
          **({"speculative": {**spec, "acceptance": spec["accepted_tokens"] / spec["draft_tokens"]}} if spec["draft_tokens"] > 0 else {})}


def vllm_chat(url:str, messages:list[dict[str, str]], reply:int, *, token_evidence:bool = False) -> dict[str, Any]:
  model = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=60).read())["data"][0]["id"]
  before = vllm_metrics(url)
  got = stream_chat(url, messages, reply, model=model, **({"return_token_ids": True} if token_evidence else {}))
  got["timings"] = vllm_timings(before, vllm_metrics(url))
  if token_evidence:
    usage=got["response"]["usage"];timings=got["timings"]
    if (usage["prompt_tokens"]!=timings["prompt_n"]+timings["cache_n"] or
        usage["completion_tokens"]!=timings["predicted_n"]):
      raise ValueError("replay response tokens disagree with server metric deltas")
    cache=(usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cache is not None and cache!=timings["cache_n"]:
      raise ValueError("replay response cache count disagrees with server metric deltas")
    # The metrics are observed around this serial request, independently of the worker ledger.
    usage["prompt_tokens_details"]={**(usage.get("prompt_tokens_details") or {}),"cached_tokens":timings["cache_n"]}
  return got


def vllm_tokenize(url:str, text:str) -> list[int]:
  return _post(url + "/tokenize", {"prompt": text, "add_special_tokens": False}, 120.0)["tokens"]


def vllm_detokenize(url:str, ids:list[int]) -> str:
  return _post(url + "/detokenize", {"tokens": ids}, 120.0)["prompt"]


VLLM_FLAGS = {"n_ctx": ("--max-model-len",), "n_batch": ("--max-num-batched-tokens",), "type_k": ("--kv-cache-dtype",)}
VLLM_LOGGED = {"n_ctx": r"max_model_len=(\d+)", "n_batch": r"max_num_batched_tokens=(\d+)", "type_k": r"kv_cache_dtype=[\"']?([a-z0-9_]+)"}


def vllm_settings(argv:list[str], log:str) -> dict[str, Any]:
  """The configuration vLLM ran: each setting from the command, else what its log says; the KV cache in the model's
  bf16 unless the command sets it ("auto" is the model's type); the chunk is max_num_batched_tokens, one output per
  chunk; the prefix cache on unless the command turns it off (vLLM's default)."""
  out: dict[str, Any] = {"source": {}}
  for key, names in VLLM_FLAGS.items():
    v = flag_value(argv, names)
    m = re.search(VLLM_LOGGED[key], log) if v is None and key in VLLM_LOGGED else None
    v = v if v is not None else (m.group(1) if m else None)
    out[key] = int(v) if v and v.isdigit() else v
    out["source"][key] = "the command" if flag_value(argv, names) is not None else ("the server's log" if m else "vLLM's default")
  if out["type_k"] in (None, "auto"):
    out["type_k"], out["source"]["type_k"] = "bf16", "the model's dtype (auto)"
  out["n_ubatch"], out["source"]["n_ubatch"] = out["n_batch"], out["source"]["n_batch"]
  out["prefix_cache"] = "--no-enable-prefix-caching" not in argv
  out["source"]["prefix_cache"] = "the command" if any("prefix-caching" in a for a in argv) else "vLLM's default"
  spec = flag_value(argv, ("--speculative-config",))
  out["speculative_config"] = json.loads(spec) if spec else None
  out["source"]["speculative_config"] = "the command" if spec else "not enabled in the command"
  return out


def vllm_attention_paths(binary:pathlib.Path, device:dict[str, Any] | None = None) -> dict[str, int]:
  """The attention kernels' instruction sites per path in the vLLM package beside this python (no GPU)."""
  site = sorted(binary.expanduser().absolute().parent.parent.glob("lib/python*/site-packages"))  # the venv, not its symlinked python
  libs = [l for s in site for l in sorted(s.glob("vllm/*.so")) + sorted(s.glob("vllm/vllm_flash_attn/*.so"))]
  return _attention(pf.library_paths(libs, device=device))


def vllm_env(argv:list[str]) -> dict[str, str]:
  """vLLM's own venv, its run settings, and no download."""
  import os
  from boltbeam.collectors import engine_common as ec, vllm_decode as vd
  return {**os.environ, **ec.venv_env(argv[0]), **vd.RUN_ENV, "HF_HUB_OFFLINE": "1"}


REPLAY_ENGINES: dict[str, dict[str, Callable[..., Any]]] = {
  "llama.cpp": {"chat": llama_chat, "tokenize": llama_tokenize, "detokenize": llama_detokenize,
                "settings": llama_settings, "weight_path": llama_weight_path, "attention_paths": llama_attention_paths,
                "formats": lambda s, weights: pf.llama_formats(s.get("type_k"), weights), "bench_flags": llama_bench_flags,
                # the one GPU a server command keeps to (split mode none: its main GPU); None, it may use them all
                "gpus": lambda argv: [int(flag_value(argv, ("-mg", "--main-gpu")) or 0)]
                if flag_value(argv, ("-sm", "--split-mode")) == "none" else None},
  "vllm": {"chat": vllm_chat, "tokenize": vllm_tokenize, "detokenize": vllm_detokenize, "settings": vllm_settings,
           "weight_path": lambda quant: {"NVFP4_G16": "nvfp4", "FP8_E4M3": "fp8", "F32": "fp32_fma"}.get(quant, "bf16"),
           "attention_paths": vllm_attention_paths, "formats": lambda s, weights: pf.vllm_formats(weights, {"kv_cache_dtype": s.get("type_k")}),
           "env": vllm_env,
           "gpus": lambda argv: [0] if int(flag_value(argv, ("-tp", "--tensor-parallel-size")) or 1) == 1 else None},
}


def server_gpus(run:pathlib.Path, provider:str, argv:list[str]) -> dict[str, Any]:
  """Which of the machine's GPUs the user's server may run on, read from its own flags: BoltBeam never pins the
  user's command. The run's telemetry samples GPU 0 alone, so it stands for the server only when that is the one GPU
  the server uses; otherwise its clock scales no turn's peaks, and `why` says so."""
  from boltbeam.workflow.layout import devices
  hw = run / "hardware_profile.json"
  n = len(devices(json.loads(hw.read_text()).get("gpu") or {})) if hw.is_file() else 1
  used = engine(provider)["gpus"](argv)
  if n <= 1 or used == [0]:
    return {"machine_gpus": n, "server_gpus": used, "telemetry_covers": True}
  where = f"may use any of the machine's {n} GPUs" if used is None else f"runs on GPU {used[0]}"
  return {"machine_gpus": n, "server_gpus": used, "telemetry_covers": False,
          "why": f"the server {where} and the telemetry samples GPU 0 alone, so its clock scales no turn's peaks"}


def engine(provider:str) -> dict[str, Callable[..., Any]]:
  if provider not in REPLAY_ENGINES:
    raise RuntimeError(f"a replay runs on {', '.join(REPLAY_ENGINES)}; {provider} has no replay adapter")
  return REPLAY_ENGINES[provider]


# --- inputs --------------------------------------------------------------------------------------------------------

def server_argv(text:str) -> list[str]:
  """The user's server command from a string (a shell line, or the contents of a command file)."""
  argv = shlex.split(text.replace("\\\n", " "))
  if not argv:
    raise ValueError("the server command is empty")
  return argv


def stage(run:pathlib.Path, trace:str | None, argv_text:str | None, pairs:int | None = None) -> int:
  """The run's replay inputs: the trace as read (a trace file, or a shape's name; none: the default shape), counts
  only, and the server command; returns the prefix the trace reaches, the run's context."""
  path = pathlib.Path(trace).expanduser() if trace else None
  trace = rt.read(path) if path and path.is_file() else rt.shape(trace or None)
  if not argv_text:
    raise ValueError("a replay needs the server command it replays on (--server-argv)")
  rt.write(trace, run / rt.FILE)
  (run / SERVER).write_text(json.dumps({"argv": server_argv(argv_text), **({"pairs": pairs} if pairs else {})}, indent=1) + "\n")
  return rt.prefix_tokens(trace)


def running_servers(ps:Callable[[], str] | None = None) -> list[list[str]]:
  """The commands of the llama-server processes running on this machine (`ps`), to offer as the server to replay
  on. Shown for confirmation; never used without it."""
  text = ps() if ps else subprocess.run(["ps", "-axo", "args="], capture_output=True, text=True).stdout
  out = []
  for line in text.splitlines():
    argv = line.split()
    if argv and pathlib.Path(argv[0]).name == "llama-server":
      out.append(argv)
  return out


def _sha256(path:pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
      h.update(block)
  return h.hexdigest()


def _version(binary:pathlib.Path) -> str | None:
  try:
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60)
  except (OSError, subprocess.TimeoutExpired):
    return None
  return next((l.strip() for l in (out.stdout + out.stderr).splitlines() if l.strip().startswith("version")), None)


def build_metadata(provider:str, version:str | None, log:str = "") -> dict[str, Any]:
  """Parse recorded engine evidence only; a Python executable's version is not a vLLM build."""
  if provider == "llama.cpp":
    text = version or ""
    m = re.search(r"build\s+(\d+),\s*commit\s+([0-9a-f]+)", text)
    if not m:
      m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text)
    return {"build": m.group(2), "build_number": int(m.group(1))} if m else {}
  if provider == "vllm":
    m = re.search(r"Initializing a V\d+ LLM engine \(v([^\s)]+)\)", log)
    return {"build": m.group(1)} if m else {}
  return {}


def model_of(argv:list[str]) -> str | None:
  for i, a in enumerate(argv):
    if a in ("-m", "--model") and i + 1 < len(argv):
      return argv[i + 1]
    if a.startswith("--model="):
      return a.split("=", 1)[1]
  return None


# --- step 4 -------------------------------------------------------------------------------------------------------

class Words:
  """The document set as the server tokenizes it, handed out in order: every turn's text is new."""

  def __init__(self, eng:dict[str, Callable[..., Any]], url:str):
    from boltbeam.collectors.session import documents
    self.eng, self.url, self.ids, self.at = eng, url, [], 0
    self.files, self.source = documents()

  def take(self, n:int) -> str:
    while len(self.ids) - self.at < n:
      if not self.files:
        raise RuntimeError(f"the document set ran out after {len(self.ids)} tokens")
      self.ids += self.eng["tokenize"](self.url, self.files.pop(0).read_text(errors="replace"))
    piece = self.ids[self.at:self.at + n]
    self.at += n
    return self.eng["detokenize"](self.url, piece) if piece else ""


def turn_row(i:int, got:dict[str, Any], prev:dict[str, Any] | None, t:dict[str, Any], tele:gpu_telemetry.Telemetry,
             t0:float) -> dict[str, Any]:
  """One turn's measures from the server's timings and the client's clock. needed: the tokens the turn added, what
  the prompt holds beyond what the cache could hold from the turn before (its prompt and its reply), and never less
  than the trace's own new tokens (a template that rewrites the history shortens the prompt); the whole prompt on the
  first turn; after compaction the server's retained prefix is still reused. recomputed: tokens processed beyond those."""
  tm, us = got["timings"], got["usage"]
  cache_n, prompt_n = int(tm.get("cache_n") or 0), int(tm.get("prompt_n") or 0)
  prefix = cache_n + prompt_n
  held = cache_n if t["compaction"] else (prev["prefix"] + prev["predicted_n"] if prev else 0)
  growth = min(prefix, max(prefix - held, t["new_tokens"] + t["tool_tokens"])) if held else prefix
  pred_n, pred_ms, prompt_ms = int(tm.get("predicted_n") or 0), float(tm.get("predicted_ms") or 0.0), float(tm.get("prompt_ms") or 0.0)
  ttft = (got["t_first"] - got["t_send"]) * 1e3
  stream = (got["t_last"] - got["t_first"]) * 1e3
  win = tele.summary(got["t_send"], got["t_last"])
  return {"scope": "turn", "workload": "replay", "i": i, "at_s": round(got["t_send"] - t0, 3), "prefix": prefix,
          "context": prefix, "cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": prompt_ms, "predicted_n": pred_n,
          "predicted_ms": pred_ms, "prompt_tokens": us.get("prompt_tokens"), "completion_tokens": us.get("completion_tokens"),
          "tokens_cached": tm.get("tokens_cached"), "trace_new": t["new_tokens"] + t["tool_tokens"],
          "trace_reply": t["reply_tokens"], "needed": growth, "recomputed": max(prompt_n - growth, 0),
          "ttft_ms": ttft, "wall_ms": (got["t_last"] - got["t_send"]) * 1e3, "overhead_ms": (got["t_last"] - got["t_send"]) * 1e3 - prompt_ms - pred_ms,
          "ttft_overhead_ms": ttft - prompt_ms,
          "decode_tok_s": (pred_n - 1) / pred_ms * 1e3 if pred_n > 1 and pred_ms else None,
          "client_tok_s": (pred_n - 1) / stream * 1e3 if pred_n > 1 and stream > 0 else None,
          "telemetry": win, "throttled": win.get("throttled"), "cold": i == 1, "compaction": t["compaction"],
          **({"draft_n": tm["draft_n"]} if tm.get("draft_n") else {}),
          **({"speculative": tm["speculative"]} if tm.get("speculative") else {})}


def server_window(i:int, prompt_n:int, cache_n:int, server:dict[str, Any]) -> dict[str, Any]:
  """What one turn's server prompt_ms measured: the same turn, the same tokens processed after the same cache, on the
  same build and command. Two arms are comparable on a turn only when these are equal."""
  return {"method": "server_turn_prompt_ms", "turn": i, "prompt_n": prompt_n, "cache_n": cache_n,
          "server_build": server.get("version"), "argv_sha256": hashlib.sha256(json.dumps(server["argv"]).encode()).hexdigest(),
          "estimator_version": ESTIMATOR_VERSION}


def _arm(run:pathlib.Path, provider:str, server:dict[str, Any], trace:dict[str, Any], tele, say:Callable[[str], None], *,
         profiled:pathlib.Path | None = None, evidence:list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str]:
  """One server start and the whole trace through it: the load row and the turn rows, each turn with its window, and
  the document set's source. profiled: the folder to run the same command under the target's vendor profiler in."""
  from boltbeam.adapters.llama_server_control import serve_argv
  from boltbeam.collectors import vendor_capture
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest
  eng, argv, rows = engine(provider), server["argv"], []
  env = eng["env"](argv) if "env" in eng else None
  if profiled:  # the profiler the llama.cpp captures use (vendor_capture), around the server, nothing else changed
    backend, profiled = get_target(load_manifest(run).get("target_id")).backend, profiled.absolute()
    profiled.mkdir(parents=True, exist_ok=True)
    plan = vendor_capture.plan(backend)
    if plan["tool"] is None:
      raise vendor_capture.NoCapture(plan["reason"])
    argv = vendor_capture.VENDORS[backend][2](plan["tool"], argv, profiled, dict(env or {}), PROFILE_FLUSH_MS)
  say(f"replay: starting the server ({pathlib.Path(argv[0]).name}) with the user's flags" + (" under the profiler" if profiled else ""))
  log = run / (SERVER_LOG if not profiled else "replay_server_profiled.log")
  with serve_argv(argv, log_path=log, environ=env, stop_timeout_s=300.0 if profiled else 5.0) as (url, command, load_s):
    text = log.read_text(errors="replace")
    settings = eng["settings"](command, text)
    if profiled:
      server["profiled_command"] = command
    else:
      server["command"], server["settings"] = command, settings
      server.update(build_metadata(provider, server.get("version"), text))
    rows.append({"scope": "load", "workload": "replay", "seconds": round(load_s, 2),
                 "source": "from the server's start to /health answering 200"})
    words = Words(eng, url)
    n_ctx = int(settings.get("n_ctx") or 0)
    system = {"role": "system", "content": words.take(trace["system_tokens"])}
    messages = [system]
    prev, t0 = None, time.monotonic()
    for t in trace["turns"]:
      if t["compaction"]:
        messages = [system]
      need = (prev["prefix"] + prev["predicted_n"] if prev and not t["compaction"] else 0) + t["new_tokens"] + \
             t["tool_tokens"] + t["reply_tokens"] + HEADROOM
      if n_ctx and need > n_ctx:
        rows.append({"scope": "stopped", "workload": "replay", "i": t["i"],
                     "reason": f"turn {t['i']} needs about {need:,} tokens; the server's context is {n_ctx:,}"})
        break
      messages.append({"role": "user", "content": words.take(t["new_tokens"]) + words.take(t["tool_tokens"])})
      say(f"replay: turn {t['i']} of {len(trace['turns'])}")
      try:
        got = eng["chat"](url, messages, t["reply_tokens"], **({"token_evidence":True} if evidence is not None else {}))
      except Exception as exc:  # noqa: BLE001  a turn the server refused ends the replay with its reason
        if evidence is not None:raise  # incomplete identity must not look like a valid shortened capture
        rows.append({"scope": "stopped", "workload": "replay", "i": t["i"], "reason": f"the server failed: {str(exc)[-300:]}"})
        break
      messages.append({"role": "assistant", "content": got["text"]})
      prev = turn_row(t["i"], got, prev, t, tele, t0)
      prev["server_window"] = server_window(t["i"], prev["prompt_n"], prev["cache_n"], server)
      rows.append(prev)
      if evidence is not None:
        evidence.append({"turn":t["i"],**got["response"]})
  if profiled:  # the server is stopped: the profiler has written its report; the timing stands without the kernels, said
    try: server["profiled_kernels"] = len(vendor_capture.VENDORS[backend][3](plan["tool"], profiled))
    except Exception as exc: server["profiled_kernels_failed"] = str(exc)[-300:]  # noqa: BLE001
  return rows, words.source


def measure(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None,
            token_evidence:bool = False) -> pathlib.Path:
  """The trace played on the user's server: timing_trace.json with a load row, a server row and a row per turn.
  llama.cpp: PAIRS pairs of (native, profiled) arms, in ORDER (NP, NP, PN, PN), every arm a fresh server on the same
  command; the profiled arms are timing_trace_profiled.json. Each arm's turns hold their windows (server_window)."""
  if (run/'forced_replay_request.json').exists():
    from boltbeam.collectors.forced_replay import collect
    return collect(run,provider=provider,say=say)
  from boltbeam.core.canonical import pretty_json
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  from boltbeam.workflow.common import load_manifest
  manifest = load_manifest(run)
  eng = engine(provider)
  if token_evidence and provider!="vllm":raise ValueError("replay token evidence requires the vLLM response adapter")
  responses=[]
  response_path=run / "replay_responses.json"
  response_path.unlink(missing_ok=True)
  trace = rt.read(run / rt.FILE)
  staged = json.loads((run / SERVER).read_text())
  argv = staged["argv"]
  pairs = int(staged.get("pairs") or PAIRS) if provider == "llama.cpp" else 1
  binary = pathlib.Path(argv[0]).expanduser()
  server = {"scope": "server", "workload": "replay", "argv": argv, "binary": str(binary), "version": _version(binary),
            "sha256": _sha256(binary) if binary.is_file() else None, "model": model_of(argv),
            "request": REQUEST_WORDS, "attention_paths": None}
  try:
    from boltbeam.workflow.common import run_gpu
    server["attention_paths"] = eng["attention_paths"](binary, run_gpu(run))
  except Exception as exc:  # noqa: BLE001  no tool to read the code: attention is then bound by its bytes, said
    server["attention_paths_failed"] = str(exc)[-300:]
  arms: dict[str, list[dict[str, Any]]] = {"native": [], "profiled": []}
  rows: list[dict[str, Any]] = []
  order = [ORDER[k % len(ORDER)] for k in range(pairs)] if provider == "llama.cpp" else ["N"]
  with gpu_telemetry.run_telemetry(run, 0, INTERVAL_S) as tele:
    if provider == "llama.cpp": _arm(run, provider, server, trace, tele, say)  # unmeasured warm-up server: disk, clocks and first use are paid here, by neither arm
    for k, pair in enumerate(order, 1):
      for arm in pair:
        name = "native" if arm == "N" else "profiled"
        got, source = _arm(run, provider, server, trace, tele, say, profiled=None if arm == "N" else run / pf.RAW / f"replay-server-pair-{k}",
                           evidence=responses if token_evidence and arm == "N" and not arms["native"] else None)
        arms[name].append({"pair": k, "position": pair.index(arm) + 1, "order": pair, "rows": got})
        server["words_source"] = source
        rows = rows or (got if arm == "N" else [])
    if token_evidence:response_path.write_text(pretty_json(responses))
  turns = [r for r in rows if r["scope"] == "turn"]
  if not turns:
    raise RuntimeError("no turn was replayed: " + "; ".join(r.get("reason", "") for r in rows if r["scope"] == "stopped"))
  worker=flag_value(argv,("--worker-cls",)) if provider=="vllm" else None
  out = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
         "workload": "replay", "provider": provider, "provider_id": provider, "collector_id": "replay",
         "timing_source": "the server's own timings per turn and the client's clock; " +
                          (f"custom worker {worker}, not an untraced reference" if worker else "untraced"),
         "contexts": [max(r["prefix"] for r in turns)], "rows": [*rows, server], "arms": arms["native"], "warmup_arm": provider == "llama.cpp", "pairs": pairs, "order": order,
         "replay_trace": {"source": trace["source"], "turns": len(trace["turns"]), "sha256": trace["source"]["sha256"]},
         "telemetry": tele.summary(), "interval_s": INTERVAL_S, "clock_policy": gpu_telemetry.run_clock_policy(run),
         "server_gpus": server_gpus(run, provider, argv),
         "measured": ["turn.prompt_ms, turn.predicted_ms (the server's timings)", "turn.ttft_ms, turn.wall_ms (the client's clock)",
                      "load.seconds"]}
  if token_evidence:
    out["response_evidence"]={"file":response_path.name,"sha256":_sha256(response_path),"turns":len(responses)}
  path = run / "timing_trace.json"
  path.write_text(pretty_json(out))
  if arms["profiled"]:
    (run / PROFILED).write_text(pretty_json({**{k: v for k, v in out.items() if k != "arms"}, "arms": arms["profiled"], "rows": arms["profiled"][0]["rows"],
        "timing_source": "the same server under the vendor profiler: the server's own timings per turn",
        **{k: v for k, v in server.items() if k.startswith("profiled_")}}))
  return path


# --- step 5 -------------------------------------------------------------------------------------------------------

def picks(turns:list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
  """The turns captured: the first warm turn (early), the one whose prefix is nearest the median prefix (mid), the
  deepest (deep), and the one with the most tokens processed (biggest) when it is none of those."""
  warm = [r for r in turns if not r["cold"] and r["prompt_n"] > 0]
  if not warm:
    return []
  by_prefix = sorted(warm, key=lambda r: r["prefix"])
  median = by_prefix[len(by_prefix) // 2]["prefix"]
  out = [("early", warm[0]), ("mid", min(warm, key=lambda r: (abs(r["prefix"] - median), r["i"]))),
         ("deep", by_prefix[-1])]
  big = max(warm, key=lambda r: (r["prompt_n"], -r["i"]))
  if all(big["i"] != r["i"] for _, r in out):
    out.append(("biggest", big))
  seen, unique = set(), []
  for name, r in out:
    if r["i"] not in seen:
      seen.add(r["i"])
      unique.append((name, r))
  return unique


def pair_turns(native:dict[str, Any], profiled:dict[str, Any] | None) -> dict[int, dict[str, Any]]:
  """Each turn's native and profiled server prompt_ms, over the pairs. A turn is comparable only when it was played in
  every pair of both traces and its window (server_window: tokens processed after the cache, build, command) is the
  same in all of them: {"window", "native_ms", "profiled_ms", "pairs": [{pair, native_ms, profiled_ms}]} are the means
  over those identical windows and each pair's values; otherwise {"refused": words}, never forced."""
  arms = {"native": native.get("arms") or [], "profiled": (profiled or {}).get("arms") or []}
  if not arms["native"] or not arms["profiled"]:
    return {}
  have = {name: {a["pair"]: {r["i"]: r for r in a["rows"] if r["scope"] == "turn"} for a in rows} for name, rows in arms.items()}
  ks = sorted(set(have["native"]) | set(have["profiled"]))
  out: dict[int, dict[str, Any]] = {}
  for i in sorted({i for name in have for p in have[name].values() for i in p}):
    seen = [(f"{name} pair {k}", have[name].get(k, {}).get(i)) for name in have for k in ks]
    base = seen[0][1] and seen[0][1]["server_window"]
    why = [f"turn {i} was not played in {lab}" for lab, r in seen if r is None] or \
          [f"turn {i} {f} {base[f]} ({seen[0][0]}) against {r['server_window'].get(f)} ({lab})"
           for lab, r in seen for f in base if r["server_window"].get(f) != base[f]][:1]
    if why:
      out[i] = {"refused": "not checked: native and profiled server windows differ: " + why[0]}
      continue
    pairs = [{"pair": k, "native_ms": have["native"][k][i]["prompt_ms"], "profiled_ms": have["profiled"][k][i]["prompt_ms"]} for k in ks]
    out[i] = {"window": base, "native_ms": sum(x["native_ms"] for x in pairs) / len(pairs),
              "profiled_ms": sum(x["profiled_ms"] for x in pairs) / len(pairs), "pairs": pairs}
  return out


def role_time(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None) -> dict[str, Any]:
  """The picked turns captured with llama-bench at their depth with the server's flags, attributed, and tied out
  to the server's prompt_ms; one decode capture at the deepest turn: replay_capture.json."""
  if (run/'forced_replay_request.json').exists():
    from boltbeam.workflow.forced_replay import checked
    data=checked(run)
    return {'method':'nsys','pairing':data['pairing'],'timing_qualification':data['timing_qualification']}
  from boltbeam.collectors import llama_bench_decode as lbd
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest, read_json
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  timing = read_json(run / "timing_trace.json")
  server = next(r for r in timing["rows"] if r["scope"] == "server")
  if "bench_flags" not in engine(provider):  # no per-turn capture for this engine: said, not guessed
    out = {"schema": "boltbeam.replay_capture.v1", "turns": [], "decode": None,
           "not_captured": f"{provider}: a replay's turns are captured with llama-bench only; this engine's turns are not captured"}
    (run / CAPTURE).write_text(json.dumps(out, indent=1) + "\n")
    return out
  flags = engine(provider)["bench_flags"](server.get("settings") or {})
  model = str(manifest["model_path"])
  out: dict[str, Any] = {"schema": "boltbeam.replay_capture.v1", "flags": flags, "tie_out": "repeat-band agreement is disclosed, not a gate; qualification is the 5% capture-overhead rule", "turns": [],
                         "decode": None}
  turns = [r for r in timing["rows"] if r["scope"] == "turn"]
  for name, r in picks(turns):
    n, c = r["prompt_n"], r["cache_n"]
    say(f"replay: capturing turn {r['i']} ({name}): {n} tokens after {c}")
    file = f"replay_capture_{name}.json"
    raw = run / pf.RAW / f"replay-{name}"
    try:
      got = pf.role_time(run, provider, root=root, say=say, depth=c, lengths=[n], truth_ms={n: r["prompt_ms"]},
                         capture=lambda run_, model_, length, target_, root_: pf.llama_capture(
                           run_, model_, length, target_, c, extra=flags, raw=raw), trace=file)
    except Exception as exc:  # noqa: BLE001  a capture that fails is that turn's reason; the others go on
      out["turns"].append({"name": name, "i": r["i"], "failed": str(exc)[-400:]})
      continue
    attr = (got.get("lengths") or {}).get(str(n)) or {}
    out["turns"].append({"name": name, "i": r["i"], "prompt_n": n, "cache_n": c, "file": file,
                         "bench_ms": attr.get("measured_ms") or attr.get("captured_ms"), "server_ms": r["prompt_ms"]})
  deep = max(turns, key=lambda r: r["prefix"])
  say(f"replay: capturing decode at the deepest turn ({deep['prefix']} tokens)")
  try:
    lbd.role_time(run, target=target, model=model, model_id=manifest["model_id"], context=deep["prefix"],
                  tokens=DECODE_TOKENS, flags=_decode_flags(flags),
                  raw=run / lbd.RAW / "replay-deep", out="replay_decode_trace.json")
    out["decode"] = {"i": deep["i"], "context": deep["prefix"], "file": "replay_decode_trace.json",
                     "server_tok_s": deep.get("decode_tok_s")}
  except Exception as exc:  # noqa: BLE001
    out["decode"] = {"i": deep["i"], "context": deep["prefix"], "failed": str(exc)[-400:]}
  (run / CAPTURE).write_text(json.dumps(out, indent=1) + "\n")
  return out


def _decode_flags(flags:list[str]) -> list[str]:
  """The bench flags a decode capture keeps: the cache types and flash attention (chunk flags shape a prompt only)."""
  out, i = [], 0
  while i < len(flags):
    if flags[i] in ("-ub", "-b"):
      i += 2
      continue
    out.append(flags[i])
    i += 1
  return out
