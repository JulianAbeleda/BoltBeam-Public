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
counts only when llama-bench's turn ties out to the server's prompt_ms within TIE_OUT; otherwise that turn's where
table is `unmeasured`. One decode capture at the deepest turn. The ceilings and the table are workflow/replay.py's.
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
from boltbeam.trace import replay_trace as rt

SERVER = "replay_server.json"  # the user's server command as given, and what the run read back from it
SERVER_LOG = "replay_server.log"
CAPTURE = "replay_capture.json"
INTERVAL_S = 0.01  # a 150 ms turn at the default 50 ms would hold three samples
HEADROOM = 64  # tokens left free under the server's context: the replay stops before a turn would need them
TIE_OUT = 0.10  # a capture stands for the server's turn only when llama-bench's turn is within this share of prompt_ms
DECODE_TOKENS = 64  # decode tokens in the deep turn's decode capture
REQUEST = {"temperature": 0.0, "ignore_eos": True, "cache_prompt": True, "stream": True,
           "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
REQUEST_WORDS = ("greedy, ignore_eos, exactly the trace's reply tokens, prompt cache on, streamed, thinking off in "
                 "the template")


# --- the llama.cpp server -----------------------------------------------------------------------------------------

def _post(url:str, body:dict[str, Any], timeout:float = 600.0) -> dict[str, Any]:
  from boltbeam.adapters.llama_server_control import _json_post
  return _json_post(url, body, timeout)


def llama_chat(url:str, messages:list[dict[str, str]], reply:int) -> dict[str, Any]:
  """One turn streamed through /v1/chat/completions: the reply text, the server's timings and usage, and the
  client's clock (monotonic) at send, first token and last token."""
  body = json.dumps({"messages": messages, "max_tokens": reply, **REQUEST}).encode()
  req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
  text, timings, usage, first = [], {}, {}, None
  sent = time.monotonic()
  with urllib.request.urlopen(req, timeout=3600) as resp:
    for raw in resp:
      line = raw.decode(errors="replace").strip()
      if not line.startswith("data:") or line == "data: [DONE]":
        continue
      chunk = json.loads(line[5:])
      for c in chunk.get("choices") or []:
        piece = (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
        if piece:
          first = first or time.monotonic()
          text.append(piece)
      timings = chunk.get("timings") or timings
      usage = chunk.get("usage") or usage
  last = time.monotonic()
  return {"text": "".join(text), "timings": timings, "usage": usage, "t_send": sent, "t_first": first or last, "t_last": last}


def llama_tokenize(url:str, text:str) -> list[int]:
  return _post(url + "/tokenize", {"content": text, "add_special": False}, 120.0)["tokens"]


def llama_detokenize(url:str, ids:list[int]) -> str:
  return _post(url + "/detokenize", {"tokens": ids}, 120.0)["content"]


FLAGS = {"n_batch": ("-b", "--batch-size"), "n_ubatch": ("-ub", "--ubatch-size"), "type_k": ("-ctk", "--cache-type-k"),
         "type_v": ("-ctv", "--cache-type-v"), "flash_attn": ("-fa", "--flash-attn")}  # llama.cpp's server and llama-bench share them
DEFAULTS = {"n_batch": 2048, "n_ubatch": 512, "type_k": "f16", "type_v": "f16", "flash_attn": "auto"}  # common/common.h
LOGGED = {"n_ctx": r"new slot, n_ctx = (\d+)", "ctx_checkpoints": r"context checkpoints enabled, max = (\d+)",
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
  return out


def llama_weight_path(quant:str) -> str | None:
  """A weight's matrix path in llama.cpp's prefill: block quants run mul_mat_q on int8 (activations quantized to
  q8_1); float weights run cuBLAS on their own type."""
  from boltbeam.quantization.quant import load_quant_registry
  q = load_quant_registry().get(str(quant).upper())
  if q is None:
    return None
  return {"F16": "fp16", "BF16": "bf16"}.get(str(quant).upper()) if q.dequant_family == "float_passthrough" else "int8"


def llama_attention_paths(binary:pathlib.Path) -> dict[str, int]:
  """The attention kernels' instruction sites per matrix path, summed over the engine's libraries (no GPU)."""
  from boltbeam.workflow.tie_out import kind_of
  found = pf.library_paths(pf.linked_libraries(binary.resolve()))
  out: dict[str, int] = {}
  for name, paths in (found.get("kernels") or {}).items():
    if kind_of({"kernel": name}) == "attention":
      for p, n in paths.items():
        out[p] = out.get(p, 0) + n
  return out


def llama_bench_flags(s:dict[str, Any]) -> list[str]:
  """llama-bench flags that match the server's configuration: cache types, chunk and flash attention (FLAGS)."""
  return [x for key in FLAGS if s.get(key) is not None for x in (FLAGS[key][0], str(s[key]))]


REPLAY_ENGINES: dict[str, dict[str, Callable[..., Any]]] = {
  "llama.cpp": {"chat": llama_chat, "tokenize": llama_tokenize, "detokenize": llama_detokenize,
                "settings": llama_settings, "weight_path": llama_weight_path, "attention_paths": llama_attention_paths,
                "formats": lambda s, weights: pf.llama_formats(s.get("type_k"), weights), "bench_flags": llama_bench_flags},
}


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


def stage(run:pathlib.Path, trace:str | None, argv_text:str | None) -> int:
  """The run's replay inputs: the trace as read (a trace file, or a shape's name; none: the default shape), counts
  only, and the server command; returns the prefix the trace reaches, the run's context."""
  path = pathlib.Path(trace).expanduser() if trace else None
  trace = rt.read(path) if path and path.is_file() else rt.shape(trace or None)
  if not argv_text:
    raise ValueError("a replay needs the server command it replays on (--server-argv)")
  rt.write(trace, run / rt.FILE)
  (run / SERVER).write_text(json.dumps({"argv": server_argv(argv_text)}, indent=1) + "\n")
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
  first turn and after a compaction. recomputed: tokens processed beyond those."""
  tm, us = got["timings"], got["usage"]
  cache_n, prompt_n = int(tm.get("cache_n") or 0), int(tm.get("prompt_n") or 0)
  prefix = cache_n + prompt_n
  held = (prev["prefix"] + prev["predicted_n"]) if prev and not t["compaction"] else 0
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
          "ttft_ms": ttft, "wall_ms": (got["t_last"] - got["t_send"]) * 1e3, "overhead_ms": ttft - prompt_ms,
          "decode_tok_s": (pred_n - 1) / pred_ms * 1e3 if pred_n > 1 and pred_ms else None,
          "client_tok_s": (pred_n - 1) / stream * 1e3 if pred_n > 1 and stream > 0 else None,
          "telemetry": win, "throttled": win.get("throttled"), "cold": i == 1, "compaction": t["compaction"],
          **({"draft_n": tm["draft_n"]} if tm.get("draft_n") else {})}


def measure(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None) -> pathlib.Path:
  """The trace played on the user's server: timing_trace.json with a load row, a server row and a row per turn."""
  from boltbeam.adapters.llama_server_control import serve_argv
  from boltbeam.core.canonical import pretty_json
  from boltbeam.vocab import SCHEMA_TIMING_TRACE
  from boltbeam.workflow.common import load_manifest
  manifest = load_manifest(run)
  eng = engine(provider)
  trace = rt.read(run / rt.FILE)
  argv = json.loads((run / SERVER).read_text())["argv"]
  binary = pathlib.Path(argv[0]).expanduser()
  server = {"scope": "server", "workload": "replay", "argv": argv, "binary": str(binary), "version": _version(binary),
            "sha256": _sha256(binary) if binary.is_file() else None, "model": model_of(argv),
            "request": REQUEST_WORDS, "attention_paths": None}
  try:
    server["attention_paths"] = eng["attention_paths"](binary)
  except Exception as exc:  # noqa: BLE001  no tool to read the code: attention is then bound by its bytes, said
    server["attention_paths_failed"] = str(exc)[-300:]
  rows: list[dict[str, Any]] = []
  say(f"replay: starting the server ({binary.name}) with the user's flags")
  with gpu_telemetry.Telemetry(0, INTERVAL_S) as tele:
    with serve_argv(argv, log_path=run / SERVER_LOG) as (url, command, load_s):
      server["command"] = command
      server["settings"] = eng["settings"](command, (run / SERVER_LOG).read_text(errors="replace"))
      rows.append({"scope": "load", "workload": "replay", "seconds": round(load_s, 2),
                   "source": "from the server's start to /health answering 200"})
      words = Words(eng, url)
      n_ctx = int(server["settings"].get("n_ctx") or 0)
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
          got = eng["chat"](url, messages, t["reply_tokens"])
        except Exception as exc:  # noqa: BLE001  a turn the server refused ends the replay with its reason
          rows.append({"scope": "stopped", "workload": "replay", "i": t["i"], "reason": f"the server failed: {str(exc)[-300:]}"})
          break
        messages.append({"role": "assistant", "content": got["text"]})
        prev = turn_row(t["i"], got, prev, t, tele, t0)
        rows.append(prev)
  turns = [r for r in rows if r["scope"] == "turn"]
  if not turns:
    raise RuntimeError("no turn was replayed: " + "; ".join(r.get("reason", "") for r in rows if r["scope"] == "stopped"))
  server["words_source"] = words.source
  out = {"schema": SCHEMA_TIMING_TRACE, "model_id": manifest["model_id"], "target_id": manifest["target_id"],
         "workload": "replay", "provider": provider, "provider_id": provider, "collector_id": "replay",
         "timing_source": "the server's own timings per turn and the client's clock, untraced",
         "contexts": [max(r["prefix"] for r in turns)], "rows": [*rows, server],
         "replay_trace": {"source": trace["source"], "turns": len(trace["turns"]), "sha256": trace["source"]["sha256"]},
         "telemetry": tele.summary(), "interval_s": INTERVAL_S, "clock_policy": gpu_telemetry.clock_policy(0),
         "measured": ["turn.prompt_ms, turn.predicted_ms (the server's timings)", "turn.ttft_ms, turn.wall_ms (the client's clock)",
                      "load.seconds"]}
  path = run / "timing_trace.json"
  path.write_text(pretty_json(out))
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


def ties_out(bench_ms:float | None, server_ms:float) -> bool:
  return bool(bench_ms) and server_ms > 0 and abs(bench_ms - server_ms) <= TIE_OUT * server_ms


def role_time(run:pathlib.Path, provider:str, *, root=None, say:Callable[[str], None] = lambda _: None) -> dict[str, Any]:
  """The picked turns captured with llama-bench at their depth with the server's flags, attributed, and tied out
  to the server's prompt_ms; one decode capture at the deepest turn: replay_capture.json."""
  from boltbeam.collectors import llama_bench_decode as lbd
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest, read_json
  manifest = load_manifest(run)
  target = get_target(manifest.get("target_id"))
  timing = read_json(run / "timing_trace.json")
  server = next(r for r in timing["rows"] if r["scope"] == "server")
  flags = engine(provider)["bench_flags"](server.get("settings") or {})
  model = str(manifest["model_path"])
  out: dict[str, Any] = {"schema": "boltbeam.replay_capture.v1", "flags": flags, "tie_out": TIE_OUT, "turns": [],
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
    bench = attr.get("measured_ms") or attr.get("captured_ms")
    out["turns"].append({"name": name, "i": r["i"], "prompt_n": n, "cache_n": c, "file": file, "bench_ms": bench,
                         "server_ms": r["prompt_ms"], "ties_out": ties_out(bench, r["prompt_ms"])})
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
