#!/usr/bin/env python3
"""Decode time per token in llama-server, at a fixed context, from the server's own `timings`.

Any python3 runs this file by path; it uses only the standard library. It starts the llama-server it is given on a
private port with the model, the context size and the engine flags it is handed (the expert placement, the MoE cache,
a draft or MTP head: what llama-bench cannot run), waits for /health, and stops that one process when it ends, also
on an error (SIGTERM, which llama-server handles and exits from, so a profiler's buffers are flushed).

How one point is measured. The prompt is `context` copies of the token " the" (tokenized by the server), sent as token
ids, prompt cache off, greedy (temperature 0, top_k 1), ignore_eos, n_predict N. The server reports predicted_n tokens
in predicted_ms; that window is decode only (the prompt is prompt_ms). With speculative decoding it also reports
draft_n and draft_n_accepted, kept as the acceptance. So

    tok/s = predicted_n / predicted_ms

It prints one JSON line per point: {"context", "batch", "tokens", "prompt_tokens", "step_ms", "tok_s_stream",
"tok_s_total", "engine"}. --once N is the capture mode: one request of N tokens and the line {"tokens", "seconds",
"engine"}, seconds the server's predicted_ms.

--prefill is the prefill mode: the prompt is real text (--corpus, the shipped Gutenberg excerpts of data/gate_corpus.json,
tokenized by the server and cut to exactly `context` tokens; repeated from the start only when the corpus is shorter),
sent as token ids with the prompt cache off, one token asked for. One warm-up and --reps timed requests; each must report
prompt_n equal to the context (and cache_n 0 where the server reports it): a prompt the server partly reused is refused.
It prints {"prefill", "context", "prompt_tokens", "samples_ms" (the server's prompt_ms), "wall_samples_ms" (client wall: request sent to reply read), "n_batch", "n_ubatch", "corpus",
"prompt_form", "request_spans_ns" (each timed request's send and reply on the host's CLOCK_MONOTONIC, so a trace can be cut to it),
"engine", and with --memory-cmd "memory": {"loaded", "warm"}, the GPU's memory right after load and after the warm-up
request, as that command prints it}. Real text, because an MoE routes random token ids to far more experts than real text and reads ~2x slow.

--prompt-form chat sends the same corpus text as a chat request instead (/v1/chat/completions, one user message,
max_tokens 1, the prompt cache off): what a chat client sends. The server wraps it in the model's chat template, and for a
model it checkpoints it ends a batch where the user message starts (server-context.cpp, the user-start break), so a chat
prompt's first batch is the template's head alone. The text is the corpus's first k tokens, k chosen (through /detokenize,
/apply-template and /tokenize) so the templated prompt is `context` tokens (refused when it cannot be fitted); each
request must report prompt_n equal to it and cache_n 0. The line also gives chat_head_tokens, the template's tokens before the text.
"""
from __future__ import annotations

import argparse, json, pathlib, signal, statistics, subprocess, sys, time, urllib.request


def call(base:str, path:str, body:dict | None = None, timeout:float = 3600.0) -> dict:
  data = json.dumps(body).encode() if body is not None else None
  req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"},
                               method="POST" if body is not None else "GET")
  with urllib.request.urlopen(req, timeout=timeout) as r:
    return json.loads(r.read().decode())


class Server:
  """The llama-server it was given, on a private port, stopped with SIGTERM when the block ends."""

  def __init__(self, argv:list[str], port:int, log:pathlib.Path, ready_s:float) -> None:
    self.base, self.ready_s = f"http://127.0.0.1:{port}", ready_s
    log.parent.mkdir(parents=True, exist_ok=True)
    self.log = open(log, "a")
    self.proc = subprocess.Popen([*argv, "--host", "127.0.0.1", "--port", str(port)], stdout=self.log, stderr=subprocess.STDOUT,
                                 start_new_session=True)  # its own group: a driver killed on timeout takes the server with it (main)

  def __enter__(self) -> "Server":
    deadline = time.time() + self.ready_s
    while time.time() < deadline:
      if self.proc.poll() is not None:  # before /health: another server on the port must never answer for this one
        raise SystemExit(f"llama-server exited {self.proc.returncode} before it was ready; see {self.log.name}")
      try:
        if call(self.base, "/health", timeout=2).get("status") == "ok":
          return self
      except OSError:
        pass
      time.sleep(0.5)
    self.__exit__()
    raise SystemExit(f"llama-server was not ready in {self.ready_s:.0f} s")

  def __exit__(self, *exc) -> None:
    if self.proc.poll() is None:
      self.proc.send_signal(signal.SIGTERM)
      try:
        self.proc.wait(timeout=120)
      except subprocess.TimeoutExpired:
        self.proc.kill()
        self.proc.wait()
    self.log.close()


def generate(base:str, prompt:list[int], n:int) -> dict:
  got = call(base, "/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 0,
                                   "ignore_eos": True, "cache_prompt": False})
  t = got.get("timings") or {}
  if t.get("predicted_n") != n:
    raise SystemExit(f"asked for {n} tokens, the server predicted {t.get('predicted_n')}")
  return t


def engine(t:dict, build:str | None) -> dict:
  out = {"build": build, "server": "llama-server"}
  if t.get("draft_n"):
    out.update(draft_n=t["draft_n"], draft_n_accepted=t.get("draft_n_accepted"),
               acceptance=(t.get("draft_n_accepted") or 0) / t["draft_n"])
  return out


def corpus_ids(base:str, path:str, n:int) -> tuple[list[int], int]:
  """n token ids of the corpus's documents in order (the server's tokenizer), and how many times the corpus was repeated."""
  text = "\n\n".join(d["text"] for d in json.loads(pathlib.Path(path).read_text())["documents"])
  ids = call(base, "/tokenize", {"content": text})["tokens"]
  if not ids:
    raise SystemExit(f"the corpus {path} tokenized to nothing")
  cycles = -(-n // len(ids))
  return (ids * cycles)[:n], cycles


def flag_int(flags:list, name:str, default:int) -> int:
  return int(flags[flags.index(name) + 1]) if name in flags else default


FORMS = ("ids", "chat")


def templated(base:str, text:str) -> tuple[list[dict], int]:
  """The chat messages for text, and how many tokens the server's chat template makes of them."""
  messages = [{"role": "user", "content": text}]
  prompt = call(base, "/apply-template", {"messages": messages})["prompt"]
  return messages, len(call(base, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"])


def user_marker(candidate:str) -> str:
  """The user message's start marker in the text a user message adds before its content, as llama.cpp reads it
  (chat-diff-analyzer.cpp detect_user_start_marker): split into markers (<...>, [...]) and text, drop the leading end
  or close markers and leading blank text, trim."""
  import re
  out, seen = [], False
  for part in (x for x in re.split(r"(<[^<>]*>|\[[^\[\]]*\])", candidate) if x):
    marker = bool(re.fullmatch(r"<[^<>]*>|\[[^\[\]]*\]", part))
    if not seen and ((marker and ("end" in part.lower() or "close" in part.lower())) or (not marker and not part.strip())):
      continue
    seen |= marker
    out.append(part)
  return "".join(out).strip()


def chat_head(base:str) -> int:
  """Where the server ends a chat prompt's first batch: the start of the user message, its role header included. The
  server finds the user message's marker by rendering the template without and with one user message (llama.cpp
  common/chat-diff-analyzer.cpp, detect_user_start_marker; when the template refuses no messages, after a user and an
  assistant turn): the marker is the text the user message adds before its content. The head is the tokens of a
  one-message prompt before that marker."""
  content = "\u2063BOLTBEAM\u2063"
  render = lambda msgs: call(base, "/apply-template", {"messages": msgs, "add_generation_prompt": False})["prompt"]  # noqa: E731
  user = {"role": "user", "content": content}
  try:
    before, after = render([]), render([user])
  except OSError:  # urllib's HTTPError: the template needs a message
    turn = [{"role": "user", "content": "\u2063BB2\u2063"}, {"role": "assistant", "content": "\u2063BB3\u2063"}]
    before, after = render(turn), render([*turn, user])
  n = 0
  while n < min(len(before), len(after)) and before[n] == after[n]:
    n += 1
  marker = user_marker(after[n:after.index(content)])
  pre = (one := render([user]))[:one.index(content)]
  head = pre[:pre.rindex(marker)] if marker and marker in pre else pre
  return len(call(base, "/tokenize", {"content": head, "add_special": True, "parse_special": True})["tokens"])


def chat_prompt(base:str, ids:list[int], n:int) -> tuple[list[dict], int]:
  """A user message of the corpus's first k tokens, k chosen so the templated prompt is n tokens (a few tries: a
  detokenized cut can tokenize back a token longer or shorter). The messages and the templated count."""
  empty, k, best = templated(base, "")[1], n, None
  for _ in range(12):
    messages, count = templated(base, call(base, "/detokenize", {"tokens": ids[:k]})["content"])
    if best is None or abs(count - n) < abs(best[1] - n):
      best = (messages, count)
    if count == n:
      break
    step = round((n - count) * k / max(count - empty, 1))  # the text's tokens per id so far
    k = max(1, k + (step or (1 if count < n else -1)))
  if best[1] != n:  # a prompt of another length is another measurement: refused, as an ids prompt of another length is
    raise SystemExit(f"could not fit the corpus text to a {n}-token chat prompt (closest {best[1]})")
  return best


def prefill(base:str, prompt:list[int] | list[dict], expect:int, form:str = "ids") -> tuple[float, float, list[int], int]:
  """(the server's prompt_ms, the client's wall ms from the request sent to the reply read, [send, reply] on the
  host's monotonic clock in ns, prompt_n) for one cold prompt: token ids to /completion, or chat messages to
  /v1/chat/completions."""
  path, body = (("/completion", {"prompt": prompt, "n_predict": 1}) if form == "ids" else
                ("/v1/chat/completions", {"messages": prompt, "max_tokens": 1}))
  m0, t0 = time.monotonic_ns(), time.perf_counter()
  got = call(base, path, {**body, "temperature": 0, "top_k": 1, "seed": 0, "cache_prompt": False})
  wall, m1 = (time.perf_counter() - t0) * 1e3, time.monotonic_ns()
  t = got.get("timings") or {}
  n = t.get("prompt_n")
  if n != expect or t.get("cache_n", 0) != 0:
    raise SystemExit(f"asked for a cold prompt of {expect} tokens ({form} form), the server read prompt_n {n} with cache_n {t.get('cache_n')}")
  return float(t["prompt_ms"]), wall, [m0, m1], int(n)


def gpu_memory(cmd:str) -> dict:
  """The GPU's memory as the --memory-cmd prints it (JSON), {"error"} when it fails: a reading, never a reason to stop."""
  try:
    out = subprocess.run(json.loads(cmd), capture_output=True, text=True, timeout=60)
    return json.loads(out.stdout.strip().splitlines()[-1])
  except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
    return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--server", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--port", type=int, default=0, help="0: a free port")
  ap.add_argument("--ctx-size", type=int, required=True, help="llama-server -c")
  ap.add_argument("--contexts", required=True)
  ap.add_argument("--tokens", type=int, default=64)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--once", type=int, default=None)
  ap.add_argument("--prefill", action="store_true", help="time a cold prefill of the first --contexts value on real text")
  ap.add_argument("--corpus", default=None, help="--prefill: data/gate_corpus.json")
  ap.add_argument("--prompt-form", default="ids", choices=FORMS, help="--prefill: token ids to /completion, or a chat request")
  ap.add_argument("--flags", default="[]", help="the engine flags, a JSON list")
  ap.add_argument("--log", default="llama-server.log")
  ap.add_argument("--ready-s", type=float, default=1800.0)
  ap.add_argument("--memory-cmd", default=None, help="--prefill: a JSON argv that prints the GPU's memory as JSON; run after load and after the warm-up")
  a = ap.parse_args()
  if a.port == 0:  # a free port of the system's choosing, never a fixed one a stale server may hold
    import socket
    with socket.socket() as s:
      s.bind(("127.0.0.1", 0))
      a.port = s.getsockname()[1]
  stop = lambda *_: sys.exit(143)  # noqa: E731  a SIGTERM to the driver unwinds the server's block
  signal.signal(signal.SIGTERM, stop)
  argv = [a.server, "-m", a.model, "-c", str(a.ctx_size), "-np", "1", *json.loads(a.flags)]
  contexts = [int(x) for x in a.contexts.split(",")]
  with Server(argv, a.port, pathlib.Path(a.log), a.ready_s) as s:
    the = call(s.base, "/tokenize", {"content": " the"})["tokens"][-1]
    build = call(s.base, "/props").get("build_info")
    if a.prefill:
      flags = json.loads(a.flags)
      memory = {"loaded": gpu_memory(a.memory_cmd)} if a.memory_cmd else None
      ids, cycles = corpus_ids(s.base, a.corpus, contexts[0])
      prompt, expect = (ids, len(ids)) if a.prompt_form == "ids" else chat_prompt(s.base, ids, contexts[0])
      prefill(s.base, prompt, expect, a.prompt_form)  # load and warm; the server must read the fitted length
      if memory is not None:
        memory["warm"] = gpu_memory(a.memory_cmd)
      got = [prefill(s.base, prompt, expect, a.prompt_form) for _ in range(a.reps)]
      samples, walls = [g[0] for g in got], [g[1] for g in got]
      if got and {g[3] for g in got} != {expect}:  # --reps 0: the warm-up pass alone (a routing trace reads that one pass)
        raise SystemExit(f"the {a.prompt_form} prompt's prompt_n moved between requests: {[g[3] for g in got]}")
      print(json.dumps({"prefill": True, "context": contexts[0], "prompt_tokens": expect, "samples_ms": samples, "wall_samples_ms": walls,
                        "prompt_form": a.prompt_form, "request_spans_ns": [g[2] for g in got],
                        **({"chat_head_tokens": chat_head(s.base)} if a.prompt_form == "chat" else {}),
                        "n_batch": flag_int(flags, "-b", 2048), "n_ubatch": flag_int(flags, "-ub", 512),
                        "corpus": {"file": pathlib.Path(a.corpus).name, "repeats": cycles}, "engine": {"build": build, "server": "llama-server"},
                        **({"memory": memory} if memory is not None else {})}), flush=True)
      return 0
    if a.once:
      prompt = [the] * contexts[0]
      generate(s.base, prompt, 2)  # load and warm
      t = generate(s.base, prompt, a.once)
      print(json.dumps({"tokens": a.once, "seconds": t["predicted_ms"] / 1000.0, "engine": engine(t, build)}), flush=True)
      return 0
    for context in contexts:
      prompt = [the] * context
      generate(s.base, prompt, 4)  # load and warm
      runs = [generate(s.base, prompt, a.tokens) for _ in range(a.reps)]
      # the first token is sampled from the prompt's logits: predicted_ms holds the other predicted_n - 1 decode steps
      per = statistics.median((t["predicted_n"] - 1) / (t["predicted_ms"] / 1000.0) for t in runs)
      print(json.dumps({"context": context, "batch": 1, "tokens": a.tokens, "prompt_tokens": runs[0].get("prompt_n"),
                        "engine": engine(runs[-1], build), "step_ms": 1000.0 / per, "tok_s_stream": per, "tok_s_total": per}), flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
