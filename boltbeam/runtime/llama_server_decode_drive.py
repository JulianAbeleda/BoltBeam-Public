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
  ap.add_argument("--flags", default="[]", help="the engine flags, a JSON list")
  ap.add_argument("--log", default="llama-server.log")
  ap.add_argument("--ready-s", type=float, default=1800.0)
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
