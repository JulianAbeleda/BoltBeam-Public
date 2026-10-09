#!/usr/bin/env python3
"""Decode time per step in Ollama, at a fixed context and batch, from Ollama's own API counters.

Any python3 runs this file by path; it uses only the standard library. It starts its own `ollama serve` on a
private port with OLLAMA_MODELS in a folder it is given, imports the GGUF with a one-line Modelfile
(FROM <gguf>), and stops the server when it ends, also on an error.

How one point is measured. B requests are sent at once (OLLAMA_NUM_PARALLEL=B, so the runner decodes them as one
batch). Each has the same prompt of `context` tokens (" the" repeated; Ollama reports the real count in
prompt_eval_count) and num_predict N with greedy sampling. Ollama reports, per request, eval_count tokens
generated in eval_duration ns; that window is decode only (the prompt is in prompt_eval_duration). So

    tok/s per stream = mean over streams of eval_count / eval_duration
    tok/s total      = sum of eval_count / the longest eval_duration (the streams decode in the same steps)

It prints one JSON line per point: {"context", "batch", "tokens", "prompt_tokens", "step_ms", "tok_s_stream",
"tok_s_total"}. --once N is the capture mode: one batch of N tokens and the line {"tokens", "seconds"}, where
seconds is the decode window (eval_duration) of the slowest stream.
"""
from __future__ import annotations

import argparse, json, os, pathlib, signal, statistics, subprocess, sys, threading, time, urllib.request

NAME = "boltbeam-decode"


def call(base:str, path:str, body:dict | None = None, timeout:float = 1800.0) -> dict:
  data = json.dumps(body).encode() if body is not None else None
  req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"},
                               method="POST" if body is not None else "GET")
  with urllib.request.urlopen(req, timeout=timeout) as r:
    text = r.read().decode()
  return json.loads(text.strip().splitlines()[-1])


class Server:
  """`ollama serve` on a private port, stopped with SIGTERM (it stops its runner first) when the block ends."""

  def __init__(self, ollama:str, port:int, models:str, parallel:int, log:pathlib.Path) -> None:
    self.base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_MODELS": models,
           "OLLAMA_NUM_PARALLEL": str(parallel), "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KEEP_ALIVE": "-1"}
    self.log = open(log, "a")
    self.proc = subprocess.Popen([ollama, "serve"], env=env, stdout=self.log, stderr=subprocess.STDOUT)

  def __enter__(self) -> "Server":
    for _ in range(300):
      try:
        call(self.base, "/api/version", timeout=2)
        return self
      except OSError:
        if self.proc.poll() is not None:
          raise SystemExit(f"ollama serve exited {self.proc.returncode}; see {self.log.name}")
        time.sleep(0.2)
    raise SystemExit("ollama serve did not answer in 60 s")

  def children(self) -> list[int]:
    """The server's child processes (its runners), from /proc on Linux; [] elsewhere."""
    out = []
    for task in pathlib.Path(f"/proc/{self.proc.pid}/task").glob("*/children"):
      out += [int(x) for x in task.read_text().split()]
    return out

  def __exit__(self, *exc) -> None:
    # Ollama ends its runner with SIGKILL, and a killed process never flushes a profiler's buffers. The runner is
    # asked to stop first with SIGTERM, which it handles and exits cleanly from; then the server is stopped.
    runners = self.children()
    for pid in runners:
      try:
        os.kill(pid, signal.SIGTERM)
      except ProcessLookupError:
        pass
    deadline = time.time() + 30
    while time.time() < deadline and any(pathlib.Path(f"/proc/{pid}").exists() for pid in runners):
      time.sleep(0.2)
    self.proc.send_signal(signal.SIGTERM)
    try:
      self.proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
      self.proc.kill()
    self.log.close()


def generate(base:str, context:int, batch:int, n:int, num_ctx:int) -> list[dict]:
  prompt = " the" * context
  body = {"model": NAME, "prompt": prompt, "raw": True, "stream": False,
          "options": {"num_predict": n, "temperature": 0, "seed": 0, "num_ctx": num_ctx, "top_k": 1}}
  out: list[dict | Exception] = [None] * batch  # type: ignore[list-item]
  def one(i:int) -> None:
    try:
      out[i] = call(base, "/api/generate", body)
    except Exception as exc:  # reported below with the other streams
      out[i] = exc
  threads = [threading.Thread(target=one, args=(i,)) for i in range(batch)]
  for t in threads: t.start()
  for t in threads: t.join()
  bad = [o for o in out if isinstance(o, Exception)]
  if bad:
    raise SystemExit(f"generate failed: {bad[0]}")
  short = [o.get("eval_count") for o in out if o.get("eval_count") != n]
  if short:
    raise SystemExit(f"asked for {n} tokens per stream, got {short}")
  return out  # type: ignore[return-value]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ollama", required=True)
  ap.add_argument("--gguf", required=True)
  ap.add_argument("--models", required=True, help="OLLAMA_MODELS folder")
  ap.add_argument("--port", type=int, default=11534)
  ap.add_argument("--contexts", default="128")
  ap.add_argument("--batches", default="1")
  ap.add_argument("--tokens", type=int, default=64)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--once", type=int, default=None)
  ap.add_argument("--log", default="ollama-serve.log")
  a = ap.parse_args()
  contexts = [int(x) for x in a.contexts.split(",")]
  batches = [int(x) for x in a.batches.split(",")]
  num_ctx = max(contexts) + max(a.tokens, a.once or 0) + 64
  for batch in (batches[:1] if a.once else batches):
    with Server(a.ollama, a.port, a.models, batch, pathlib.Path(a.log)) as s:
      mf = pathlib.Path(a.models) / "Modelfile"
      mf.write_text(f"FROM {a.gguf}\n")
      env = {**os.environ, "OLLAMA_HOST": s.base.removeprefix("http://")}
      made = subprocess.run([a.ollama, "create", NAME, "-f", str(mf)], env=env, capture_output=True, text=True)
      if made.returncode != 0:
        raise SystemExit(f"ollama create exited {made.returncode}: {made.stderr.strip()[-300:]}")
      if a.once:
        generate(s.base, contexts[0], batch, 2, num_ctx)  # load and warm
        got = generate(s.base, contexts[0], batch, a.once, num_ctx)
        print(json.dumps({"tokens": a.once, "seconds": max(o["eval_duration"] for o in got) / 1e9}), flush=True)
        return 0
      for context in contexts:
        generate(s.base, context, batch, 4, num_ctx)  # load and warm
        stream, total, prompt_tokens = [], [], None
        for _ in range(a.reps):
          got = generate(s.base, context, batch, a.tokens, num_ctx)
          stream.append(statistics.mean(o["eval_count"] / (o["eval_duration"] / 1e9) for o in got))
          total.append(sum(o["eval_count"] for o in got) / (max(o["eval_duration"] for o in got) / 1e9))
          prompt_tokens = prompt_tokens or got[0].get("prompt_eval_count")  # later calls reuse the cached prompt
        per = statistics.median(stream)
        print(json.dumps({"context": context, "batch": batch, "tokens": a.tokens, "prompt_tokens": prompt_tokens,
                          "step_ms": 1000.0 / per, "tok_s_stream": per, "tok_s_total": statistics.median(total)}),
              flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
