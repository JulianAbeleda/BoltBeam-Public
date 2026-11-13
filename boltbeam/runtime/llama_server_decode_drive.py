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

    tok/s = (predicted_n - 1) / predicted_ms

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

--mtp-prompts is MTP's measurement at one draft count (collectors/llama_mtp.py; k is the server's --spec-draft-n-max, 0
no draft head): one user turn through the model's chat template (/apply-template, /tokenize), sent as token ids,
greedy, ignore_eos, --tokens long, at the short depth (each prompt of the set) and the long one (each document of
--long-corpus cut to --long-tokens tokens by the server's tokenizer at a paragraph break, then its task:
runtime/mtp_shared.long_prompts, the same ids at every k). Each timed request waits for the quiet-CPU gate
(mtp_shared.quiet) and must be read cold (prompt_n its length, cache_n 0). One line per prompt and depth in
collectors/mtp.py's row shape, with the token diversity of its context and reply (mtp_shared.diversity): decode tok/s (predicted_n - 1) / predicted_ms, the
server's own predicted_per_second (server-common.h n_gen_tps: the first token is free); the acceptance is the server's
spec-decode counters (--metrics) read before and after the request, as vLLM's are: verify steps, draft tokens, accepted
tokens and accepted per position, checked against the reply's draft_n and draft_n_accepted; span_ns, the request sent and
its reply read on the host's CLOCK_MONOTONIC (a trace is cut to it: collectors/time_budget.request_decode_budget), and
with --first-token-mark (each request streamed) first_token_ns, when its first generated token arrived, and with
--stream-control the same request's decode tok/s unstreamed right after. Then one line naming the engine, with the
corpus's own diversity on this tokenizer (real_text), which collectors/mtp.guard holds the rows to, and where the server
put its weights when --residency-probe read them (residency).
"""
from __future__ import annotations

import argparse, http.client, json, os, pathlib, re, signal, statistics, subprocess, sys, time, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # run by path or imported: the MTP drivers' shared module
from mtp_shared import delta, diversity, long_prompts, quiet, reference, sha  # noqa: E402


def call(base:str, path:str, body:dict | None = None, timeout:float = 3600.0, text:bool = False):
  data = json.dumps(body).encode() if body is not None else None
  req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"},
                               method="POST" if body is not None else "GET")
  with urllib.request.urlopen(req, timeout=timeout) as r:
    got = r.read().decode()
    return got if text else json.loads(got)


class Server:
  """The llama-server it was given, on a private port, in a process group of its own, stopped (the whole group: SIGTERM,
  then SIGKILL after 120 s) on every way out of this driver it can see: the block's end, an error, SIGTERM, SIGINT or
  --deadline-s (main turns them into exits), also while the server is still loading (__enter__ stops it before it
  raises). A SIGKILL of this driver cannot be caught: pgid_file names the group, and the caller stops it
  (collectors/mtp.stop_group)."""

  def __init__(self, argv:list[str], port:int, log:pathlib.Path, ready_s:float, pgid_file:str | None = None) -> None:
    self.base, self.ready_s, self.path = f"http://127.0.0.1:{port}", ready_s, log
    log.parent.mkdir(parents=True, exist_ok=True)
    self.log = open(log, "a")
    self.proc = subprocess.Popen([*argv, "--host", "127.0.0.1", "--port", str(port)], stdout=self.log, stderr=subprocess.STDOUT,
                                 start_new_session=True)  # its group's id is its pid
    if pgid_file:
      pathlib.Path(pgid_file).write_text(str(self.proc.pid))

  def __enter__(self) -> "Server":
    try:
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
      raise SystemExit(f"llama-server was not ready in {self.ready_s:.0f} s")
    except BaseException:  # a signal or a refusal while it loads: the with block's __exit__ never runs
      self.__exit__()
      raise

  def __exit__(self, *exc) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
      signal.signal(sig, signal.SIG_IGN)  # a second signal must not cut the stop short
    for sig, wait in ((signal.SIGTERM, 120), (signal.SIGKILL, None)):
      try:
        os.killpg(self.proc.pid, sig)
      except ProcessLookupError:
        break
      try:
        self.proc.wait(timeout=wait)
      except subprocess.TimeoutExpired:
        continue
    self.log.close()


  def said(self) -> str:
    self.log.flush()
    return self.path.read_text(errors="replace") if self.path.is_file() else ""


WORDS = pathlib.Path(__file__).resolve().parents[1] / "data" / "plain_words.json"
NO_FIT = "common_fit_params: failed to fit params to free device memory"  # common/fit.cpp: its projection misses the free-memory target
OUT_OF_MEMORY = re.compile(r"CUDA error: (out of memory|the resource allocation failed)|cudaMalloc failed|failed to allocate CUDA|ErrorOutOfDeviceMemory|hipErrorOutOfMemory")
FIT_END = re.compile(r"(successfully fit params|failed to fit params|encountered an error while trying to fit params)")


def words() -> dict:
  return json.loads(WORDS.read_text())["llama_server"]


def where(k:int, ctx:int) -> dict:
  return {"k": k, "ctx": ctx, "with": words()["with_head"] if k else ""}


def failure(log:str, k:int, ctx:int) -> str:
  """Why the server dropped a request, from its log, in words: out of GPU memory (cuBLAS's handle, or another allocation),
  else the log's last error line."""
  w = words()
  if OUT_OF_MEMORY.search(log):
    what = w["cublas"] if "cublasCreate" in log else w["allocation"]
    return w["out_of_memory"].format(what=what, **where(k, ctx))
  errors = [ln.strip() for ln in log.splitlines() if " E " in ln or "error" in ln.lower()]
  return w["stopped"].format(last=errors[-1] if errors else "nothing", **where(k, ctx))


def fit_outcome(log:str) -> dict:
  """llama.cpp's own fit of a setup to the GPU's free memory (common/fit.cpp, logged at -lv 4), as the server logged it:
  {"fits", "context" (the context it reduced to; None: the model's own context fits), "why"}. Fitting only by moving
  layers off the GPU is no fit here: a measurement keeps every layer on it."""
  if m := re.search(r"(?:failed to fit params to free device memory|encountered an error while trying to fit params to free device memory): (.*)", log):
    return {"fits": False, "context": None, "why": m.group(1).strip()}
  if "no changes needed" in log:
    return {"fits": True, "context": None, "why": "the model's own context fits"}
  reduced = re.findall(r"context size reduced from (\d+) to (\d+)", log)
  if reduced and "entire model can be fit by reducing context" in log:
    return {"fits": True, "context": int(reduced[-1][1]), "why": f"the context reduced from {int(reduced[-1][0]):,} to {int(reduced[-1][1]):,}"}
  return {"fits": False, "context": None, "why": "only with layers moved off the GPU, even at the smallest context"}


def fit_probe(s:"Server") -> dict:
  """The server's fit, read from its log as soon as it is logged (it runs before the weights load), then the server stopped."""
  deadline, log = time.time() + s.ready_s, ""
  while time.time() < deadline and not FIT_END.search(log := s.said()):
    if s.proc.poll() is not None:
      break
    time.sleep(0.2)
  s.__exit__()
  return fit_outcome(log)


def body(prompt:list[int], n:int) -> dict:
  return {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 0, "ignore_eos": True, "cache_prompt": False, "return_tokens": True}


def complete(base:str, prompt:list[int], n:int, cold:bool = False) -> dict:
  """The server's whole reply to a greedy, cold, ignore_eos request for n tokens, its token ids with it. cold: the
  server must have read the whole prompt (prompt_n its length, cache_n 0), as a prefill is held to."""
  got = call(base, "/completion", body(prompt, n))
  checked(got.get("timings") or {}, prompt, n, cold)
  return got


def stream(base:str, path:str, doc:dict, timeout:float = 3600.0):
  """The server's streamed reply (server-sent events), one dict per data line, as each arrives."""
  req = urllib.request.Request(base + path, data=json.dumps(doc).encode(), headers={"Content-Type": "application/json"}, method="POST")
  with urllib.request.urlopen(req, timeout=timeout) as r:
    for raw in r:
      line = raw.decode().strip()
      if line.startswith("data: ") and line[6:] != "[DONE]":
        yield json.loads(line[6:])


def complete_marked(base:str, prompt:list[int], n:int) -> tuple[dict, int | None]:
  """complete(cold=True), streamed: the reply and the host's CLOCK_MONOTONIC when the first generated token arrived (the
  server sends it once it has read the prompt's last result and sampled it: every forward the prompt took is before it)."""
  mark, toks, last = None, [], {}
  for d in stream(base, "/completion", {**body(prompt, n), "stream": True}):
    if mark is None and (d.get("tokens") or d.get("content")):
      mark = time.monotonic_ns()
    toks += d.get("tokens") or []
    last = d
  checked(last.get("timings") or {}, prompt, n, True)
  return {"timings": last["timings"], "tokens": toks}, mark


BUFFER = re.compile(r"(\S+) model buffer size = +([0-9]+)\.?([0-9]*) MiB")  # llama-model.cpp load_tensors: "%12s model buffer size = %8.2f MiB"
PROBE_VERBOSITY = 4  # the -lv a load must run at to print that line: llama's INFO maps to LOG_LEVEL_TRACE, 4 (common/log.cpp
# common_log_get_verbosity), and common_log_default_callback prints only at or under the threshold, 3 by default (LOG_DEFAULT_LLAMA)


def residency(text:str) -> dict:
  """Where the server put its weights, from a load log that prints them (PROBE_VERBOSITY): each buffer's MiB of the
  model's weights summed over its lines per buffer name (CUDA0, CPU_Mapped, ...: llama.cpp prints one line per buffer,
  so a model mapped in two ranges has two CPU_Mapped lines), the target model's apart from the draft head's (its lines
  come after common_speculative_init's "loading draft model"), and each part's rounding: half the last printed digit,
  per line. text: the server's log as Server.said reads it. {"target": None} when it holds no such line."""
  cut = text.find("loading draft model")
  def read(t:str) -> tuple[dict[str, float] | None, float]:
    found = BUFFER.findall(t)
    out: dict[str, float] = {}
    for name, whole, frac in found:
      out[name] = out.get(name, 0.0) + float(f"{whole}.{frac or 0}")
    return (out or None), sum(0.5 * 10 ** -len(frac) for _, _, frac in found)
  (target, t_round), (draft, d_round) = (read(text), (None, 0.0)) if cut < 0 else (read(text[:cut]), read(text[cut:]))
  return {"target": target, "draft": draft, "rounding_mib": {"target": t_round, "draft": d_round}}


def probe_residency(argv:list[str], a) -> dict:
  """--residency-probe: a load of its own, untimed, so the measured server runs at the default verbosity; its log a fresh
  file, so a reused --log never answers for this run."""
  probe = pathlib.Path(a.log).with_suffix(".residency.log")
  probe.unlink(missing_ok=True)
  with Server([*argv, "-lv", str(PROBE_VERBOSITY)], a.port, probe, a.ready_s, a.pgid_file) as s:
    text = s.said()  # read as the fit probe and a dropped request read it: one place reads the server's log
  return {**residency(text), "how": f"a load of the same command at -lv {PROBE_VERBOSITY} before the measured one"}


def checked(t:dict, prompt:list[int], n:int, cold:bool) -> None:
  if t.get("predicted_n") != n:
    raise SystemExit(f"asked for {n} tokens, the server predicted {t.get('predicted_n')}")
  if cold and (t.get("prompt_n") != len(prompt) or t.get("cache_n", 0) != 0):
    raise SystemExit(f"asked for a cold prompt of {len(prompt)} tokens, the server read prompt_n {t.get('prompt_n')} with cache_n {t.get('cache_n')}")


def generate(base:str, prompt:list[int], n:int) -> dict:
  return complete(base, prompt, n)["timings"]


SPEC = {"llamacpp:spec_decode_num_drafts_total": "drafts", "llamacpp:spec_decode_num_draft_tokens_total": "draft_tokens",
        "llamacpp:spec_decode_num_accepted_tokens_total": "accepted_tokens"}  # llama-server --metrics, server-task.cpp to_metrics
PER_POSITION = re.compile(r'llamacpp:spec_decode_num_accepted_tokens_per_pos_total\{position="(\d+)"\}')


def spec_counters(base:str) -> dict:
  """The server's cumulative spec-decode counters (/metrics, --metrics), as vLLM's are read: the verify steps (drafts),
  draft tokens, accepted tokens (a replayed checkpoint's step counted once, server-context.cpp) and accepted per draft
  position. A request's are added when its slot is released, before the next task is read, so a read after a reply holds it."""
  out, per = {v: 0 for v in SPEC.values()}, {}
  for line in call(base, "/metrics", text=True).splitlines():
    name, _, value = line.rpartition(" ")
    if name in SPEC:
      out[SPEC[name]] = int(float(value))
    elif m := PER_POSITION.fullmatch(name):
      per[int(m.group(1))] = int(float(value))
  return {**out, "accepted_per_position": [per[i] for i in sorted(per)]}


def mtp_run(base:str, a, build:str | None, flags:list[str]) -> None:
  """--mtp-prompts: one draft count's rows (the module docstring) and the engine line."""
  chat = lambda text: template_ids(base, [{"role": "user", "content": text}])  # noqa: E731
  ids = {p["id"]: chat(p["text"]) for p in json.loads(pathlib.Path(a.mtp_prompts).read_text())["prompts"]}
  generate(base, list(range(1000, 1064)), 32)  # load and warm
  messages, docs = long_prompts(a.long_corpus, a.long_tokens, lambda text: call(base, "/tokenize", {"content": text})["tokens"],
                                lambda x: call(base, "/detokenize", {"tokens": x})["content"])
  long_ids, real = {d: chat(m) for d, m in messages.items()}, reference(docs, a.tokens)
  for depth, source in (("short", ids), ("long", long_ids)):
    for pid, prompt in source.items():
      cpu = quiet(a.quiet_s)
      before = spec_counters(base)
      m0, t0 = time.monotonic_ns(), time.perf_counter()
      got, mark = complete_marked(base, prompt, a.tokens) if a.first_token_mark else (complete(base, prompt, a.tokens, cold=True), None)
      wall, t, m1 = time.perf_counter() - t0, got["timings"], time.monotonic_ns()
      spec = delta(spec_counters(base), before)
      if a.k and not spec["drafts"]:
        raise SystemExit(f"{pid} {depth}: k={a.k} and the server drafted nothing: the draft head did not run")
      if (spec["draft_tokens"], spec["accepted_tokens"]) != (t.get("draft_n", 0), t.get("draft_n_accepted", 0)):
        raise SystemExit(f"{pid} {depth}: /metrics counted {spec['draft_tokens']} drafted, {spec['accepted_tokens']} accepted; the reply "
                         f"{t.get('draft_n')}, {t.get('draft_n_accepted')}: another request ran between the reads")
      control = None
      if a.first_token_mark and a.stream_control:  # the same request unstreamed, right after: what streaming costs the decode
        c = complete(base, prompt, a.tokens, cold=True)["timings"]
        control = {"decode_tok_s": (c["predicted_n"] - 1) / (c["predicted_ms"] / 1000.0), "how": "the same request unstreamed, right after"}
      reply = list(got["tokens"])
      print(json.dumps({"k": a.k, "prompt": pid, "depth": depth, "context": len(prompt), "tokens": t["predicted_n"],
                        "decode_tok_s": (t["predicted_n"] - 1) / (t["predicted_ms"] / 1000.0), "wall_s": wall, "spec": spec, "cpu": cpu,
                        "prompt_sha": sha(prompt), "token_ids_sha": sha(reply), "span_ns": [m0, m1],
                        **({"first_token_ns": mark, "streamed": True, "unstreamed": control} if a.first_token_mark else {}),
                        "text": {"context": diversity(prompt, a.tokens), "reply": diversity(prompt[-a.tokens:] + reply, a.tokens)}, "reply_ids": reply}), flush=True)
  print(json.dumps({"engine": {"llama.cpp": build, "server": "llama-server", "k": a.k, "ctx_size": a.ctx_size, "flags": flags, "real_text": real,
                               "residency": a.residency_read}}), flush=True)


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


def template_ids(base:str, messages:list[dict]) -> list[int]:
  """The token ids the server's chat template makes of the messages, as it reads a chat request."""
  prompt = call(base, "/apply-template", {"messages": messages})["prompt"]
  return call(base, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]


def templated(base:str, text:str) -> tuple[list[dict], int]:
  """The chat messages for text, and how many tokens the server's chat template makes of them."""
  messages = [{"role": "user", "content": text}]
  return messages, len(template_ids(base, messages))


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
  ap.add_argument("--contexts", default="", help="the decode depths (not with --mtp-prompts, whose prompts set them)")
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
  ap.add_argument("--mtp-prompts", default=None, help="MTP's measurement on this boltbeam.mtp_prompts.v1 set (the module docstring)")
  ap.add_argument("--k", type=int, default=0, help="--mtp-prompts: the draft count the flags set (0: no draft head)")
  ap.add_argument("--long-tokens", type=int, default=0, help="--mtp-prompts: the long depth's document tokens; 0: short only")
  ap.add_argument("--long-corpus", default=None, help="--mtp-prompts: the long depth's documents, a boltbeam.mtp_long_corpus.v1 file")
  ap.add_argument("--first-token-mark", action="store_true", help="--mtp-prompts: stream each timed request and record first_token_ns, "
                  "when its first generated token arrived (the time budget's prompt mark); the same for a traced and an untraced run")
  ap.add_argument("--stream-control", action="store_true", help="--first-token-mark: run each timed request again unstreamed right "
                  "after and record its decode tok/s (unstreamed), so streaming's own cost is measured")
  ap.add_argument("--residency-probe", action="store_true", help="--mtp-prompts: load the server once first with the same flags at "
                  "the verbosity its weight buffers are printed at (PROBE_VERBOSITY), record where the weights went (residency), stop it, then measure")
  ap.add_argument("--quiet-s", type=float, default=600.0, help="--mtp-prompts: how long to wait for no build or test runner")
  ap.add_argument("--deadline-s", type=int, default=0, help="exit (stopping the server) after this many seconds; 0: none")
  ap.add_argument("--pgid-file", default=None, help="write the server's process group here, for a caller to stop it if this driver is killed")
  ap.add_argument("--fit-probe", type=int, default=0, help="the smallest context: start the server with its own fit on (no -c, no -ngl, -fitc this), print "
                  "what it fits ({\"fit\"}, fit_outcome) and stop it before the weights load")
  a = ap.parse_args()
  if a.port == 0:  # a free port of the system's choosing, never a fixed one a stale server may hold
    import socket
    with socket.socket() as s:
      s.bind(("127.0.0.1", 0))
      a.port = s.getsockname()[1]
  stop = lambda *_: sys.exit(143)  # noqa: E731  SIGTERM, SIGINT and --deadline-s unwind the server's block (Server)
  for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
    signal.signal(sig, stop)
  signal.alarm(a.deadline_s)
  if a.fit_probe:
    s = Server([a.server, "-m", a.model, "-np", "1", "-fitc", str(a.fit_probe), "-lv", str(PROBE_VERBOSITY), *json.loads(a.flags)], a.port, pathlib.Path(a.log), a.ready_s, a.pgid_file)
    print(json.dumps({"fit": fit_probe(s)}), flush=True)
    return 0
  argv = [a.server, "-m", a.model, "-c", str(a.ctx_size), "-np", "1", *json.loads(a.flags)]
  contexts = [int(x) for x in a.contexts.split(",") if x]
  if bool(contexts) == bool(a.mtp_prompts) or (a.mtp_prompts and not a.long_corpus):
    ap.error("give --contexts or --mtp-prompts, and --mtp-prompts with --long-corpus: its real text is what replies are held to")
  a.residency_read = {"target": None, "why": "not probed (--residency-probe)"}
  if a.mtp_prompts and a.residency_probe:
    a.residency_read = probe_residency(argv, a)
  with Server(argv, a.port, pathlib.Path(a.log), a.ready_s, a.pgid_file) as s:
    the = call(s.base, "/tokenize", {"content": " the"})["tokens"][-1]
    build = call(s.base, "/props").get("build_info")
    if a.mtp_prompts:
      if NO_FIT in s.said():  # its own projection misses its free-memory target: the first request would run out
        raise SystemExit(words()["no_fit"].format(**where(a.k, a.ctx_size)))
      try:
        mtp_run(s.base, a, build, json.loads(a.flags))
      except (OSError, http.client.HTTPException) as exc:  # the connection dropped: the log says why
        raise SystemExit(failure(s.said(), a.k, a.ctx_size)) from exc
      return 0
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
