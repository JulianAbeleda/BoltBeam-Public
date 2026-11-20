"""One load probe: does this server command load, and what GPU memory does it leave?

    boltbeam load-probe --server-argv 'VAR=value ... SERVER ARGS' --out DIR [--session S.json --turns 3]

The server starts as session-ab starts an arm (adapters/llama_server_control.serve_argv: its own process group, a free
loopback port) and is ready when its own /health answers 200, never when its log says a word. A load that fails is
the process's exit status, as the OS returned it (a signal's negative number), or no /health within --timeout (a hang:
its group is killed). Why it failed is read from the log only for the one fact no structured signal carries, out of
memory, by the engine's own row (data/engine_adapters.json out_of_memory, tested against the pinned source). The GPU's
free memory is read from its driver (NVML, the amdgpu sysfs): before the start, at ready, and its lowest over the
load and the optional short session, which runs as session-ab plays its turns and is stopped by its hang guard. The
record (FILE, boltbeam.spend_probe.v1) is a measured probe row tune --measured reads; a probe with no session has no
lowest of a session, so it says whether the setup loads and is not a memory anchor.

--stress (after the session span): what a user's next prompts add to the memory a session left, the runtime headroom a
fit must keep. Prompt lengths sweep 1 token to what the context holds, crossing each chunk boundary (stress_lengths: the
engine's own batch and chunk sizes), each sent as a new conversation, resumed with a follow-up of the same length
and scored as a continuation with logprobs (a 256k gate's scored requests grew 22 MiB where plain prompts grew none), round after round until a round lowers the free memory no further (a plateau) or ROUNDS end. stress_growth_mib:
the session's lowest free less the lowest over the stress (what tune adds to each fit's margin)."""
from __future__ import annotations

import os
import pathlib
import re
import shlex
import signal
import time
from typing import Any, Callable

from boltbeam.collectors import session_ab as sab
from boltbeam.vocab import SCHEMA_SPEND_PROBE as PROBE

FILE = "load_probe.json"
ROUNDS = 4  # stress rounds at most: each new prompt shape can add a graph instance; a round adding none is the plateau
SPAN = 64  # words either side of a chunk boundary: more than a chat template's own tokens
SCORED = {"logprobs": True, "top_logprobs": 1, "samplers": ["temperature"]}  # as a scoring client sends (session-gate): its sampler chain differs
VARIANTS = {"temperature": {"temperature": 0.7}, "top_p": {"temperature": 0.7, "top_p": 0.9},
            "top_k_min_p": {"temperature": 0.7, "top_k": 40, "min_p": 0.05}, "repeat_penalty": {"repeat_penalty": 1.1},
            "json_schema": {"response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}},
            "grammar": {"grammar": 'root ::= "a"+'}, "logprobs_5": {"logprobs": True, "top_logprobs": 5}}  # the settings clients send: each can rebuild the sampler's buffers
STRESS_REPLY = 32  # reply tokens a stress request decodes (the draft's shapes run too)
INTERVAL_S = 0.25  # the GPU memory sampler: a peak shorter than this between two samples is not seen
ADAPTERS = pathlib.Path(__file__).resolve().parents[1] / "data" / "engine_adapters.json"


def _words(key:str, **kw:Any) -> str:
  from boltbeam.report.plain import words
  return words()["load_probe"][key].format(**kw)


def out_of_memory(engine:str | None, log:str) -> dict[str, Any]:
  """Whether the engine's log says a GPU allocation failed, by its adapter row (engine None: any engine's row, for a
  record that does not name its engine): {"out_of_memory", "line"}; None when there is no row (unknown, not no)."""
  import json
  rows = json.loads(ADAPTERS.read_text())["engines"]
  row = "|".join(f"(?:{r['out_of_memory']})" for e, r in rows.items() if r.get("out_of_memory") and engine in (None, e))
  if not row:
    return {"out_of_memory": None, "line": None}
  hit = next((ln.strip() for ln in log.splitlines() if re.search(row, ln)), None)
  return {"out_of_memory": hit is not None, "line": hit[-300:] if hit else None}


def cuda_graphs(engine:str | None, log:str, env:dict[str, str]) -> dict[str, Any] | None:
  """The CUDA graphs a run built, from the engine's stats lines (its adapter row graph_stats; the line is printed only
  with the stats switch on): per backend context the most chunks one graph took (highest chunk + 1, which the backend
  sizes as ceil(nodes / chunk size)), their sum (chunks: what a cap on graph keys bounds per key) and the chunk size
  they were split at. None: no row for the engine, or no stats line (the switch was off)."""
  import json
  row = (json.loads(ADAPTERS.read_text())["engines"].get(engine or "") or {}).get("graph_stats")
  if not row:
    return None
  most: dict[str, int] = {}
  for m in re.finditer(row["line"], log):
    most[m["ctx"]] = max(most.get(m["ctx"], 0), int(m["chunk"]) + 1)
  if not most:
    return None
  size = env.get(row["chunk_env"]) or os.environ.get(row["chunk_env"])
  per = sorted(most.values(), reverse=True)
  return {"contexts": per, "chunks": sum(per), "chunk_size": int(size) if size and size.isdigit() else row["chunk_default"]}


def machine_id() -> str:
  """This machine, as a record names it: a hash of its host name (records travel; the name itself is not written), so
  a reader knows a record measured on itself (workflow/spend.machine_shift)."""
  import hashlib
  import socket
  return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]


def exit_words(code:int | None) -> dict[str, Any]:
  """An exit status as the OS returned it: the code, and the signal's name when a signal ended the process."""
  if code is None:
    return {"code": None, "signal": None}
  return {"code": code, "signal": signal.Signals(-code).name if code < 0 and -code in signal.valid_signals() else None}


def free_mib(source) -> float | None:
  """The GPU's free memory now, MiB, from its driver (total less used); None with no source."""
  used, total = source.memory_mib() if source is not None else (None, None)
  return round(total - used, 1) if used is not None and total is not None else None


def low_between(tele, total:float | None, t0:float, t1:float) -> float | None:
  """The lowest free memory sampled in [t0, t1], MiB."""
  used = [x["mem_mib"] for x in list(tele.samples) if x.get("mem_mib") is not None and t0 <= x["t"] <= t1]
  return round(total - max(used), 1) if used and total else None


def stress_lengths(n_batch:int, n_ubatch:int, cap:int) -> list[int]:
  """Prompt lengths in filler words (about a token each): powers of two below the chunk, each multiple of the chunk up to
  twice the batch with neighbours SPAN under and over it (every chunk boundary crossed whatever the chat template adds),
  then doubling from four batches to cap, what the context holds (a resumed request carries two)."""
  out = {1 << i for i in range(max(n_ubatch, 1).bit_length())}
  for m in range(n_ubatch, 2 * n_batch + n_ubatch + 1, n_ubatch):
    out |= {m - SPAN, m, m + SPAN}
  m = 4 * n_batch
  while m <= cap:  # then doubling to what the context holds: a long paste after a short chat (a 4.6k-token switch grew 22 MiB at 256k)
    out.add(m)
    m *= 2
  return sorted(x for x in out if 0 < x <= cap)


def stress(chat:Callable, url:str, lengths:list[int], low:Callable[[float, float], float | None], say:Callable[[str], None],
           settle:Callable[[], None] = lambda: time.sleep(2 * INTERVAL_S)) -> dict[str, Any]:
  """New, resumed and scored (logprobs) conversations at each length, round after round, until a round's own lowest free is no lower than
  the round before's (a plateau). Each request's processed prompt tokens (the server's prompt_n) are kept: the record
  says which chunk boundaries were crossed, not the lengths asked."""
  rounds, seen = [], []
  for r in range(ROUNDS):
    t0 = time.monotonic()
    for i, n in enumerate(lengths):
      first = [{"role": "user", "content": f"r{r} c{i}" + " a" * n}]  # a new conversation, never in the prompt cache
      got = chat(url, first, STRESS_REPLY)
      again = chat(url, first + [{"role": "assistant", "content": got["text"]}, {"role": "user", "content": " b" * n}], STRESS_REPLY)  # resumed
      probs = chat(url, first + [{"role": "assistant", "content": got["text"]}], 1, **SCORED)  # a scored continuation
      seen += [int((g.get("timings") or {}).get("prompt_n") or 0) for g in (got, again, probs)]
    settle()  # the sampler catches the last request's allocation
    f = low(t0, time.monotonic())
    rounds.append({"requests": 3 * len(lengths), "free_low_mib": f})
    say(_words("stress_round", r=r + 1, free=f))
    if len(rounds) > 1 and f is not None and rounds[-2]["free_low_mib"] is not None and f >= rounds[-2]["free_low_mib"]:
      break
  t0, n, varied = time.monotonic(), lengths[len(lengths) // 2], {}
  for k, extra in VARIANTS.items():  # one request each after the rounds: a client changing its sampler settings
    try:
      chat(url, [{"role": "user", "content": f"v {k}" + " a" * n}], STRESS_REPLY, **extra)
      varied[k] = "ok"
    except RuntimeError as exc:  # the server refused this setting (an HTTP error): said, the rest still run
      varied[k] = str(exc)[-120:]
  settle()
  plateau = len(rounds) > 1 and rounds[-1]["free_low_mib"] is not None and rounds[-2]["free_low_mib"] is not None and \
    rounds[-1]["free_low_mib"] >= rounds[-2]["free_low_mib"]
  return {"lengths": lengths, "rounds": rounds, "plateau": plateau, "prompt_n": sorted(set(seen)),
          "variants": varied, "variants_free_low_mib": low(t0, time.monotonic())}


def run(*, out:str, server_argv:str, engine:str = "llama.cpp", session:str | None = None, turns:str | None = None,
        history:str = "recorded", timeout_s:float = 600.0, name:str | None = None, say:Callable[[str], None] = print, stress_on:bool = False,
        machine:Callable[[], tuple[Any, str | None]] | None = None, held:Callable[[str], dict] | None = None) -> dict[str, Any]:
  """Start the server once, read whether it loaded and its free memory, run the session span if given, stop it: the record."""
  from boltbeam.adapters.llama_server_control import ServerExited, serve_argv
  from boltbeam.collectors import gpu_telemetry
  from boltbeam.collectors.providers import gpu_free
  from boltbeam.collectors.replay import _sha256, _version, engine as engine_of
  env, argv = sab.split_env(shlex.split(server_argv.replace("\\\n", " ")))
  if not argv:
    raise sab.Refused(sab._words("bad_server"))
  sess = span = None
  if session:
    sess, _ = sab.read_session(session, history)
    span = sab.parse_span(turns or ("3" if len(sess["turns"]) >= 3 else None), len(sess["turns"]))
  outdir = pathlib.Path(out).expanduser()
  if (outdir / FILE).exists():
    raise sab.Refused(_words("exists", path=outdir / FILE))
  outdir.mkdir(parents=True, exist_ok=True)
  source, backend = (machine or sab._machine)()
  if backend and not (g := (held or gpu_free)(backend))["free"]:  # another program's memory would read as this setup's
    raise sab.Refused(_words("held", why=g["reason"]))
  binary = pathlib.Path(argv[0]).expanduser()
  log = outdir / "server.log"
  rec: dict[str, Any] = {"schema": PROBE, "name": name or outdir.name, "argv": argv, "env": env, "engine": engine,
                         "loaded": False, "session": None, "turns": span, "start": sab.now_words(),
                         "binary": {"path": str(binary), "sha256": _sha256(binary) if binary.is_file() else None,
                                    "version": _version(binary) if binary.is_file() else None},
                         "free_before_mib": free_mib(source), "machine_id": machine_id(), "free_ready_mib": None, "sampled_every_s": INTERVAL_S,
                         "log": str(log), "failure": None}
  proc: list[Any] = []
  say(_words("start", name=rec["name"]))
  with gpu_telemetry.opened(source or gpu_telemetry.Unsampled(sab._words("no_gpu")), 0, INTERVAL_S) as tele:
    try:
      with serve_argv(argv, log_path=log, startup_timeout_s=timeout_s, environ={**os.environ, **env}, started=proc.append) as (url, command, load_s):
        rec.update(loaded=True, command=command, load_s=round(load_s, 2), free_ready_mib=free_mib(source), ready_t=time.monotonic())
        say(_words("ready", s=load_s, free=rec["free_ready_mib"]))
        if sess is not None:
          try:
            eng = engine_of(engine)
            eng["chat"](url, *sab.WARMUP)
            rows = sab.play(eng, url, sess, None, None, history, proc[0].pid, tele, say, span)
            rec["session"] = len(rows) == span[1] - span[0] + 1
          except Exception as exc:  # noqa: BLE001  the server died or refused mid-session: said, the probe still records
            rec["session"] = False
            rec["failure"] = {"kind": "session", "why": str(exc)[-300:], "exit": exit_words(proc[0].poll())}
          if stress_on and rec["session"]:
            time.sleep(2 * INTERVAL_S)
            rec["stress_t0"] = time.monotonic()  # the session's lowest is every sample after ready and before this
            tot = source.memory_mib()[1] if source is not None else None
            try:
              st = eng["settings"](argv, log.read_text(errors="replace") if log.is_file() else "")
              if not st.get("n_batch") or not st.get("n_ubatch"):
                raise ValueError(_words("stress_no_chunk", engine=engine))
              cap = (int(st["n_ctx"]) - 4 * STRESS_REPLY) // 2 if st.get("n_ctx") else 1 << 30
              rec["stress"] = stress(eng["chat"], url, stress_lengths(int(st["n_batch"]), int(st["n_ubatch"]), cap),
                                     lambda a, b: low_between(tele, tot, a, b), say)
            except Exception as exc:  # noqa: BLE001  the server died under the stress (what it probes): kept with its cause
              rec["stress"] = {"failure": {"kind": "stress", "why": str(exc)[-300:], "exit": exit_words(proc[0].poll()),
                                           **out_of_memory(engine, log.read_text(errors="replace") if log.is_file() else "")}}
        rec["exit"] = sab.stop(proc[0], load_s)  # the hang guard: tearing down never needs longer than bringing up took
    except ServerExited as exc:
      rec["failure"] = {"kind": "exited", "exit": exit_words(exc.returncode)}
    except TimeoutError:
      rec["failure"] = {"kind": "not_ready", "timeout_s": timeout_s, "why": _words("not_ready", s=timeout_s)}
    samples, total = list(tele.samples), (source.memory_mib()[1] if source is not None else None)
  frees = [total - s["mem_mib"] for s in samples if s.get("mem_mib") is not None and total] + \
          [x for x in (rec["free_ready_mib"],) if x is not None]
  rec["free_min_mib"] = round(min(frees), 1) if frees else None
  ready_t, cut = rec.get("ready_t"), rec.pop("stress_t0", None)
  if cut is not None:  # a stressed probe's own lowest is its session's: the stress's lowest is its own field
    stressed = [total - s["mem_mib"] for s in samples if s["t"] >= cut and s.get("mem_mib") is not None and total]
    samples = [s for s in samples if s["t"] < cut]
    frees = [total - s["mem_mib"] for s in samples if s.get("mem_mib") is not None and total] + [x for x in (rec["free_ready_mib"],) if x is not None]
    rec["free_min_mib"] = round(min(frees), 1) if frees else None
  after = [total - s["mem_mib"] for s in samples if ready_t and s["t"] >= ready_t and s.get("mem_mib") is not None and total]
  rec["free_min_after_ready_mib"] = round(min(after + [rec["free_ready_mib"]]), 1) if rec["free_ready_mib"] is not None else None
  rec["memory_total_mib"] = round(total, 1) if total else None
  if cut is not None and rec.get("stress") and after and stressed:
    rec["stress_free_min_mib"] = round(min(stressed), 1)
    rec["stress_growth_mib"] = round(max(0.0, min(after + [rec["free_ready_mib"]]) - min(stressed)), 1)  # beyond the session's lowest
    rec["stress"]["lower_bound"] = bool(rec["stress"].get("failure")) or not rec["stress"].get("plateau")
  if rec["failure"]:
    rec["failure"].update(out_of_memory(engine, log.read_text(errors="replace") if log.is_file() else ""))
  if log.is_file() and (cg := cuda_graphs(engine, log.read_text(errors="replace"), env)):
    rec["cuda_graphs"] = cg  # the graph cap's chunks for tune (spend headroom graph_cap)
  rec.pop("ready_t", None)
  rec["end"] = sab.now_words()
  sab._write(outdir / FILE, rec)
  return rec


def summary(rec:dict[str, Any]) -> str:
  """One line: loaded or not, why not, and the free memory read."""
  f = rec.get("failure") or {}
  if not rec["loaded"]:
    ex = f.get("exit") or {}
    why = _words("not_ready", s=f.get("timeout_s")) if f.get("kind") == "not_ready" else \
      _words("exited", code=ex.get("code"), sig=f" ({ex['signal']})" if ex.get("signal") else "")
    oom = {True: _words("oom", line=f.get("line")), False: _words("no_oom"), None: _words("oom_unknown")}[f.get("out_of_memory")]
    return _words("failed", name=rec["name"], why=why, oom=oom, low=rec.get("free_min_mib"))
  sess = "" if rec["session"] is None else _words("session_ok" if rec["session"] else "session_failed", why=f.get("why", ""))
  return _words("loaded", name=rec["name"], ready=rec["free_ready_mib"], low=rec["free_min_mib"], session=sess)
