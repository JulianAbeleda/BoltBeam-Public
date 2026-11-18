"""Paired session benchmarks on the user's own server: arms, alternating order, one record.

    boltbeam session-ab --server-argv 'SERVER ARGS' --session S.json --arm off= --arm on='VAR=1' --repeats 3 --out DIR

An arm is the server command with its own environment switches, extra flags (appended last) or binary (parse_arm). A
switch any arm sets is removed from the arms that do not set it, so "off" is off whatever the shell holds. Arms run
in blocks, each in the reverse order of the block before (A B | B A | A B; P Q C | C Q P | P Q C), so drift falls on
every arm alike. Each arm is a fresh server: the start gate (gate), one unmeasured warm-up request, then the session's
turns through the engine's chat adapter (collectors/replay.REPLAY_ENGINES: the timings arrive on its keys), each turn
timed from the last request byte sent; then the server is stopped by its process group and killed when it has not
exited within its own measured load time (the hang guard; a server hung at exit holds the GPU). An arm whose gate
failed, whose power limit moved or whose session did not finish is invalid and run again, never averaged. The record
(FILE) is rewritten after every arm, so --resume carries on after a crash. Every number is analysed in
workflow/session_ab.py; nothing here judges.

The session is the user's file and stays theirs: the record keeps its path and sha256, never its text.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import statistics
import time
from typing import Any, Callable

from boltbeam.vocab import SCHEMA_SESSION_AB

FILE = "session_ab.json"
ENV_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
HISTORY = ("own", "recorded")  # what goes back into the history after a turn: the server's reply, or the session's
WARMUP = ([{"role": "user", "content": "hi"}], 8)  # one short unmeasured request per server: its first use paid here
GATE_INTERVAL_S = 1.0  # the start gate's sampling interval
GATE_WINDOW = 10  # samples the cooling slope is fitted over (the decision is the slope's own 95% interval)
TELEMETRY_S = 0.5  # the arm's GPU sampler
LIMIT_STEP_W = 1e-3  # a power limit is read in mW: two limits differ by one step
HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


class Refused(ValueError):
  """An input or a record the benchmark cannot run on, with the reason."""


def _words(key:str, **kw:Any) -> str:
  from boltbeam.report.plain import words
  return words()["session_ab"][key].format(**kw)


# --- inputs --------------------------------------------------------------------------------------------------------

def _digest(path:pathlib.Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def read_session(path:str, history:str) -> tuple[dict[str, Any], dict[str, Any]]:
  """The session ({"system", "turns": [{"user", "assistant", "reply_tokens"}]}) and its identity (path, sha256)."""
  p = pathlib.Path(path).expanduser()
  s = json.loads(p.read_text())
  turns = s.get("turns") if isinstance(s, dict) else None
  if not isinstance(turns, list) or not turns or not isinstance(s.get("system", ""), str):
    raise Refused(_words("bad_session", path=p))
  for n, t in enumerate(turns, 1):
    if not isinstance(t.get("user"), str) or not isinstance(t.get("reply_tokens"), int) or (history == "recorded" and not isinstance(t.get("assistant"), str)):
      raise Refused(_words("bad_turn", n=n, history=history))
  return s, {"path": str(p.absolute()), "sha256": _digest(p), "turns": len(turns)}


def read_list(path:str | None, key:str, n:int) -> tuple[list[float] | None, dict[str, Any] | None]:
  """A per-turn list (reply lengths, gaps in seconds): a JSON list or an object holding it under key; one per turn."""
  if not path:
    return None, None
  p = pathlib.Path(path).expanduser()
  d = json.loads(p.read_text())
  got = d.get(key) if isinstance(d, dict) else d
  if not isinstance(got, list) or len(got) < n or any(not isinstance(x, (int, float)) or x < 0 for x in got[:n]):
    raise Refused(_words("bad_list", path=p, key=key, n=n))
  return list(got[:n]), {"path": str(p.absolute()), "sha256": _digest(p)}


def split_env(tokens:list[str]) -> tuple[dict[str, str], list[str]]:
  """Leading VAR=value tokens as an environment, and the rest."""
  env, i = {}, 0
  while i < len(tokens) and ENV_TOKEN.match(tokens[i]):
    k, v = tokens[i].split("=", 1)
    env[k], i = v, i + 1
  return env, tokens[i:]


def parse_arm(text:str) -> dict[str, Any]:
  """NAME=[VAR=value ...] [BINARY] [flags ...]: the arm's switches, a binary in place of the command's (a path to a file,
  first after the switches) and flags appended to the command."""
  name, _, spec = text.partition("=")
  if not NAME.match(name):
    raise Refused(_words("bad_arm", text=text))
  env, rest = split_env(shlex.split(spec))
  binary = None
  if rest and not rest[0].startswith("-") and pathlib.Path(rest[0]).expanduser().is_file():
    binary, rest = str(pathlib.Path(rest[0]).expanduser().absolute()), rest[1:]
  return {"name": name, "env": env, "binary": binary, "flags": rest}


def order(names:list[str], repeats:int) -> list[list[Any]]:
  """[block, name] in run order: each block the reverse of the one before."""
  return [[k, n] for k in range(repeats) for n in (names if k % 2 == 0 else names[::-1])]


def arm_command(base:dict[str, Any], arms:list[dict[str, Any]], arm:dict[str, Any], wrap:list[str],
                environ:dict[str, str] | None = None) -> tuple[list[str], dict[str, str]]:
  """The arm's server command and environment: the base's, its binary if it names one, its flags last, its switches
  set, and the switches only other arms set removed."""
  argv = [arm["binary"] or base["argv"][0], *base["argv"][1:], *arm["flags"]]
  env = dict(os.environ if environ is None else environ) | base["env"]
  for a in arms:
    for k in a["env"]:
      if k not in base["env"]:
        env.pop(k, None)
  env |= arm["env"]
  if wrap:
    argv = [shutil.which(wrap[0]) or wrap[0], *wrap[1:], *argv]
  return argv, env


# --- conditions ----------------------------------------------------------------------------------------------------

def load1() -> float | None:
  try:
    return float(pathlib.Path("/proc/loadavg").read_text().split()[0])
  except (OSError, ValueError, IndexError):
    return None


def host_temps(root:str = "/sys/class/hwmon") -> dict[str, float]:
  """Every host temperature sensor the kernel exposes, by chip and label, in C."""
  out = {}
  for t in sorted(pathlib.Path(root).glob("hwmon*/temp*_input")):
    try:
      chip = (t.parent / "name").read_text().strip()
      label = t.with_name(t.name.replace("_input", "_label"))
      out[f"{chip}/{label.read_text().strip() if label.is_file() else t.name}"] = int(t.read_text()) / 1000
    except (OSError, ValueError):
      continue
  return out


def _process_ticks(pgid:int | None) -> dict[int, int]:
  """CPU time (user + system, clock ticks) of every process outside the server's process group and this client."""
  out, me = {}, os.getpid()
  for d in os.listdir("/proc"):
    if d.isdigit() and int(d) != me:
      try:
        st = pathlib.Path(f"/proc/{d}/stat").read_text()
      except OSError:
        continue
      rest = st[st.rindex(")") + 2:].split()
      if int(rest[2]) != pgid and int(rest[1]) != me:  # rest[1] ppid: the client's own tools (the gate's probes)
        out[int(d)] = int(rest[11]) + int(rest[12])
  return out


def cpu_mark(pgid:int | None) -> tuple[float, dict[int, int]] | None:
  """The clock and every other process's CPU ticks: foreign_cores of two marks."""
  try:
    return time.monotonic(), _process_ticks(pgid)
  except (OSError, ValueError, IndexError):
    return None


def foreign_cores(a, b) -> float | None:
  """Cores' worth of CPU other programs used between two marks (per process, so one that exits takes nothing away;
  interrupt time the server's own GPU work causes is not in it)."""
  if not a or not b or b[0] <= a[0]:
    return None
  return sum(t - a[1].get(pid, 0) for pid, t in b[1].items()) / HZ / (b[0] - a[0])


def now_words() -> dict[str, Any]:
  t = datetime.datetime.now().astimezone()
  return {"at": t.isoformat(timespec="seconds"), "epoch": t.timestamp(), "load1": load1()}


# --- the start gate ------------------------------------------------------------------------------------------------

def cooling(temps:list[tuple[float, float]]) -> bool | None:
  """Whether the GPU is still cooling: the least-squares slope of the last GATE_WINDOW samples is below zero beyond
  its own 95% interval. None until the window is full."""
  from boltbeam.math.stats import t_quantile
  if len(temps) < GATE_WINDOW:
    return None
  xs, ys = zip(*temps[-GATE_WINDOW:])
  mx, my = statistics.fmean(xs), statistics.fmean(ys)
  sxx = sum((x - mx) ** 2 for x in xs)
  if sxx <= 0:
    return None
  slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
  resid = sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys)) / (len(xs) - 2)
  return slope + t_quantile(0.975, len(xs) - 2) * (resid / sxx) ** 0.5 < 0


def gate(source, backend:str | None, cap_w:float | None, bound_s:float | None, *, free:Callable[[str], dict] | None = None,
         clock:Callable[[], float] = time.monotonic, sleep:Callable[[float], None] = time.sleep) -> dict[str, Any]:
  """Wait until no other program holds the GPU, its power limit is the cap, and it has stopped cooling. bound_s: how
  long to wait (the previous arm's measured length; None before the first arm: settling always ends, and a GPU held or
  off its cap there is refused at once). Returns what was seen; passed False with the reason when the bound ran out."""
  from boltbeam.collectors.providers import gpu_free
  free = free or gpu_free
  t0, temps, idle = clock(), [], []
  mark = cpu_mark(None)
  while True:
    held = free(backend) if backend else {"free": True, "reason": ""}
    s = source.sample() if source else {}
    limit = source.power_limit_w() if source else None
    if s.get("temp_c") is not None:
      temps.append((clock(), float(s["temp_c"])))
    m = cpu_mark(None)
    idle.append(foreign_cores(mark, m))
    mark = m
    off_cap = cap_w is not None and limit is not None and abs(limit - cap_w) >= LIMIT_STEP_W
    still = cooling(temps) if source and temps else False
    waited = clock() - t0
    why = (held["reason"] if not held["free"] else _words("gate_cap", limit=limit, cap=cap_w) if off_cap
           else _words("gate_cooling") if still is not False else None)
    if why is None or (bound_s is None and (not held["free"] or off_cap)) or (bound_s is not None and waited >= bound_s):
      return {"passed": why is None, "why": why, "waited_s": round(waited, 1), "power_limit_w": limit,
              "temp_c": temps[-1][1] if temps else None, "foreign_cores": [round(x, 3) for x in idle if x is not None]}
    sleep(GATE_INTERVAL_S)


# --- one arm -------------------------------------------------------------------------------------------------------

def _group_alive(pgid:int) -> bool:
  try:
    os.killpg(pgid, 0)
    return True
  except (ProcessLookupError, PermissionError):
    return False


def stop(process, timeout_s:float, *, sleep:Callable[[float], None] = time.sleep) -> dict[str, Any]:
  """SIGTERM to the server's process group; SIGKILL to it when the group has not exited within timeout_s; survived
  when it still has not after as long again."""
  pgid, t0, killed = process.pid, time.monotonic(), False
  try:
    os.killpg(pgid, signal.SIGTERM)
  except ProcessLookupError:
    pass
  while process.poll() is None or _group_alive(pgid):
    waited = time.monotonic() - t0
    if waited >= timeout_s and not killed:
      killed = True
      try:
        os.killpg(pgid, signal.SIGKILL)
      except ProcessLookupError:
        break
    elif waited >= 2 * timeout_s:  # SIGKILL did not take it (a process stuck in the driver): said, never waited on forever
      return {"seconds": round(waited, 2), "timeout_s": round(timeout_s, 1), "killed": True, "survived": True}
    sleep(0.1)
  return {"seconds": round(time.monotonic() - t0, 2), "timeout_s": round(timeout_s, 1), "killed": killed}


def play(eng:dict[str, Callable[..., Any]], url:str, session:dict[str, Any], lengths:list[float] | None,
         gaps:list[float] | None, history:str, pgid:int | None, tele, say:Callable[[str], None], span:list[int],
         sleep:Callable[[float], None] = time.sleep, fill:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """The session's turns span[0]..span[1], one row each: the server's timings, the client's clock from the last
  request byte, the reply's sha256, other programs' CPU while the turn ran, the GPU clock and temperature over it.
  A span that starts past turn 1 first fills the cache, untimed, with the history up to the user message before it
  (the recorded replies): the first timed turn then processes what it would in a full run with the recorded history.
  fill: where the fill's own timings go."""
  msgs, rows = [{"role": "system", "content": session.get("system", "")}], []
  for t in session["turns"][:span[0] - 1]:
    msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["assistant"]}]
  if span[0] > 1:
    got = eng["chat"](url, msgs[:-1], 1)
    if fill is not None:
      fill.update(turns=span[0] - 1, prompt_tokens=got["usage"].get("prompt_tokens"), prompt_ms=got["timings"].get("prompt_ms"),
                  wall_ms=(got["t_last"] - got["t_send"]) * 1e3)
  for i, t in enumerate(session["turns"][span[0] - 1:span[1]], span[0]):
    gap = float(gaps[i - 1]) if gaps else 0.0
    if gap:
      sleep(gap)
    msgs.append({"role": "user", "content": t["user"]})
    want = int(lengths[i - 1]) if lengths else t["reply_tokens"]
    a, load = cpu_mark(pgid), load1()
    got = eng["chat"](url, msgs, want)
    b = cpu_mark(pgid)
    tm, us = got["timings"], got["usage"]
    win = tele.summary(got["t_send"], got["t_last"]) if tele else {}
    rows.append({"i": i, "gap_s": gap, "reply_wanted": want, "prompt_tokens": us.get("prompt_tokens"),
                 "cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                 "predicted_n": tm.get("predicted_n") or us.get("completion_tokens"), "predicted_ms": tm.get("predicted_ms"),
                 "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
                 "ttft_ms": (got["t_first"] - got["t_sent"]) * 1e3, "send_ms": (got["t_sent"] - got["t_send"]) * 1e3,
                 "wall_ms": (got["t_last"] - got["t_send"]) * 1e3, "sha256": hashlib.sha256(got["text"].encode()).hexdigest(),
                 "foreign_cores": foreign_cores(a, b), "load1": load,
                 "sm_mhz": (win.get("sm_mhz") or {}).get("mean"), "temp_c_max": win.get("temp_c_max")})
    msgs.append({"role": "assistant", "content": got["text"] if history == "own" else t["assistant"]})
    say(f"  turn {i}: prompt {tm.get('prompt_ms') or 0:.0f} ms, TTFT {rows[-1]['ttft_ms']:.0f} ms")
  return rows


def run_arm(ctx:dict[str, Any], arm:dict[str, Any], block:int, attempt:int, bound_s:float | None) -> dict[str, Any]:
  """One fresh server for one arm through the gate, the session and the hang guard: its record."""
  from boltbeam.adapters.llama_server_control import serve_argv
  from boltbeam.collectors import gpu_telemetry
  from boltbeam.collectors.replay import _sha256, _version
  inp, say = ctx["inputs"], ctx["say"]
  argv, env = arm_command(ctx["base"], ctx["arms"], arm, inp["wrap"])
  binary = pathlib.Path(arm["binary"] or ctx["base"]["argv"][0]).expanduser()
  rec: dict[str, Any] = {"name": arm["name"], "block": block, "attempt": attempt, "valid": False, "why": None,
                         "binary": {"path": str(binary), "sha256": _sha256(binary) if binary.is_file() else None,
                                    "version": _version(binary) if binary.is_file() else None},
                         "switches": {k: env.get(k) for a in ctx["arms"] for k in a["env"]}}
  say(f"session-ab: block {block + 1}, arm {arm['name']} (attempt {attempt + 1}): start gate")
  rec["gate"] = g = ctx["gate"](ctx["source"], ctx["backend"], ctx["cap_w"], bound_s)
  rec["start"], rec["host_temps_start"] = now_words(), host_temps()
  if not g["passed"]:
    rec["why"] = _words("gate_failed", why=g["why"])
    return rec
  proc: list[Any] = []
  log = ctx["out"] / "logs" / f"{block + 1}-{arm['name']}-{attempt + 1}.log"
  with gpu_telemetry.opened(ctx["source"] or gpu_telemetry.Unsampled(_words("no_gpu")), 0, TELEMETRY_S) as tele:
    try:
      with serve_argv(argv, log_path=log, environ=env, started=proc.append) as (url, command, load_s):
        rec["command"], rec["load_s"] = command, round(load_s, 2)
        ctx["engine"]["chat"](url, *WARMUP)
        rec["fill"] = {}
        rec["turns"] = play(ctx["engine"], url, ctx["session"], ctx["lengths"], ctx["gaps"], inp["history"],
                            proc[0].pid, tele, say, inp["turns"], fill=rec["fill"])
        if ctx.get("hook"):  # after the timed turns, the server still up: the user's own check (a quality gate), untimed
          rec["hook"] = arm_hook(ctx["hook"], url, rec, ctx["out"], inp["session"]["path"], inp["turns"])
        rec["exit"] = stop(proc[0], load_s)  # tearing down never needs longer than bringing up took
        if rec["exit"]["killed"]:
          say(_words("exit_hang", pid=proc[0].pid, s=rec["exit"]["timeout_s"]))
    except Exception as exc:  # noqa: BLE001  a server that fails is this arm's reason; the run goes on
      rec["why"] = _words("server_failed", why=str(exc)[-300:])
    rec["telemetry"] = tele.summary()
  rec["end"], rec["host_temps_end"] = now_words(), host_temps()
  rec["power_limit_end_w"] = ctx["source"].power_limit_w() if ctx["source"] else None
  want = inp["turns"][1] - inp["turns"][0] + 1
  if rec["why"] is None:
    moved = g["power_limit_w"] is not None and rec["power_limit_end_w"] is not None and abs(g["power_limit_w"] - rec["power_limit_end_w"]) >= LIMIT_STEP_W
    short = len(rec.get("turns") or []) < want
    rec["why"] = _words("survived") if (rec.get("exit") or {}).get("survived") else \
                 _words("cap_moved", a=g["power_limit_w"], b=rec["power_limit_end_w"]) if moved else \
                 _words("short", n=len(rec.get("turns") or []), want=want) if short else None
  rec["valid"] = rec["why"] is None
  return rec


def parse_span(text:str | None, n:int) -> list[int]:
  """"A-B" (or "N", the first N) as [first, last] turn numbers within the session; all turns by default."""
  if not text:
    return [1, n]
  a, _, b = str(text).partition("-")
  try:
    span = [int(a), int(b)] if b else [1, int(a)]
  except ValueError:
    raise Refused(_words("bad_span", turns=text)) from None
  if not 1 <= span[0] <= span[1] <= n:
    raise Refused(_words("bad_span", turns=text))
  return span


def arm_hook(cmd:str, url:str, rec:dict[str, Any], out:pathlib.Path, session:str = "", turns:list[int] | None = None) -> dict[str, Any]:
  """The user's command (a shell line) once per arm, the server still up, its URL and the arm in the environment:
  SESSION_AB_URL, SESSION_AB_ARM, SESSION_AB_BLOCK, SESSION_AB_OUT, SESSION_AB_SESSION (path), SESSION_AB_TURNS ([A, B])
  and SESSION_AB_COMMAND (the arm's server argv, JSON). Its exit code and output's tail are recorded;
  it never makes an arm invalid and is never timed."""
  import subprocess
  t0 = time.monotonic()
  env = {**os.environ, "SESSION_AB_URL": url, "SESSION_AB_ARM": rec["name"], "SESSION_AB_BLOCK": str(rec["block"] + 1),
         "SESSION_AB_OUT": str(out), "SESSION_AB_SESSION": session, "SESSION_AB_TURNS": json.dumps(turns),
         "SESSION_AB_COMMAND": json.dumps(rec.get("command") or [])}
  got = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
  return {"rc": got.returncode, "seconds": round(time.monotonic() - t0, 1), "tail": (got.stdout + got.stderr)[-500:]}


# --- the run -------------------------------------------------------------------------------------------------------

def _write(path:pathlib.Path, record:dict[str, Any]) -> None:
  tmp = path.with_suffix(".tmp")
  tmp.write_text(json.dumps(record, indent=1) + "\n")
  tmp.replace(path)


def _machine() -> tuple[Any, str | None]:
  """The telemetry source and the backend of this machine's first GPU (none: no gate on it, said in the record)."""
  from boltbeam.collectors.gpu_telemetry import source_for
  from boltbeam.target.gpu import first_gpu
  from boltbeam.target.hardware import hardware_profile
  from boltbeam.target.vendors import vendor
  device = first_gpu(hardware_profile())
  if not device:
    return None, None
  source = source_for(device)
  return (None if source is None or source.__class__.__name__ == "Unsampled" else source), vendor(device.get("vendor")).backend


def run(*, out:str, server_argv:str, arms:list[str], session:str, repeats:int, engine:str, history:str = "own",
        lengths:str | None = None, gaps:str | None = None, wrap:str = "", power_cap_w:float | None = None,
        retries:int = 2, turns:str | None = None, resume:bool = False, hook:str = "", say:Callable[[str], None] = print,
        machine:Callable[[], tuple[Any, str | None]] = _machine, gate_fn=gate) -> dict[str, Any]:
  """Every arm of the plan, valid ones kept, invalid ones run again (at most `retries` times each); the record."""
  from boltbeam.collectors.replay import engine as engine_of
  if history not in HISTORY:
    raise Refused(_words("bad_history", history=history, ok=", ".join(HISTORY)))
  parsed = [parse_arm(a) for a in arms]
  names = [a["name"] for a in parsed]
  if len(set(names)) != len(names) or len(names) < 2 or repeats < 1:
    raise Refused(_words("bad_plan"))
  sess, sess_id = read_session(session, history)
  n = len(sess["turns"])
  span = parse_span(turns, n)
  if span[0] > 1 and any(not isinstance(t.get("assistant"), str) for t in sess["turns"][:span[0] - 1]):
    raise Refused(_words("bad_span", turns=turns))
  lens, lens_id = read_list(lengths, "lengths", n)
  gap, gap_id = read_list(gaps, "gaps_s", n)
  base_env, base_argv = split_env(shlex.split(server_argv.replace("\\\n", " ")))
  if not base_argv:
    raise Refused(_words("bad_server"))
  inputs = {"server_argv": base_argv, "server_env": base_env, "arms": parsed, "session": sess_id, "lengths": lens_id,
            "gaps": gap_id, "history": history, "repeats": repeats, "order": order(names, repeats), "engine": engine,
            "wrap": shlex.split(wrap), "power_cap_w": power_cap_w, "turns": span,
            **({"hook": hook} if hook else {})}
  outdir = pathlib.Path(out).expanduser()
  path = outdir / FILE
  if path.exists() and not resume:
    raise Refused(_words("exists", path=path))
  record = json.loads(path.read_text()) if path.exists() else {"schema": SCHEMA_SESSION_AB, "inputs": inputs, "arms": []}
  if record.get("inputs") != inputs:
    raise Refused(_words("resume_differs", path=path))
  outdir.mkdir(parents=True, exist_ok=True)
  source, backend = machine()
  record["machine"] = {"gpu_sampled": source is not None, "backend": backend}
  cap = power_cap_w if power_cap_w is not None else next((a["gate"]["power_limit_w"] for a in record["arms"] if a.get("gate")), None)
  if cap is None and source is not None:
    cap = source.power_limit_w()
  record["power_cap_w"] = cap
  ctx = {"inputs": inputs, "base": {"argv": base_argv, "env": base_env}, "arms": parsed, "session": sess, "lengths": lens,
         "gaps": gap, "engine": engine_of(engine), "source": source, "backend": backend, "cap_w": cap, "out": outdir,
         "say": say, "gate": gate_fn, "hook": hook}
  by = {a["name"]: a for a in parsed}
  for block, name in inputs["order"]:
    tried = [a for a in record["arms"] if a["block"] == block and a["name"] == name]
    if any(a["valid"] for a in tried):
      continue
    now = 0  # attempts in this invocation: a --resume after a stop gets its retries again
    while now <= retries:
      now += 1
      lengths_s = [sum(t["wall_ms"] for t in a.get("turns") or []) / 1e3 for a in record["arms"] if a["valid"]]
      rec = run_arm(ctx, by[name], block, len(tried), lengths_s[-1] if lengths_s else None)
      record["arms"].append(rec)
      tried.append(rec)
      _write(path, record)
      say(f"session-ab: {name} block {block + 1}: " + ("valid" if rec["valid"] else f"INVALID: {rec['why']}"))
      if rec["valid"]:
        break
    else:
      record["stopped"] = _words("retries", name=name, block=block + 1, n=retries + 1)
      _write(path, record)
      return record
  record.pop("stopped", None)
  _write(path, record)
  return record
