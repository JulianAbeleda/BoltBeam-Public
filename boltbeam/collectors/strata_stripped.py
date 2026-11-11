"""UnbiasedStrata (fork row `strata-stripped`) as a measured engine: its OpenAI-style front end, one request at a time.

Starts `unbiased-strata` with the model and flags asked, waits for /health, times prefill and decode with BoltBeam's
prompts (" the" x N, batch 1, temperature 0) from the server's own `timings`, and stops the one process it started.
Every mismatch raises Refused with a plain reason; nothing is filled in. No model is touched by the tests: they use a
fake server and fake binaries.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import time
import urllib.request
from typing import Any

from boltbeam.collectors import engine_families as ef
from boltbeam.core.refused import Refused

ENV = "BOLTBEAM_STRATA_STRIPPED"


def row() -> dict[str, Any]:
  return next(f for fam in ef.families() for f in fam["forks"] if f["fork"] == "strata-stripped")


def locate(env:dict[str, str] | None = None, home:pathlib.Path | None = None) -> pathlib.Path:
  found = ef.find(row(), env, home)
  if not found or not pathlib.Path(found).is_file():
    raise Refused(f"unbiased-strata was not found: set ${ENV} to the binary or build it at {row()['paths'][0]}")
  return pathlib.Path(found)


def _run(argv:list[str], cwd:pathlib.Path) -> str:
  try:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
  except (OSError, subprocess.SubprocessError):
    return ""


def build(binary:pathlib.Path) -> dict[str, Any]:
  """The build as it is: the checkout's commit, the engine's version, the binary's sha256. Refuses a pin mismatch."""
  pin = row()
  commit = _run(["git", "rev-parse", "HEAD"], binary.parent)
  if not commit:
    raise Refused(f"cannot read the commit of {binary}: it is not inside a git checkout, and the row pins {pin['commit']}")
  if not commit.startswith(pin["commit"]):
    raise Refused(f"{binary} is at commit {commit[:12]}, the row pins {pin['commit']}")
  engine = binary.parent / "strata"
  version = _run([str(engine), "--version"], binary.parent) if engine.is_file() else ""
  if not version:
    raise Refused(f"cannot read the engine version from {engine} --version")
  return {"commit": commit, "engine_version": version, "binary": str(binary),
          "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}


def port_free(port:int, host:str = "127.0.0.1") -> bool:
  with socket.socket() as s:
    return s.connect_ex((host, port)) != 0


def get(base:str, path:str, timeout:float = 5.0) -> dict[str, Any]:
  with urllib.request.urlopen(base + path, timeout=timeout) as r:
    return json.loads(r.read())


def chat(base:str, context:int, tokens:int, timeout:float = 1800.0) -> dict[str, Any]:
  body = {"messages": [{"role": "user", "content": " the" * context}], "max_tokens": tokens, "temperature": 0, "top_k": 1,
          "stream": False, "ignore_eos": True}
  req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=timeout) as r:
    return json.loads(r.read())


def start(binary:pathlib.Path, model:str, *, port:int, alias:str, engine_args:list[str], timeout_s:float,
          host:str = "127.0.0.1", log:pathlib.Path | None = None) -> tuple[subprocess.Popen, str]:
  """Start the front end on `port`, wait for /health, and check what it reports against what was asked."""
  if not port_free(port, host):
    raise Refused(f"port {port} is already in use: another server holds it, and this adapter will not touch it")
  argv = [str(binary), "--engine", str(binary.parent / "strata"), "-m", model, "--host", host, "--port", str(port), "--alias", alias,
          "--temp", "0", "--top-k", "1", "--", *engine_args]
  out = open(log, "w") if log else subprocess.DEVNULL
  proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
  base = f"http://{host}:{port}"
  try:
    deadline = time.monotonic() + timeout_s
    while True:
      if proc.poll() is not None:
        raise Refused(f"unbiased-strata exited with code {proc.returncode} before it was healthy")
      try:
        if get(base, "/health").get("status") == "ok":
          break
      except (OSError, ValueError):
        pass
      if time.monotonic() > deadline:
        raise Refused(f"unbiased-strata was not healthy within {timeout_s:g} s")
      time.sleep(0.2)
    props = get(base, "/props")
    if props.get("model_path") != model or props.get("model_alias") != alias:
      raise Refused(f"the server reports model {props.get('model_path')!r} as {props.get('model_alias')!r}, asked {model!r} as {alias!r}")
  except BaseException:
    stop(proc)
    raise
  return proc, base


def stop(proc:subprocess.Popen) -> None:
  """Stop the process group this adapter started, and only that one."""
  if proc.poll() is None:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
      proc.wait(30)
    except subprocess.TimeoutExpired:
      os.killpg(proc.pid, signal.SIGKILL)
      proc.wait()


def timed(base:str, context:int, tokens:int, *, mtp:bool, wall_clock_ok:bool = False) -> dict[str, Any]:
  """One prefill and one decode at `context`, from the server's timings; wall clock only when allowed, and recorded."""
  def one(n:int) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    r = chat(base, context, n)
    return r, time.perf_counter() - t0
  (pre, wall_pre), (dec, wall_dec) = one(1), one(tokens)
  rows: dict[str, Any] = {"context": context, "tokens": tokens, "batch": 1, "prompt_tokens": dec.get("usage", {}).get("prompt_tokens")}
  for name, key, reply, n, wall in (("prefill_tok_s", "prompt_per_second", pre, rows["prompt_tokens"], wall_pre),
                                    ("decode_tok_s", "predicted_per_second", dec, tokens, wall_dec)):
    got = (reply.get("timings") or {}).get(key)
    if got:
      rows[name], rows[name + "_method"] = got, "server timings"
    elif wall_clock_ok and n:
      rows[name], rows[name + "_method"] = n / wall, "wall clock (includes the other phase and HTTP)"
    else:
      raise Refused(f"the response has no timings.{key} and wall clock is not allowed")
  drafts = (dec.get("timings") or {}).get("draft_n", 0)
  if mtp and not drafts:
    raise Refused("asked for an MTP pack but the server reports no drafts (timings.draft_n)")
  if drafts and not mtp:
    raise Refused(f"the server drafted {drafts} tokens but no MTP pack was asked for")
  rows["draft_n"], rows["draft_n_accepted"] = drafts, (dec.get("timings") or {}).get("draft_n_accepted", 0)
  return rows


def check_pack(pack:str | None, sha256:str | None) -> None:
  """The MTP pack asked for: its dense.bin must hash to the pinned value (the server does not report which pack it read)."""
  if pack and sha256:
    got = hashlib.sha256((pathlib.Path(pack) / "dense.bin").read_bytes()).hexdigest()
    if got != sha256:
      raise Refused(f"the MTP pack {pack} has dense.bin sha256 {got[:12]}, asked for {sha256[:12]}")


def measure(model:str, contexts:list[int], *, engine_args:list[str], mtp:bool, alias:str = "boltbeam-strata", port:int = 18090,
            tokens:int = 64, timeout_s:float = 3600.0, wall_clock_ok:bool = False, env:dict[str, str] | None = None,
            log:pathlib.Path | None = None, mtp_pack:str | None = None, mtp_sha256:str | None = None,
            run:pathlib.Path | None = None) -> dict[str, Any]:
  """run: the run folder whose engine_settings record the arguments this engine was started with (collectors/engine_args)."""
  check_pack(mtp_pack, mtp_sha256)
  binary = locate(env)
  rec = build(binary)
  proc, base = start(binary, model, port=port, alias=alias, engine_args=engine_args, timeout_s=timeout_s, log=log)
  if run is not None:
    from boltbeam.collectors import engine_args as recorded
    recorded.record(run, engine_args, engine="strata-stripped")
  try:
    return {**rec, "model": model, "engine_args": engine_args, "points": [timed(base, c, tokens, mtp=mtp, wall_clock_ok=wall_clock_ok) for c in contexts]}
  finally:
    stop(proc)
