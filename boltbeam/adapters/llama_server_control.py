"""Exact-token llama.cpp correctness canary using the completion server.

This adapter is not a second benchmark authority. It starts the pinned llama.cpp
runtime, supplies the exact token ids emitted by tinygrad's authority, and
normalizes one fixed-depth decode result for the matched-control runner. Its
request time is diagnostic only; llama-bench remains timing authority.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import signal
import socket

from boltbeam.core.canonical import sha256_hex
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from boltbeam.vocab import SCHEMA_RUNTIME_OUTPUT_IDENTITY

JsonPost = Callable[[str, Mapping[str, Any], float], dict[str, Any]]


def token_evidence(tokens:Sequence[int]) -> dict[str, Any]:
  values = [int(token) for token in tokens]
  encoded = ",".join(map(str, values)).encode()
  return {
    "count": len(values),
    "sha256": sha256_hex(encoded),
    "first_token_ids": values[:16],
    "token_ids": values,
  }


def _json_post(url:str, payload:Mapping[str, Any], timeout_s:float) -> dict[str, Any]:
  request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                   headers={"Content-Type": "application/json"}, method="POST")
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    value = json.loads(response.read().decode())
  if not isinstance(value, dict): raise ValueError("llama server returned a non-object response")
  return value


def _completion(post:JsonPost, base_url:str, prompt_tokens:Sequence[int], *, seed:int,
                cache_prompt:bool, timeout_s:float) -> tuple[int, dict[str, Any]]:
  payload = {
    "prompt": [int(token) for token in prompt_tokens],
    "n_predict": 1,
    "temperature": 0.0,
    "seed": int(seed),
    "cache_prompt": bool(cache_prompt),
    "return_tokens": True,
    "n_probs": 1,
  }
  response = post(base_url.rstrip("/") + "/completion", payload, timeout_s)
  tokens = response.get("tokens")
  if not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], int):
    raise ValueError("llama completion did not return exactly one token id")
  if int(response.get("tokens_predicted", 1)) != 1:
    raise ValueError("llama completion did not predict exactly one token")
  return int(tokens[0]), response


def measure_fixed_depth_output(prompt_tokens:Sequence[int], *, warmups:int, seed:int = 20260617,
                               base_url:str = "http://127.0.0.1:8080", post:JsonPost = _json_post,
                               clock_ns:Callable[[], int] = time.perf_counter_ns,
                               timeout_s:float = 120.0) -> dict[str, Any]:
  """Measure the same post-prefill token lifecycle used by tinygrad's W row.

  tinygrad consumes one sampled token immediately after prompt prefill, advances
  the same cache for every warmup token, then starts a clean measured cycle. The
  llama control mirrors that lifecycle with exact prompt-token prefixes.
  """
  prompt = [int(token) for token in prompt_tokens]
  if not prompt: raise ValueError("matched llama control requires prompt token ids")
  if warmups < 0: raise ValueError("warmups must be non-negative")

  if warmups:
    warmup_token, _ = _completion(post, base_url, prompt, seed=seed,
                                  cache_prompt=False, timeout_s=timeout_s)
    warmup_prompt = [*prompt, warmup_token]
    for _ in range(warmups):
      warmup_token, _ = _completion(post, base_url, warmup_prompt, seed=seed,
                                    cache_prompt=True, timeout_s=timeout_s)
      warmup_prompt.append(warmup_token)

  prelude, prelude_response = _completion(post, base_url, prompt, seed=seed,
                                           cache_prompt=False, timeout_s=timeout_s)
  if prelude_response.get("tokens_evaluated") != len(prompt):
    raise ValueError("llama correctness canary did not evaluate the exact prompt-token count")
  started = clock_ns()
  generated, measured_response = _completion(post, base_url, [*prompt, prelude], seed=seed,
                                              cache_prompt=True, timeout_s=timeout_s)
  elapsed_ns = clock_ns() - started
  timings = measured_response.get("timings") if isinstance(measured_response.get("timings"), dict) else {}
  return {
    "schema": SCHEMA_RUNTIME_OUTPUT_IDENTITY,
    "provider_id": "llama.cpp/server",
    "claim_scope": "correctness_only",
    "prompt": token_evidence(prompt),
    "prelude": token_evidence([prelude]),
    "generated": token_evidence([generated]),
    "warmups": warmups,
    "decode_tokens": 1,
    "diagnostic_request_wall_ns": elapsed_ns,
    "runtime_predicted_ms": timings.get("predicted_ms"),
    "tokens_evaluated": measured_response.get("tokens_evaluated"),
    "cache_prompt": True,
    "sampling": {"temperature": 0.0, "seed": seed},
    "response_identity": {
      "prelude_generation_settings": prelude_response.get("generation_settings"),
      "measured_generation_settings": measured_response.get("generation_settings"),
    },
  }


def _free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return int(sock.getsockname()[1])


def _wait_healthy(base_url:str, process:subprocess.Popen, timeout_s:float) -> None:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if process.poll() is not None: raise RuntimeError(f"llama server exited with status {process.returncode}")
    try:
      with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=1) as response:
        if response.status == 200: return
    except (OSError, urllib.error.URLError):
      time.sleep(0.1)
  raise TimeoutError(f"llama server did not become healthy within {timeout_s:g}s")


@contextlib.contextmanager
def _running(command:list[str], base_url:str, log_path:str | pathlib.Path, startup_timeout_s:float,
             environ:Mapping[str, str] | None, stop_timeout_s:float = 5.0):
  """The command started in its own session with its output in the log, waited on until /health answers; stopped
  (its whole process group) when the block ends: SIGTERM, then SIGKILL after stop_timeout_s (a traced server writes
  its trace as it exits, which takes far longer than an untraced one's shutdown)."""
  log = pathlib.Path(log_path)
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open("w", encoding="utf-8") as stream:
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                               env=dict(os.environ if environ is None else environ), start_new_session=True)
    try:
      _wait_healthy(base_url, process, startup_timeout_s)
      yield
    finally:
      if process.poll() is None:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        deadline = time.monotonic() + stop_timeout_s
        try: process.wait(timeout=stop_timeout_s)
        except subprocess.TimeoutExpired:
          try: os.killpg(process.pid, signal.SIGKILL)
          except ProcessLookupError: pass
          process.wait(timeout=5)
        while time.monotonic() < deadline:  # the group outlives its leader while a child (vLLM's engine) still exits
          try: os.killpg(process.pid, 0)
          except ProcessLookupError: break
          time.sleep(0.2)


@contextlib.contextmanager
def serve(*, executable:str | pathlib.Path, model:str | pathlib.Path, context_size:int, log_path:str | pathlib.Path,
          startup_timeout_s:float = 180.0, environ:Mapping[str, str] | None = None, extra:Sequence[str] = ()):
  """One clean llama server on a free local port for the block: yields (base url, its command), then stops it."""
  server = pathlib.Path(executable).expanduser().absolute()
  model_path = pathlib.Path(model).expanduser().absolute()
  if not server.is_file(): raise FileNotFoundError(f"llama server not found: {server}")
  if not model_path.is_file(): raise FileNotFoundError(f"model not found: {model_path}")
  port = _free_port()
  base_url = f"http://127.0.0.1:{port}"
  command = [str(server), "--model", str(model_path), "--host", "127.0.0.1", "--port", str(port),
             "--ctx-size", str(context_size), "--n-gpu-layers", "99", "--no-warmup", *extra]
  with _running(command, base_url, log_path, startup_timeout_s, environ):
    yield base_url, command


OWN_FLAGS = ("--host", "--port")  # the only flags serve_argv replaces in a user's command


def own_argv(argv:Sequence[str], port:int) -> list[str]:
  """The user's server command with only --host and --port replaced (a loopback host and this port); nothing added."""
  out, skip = [], False
  for a in argv:
    if skip:
      skip = False
      continue
    if a in OWN_FLAGS:
      skip = True
      continue
    if a.split("=", 1)[0] in OWN_FLAGS:
      continue
    out.append(a)
  return [*out, "--host", "127.0.0.1", "--port", str(port)]


@contextlib.contextmanager
def serve_argv(argv:Sequence[str], *, log_path:str | pathlib.Path, startup_timeout_s:float = 600.0,
               environ:Mapping[str, str] | None = None, stop_timeout_s:float = 5.0):
  """The user's own server command on a free local port: yields (base url, the command run, seconds from start to
  /health answering), then stops it."""
  if not argv or not pathlib.Path(argv[0]).expanduser().is_file():
    raise FileNotFoundError(f"llama server not found: {argv[0] if argv else '(empty command)'}")
  port = _free_port()
  base_url = f"http://127.0.0.1:{port}"
  command = own_argv([str(pathlib.Path(argv[0]).expanduser()), *argv[1:]], port)
  began = time.monotonic()
  with _running(command, base_url, log_path, startup_timeout_s, environ, stop_timeout_s):
    yield base_url, command, time.monotonic() - began


def run_llama_server_control(*, executable:str | pathlib.Path, model:str | pathlib.Path,
                             prompt_tokens:Sequence[int], warmups:int, seed:int = 20260617,
                             context_size:int, log_path:str | pathlib.Path,
                             startup_timeout_s:float = 180.0, request_timeout_s:float = 120.0,
                             environ:Mapping[str, str] | None = None) -> dict[str, Any]:
  """Start one clean llama server, collect one matched row, and tear it down."""
  with serve(executable=executable, model=model, context_size=context_size, log_path=log_path,
             startup_timeout_s=startup_timeout_s, environ=environ) as (base_url, command):
    result = measure_fixed_depth_output(prompt_tokens, warmups=warmups, seed=seed, base_url=base_url,
                                        timeout_s=request_timeout_s)
  result["runtime_command"] = command
  result["runtime_log"] = str(pathlib.Path(log_path))
  return result


__all__ = ["measure_fixed_depth_output", "own_argv", "run_llama_server_control", "serve", "serve_argv", "token_evidence"]
