"""Normalize llama-bench's fixed-depth timing without claiming correctness.

The first ``warmups`` repetitions are explicitly discarded. Only the final
sample is timing authority; exact prompt/output identity belongs to the
separate llama-server correctness canary.
"""
from __future__ import annotations

import json
import pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

CommandRunner = Callable[[Sequence[str], pathlib.Path | None, Mapping[str, str] | None, float], tuple[int, str, str]]


def parse_fixed_depth_timing(stdout:str, *, depth:int, warmups:int) -> dict[str, Any]:
  payload = json.loads(stdout)
  if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
    raise ValueError("llama-bench must return exactly one JSON row")
  row = payload[0]
  if row.get("n_depth") != depth or row.get("n_gen") != 1 or row.get("n_prompt") != 0:
    raise ValueError("llama-bench row does not match fixed-depth one-token timing")
  samples_ns, samples_ts = row.get("samples_ns"), row.get("samples_ts")
  expected = warmups + 1
  if not isinstance(samples_ns, list) or not isinstance(samples_ts, list) or \
     len(samples_ns) != expected or len(samples_ts) != expected:
    raise ValueError(f"llama-bench must return {expected} repetitions")
  if any(not isinstance(value, (int, float)) or value <= 0 for value in [*samples_ns, *samples_ts]):
    raise ValueError("llama-bench returned a non-positive timing sample")
  return {
    "source": "llama-bench/final-repetition",
    "wall_ns": int(samples_ns[-1]),
    "tok_s": float(samples_ts[-1]),
    "warmups": warmups,
    "discarded_warmup_ns": [int(value) for value in samples_ns[:-1]],
    "discarded_warmup_tok_s": [float(value) for value in samples_ts[:-1]],
    "raw_row": row,
  }


def run_fixed_depth_timing(*, executable:str | pathlib.Path, model:str | pathlib.Path,
                           depth:int, warmups:int, run_command:CommandRunner,
                           cwd:str | pathlib.Path | None = None,
                           raw_output_path:str | pathlib.Path | None = None,
                           timeout_s:float = 1800.0) -> tuple[dict[str, Any], dict[str, Any]]:
  bench = pathlib.Path(executable).expanduser().absolute()
  model_path = pathlib.Path(model).expanduser().absolute()
  if not bench.is_file(): raise FileNotFoundError(f"llama-bench not found: {bench}")
  if not model_path.is_file(): raise FileNotFoundError(f"model not found: {model_path}")
  command = [str(bench), "-m", str(model_path), "-p", "0", "-n", "1", "-d", str(depth),
             "-ngl", "99", "-r", str(warmups + 1), "--no-warmup", "-o", "json"]
  code, stdout, stderr = run_command(command, pathlib.Path(cwd) if cwd else None, None, timeout_s)
  raw = {"command": command, "exit_code": code, "stdout": stdout, "stderr": stderr}
  if raw_output_path is not None:
    path = pathlib.Path(raw_output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  if code != 0: raise RuntimeError(f"llama-bench exited {code}: {stderr.strip() or stdout.strip()}")
  timing = parse_fixed_depth_timing(stdout, depth=depth, warmups=warmups)
  return timing, raw


__all__ = ["parse_fixed_depth_timing", "run_fixed_depth_timing"]
