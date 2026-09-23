"""Fail-closed, serial matched-control evidence for fixed-depth decode.

BoltBeam owns orchestration and validation only. tinygrad and llama.cpp remain
the runtime executors, and their raw artifacts are retained beside normalized
row manifests.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence

from boltbeam.core.canonical import sha256_file, sha256_hex
from datetime import datetime, timezone
from typing import Any

from boltbeam.adapters.llama_bench_control import run_fixed_depth_timing
from boltbeam.adapters.llama_server_control import run_llama_server_control
from boltbeam.adapters.tinygrad_decode import validate_decode_authority
from boltbeam.vocab import (SCHEMA_MATCHED_CONTROL_PROTOCOL, SCHEMA_MATCHED_CONTROL_ROW,
                            SCHEMA_MATCHED_CONTROL_SUMMARY, SCHEMA_RUNTIME_OUTPUT_IDENTITY,
                            SCHEMA_TINYGRAD_GRAPH_ADMISSION_CENSUS)
from boltbeam.workflow.autoscan import _hardware_profile
from boltbeam.workflow.common import read_json, write_json

CommandRunner = Callable[[Sequence[str], pathlib.Path | None, Mapping[str, str] | None, float], tuple[int, str, str]]
CommandExecutor = Callable[..., tuple[int, str, str, Mapping[str, Any] | None]]
LlamaExecutor = Callable[..., dict[str, Any]]
LlamaTimingExecutor = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


def _utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _run_command(argv:Sequence[str], cwd:pathlib.Path | None = None,
                 env:Mapping[str, str] | None = None, timeout_s:float = 30.0) -> tuple[int, str, str]:
  try:
    proc = subprocess.run(list(argv), cwd=str(cwd) if cwd else None, env=dict(env) if env else None,
                          capture_output=True, text=True, timeout=timeout_s, check=False)
    return proc.returncode, proc.stdout, proc.stderr
  except (OSError, subprocess.TimeoutExpired) as exc:
    return 127, "", f"{type(exc).__name__}: {exc}"


default_command_runner = _run_command
utc_now = _utc_now


def _sha256(path:pathlib.Path) -> str:
  return sha256_file(path)


def _text_sha256(value:str) -> str:
  return sha256_hex(value.encode())


def _command_fact(argv:Sequence[str], run_command:CommandRunner, *, timeout_s:float = 30.0) -> dict[str, Any]:
  code, stdout, stderr = run_command(tuple(argv), None, None, timeout_s)
  return {
    "argv": list(argv), "exit_code": code,
    "stdout": stdout.strip(), "stderr": stderr.strip(),
    "output_sha256": _text_sha256(stdout + "\nSTDERR\n" + stderr),
    "status": "observed" if code == 0 else "unavailable",
  }


def _git(root:pathlib.Path, args:Sequence[str], run_command:CommandRunner) -> str | None:
  code, stdout, _ = run_command(("git", "-C", str(root), *args), None, None, 30.0)
  return stdout.strip() if code == 0 else None


def _git_identity(root:pathlib.Path, run_command:CommandRunner, *, subtree:str | None = None) -> dict[str, Any]:
  commit = _git(root, ("rev-parse", "HEAD"), run_command)
  tree = _git(root, ("rev-parse", "HEAD^{tree}"), run_command)
  status = _git(root, ("status", "--porcelain"), run_command)
  subtree_tree = _git(root, ("rev-parse", f"HEAD:{subtree}"), run_command) if subtree else None
  valid_hash = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None
  return {
    "root": str(root), "commit": commit, "tree": tree,
    "subtree": subtree, "subtree_tree": subtree_tree,
    "dirty": status is None or bool(status), "tracked_status": status,
    "status": "observed" if valid_hash(commit) and valid_hash(tree) and (not subtree or valid_hash(subtree_tree)) else "unavailable",
  }


def _file_identity(path:pathlib.Path) -> dict[str, Any]:
  if not path.is_file(): return {"path": str(path), "status": "unavailable", "sha256": None}
  stat = path.stat()
  return {"path": str(path), "status": "content_hashed", "sha256": _sha256(path),
          "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _toolchain_identity(run_command:CommandRunner) -> dict[str, Any]:
  facts = {
    "macos": _command_fact(("sw_vers",), run_command),
    "xcode": _command_fact(("xcodebuild", "-version"), run_command),
    "xctrace": _command_fact(("xcrun", "xctrace", "version"), run_command),
    "metal_compiler_path": _command_fact(("xcrun", "--find", "metal"), run_command),
    "clang_compiler_path": _command_fact(("xcrun", "--find", "clang"), run_command),
    "clang_version": _command_fact(("xcrun", "clang", "--version"), run_command),
  }
  for compiler in ("metal", "clang"):
    path_fact = facts[f"{compiler}_compiler_path"]
    compiler_path = path_fact["stdout"] if path_fact["status"] == "observed" else ""
    facts[f"{compiler}_compiler_binary"] = _file_identity(pathlib.Path(compiler_path)) if compiler_path else {
      "path": None, "status": "unavailable", "sha256": None,
    }
  facts["python"] = {"version": platform.python_version(), "executable": sys.executable,
                     "executable_identity": _file_identity(pathlib.Path(sys.executable))}
  return facts


def _apple_hardware(run_command:CommandRunner) -> dict[str, Any]:
  def adapter(argv:tuple[str, ...]) -> tuple[int, str, str]:
    return run_command(argv, None, None, 30.0)
  return _hardware_profile(run_command=adapter, system_name="Darwin")


def build_environment_identity(*, model:str | pathlib.Path,
                               repositories:Mapping[str, tuple[str | pathlib.Path, str | None]],
                               executables:Mapping[str, str | pathlib.Path], target_id:str,
                               run_command:CommandRunner = _run_command) -> dict[str, Any]:
  """Build the shared immutable environment envelope used by control protocols."""
  model_path = pathlib.Path(model).expanduser().absolute()
  if not model_path.is_file(): raise FileNotFoundError(f"model not found: {model_path}")
  return {
    "model": _file_identity(model_path),
    "target": {"target_id": target_id, "backend": "Metal"},
    "repositories": {name:_git_identity(pathlib.Path(root).expanduser().absolute(), run_command, subtree=subtree)
                     for name,(root,subtree) in repositories.items()},
    "executables": {name:_file_identity(pathlib.Path(path).expanduser().absolute())
                    for name,path in executables.items()},
    "toolchain": _toolchain_identity(run_command),
    "hardware": _apple_hardware(run_command),
  }


def environment_preflight_errors(plan:Mapping[str, Any]) -> list[str]:
  """Validate the shared identity envelope without protocol-specific assumptions."""
  errors = []
  if plan.get("model", {}).get("status") != "content_hashed": errors.append("model_content_hash")
  if plan.get("target", {}).get("target_id") != plan.get("hardware", {}).get("gpu", {}).get("target_id"):
    errors.append("target_resolution")
  for name, identity in dict(plan.get("repositories") or {}).items():
    if identity.get("status") != "observed": errors.append(f"{name}_revision")
    if identity.get("dirty"): errors.append(f"{name}_dirty")
  for name, identity in dict(plan.get("executables") or {}).items():
    if identity.get("status") != "content_hashed": errors.append(f"{name}_identity")
  for name in ("macos", "xcode", "xctrace", "metal_compiler_path", "clang_compiler_path", "clang_version"):
    if plan.get("toolchain", {}).get(name, {}).get("status") != "observed": errors.append(name)
  for compiler in ("metal", "clang"):
    if plan.get("toolchain", {}).get(f"{compiler}_compiler_binary", {}).get("status") != "content_hashed":
      errors.append(f"{compiler}_compiler_binary_identity")
  return errors


def build_protocol(*, model:str | pathlib.Path, tinygrad_root:str | pathlib.Path,
                   llama_root:str | pathlib.Path, llama_bench:str | pathlib.Path,
                   llama_server:str | pathlib.Path,
                   output_root:str | pathlib.Path, target_id:str = "apple_m4_10c",
                   depth:int = 128, warmups:int = 2, samples:int = 5,
                   minimum_free_memory_percent:int = 10,
                   run_id:str | None = None, run_command:CommandRunner = _run_command) -> dict[str, Any]:
  if depth < 1 or warmups < 2 or samples < 5:
    raise ValueError("matched control requires depth>=1, warmups>=2, and samples>=5")
  model_path = pathlib.Path(model).expanduser().absolute()
  tg_root = pathlib.Path(tinygrad_root).expanduser().absolute()
  llama_repo = pathlib.Path(llama_root).expanduser().absolute()
  bench = pathlib.Path(llama_bench).expanduser().absolute()
  server = pathlib.Path(llama_server).expanduser().absolute()
  out = pathlib.Path(output_root).expanduser().absolute()
  if not model_path.is_file(): raise FileNotFoundError(f"model not found: {model_path}")
  tinygrad_python = tg_root / ".venv" / "bin" / "python"
  order = [item for index in range(1, samples + 1) for item in (f"tinygrad-{index:02d}", f"llama-{index:02d}")]
  identity = build_environment_identity(model=model_path, target_id=target_id, run_command=run_command,
    repositories={"boltbeam":(pathlib.Path(__file__).resolve().parents[2], None),
                  "tinygrad":(tg_root, "tinygrad"), "llama_cpp":(llama_repo, None)},
    executables={"tinygrad_python":tinygrad_python, "llama_bench":bench, "llama_server":server})
  plan = {
    "schema": SCHEMA_MATCHED_CONTROL_PROTOCOL,
    "run_id": run_id or f"metal-matched-control-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    "created_at": _utc_now(),
    "output_root": str(out),
    **identity,
    "workload": {"phase": "decode", "fixed_depth": depth, "decode_tokens": 1,
                 "warmups": warmups, "samples_per_runtime": samples,
                 "tinygrad_measured_lifecycle": "prompt prefill + one unmeasured prelude token + one measured token",
                 "llama_timing_lifecycle": "llama-bench depth cache + discarded repetitions + final measured repetition",
                 "llama_correctness_lifecycle": "exact prompt token ids + one prelude token + one output token",
                 "timing_and_correctness_claims_separate": True},
    "validity": {
      "serial_only": True, "require_ac_power": True, "require_nominal_thermal": True,
      "minimum_free_memory_percent": minimum_free_memory_percent,
      "maximum_relative_mad": 0.05,
      "allow_dirty_repositories": False,
      "unknown_dynamic_probe_invalidates": True,
    },
    "order": order,
    "commands": {
      "tinygrad": [str(tinygrad_python), "extra/llm_research/decode/decode_runtime_overhead.py",
                   "--model", str(model_path), "--ckpts", str(depth), "--nmeas", "1", "--reps", "1",
                   "--warmup-decode", str(warmups), "--out", "<row>/authority.json",
                   "--graph-admission-out", "<row>/graph-admission-census.json"],
      "llama_timing": [str(bench), "-m", str(model_path), "-p", "0", "-n", "1", "-d", str(depth),
                       "-ngl", "99", "-r", str(warmups + 1), "--no-warmup", "-o", "json"],
      "llama_correctness": [str(server), "--model", str(model_path), "--ctx-size", str(depth + 16),
                            "--n-gpu-layers", "99", "--no-warmup", "<dynamic-host-port>"],
    },
  }
  plan["preflight"] = protocol_preflight(plan)
  return plan


def protocol_preflight(plan:Mapping[str, Any]) -> dict[str, Any]:
  errors = []
  if plan.get("schema") != SCHEMA_MATCHED_CONTROL_PROTOCOL: errors.append("schema")
  errors.extend(protocol_definition_errors(plan))
  output_root = plan.get("output_root")
  if not isinstance(output_root, str) or not output_root or pathlib.Path(output_root).exists():
    errors.append("output_root_not_fresh")
  errors.extend(environment_preflight_errors(plan))
  return {"status": "ready" if not errors else "blocked", "errors": errors}


def protocol_definition_errors(plan:Mapping[str, Any]) -> list[str]:
  """Validate the immutable MR0 measurement contract independently of CLI construction."""
  errors = []
  workload = plan.get("workload") if isinstance(plan.get("workload"), Mapping) else {}
  depth, warmups, samples = workload.get("fixed_depth"), workload.get("warmups"), workload.get("samples_per_runtime")
  if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1: errors.append("fixed_depth")
  if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 2: errors.append("warmups")
  if not isinstance(samples, int) or isinstance(samples, bool) or samples < 5: errors.append("samples_per_runtime")
  if isinstance(samples, int) and not isinstance(samples, bool) and samples >= 1:
    required = [item for index in range(1, samples + 1) for item in (f"tinygrad-{index:02d}", f"llama-{index:02d}")]
    if plan.get("order") != required: errors.append("interleaved_order")
  else: errors.append("interleaved_order")
  return errors


def _probe_memory(stdout:str) -> int | None:
  match = re.search(r"System-wide memory free percentage:\s*(\d+)%", stdout, re.I)
  return int(match.group(1)) if match else None


def _probe_power(stdout:str) -> str | None:
  match = re.search(r"Now drawing from ['\"]([^'\"]+)['\"]", stdout)
  return match.group(1) if match else None


def _probe_thermal(stdout:str) -> str | None:
  if re.search(r"no thermal warning", stdout, re.I): return "nominal"
  values = {key.lower(): int(value) for key, value in re.findall(
    r"(CPU_Speed_Limit|Scheduler_Limit|Thermal_Level)\s*=\s*(\d+)", stdout)}
  if values:
    nominal = values.get("cpu_speed_limit", 100) == 100 and values.get("scheduler_limit", 100) == 100 and values.get("thermal_level", 0) == 0
    return "nominal" if nominal else "constrained"
  return None


def capture_dynamic_validity(validity:Mapping[str, Any], *, run_command:CommandRunner = _run_command) -> dict[str, Any]:
  memory = _command_fact(("memory_pressure", "-Q"), run_command)
  power = _command_fact(("pmset", "-g", "batt"), run_command)
  thermal = _command_fact(("pmset", "-g", "therm"), run_command)
  free = _probe_memory(memory["stdout"]) if memory["status"] == "observed" else None
  source = _probe_power(power["stdout"]) if power["status"] == "observed" else None
  thermal_state = _probe_thermal(thermal["stdout"]) if thermal["status"] == "observed" else None
  errors = []
  if free is None: errors.append("memory_pressure_unknown")
  elif free < int(validity["minimum_free_memory_percent"]): errors.append("memory_pressure_below_floor")
  if validity.get("require_ac_power") and source != "AC Power": errors.append("not_ac_power")
  if validity.get("require_nominal_thermal") and thermal_state != "nominal": errors.append("thermal_not_nominal")
  return {
    "captured_at": _utc_now(), "memory_pressure": memory, "power": power, "thermal": thermal,
    "free_memory_percent": free, "power_source": source, "thermal_state": thermal_state,
    "status": "valid" if not errors else "invalid", "invalidation_reasons": errors,
  }


def reconcile_graph_census(authority:Mapping[str, Any], census:Mapping[str, Any]) -> dict[str, Any]:
  if census.get("schema") != SCHEMA_TINYGRAD_GRAPH_ADMISSION_CENSUS: raise ValueError("unexpected graph census schema")
  counts = census.get("counts") if isinstance(census.get("counts"), dict) else {}
  logical = int(counts.get("logical_calls", -1))
  assigned = sum(int(counts.get(key, 0)) for key in
                 ("graph_members", "direct_calls", "ignored_slice_nodes", "constructor_failures"))
  records = census.get("records") if isinstance(census.get("records"), list) else []
  identity_fields = ("program_hash", "source_sha256", "binary_sha256")
  identity_rows = [{field:row.get(field) for field in identity_fields} for row in records]
  missing_identity = sum(1 for row in identity_rows if any(
    not isinstance(row[field], str) or re.fullmatch(r"[0-9a-f]{64}", row[field]) is None for field in identity_fields))
  unknown = sum(1 for row in records if row.get("reason") == "unknown" or row.get("decision") == "unknown")
  batch_sum = sum(int(row.get("size", 0)) for row in census.get("batches", []) if isinstance(row, dict))
  authority_programs = authority["rows"][0]["programs_per_token_by_route"][authority["rows"][0]["route"]]
  errors = []
  if logical < 0 or logical != assigned: errors.append("logical_call_reconciliation")
  if len(records) != logical: errors.append("record_count")
  if batch_sum != int(counts.get("graph_members", -1)): errors.append("batch_member_count")
  if unknown: errors.append("unknown_decisions")
  if missing_identity: errors.append("missing_program_identity")
  if authority_programs != logical: errors.append("authority_program_count")
  if census.get("capture", {}).get("fixed_depth") != authority["rows"][0]["fixed_depth"]: errors.append("fixed_depth")
  return {"status": "valid" if not errors else "invalid", "errors": errors,
          "logical_calls": logical, "assigned_calls": assigned, "unknown_decisions": unknown,
          "graph_members": counts.get("graph_members"), "graph_batches": counts.get("graph_batches"),
          "direct_calls": counts.get("direct_calls"), "missing_program_identities": missing_identity,
          "program_identity_sha256":None if missing_identity else _text_sha256(json.dumps(identity_rows, sort_keys=True, separators=(",", ":"))),
          "census_sha256": None}


def tinygrad_output_identity(authority:Mapping[str, Any]) -> dict[str, Any]:
  row = authority["rows"][0]
  prompt = row.get("prompt_evidence") or {}
  validated_token_ids(prompt, label="tinygrad prompt", expected_count=row.get("fixed_depth"))
  generated = row.get("generated_token_evidence")
  prelude = row.get("prelude_token_evidence")
  if not isinstance(prelude, list) or len(prelude) != 1:
    raise ValueError("tinygrad authority lacks one prelude-token evidence row")
  if not isinstance(generated, list) or len(generated) != 1:
    raise ValueError("tinygrad authority lacks one generated-token evidence row")
  validated_token_ids(prelude[0], label="tinygrad prelude", expected_count=1)
  validated_token_ids(generated[0], label="tinygrad generated", expected_count=1)
  return {"prompt": prompt, "prelude": prelude[0], "generated": generated[0]}


def validated_token_ids(evidence:Mapping[str, Any], *, label:str, expected_count:int | None = None) -> list[int]:
  """Return exact ids only when every redundant token-evidence field agrees."""
  if not isinstance(evidence, Mapping): raise ValueError(f"{label} evidence is missing")
  tokens = evidence.get("token_ids")
  if not isinstance(tokens, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in tokens):
    raise ValueError(f"{label} lacks exact integer token ids")
  if evidence.get("count") != len(tokens) or expected_count is not None and len(tokens) != expected_count:
    raise ValueError(f"{label} token count mismatch")
  if evidence.get("first_token_ids") != tokens[:16]: raise ValueError(f"{label} token prefix mismatch")
  if evidence.get("sha256") != _text_sha256(",".join(map(str, tokens))):
    raise ValueError(f"{label} token hash mismatch")
  return tokens


def require_matching_output_identity(expected:Mapping[str, Any], observed:Mapping[str, Any]) -> None:
  if observed.get("schema") != SCHEMA_RUNTIME_OUTPUT_IDENTITY:
    raise ValueError("llama output identity schema mismatch")
  for name, count in (("prompt", len(expected["prompt"]["token_ids"])), ("prelude", 1), ("generated", 1)):
    wanted = validated_token_ids(expected[name], label=f"tinygrad {name}", expected_count=count)
    got = validated_token_ids(observed.get(name, {}), label=f"llama {name}", expected_count=count)
    if got != wanted: raise ValueError(f"cross-runtime {name}-token identity mismatch")


def summarize_dispersion(samples:Sequence[float | int], *, expected_samples:int,
                         maximum_relative_mad:float = 0.05) -> dict[str, Any]:
  """Central robust stability authority for serialized timing samples."""
  values = [float(value) for value in samples]
  median = statistics.median(values) if values else None
  mad = statistics.median(abs(value - median) for value in values) if median is not None else None
  relative_mad = mad / median if median and mad is not None else None
  complete = len(values) == expected_samples
  stable = complete and relative_mad is not None and relative_mad <= maximum_relative_mad
  return {"sample_count":len(values), "expected_samples":expected_samples, "median":median,
          "minimum":min(values) if values else None, "maximum":max(values) if values else None,
          "mean":statistics.mean(values) if values else None,
          "stdev":statistics.stdev(values) if len(values) > 1 else None,
          "median_absolute_deviation":mad, "relative_mad":relative_mad,
          "maximum_relative_mad":maximum_relative_mad, "status":"stable" if stable else "unstable"}


def run_tinygrad_authority_row(plan:Mapping[str, Any], row_dir:pathlib.Path, *,
                               run_command:CommandRunner = _run_command,
                               environment:Mapping[str, str] | None = None,
                               command_executor:CommandExecutor | None = None,
                               reconciliation:Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]] = reconcile_graph_census
                               ) -> dict[str, Any]:
  """Execute and normalize the one shared fixed-depth tinygrad authority row."""
  authority_path, census_path = row_dir / "authority.json", row_dir / "graph-admission-census.json"
  command = [part.replace("<row>/authority.json", str(authority_path)).replace(
    "<row>/graph-admission-census.json", str(census_path)) for part in plan["commands"]["tinygrad"]]
  env = dict(os.environ); env.update({"DEV":"METAL", "PYTHONPATH":".", "PYTHONHASHSEED":"0"})
  if environment: env.update({str(key):str(value) for key,value in environment.items()})
  cwd = pathlib.Path(plan["repositories"]["tinygrad"]["root"])
  if command_executor is None:
    code, stdout, stderr = run_command(command, cwd, env, 1800.0); auxiliary = None
  else:
    code, stdout, stderr, auxiliary = command_executor(command=command, cwd=cwd, environ=env,
      timeout_s=1800.0, row_dir=row_dir, authority_path=authority_path)
  process_path = row_dir / "tinygrad-process.json"
  write_json(process_path, {"command":command, "environment":dict(environment or {}),
                            "exit_code":code, "stdout":stdout, "stderr":stderr})
  if code != 0: raise RuntimeError(f"tinygrad row exited {code}: {stderr.strip() or stdout.strip()}")
  authority, census = read_json(authority_path), read_json(census_path)
  validate_decode_authority(authority)
  if authority["workload"]["warmup_decode"] != plan["workload"]["warmups"]: raise ValueError("tinygrad warmup mismatch")
  if authority["rows"][0]["fixed_depth"] != plan["workload"]["fixed_depth"]: raise ValueError("tinygrad depth mismatch")
  output = tinygrad_output_identity(authority)
  reconciled = reconciliation(authority, census)
  reconciled["census_sha256"] = _sha256(census_path)
  if reconciled["status"] != "valid":
    raise ValueError("graph census reconciliation failed: " + ", ".join(reconciled["errors"]))
  return {
    "authority":authority, "census":census, "output":output, "reconciliation":reconciled,
    "measurement":{"wall_ns":int(round(authority["rows"][0]["wall_ms_W"] * 1_000_000)),
                   "tok_s":authority["rows"][0]["tok_s_W"], "source":"tinygrad_fixed_depth_W"},
    "raw":{"authority":str(authority_path), "authority_sha256":_sha256(authority_path),
           "graph_census":str(census_path), "process":str(process_path), "process_sha256":_sha256(process_path)},
    "auxiliary":dict(auxiliary) if isinstance(auxiliary, Mapping) else None,
  }


def validate_protocol(plan:Mapping[str, Any]) -> None:
  if plan.get("schema") != SCHEMA_MATCHED_CONTROL_PROTOCOL: raise ValueError("unexpected matched-control protocol schema")
  if plan.get("preflight", {}).get("status") != "ready":
    raise ValueError("matched-control protocol preflight is blocked: " + ", ".join(plan.get("preflight", {}).get("errors", [])))
  if errors := protocol_definition_errors(plan):
    raise ValueError("matched-control protocol definition is invalid: " + ", ".join(errors))


def row_identity(plan:Mapping[str, Any], executable_names:Sequence[str]) -> dict[str, Any]:
  """Project one protocol's shared identity into a self-contained row envelope."""
  return {
    "model": plan["model"], "repositories": plan["repositories"], "target": plan["target"],
    "hardware": plan["hardware"], "toolchain": plan["toolchain"],
    "executables": {name: plan["executables"][name] for name in executable_names},
  }


def environment_drift(plan:Mapping[str, Any], *, run_command:CommandRunner = _run_command) -> list[str]:
  """Re-resolve revisions and binaries between every serialized row."""
  errors = []
  observed = {name:_git_identity(pathlib.Path(identity["root"]), run_command, subtree=identity.get("subtree"))
              for name,identity in plan["repositories"].items()}
  for name, current in observed.items():
    expected = plan["repositories"][name]
    for field in ("commit", "tree", "subtree_tree"):
      if current.get(field) != expected.get(field): errors.append(f"{name}_{field}_drift")
    if current.get("dirty"): errors.append(f"{name}_dirty")
  model = pathlib.Path(plan["model"]["path"])
  stat = model.stat()
  if stat.st_size != plan["model"]["size_bytes"] or stat.st_mtime_ns != plan["model"]["mtime_ns"]:
    errors.append("model_stat_drift")
  for name, expected in plan["executables"].items():
    current = _file_identity(pathlib.Path(expected["path"]))
    if current.get("sha256") != expected.get("sha256"): errors.append(f"{name}_binary_drift")
  return errors


def _row_identity(plan:Mapping[str, Any], provider:str) -> dict[str, Any]:
  names = ("tinygrad_python",) if provider == "tinygrad" else ("llama_bench", "llama_server")
  return row_identity(plan, names)


def summarize_matched_control(plan:Mapping[str, Any], rows:list[Mapping[str, Any]], *,
                              stop_reasons:Sequence[str] = ()) -> dict[str, Any]:
  """Summarize rows and enforce the shared robust timing-stability gate."""
  expected = len(plan["order"])
  valid_rows = [row for row in rows if row["status"] == "valid"]
  expected_samples = int(plan["workload"]["samples_per_runtime"])
  maximum_relative_mad = float(plan.get("validity", {}).get("maximum_relative_mad", 0.05))
  per_provider = {}
  for provider in ("tinygrad", "llama.cpp"):
    selected = [row for row in valid_rows if row["provider_id"] == provider]
    walls = [row["measurement"]["wall_ns"] for row in selected]
    per_provider[provider] = {"valid_samples":len(selected), "wall_ns":walls,
      "median_wall_ns":statistics.median(walls) if walls else None,
      "median_tok_s":statistics.median(row["measurement"]["tok_s"] for row in selected) if selected else None,
      "dispersion":summarize_dispersion(walls, expected_samples=expected_samples,
                                        maximum_relative_mad=maximum_relative_mad)}
  stability_blockers = [f"{provider}:unstable_timing" for provider,facts in per_provider.items()
                        if facts["dispersion"]["status"] != "stable"]
  observed_order = [row["row_id"] for row in rows]
  order_blockers = [] if observed_order == plan["order"] else ["interleaved_order_mismatch"]
  complete = len(rows) == expected and len(valid_rows) == expected and not stability_blockers and not order_blockers
  out = pathlib.Path(plan["output_root"])
  return {"schema":SCHEMA_MATCHED_CONTROL_SUMMARY, "run_id":plan["run_id"],
          "status":"complete" if complete else "inconclusive", "completed_at":_utc_now(),
          "expected_rows":expected, "observed_rows":len(rows), "valid_rows":len(valid_rows),
          "order":observed_order, "required_order":plan["order"],
          "warmups":plan["workload"]["warmups"], "providers":per_provider,
          "stability":{"status":"stable" if not stability_blockers else "unstable",
                       "maximum_relative_mad":maximum_relative_mad, "blockers":stability_blockers},
          "stop_reasons":[*stop_reasons, *order_blockers],
          "rows":[str(out / "rows" / row["row_id"] / "row.json") for row in rows]}


def execute_protocol(plan:Mapping[str, Any], *, run_command:CommandRunner = _run_command,
                     llama_executor:LlamaExecutor = run_llama_server_control,
                     llama_timing_executor:LlamaTimingExecutor = run_fixed_depth_timing,
                     dynamic_probe:Callable[..., dict[str, Any]] = capture_dynamic_validity) -> dict[str, Any]:
  validate_protocol(plan)
  out = pathlib.Path(plan["output_root"])
  out.mkdir(parents=True, exist_ok=False)
  write_json(out / "protocol.json", dict(plan))
  rows, stop_reasons = [], []
  paired_tinygrad: dict[int, dict[str, Any]] = {}
  tinygrad_program_identity_sha256: str | None = None
  for position, row_id in enumerate(plan["order"]):
    provider, index_text = row_id.split("-", 1)
    index = int(index_text)
    row_dir = out / "rows" / row_id
    row_dir.mkdir(parents=True)
    started_at = _utc_now()
    drift = environment_drift(plan, run_command=run_command)
    before = dynamic_probe(plan["validity"], run_command=run_command)
    if drift or before.get("status") != "valid":
      row = {"schema": SCHEMA_MATCHED_CONTROL_ROW, "run_id": plan["run_id"], "row_id": row_id,
             "order_index": position, "provider_id": provider, "started_at": started_at, "ended_at": _utc_now(),
             "status": "invalidated", "invalidation_reasons": [*drift, *before.get("invalidation_reasons", [])],
             "before": before, "identity": _row_identity(plan, provider)}
      write_json(row_dir / "row.json", row); rows.append(row); stop_reasons.extend(row["invalidation_reasons"]); break

    try:
      if provider == "tinygrad":
        normalized = run_tinygrad_authority_row(plan, row_dir, run_command=run_command)
        output, reconciliation = normalized["output"], normalized["reconciliation"]
        measurement, raw = normalized["measurement"], normalized["raw"]
        observed_program_identity = reconciliation["program_identity_sha256"]
        if tinygrad_program_identity_sha256 is None: tinygrad_program_identity_sha256 = observed_program_identity
        elif observed_program_identity != tinygrad_program_identity_sha256:
          raise ValueError("tinygrad generated program identity drift")
        paired_tinygrad[index] = {"output": output, "reconciliation": reconciliation}
      elif provider == "llama":
        paired = paired_tinygrad.get(index)
        if paired is None: raise ValueError("llama row has no preceding paired tinygrad prompt")
        prompt_tokens = paired["output"]["prompt"]["token_ids"]
        timing, timing_raw = llama_timing_executor(executable=plan["executables"]["llama_bench"]["path"],
          model=plan["model"]["path"], depth=plan["workload"]["fixed_depth"],
          warmups=plan["workload"]["warmups"], run_command=run_command,
          cwd=plan["repositories"]["llama_cpp"]["root"], raw_output_path=row_dir / "llama-bench.json")
        timing_path = row_dir / "llama-bench.json"
        if not timing_path.is_file(): write_json(timing_path, timing_raw)
        bench_commit = timing.get("raw_row", {}).get("build_commit")
        expected_commit = plan["repositories"]["llama_cpp"]["commit"]
        if not isinstance(bench_commit, str) or not expected_commit.startswith(bench_commit):
          raise ValueError("llama-bench build revision does not match the pinned llama.cpp checkout")
        output = llama_executor(executable=plan["executables"]["llama_server"]["path"],
          model=plan["model"]["path"], prompt_tokens=prompt_tokens, warmups=plan["workload"]["warmups"],
          context_size=plan["workload"]["fixed_depth"] + 16, log_path=row_dir / "llama-server.log")
        output_path = row_dir / "output-identity.json"; write_json(output_path, output)
        if output.get("warmups") != plan["workload"]["warmups"]: raise ValueError("llama warmup mismatch")
        require_matching_output_identity(paired["output"], output)
        reconciliation = dict(paired["reconciliation"])
        measurement = timing
        raw = {"output_identity": str(output_path), "output_identity_sha256": _sha256(output_path),
               "llama_bench": str(timing_path), "llama_bench_sha256": _sha256(timing_path),
               "paired_graph_census_sha256": reconciliation["census_sha256"]}
      else:
        raise ValueError(f"unknown matched-control provider {provider!r}")

      after = dynamic_probe(plan["validity"], run_command=run_command)
      invalidation = [*environment_drift(plan, run_command=run_command), *after.get("invalidation_reasons", [])]
      row = {"schema": SCHEMA_MATCHED_CONTROL_ROW, "run_id": plan["run_id"], "row_id": row_id,
             "order_index": position, "provider_id": "llama.cpp" if provider == "llama" else "tinygrad",
             "started_at": started_at, "ended_at": _utc_now(), "status": "valid" if not invalidation else "invalidated",
             "invalidation_reasons": invalidation, "workload": plan["workload"],
             "identity": _row_identity(plan, provider),
             "before": before, "after": after, "measurement": measurement,
             "output_identity": output, "graph_reconciliation": reconciliation, "raw": raw}
      write_json(row_dir / "row.json", row); rows.append(row)
      if invalidation: stop_reasons.extend(invalidation); break
    except Exception as exc:
      after = dynamic_probe(plan["validity"], run_command=run_command)
      row = {"schema": SCHEMA_MATCHED_CONTROL_ROW, "run_id": plan["run_id"], "row_id": row_id,
             "order_index": position, "provider_id": provider, "started_at": started_at, "ended_at": _utc_now(),
             "status": "blocked", "invalidation_reasons": [f"{type(exc).__name__}: {exc}"],
             "before": before, "after": after,
             "raw_artifacts": [str(path) for path in sorted(row_dir.iterdir()) if path.is_file()],
             "identity": _row_identity(plan, provider)}
      write_json(row_dir / "row.json", row); rows.append(row); stop_reasons.extend(row["invalidation_reasons"]); break

  summary = summarize_matched_control(plan, rows, stop_reasons=stop_reasons)
  write_json(out / "summary.json", summary)
  return summary


__all__ = ["build_protocol", "capture_dynamic_validity", "execute_protocol", "protocol_preflight",
           "build_environment_identity", "default_command_runner", "environment_drift", "environment_preflight_errors",
           "protocol_definition_errors", "reconcile_graph_census", "require_matching_output_identity", "row_identity", "run_tinygrad_authority_row",
           "summarize_dispersion", "summarize_matched_control", "tinygrad_output_identity", "utc_now", "validate_protocol",
           "validated_token_ids"]
