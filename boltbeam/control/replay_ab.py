"""Paired Metal hybrid-replay A/B orchestration over one shared authority.

This module contributes only the strategy comparison. Environment identity,
dynamic validity, tinygrad execution, exact-token evidence, and base graph
census reconciliation are reused from :mod:`boltbeam.control.matched_control`.
"""
from __future__ import annotations

import pathlib
import random
import statistics
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from boltbeam.control.matched_control import (CommandRunner, build_environment_identity, capture_dynamic_validity,
  default_command_runner, environment_drift, environment_preflight_errors, reconcile_graph_census,
  row_identity, run_tinygrad_authority_row, summarize_dispersion, utc_now, validated_token_ids)
from boltbeam.collectors.metal_system_trace import capture_command_metal_trace
from boltbeam.vocab import SCHEMA_REPLAY_AB_PROTOCOL, SCHEMA_REPLAY_AB_ROW, SCHEMA_REPLAY_AB_SUMMARY
from boltbeam.workflow.common import read_json, write_json

AuthorityExecutor = Callable[..., dict[str, Any]]

STRATEGIES = {
  "control": {"environment":"0", "configured_strategy":"partitioned_control"},
  "hybrid": {"environment":"1", "configured_strategy":"hybrid_icb_direct"},
}


def build_protocol(*, model:str | pathlib.Path, tinygrad_root:str | pathlib.Path,
                   output_root:str | pathlib.Path, target_id:str = "apple_m4_10c",
                   depth:int = 128, warmups:int = 2, samples:int = 5,
                   minimum_free_memory_percent:int = 10, run_id:str | None = None,
                   run_command:CommandRunner = default_command_runner) -> dict[str, Any]:
  if depth < 1 or warmups < 2 or samples < 5:
    raise ValueError("replay A/B requires depth>=1, warmups>=2, and samples>=5")
  tg_root = pathlib.Path(tinygrad_root).expanduser().absolute()
  tinygrad_python = tg_root / ".venv" / "bin" / "python"
  model_path = pathlib.Path(model).expanduser().absolute()
  identity = build_environment_identity(model=model_path, target_id=target_id, run_command=run_command,
    repositories={"boltbeam":(pathlib.Path(__file__).resolve().parents[2], None), "tinygrad":(tg_root, "tinygrad")},
    executables={"tinygrad_python":tinygrad_python})
  order = [arm for index in range(1, samples + 1) for arm in (f"control-{index:02d}", f"hybrid-{index:02d}")]
  plan = {
    "schema":SCHEMA_REPLAY_AB_PROTOCOL,
    "run_id":run_id or f"metal-replay-ab-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    "created_at":utc_now(), "output_root":str(pathlib.Path(output_root).expanduser().absolute()),
    **identity,
    "workload":{"model_family":"Qwen3-8B", "phase":"decode", "fixed_depth":depth,
                "decode_tokens":1, "warmups":warmups, "samples_per_strategy":samples,
                "measured_lifecycle":"prompt prefill + one unmeasured prelude token + one measured token"},
    "strategies":STRATEGIES,
    "validity":{"serial_only":True, "require_ac_power":True, "require_nominal_thermal":True,
                "minimum_free_memory_percent":minimum_free_memory_percent,
                "maximum_relative_mad":0.05,
                "maximum_process_memory_regression_percent":3.0,
                "allow_dirty_repositories":False, "unknown_dynamic_probe_invalidates":True},
    "order":order,
    "commands":{"tinygrad":[str(tinygrad_python), "extra/llm_research/decode/decode_runtime_overhead.py",
      "--model", str(model_path), "--ckpts", str(depth), "--nmeas", "1", "--reps", "1",
      "--warmup-decode", str(warmups), "--skip-dispatch-diagnostic", "--out", "<row>/authority.json",
      "--graph-admission-out", "<row>/graph-admission-census.json"]},
  }
  errors = protocol_definition_errors(plan)
  if pathlib.Path(plan["output_root"]).exists(): errors.append("output_root_not_fresh")
  errors.extend(environment_preflight_errors(plan))
  plan["preflight"] = {"status":"ready" if not errors else "blocked", "errors":errors}
  return plan


def protocol_definition_errors(plan:Mapping[str, Any]) -> list[str]:
  workload = plan.get("workload") if isinstance(plan.get("workload"), Mapping) else {}
  depth, warmups, samples = workload.get("fixed_depth"), workload.get("warmups"), workload.get("samples_per_strategy")
  errors = []
  if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1: errors.append("fixed_depth")
  if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 2: errors.append("warmups")
  if not isinstance(samples, int) or isinstance(samples, bool) or samples < 5: errors.append("samples_per_strategy")
  if isinstance(samples, int) and not isinstance(samples, bool) and samples >= 1:
    expected = [arm for index in range(1, samples + 1) for arm in (f"control-{index:02d}", f"hybrid-{index:02d}")]
    if plan.get("order") != expected: errors.append("interleaved_order")
  else: errors.append("interleaved_order")
  return errors


def validate_protocol(plan:Mapping[str, Any]) -> None:
  if plan.get("schema") != SCHEMA_REPLAY_AB_PROTOCOL: raise ValueError("unexpected replay A/B protocol schema")
  if plan.get("preflight", {}).get("status") != "ready":
    raise ValueError("replay A/B preflight is blocked: " + ", ".join(plan.get("preflight", {}).get("errors", [])))
  if errors := protocol_definition_errors(plan):
    raise ValueError("replay A/B protocol definition is invalid: " + ", ".join(errors))


def reconcile_replay_census(authority:Mapping[str, Any], census:Mapping[str, Any], *, arm:str) -> dict[str, Any]:
  if arm not in STRATEGIES: raise ValueError(f"unknown replay arm {arm!r}")
  result = reconcile_graph_census(authority, census)
  errors = list(result["errors"])
  records = census.get("records") if isinstance(census.get("records"), list) else []
  limited = [record for record in records if record.get("admission_reason") == "backend_buffer_offset_width"]
  facts = authority.get("runtime_settings", {}).get("metal_replay")
  expected_strategy = STRATEGIES[arm]["configured_strategy"]
  if not isinstance(facts, dict) or facts.get("schema") != "tinygrad.metal_replay.v1": errors.append("metal_replay_facts")
  elif facts.get("configured_strategy") != expected_strategy: errors.append("configured_strategy")
  last_graph = facts.get("last_graph") if isinstance(facts, dict) else None
  if not isinstance(last_graph, dict) or last_graph.get("strategy") != expected_strategy: errors.append("last_graph_strategy")
  elif last_graph.get("committed") is not True: errors.append("last_graph_not_committed")
  if not limited: errors.append("backend_offset_mechanism_not_exercised")
  if arm == "control":
    if any(record.get("supported") is not False or record.get("assignment") != "direct" for record in limited):
      errors.append("control_backend_offset_assignment")
  else:
    if any(record.get("supported") is not True or record.get("assignment") != "graph" for record in limited):
      errors.append("hybrid_backend_offset_assignment")
    if result.get("direct_calls") != 0 or result.get("graph_members") != result.get("logical_calls"):
      errors.append("hybrid_not_fully_graphed")
  result.update({"status":"valid" if not errors else "invalid", "errors":errors, "arm":arm,
                 "configured_strategy":expected_strategy, "backend_offset_calls":len(limited),
                 "admission_reason_histogram":census.get("admission_reason_histogram", {}),
                 "program_hashes":[record.get("program_hash") for record in records],
                 "compiled_identity_set":sorted({(record.get("source_sha256"), record.get("binary_sha256"))
                                                  for record in records
                                                  if record.get("source_sha256") and record.get("binary_sha256")})})
  return result


def reconcile_pair(control:Mapping[str, Any], hybrid:Mapping[str, Any], *,
                   maximum_process_memory_regression_percent:float = 3.0) -> dict[str, Any]:
  errors = []
  for key in ("prompt", "prelude", "generated"):
    ctoken = validated_token_ids(control["output"][key], label=f"control {key}")
    htoken = validated_token_ids(hybrid["output"][key], label=f"hybrid {key}")
    if ctoken != htoken: errors.append(f"{key}_identity")
  cgraph, hgraph = control["reconciliation"], hybrid["reconciliation"]
  if cgraph.get("logical_calls") != hgraph.get("logical_calls"): errors.append("logical_calls")
  if cgraph.get("program_hashes") != hgraph.get("program_hashes"): errors.append("program_order_identity")
  ccompiled, hcompiled = cgraph.get("compiled_identity_set"), hgraph.get("compiled_identity_set")
  if not ccompiled or not hcompiled or ccompiled != hcompiled: errors.append("compiled_identity_set")
  if cgraph.get("admission_reason_histogram") != hgraph.get("admission_reason_histogram"):
    errors.append("admission_reason_identity")
  if hgraph.get("graph_members", 0) < cgraph.get("graph_members", 0): errors.append("graph_members_regressed")
  if hgraph.get("direct_calls", 0) > cgraph.get("direct_calls", 0): errors.append("direct_calls_regressed")
  control_memory = control.get("authority", {}).get("runtime_settings", {}).get("memory_facts", {})
  hybrid_memory = hybrid.get("authority", {}).get("runtime_settings", {}).get("memory_facts", {})
  # Graph-retained resource counts may legitimately differ when the same process-resident buffers move from direct
  # calls into an ICB. The invariant is the process-wide tinygrad-owned resident allocation, not one graph's subset.
  resident_identity_fields = ("allocation_scope", "tinygrad_live_buffer_bytes")
  if any(control_memory.get(key) is None or hybrid_memory.get(key) is None for key in resident_identity_fields):
    errors.append("resident_buffer_identity_missing")
  elif any(control_memory[key] != hybrid_memory[key] for key in resident_identity_fields):
    errors.append("resident_buffer_identity")
  memory_comparisons, memory_regressions = {}, []
  for key in ("current_allocated_bytes", "tinygrad_live_buffer_bytes"):
    before, after = control_memory.get(key), hybrid_memory.get(key)
    delta_percent = 100 * (after-before) / before if isinstance(before, int) and before > 0 and isinstance(after, int) else None
    memory_comparisons[key] = {"control":before, "hybrid":after, "delta_percent":delta_percent}
    if delta_percent is not None and delta_percent > maximum_process_memory_regression_percent:
      memory_regressions.append(key)
  return {"status":"valid" if not errors else "invalid", "errors":errors,
          "logical_calls":cgraph.get("logical_calls"),
          "control_graph_members":cgraph.get("graph_members"), "hybrid_graph_members":hgraph.get("graph_members"),
          "control_direct_calls":cgraph.get("direct_calls"), "hybrid_direct_calls":hgraph.get("direct_calls"),
          "process_memory_gate":{"status":"pass" if not memory_regressions else "fail",
            "maximum_regression_percent":maximum_process_memory_regression_percent,
            "regressions":memory_regressions, "comparisons":memory_comparisons,
            "resident_facts_scope":"process_tinygrad_live_bytes; per-graph structural counts retained separately"}}


def replay_execution_identity(normalized:Mapping[str, Any]) -> dict[str, Any]:
  """Identity that must remain invariant across every control and hybrid row."""
  output = normalized["output"]
  memory = normalized.get("authority", {}).get("runtime_settings", {}).get("memory_facts", {})
  reconciliation = normalized["reconciliation"]
  return {
    "tokens":{name:validated_token_ids(output[name], label=f"replay {name}") for name in ("prompt", "prelude", "generated")},
    "program_hashes":reconciliation.get("program_hashes"),
    "compiled_identity_set":reconciliation.get("compiled_identity_set"),
    "resident_buffers":{key:memory.get(key) for key in ("allocation_scope", "tinygrad_live_buffer_bytes")},
  }


def _bootstrap_mean_ci(values:list[float], *, resamples:int = 10_000, seed:int = 20260730) -> dict[str, Any]:
  if len(values) < 2:
    return {"status":"unavailable", "reason":"fewer_than_two_complete_pairs", "confidence":0.95,
            "estimator":"mean_paired_delta_wall_percent", "resamples":resamples, "seed":seed,
            "lower":None, "upper":None}
  rng = random.Random(seed)
  estimates = sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(resamples))
  return {"status":"available", "confidence":0.95, "estimator":"mean_paired_delta_wall_percent",
          "resamples":resamples, "seed":seed,
          "lower":estimates[int(0.025 * resamples)], "upper":estimates[int(0.975 * resamples) - 1]}


def paired_statistics(rows:list[Mapping[str, Any]], samples:int, *, maximum_relative_mad:float = 0.05) -> dict[str, Any]:
  by_id = {row["row_id"]:row for row in rows if row.get("status") == "valid"}
  pairs = []
  for index in range(1, samples + 1):
    control, hybrid = by_id.get(f"control-{index:02d}"), by_id.get(f"hybrid-{index:02d}")
    if control is None or hybrid is None: continue
    cwall, hwall = control["measurement"]["wall_ns"], hybrid["measurement"]["wall_ns"]
    cgpu = control.get("evidence", {}).get("profiler", {}).get("selected_gpu_union_us")
    hgpu = hybrid.get("evidence", {}).get("profiler", {}).get("selected_gpu_union_us")
    pairs.append({"pair_index":index, "control_wall_ns":cwall, "hybrid_wall_ns":hwall,
                  "delta_wall_ns":hwall-cwall, "delta_wall_percent":100*(hwall-cwall)/cwall,
                  "latency_speedup":cwall/hwall,
                  "throughput_retention":hybrid["measurement"]["tok_s"] / control["measurement"]["tok_s"],
                  "control_gpu_union_us":cgpu, "hybrid_gpu_union_us":hgpu,
                  "delta_gpu_union_percent":100*(hgpu-cgpu)/cgpu if isinstance(cgpu, (int, float)) and cgpu > 0 and isinstance(hgpu, (int, float)) else None})
  deltas = [pair["delta_wall_ns"] for pair in pairs]
  percents = [pair["delta_wall_percent"] for pair in pairs]
  retentions = [pair["throughput_retention"] for pair in pairs]
  control_walls = [pair["control_wall_ns"] for pair in pairs]
  hybrid_walls = [pair["hybrid_wall_ns"] for pair in pairs]
  gpu_percents = [pair["delta_gpu_union_percent"] for pair in pairs if pair["delta_gpu_union_percent"] is not None]
  arm_dispersion = {"control":summarize_dispersion(control_walls, expected_samples=samples,
                                                     maximum_relative_mad=maximum_relative_mad),
                    "hybrid":summarize_dispersion(hybrid_walls, expected_samples=samples,
                                                   maximum_relative_mad=maximum_relative_mad)}
  stability = "stable" if all(value["status"] == "stable" for value in arm_dispersion.values()) else "unstable"
  return {"complete_pairs":len(pairs), "pairs":pairs,
          "median_delta_wall_ns":statistics.median(deltas) if deltas else None,
          "mean_delta_wall_ns":statistics.mean(deltas) if deltas else None,
          "stdev_delta_wall_ns":statistics.stdev(deltas) if len(deltas) > 1 else None,
          "median_delta_wall_percent":statistics.median(percents) if percents else None,
          "median_delta_gpu_union_percent":statistics.median(gpu_percents) if len(gpu_percents) == samples else None,
          "median_throughput_retention":statistics.median(retentions) if retentions else None,
          "bootstrap_ci":_bootstrap_mean_ci(percents),
          "strategy_dispersion":{"status":stability, "arms":arm_dispersion},
          "three_percent_improvement_gate":("pass" if len(percents) == samples and statistics.median(percents) <= -3.0 else "fail")}


def _trace_command_executor(plan:Mapping[str, Any]) -> Callable[..., tuple[int, str, str, Mapping[str, Any] | None]]:
  """Adapt the shared Metal System Trace collector to one authority row."""
  def execute(*, command, cwd, environ, timeout_s, row_dir, authority_path):
    census_path = pathlib.Path(row_dir) / "graph-admission-census.json"
    def measurement_loader() -> Mapping[str, Any]:
      authority, census = read_json(authority_path), read_json(census_path)
      source = authority["rows"][0]
      counts = census["counts"]
      return {"wall_us":float(source["wall_ms_W"]) * 1000.0, "tok_s":float(source["tok_s_W"]),
              "samples_us":[float(source["wall_ms_W"]) * 1000.0],
              "runtime_expected_command_buffers":int(counts["graph_batches"]) + int(counts["direct_calls"])}
    code, stdout, stderr, trace = capture_command_metal_trace(command=command, cwd=cwd, environ=environ,
      trace_dir=pathlib.Path(row_dir) / "metal-system-trace", model_id=plan["model"]["sha256"],
      target_id=plan["target"]["target_id"], context=int(plan["workload"]["fixed_depth"]),
      measurement_loader=measurement_loader, time_limit_s=int(timeout_s))
    trace_path = pathlib.Path(row_dir) / "metal-system-trace.json"
    if trace is not None: write_json(trace_path, trace)
    return code, stdout, stderr, {"metal_system_trace":trace, "metal_system_trace_path":str(trace_path)}
  return execute


def _row_evidence(normalized:Mapping[str, Any]) -> dict[str, Any]:
  auxiliary = normalized.get("auxiliary") if isinstance(normalized.get("auxiliary"), Mapping) else {}
  trace = auxiliary.get("metal_system_trace") if isinstance(auxiliary.get("metal_system_trace"), Mapping) else None
  whole = next((row for row in trace.get("rows", []) if row.get("scope") == "whole_step"), {}) if trace else {}
  profiler = dict(auxiliary.get("profiler_facts") or whole)
  memory = normalized.get("authority", {}).get("runtime_settings", {}).get("memory_facts")
  records = normalized.get("census", {}).get("records", [])
  reconciliation = normalized.get("reconciliation", {})
  executable_records = [record for record in records if record.get("assignment") in {"graph", "direct"}]
  compiled = sorted({(record.get("source_sha256"), record.get("binary_sha256")) for record in executable_records
                     if record.get("source_sha256") and record.get("binary_sha256")})
  compiled_complete = bool(executable_records) and len(compiled) > 0 and all(
    record.get("source_sha256") and record.get("binary_sha256") for record in executable_records)
  profiler_complete = (profiler.get("measurement_binding_status") == "bound" and
    isinstance(profiler.get("selected_gpu_union_us"), (int, float)) and profiler["selected_gpu_union_us"] > 0 and
    isinstance(profiler.get("selected_command_buffer_count"), int) and profiler["selected_command_buffer_count"] > 0 and
    isinstance(profiler.get("command_buffer_count"), int) and profiler["command_buffer_count"] > 0 and
    profiler.get("selected_command_buffer_count") == reconciliation.get("graph_batches", 0) + reconciliation.get("direct_calls", 0))
  memory_complete = isinstance(memory, Mapping) and memory.get("resident_scope") == "last_committed_graph" and all(
    isinstance(memory.get(key), int) and memory[key] > 0 for key in (
    "recommended_max_working_set_bytes", "current_allocated_bytes", "tinygrad_live_buffer_bytes",
    "resident_buffer_count", "resident_buffer_bytes"))
  return {"profiler":{"measurement_binding_status":profiler.get("measurement_binding_status"),
                       "selected_gpu_union_us":profiler.get("selected_gpu_union_us"),
                       "selected_command_buffer_count":profiler.get("selected_command_buffer_count"),
                       "command_buffer_count":profiler.get("command_buffer_count"),
                       "expected_selected_command_buffer_count":reconciliation.get("graph_batches", 0) + reconciliation.get("direct_calls", 0),
                       "graph_batch_count":reconciliation.get("graph_batches"),
                       "direct_call_count":reconciliation.get("direct_calls"),
                       "dispatch_count":reconciliation.get("logical_calls")},
          "memory":dict(memory) if isinstance(memory, Mapping) else None,
          "compiled":{"identities":[{"source_sha256":source, "binary_sha256":binary} for source,binary in compiled],
                      "record_count":len(executable_records)},
          "gates":{"profiler":profiler_complete, "memory":memory_complete, "compiled":compiled_complete},
          "raw":{"metal_system_trace":auxiliary.get("metal_system_trace_path"),
                 "metal_system_trace_artifacts":trace.get("aux_sources", {}).get("capture_artifacts") if trace else None}}


def _evidence_readiness(rows:list[Mapping[str, Any]], expected:int, stats:Mapping[str, Any]) -> dict[str, Any]:
  blockers = []
  for row in rows:
    if row.get("status") != "valid": continue
    gates = row.get("evidence", {}).get("gates", {})
    for gate in ("profiler", "memory", "compiled"):
      ready = gates.get(gate)
      if ready is not True: blockers.append(f"{row['row_id']}:{gate}")
  if len(rows) != expected: blockers.append("missing_rows")
  if stats.get("complete_pairs") * 2 != expected: blockers.append("missing_pairs")
  bootstrap = stats.get("bootstrap_ci", {})
  if bootstrap.get("status") != "available" or bootstrap.get("resamples") != 10_000: blockers.append("bootstrap_ci")
  if stats.get("median_delta_gpu_union_percent") is None: blockers.append("gpu_union_pairs")
  stability = stats.get("strategy_dispersion", {}).get("status")
  memory_regressions = [row["row_id"] for row in rows if (row.get("pair_reconciliation") or {}).get(
    "process_memory_gate", {}).get("status") == "fail"]
  status = "inconclusive" if blockers else ("refuted" if memory_regressions else ("ready" if stability == "stable" else "unstable"))
  return {"status":status, "blockers":blockers, "performance_stability":stability,
          "process_memory_regressions":memory_regressions,
          "required":["selected_gpu_union", "command_buffer_counts", "batch_direct_dispatch_counts", "memory_residency",
                      "compiled_source_binary_identity", "paired_bootstrap_ci", "stable_arm_dispersion"]}


def _verdict(*, complete:bool, readiness:Mapping[str, Any], stats:Mapping[str, Any]) -> str:
  if not complete or readiness.get("status") == "inconclusive": return "INCONCLUSIVE"
  if readiness.get("status") in {"unstable", "refuted"}: return "REPLAY_REFUTED"
  ci = stats["bootstrap_ci"]
  if stats.get("three_percent_improvement_gate") == "pass" and ci["upper"] < 0: return "REPLAY_WIN"
  if stats.get("median_delta_wall_percent", 0) > 0: return "REPLAY_REFUTED"
  return "REPLAY_NEUTRAL"


def execute_protocol(plan:Mapping[str, Any], *, run_command:CommandRunner = default_command_runner,
                     authority_executor:AuthorityExecutor = run_tinygrad_authority_row,
                     dynamic_probe:Callable[..., dict[str, Any]] = capture_dynamic_validity,
                     command_executor:Callable[..., tuple[int, str, str, Mapping[str, Any] | None]] | None = None) -> dict[str, Any]:
  validate_protocol(plan)
  out = pathlib.Path(plan["output_root"]); out.mkdir(parents=True, exist_ok=False)
  write_json(out / "protocol.json", dict(plan))
  rows, stop_reasons, paired = [], [], {}
  run_execution_identity: dict[str, Any] | None = None
  for position,row_id in enumerate(plan["order"]):
    arm,index_text = row_id.split("-", 1); pair_index = int(index_text)
    row_dir = out / "rows" / row_id; row_dir.mkdir(parents=True)
    started_at = utc_now()
    drift = environment_drift(plan, run_command=run_command)
    before = dynamic_probe(plan["validity"], run_command=run_command)
    if drift or before.get("status") != "valid":
      reasons = [*drift, *before.get("invalidation_reasons", [])]
      row = {"schema":SCHEMA_REPLAY_AB_ROW, "run_id":plan["run_id"], "row_id":row_id,
             "order_index":position, "pair_index":pair_index, "arm":arm, "started_at":started_at,
             "ended_at":utc_now(), "status":"invalidated", "invalidation_reasons":reasons,
             "before":before, "identity":row_identity(plan, ("tinygrad_python",))}
      write_json(row_dir / "row.json", row); rows.append(row); stop_reasons.extend(reasons); break
    try:
      normalized = authority_executor(plan, row_dir, run_command=run_command,
        environment={"METAL_HYBRID_REPLAY":STRATEGIES[arm]["environment"]},
        command_executor=command_executor or _trace_command_executor(plan),
        reconciliation=lambda authority,census: reconcile_replay_census(authority, census, arm=arm))
      observed_identity = replay_execution_identity(normalized)
      if run_execution_identity is None: run_execution_identity = observed_identity
      elif observed_identity != run_execution_identity: raise ValueError("replay execution identity drift")
      if arm == "control": paired[pair_index] = {"control":normalized}
      else:
        if pair_index not in paired: raise ValueError("hybrid row has no preceding paired control")
        pair_reconciliation = reconcile_pair(paired[pair_index]["control"], normalized,
          maximum_process_memory_regression_percent=float(plan.get("validity", {}).get(
            "maximum_process_memory_regression_percent", 3.0)))
        if pair_reconciliation["status"] != "valid":
          raise ValueError("paired replay reconciliation failed: " + ", ".join(pair_reconciliation["errors"]))
        paired[pair_index]["hybrid"] = normalized
      after = dynamic_probe(plan["validity"], run_command=run_command)
      invalidation = [*environment_drift(plan, run_command=run_command), *after.get("invalidation_reasons", [])]
      evidence = _row_evidence(normalized)
      row = {"schema":SCHEMA_REPLAY_AB_ROW, "run_id":plan["run_id"], "row_id":row_id,
             "order_index":position, "pair_index":pair_index, "arm":arm,
             "strategy":STRATEGIES[arm], "started_at":started_at, "ended_at":utc_now(),
             "status":"valid" if not invalidation else "invalidated", "invalidation_reasons":invalidation,
             "workload":plan["workload"], "identity":row_identity(plan, ("tinygrad_python",)),
             "before":before, "after":after, "measurement":normalized["measurement"],
             "output_identity":normalized["output"], "graph_reconciliation":normalized["reconciliation"],
             "pair_reconciliation":pair_reconciliation if arm == "hybrid" else None,
             "evidence":evidence, "raw":{**normalized["raw"], **evidence["raw"]}}
      write_json(row_dir / "row.json", row); rows.append(row)
      if invalidation: stop_reasons.extend(invalidation); break
    except Exception as exc:
      after = dynamic_probe(plan["validity"], run_command=run_command)
      reasons = [f"{type(exc).__name__}: {exc}"]
      row = {"schema":SCHEMA_REPLAY_AB_ROW, "run_id":plan["run_id"], "row_id":row_id,
             "order_index":position, "pair_index":pair_index, "arm":arm, "started_at":started_at,
             "ended_at":utc_now(), "status":"blocked", "invalidation_reasons":reasons,
             "before":before, "after":after, "identity":row_identity(plan, ("tinygrad_python",)),
             "raw_artifacts":[str(path) for path in sorted(row_dir.iterdir()) if path.is_file()]}
      write_json(row_dir / "row.json", row); rows.append(row); stop_reasons.extend(reasons); break
  expected = len(plan["order"]); valid = [row for row in rows if row["status"] == "valid"]
  statistics_result = paired_statistics(rows, int(plan["workload"]["samples_per_strategy"]),
    maximum_relative_mad=float(plan.get("validity", {}).get("maximum_relative_mad", 0.05)))
  observed_order = [row["row_id"] for row in rows]
  order_matches = observed_order == plan["order"]
  complete = len(rows) == expected and len(valid) == expected and statistics_result["complete_pairs"] * 2 == expected and order_matches
  if not order_matches: stop_reasons.append("interleaved_order_mismatch")
  readiness = _evidence_readiness(rows, expected, statistics_result)
  summary = {"schema":SCHEMA_REPLAY_AB_SUMMARY, "run_id":plan["run_id"],
             "status":"complete" if complete else "inconclusive", "completed_at":utc_now(),
             "verdict":_verdict(complete=complete, readiness=readiness, stats=statistics_result),
             "evidence_readiness":readiness,
             "expected_rows":expected, "observed_rows":len(rows), "valid_rows":len(valid),
             "order":observed_order, "required_order":plan["order"],
             "paired_statistics":statistics_result, "stop_reasons":stop_reasons,
             "rows":[str(out / "rows" / row["row_id"] / "row.json") for row in rows]}
  write_json(out / "summary.json", summary)
  return summary


__all__ = ["STRATEGIES", "build_protocol", "execute_protocol", "paired_statistics", "protocol_definition_errors",
           "reconcile_pair", "reconcile_replay_census", "replay_execution_identity", "validate_protocol"]
