"""Provider-neutral calibration microbenchmark contracts and orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from boltbeam.artifacts.base import sha256_json

SPEC_SCHEMA = "boltbeam.mmq_calibration_spec.v1"
MEASUREMENT_SCHEMA = "boltbeam.mmq_calibration_measurement.v1"
RUN_SCHEMA = "boltbeam.mmq_calibration_run.v1"
KINDS = frozenset(("launch", "dependent_latency", "independent_throughput", "issue_compatibility",
                   "global_memory", "global_store", "lds", "barrier_wait", "occupancy", "grid_tail"))


@dataclass(frozen=True)
class CalibrationSpec:
  benchmark_id: str
  parameter_name: str
  benchmark_kind: str
  unit: str
  argv: tuple[str, ...]
  system_snapshot_id: str
  warmups: int = 3
  repetitions: int = 30
  order: str = "randomized_interleaved"
  seed: int = 0
  controls: Mapping[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    if not self.benchmark_id or "." not in self.parameter_name: raise ValueError("benchmark id and namespaced parameter are required")
    if self.benchmark_kind not in KINDS: raise ValueError(f"unknown benchmark kind {self.benchmark_kind!r}")
    if not self.unit or not self.system_snapshot_id: raise ValueError("unit and system snapshot are required")
    if not self.argv or any(not isinstance(x, str) or not x for x in self.argv): raise ValueError("argv array is required")
    if self.warmups < 3 or self.repetitions < 30: raise ValueError("calibration requires 3 warmups and 30 repetitions")
    if self.order != "randomized_interleaved": raise ValueError("calibration order must be randomized_interleaved")

  @property
  def spec_id(self) -> str: return sha256_json(self.semantic_json())

  def semantic_json(self) -> dict[str, Any]:
    return {"benchmark_id": self.benchmark_id, "parameter_name": self.parameter_name,
      "benchmark_kind": self.benchmark_kind, "unit": self.unit, "argv": list(self.argv),
      "system_snapshot_id": self.system_snapshot_id, "warmups": self.warmups, "repetitions": self.repetitions,
      "order": self.order, "seed": self.seed, "controls": dict(self.controls)}

  def to_json(self) -> dict[str, Any]: return {"schema": SPEC_SCHEMA, "spec_id": self.spec_id, **self.semantic_json()}

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "CalibrationSpec":
    if data.get("schema") != SPEC_SCHEMA: raise ValueError(f"expected {SPEC_SCHEMA}")
    spec = CalibrationSpec(data["benchmark_id"], data["parameter_name"], data["benchmark_kind"], data["unit"],
      tuple(data["argv"]), data["system_snapshot_id"], int(data.get("warmups", 3)), int(data.get("repetitions", 30)),
      data.get("order", "randomized_interleaved"), int(data.get("seed", 0)), dict(data.get("controls", {})))
    if data.get("spec_id") != spec.spec_id: raise ValueError("spec_id mismatch")
    return spec


@dataclass(frozen=True)
class CalibrationMeasurement:
  spec: CalibrationSpec
  status: str
  samples: tuple[float, ...] = ()
  clock_hz_samples: tuple[float, ...] = ()
  producer: str = ""
  error: str = ""

  def __post_init__(self):
    if self.status not in ("measured", "blocked", "unsupported"): raise ValueError("invalid measurement status")
    if self.status == "measured":
      if len(self.samples) < self.spec.repetitions: raise ValueError("measured calibration has too few samples")
      if any(not isinstance(x, (int, float)) or x <= 0 for x in self.samples): raise ValueError("samples must be positive")
      if len(self.clock_hz_samples) < self.spec.repetitions or any(x <= 0 for x in self.clock_hz_samples):
        raise ValueError("measured calibration requires per-sample clocks")
      if not self.producer: raise ValueError("measured calibration requires producer identity")
    elif not self.error: raise ValueError("non-measured calibration requires an error")

  @property
  def measurement_id(self) -> str: return sha256_json(self.semantic_json())

  def semantic_json(self) -> dict[str, Any]:
    return {"spec": self.spec.to_json(), "status": self.status, "samples": list(self.samples),
            "clock_hz_samples": list(self.clock_hz_samples), "producer": self.producer, "error": self.error}

  def to_json(self) -> dict[str, Any]: return {"schema": MEASUREMENT_SCHEMA, "measurement_id": self.measurement_id, **self.semantic_json()}

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "CalibrationMeasurement":
    if data.get("schema") != MEASUREMENT_SCHEMA: raise ValueError(f"expected {MEASUREMENT_SCHEMA}")
    row = CalibrationMeasurement(CalibrationSpec.from_json(data["spec"]), data["status"], tuple(data.get("samples", ())),
                                 tuple(data.get("clock_hz_samples", ())), data.get("producer", ""), data.get("error", ""))
    if data.get("measurement_id") != row.measurement_id: raise ValueError("measurement_id mismatch")
    return row


@dataclass(frozen=True)
class CalibrationRun:
  measurements: tuple[CalibrationMeasurement, ...]

  def to_json(self) -> dict[str, Any]:
    return {"schema": RUN_SCHEMA, "measurements": [m.to_json() for m in self.measurements]}

  @staticmethod
  def from_json(data: Mapping[str, Any]) -> "CalibrationRun":
    if data.get("schema") != RUN_SCHEMA: raise ValueError(f"expected {RUN_SCHEMA}")
    return CalibrationRun(tuple(CalibrationMeasurement.from_json(row) for row in data.get("measurements", ())))


def run_calibration_plan(specs: tuple[CalibrationSpec, ...], producer: Any) -> CalibrationRun:
  # producer duck contract: measure(spec) -> CalibrationMeasurement. No real producer class exists yet;
  # the Protocol that used to declare this had zero implementations (LN-120).
  """Failure isolation is intentional: one blocked microbenchmark does not stop the matrix."""
  rows = []
  for spec in specs:
    try: measurement = producer.measure(spec)
    except Exception as exc:
      measurement = CalibrationMeasurement(spec, "blocked", producer=type(producer).__name__, error=str(exc))
    if measurement.spec.spec_id != spec.spec_id: raise ValueError("producer returned measurement for the wrong spec")
    rows.append(measurement)
  return CalibrationRun(tuple(rows))


def mmq_cycle_calibration_plan(system_snapshot_id: str, argv_prefix: tuple[str, ...], *, seed: int = 0) -> tuple[CalibrationSpec, ...]:
  """Minimum independent training matrix for every parameter consumed by cycle_model v1."""
  if not argv_prefix: raise ValueError("calibration producer argv prefix is required")
  rows = [
    ("clock", "gpu.clock_hz", "launch", "Hz"),
    ("launch", "perf.launch_cycles", "launch", "cycles"),
    ("tail", "perf.tail_efficiency", "grid_tail", "fraction"),
  ]
  rows += [(f"issue-{domain}", f"perf.issue_rate.{domain}", "independent_throughput", "instructions/cycle")
           for domain in ("valu", "salu", "vmem", "lds")]
  classes = ("branch_predicate", "global_load", "global_store", "lds_load", "lds_store", "salu",
             "valu_float", "valu_int", "waitcnt")
  rows += [(f"latency-{kind}", f"perf.latency.{kind}", "dependent_latency" if kind != "waitcnt" else "barrier_wait", "cycles")
           for kind in classes]
  controls = {"training_scope": "generated_microbenchmark", "final_holdout_excluded": True}
  return tuple(CalibrationSpec(benchmark_id, parameter, kind, unit,
    (*argv_prefix, "--benchmark-id", benchmark_id, "--seed", str(seed)), system_snapshot_id,
    seed=seed, controls=controls) for benchmark_id, parameter, kind, unit in rows)


def scheduling_wall_transfer_plan(system_snapshot_id: str, argv_prefix: tuple[str, ...], *, seed: int = 0) -> tuple[CalibrationSpec, ...]:
  """Post-v2 residual probe: leave the launch-dominated timing regime without using MMQ candidates."""
  rows = []
  for policy in ("profile_standard", "auto"):
    for chain in (1024, 4096, 16384):
      benchmark_id = f"long-valu.{policy}.n{chain}"
      controls = {"training_scope": "generated_microbenchmark", "final_holdout_excluded": True,
                  "clock_policy": policy, "chain_length": chain,
                  "required_counters": ["SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_BUSY_CYCLES"],
                  "sq_wait_any_policy": "overlap_diagnostic_only_not_additive", "live_clock_required": True}
      rows.append(CalibrationSpec(benchmark_id, f"perf.wall_transfer.{policy}.n{chain}",
        "independent_throughput", "ms", (*argv_prefix, "--benchmark-id", benchmark_id, "--seed", str(seed)),
        system_snapshot_id, seed=seed, controls=controls))
  return tuple(rows)


def kernel_runner_scope_plan(system_snapshot_id: str, argv_prefix: tuple[str, ...], *, seed: int = 0) -> tuple[CalibrationSpec, ...]:
  """Bound kernel/device time separately from runner setup, synchronization, and readback."""
  if not argv_prefix: raise ValueError("calibration producer argv prefix is required")
  rows = []
  for repeats in (1, 16, 64, 256):
    benchmark_id = f"looped-valu.auto.r{repeats}"
    controls = {"training_scope": "generated_microbenchmark", "final_holdout_excluded": True,
      "clock_policy": "auto", "runtime_repetitions": repeats, "source_unrolled": False,
      "compile_once": True, "preallocated_buffers": True,
      "timing_channels": ["kernel_device_ms", "enqueue_sync_ms", "setup_ms", "readback_ms", "invocation_wall_ms"],
      "required_counters": ["SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_BUSY_CYCLES"],
      "sq_wait_any_policy": "overlap_diagnostic_only_not_additive", "live_clock_required": True}
    rows.append(CalibrationSpec(benchmark_id, f"perf.kernel_runner_scope.auto.r{repeats}",
      "independent_throughput", "ms", (*argv_prefix, "--benchmark-id", benchmark_id, "--seed", str(seed)),
      system_snapshot_id, seed=seed, controls=controls))
  return tuple(rows)


def false_site_grid_transfer_plan(system_snapshot_id: str, argv_prefix: tuple[str, ...], *, seed: int = 0) -> tuple[CalibrationSpec, ...]:
  """Test whether false branch/store cost transfers from one workgroup to candidate-scale grids."""
  if not argv_prefix: raise ValueError("calibration producer argv prefix is required")
  rows=[]
  for workgroups in (1,32,96,256):
    for false_sites in (0,128,256):
      for lds_stage in (False,True):
        benchmark_id=f"false-grid.wg{workgroups}.n{false_sites}.lds{int(lds_stage)}"
        controls={"training_scope":"generated_microbenchmark","final_holdout_excluded":True,
          "target_metric":"kernel_device_ms","gpu_timestamp_only":True,"workgroups":workgroups,
          "false_sites":false_sites,"lds_stage":lds_stage,"candidate_binary":False,
          "matched_resources_required":True,"production_dispatch_changed":False}
        rows.append(CalibrationSpec(benchmark_id,f"perf.false_site_grid.{benchmark_id}","grid_tail","ms",
          (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def exact_isa_frontend_plan(system_snapshot_id: str, argv_prefix: tuple[str,...], *, seed:int=0)->tuple[CalibrationSpec,...]:
  """Generated front-end probe admitted by final ISA counts rather than source-level knobs."""
  rows=[]
  for sites in (0,64,128,256):
    benchmark_id=f"exact-frontend.wg256.n{sites}"
    controls={"training_scope":"generated_microbenchmark","final_holdout_excluded":True,
      "target_metric":"kernel_device_ms","gpu_timestamp_only":True,"workgroups":256,"candidate_binary":False,
      "admission":"final_isa_exact","required_final_isa":{"scalar_branch_sites":sites,
        "global_store_b32_sites":sites,"predicate_sites_min":2*sites},
      "prevent_vectorization":True,"prevent_branch_collapse":True,"matched_resources_required":True}
    rows.append(CalibrationSpec(benchmark_id,f"perf.exact_frontend.n{sites}","independent_throughput","ms",
      (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def host_invocation_structure_plan(system_snapshot_id:str,argv_prefix:tuple[str,...],*,seed:int=0)->tuple[CalibrationSpec,...]:
  """Independent host probe with MMQ-shaped fixed structure and separately timed orchestration phases."""
  rows=[]
  for sites in (0,64,128,256):
    benchmark_id=f"host-mmq-structure.n{sites}"
    controls={"training_scope":"generated_host_microbenchmark","final_holdout_excluded":True,"false_sites":sites,
      "candidate_binary":False,"mmq_shaped_fixed_uops":True,"preallocated_device_buffers":True,
      "timing_channels":["custom_kernel_uop_construction","schedule_creation","input_construct_realize_transfer",
        "output_allocation","compile_cache_lookup","enqueue_sync_overhead","output_readback","lifecycle_validation_views"],
      "kernel_device_time_excluded":"modeled_separately_by_v7"}
    rows.append(CalibrationSpec(benchmark_id,f"perf.host_invocation_structure.n{sites}","launch","ms",
      (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def exact_host_structure_plan(system_snapshot_id:str,argv_prefix:tuple[str,...],*,seed:int=0)->tuple[CalibrationSpec,...]:
  """Separate fixed graph/source complexity from false-site growth in host construction and scheduling."""
  rows=[]
  for base_uops in (32,256,768):
    for sites in (0,128,256):
      benchmark_id=f"host-exact.u{base_uops}.n{sites}"
      controls={"training_scope":"generated_host_microbenchmark","final_holdout_excluded":True,"candidate_binary":False,
        "base_sink_uops":base_uops,"false_sites":sites,"rendered_source_identity_required":True,
        "timing_channels":["uop_construction","schedule_creation","warmed_compile_cache_lookup"],
        "device_time_excluded":True,"preallocated_device_buffers":True}
      rows.append(CalibrationSpec(benchmark_id,f"perf.host_exact_structure.u{base_uops}.n{sites}","launch","ms",
        (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def gated_host_topology_plan(system_snapshot_id:str,argv_prefix:tuple[str,...],*,seed:int=0)->tuple[CalibrationSpec,...]:
  """Minimal high-base check of generated false-site topology after v3 baseline extension."""
  rows=[]
  for base_uops in (1024,1280):
    for sites in (0,255):
      benchmark_id=f"host-gated-topology.u{base_uops}.n{sites}"
      controls={"training_scope":"generated_host_microbenchmark","final_holdout_excluded":True,"candidate_binary":False,
        "base_sink_uops":base_uops,"false_sites":sites,"candidate_shaped_false_site_topology":True,
        "exact_sink_uop_and_source_identity_required":True,"device_time_excluded":True,
        "timing_channels":["uop_construction","schedule_creation"]}
      rows.append(CalibrationSpec(benchmark_id,f"perf.host_gated_topology.u{base_uops}.n{sites}","launch","ms",
        (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def ownership_and_fanin_plan(system_snapshot_id:str,argv_prefix:tuple[str,...],*,seed:int=0)->tuple[CalibrationSpec,...]:
  """Minimal exact-UOp probe for the isolated ownership-predicate AND fan-in mismatch."""
  rows=[]
  for variant,sites,and_sites in (("baseline",0,0),("single_and",255,256),("double_and",255,510)):
    benchmark_id=f"host-owner-fanin.{variant}"
    controls={"training_scope":"generated_host_microbenchmark","final_holdout_excluded":True,"candidate_binary":False,
      "base_sink_uops":405,"false_sites":sites,"ownership_and_sites":and_sites,"topology_variant":variant,
      "exact_uop_admission":{"store_sites":4+sites,"index_sites":39+sites,"cmpne_sites":2+(64 if sites else 0),"and_sites":34+and_sites},
      "device_time_excluded":True,"timing_channels":["uop_construction","schedule_creation"]}
    rows.append(CalibrationSpec(benchmark_id,f"perf.host_owner_fanin.{variant}","launch","ms",
      (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def ownership_scaffolding_plan(system_snapshot_id:str,argv_prefix:tuple[str,...],*,seed:int=0)->tuple[CalibrationSpec,...]:
  """Vary only noncandidate CMPLT scaffolding around the exact ownership delta."""
  rows=[]
  for cmplt in (0,128,256):
    benchmark_id=f"host-owner-scaffold.c{cmplt}"
    controls={"training_scope":"generated_host_microbenchmark","final_holdout_excluded":True,"candidate_binary":False,
      "core_delta":{"STORE":255,"INDEX":255,"AND":256,"CMPNE":64},"cmplt_scaffolding":cmplt,
      "exact_full_histogram_required":True,"candidate_total_uops_target":1246,"device_time_excluded":True,
      "timing_channels":["uop_construction","schedule_creation","warmed_compile_cache_lookup"]}
    rows.append(CalibrationSpec(benchmark_id,f"perf.host_owner_scaffold.c{cmplt}","launch","ms",
      (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def uop_backbone_topology_plan(system_snapshot_id:str,argv_prefix:tuple[str,...],*,seed:int=0)->tuple[CalibrationSpec,...]:
  """Hold total/core ownership fixed while restoring candidate arithmetic and control backbone topology."""
  rows=[]
  for variant in ("comparison_filler","arithmetic_backbone","full_candidate_backbone"):
    benchmark_id=f"host-uop-backbone.{variant}"
    controls={"training_scope":"generated_host_microbenchmark","final_holdout_excluded":True,"candidate_binary":False,
      "sink_uops":1246,"core_delta":{"STORE":255,"INDEX":255,"AND":256,"CMPNE":64},"variant":variant,
      "exact_full_histogram_required":True,"dependency_depth_recorded":True,"python_construction_events_recorded":True,
      "device_time_excluded":True,"timing_channels":["uop_construction","schedule_creation"]}
    rows.append(CalibrationSpec(benchmark_id,f"perf.host_uop_backbone.{variant}","launch","ms",
      (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def uop_depth_sharing_plan(system_snapshot_id:str,argv_prefix:tuple[str,...],*,seed:int=0)->tuple[CalibrationSpec,...]:
  """Minimal fixed-size factorial for dependency depth and common-subexpression fanout."""
  rows=[]
  for depth in (32,64):
    for fanout in (256,512):
      benchmark_id=f"host-uop-shape.d{depth}.f{fanout}"
      controls={"training_scope":"generated_host_microbenchmark","final_holdout_excluded":True,"candidate_binary":False,
        "sink_uops":1246,"dependency_depth_target":depth,"max_fanout_target":fanout,
        "exact_opcode_histogram_fixed":True,"edge_count_target_range":[3050,3200],"shared_extra_edges_target_range":[1800,1950],
        "python_builder_event_count_fixed":True,"device_time_excluded":True,"timing_channels":["uop_construction"]}
      rows.append(CalibrationSpec(benchmark_id,f"perf.host_uop_shape.d{depth}.f{fanout}","launch","ms",
        (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def python_builder_event_plan(system_snapshot_id:str,argv_prefix:tuple[str,...],*,seed:int=0)->tuple[CalibrationSpec,...]:
  """Factor attempted Python helper calls that persistent UOp canonicalization hides from final graphs."""
  rows=[]
  for quant_groups in (0,8):
    for reduce_expansions in (0,1):
      for writeback_iterations in (1,256):
        benchmark_id=f"host-builder.q{quant_groups}.r{reduce_expansions}.w{writeback_iterations}"
        controls={"training_scope":"generated_host_microbenchmark","final_holdout_excluded":True,"candidate_binary":False,
          "quant_group_helper_calls":quant_groups,"quant_helper_calls":quant_groups,"reduce_expansion_calls":reduce_expansions,
          "writeback_iterations":writeback_iterations,"expected_eq_call_attempts":8+2*writeback_iterations,
          "expected_store_call_attempts":3+writeback_iterations,"final_sink_uops":1246,"final_opcode_histogram_fixed":True,
          "instrumentation":"explicit_helper_timers","timer_overhead_calibration_samples":2000,
          "profile_crosscheck":"untimed_sys_setprofile_call_counts","record_per_helper_samples":True,
          "device_time_excluded":True,"timing_channels":["total_uop_construction","group_params","quant","reduce","eq_attempts","store_attempts","canonicalization_residual"]}
        rows.append(CalibrationSpec(benchmark_id,f"perf.host_builder_events.q{quant_groups}.r{reduce_expansions}.w{writeback_iterations}","launch","ms",
          (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


def writeback_operand_interaction_plan(system_snapshot_id:str,argv_prefix:tuple[str,...],*,seed:int=0)->tuple[CalibrationSpec,...]:
  """Isolate writeback event count from equality/index/store operand construction complexity."""
  rows=[]
  for iterations in (1,256):
    for operands in ("simple_const_1d","candidate_shaped_special_2d_deep_value"):
      benchmark_id=f"host-writeback-operands.w{iterations}.{operands}"
      controls={"training_scope":"generated_host_microbenchmark","final_holdout_excluded":True,"candidate_binary":False,
        "writeback_iterations":iterations,"operand_topology":operands,"final_sink_uops":1246,"final_opcode_histogram_fixed":True,
        "eq_lhs":"const" if operands.startswith("simple") else "shared_special_gidx",
        "store_index_rank":1 if operands.startswith("simple") else 2,"store_value":"const" if operands.startswith("simple") else "shared_deep_value",
        "instrumentation":"explicit_suboperation_timers","timer_overhead_calibration_samples":2000,
        "profile_crosscheck":"untimed exact call counts","device_time_excluded":True,
        "timing_channels":["eq_operand_build","eq_call","index_operand_build","store_call","hash_canonicalization","total"]}
      rows.append(CalibrationSpec(benchmark_id,f"perf.host_writeback_operands.w{iterations}.{operands}","launch","ms",
        (*argv_prefix,"--benchmark-id",benchmark_id,"--seed",str(seed)),system_snapshot_id,seed=seed,controls=controls))
  return tuple(rows)


__all__ = ["CalibrationMeasurement", "CalibrationRun", "CalibrationSpec",
           "exact_host_structure_plan", "exact_isa_frontend_plan", "false_site_grid_transfer_plan", "gated_host_topology_plan", "host_invocation_structure_plan", "kernel_runner_scope_plan", "mmq_cycle_calibration_plan", "ownership_and_fanin_plan", "ownership_scaffolding_plan", "python_builder_event_plan", "run_calibration_plan", "uop_backbone_topology_plan", "uop_depth_sharing_plan", "writeback_operand_interaction_plan",
           "scheduling_wall_transfer_plan"]
