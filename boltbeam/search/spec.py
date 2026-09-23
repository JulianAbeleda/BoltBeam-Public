"""Schema authority for the bounded decode/context machine-search layer.

Single source of truth for the *vocabulary* and *record shapes* of the
table-driven search loop:

    search spec -> candidate generator -> isolated runner -> scorer -> accepted policy

This is the BoltBeam-owned port of tinygrad's historical `extra.qk_search_spec`.
It is pure schema/data validation: no hardware execution, no tinygrad imports,
and no runtime default changes.
"""
from __future__ import annotations

import json
import math
import re
import pathlib

from boltbeam.core.canonical import canonical_json as _canonical_json, sha256_hex as _sha256_hex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Phase(str, Enum):
  """Optimization phase; orthogonal search spaces should not be merged."""
  DECODE = "decode"
  LONG_CONTEXT_DECODE = "long_context_decode"
  PREFILL = "prefill"


class Model(str, Enum):
  """Search target model."""
  QWEN3_8B = "qwen3_8b"
  QWEN3_14B = "qwen3_14b"
  QWEN3_32B = "qwen3_32b"


class OpScope(str, Enum):
  """The op a search row targets."""
  Q4K_GEMV = "q4k_gemv"
  Q6K_GEMV = "q6k_gemv"
  ATTENTION = "attention"
  FFN_DOWN = "ffn_down"
  FFN_GATE_UP = "ffn_gate_up"
  LM_HEAD = "lm_head"
  SCHEDULER = "scheduler"


class SearchSpace(str, Enum):
  """The lever a search row explores."""
  PRIMITIVE_POLICY = "primitive_policy"
  DEMOTION = "demotion"
  FLASH_THRESHOLD = "flash_threshold"
  FLASH_VARIANT = "flash_variant"
  STORAGE = "storage"
  SCHEDULE = "schedule"
  LDS_BLOCKING = "lds_blocking"


class Objective(str, Enum):
  """What a search row maximizes/minimizes."""
  TOK_S = "tok_s"
  HBM_PCT = "hbm_pct"
  SERVING_LATENCY = "serving_latency"


# The fork is AMD-only; backend is a schema field so the invariant is explicit
# (a non-AMD backend on a QK primitive path is rejected at the runtime gate too).
BACKENDS: frozenset[str] = frozenset({"AMD"})

# qwen3_8b -> "8B": bridges Model values to the existing LLAMA_REFS / model-bytes keys.
_MODEL_SIZE_KEY: dict[str, str] = {Model.QWEN3_8B.value: "8B", Model.QWEN3_14B.value: "14B", Model.QWEN3_32B.value: "32B"}
LLAMA_REFS: dict[str, float] = {"8B": 101.2, "14B": 65.8, "32B": 30.8}
DEFAULT_MODEL_BYTES: dict[str, int] = {
  "8B": 5_027_783_488,
  "14B": 9_001_752_960,
  "32B": 19_762_149_024,
}
from boltbeam.target.targets import DEFAULT_PEAK_MEM_GBS  # single source: target registry (LN-130)

# BACKENDS is a v1-wire-format artifact: the v2 candidate path carries backend inside the target
# facts and bypasses validate_backend entirely (LN-120 decision record).


def _validate(enum_cls:type[Enum], value:str, label:str) -> str:
  """Return `value` if it is a valid member of `enum_cls`, else raise ValueError."""
  try:
    return enum_cls(value).value
  except ValueError:
    raise ValueError(f"unknown {label} {value!r}; expected one of {sorted(m.value for m in enum_cls)}")


def validate_phase(value:str) -> str: return _validate(Phase, value, "phase")
def validate_model(value:str) -> str: return _validate(Model, value, "model")
def validate_op_scope(value:str) -> str: return _validate(OpScope, value, "op_scope")
def validate_search_space(value:str) -> str: return _validate(SearchSpace, value, "search_space")
def validate_objective(value:str) -> str: return _validate(Objective, value, "objective")


def validate_backend(value:str) -> str:
  if value not in BACKENDS: raise ValueError(f"unknown backend {value!r}; expected one of {sorted(BACKENDS)}")
  return value


def model_size_key(model:str) -> str:
  """Map a Model value (`qwen3_8b`) to the size key (`8B`) used by LLAMA_REFS / model bytes."""
  if model not in _MODEL_SIZE_KEY: raise ValueError(f"unknown model {model!r}; expected one of {sorted(_MODEL_SIZE_KEY)}")
  return _MODEL_SIZE_KEY[model]


def _validate_ctx_range(ctx_range:tuple[int, int]) -> tuple[int, int]:
  if not (isinstance(ctx_range, (tuple, list)) and len(ctx_range) == 2):
    raise ValueError(f"ctx_range must be a 2-tuple, got {ctx_range!r}")
  lo, hi = ctx_range
  if not (isinstance(lo, int) and isinstance(hi, int)) or isinstance(lo, bool) or isinstance(hi, bool):
    raise ValueError(f"ctx_range bounds must be ints, got {ctx_range!r}")
  if lo < 1: raise ValueError(f"ctx_range lower bound must be >= 1, got {lo}")
  if hi < lo: raise ValueError(f"ctx_range must be non-decreasing, got [{lo}, {hi}]")
  return (lo, hi)


FULL_KERNEL_CANDIDATE_SCHEMA_VERSION = "boltbeam.full_kernel_candidate.v1"
FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION = "boltbeam.full_kernel_candidate.v2"
FULL_KERNEL_CANDIDATE_V3_SCHEMA_VERSION = "boltbeam.full_kernel_candidate.v3"
FULL_KERNEL_CANDIDATE_SCHEMA_VERSIONS = frozenset((FULL_KERNEL_CANDIDATE_SCHEMA_VERSION,
                                                   FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION,
                                                   FULL_KERNEL_CANDIDATE_V3_SCHEMA_VERSION))
TargetCapabilityValidator = Callable[[dict[str, Any]], None]


def _strict_keys(d:dict[str, Any], required:set[str], label:str) -> None:
  if not isinstance(d, dict): raise ValueError(f"{label} must be an object")
  missing, unknown = required - set(d), set(d) - required
  if missing: raise ValueError(f"{label} missing fields {sorted(missing)}")
  if unknown: raise ValueError(f"{label} has unknown fields {sorted(unknown)}")


def _positive_int(value:Any, label:str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{label} must be a positive int, got {value!r}")
  return value


def _nonempty_str(value:Any, label:str) -> str:
  if not isinstance(value, str) or not value: raise ValueError(f"{label} must be a non-empty string")
  return value


def _string_list(value:Any, label:str) -> list[str]:
  if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v for v in value):
    raise ValueError(f"{label} must be a non-empty list of strings")
  return value


@dataclass(frozen=True)
class FullKernelCandidate:
  """Strict v1/v2/v3 reader for one compiler-owned full-kernel candidate.

  V1 remains an immutable AMD compatibility wire format. V2 is the one
  target-neutral authoring format: it describes semantic workload identity and a
  backend-neutral plan, while target capability authority is injected by the
  caller. Neither format contains generated code, roofline results, or policy.
  """
  payload: dict[str, Any]
  _canonical_document: str = field(init=False, repr=False, compare=False)

  def __post_init__(self):
    # Normalize through JSON so callers cannot mutate nested values after hashing.
    try: payload = json.loads(json.dumps(self.payload, allow_nan=False))
    except (TypeError, ValueError) as exc: raise ValueError(f"full_kernel_candidate must be JSON data: {exc}") from exc
    self._validate(payload)
    object.__setattr__(self, "payload", payload)
    object.__setattr__(self, "_canonical_document", _canonical_json(payload))

  @staticmethod
  def _validate(p:dict[str, Any]) -> None:
    if not isinstance(p, dict): raise ValueError("full_kernel_candidate must be an object")
    version = p.get("schema_version")
    if version == FULL_KERNEL_CANDIDATE_SCHEMA_VERSION:
      _validate_v1_candidate(p)
    elif version == FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION:
      _validate_v2_candidate(p)
    elif version == FULL_KERNEL_CANDIDATE_V3_SCHEMA_VERSION:
      _validate_v3_candidate(p)
    else:
      raise ValueError(f"unsupported full-kernel candidate schema_version {version!r}")

  @property
  def schema_version(self) -> str: return self.payload["schema_version"]

  @property
  def target(self) -> dict[str, Any]:
    """Fresh exact target identity shared by v1 and v2 callers."""
    return json.loads(self.canonical_json())["workload"]["target"]

  @property
  def workload(self) -> dict[str, Any]:
    """Fresh exact workload identity; safe for request/evidence joins."""
    return json.loads(self.canonical_json())["workload"]

  def common_plan(self) -> dict[str, Any]:
    """Return the version-neutral read model without changing wire identity.

    The adapter deliberately preserves v1's original hash and stored bytes. The
    common plan is for consumers such as expansion/search, never a replacement
    serialization. It keeps workload accounting inputs (shape, dtype, quant and
    layout) separate from schedule choices and carries no computed roofline or
    policy fields.
    """
    p = self.payload
    if self.schema_version in (FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION, FULL_KERNEL_CANDIDATE_V3_SCHEMA_VERSION):
      return json.loads(self.canonical_json())
    workload = p["workload"]
    # V1 has no quantization field. Preserve that absence as ``unknown`` rather
    # than inventing a semantic/roofline input during compatibility reading.
    operands = {name: {"dtype": workload["dtypes"][name], "layout": workload["layout"][name],
                       "quantization": "unknown"} for name in ("a", "b", "c")}
    return {
      "schema_version": "boltbeam.full_kernel_candidate.common.v1",
      "workload": {"profile": workload["profile"], "role": workload["role"],
                   "operation": "matmul", "shape": dict(workload["shape"]),
                   "operands": operands, "accumulator_dtype": workload["dtypes"]["accumulator"],
                   "target": dict(workload["target"])},
      "schedule": {"legacy_v1": json.loads(json.dumps(p["schedule"]))},
      "static_constraints": {"legacy_v1": json.loads(json.dumps(p["static_constraints"]))},
      "applicability": json.loads(json.dumps(p["applicability"])),
    }

  def validate_target(self, validator: TargetCapabilityValidator) -> None:
    """Apply registry-owned target vocabulary/capability validation on demand."""
    if not callable(validator): raise ValueError("target validator must be callable")
    validator(self.target)

  def to_dict(self) -> dict[str, Any]:
    return json.loads(self.canonical_json())

  def canonical_json(self) -> str:
    return self._canonical_document

  @property
  def candidate_hash(self) -> str:
    return _sha256_hex(self.canonical_json().encode("ascii"))

  @staticmethod
  def from_dict(d:dict[str, Any]) -> "FullKernelCandidate":
    return FullKernelCandidate(d)


def _validate_v1_candidate(p:dict[str, Any]) -> None:
    root = {"schema_version", "workload", "schedule", "static_constraints", "applicability"}
    _strict_keys(p, root, "full_kernel_candidate")
    if p["schema_version"] != FULL_KERNEL_CANDIDATE_SCHEMA_VERSION:
      raise ValueError(f"unsupported full-kernel candidate schema_version {p['schema_version']!r}")

    workload = p["workload"]
    _strict_keys(workload, {"profile", "role", "shape", "dtypes", "layout", "target"}, "workload")
    _nonempty_str(workload["profile"], "workload.profile")
    _nonempty_str(workload["role"], "workload.role")
    for group in ("shape", "dtypes", "layout", "target"):
      expected = {"m", "n", "k"} if group == "shape" else ({"a", "b", "c", "accumulator"} if group == "dtypes" else
                 ({"a", "b", "c"} if group == "layout" else {"backend", "arch", "wave_size"}))
      _strict_keys(workload[group], expected, f"workload.{group}")
    for dim in ("m", "n", "k"): _positive_int(workload["shape"][dim], f"workload.shape.{dim}")
    for key in ("a", "b", "c", "accumulator"): _nonempty_str(workload["dtypes"][key], f"workload.dtypes.{key}")
    for key in ("a", "b", "c"): _nonempty_str(workload["layout"][key], f"workload.layout.{key}")
    _nonempty_str(workload["target"]["backend"], "workload.target.backend")
    _nonempty_str(workload["target"]["arch"], "workload.target.arch")
    _positive_int(workload["target"]["wave_size"], "workload.target.wave_size")

    schedule = p["schedule"]
    groups = {"tile", "waves", "threads", "lane_ownership", "cooperative_load", "lds", "pipeline", "wmma",
              "dependency_policy", "residency", "epilogue", "numerical_mode"}
    _strict_keys(schedule, groups, "schedule")
    _strict_keys(schedule["tile"], {"m", "n", "k"}, "schedule.tile")
    _strict_keys(schedule["waves"], {"m", "n"}, "schedule.waves")
    for group in ("tile", "waves"):
      for key, value in schedule[group].items(): _positive_int(value, f"schedule.{group}.{key}")
    _positive_int(schedule["threads"], "schedule.threads")
    _strict_keys(schedule["cooperative_load"], {"a", "b"}, "schedule.cooperative_load")
    for operand in ("a", "b"):
      load = schedule["cooperative_load"][operand]
      _strict_keys(load, {"lane_mapping", "vector_width", "alignment"}, f"schedule.cooperative_load.{operand}")
      _nonempty_str(load["lane_mapping"], f"schedule.cooperative_load.{operand}.lane_mapping")
      _positive_int(load["vector_width"], f"schedule.cooperative_load.{operand}.vector_width")
      _positive_int(load["alignment"], f"schedule.cooperative_load.{operand}.alignment")
    _strict_keys(schedule["lds"], {"windows", "strides", "padding", "banks", "store_vector_width", "load_vector_width"}, "schedule.lds")
    _strict_keys(schedule["pipeline"], {"buffer_count", "stage_count", "epoch_graph"}, "schedule.pipeline")
    _strict_keys(schedule["wmma"], {"instruction_family", "fragment_layout", "accumulator_ownership"}, "schedule.wmma")
    _strict_keys(schedule["dependency_policy"], {"waitcnt", "barriers"}, "schedule.dependency_policy")
    _strict_keys(schedule["residency"], {"preload", "resident", "reuse"}, "schedule.residency")
    _strict_keys(schedule["epilogue"], {"lane_mapping", "vector_width"}, "schedule.epilogue")
    for field in ("lane_ownership", "numerical_mode"): _nonempty_str(schedule[field], f"schedule.{field}")
    for group in ("lds", "pipeline", "wmma", "dependency_policy", "residency", "epilogue"):
      for key, value in schedule[group].items():
        if key in {"buffer_count", "stage_count", "vector_width", "padding", "banks", "store_vector_width", "load_vector_width"}:
          _positive_int(value, f"schedule.{group}.{key}")
        elif key in {"windows", "strides", "epoch_graph", "waitcnt", "barriers", "preload", "resident", "reuse"}:
          if not isinstance(value, (dict, list)): raise ValueError(f"schedule.{group}.{key} must be an object or list")
        else: _nonempty_str(value, f"schedule.{group}.{key}")

    constraints = p["static_constraints"]
    _strict_keys(constraints, {"max_lds_bytes", "max_vgpr_per_thread", "allow_spill"}, "static_constraints")
    _positive_int(constraints["max_lds_bytes"], "static_constraints.max_lds_bytes")
    _positive_int(constraints["max_vgpr_per_thread"], "static_constraints.max_vgpr_per_thread")
    if not isinstance(constraints["allow_spill"], bool): raise ValueError("static_constraints.allow_spill must be bool")

    applicability = p["applicability"]
    _strict_keys(applicability, {"exact_shape", "profiles", "roles", "targets"}, "applicability")
    if not isinstance(applicability["exact_shape"], bool): raise ValueError("applicability.exact_shape must be bool")
    for key in ("profiles", "roles", "targets"): _string_list(applicability[key], f"applicability.{key}")


def _validate_v2_candidate(p:dict[str, Any]) -> None:
  root = {"schema_version", "workload", "schedule", "static_constraints", "correctness", "memory_budget",
          "provenance", "applicability"}
  _strict_keys(p, root, "full_kernel_candidate")
  if p["schema_version"] != FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION:
    raise ValueError(f"unsupported full-kernel candidate schema_version {p['schema_version']!r}")

  workload = p["workload"]
  _strict_keys(workload, {"profile", "model_sha256", "phase", "role", "operation", "shape", "operands",
                          "accumulator_dtype", "target"}, "workload")
  for field in ("profile", "phase", "role", "operation", "accumulator_dtype"):
    _nonempty_str(workload[field], f"workload.{field}")
  if not isinstance(workload["model_sha256"], str) or not _SHA256_RE.fullmatch(workload["model_sha256"]):
    raise ValueError("workload.model_sha256 must be a lowercase SHA-256 hex digest")
  _strict_keys(workload["shape"], {"m", "n", "k"}, "workload.shape")
  for dim in ("m", "n", "k"): _positive_int(workload["shape"][dim], f"workload.shape.{dim}")
  _validate_target_identity(workload["target"])
  _strict_keys(workload["operands"], {"a", "b", "c"}, "workload.operands")
  for name in ("a", "b", "c"):
    operand = workload["operands"][name]
    _strict_keys(operand, {"dtype", "layout", "quantization"}, f"workload.operands.{name}")
    for field in ("dtype", "layout", "quantization"):
      _nonempty_str(operand[field], f"workload.operands.{name}.{field}")

  schedule = p["schedule"]
  _strict_keys(schedule, {"plan_kind", "transforms", "tile", "launch", "mapping", "memory", "pipeline", "compute", "numerical_mode"}, "schedule")
  if schedule["plan_kind"] not in {"tinygrad_heuristic.v1", "tinygrad_opt_sequence.v1"}:
    raise ValueError("schedule.plan_kind must be tinygrad_heuristic.v1 or tinygrad_opt_sequence.v1")
  if not isinstance(schedule["transforms"], list): raise ValueError("schedule.transforms must be an ordered list")
  if schedule["plan_kind"] == "tinygrad_heuristic.v1" and schedule["transforms"]:
    raise ValueError("tinygrad heuristic control cannot carry explicit transforms")
  for index, transform in enumerate(schedule["transforms"]):
    _strict_keys(transform, {"op", "axis", "arg"}, f"schedule.transforms[{index}]")
    _nonempty_str(transform["op"], f"schedule.transforms[{index}].op")
    if transform["axis"] is not None and (not isinstance(transform["axis"], int) or isinstance(transform["axis"], bool)):
      raise ValueError(f"schedule.transforms[{index}].axis must be an int or null")
    _validate_transform_arg(transform["arg"], f"schedule.transforms[{index}].arg")
  _strict_keys(schedule["tile"], {"m", "n", "k"}, "schedule.tile")
  for dim in ("m", "n", "k"): _positive_int(schedule["tile"][dim], f"schedule.tile.{dim}")
  _strict_keys(schedule["launch"], {"threads"}, "schedule.launch")
  _positive_int(schedule["launch"]["threads"], "schedule.launch.threads")
  _strict_keys(schedule["mapping"], {"lane_policy"}, "schedule.mapping")
  _nonempty_str(schedule["mapping"]["lane_policy"], "schedule.mapping.lane_policy")
  _strict_keys(schedule["memory"], {"a", "b", "c"}, "schedule.memory")
  for name in ("a", "b", "c"):
    memory = schedule["memory"][name]
    _strict_keys(memory, {"space", "vector_width", "alignment"}, f"schedule.memory.{name}")
    if memory["space"] not in {"global", "shared", "private"}:
      raise ValueError(f"schedule.memory.{name}.space must be global, shared, or private")
    _positive_int(memory["vector_width"], f"schedule.memory.{name}.vector_width")
    _positive_int(memory["alignment"], f"schedule.memory.{name}.alignment")
  _strict_keys(schedule["pipeline"], {"stage_count"}, "schedule.pipeline")
  _positive_int(schedule["pipeline"]["stage_count"], "schedule.pipeline.stage_count")
  _strict_keys(schedule["compute"], {"family"}, "schedule.compute")
  _nonempty_str(schedule["compute"]["family"], "schedule.compute.family")
  _nonempty_str(schedule["numerical_mode"], "schedule.numerical_mode")

  constraints = p["static_constraints"]
  _strict_keys(constraints, {"max_local_memory_bytes", "max_registers_per_thread", "spill_policy"}, "static_constraints")
  for field in ("max_local_memory_bytes", "max_registers_per_thread"):
    if constraints[field] is not None: _positive_int(constraints[field], f"static_constraints.{field}")
  if constraints["spill_policy"] not in {"forbid", "allow", "unknown"}:
    raise ValueError("static_constraints.spill_policy must be forbid, allow, or unknown")

  correctness = p["correctness"]
  _strict_keys(correctness, {"oracle", "atol", "rtol"}, "correctness")
  _nonempty_str(correctness["oracle"], "correctness.oracle")
  for field in ("atol", "rtol"):
    value = correctness[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
      raise ValueError(f"correctness.{field} must be finite and non-negative")

  memory = p["memory_budget"]
  _strict_keys(memory, {"status", "bytes"}, "memory_budget")
  if memory["status"] not in {"bounded", "unavailable"}: raise ValueError("memory_budget.status must be bounded or unavailable")
  if memory["status"] == "unavailable" and memory["bytes"] is not None:
    raise ValueError("unavailable memory budget must have null bytes")
  if memory["status"] == "bounded": _positive_int(memory["bytes"], "memory_budget.bytes")

  provenance = p["provenance"]
  _strict_keys(provenance, {"generator_id", "generator_revision", "schema_revision"}, "provenance")
  for field in ("generator_id", "generator_revision", "schema_revision"):
    _nonempty_str(provenance[field], f"provenance.{field}")
  if provenance["schema_revision"] != FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION:
    raise ValueError("provenance.schema_revision must match candidate schema_version")

  applicability = p["applicability"]
  _strict_keys(applicability, {"exact_shape", "profiles", "roles", "targets"}, "applicability")
  if not isinstance(applicability["exact_shape"], bool): raise ValueError("applicability.exact_shape must be bool")
  for key in ("profiles", "roles", "targets"): _string_list(applicability[key], f"applicability.{key}")


_PRIMITIVE_PLAN_OPS = frozenset({
  "global_load", "global_store", "workgroup_load", "workgroup_store", "workgroup_barrier", "abs", "add", "sub", "mul", "div",
  "precise_div", "select", "round_away", "clamp", "cast", "bitcast", "and", "or", "shift", "subgroup_shuffle",
  "subgroup_reduce_max", "subgroup_reduce_sum", "pack_bytes", "unpack_bits", "dot", "matrix_mma", "online_softmax",
})


def _validate_v3_candidate(p:dict[str, Any]) -> None:
  """Validate a backend-neutral primitive DAG without interpreting or executing it."""
  root = {"schema_version", "workload", "schedule", "static_constraints", "correctness", "memory_budget",
          "provenance", "applicability"}
  _strict_keys(p, root, "full_kernel_candidate")
  if p["schema_version"] != FULL_KERNEL_CANDIDATE_V3_SCHEMA_VERSION:
    raise ValueError(f"unsupported full-kernel candidate schema_version {p['schema_version']!r}")

  workload = p["workload"]
  _strict_keys(workload, {"profile", "model_sha256", "phase", "role", "operation", "shape", "operands", "target"}, "workload")
  for field in ("profile", "phase", "role", "operation"): _nonempty_str(workload[field], f"workload.{field}")
  if not isinstance(workload["model_sha256"], str) or not _SHA256_RE.fullmatch(workload["model_sha256"]):
    raise ValueError("workload.model_sha256 must be a lowercase SHA-256 hex digest")
  if not isinstance(workload["shape"], dict) or not workload["shape"]:
    raise ValueError("workload.shape must be a non-empty named-axis object")
  for axis, extent in workload["shape"].items():
    _nonempty_str(axis, "workload.shape axis"); _positive_int(extent, f"workload.shape.{axis}")
  if not isinstance(workload["operands"], dict) or not workload["operands"]:
    raise ValueError("workload.operands must be a non-empty named-operand object")
  for name, operand in workload["operands"].items():
    _nonempty_str(name, "workload operand name")
    _strict_keys(operand, {"direction", "dtype", "layout", "quantization"}, f"workload.operands.{name}")
    if operand["direction"] not in {"in", "out", "inout"}: raise ValueError(f"workload.operands.{name}.direction is invalid")
    for field in ("dtype", "layout", "quantization"): _nonempty_str(operand[field], f"workload.operands.{name}.{field}")
  _validate_target_identity(workload["target"])

  schedule = p["schedule"]
  _strict_keys(schedule, {"plan_kind", "launch", "axes", "nodes", "outputs", "parameters"}, "schedule")
  if schedule["plan_kind"] != "tinygrad_primitive_graph.v1":
    raise ValueError("schedule.plan_kind must be tinygrad_primitive_graph.v1")
  launch = schedule["launch"]
  _strict_keys(launch, {"dispatch", "workgroup"}, "schedule.launch")
  for field in ("dispatch", "workgroup"):
    dims = launch[field]
    if not isinstance(dims, list) or len(dims) != 3: raise ValueError(f"schedule.launch.{field} must have three dimensions")
    for index, extent in enumerate(dims): _positive_int(extent, f"schedule.launch.{field}[{index}]")
  axes = schedule["axes"]
  if not isinstance(axes, list) or not axes: raise ValueError("schedule.axes must be a non-empty list")
  axis_names = set()
  for index, axis in enumerate(axes):
    _strict_keys(axis, {"name", "kind", "extent"}, f"schedule.axes[{index}]")
    name = _nonempty_str(axis["name"], f"schedule.axes[{index}].name")
    if name in axis_names: raise ValueError(f"duplicate schedule axis {name!r}")
    axis_names.add(name)
    if axis["kind"] not in {"dispatch", "workgroup", "subgroup", "serial", "reduce"}: raise ValueError(f"schedule.axes[{index}].kind is invalid")
    _positive_int(axis["extent"], f"schedule.axes[{index}].extent")
  nodes = schedule["nodes"]
  if not isinstance(nodes, list) or not nodes: raise ValueError("schedule.nodes must be a non-empty list")
  available = set(workload["operands"]) | axis_names
  for index, node in enumerate(nodes):
    _strict_keys(node, {"id", "op", "inputs", "attrs"}, f"schedule.nodes[{index}]")
    node_id = _nonempty_str(node["id"], f"schedule.nodes[{index}].id")
    if node_id in available: raise ValueError(f"duplicate primitive value {node_id!r}")
    if node["op"] not in _PRIMITIVE_PLAN_OPS: raise ValueError(f"unknown primitive op {node['op']!r}")
    if not isinstance(node["inputs"], list) or any(ref not in available for ref in node["inputs"]):
      raise ValueError(f"schedule.nodes[{index}].inputs must reference operands, axes, or prior nodes")
    if not isinstance(node["attrs"], dict): raise ValueError(f"schedule.nodes[{index}].attrs must be an object")
    available.add(node_id)
  outputs = schedule["outputs"]
  if not isinstance(outputs, list) or not outputs or any(ref not in available for ref in outputs):
    raise ValueError("schedule.outputs must reference generated values")
  if not isinstance(schedule["parameters"], dict): raise ValueError("schedule.parameters must be an object")

  constraints = p["static_constraints"]
  _strict_keys(constraints, {"max_workgroup_memory_bytes", "max_private_memory_bytes", "max_registers_per_thread", "spill_policy"}, "static_constraints")
  for field in ("max_workgroup_memory_bytes", "max_private_memory_bytes", "max_registers_per_thread"):
    if constraints[field] is not None: _positive_int(constraints[field], f"static_constraints.{field}")
  if constraints["spill_policy"] not in {"forbid", "allow", "unknown"}: raise ValueError("static_constraints.spill_policy is invalid")
  correctness = p["correctness"]
  _strict_keys(correctness, {"oracle", "atol", "rtol"}, "correctness"); _nonempty_str(correctness["oracle"], "correctness.oracle")
  for field in ("atol", "rtol"):
    value = correctness[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
      raise ValueError(f"correctness.{field} must be finite and non-negative")
  memory = p["memory_budget"]
  _strict_keys(memory, {"status", "bytes"}, "memory_budget")
  if memory["status"] not in {"bounded", "unavailable"}: raise ValueError("memory_budget.status is invalid")
  if memory["status"] == "bounded": _positive_int(memory["bytes"], "memory_budget.bytes")
  elif memory["bytes"] is not None: raise ValueError("unavailable memory budget must have null bytes")
  provenance = p["provenance"]
  _strict_keys(provenance, {"generator_id", "generator_revision", "schema_revision"}, "provenance")
  for field in provenance: _nonempty_str(provenance[field], f"provenance.{field}")
  if provenance["schema_revision"] != FULL_KERNEL_CANDIDATE_V3_SCHEMA_VERSION:
    raise ValueError("provenance.schema_revision must match candidate schema_version")
  applicability = p["applicability"]
  _strict_keys(applicability, {"exact_shape", "profiles", "roles", "targets"}, "applicability")
  if applicability["exact_shape"] is not True: raise ValueError("primitive plans require exact_shape=true")
  for key in ("profiles", "roles", "targets"): _string_list(applicability[key], f"applicability.{key}")


def _validate_target_identity(target:Any) -> None:
  _strict_keys(target, {"target_id", "backend", "arch", "subgroup_size", "resolved_target_hash"}, "workload.target")
  _nonempty_str(target["target_id"], "workload.target.target_id")
  _nonempty_str(target["backend"], "workload.target.backend")
  _nonempty_str(target["arch"], "workload.target.arch")
  _positive_int(target["subgroup_size"], "workload.target.subgroup_size")
  if not isinstance(target["resolved_target_hash"], str) or not _SHA256_RE.fullmatch(target["resolved_target_hash"]):
    raise ValueError("workload.target.resolved_target_hash must be a lowercase SHA-256 hex digest")


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_transform_arg(value:Any, label:str) -> None:
  if value is None: return
  if isinstance(value, int) and not isinstance(value, bool): return
  if isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value): return
  raise ValueError(f"{label} must be an int, list of ints, or null")


@dataclass(frozen=True)
class Constraints:
  """Bounds a search row must respect; encoded as data and validated on construction."""
  exact_required: bool = True
  dnll_epsilon: float = 0.0
  max_storage_mb: int | None = None
  ctx_range: tuple[int, int] = (1, 4096)
  no_remote_execution: bool = True

  def __post_init__(self):
    if not isinstance(self.exact_required, bool): raise ValueError("exact_required must be bool")
    if not isinstance(self.no_remote_execution, bool): raise ValueError("no_remote_execution must be bool")
    if not isinstance(self.dnll_epsilon, (int, float)) or isinstance(self.dnll_epsilon, bool) or self.dnll_epsilon < 0:
      raise ValueError(f"dnll_epsilon must be a non-negative number, got {self.dnll_epsilon!r}")
    if self.max_storage_mb is not None and (not isinstance(self.max_storage_mb, int) or isinstance(self.max_storage_mb, bool) or self.max_storage_mb <= 0):
      raise ValueError(f"max_storage_mb must be None or a positive int, got {self.max_storage_mb!r}")
    object.__setattr__(self, "ctx_range", _validate_ctx_range(self.ctx_range))

  def to_dict(self) -> dict[str, Any]:
    return {"exact_required": self.exact_required, "dnll_epsilon": self.dnll_epsilon,
            "max_storage_mb": self.max_storage_mb, "ctx_range": list(self.ctx_range),
            "no_remote_execution": self.no_remote_execution}

  @staticmethod
  def from_dict(d:dict[str, Any]) -> "Constraints":
    no_remote = d.get("no_remote_execution", d.get("no_beam_remote", True))
    return Constraints(exact_required=d.get("exact_required", True), dnll_epsilon=d.get("dnll_epsilon", 0.0),
                       max_storage_mb=d.get("max_storage_mb"), ctx_range=tuple(d.get("ctx_range", (1, 4096))),
                       no_remote_execution=no_remote)


@dataclass(frozen=True)
class SearchRow:
  """One bounded search experiment. Constructing it validates every field."""
  row_id: str
  phase: str
  model: str
  op_scope: str
  backend: str
  search_space: str
  objective: str
  constraints: Constraints = field(default_factory=Constraints)
  full_kernel_candidate: FullKernelCandidate | None = None

  def __post_init__(self):
    if not isinstance(self.row_id, str) or not self.row_id: raise ValueError("row_id must be a non-empty string")
    if self.full_kernel_candidate is not None and not isinstance(self.full_kernel_candidate, FullKernelCandidate):
      raise ValueError("full_kernel_candidate must be a FullKernelCandidate or None")
    object.__setattr__(self, "phase", validate_phase(self.phase))
    object.__setattr__(self, "model", validate_model(self.model))
    object.__setattr__(self, "op_scope", validate_op_scope(self.op_scope))
    if self.full_kernel_candidate is not None and self.full_kernel_candidate.schema_version == FULL_KERNEL_CANDIDATE_V2_SCHEMA_VERSION:
      if not isinstance(self.backend, str) or not self.backend:
        raise ValueError("v2 candidate search-row backend must be a non-empty string")
      if self.backend != self.full_kernel_candidate.target["backend"]:
        raise ValueError("v2 candidate search-row backend must match candidate target")
    else:
      object.__setattr__(self, "backend", validate_backend(self.backend))
    object.__setattr__(self, "search_space", validate_search_space(self.search_space))
    object.__setattr__(self, "objective", validate_objective(self.objective))
    if not isinstance(self.constraints, Constraints): raise ValueError("constraints must be a Constraints")

  def to_dict(self) -> dict[str, Any]:
    out = {"id": self.row_id, "phase": self.phase, "model": self.model, "op_scope": self.op_scope,
            "backend": self.backend, "search_space": self.search_space, "objective": self.objective,
            "constraints": self.constraints.to_dict()}
    if self.full_kernel_candidate is not None:
      out["full_kernel_candidate"] = self.full_kernel_candidate.to_dict()
      out["candidate_hash"] = self.full_kernel_candidate.candidate_hash
    return out

  @staticmethod
  def from_dict(d:dict[str, Any]) -> "SearchRow":
    candidate = FullKernelCandidate.from_dict(d["full_kernel_candidate"]) if "full_kernel_candidate" in d else None
    if "candidate_hash" in d and (candidate is None or d["candidate_hash"] != candidate.candidate_hash):
      raise ValueError("candidate_hash does not match full_kernel_candidate")
    return SearchRow(row_id=d["id"], phase=d["phase"], model=d["model"], op_scope=d["op_scope"],
                     backend=d["backend"], search_space=d["search_space"], objective=d["objective"],
                     constraints=Constraints.from_dict(d.get("constraints", {})), full_kernel_candidate=candidate)


def assemble_search_row(*, row_id:str, phase:str, model:str, op_scope:str, backend:str, search_space:str,
                        objective:str, constraints:Constraints | None = None) -> dict[str, Any]:
  """Single source of truth for a search-row dict — validates and returns the canonical shape.

  This is the *only* sanctioned way to add an experiment: a new row, not a new
  script (anti-re-sprawl). Delegates validation to `SearchRow.__post_init__`.
  """
  return SearchRow(row_id=row_id, phase=phase, model=model, op_scope=op_scope, backend=backend,
                   search_space=search_space, objective=objective,
                   constraints=constraints if constraints is not None else Constraints()).to_dict()


@dataclass(frozen=True)
class AcceptedPolicy:
  """A durable, runtime-consumable accepted-policy record (the doc's accepted artifact)."""
  model: str
  phase: str
  backend: str
  ctx_range: tuple[int, int]
  objective: str
  baseline_tok_s: float
  accepted_tok_s: float
  quality_gate: str
  exactness: str
  commit: str
  memory_cap_mb: int | None = None
  hardware: str = "required"

  def __post_init__(self):
    object.__setattr__(self, "phase", validate_phase(self.phase))
    object.__setattr__(self, "model", validate_model(self.model))
    object.__setattr__(self, "backend", validate_backend(self.backend))
    object.__setattr__(self, "objective", validate_objective(self.objective))
    object.__setattr__(self, "ctx_range", _validate_ctx_range(self.ctx_range))
    for name in ("baseline_tok_s", "accepted_tok_s"):
      val = getattr(self, name)
      if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
        raise ValueError(f"{name} must be a non-negative number, got {val!r}")
    for name in ("quality_gate", "exactness", "commit", "hardware"):
      val = getattr(self, name)
      if not isinstance(val, str) or not val: raise ValueError(f"{name} must be a non-empty string")
    if self.memory_cap_mb is not None and (not isinstance(self.memory_cap_mb, int) or isinstance(self.memory_cap_mb, bool) or self.memory_cap_mb <= 0):
      raise ValueError(f"memory_cap_mb must be None or a positive int, got {self.memory_cap_mb!r}")

  def to_dict(self) -> dict[str, Any]:
    return {"model": self.model, "phase": self.phase, "backend": self.backend, "ctx_range": list(self.ctx_range),
            "objective": self.objective, "baseline_tok_s": self.baseline_tok_s, "accepted_tok_s": self.accepted_tok_s,
            "quality_gate": self.quality_gate, "exactness": self.exactness, "memory_cap_mb": self.memory_cap_mb,
            "hardware": self.hardware, "commit": self.commit}

  @staticmethod
  def from_dict(d:dict[str, Any]) -> "AcceptedPolicy":
    return AcceptedPolicy(model=d["model"], phase=d["phase"], backend=d["backend"],
                          ctx_range=tuple(d["ctx_range"]), objective=d["objective"],
                          baseline_tok_s=d["baseline_tok_s"], accepted_tok_s=d["accepted_tok_s"],
                          quality_gate=d["quality_gate"], exactness=d["exactness"], commit=d["commit"],
                          memory_cap_mb=d.get("memory_cap_mb"), hardware=d.get("hardware", "required"))


def baseline(model:str) -> dict[str, Any]:
  """Return the canonical baseline numbers for a model."""
  size = model_size_key(validate_model(model))
  return {"size": size, "llama_tok_s": LLAMA_REFS[size], "model_bytes": DEFAULT_MODEL_BYTES[size],
          "hbm_peak_gbs": DEFAULT_PEAK_MEM_GBS}


def from_generated_policy(policy:dict[str, Any], *, model:str, baseline_tok_s:float, accepted_tok_s:float,
                          ctx_range:tuple[int, int] = (1, 4096), objective:str = Objective.TOK_S.value) -> AcceptedPolicy:
  """Read-only adapter: map an existing `qk_generated_policy` artifact -> AcceptedPolicy.

  Proves the new schema models the real accepted artifacts under
  `bench/qk-shared-storage-20260612/*/policy.json`. Does NOT mutate or write back the
  artifact. tok/s figures come from the experiment matrix (the artifact does not store
  the explicit-vs-generated comparison), so the caller supplies them.
  """
  if policy.get("kind") != "qk_generated_policy":
    raise ValueError(f"not a qk_generated_policy artifact (kind={policy.get('kind')!r})")
  if policy.get("generator_version") not in (0, 1):
    raise ValueError(f"unsupported generator_version {policy.get('generator_version')!r}")
  commit = policy.get("commit")
  if not isinstance(commit, str) or not commit: raise ValueError("artifact missing string commit")
  cap_bytes = (policy.get("storage_policy") or {}).get("cap_bytes")
  memory_cap_mb = None if cap_bytes in (None, 0) else max(1, int(cap_bytes) // (1024 * 1024))
  # These generated policies are exact (Q4_K/Q6_K dequant is lossless) per the decode arc.
  return AcceptedPolicy(model=model, phase=Phase.DECODE.value, backend="AMD", ctx_range=ctx_range,
                        objective=objective, baseline_tok_s=baseline_tok_s, accepted_tok_s=accepted_tok_s,
                        quality_gate="dNLL <= baseline + epsilon", exactness="byte-identical",
                        commit=commit, memory_cap_mb=memory_cap_mb, hardware="required")


def load_search_rows(path:pathlib.Path) -> list[SearchRow]:
  """Load + validate a search-spec table (JSONL, unique ids)."""
  rows: list[dict[str, Any]] = []
  seen: set[str] = set()
  for lineno, line in enumerate(path.read_text().splitlines(), 1):
    line = line.strip()
    if not line:
      continue
    row = json.loads(line)
    if not isinstance(row, dict):
      raise ValueError(f"{path}:{lineno}: expected JSON object")
    row_id = row.get("id")
    if row_id in seen:
      raise ValueError(f"{path}:{lineno}: duplicate id {row_id!r}")
    seen.add(row_id)
    rows.append(row)
  return [SearchRow.from_dict(row) for row in rows]


def save_search_rows(path:pathlib.Path, rows:list[SearchRow]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows))


def load_accepted_policy(path:pathlib.Path) -> AcceptedPolicy:
  data = json.loads(path.read_text())
  if not isinstance(data, dict):
    raise ValueError(f"{path}: expected JSON object")
  return AcceptedPolicy.from_dict(data)


def save_accepted_policy(path:pathlib.Path, policy:AcceptedPolicy) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n")
