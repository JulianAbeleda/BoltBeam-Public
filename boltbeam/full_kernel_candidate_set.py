"""Immutable manifests of exact Tinygrad full-kernel candidates."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable

from boltbeam.core.canonical import canonical_json as _canonical_json, sha256_hex as _sha256_hex
from boltbeam.search.spec import FullKernelCandidate
from boltbeam.target.targets import load_target_registry, target_aliases

SCHEMA = "boltbeam.full_kernel_candidate_set.v1"
QWEN3_8B_PROFILE = "qwen3_8b_q4k_m_gfx1100"
QWEN3_8B_ROLE_SHAPES = (
  ("attn_kv", 512, 1024, 4096),
  ("attn_qo", 512, 4096, 4096),
  ("ffn_down", 512, 4096, 12288),
  ("ffn_gate_up", 512, 12288, 4096),
)

ExactCandidateIndex = tuple[str, str, int, int, int, str, str, int]


def _arch_of(target_id: str) -> str:
  """Registry-owned arch spelling for a target id (its vendor-stripped alias)."""
  aliases = sorted(target_aliases(target_id) - {target_id})
  if len(aliases) != 1:
    raise ValueError(f"cannot derive a single arch spelling for target {target_id!r}: {aliases}")
  return aliases[0]


def _device_neutral_id(profile_id: str, arch: str) -> str:
  """Strip a trailing target-label suffix a profile id happens to carry.

  The model identity is family/size/quant; the ``_gfx1100``-style label is a naming
  artifact, not a fact anything here derives from (tinygrad M1b precedent). Deriving
  the neutral form from the caller-supplied arch instead of hand-writing a second
  literal per profile keeps the two from drifting apart.
  """
  suffix = f"_{arch}"
  return profile_id[:-len(suffix)] if profile_id.endswith(suffix) else profile_id


QWEN3_8B_NEUTRAL_ID = _device_neutral_id(QWEN3_8B_PROFILE, _arch_of("amd_gfx1100"))

# Device-neutral aliases: the same model profile, addressable without naming a target.
# Existing device-labelled ids are retained verbatim and keep resolving unchanged.
PROFILE_ALIASES = {QWEN3_8B_NEUTRAL_ID: QWEN3_8B_PROFILE}


def resolve_profile_id(profile_id: str) -> str:
  """Resolve a profile id (canonical or device-neutral alias) to its canonical spelling."""
  return PROFILE_ALIASES.get(profile_id, profile_id)


def _exact_index(candidate:FullKernelCandidate) -> ExactCandidateIndex:
  workload, applicability = candidate.workload, candidate.payload["applicability"]
  shape, target = workload["shape"], workload["target"]
  if candidate.schema_version == "boltbeam.full_kernel_candidate.v2":
    expected_target = f"{target['target_id']}:subgroup{target['subgroup_size']}"
    target_key, lane_width = target["target_id"], target["subgroup_size"]
  else:
    expected_target = f"{target['backend']}:{target['arch']}:wave{target['wave_size']}"
    target_key, lane_width = target["arch"], target["wave_size"]
  if applicability["exact_shape"] is not True:
    raise ValueError("candidate-set entries must require an exact shape")
  if applicability["profiles"] != [workload["profile"]]:
    raise ValueError("candidate applicability profiles must exactly match its workload profile")
  if applicability["roles"] != [workload["role"]]:
    raise ValueError("candidate applicability roles must exactly match its workload role")
  if applicability["targets"] != [expected_target]:
    raise ValueError("candidate applicability targets must exactly match its workload target")
  return (workload["profile"], workload["role"], shape["m"], shape["n"], shape["k"],
          target["backend"], target_key, lane_width)


@dataclass(frozen=True, init=False)
class FullKernelCandidateEntry:
  canonical_identity: str
  _candidate_json: str = field(repr=False, compare=True)
  index: ExactCandidateIndex

  def __init__(self, canonical_identity:str, candidate:FullKernelCandidate) -> None:
    if not isinstance(candidate, FullKernelCandidate):
      raise ValueError("candidate-set entry candidate must be a FullKernelCandidate")
    if canonical_identity != candidate.candidate_hash:
      raise ValueError("candidate-set entry canonical_identity does not match its payload hash")
    object.__setattr__(self, "canonical_identity", canonical_identity)
    object.__setattr__(self, "_candidate_json", candidate.canonical_json())
    object.__setattr__(self, "index", _exact_index(candidate))

  @property
  def candidate(self) -> FullKernelCandidate:
    return FullKernelCandidate.from_dict(json.loads(self._candidate_json))

  @staticmethod
  def from_candidate(candidate:FullKernelCandidate) -> "FullKernelCandidateEntry":
    return FullKernelCandidateEntry(candidate.candidate_hash, candidate)

  def to_dict(self) -> dict[str, Any]:
    return {"canonical_identity": self.canonical_identity, "payload": json.loads(self._candidate_json)}


@dataclass(frozen=True)
class FullKernelCandidateSet:
  entries: tuple[FullKernelCandidateEntry, ...]
  _canonical_body: str = field(init=False, repr=False, compare=False)
  _by_index: dict[ExactCandidateIndex, FullKernelCandidateEntry] = field(init=False, repr=False, compare=False)

  def __post_init__(self) -> None:
    entries = tuple(sorted(self.entries, key=lambda entry: (entry.index, entry.canonical_identity)))
    if not entries: raise ValueError("candidate set must contain at least one entry")
    if any(not isinstance(entry, FullKernelCandidateEntry) for entry in entries):
      raise ValueError("candidate set entries must be FullKernelCandidateEntry values")
    identities = [entry.canonical_identity for entry in entries]
    if len(identities) != len(set(identities)): raise ValueError("duplicate candidate identity in candidate set")
    indexes = [entry.index for entry in entries]
    if len(indexes) != len(set(indexes)): raise ValueError("duplicate exact candidate index in candidate set")
    body = {"schema": SCHEMA, "entries": [entry.to_dict() for entry in entries]}
    object.__setattr__(self, "entries", entries)
    object.__setattr__(self, "_canonical_body", _canonical_json(body))
    object.__setattr__(self, "_by_index", dict(zip(indexes, entries)))

  @property
  def set_hash(self) -> str: return _sha256_hex(self._canonical_body.encode("ascii"))

  def canonical_json(self) -> str: return self._canonical_body

  def to_dict(self) -> dict[str, Any]:
    return {**json.loads(self._canonical_body), "set_hash": self.set_hash}

  def lookup(self, index:ExactCandidateIndex) -> FullKernelCandidateEntry|None: return self._by_index.get(index)

  @staticmethod
  def build(candidates:Iterable[FullKernelCandidate]) -> "FullKernelCandidateSet":
    return FullKernelCandidateSet(tuple(FullKernelCandidateEntry.from_candidate(candidate) for candidate in candidates))

  @staticmethod
  def from_dict(value:dict[str, Any]) -> "FullKernelCandidateSet":
    if not isinstance(value, dict) or set(value) not in ({"schema", "entries"}, {"schema", "entries", "set_hash"}):
      raise ValueError("candidate-set document has invalid fields")
    if value.get("schema") != SCHEMA: raise ValueError(f"candidate-set schema must be {SCHEMA}")
    if not isinstance(value.get("entries"), list): raise ValueError("candidate-set entries must be a list")
    entries = []
    for row in value["entries"]:
      if not isinstance(row, dict) or set(row) != {"canonical_identity", "payload"}:
        raise ValueError("candidate-set entry has invalid fields")
      entries.append(FullKernelCandidateEntry(str(row["canonical_identity"]), FullKernelCandidate.from_dict(row["payload"])))
    result = FullKernelCandidateSet(tuple(entries))
    if "set_hash" in value and value["set_hash"] != result.set_hash:
      raise ValueError("candidate-set set_hash does not match its canonical entries")
    return result


def build_qwen3_8b_buffer2_candidate_set(seed:FullKernelCandidate|dict[str, Any],
                                         target_id: str = "amd_gfx1100",
                                         schedule_template: dict[str, Any] | None = None) -> FullKernelCandidateSet:
  """Clone one admitted buffer-2 schedule into four exact 8B workload descriptors.

  The model identity comes from the seed; the target is a registry-resolved parameter.
  The legacy default (``amd_gfx1100``) stamps byte-identical output; any registered
  target mints a valid set whose profile id carries that target's arch instead.

  ``schedule_template`` is a tinygrad-derived typed schedule body
  (``{"schedule": ..., "static_constraints": ...}``, option A of
  ``target-schedule-derivation-scope-20260801.md``). When given, the builder
  stops cloning the seed's schedule/static_constraints, validates the template's
  pipeline family, and stamps workload/applicability onto it exactly as it
  stamps identity -- the mint cannot silently inherit another target's
  vocabulary because it must be given a typed schedule.
  """
  registry = load_target_registry()
  if target_id not in registry:
    known = ", ".join(sorted(registry))
    raise ValueError(f"unknown target_id {target_id!r}; known targets: {known}")
  target = registry[target_id]
  arch = _arch_of(target_id)
  seed_candidate = seed if isinstance(seed, FullKernelCandidate) else FullKernelCandidate.from_dict(seed)
  if schedule_template is not None:
    if not isinstance(schedule_template, dict):
      raise ValueError("schedule template must be an object")
    schedule = schedule_template.get("schedule")
    static_constraints = schedule_template.get("static_constraints")
    if not isinstance(schedule, dict) or not isinstance(static_constraints, dict):
      raise ValueError("schedule template must carry schedule and static_constraints objects")
    pipeline = schedule.get("pipeline")
    if not isinstance(pipeline, dict) or (pipeline.get("buffer_count"), pipeline.get("stage_count")) != (2, 1):
      raise ValueError("8B schedule template must describe a two-buffer stage-1 schedule")
  else:
    schedule = seed_candidate.payload["schedule"]
    static_constraints = seed_candidate.payload["static_constraints"]
    pipeline = schedule["pipeline"]
    if (pipeline["buffer_count"], pipeline["stage_count"]) != (2, 1):
      raise ValueError("8B candidate-set seed must describe a two-buffer stage-1 schedule")
  seed_workload = seed_candidate.payload["workload"]
  seed_arch = seed_workload["target"].get("arch") or arch
  if _device_neutral_id(seed_workload["profile"], seed_arch) != QWEN3_8B_NEUTRAL_ID:
    raise ValueError(f"8B candidate-set seed profile must denote the {QWEN3_8B_NEUTRAL_ID} model")
  gate_up = next(shape for role,*shape in QWEN3_8B_ROLE_SHAPES if role == "ffn_gate_up")
  shape = seed_workload["shape"]
  if seed_workload["role"] != "ffn_gate_up" or [shape["m"], shape["n"], shape["k"]] != gate_up:
    raise ValueError("8B candidate-set seed must be the exact ffn_gate_up workload")
  profile_id = f"{QWEN3_8B_NEUTRAL_ID}_{arch}"
  expected_target = f"{target.backend}:{arch}:wave{target.wave_size}"
  out = []
  for role,m,n,k in QWEN3_8B_ROLE_SHAPES:
    payload = seed_candidate.to_dict()
    workload = payload["workload"]
    workload["profile"], workload["role"] = profile_id, role
    workload["shape"] = {"m":m, "n":n, "k":k}
    workload["target"] = {"backend": target.backend, "arch": arch, "wave_size": target.wave_size}
    payload["applicability"] = {"exact_shape":True, "profiles":[profile_id], "roles":[role],
                                 "targets":[expected_target]}
    if schedule_template is not None:
      payload["schedule"] = json.loads(json.dumps(schedule))
      payload["static_constraints"] = json.loads(json.dumps(static_constraints))
    out.append(FullKernelCandidate(payload))
  return FullKernelCandidateSet.build(out)


def candidate_from_proven_document(value:Any) -> FullKernelCandidate:
  """Extract one full-kernel candidate from a supported BoltBeam artifact envelope."""
  candidates:list[tuple[dict[str, Any], str|None]] = []

  def collect(node:Any) -> None:
    if not isinstance(node, dict):
      raise ValueError("candidate document must be a JSON object")
    if node.get("schema_version") == "boltbeam.full_kernel_candidate.v1":
      candidates.append((node, None))
    elif "payload" in node and ("canonical_identity" in node or "candidate_hash" in node):
      identity = node.get("canonical_identity", node.get("candidate_hash"))
      candidates.append((node["payload"], str(identity)))
    elif "full_kernel_candidate" in node:
      candidates.append((node["full_kernel_candidate"],
                         str(node["candidate_hash"]) if "candidate_hash" in node else None))
    elif isinstance(node.get("rows"), list):
      for row in node["rows"]:
        if not isinstance(row, dict) or not isinstance(row.get("search_row"), dict):
          raise ValueError("candidate manifest rows must contain search_row objects")
        collect(row["search_row"])
    else:
      raise ValueError("unsupported candidate document envelope")

  collect(value)
  if len(candidates) != 1:
    raise ValueError(f"candidate document must resolve to exactly one candidate, found {len(candidates)}")
  payload, supplied_identity = candidates[0]
  candidate = FullKernelCandidate.from_dict(payload)
  if supplied_identity is not None and supplied_identity != candidate.candidate_hash:
    raise ValueError("supplied candidate identity does not match its payload hash")
  return candidate


def deterministic_candidate_set_json(candidate_set:FullKernelCandidateSet) -> str:
  """Serialize stable bytes in Tinygrad's strict candidate-set transport schema."""
  if not isinstance(candidate_set, FullKernelCandidateSet):
    raise ValueError("candidate_set must be a FullKernelCandidateSet")
  return json.dumps(json.loads(candidate_set.canonical_json()), indent=2, sort_keys=True,
                    ensure_ascii=True, allow_nan=False) + "\n"
