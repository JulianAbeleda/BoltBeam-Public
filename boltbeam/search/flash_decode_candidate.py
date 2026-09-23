"""Flash decode candidate descriptor vocabulary for the generic search loop.

BoltBeam proposes flash decode geometry as a hashed descriptor, exactly like
``FullKernelCandidate``: the tile and (optional) combine fields are the wire
payload, every field participates in canonical JSON, and the candidate hash is
the sha256 of that canonical document. A change to any field is a new identity
and therefore a distinct measured binary.

This module is deliberately free of vendor logic: legality is checked against
the candidate's own target and caller-supplied target facts, one key per fact
(``boltbeam.search.target_fact_keys``), never by backend or architecture names.
Tinygrad owns admission/codegen; this module only validates descriptors and
fact-driven legality so the population, static-assessment
(BubbleBeam/FutureSight), and provider layers share one identity and legality
authority.
"""
from __future__ import annotations

import hashlib, json, math, re
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from boltbeam.search.target_fact_keys import MAX_LOCAL_MEMORY, MAX_THREADS, read_target_limits

FLASH_DECODE_CANDIDATE_SCHEMA_VERSION = "boltbeam.flash_decode_candidate.v1"
REDUCE_STRUCTURES = frozenset({"staged", "inline"})
# KV_BOTH stages K and V blocks in shared memory; K_ONLY stages K and reads V from global memory.
STAGING_MODES = frozenset({"KV_BOTH", "K_ONLY"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TargetCapabilityValidator = Callable[[dict[str, Any]], None]


def _canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _strict_keys(d: dict[str, Any], required: set[str], label: str) -> None:
  if not isinstance(d, dict): raise ValueError(f"{label} must be an object")
  missing, unknown = required - set(d), set(d) - required
  if missing: raise ValueError(f"{label} missing fields {sorted(missing)}")
  if unknown: raise ValueError(f"{label} has unknown fields {sorted(unknown)}")


def _positive_int(value: Any, label: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{label} must be a positive int, got {value!r}")
  return value


def _optional_positive_int(value: Any, label: str) -> int | None:
  if value is None: return None
  return _positive_int(value, label)


def _power_of_two(value: int, label: str) -> int:
  # Shuffle reduces are XOR butterflies: offsets width/2..1 pair lanes only for a power-of-two width.
  if value & (value - 1): raise ValueError(f"{label} must be a power of two, got {value}")
  return value


def _one_of(value: Any, allowed: frozenset[str], label: str) -> str:
  if not isinstance(value, str) or value not in allowed:
    raise ValueError(f"{label} must be one of {sorted(allowed)}, got {value!r}")
  return value


def _nonempty_str(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value: raise ValueError(f"{label} must be a non-empty string")
  return value


def _bool(value: Any, label: str) -> bool:
  if not isinstance(value, bool): raise ValueError(f"{label} must be a bool, got {value!r}")
  return value


@dataclass(frozen=True)
class FlashDecodeTileSpec:
  """Tile geometry for one flash decode candidate.

  Defaults reproduce the current production descriptor: 32 physical lanes, one
  score group per lane, workgroup warps derived from the query group, a 16-token
  block, a single stage width, and the staged shuffle-ladder reduce. ``warps``
  and ``query_group_size`` are ``None`` until resolved by the derived geometry.
  """
  Hq: int
  Hd: int
  Hkv: int
  MAXC: int
  split_count: int
  staging: str = "KV_BOTH"
  quant: bool = False
  rope: bool = False
  token_block: int = 16
  lane_width: int = 32
  score_group_width: int | None = None
  warps: int | None = None
  query_group_size: int | None = None
  stage_width: int = 1
  reduce_structure: str = "staged"
  dot_pair_width: int = 2

  def __post_init__(self) -> None:
    for name in ("Hq", "Hd", "Hkv", "MAXC", "split_count", "token_block", "lane_width", "stage_width",
                 "dot_pair_width"):
      _positive_int(getattr(self, name), f"tile.{name}")
    for name in ("score_group_width", "warps", "query_group_size"):
      _optional_positive_int(getattr(self, name), f"tile.{name}")
    if self.Hq % self.Hkv != 0:
      raise ValueError(f"tile.Hq must be divisible by tile.Hkv, got Hq={self.Hq} Hkv={self.Hkv}")
    _one_of(self.staging, STAGING_MODES, "tile.staging")
    _bool(self.quant, "tile.quant")
    _bool(self.rope, "tile.rope")
    _one_of(self.reduce_structure, REDUCE_STRUCTURES, "tile.reduce_structure")
    _power_of_two(self.lane_width, "tile.lane_width")
    if self.Hd % (self.lane_width * self.dot_pair_width) != 0:
      raise ValueError("tile.Hd must be divisible by tile.lane_width * tile.dot_pair_width")
    # Every lane owns part of each dot product, so the score reduce spans all lanes.
    if self.score_group_width is not None and self.score_group_width != self.lane_width:
      raise ValueError(f"tile.score_group_width must be null or equal tile.lane_width, got {self.score_group_width}")
    # A query group is a subset of the G query heads that share one KV head.
    if self.query_group_size is not None and self.query_group_size > self.G:
      raise ValueError(f"tile.query_group_size must be in 1..Hq/Hkv={self.G}, got {self.query_group_size}")
    # One warp computes one head, so fewer warps than the query group leave heads unwritten.
    if self.resolved_warps < self.QG:
      raise ValueError(f"tile.warps must be >= the query group size {self.QG}, got {self.warps}")

  def to_dict(self) -> dict[str, Any]:
    return {"Hq": self.Hq, "Hd": self.Hd, "Hkv": self.Hkv, "MAXC": self.MAXC, "split_count": self.split_count,
            "staging": self.staging, "quant": self.quant, "rope": self.rope, "token_block": self.token_block,
            "lane_width": self.lane_width, "score_group_width": self.score_group_width, "warps": self.warps,
            "query_group_size": self.query_group_size, "stage_width": self.stage_width,
            "reduce_structure": self.reduce_structure, "dot_pair_width": self.dot_pair_width}

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "FlashDecodeTileSpec":
    if not isinstance(d, dict): raise ValueError("tile must be an object")
    unknown = set(d) - set(FlashDecodeTileSpec.__dataclass_fields__)
    if unknown: raise ValueError(f"tile has unknown fields {sorted(unknown)}")
    fields = {name: d.get(name, default) for name, default in (
      ("Hq", None), ("Hd", None), ("Hkv", None), ("MAXC", None), ("split_count", None),
      ("staging", "KV_BOTH"), ("quant", False), ("rope", False), ("token_block", 16),
      ("lane_width", 32), ("score_group_width", None), ("warps", None), ("query_group_size", None),
      ("stage_width", 1), ("reduce_structure", "staged"), ("dot_pair_width", 2))}
    return FlashDecodeTileSpec(**fields)

  # Derived geometry (scope: descriptor contract section 4.1).
  @property
  def G(self) -> int:
    """Number of query heads per KV head group."""
    return self.Hq // self.Hkv

  @property
  def QG(self) -> int:
    """Query group size; falls back to the KV-group count."""
    return self.G if self.query_group_size is None else self.query_group_size

  @property
  def resolved_warps(self) -> int:
    """Workgroup warps; falls back to the query group size."""
    return self.QG if self.warps is None else self.warps

  @property
  def threads(self) -> int:
    return self.lane_width * self.resolved_warps

  @property
  def group_width(self) -> int:
    """Score-reduce group width; defaults to the physical lane width."""
    return self.lane_width if self.score_group_width is None else self.score_group_width

  @property
  def R(self) -> int:
    """Number of Hd elements per lane."""
    return self.Hd // self.lane_width

  @property
  def RP(self) -> int:
    """Number of half2 dot pairs per lane."""
    return self.Hd // (self.lane_width * self.dot_pair_width)

  @property
  def STAGES(self) -> int:
    """Token-block LDS stage count."""
    return math.ceil(self.token_block * self.Hd / self.threads)

  @property
  def reduce_stages(self) -> int:
    """Score-reduce shuffle ladder length: 1 inline, else ceil(log2(group_width)), at least 1.

    Not ``STAGES``: that counts token-block LDS stages, this counts shuffle steps.
    """
    if self.reduce_structure == "inline": return 1
    return max(1, (self.group_width - 1).bit_length())

  def kv_tile_bytes(self) -> int:
    """Declared shared memory of one tile workgroup: fp16 K and V blocks, or the K block alone for K_ONLY.

    Each workgroup owns one (KV head, split) pair, so the bytes depend on neither
    ``Hkv`` nor ``split_count``. The runtime's fixed per-launch reservation is a
    target fact and is not included. The provider's own ``admit`` remains the
    binding authority; this is a BoltBeam-side static bound.
    """
    return self.token_block * self.Hd * 2 * (2 if self.staging == "KV_BOTH" else 1)


@dataclass(frozen=True)
class FlashDecodeCombineSpec:
  """Optional combine geometry; a null ``lane_width`` is resolved to the tile lane width before hashing."""
  stride: int | None = None
  output_fp16: bool = False
  lane_width: int | None = None

  def __post_init__(self) -> None:
    _optional_positive_int(self.stride, "combine.stride")
    _bool(self.output_fp16, "combine.output_fp16")
    if _optional_positive_int(self.lane_width, "combine.lane_width") is not None:
      _power_of_two(self.lane_width, "combine.lane_width")

  def to_dict(self) -> dict[str, Any]:
    return {"stride": self.stride, "output_fp16": self.output_fp16, "lane_width": self.lane_width}

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "FlashDecodeCombineSpec":
    if not isinstance(d, dict): raise ValueError("combine must be an object")
    unknown = set(d) - {"stride", "output_fp16", "lane_width"}
    if unknown: raise ValueError(f"combine has unknown fields {sorted(unknown)}")
    return FlashDecodeCombineSpec(stride=d.get("stride"), output_fp16=d.get("output_fp16", False),
                                  lane_width=d.get("lane_width"))

  def scratch_bytes(self, tile: FlashDecodeTileSpec) -> int:
    """Declared shared memory of the combine workgroup: one fp32 weight per split.

    The combine output goes straight to global memory, so ``output_fp16`` does
    not change it; the split count is the tile's.
    """
    return tile.split_count * 4


def _validate_target_identity(target: Any) -> dict[str, Any]:
  _strict_keys(target, {"target_id", "backend", "arch", "subgroup_size", "resolved_target_hash"}, "target")
  for field_name in ("target_id", "backend", "arch"):
    _nonempty_str(target[field_name], f"target.{field_name}")
  _positive_int(target["subgroup_size"], "target.subgroup_size")
  if not isinstance(target["resolved_target_hash"], str) or not _SHA256_RE.fullmatch(target["resolved_target_hash"]):
    raise ValueError("target.resolved_target_hash must be a lowercase SHA-256 hex digest")
  return target


def _normalize_payload(p: dict[str, Any]) -> dict[str, Any]:
  """Fill descriptor defaults so partial spellings canonicalize identically."""
  if not isinstance(p, dict): raise ValueError("flash_decode_candidate must be an object")
  _strict_keys(p, {"schema_version", "descriptor", "target", "provenance"}, "flash_decode_candidate")
  if p["schema_version"] != FLASH_DECODE_CANDIDATE_SCHEMA_VERSION:
    raise ValueError(f"unsupported flash decode candidate schema_version {p['schema_version']!r}")
  descriptor = p["descriptor"]
  _strict_keys(descriptor, {"tile", "combine"}, "descriptor")
  tile = FlashDecodeTileSpec.from_dict(descriptor["tile"])
  combine = descriptor["combine"]
  normalized_combine = None
  if combine is not None:
    combine_spec = FlashDecodeCombineSpec.from_dict(combine)
    # The combine emitter uses the tile lane width when none is given: one binary, one identity.
    if combine_spec.lane_width is None: combine_spec = replace(combine_spec, lane_width=tile.lane_width)
    if tile.Hd % combine_spec.lane_width != 0:
      raise ValueError("combine.lane_width must divide tile.Hd")
    normalized_combine = combine_spec.to_dict()
  provenance = p["provenance"]
  _strict_keys(provenance, {"generator_id", "generator_revision", "schema_revision"}, "provenance")
  for field_name in ("generator_id", "generator_revision", "schema_revision"):
    _nonempty_str(provenance[field_name], f"provenance.{field_name}")
  if provenance["schema_revision"] != FLASH_DECODE_CANDIDATE_SCHEMA_VERSION:
    raise ValueError("provenance.schema_revision must match candidate schema_version")
  return {"schema_version": p["schema_version"],
          "descriptor": {"tile": tile.to_dict(), "combine": normalized_combine},
          "target": _validate_target_identity(p["target"]),
          "provenance": dict(provenance)}


@dataclass(frozen=True)
class FlashDecodeCandidate:
  """Strict v1 reader for one flash decode descriptor.

  The payload is normalized through JSON, validated, and hashed exactly once:
  ``candidate_hash`` is the sha256 of the canonical JSON document and covers
  every descriptor, target, and provenance field. No computed geometry or
  policy field may be added to the hashed document.
  """
  payload: dict[str, Any]
  _canonical_document: str = field(init=False, repr=False, compare=False)

  def __post_init__(self) -> None:
    try:
      payload = json.loads(json.dumps(self.payload, allow_nan=False))
    except (TypeError, ValueError) as exc:
      raise ValueError(f"flash_decode_candidate must be JSON data: {exc}") from exc
    normalized = _normalize_payload(payload)
    object.__setattr__(self, "payload", normalized)
    object.__setattr__(self, "_canonical_document", _canonical_json(normalized))

  @property
  def schema_version(self) -> str: return self.payload["schema_version"]

  @property
  def descriptor(self) -> dict[str, Any]: return json.loads(self.canonical_json())["descriptor"]

  @property
  def tile(self) -> FlashDecodeTileSpec:
    return FlashDecodeTileSpec.from_dict(json.loads(self.canonical_json())["descriptor"]["tile"])

  @property
  def combine(self) -> FlashDecodeCombineSpec | None:
    raw = json.loads(self.canonical_json())["descriptor"]["combine"]
    return None if raw is None else FlashDecodeCombineSpec.from_dict(raw)

  @property
  def target(self) -> dict[str, Any]: return json.loads(self.canonical_json())["target"]

  @property
  def provenance(self) -> dict[str, Any]: return json.loads(self.canonical_json())["provenance"]

  def canonical_json(self) -> str: return self._canonical_document

  @property
  def candidate_hash(self) -> str:
    return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

  def to_dict(self) -> dict[str, Any]: return json.loads(self.canonical_json())

  @staticmethod
  def from_dict(d: dict[str, Any]) -> "FlashDecodeCandidate":
    return FlashDecodeCandidate(d)

  def envelope(self) -> dict[str, Any]:
    """Static-assessment wire form: the canonical payload plus the outer ``candidate_hash``.

    The outer hash is a label for readers, never part of the hashed document.
    """
    return {**self.to_dict(), "candidate_hash": self.candidate_hash}

  @staticmethod
  def from_envelope(d: dict[str, Any]) -> "FlashDecodeCandidate":
    """Read a payload with or without the outer ``candidate_hash``.

    A present hash must equal the recomputed identity, so a stale or edited
    envelope fails closed instead of being relabelled.
    """
    if not isinstance(d, dict): raise ValueError("flash_decode_candidate envelope must be an object")
    candidate = FlashDecodeCandidate({key: value for key, value in d.items() if key != "candidate_hash"})
    if "candidate_hash" in d and d["candidate_hash"] != candidate.candidate_hash:
      raise ValueError("flash_decode_candidate envelope candidate_hash does not match its canonical payload")
    return candidate

  def launch_local_memory_bytes(self, reserved_bytes: int | None = None) -> dict[str, int]:
    """Shared memory each kernel launch needs: the tile launch, and the combine launch when present.

    The launches run separately, so each is held against the target limit on its
    own, never their sum. ``reserved_bytes`` is the target's fixed per-launch
    runtime reservation when the target supplies one; there is no default.
    """
    tile, combine = self.tile, self.combine
    declared = {"tile": tile.kv_tile_bytes()} | ({} if combine is None else {"combine": combine.scratch_bytes(tile)})
    return {launch: size + (reserved_bytes or 0) for launch, size in declared.items()}

  def validate_target(self, validator: TargetCapabilityValidator) -> None:
    """Apply a caller-owned target capability validator on demand."""
    if not callable(validator): raise ValueError("target validator must be callable")
    validator(self.target)


def _missing_fact(key: str) -> tuple[str, str]:
  return f"missing_fact:{key}", f"target fact {key} is missing, so this limit is unverified"


def flash_legality_violations(candidate: FlashDecodeCandidate, target_facts: dict[str, Any]) -> list[tuple[str, str]]:
  """Static legality of one descriptor against its own target and caller-supplied target facts.

  Facts are read by one key each (``boltbeam.search.target_fact_keys``), so a
  CUDA, Metal, or any future backend supplies the same numbers and gets the same
  verdict. The subgroup size is the candidate's own ``target.subgroup_size``. A
  present but invalid fact raises; a missing thread or shared-memory limit is a
  ``missing_fact:<key>`` violation, never a pass. The provider's ``admit``
  remains the binding authority for live limits.

  Each violation is ``(code, message)``: ``code`` is a stable reason token for
  static rejections, ``message`` is the text ``flash_legality_errors`` returns.
  """
  if not isinstance(candidate, FlashDecodeCandidate):
    candidate = FlashDecodeCandidate.from_dict(candidate)
  if not isinstance(target_facts, dict):
    raise ValueError("target_facts must be an object of target capability facts")
  limits = read_target_limits(target_facts)
  tile = candidate.tile
  violations: list[tuple[str, str]] = []

  # Staged and inline reduces both shuffle lane ^ offset; a shuffle cannot cross a hardware subgroup.
  subgroup = candidate.target["subgroup_size"]
  if subgroup % tile.lane_width != 0:
    violations.append(("lane_width_not_in_subgroup",
                       f"lane_width {tile.lane_width} must divide the target subgroup_size {subgroup}"))
  if limits.max_threads is None:
    violations.append(_missing_fact(MAX_THREADS))
  elif tile.threads > limits.max_threads:
    violations.append(("over_threads", f"workgroup threads {tile.threads} exceed target limit {limits.max_threads}"))
  if limits.max_local_memory is None:
    violations.append(_missing_fact(MAX_LOCAL_MEMORY))
  else:
    for launch, required in candidate.launch_local_memory_bytes(limits.reserved_local_memory).items():
      if required > limits.max_local_memory:
        violations.append(("over_local_memory", f"{launch} launch needs {required} bytes of shared memory, "
                                                f"over the target limit {limits.max_local_memory}"))
  return violations


def flash_legality_errors(candidate: FlashDecodeCandidate, target_facts: dict[str, Any]) -> list[str]:
  """Static legality messages for one descriptor; the rules live in ``flash_legality_violations``."""
  return [message for _code, message in flash_legality_violations(candidate, target_facts)]


def validate_flash_legality(candidate: FlashDecodeCandidate, target_facts: dict[str, Any]) -> None:
  """Fail closed unless the descriptor is legal for the supplied target facts."""
  errors = flash_legality_errors(candidate, target_facts)
  if errors:
    raise ValueError("flash candidate is illegal for target facts: " + "; ".join(errors))


__all__ = ["FLASH_DECODE_CANDIDATE_SCHEMA_VERSION", "FlashDecodeCandidate", "FlashDecodeCombineSpec",
           "FlashDecodeTileSpec", "REDUCE_STRUCTURES", "STAGING_MODES", "TargetCapabilityValidator",
           "flash_legality_errors", "flash_legality_violations", "validate_flash_legality"]
