"""Portable counter registry for profiler counters."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

from .ontology import DERIVED, MEASURED, PROXY, UNSUPPORTED, tier_observation, unsupported


SCHEMA_COUNTER_REGISTRY = "boltbeam.counter_registry.v1"

_DEFAULT_COUNTER_REGISTRY = pathlib.Path(__file__).resolve().parents[1] / "data" / "counter_registry.json"

_ALLOWED_EXACTNESS = {"measured", "derived", "proxy", "estimated"}
_ALLOWED_HIGHER_IS = {"better", "worse", "contextual"}
_ALLOWED_SCOPES = {"global", "per_kernel", "per_operand", "controlled_proxy"}


@dataclass(frozen=True)
class CounterDefinition:
  counter_id: str
  unit: str
  concept: str
  semantic_definition: str
  higher_is: str
  exactness: str
  formula: str | None
  provider_mappings: dict[str, dict[str, Any]]
  required_raw_metrics: tuple[str, ...]
  valid_range: tuple[float | None, float | None]
  normalization_notes: str
  known_non_equivalences: tuple[str, ...]


class CounterRegistryError(ValueError):
  """Raised when a counter registry file is malformed."""


class CounterLookupError(KeyError):
  """Raised when a counter cannot be found in the registry."""


class CounterFragmentError(ValueError):
  """Raised when counter evidence would overstate its collection or attribution scope."""


@dataclass(frozen=True)
class CounterOutcome:
  counter_id: str
  status: str
  reason: str | None = None
  value: float | None = None
  unit: str | None = None
  scope: str = "per_kernel"
  exactness: str | None = None
  source: str | None = None
  calibration_id: str | None = None

  def to_json(self) -> dict[str, Any]:
    out = {"counter": self.counter_id, "status": self.status, "scope": self.scope, "value": self.value}
    for key in ("reason", "unit", "exactness", "source", "calibration_id"):
      value = getattr(self, key)
      if value is not None: out[key] = value
    return out


def unsupported_counter(counter_id: str, reason: str, *, scope: str = "per_kernel") -> dict[str, Any]:
  """Typed unsupported outcome; unavailable is distinct from a measured zero."""
  if scope not in _ALLOWED_SCOPES: raise CounterFragmentError(f"invalid counter scope {scope!r}")
  return CounterOutcome(counter_id, UNSUPPORTED, reason=reason, scope=scope).to_json()


def normalize_counter_fragment(fragment: dict[str, Any], *, registry: dict[str, CounterDefinition] | None = None) -> dict[str, Any]:
  """Normalize one provider counter pass with fail-closed replay and attribution rules.

  Replay is allowed for a dedicated counter pass, but multiplexed/wrapped samples
  require an explicit correction. Request-to-byte conversion requires a calibration
  bound to this target and experiment scope.
  """
  source_registry = registry or load_counter_registry()
  scope = fragment.get("scope", "per_kernel")
  if scope not in _ALLOWED_SCOPES: raise CounterFragmentError(f"invalid counter scope {scope!r}")
  if scope == "per_operand" and not fragment.get("attribution"):
    raise CounterFragmentError("per_operand counters require address, differential, trace, or algebraic attribution")
  collection = fragment.get("collection") or {}
  pass_kind = collection.get("pass", "counter")
  if collection.get("replay") and pass_kind != "counter":
    raise CounterFragmentError("replayed counters must use a dedicated counter pass, not a timing pass")
  if (collection.get("multiplexed") or collection.get("wrapped")) and not collection.get("correction"):
    raise CounterFragmentError("multiplexed or wrapped counters require a documented correction")
  baseline = fragment.get("baseline") or {}
  outcomes, observations = [], []
  available_rows = [row for row in (fragment.get("counters") or []) if row.get("status") != UNSUPPORTED]
  if available_rows:
    missing_identity = [key for key in ("target_id", "experiment_id", "binary_hash", "session_id") if not fragment.get(key)]
    if missing_identity:
      raise CounterFragmentError(f"available counters require identity fields: {', '.join(missing_identity)}")
  for row in fragment.get("counters") or []:
    counter_id = str(row.get("counter") or row.get("counter_id") or "")
    if row.get("status") == UNSUPPORTED:
      outcomes.append(unsupported_counter(counter_id, row.get("reason") or "provider_counter_unavailable", scope=scope))
      continue
    definition = get_counter(counter_id, source_registry)
    if row.get("value") is None: raise CounterFragmentError(f"counter {counter_id!r} has no value or typed status")
    value = float(row["value"]) - float(baseline.get(counter_id, 0))
    if value < 0: raise CounterFragmentError(f"baseline exceeds counter {counter_id!r}")
    exactness = row.get("exactness", definition.exactness)
    calibration = row.get("calibration")
    if row.get("convert_requests_to_bytes"):
      if not calibration or not calibration.get("calibration_id") or not calibration.get("bytes_per_request"):
        raise CounterFragmentError(f"counter {counter_id!r} request-to-byte conversion requires calibration identity")
      if calibration.get("target_id") != fragment.get("target_id"):
        raise CounterFragmentError(f"counter {counter_id!r} calibration target mismatch")
      if calibration.get("experiment_scope") not in {fragment.get("experiment_id"), "global"}:
        raise CounterFragmentError(f"counter {counter_id!r} calibration is outside experiment scope")
      value *= float(calibration["bytes_per_request"])
      exactness = PROXY
    if scope == "controlled_proxy" and exactness == MEASURED: exactness = PROXY
    outcome = CounterOutcome(counter_id, "available", value=value, unit="bytes" if row.get("convert_requests_to_bytes") else definition.unit,
                             scope=scope, exactness=exactness, source=fragment.get("tool"),
                             calibration_id=(calibration or {}).get("calibration_id"))
    outcomes.append(outcome.to_json())
    tier = row.get("tier")
    if tier:
      observation = row.get("observation", "requests")
      if observation not in {"bytes", "requests", "hits", "misses", "hit_rate"}:
        raise CounterFragmentError(f"invalid tier observation kind {observation!r}")
      if observation == "hit_rate" and definition.unit == "%": value /= 100.0
      kwargs = {observation: value}
      observations.append(tier_observation(tier, scope=scope, source=counter_id,
                                           status=PROXY if exactness in {"proxy", "estimated"} else (DERIVED if exactness == "derived" else MEASURED),
                                           calibration_id=(calibration or {}).get("calibration_id"), **kwargs))
  return {"status": "ok", "scope": scope, "experiment_id": fragment.get("experiment_id"),
          "binary_hash": fragment.get("binary_hash"), "session_id": fragment.get("session_id"),
          "collection": collection, "outcomes": outcomes, "tier_observations": observations}


def _normalize_counter_id(counter_id: str) -> str:
  """Collapse punctuation/spacing for scattered-name collision checks."""
  return "".join(ch.lower() for ch in counter_id if ch.isalnum())


def _validate_counter_row(row: dict[str, Any], seen: dict[str, str]) -> CounterDefinition:
  required_fields = {
    "counter_id",
    "unit",
    "concept",
    "semantic_definition",
    "higher_is",
    "exactness",
    "provider_mappings",
    "required_raw_metrics",
    "valid_range",
    "normalization_notes",
    "known_non_equivalences",
  }
  missing = sorted(required_fields - set(row))
  if missing:
    raise CounterRegistryError(f"counter row missing required fields: {', '.join(missing)}")

  counter_id = str(row["counter_id"]).strip()
  if not counter_id:
    raise CounterRegistryError("counter_id cannot be empty")

  collapsed = _normalize_counter_id(counter_id)
  if not collapsed:
    raise CounterRegistryError(f"{counter_id!r} is not a valid counter identifier")
  duplicate = seen.get(collapsed)
  if duplicate:
    raise CounterRegistryError(
      f"parallel scattered counter ids {duplicate!r} and {counter_id!r} collapse to the same normalized name"
    )
  seen[collapsed] = counter_id

  higher_is = str(row["higher_is"]).strip()
  if higher_is not in _ALLOWED_HIGHER_IS:
    raise CounterRegistryError(f"invalid higher_is for {counter_id!r}: {higher_is!r}")

  exactness = str(row["exactness"]).strip()
  if exactness not in _ALLOWED_EXACTNESS:
    raise CounterRegistryError(f"invalid exactness for {counter_id!r}: {exactness!r}")

  unit = str(row["unit"]).strip()
  if not unit:
    raise CounterRegistryError(f"unit missing for counter {counter_id!r}")

  concept = str(row["concept"]).strip()
  if not concept:
    raise CounterRegistryError(f"concept missing for counter {counter_id!r}")

  semantic_definition = str(row["semantic_definition"]).strip()
  if not semantic_definition:
    raise CounterRegistryError(f"semantic_definition missing for counter {counter_id!r}")

  normalization_notes = str(row["normalization_notes"]).strip()
  known_non_equivalences = row["known_non_equivalences"]
  if not isinstance(known_non_equivalences, list):
    raise CounterRegistryError(f"known_non_equivalences for {counter_id!r} must be a list")
  if any(not isinstance(item, str) for item in known_non_equivalences):
    raise CounterRegistryError(f"known_non_equivalences for {counter_id!r} must contain only strings")

  required_raw_metrics = row["required_raw_metrics"]
  if not isinstance(required_raw_metrics, list):
    raise CounterRegistryError(f"required_raw_metrics for {counter_id!r} must be a list")
  if any(not isinstance(item, str) or not item for item in required_raw_metrics):
    raise CounterRegistryError(f"required_raw_metrics for {counter_id!r} must be non-empty strings")

  valid_range = row["valid_range"]
  if not isinstance(valid_range, list) or len(valid_range) != 2:
    raise CounterRegistryError(f"valid_range for {counter_id!r} must be a 2-item list")
  low, high = valid_range
  for bound in (low, high):
    if bound is not None and not isinstance(bound, (int, float)):
      raise CounterRegistryError(f"valid_range for {counter_id!r} has non-numeric bound {bound!r}")
  if low is not None and high is not None and low > high:
    raise CounterRegistryError(f"valid_range for {counter_id!r} must be low <= high")

  provider_mappings = row["provider_mappings"]
  if not isinstance(provider_mappings, dict):
    raise CounterRegistryError(f"provider_mappings for {counter_id!r} must be an object")
  for provider, mapping in provider_mappings.items():
    if not isinstance(provider, str):
      raise CounterRegistryError(f"provider_mappings keys for {counter_id!r} must be strings")
    if not isinstance(mapping, dict):
      raise CounterRegistryError(f"provider_mappings[{provider!r}] for {counter_id!r} must be an object")

  formula = row.get("formula")
  if formula is not None and not isinstance(formula, str):
    raise CounterRegistryError(f"formula for {counter_id!r} must be a string when present")

  return CounterDefinition(
    counter_id=counter_id,
    unit=unit,
    concept=concept,
    semantic_definition=semantic_definition,
    higher_is=higher_is,
    exactness=exactness,
    formula=formula,
    provider_mappings=provider_mappings,
    required_raw_metrics=tuple(required_raw_metrics),
    valid_range=(low, high),
    normalization_notes=normalization_notes,
    known_non_equivalences=tuple(known_non_equivalences),
  )


def load_counter_registry(path: str | pathlib.Path | None = None) -> dict[str, CounterDefinition]:
  """Load and validate a counter registry file into an immutable mapping."""
  registry_path = pathlib.Path(path) if path is not None else _DEFAULT_COUNTER_REGISTRY
  data = json.loads(registry_path.read_text())

  if data.get("schema") != SCHEMA_COUNTER_REGISTRY:
    raise CounterRegistryError(f"{registry_path} missing schema {SCHEMA_COUNTER_REGISTRY!r}")

  raw_counters = data.get("counters")
  if not isinstance(raw_counters, list):
    raise CounterRegistryError(f"{registry_path} counters must be a list")
  if not raw_counters:
    raise CounterRegistryError(f"{registry_path} has no counters")

  seen: dict[str, str] = {}
  rows: dict[str, CounterDefinition] = {}
  for row in raw_counters:
    if not isinstance(row, dict):
      raise CounterRegistryError(f"{registry_path} counter rows must be objects")
    normalized = _validate_counter_row(row, seen)
    rows[normalized.counter_id] = normalized

  return rows


def get_counter(counter_id: str, registry: dict[str, CounterDefinition] | None = None) -> CounterDefinition:
  """Return a counter definition by id, with permissive normalized matching."""
  source = registry or load_counter_registry()
  try:
    return source[counter_id]
  except KeyError:
    needle = _normalize_counter_id(counter_id)
    for key, definition in source.items():
      if _normalize_counter_id(key) == needle:
        return definition
    raise CounterLookupError(f"unknown counter_id {counter_id!r}")


def list_counter_ids(*, registry: dict[str, CounterDefinition] | None = None) -> tuple[str, ...]:
  """Return all registered counter ids in stable sorted order."""
  source = registry or load_counter_registry()
  return tuple(sorted(source))


def lookup_counter_ids_by_concept(concept: str, *, registry: dict[str, CounterDefinition] | None = None) -> tuple[str, ...]:
  """Return counter ids that belong to a concept bucket."""
  source = registry or load_counter_registry()
  return tuple(sorted(cid for cid, definition in source.items() if definition.concept == concept))
