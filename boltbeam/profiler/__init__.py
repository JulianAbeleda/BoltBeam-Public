"""Profiler registry helpers for BoltBeam."""

from .capabilities import (
  ProfilerCapability,
  SCHEMA_PROFILER_CAPABILITIES,
  capabilities_for_provider,
  capabilities_for_tool,
  load_profiler_capabilities,
  profiler_capability,
  resolve_profiler_capability,
)
from .counters import (
  CounterDefinition,
  CounterLookupError,
  CounterRegistryError,
  SCHEMA_COUNTER_REGISTRY,
  get_counter,
  list_counter_ids,
  lookup_counter_ids_by_concept,
  load_counter_registry,
)

COUNTER_REGISTRY = load_counter_registry()
PROFILER_CAPABILITIES = load_profiler_capabilities()

__all__ = [
  "CounterDefinition",
  "CounterLookupError",
  "CounterRegistryError",
  "ProfilerCapability",
  "SCHEMA_COUNTER_REGISTRY",
  "SCHEMA_PROFILER_CAPABILITIES",
  "COUNTER_REGISTRY",
  "PROFILER_CAPABILITIES",
  "capabilities_for_provider",
  "capabilities_for_tool",
  "get_counter",
  "list_counter_ids",
  "load_counter_registry",
  "load_profiler_capabilities",
  "lookup_counter_ids_by_concept",
  "profiler_capability",
  "resolve_profiler_capability",
]
