"""One key per target fact, read in one place.

BubbleBeam, FutureSight and the flash decode legality rules read the same chip
limits. Each fact has exactly one key here; a provider that spells a fact
differently is translated in its adapter, never by an alias list, and a
different quantity (a per-CU ``lds_bytes_per_cu``) is never read as a
per-threadgroup limit. A present fact must be a positive int or reading fails
closed; an absent fact reads as ``None`` and each rule states what absence
means. The subgroup size is not read here: each candidate carries it in its
own target block.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_THREADS = "max_threads_per_threadgroup"
MAX_LOCAL_MEMORY = "max_threadgroup_memory_bytes"
# Fixed shared-memory bytes the runtime adds to every launch on top of the kernel's own arrays.
RESERVED_LOCAL_MEMORY = "reserved_threadgroup_memory_bytes"


def positive_fact(target_facts: Mapping[str, Any], key: str) -> int | None:
  """The fact's value, ``None`` when absent; a present non-positive or non-int value raises."""
  value = target_facts.get(key)
  if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
    raise ValueError(f"{key} must be positive")
  return value


@dataclass(frozen=True)
class TargetLimits:
  """Per-threadgroup limits of one target; ``None`` means the target did not supply the fact."""
  max_threads: int | None
  max_local_memory: int | None
  reserved_local_memory: int | None


def read_target_limits(target_facts: Mapping[str, Any]) -> TargetLimits:
  if not isinstance(target_facts, Mapping): raise ValueError("target facts must be a mapping")
  return TargetLimits(positive_fact(target_facts, MAX_THREADS), positive_fact(target_facts, MAX_LOCAL_MEMORY),
                      positive_fact(target_facts, RESERVED_LOCAL_MEMORY))


__all__ = ["MAX_LOCAL_MEMORY", "MAX_THREADS", "RESERVED_LOCAL_MEMORY", "TargetLimits", "positive_fact",
           "read_target_limits"]
