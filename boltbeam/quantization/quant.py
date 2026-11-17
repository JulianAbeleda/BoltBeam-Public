"""Quant capability loader (audit A3): the `capability` of each weights row in data/formats.json, the one format table.

Quant BEHAVIOR is DATA, not code branches. Search emission, priority ranking, and reachability all ask this module
instead of hardcoding `if quant == "Q6_K"`. A quant without a capability is `unsupported_quant`
(search-space-incomplete): surfaced explicitly, never silently dropped.
"""
from __future__ import annotations

import functools
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from boltbeam import formats
from boltbeam.vocab import QuantSupport, SCHEMA_QUANT_REGISTRY

_DEFAULT_PATH = formats.PATH


@dataclass(frozen=True)
class QuantCapability:
  name: str
  ggml_type: int
  block_elems: int                       # elements per (super)block
  block_bytes: int                       # bytes per (super)block — the load-bearing byte-cost fact
  dequant_family: str                    # k_quant / legacy_scale / codebook (IQ: through a value table) / float_passthrough
  route_families: tuple[str, ...] = ()    # route families the emitter may offer for this quant
  refuted_routes: tuple[str, ...] = ()    # route axes known-refuted for this quant (ledger consistency)
  pending_routes: tuple[str, ...] = ()    # desired routes not yet expressible (search-space-incomplete)
  packed_word_bits: int | None = None

  @property
  def bytes_per_elem(self) -> float:
    return self.block_bytes / self.block_elems if self.block_elems else 0.0

  @property
  def support(self) -> str:
    return QuantSupport.SUPPORTED.value if self.route_families else QuantSupport.KNOWN_NO_ROUTE.value

  def to_json(self) -> dict[str, Any]:
    return {"name": self.name, "ggml_type": self.ggml_type, "block_elems": self.block_elems,
            "block_bytes": self.block_bytes, "dequant_family": self.dequant_family,
            "route_families": list(self.route_families), "refuted_routes": list(self.refuted_routes),
            "pending_routes": list(self.pending_routes), "packed_word_bits": self.packed_word_bits,
            "support": self.support}


def _from_row(d:dict[str, Any]) -> QuantCapability:
  c = d["capability"]
  return QuantCapability(name=d["name"], ggml_type=d.get("ggml", -1), block_elems=d["block"][0], block_bytes=d["block"][1],
                         dequant_family=c["dequant_family"], route_families=tuple(c.get("route_families", ())),
                         refuted_routes=tuple(c.get("refuted_routes", ())), pending_routes=tuple(c.get("pending_routes", ())),
                         packed_word_bits=c.get("packed_word_bits"))


def load_quant_registry(path:str | pathlib.Path | None = None, *, activations:bool = False) -> dict[str, QuantCapability]:
  """The weight quants; with activations, also the formats an engine only reads and writes activations in (a row
  marked "activation_only": q8_1, fp32), which no weight route or limit may take."""
  p = pathlib.Path(path) if path else _DEFAULT_PATH
  return dict(_registry(str(p), p.stat().st_mtime_ns, activations))


@functools.lru_cache(maxsize=8)
def _registry(p:str, _mtime:int, activations:bool) -> dict[str, QuantCapability]:
  """One parse per file version: the roofline asks per GEMM, about a million times for a long replay."""
  data = json.loads(pathlib.Path(p).read_text())
  if data.get("schema") != SCHEMA_QUANT_REGISTRY:
    raise ValueError(f"{p} is not a {SCHEMA_QUANT_REGISTRY} registry")
  rows = [r for group in data["weights"].values() for r in group if "capability" in r]
  return {r["name"]: _from_row(r) for r in rows if activations or not r["capability"].get("activation_only")}


def quant_capability(name:str, path:str | pathlib.Path | None = None) -> QuantCapability | None:
  """The capability for a quant NAME, or None if the quant is unsupported (absent from the registry)."""
  return load_quant_registry(path).get(name)


def quant_support(name:str, path:str | pathlib.Path | None = None) -> str:
  """QuantSupport value for a quant name: supported / known_no_route / unsupported_quant."""
  cap = quant_capability(name, path)
  return cap.support if cap else QuantSupport.UNSUPPORTED.value


def route_families_for(name:str, path:str | pathlib.Path | None = None) -> tuple[str, ...]:
  cap = quant_capability(name, path)
  return cap.route_families if cap else ()
