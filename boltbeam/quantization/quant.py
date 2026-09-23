"""Quant capability registry loader (audit A3).

Quant BEHAVIOR is DATA (`data/quants.json`), not code branches. Search emission, priority ranking, and
reachability all ask this module instead of hardcoding `if quant == "Q6_K"`. A quant absent from the registry
is `unsupported_quant` (search-space-incomplete) — surfaced explicitly, never silently dropped.

Note the separation of concerns: `profile/gguf.py` still owns decoding a raw GGML type id into a quant NAME
for any GGUF (broad). This registry adds the capability facts for the subset BoltBeam can actually reason
about (narrow, extended by a data edit).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from boltbeam.vocab import QuantSupport, SCHEMA_QUANT_REGISTRY

_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "quants.json"


@dataclass(frozen=True)
class QuantCapability:
  name: str
  ggml_type: int
  block_elems: int                       # elements per (super)block
  block_bytes: int                       # bytes per (super)block — the load-bearing byte-cost fact
  dequant_family: str                    # k_quant / legacy_scale / float_passthrough
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
  return QuantCapability(name=d["name"], ggml_type=int(d["ggml_type"]), block_elems=int(d["block_elems"]),
                         block_bytes=int(d["block_bytes"]), dequant_family=d["dequant_family"],
                         route_families=tuple(d.get("route_families", ())),
                         refuted_routes=tuple(d.get("refuted_routes", ())),
                         pending_routes=tuple(d.get("pending_routes", ())),
                         packed_word_bits=d.get("packed_word_bits"))


def load_quant_registry(path:str | pathlib.Path | None = None) -> dict[str, QuantCapability]:
  p = pathlib.Path(path) if path else _DEFAULT_PATH
  data = json.loads(p.read_text())
  if data.get("schema") != SCHEMA_QUANT_REGISTRY:
    raise ValueError(f"{p} is not a {SCHEMA_QUANT_REGISTRY} registry")
  caps = {row["name"]: _from_row(row) for row in data["quants"]}
  return caps


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
