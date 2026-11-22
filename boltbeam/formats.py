"""The one format registry: every weight, KV-cache and activation format BoltBeam knows (data/formats.json), read by
every component. No other module lists formats (sz.py's formats authority fails a commit that does). A format the
table does not hold is refused in words (unknown()), never guessed, skipped or counted as another. docs/formats.md."""
from __future__ import annotations

import functools, importlib, json, pathlib
from typing import Any

PATH = pathlib.Path(__file__).resolve().parent / "data" / "formats.json"
KINDS = ("weights", "kv_cache", "activations")
STATUSES = ("supported", "refuses", "n/a", "unaudited")


@functools.lru_cache(maxsize=1)
def table() -> dict[str, Any]: return json.loads(PATH.read_text(encoding="utf-8"))


def components() -> tuple[str, ...]: return tuple(table()["components"])


@functools.lru_cache(maxsize=None)
def rows(kind:str = "weights") -> tuple[dict[str, Any], ...]:
  """A kind's rows, each with its family and kind; a same_as row takes its weights row's layout and ids."""
  return tuple({**{k: v for k, v in (row(r["same_as"]) if "same_as" in r else {}).items() if k in ("ggml", "block", "bits")},
                **r, "family": family, "kind": kind} for family, group in table()[kind].items() for r in group)


@functools.lru_cache(maxsize=None)
def _index(kind:str) -> dict[Any, dict[str, Any]]:
  """Every spelling of a row: its aliases, torch name, safetensors dtype and name; ("ggml", id) and ("onnx", id)."""
  own = [r for r in rows(kind) if "same_as" not in r]
  return {**{a: r for r in rows(kind) for a in r.get("aliases", ())}, **{r[k]: r for r in rows(kind) for k in ("torch", "safetensors", "name") if k in r},
          **{(k, r[k]): r for r in own for k in ("ggml", "onnx") if k in r}}


# A format's row by any of its spellings (exact), or None.
def row(name:Any, kind:str = "weights") -> dict[str, Any] | None: return _index(kind).get(name)


def unknown(fmt:Any, where:str) -> str: return f"unknown format {fmt} in {where}: add a row to data/formats.json"


class UnknownFormat(ValueError):
  """A format the registry does not hold, refused in words; formats names each one (a screen says them in a line)."""
  def __init__(self, message:str, formats:Any = ()):
    super().__init__(message)
    self.formats = tuple(formats)


def require(fmt:Any, where:str, kind:str = "weights") -> dict[str, Any]:
  """The row, or a refusal in words: the one path every reader takes for a format it meets."""
  if (r := row(fmt, kind)) is None:
    label = f"{fmt[0].upper()} type {fmt[1]}" if isinstance(fmt, tuple) else fmt
    raise UnknownFormat(unknown(label, where), [label])
  return r


# BoltBeam's name for an engine's spelling of a format (a torch dtype, an alias, a -ctk or --kv-cache-dtype word).
def canonical(word:Any) -> str | None: return r["name"] if (r := row(word)) else (row(word, "kv_cache") or {}).get("same_as")


# The 16-bit float with a matrix path of its own (F16, BF16) this word spells, or None.
def half(word:Any) -> str | None: return r["name"] if (r := row(canonical(word))) and r.get("half_path") else None


def elem_bytes(word:Any) -> float | None: return r["block"][1] / r["block"][0] if (r := row(canonical(word))) and "block" in r else None


# {format name: its value} for every row that has this field (ggml, block, capability, llama_mmq ...).
def field(name:str, kind:str = "weights") -> dict[str, Any]: return {r["name"]: r[name] for r in rows(kind) if r.get(name) is not None and "same_as" not in r}


# The formats whose capability sets this flag (packed_linear ...).
def capable(flag:str) -> list[str]: return [q for q, c in field("capability").items() if c.get(flag)]


def ggml_names() -> dict[int, str]: return {i: n for n, i in sorted(field("ggml").items(), key=lambda x: x[1])}


# A model header's tensor type id as BoltBeam's format name (the id itself when the registry lacks it).
def header_type(type_id:int) -> str: return ggml_names().get(type_id, str(type_id))


# ggml type id -> (values per block, bytes per block).
def layouts() -> dict[int, tuple[int, int]]: return {r["ggml"]: tuple(r["block"]) for r in rows() if "ggml" in r and "same_as" not in r}


# An engine's KV cache words the roofline sizes: word -> the weights format it is.
def kv_words(engine:str) -> dict[str, str]: return {r["name"]: r["same_as"] for r in rows("kv_cache") if engine in r["engines"] and r["status"]["roofline"] == "supported"}


# The KV types recommend offers an engine, by label, full precision first: bytes per element.
def kv_offer(engine:str) -> dict[str, float]: return {r["offer"]: r["block"][1] / r["block"][0] for r in rows("kv_cache") if engine in r["engines"] and "offer" in r}


def refusal(component:str, fmt:Any, kind:str = "weights") -> str | None:
  """None when the component supports the format, else the words it refuses with (the row's, or unknown())."""
  if (r := row(fmt, kind)) is None: return unknown(fmt, component)
  s = r["status"][component]
  return None if s == "supported" else f"{fmt}: {s['refuses']}" if isinstance(s, dict) else f"{fmt}: {component} is {s} for this format"


def supports(component:str, fmt:Any, kind:str = "weights") -> bool: return refusal(component, fmt, kind) is None


@functools.lru_cache(maxsize=None)
def dequant(fmt:str, kind:str = "dequant"):  # BoltBeam's own per-block dequant for a format (its row's dequant reference), or None;
  # kind "fields": its integer fields reader (the values, scale and sub-scales an integer-dot kernel reads), or None
  return getattr(importlib.import_module(ref.split(":")[0]), ref.split(":")[1]) if (ref := (row(fmt) or {}).get(kind)) else None


# The config.json quant_method values the weights rows declare.
def quant_methods() -> set[str]: return {m for r in rows() for m in r.get("quant_method", ())}
