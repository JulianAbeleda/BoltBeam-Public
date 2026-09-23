"""Model -> required-primitive vocabulary extractor.

Answers: "what is the COMPLETE set of codegen primitives this model needs, and which can the substrate actually
generate?" Derives the vocabulary from the model itself — weight roles+quants give the GEMV primitives, the
architecture gives the fixed decoder ops — then cross-references each ISA primitive against what the target's
substrate can lower (target.dot_primitives_lowered + the known lowered/backlog substrate set). The backlog is the
exact "lower all" work remaining to fully + optimally generate the model (pure machine search).

Pure analysis (no kernels, no execution)."""
from __future__ import annotations
import json, pathlib
from dataclasses import dataclass, field
from typing import Any

from boltbeam.profile.ir import ModelProfile, TargetProfile

_VOCAB_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "primitive_vocab.json"

def load_primitive_vocab(path: str | pathlib.Path | None = None) -> dict[str, dict[str, Any]]:
  """The canonical primitive registry (single source of truth) keyed by primitive name."""
  data = json.loads(pathlib.Path(path or _VOCAB_PATH).read_text())
  if data.get("schema") != "boltbeam.primitive_vocab.v1": raise ValueError("not a primitive_vocab.v1 registry")
  return {p["name"]: p for p in data["primitives"]}

# ---------------------------------------------------------------------------
# op-class -> the ISA primitives it decomposes into. Weight ops are keyed by quant; arch ops are fixed for a
# decoder transformer (qwen3-family). "dot" is resolved to a concrete lowered dot per quant at extraction time.
# ---------------------------------------------------------------------------
_WEIGHT_OP = {
  "Q4_K": ("q4k_gemv", ["global_load", "dequant_bitunpack", "dequant_convert", "dot", "xlane_reduce"]),
  "Q6_K": ("q6k_gemv", ["global_load", "dequant_bitunpack", "dequant_convert", "dot", "xlane_reduce"]),
  "Q8_0": ("q8_gemv",  ["global_load", "dequant_convert", "dot", "xlane_reduce"]),
}
# The dot CLASSES each quant's GEMV could use — semantics, not instruction names: dequant to f16 and use an
# f16 dot, or quantize x to q8_1 and use an int8 dot. Which concrete instruction serves a class is a property
# of the TARGET (target.dot_primitives x the registry's dot_class), never of this module. Naming AMD opcodes
# here is what made `--target nvidia_sm89` confidently report v_dot2_f32_f16.
_QUANT_DOT_CLASS = {"Q4_K": ("f16", "int8"), "Q6_K": ("f16",), "Q8_0": ("int8",)}
_DEFAULT_DOT_CLASS = ("f16",)

_ARCH_DECODER_OPS = [
  ("rmsnorm",          ["reduce", "elementwise", "fused_reduce_scale"]),
  ("qk_norm",          ["reduce", "elementwise", "fused_reduce_scale"]),
  ("silu_swiglu",      ["transcendental", "elementwise"]),
  ("rope",             ["transcendental", "elementwise"]),
  ("residual_add",     ["elementwise"]),
  ("flash_decode_attn",["dot", "transcendental", "vec_store_to_reg", "xlane_reduce", "lds_stage"]),
  ("kv_cache_rw",      ["global_load", "elementwise"]),
  ("sample",           ["reduce", "elementwise"]),
]


@dataclass(frozen=True)
class VocabItem:
  op: str                       # the high-level op class (q4k_gemv, rmsnorm, flash_decode_attn, ...)
  source: str                   # "weight:<role>" or "arch"
  primitive: str                # the concrete ISA primitive
  lowered: bool
  status_reason: str
  value: str = ""               # registry `value`: foundational | low | med | high | generality

@dataclass(frozen=True)
class ModelVocabulary:
  model_id: str
  target_id: str
  items: tuple[VocabItem, ...]
  authorized: bool = True          # False when the target cannot support a lowering claim at all
  authorization_reason: str = ""

  def primitives(self) -> dict[str, bool]:
    """distinct primitive -> lowered? (a primitive counts as lowered only if lowered everywhere it appears)."""
    out: dict[str, bool] = {}
    for it in self.items:
      out[it.primitive] = out.get(it.primitive, True) and it.lowered
    return out

  def backlog(self) -> tuple[str, ...]:
    return tuple(sorted(p for p, low in self.primitives().items() if not low))

  def values(self) -> dict[str, str]:
    """distinct primitive -> its registry `value` tier."""
    return {it.primitive: it.value for it in self.items}

  def coverage(self) -> tuple[int, int]:
    """Raw lowered/total over EVERY required primitive. Kept for compatibility, but see `lever_coverage`:
    this ratio averages always-green foundational primitives with real levers and reads far better than the
    situation is."""
    prims = self.primitives(); return sum(1 for v in prims.values() if v), len(prims)

  def levers(self) -> dict[str, bool]:
    """The primitives that are actually a CHOICE -> lowered?

    A foundational primitive (global_load, elementwise, reduce, ...) is lowered on every target and can never
    be otherwise, so counting it says nothing about what is left to build. Excluding them is what makes the
    coverage number mean something."""
    vals = self.values()
    return {p: low for p, low in self.primitives().items() if vals.get(p) != "foundational"}

  def lever_coverage(self) -> tuple[int, int]:
    lev = self.levers(); return sum(1 for v in lev.values() if v), len(lev)

  def to_json(self) -> dict[str, Any]:
    lowered, total = self.coverage()
    lev = self.levers()
    lev_lowered, lev_total = self.lever_coverage()
    vals, reasons = self.values(), {i.primitive: i.status_reason for i in self.items}

    by_value: dict[str, dict[str, int]] = {}
    for prim, low in sorted(lev.items()):
      tier = vals.get(prim) or "unknown"
      slot = by_value.setdefault(tier, {"built": 0, "total": 0})
      slot["total"] += 1
      slot["built"] += 1 if low else 0

    foundational = sorted(p for p in self.primitives() if vals.get(p) == "foundational")
    return {"schema": "boltbeam.model_vocabulary.v1", "model_id": self.model_id, "target_id": self.target_id,
            "authorized": self.authorized, "authorization_reason": self.authorization_reason,
            "coverage": {"lowered": lowered, "total": total, "backlog": list(self.backlog())},
            # the number that means something: foundational primitives are always lowered and are never a
            # choice, so they are excluded rather than averaged in
            "levers": {"built": lev_lowered, "total": lev_total, "by_value": by_value,
                       "missing": [{"primitive": p, "value": vals.get(p) or "unknown",
                                    "reason": reasons.get(p, "")} for p in sorted(lev) if not lev[p]],
                       "note": "foundational primitives excluded: always lowered, never a choice"},
            "foundational": {"count": len(foundational), "primitives": foundational,
                             "all_lowered": all(self.primitives()[p] for p in foundational)},
            "primitives": {p: low for p, low in sorted(self.primitives().items())},
            "items": [{"op": i.op, "source": i.source, "primitive": i.primitive, "lowered": i.lowered,
                       "value": i.value, "reason": i.status_reason} for i in self.items]}


def _dots_for_quant(quant: str | None, target: TargetProfile,
                    registry: dict[str, dict[str, Any]]) -> list[str]:
  """Resolve the abstract `dot` to concrete primitives THIS target provides.

  A class the target cannot serve becomes a synthetic `dot:<class>` gap marker rather than borrowing another
  target's instruction — the whole point of failing closed is that a missing capability reads as missing.
  """
  out: list[str] = []
  for cls in _QUANT_DOT_CLASS.get(quant or "", _DEFAULT_DOT_CLASS):
    served = [p for p in getattr(target, "dot_primitives", ())
              if (registry.get(p) or {}).get("dot_class") == cls]
    out.extend(served or [f"dot:{cls}"])
  return out


def _prim_status(prim: str, target: TargetProfile,
                 registry: dict[str, dict[str, Any]]) -> tuple[bool, str, str]:
  """-> (lowered, reason, value_tier). Unknown primitives stay conservatively backlog AND non-foundational,
  so an unrecognised primitive can never quietly inflate the lever count."""
  if prim.startswith("dot:"):
    cls = prim.split(":", 1)[1]
    return False, (f"BACKLOG: target {target.target_id} declares no dot primitive serving class {cls!r} "
                   f"(add one to data/targets.json dot_primitives and tag its dot_class in "
                   f"data/primitive_vocab.json)"), "unknown"
  entry = registry.get(prim)
  if entry is not None:
    low = bool(entry.get("lowered"))
    if low and entry.get("dot_class") is not None:
      # the registry records that SOMETHING lowers it; it is only lowered HERE if this target has the
      # instruction and has a generated lowering for it
      hw = prim in getattr(target, "dot_primitives", ())
      gen = prim in getattr(target, "dot_primitives_lowered", ())
      if not hw:
        return False, f"BACKLOG: {target.target_id} has no {prim} instruction", str(entry.get("value") or "unknown")
      if not gen:
        return False, (f"BACKLOG: {target.target_id} has {prim} but no generated lowering "
                       f"(not in dot_primitives_lowered)"), str(entry.get("value") or "unknown")
    reason = entry.get("mechanism") if low else ("BACKLOG: " + entry.get("reason", "no lowering"))
    return low, reason, str(entry.get("value") or "unknown")
  if prim in getattr(target, "dot_primitives", ()):
    return False, "BACKLOG: hardware dot with no generated lowering", "unknown"
  return False, "BACKLOG: unrecognized primitive (no lowering recorded in registry)", "unknown"


def extract_model_vocabulary(model: ModelProfile, target: TargetProfile,
                             registry: dict[str, dict[str, Any]] | None = None) -> ModelVocabulary:
  """Derive the complete required-primitive vocabulary of `model` on `target`, with per-primitive lowering status."""
  reg = registry if registry is not None else load_primitive_vocab()
  items: list[VocabItem] = []
  def add(op: str, source: str, prims: list[str], quant: str | None = None):
    for p in prims:
      concrete = _dots_for_quant(quant, target, reg) if p == "dot" else [p]
      for cp in concrete:
        low, reason, value = _prim_status(cp, target, reg)
        items.append(VocabItem(op, source, cp, low, reason, value))
  # 1) weight-derived GEMV ops (analyze the weights): one op per distinct (role, quant)
  seen: set[tuple[str, str]] = set()
  for r in model.roles:
    key = (r.role, r.quant)
    if key in seen: continue
    seen.add(key)
    opname, prims = _WEIGHT_OP.get(r.quant, ("unknown_gemv", ["global_load", "dot"]))
    add(opname, f"weight:{r.role}", prims, quant=r.quant)
  # 2) arch-derived decoder ops (fixed for the transformer)
  for opname, prims in _ARCH_DECODER_OPS:
    add(opname, "arch", prims)
  # dedup identical items
  uniq = tuple(dict.fromkeys(items))
  # fail closed on a target we have no lowering evidence for: still report what the model NEEDS (that is
  # useful and target-independent), but refuse to call any of it lowered.
  authorized = bool(getattr(target, "is_complete", True))
  reason = "" if authorized else (
    f"target {target.target_id} is backend_status={getattr(target, 'backend_status', 'unknown')!r}: "
    f"no lowering evidence exists, so no primitive may be reported as lowered")
  if not authorized:
    # demote anything that CLAIMED a lowering, but keep reasons that are already specific — a dot-class gap
    # says exactly which capability is missing and how to declare it, which the blanket message would erase
    uniq = tuple(i if not i.lowered
                 else VocabItem(i.op, i.source, i.primitive, False, "BACKLOG: " + reason, i.value)
                 for i in uniq)
  return ModelVocabulary(model.model_id, target.target_id, uniq, authorized, reason)
