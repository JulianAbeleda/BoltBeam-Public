"""Source-level policy guard (audit A9).

The decision path — search emission, route families, seed policy, and evaluation — must be driven by profile
facts and the quant/target registries, NOT by hardcoded model names, quant-name branches, or target-name
branches. This module scans those specific modules and returns violations, so the human rule
("keep the decision path model/quant/target-agnostic") is machine-enforced (coding-principles: "human-facing
and machine-enforced").

The tinygrad handoff generator (`analyze.py`) is intentionally excluded from the target-name rule: it emits
backend-specific run commands and is the integration surface, not a decision surface. It is still checked for
model-name and quant-branch leakage.
"""
from __future__ import annotations

import pathlib
import re
import tokenize
from dataclasses import dataclass
from io import StringIO

from boltbeam.target.targets import TARGETS, target_aliases

_PKG = pathlib.Path(__file__).resolve().parent.parent

# modules whose LOGIC must be registry-driven (not the data files, not the registries themselves).
# The machine scan is a decision surface too: it decides which registry row you are running on.
_DECISION_PATH = ("search/emit.py", "search/families.py", "policy/emit.py", "eval/evaluator.py",
                  "workflow/autoscan.py")
_HANDOFF = ("analyze.py",)   # model/quant checked, target-name allowed (backend-specific command emission)

_QUANT_TOKENS = ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "BF16")
_MODEL_TOKENS = ("qwen", "llama", "mixtral", "deepseek", "mistral")
# Both target lists are read off the registry, not listed here, so a new row is covered the day it lands and
# a re-measured number is not pinned to the figure that happened to be true when this rule was written.
_TARGET_TOKENS = tuple(sorted(frozenset().union(*(target_aliases(name) for name in TARGETS))))


def _target_number_tokens() -> tuple[str, ...]:
  """Every speed the registry states, in both spellings a source file could write it.

  Hardware constants are target facts: they belong in data/targets.json or math/roofline, never inline in
  the decision path (LN-130).
  """
  values = [float(t.memory_bandwidth_gbs) for t in TARGETS.values() if t.memory_bandwidth_gbs]
  values += [float(v) for t in TARGETS.values() for v in t.peak_tflops.values()]
  return tuple(sorted({spelling for v in values for spelling in (f"{v}", f"{v:g}")}))


_TARGET_NUMBER_TOKENS = _target_number_tokens()


@dataclass(frozen=True)
class SourceViolation:
  rule: str
  module: str
  token: str

  def __str__(self) -> str:
    return f"[{self.rule}] {self.module}: hardcoded {self.token!r}"


def _code_strings(src:str) -> str:
  """Return the source with comments and docstrings removed, so a token only counts when it appears in real
  code (a docstring mentioning a quant name is fine; a `== "Q6_K"` branch is not)."""
  out = []
  try:
    toks = list(tokenize.generate_tokens(StringIO(src).readline))
  except tokenize.TokenError:
    return src
  prev_type = tokenize.INDENT
  for tok in toks:
    if tok.type == tokenize.COMMENT:
      continue
    if tok.type == tokenize.STRING:
      # a bare string statement (docstring) has NEWLINE/INDENT/DEDENT before it; skip those, keep f-string
      # expressions that are used in code by still emitting non-docstring strings' *content* is risky, so we
      # simply drop all string literals — a decision branch on a quant name is an OP/keyword pattern, not a
      # bare string, and hardcoded target/model ids we care about show up as NAME/attribute or dict keys too.
      if prev_type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
        continue
      out.append(tok.string)          # keep in-expression string literals (e.g. dict keys, comparisons)
    else:
      out.append(tok.string)
    if tok.type not in (tokenize.NL,):
      prev_type = tok.type
  return " ".join(out)


def _scan_file(rel:str, check_targets:bool) -> list[SourceViolation]:
  code = _code_strings((_PKG / rel).read_text())
  viols:list[SourceViolation] = []
  for tok in _QUANT_TOKENS:
    if re.search(rf"\b{re.escape(tok)}\b", code):
      viols.append(SourceViolation("quant_branch_outside_registry", rel, tok))
  for tok in _MODEL_TOKENS:
    if re.search(rf"\b{re.escape(tok)}\b", code, re.IGNORECASE):
      viols.append(SourceViolation("model_name_in_decision_path", rel, tok))
  if check_targets:
    for tok in _TARGET_TOKENS:
      if re.search(rf"\b{re.escape(tok)}\b", code):
        viols.append(SourceViolation("target_name_outside_descriptors", rel, tok))
    for tok in _TARGET_NUMBER_TOKENS:
      if re.search(rf"(?<![\d.]){re.escape(tok)}(?![\d.])", code):
        viols.append(SourceViolation("target_number_outside_registry", rel, tok))
  return viols


def scan_source_violations() -> list[SourceViolation]:
  viols:list[SourceViolation] = []
  for rel in _DECISION_PATH:
    viols += _scan_file(rel, check_targets=True)
  for rel in _HANDOFF:
    viols += _scan_file(rel, check_targets=False)
  return viols
