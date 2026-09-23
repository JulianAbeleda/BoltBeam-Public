"""Closed-question policy consistency checks for supplied report text.

This module is intentionally dependency-free and has no repository-specific
default file list. Callers provide paths and/or text, and the scanner reports
lines that re-open closed policy questions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence


_87_OK = (
  "ctx", "ms", "coincid", "ambig", "11.4", "contextual", "empty-kv", "empty kv", "never quote",
  "ms/token", "ms per", "numeric", "provenance", "non-headline", "two real", "opposite", "reconcil",
  "sources", "trap", "separate", "bare", "quote", "never", "85", "86", "87.0", "87.5", "87.62",
  "87.9", "us\\b",
)
_DECIDED_OK = (
  "stays off", "stay off", "decided", "do not flip", "don't flip", "not flip", "off (decided",
  "opt-in", "no decode benefit", "stays **off", "default off", "remains off", "resident during decode",
)
_HEAD_OK = ("curve", "~67", "67%", "non-headline", "ambig", "never", "contextual", "not the", "peak", "coincid")
_META = (
  "guardrail", "consistency check", "consistency: ", "re-open", "re-opens", "reopen", "stale reference",
  "stale ref", "banned", "qk_policy_consistency", "checker", "this file fails", "exit 1", "audit",
)
_DECODE_RESTED_OK = (
  "rested", "closed", "refuted", "no-go", "no go", "north-star", "north star", "not promote",
  "rest_decode", "rest decode", "not a default", "not a bounded", "do not pursue", "historical",
  "superseded", "no funded bounded", "failed w==d", "promotion failed", "exhausted", "do not",
  "don't", "opt-in", "owner", "not promoted", "below promotion", "sub-gate", "marginal", "rest",
)


@dataclass(frozen=True)
class PolicyViolation:
  path: str
  line_number: int
  reason: str
  line: str

  def format(self) -> str:
    return f"{self.path}:{self.line_number}  {self.reason}\n      > {self.line.strip()[:140]}"


def scan_text(path: str, text: str) -> list[PolicyViolation]:
  """Scan one supplied text blob for closed-question policy violations."""
  return _scan_lines(path, text.splitlines())


def scan_documents(documents: Mapping[str, str] | Iterable[tuple[str, str]]) -> list[PolicyViolation]:
  """Scan supplied ``path -> text`` documents in deterministic order."""
  if isinstance(documents, Mapping):
    items = sorted(documents.items())
  else:
    items = list(documents)
  violations: list[PolicyViolation] = []
  for path, text in items:
    violations.extend(scan_text(path, text))
  return violations


def scan_paths(paths: Iterable[str | Path]) -> list[PolicyViolation]:
  """Read and scan supplied filesystem paths."""
  violations: list[PolicyViolation] = []
  for path in paths:
    p = Path(path)
    violations.extend(scan_text(str(p), p.read_text(errors="ignore")))
  return violations


def format_violations(violations: Sequence[PolicyViolation]) -> str:
  """Render violations using the original guardrail's compact line format."""
  return "\n".join(v.format() for v in violations)


def _scan_lines(path: str, lines: list[str]) -> list[PolicyViolation]:
  out: list[PolicyViolation] = []
  low = [line.lower() for line in lines]
  for i, line in enumerate(lines):
    l = low[i]
    win = " ".join(low[max(0, i - 1):i + 2])
    if any(marker in win for marker in _META):
      continue
    _check_line(out, path, i + 1, line, l, win)
  return out


def _check_line(out: list[PolicyViolation], path: str, line_number: int, line: str, l: str, win: str) -> None:
  if re.search(r"87\.6\b", line) and not _has_any(win, _87_OK):
    out.append(PolicyViolation(path, line_number, "bare `87.6` without context", line))
  if (
    "owner call" in l
    and ("prefill_v2" in l or "prefill default" in l or "global default" in l)
    and not _has_any(win, _DECIDED_OK)
  ):
    out.append(PolicyViolation(path, line_number, "open `PREFILL_V2` owner call (decision is OFF)", line))
  if "flip" in l and "prefill_v2" in l and "auto" in l and not _has_any(win, _DECIDED_OK):
    out.append(PolicyViolation(path, line_number, "affirmative 'flip global PREFILL_V2=auto'", line))
  if "decode headline" in l and re.search(r"\b8[5-9]\b|87\.", line) and not _has_any(win, _HEAD_OK):
    out.append(PolicyViolation(path, line_number, "`87` presented as the decode headline", line))
  if (
    re.search(r"bounded.*fusion|micro-?fusion", l)
    and any(word in l for word in ("current", "next work", "todo", "implement now", "tactical", "in progress"))
    and not _has_any(win, ("closed", "no-go", "no go", "exhausted", "refuted", "historical", "superseded"))
  ):
    out.append(PolicyViolation(path, line_number, "bounded decode fusion as current work (it is closed)", line))
  if (
    _has_any(l, ("vector_flash_decode_tile", "vector-tile", "fused+coop", "fused + coop"))
    and _has_any(l, ("fundable", "next build", "next bounded", "live decode lever", "live lever",
                     "open frontier", "is the next", "next scope"))
    and not _has_any(win, _DECODE_RESTED_OK)
  ):
    out.append(PolicyViolation(path, line_number,
                               "vector-tile/fused+coop as an OPEN/next bounded decode build (bounded decode is rested)",
                               line))
  if "flash_l=64" in l and _has_any(l, ("promote", "default", "ship", "should be")) and not _has_any(win, _DECODE_RESTED_OK):
    out.append(PolicyViolation(path, line_number, "FLASH_L=64 promotion/default proposed (it failed W==D; do not promote)", line))
  if re.search(r"bounded decode.*(open|fundable|next build|remains open|still open)", l) and not _has_any(win, _DECODE_RESTED_OK):
    out.append(PolicyViolation(path, line_number, "bounded decode framed as open (it is rested)", line))
  if (
    re.search(r"wmma.{0,40}decode|decode.{0,40}wmma", l)
    and re.search(r"is (the )?(llama|decode|remaining|next).{0,20}(lever|path|gap)|llama'?s.{0,20}(decode|path)|reopen wmma|wmma.{0,20}is llama", l)
    and not _has_any(win, ("not wmma", "non-wmma", "no wmma", "refuted", "closed", "below threshold", "prefill", "rest"))
  ):
    out.append(PolicyViolation(path, line_number, "WMMA decode claimed as llama's path / the lever (refuted: llama decode is non-WMMA vector)", line))
  if (
    "mmvq" in l
    and re.search(r"is (the )?(decode )?gap|is the (decode )?lever|next bounded|fundable|is the next", l)
    and not _has_any(win, ("closed", "parity", "do not", "not the gap", "refuted", "stays closed", "not a"))
  ):
    out.append(PolicyViolation(path, line_number, "MMVQ presented as the decode gap / next lever (refuted: wall parity, stays closed)", line))


def _has_any(text: str, needles: Iterable[str]) -> bool:
  return any(needle in text for needle in needles)
