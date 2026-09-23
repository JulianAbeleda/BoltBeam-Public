"""Shared search-tool primitives ported from tinygrad's QK search tooling.

- GROUPINGS            : the lane_grouping -> rows-per-wave map used by the topology authors.
- grammar_max_candidates(): the bounded-search candidate cap, read from the grammar JSON (data, not a code constant).
- ledger_candidate_ids(path): a robust (KeyError-safe) reader for the candidate_id set from a JSONL ledger.

Deliberately small + dependency-light (stdlib only). Does NOT merge the two factorizers (factor_pairs vs
factor_pos_lanes are different functions) nor the GPU-harness burst() (only 2 sites, authority harness) — see the R1
scope's Phase 5 notes.
"""
from __future__ import annotations
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
_GRAMMAR = ROOT / "bench/qk-search-spaces/topology_grammar_v1.json"

# lane_grouping -> rows-per-wave. half_warp is refuted (decode_q6k_direct_refuted); subgroup folds into 1row on wave32.
GROUPINGS: dict[str, int] = {"1row_per_warp": 1, "2rows_per_warp": 2}


def grammar_max_candidates(default: int = 64) -> int:
  """The bounded-search candidate cap (data, from the grammar JSON `max_candidates`). Falls back to `default`."""
  try:
    return int(json.load(open(_GRAMMAR)).get("max_candidates", default))
  except Exception:
    return default


def ledger_candidate_ids(path: str | pathlib.Path) -> set[str]:
  """Read the set of candidate_id values from a JSONL ledger, tolerant of rows lacking the key or malformed lines.
  (Fixes the divergence where one reader used d['candidate_id'] (KeyError) and another used d.get(...).)"""
  p = pathlib.Path(path)
  out: set[str] = set()
  if not p.exists():
    return out
  for line in open(p):
    line = line.strip()
    if not line:
      continue
    try:
      out.add(json.loads(line).get("candidate_id", ""))
    except Exception:
      continue
  out.discard("")
  return out
