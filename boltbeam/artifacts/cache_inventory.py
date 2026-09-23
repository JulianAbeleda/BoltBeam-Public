from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = (
  "bench/qk-search-spaces", "bench/qk-lanemap-template-ir", "bench/qk-lanemap-template-audit",
  "bench/qk-topology-author", "bench/qk-quant-semantics-audit", "bench/qk-profile-opener",
  "bench/qk-target-features", "bench/qk-template-candidate-gate", "bench/qk-new-profile-search",
  "bench/qk-candidate-evaluator",
)

SPEED_KEYS = ("tok_s", "tok/s", "w==d", "wd_table", "whole_prefill", "pmc", "wall_share", "gbps", "tflops", "median_tok")
CORR_KEYS = ("token_match", "argmax", "route_attribution", "route_bound", "logit", "correct", "byte_identical", "fallback")


def _peek(path:Path) -> tuple[bool, bool, bool]:
  try:
    data = json.loads(path.read_text())
  except Exception:
    return (False, False, False)
  text = json.dumps(data).lower()
  has_speed = any(k in text for k in SPEED_KEYS)
  has_corr = any(k in text for k in CORR_KEYS)
  has_meta = isinstance(data, dict) and isinstance(data.get("cache"), dict) and data["cache"].get("schema", "").startswith("qk_artifact_cache")
  return has_speed, has_corr, has_meta


def classify_artifact(relpath:str, has_speed:bool, has_corr:bool) -> str:
  p = relpath.lower()
  is_eval = any(s in p for s in ("candidate-evaluator", "template-candidate-gate"))
  static_root = any(s in p for s in ("qk-search-spaces", "lanemap-template", "topology-author",
                                     "quant-semantics", "profile-opener", "target-features", "new-profile-search"))
  if is_eval:
    if has_speed: return "C_speed"
    if has_corr: return "B_correctness"
    return "A_static"
  if static_root: return "A_static"
  if has_speed: return "C_speed"
  if has_corr: return "B_correctness"
  return "A_static"


def build_inventory(root:str | Path, roots:tuple[str, ...] | list[str]=DEFAULT_ROOTS) -> dict[str, Any]:
  repo = Path(root)
  rows, unclassified = [], []
  for rel_root in roots:
    base = repo / rel_root
    if not base.exists():
      continue
    for path in sorted(base.rglob("*.json")):
      rel = str(path.relative_to(repo))
      has_speed, has_corr, has_meta = _peek(path)
      cls = classify_artifact(rel, has_speed, has_corr)
      rows.append({
        "path": rel,
        "class": cls,
        "has_cache_meta": has_meta,
        "has_speed_evidence": has_speed,
        "has_correctness_evidence": has_corr,
        "needs_wrapping": not has_meta,
        "wrap_phase": {"A_static": "C2", "B_correctness": "C3", "C_speed": "C3"}[cls],
      })
      if cls not in ("A_static", "B_correctness", "C_speed"):
        unclassified.append(rel)
  counts = {c: sum(1 for row in rows if row["class"] == c) for c in ("A_static", "B_correctness", "C_speed")}
  return {
    "schema": "boltbeam.artifact_cache_inventory.v1",
    "verdict": "C0_BLOCKED_ARTIFACTS_UNCLASSIFIED" if unclassified else "C0_PASS_CACHE_INVENTORY_PINNED",
    "total_artifacts": len(rows),
    "class_counts": counts,
    "need_wrapping": sum(1 for row in rows if row["needs_wrapping"]),
    "unclassified": unclassified,
    "rows": rows,
  }


def inventory_markdown(report:dict[str, Any]) -> str:
  counts = report["class_counts"]
  lines = [
    "# Artifact Cache Inventory",
    "",
    f"**Verdict:** `{report['verdict']}`",
    "",
    f"{report['total_artifacts']} artifacts: A_static={counts['A_static']}, "
    f"B_correctness={counts['B_correctness']}, C_speed={counts['C_speed']}; "
    f"{report['need_wrapping']} need cache-metadata wrapping.",
    "",
    "| class | wrap phase | count | reuse rule |",
    "|---|---|---:|---|",
    f"| A_static | C2 | {counts['A_static']} | reuse by hash(inputs + code); no GPU |",
    f"| B_correctness | C3 | {counts['B_correctness']} | reuse only if inputs+code+runtime fingerprints match |",
    f"| C_speed | C3 | {counts['C_speed']} | historical by default; promotion reruns unless cached speed is explicitly accepted |",
    "",
    "## Artifacts",
    "",
    "| path | class | speed? | correctness? | needs wrap |",
    "|---|---|:--:|:--:|:--:|",
  ]
  for row in report["rows"]:
    lines.append(f"| `{row['path']}` | {row['class']} | {'Y' if row['has_speed_evidence'] else ''} | "
                 f"{'Y' if row['has_correctness_evidence'] else ''} | {'Y' if row['needs_wrapping'] else ''} |")
  return "\n".join(lines) + "\n"
