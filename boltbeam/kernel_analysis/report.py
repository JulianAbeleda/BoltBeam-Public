"""KA-8: orchestration, rendering, and exit-code policy for `investigate-kernels`.

One entry point ingests mixed fragments, judges eligibility, contrasts eligible pairs, diagnoses each
candidate, and recommends the next bounded experiment — then renders deterministic JSON + Markdown.
A blocked or inconclusive result is still a successful analysis (exit 0). Known-but-invalid evidence
exits 2; nothing supported exits 3; an internal invariant failure exits 4.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boltbeam.kernel_analysis.adapters import IngestResult, ingest_kernel_artifacts
from boltbeam.kernel_analysis.classify import classify_kernel_evidence
from boltbeam.kernel_analysis.contrast import PairContrast, contrast_pair
from boltbeam.kernel_analysis.diagnosis import diagnose
from boltbeam.kernel_analysis.eligibility import evaluate_all, performance_eligibility
from boltbeam.kernel_analysis.model import KernelAnalysis, KernelEvidence
from boltbeam.kernel_analysis.recommend import recommend
from boltbeam.kernel_analysis.selection import rank_measured_matrix

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_NO_EVIDENCE = 3
EXIT_INTERNAL = 4


@dataclass(frozen=True)
class Investigation:
  analysis: KernelAnalysis
  exit_code: int
  markdown: str


def _pairs(evidence: tuple[KernelEvidence, ...]) -> list[tuple[KernelEvidence, KernelEvidence]]:
  by_id = {e.candidate.candidate_id: e for e in evidence}
  pairs: list[tuple[KernelEvidence, KernelEvidence]] = []
  declared = False
  for ev in evidence:
    comp_id = ev.candidate.comparator_id
    if comp_id and comp_id in by_id and comp_id != ev.candidate.candidate_id:
      pairs.append((ev, by_id[comp_id]))
      declared = True
  if not declared and len(evidence) == 2:
    pairs.append((evidence[0], evidence[1]))
  return pairs


def _conclusion_from_contrast(c: PairContrast) -> dict[str, Any]:
  timing = c.timing
  if not c.performance_eligible or timing.get("qualified") in (None, "unavailable"):
    return {
      "scope": "pair", "candidate": c.candidate_id, "comparator": c.comparator_id,
      "statement": "no performance conclusion; comparison is blocked",
      "truth_status": "unknown", "confidence": "unknown", "selection": "blocked",
      "selection_confidence": "unknown", "mechanism_confidence": "unknown",
      "supporting_fact_ids": [], "contradicting_fact_ids": [], "blockers": list(c.blockers),
    }
  qualified = timing["qualified"]
  if qualified == "tie":
    statement = "no performance change: the difference is within the combined noise floor"
    confidence = "medium"
    selection = "inconclusive"
  elif qualified == "win":
    statement = (f"candidate is faster (median {timing['candidate_median_ms']:.4g} ms vs "
                 f"{timing['comparator_median_ms']:.4g} ms, speedup {timing['speedup']:.3g}x)")
    confidence = "high"
    selection = "promote"
  else:
    statement = (f"candidate is slower (median {timing['candidate_median_ms']:.4g} ms vs "
                 f"{timing['comparator_median_ms']:.4g} ms)")
    confidence = "high"
    selection = "retain"
  mechanism_confidence = "unknown"
  if c.mechanism.get("status") == "present":
    rows = [c.operand_deltas.get(op, {}) for op in c.mechanism.get("changed_operands", [])]
    confidences = {side.get("candidate") for row in rows
                   for side in (row.get("classification_confidence", {}),)}
    if confidences - {None, "unknown"}:
      mechanism_confidence = "low"
  return {
    "scope": "pair", "candidate": c.candidate_id, "comparator": c.comparator_id,
    "statement": statement, "truth_status": "measured", "confidence": confidence,
    "selection": selection, "selection_confidence": confidence,
    "mechanism_confidence": mechanism_confidence,
    "supporting_fact_ids": list(c.used_facts), "contradicting_fact_ids": [],
  }


def investigate(paths: list[str | Path], question: str) -> Investigation:
  try:
    return _investigate(paths, question)
  except Exception as exc:  # internal invariant failure -> exit 4, never a silent pass
    analysis = KernelAnalysis(question=question or "(none)", candidates=(),
                              eligibility={"error": str(exc)}, comparability={})
    return Investigation(analysis=analysis, exit_code=EXIT_INTERNAL,
                         markdown=f"# Kernel analysis failed\n\nInternal invariant failure: {exc}\n")


def _investigate(paths: list[str | Path], question: str) -> Investigation:
  ingest: IngestResult = ingest_kernel_artifacts(paths)
  evidence = ingest.evidence

  if not evidence:
    exit_code = EXIT_INVALID if ingest.invalid else EXIT_NO_EVIDENCE
    reason = "known but invalid evidence" if ingest.invalid else "no supported evidence"
    analysis = KernelAnalysis(
      question=question or "(none)", candidates=(),
      eligibility={"reason": reason,
                   "invalid": [r.to_json() for r in ingest.invalid],
                   "unsupported": [r.to_json() for r in ingest.unsupported]},
      comparability={})
    return Investigation(analysis=analysis, exit_code=exit_code,
                         markdown=_render(analysis, ingest))

  investigation = analyze_evidence(evidence, question)
  return Investigation(analysis=investigation.analysis, exit_code=EXIT_OK,
                       markdown=_render(investigation.analysis, ingest))


def analyze_evidence(evidence: tuple[KernelEvidence, ...], question: str) -> Investigation:
  """Analyze already-ingested evidence (no file IO). Always a successful analysis (exit 0)."""
  # Adapters normalize provider facts; interpretation belongs to the pure classifier.
  evidence = tuple(classify_kernel_evidence(ev) for ev in evidence)
  eligibility = {ev.candidate.candidate_id: {k: v.to_json() for k, v in evaluate_all(ev).items()}
                 for ev in evidence}

  pairs = _pairs(evidence)
  contrasts: list[PairContrast] = []
  comparability: dict[str, Any] = {}
  for cand, comp in pairs:
    c = contrast_pair(cand, comp)
    contrasts.append(c)
    comparability[f"{cand.candidate.candidate_id}->{comp.candidate.candidate_id}"] = {
      "performance_eligible": c.performance_eligible, "blockers": list(c.blockers)}

  contrast_by_cand = {c.candidate_id: c for c in contrasts}
  hypotheses: list[dict[str, Any]] = []
  hyps_by_cand: dict[str, tuple] = {}
  for ev in evidence:
    cid = ev.candidate.candidate_id
    hs = diagnose(ev, contrast=contrast_by_cand.get(cid))
    hyps_by_cand[cid] = hs
    hypotheses.extend(h.to_json() for h in hs)

  conclusions = [_conclusion_from_contrast(c) for c in contrasts]
  baseline_ids = {ev.candidate.comparator_id for ev in evidence
                  if ev.candidate.comparator_id and ev.candidate.comparator_id in {x.candidate.candidate_id for x in evidence}}
  if len(evidence) >= 2 and len(baseline_ids) == 1:
    matrix = rank_measured_matrix(evidence, baseline_id=next(iter(baseline_ids)))
    statement = {"promote": f"measured matrix winner is {matrix.winner_id}",
                 "retain": f"measured matrix retains baseline {matrix.baseline_id}",
                 "inconclusive": "measured matrix is inconclusive because top uncertainty intervals overlap",
                 "unsupported": "measured matrix is unsupported", "blocked": "measured matrix is blocked"}[matrix.status]
    conclusions.append({"scope": "matrix", **matrix.to_json(), "statement": statement,
                        "truth_status": "measured" if matrix.status in {"promote", "retain", "inconclusive"} else "unknown",
                        "confidence": matrix.selection_confidence, "selection": matrix.status,
                        "supporting_fact_ids": [ev.evidence_id for ev in evidence
                                                if any(row.candidate_id == ev.candidate.candidate_id and row.eligible
                                                       for row in matrix.rows)],
                        "contradicting_fact_ids": []})
    comparability["measured_matrix"] = {"status": matrix.status, "winner_id": matrix.winner_id,
                                         "blockers": list(matrix.blockers)}
  unknowns = _collect_unknowns(evidence, contrasts)

  recommendation = None
  for cand, comp in pairs:
    rec = recommend(cand, comp, contrast=contrast_by_cand.get(cand.candidate.candidate_id),
                    hypotheses=hyps_by_cand.get(cand.candidate.candidate_id, ()))
    if rec is not None:
      recommendation = rec.to_json()
      break
  if recommendation is None and not pairs:
    # single candidate: recommend on it alone
    rec = recommend(evidence[0], hypotheses=hyps_by_cand.get(evidence[0].candidate.candidate_id, ()))
    recommendation = rec.to_json() if rec else None

  analysis = KernelAnalysis(
    question=question or "(none)", candidates=evidence, eligibility=eligibility,
    comparability=comparability,
    contrasts=tuple(c.to_json() for c in contrasts),
    hypotheses=tuple(hypotheses), conclusions=tuple(conclusions), unknowns=tuple(unknowns),
    recommendation=recommendation, evidence_refs=tuple(e.evidence_id for e in evidence),
  )
  empty = IngestResult(evidence=evidence, orphans=(), unsupported=(), invalid=())
  return Investigation(analysis=analysis, exit_code=EXIT_OK, markdown=_render(analysis, empty))


def _collect_unknowns(evidence, contrasts) -> list[dict[str, Any]]:
  out = []
  for c in contrasts:
    for fact in c.missing_facts:
      out.append({"scope": f"{c.candidate_id}->{c.comparator_id}", "missing": fact})
  for ev in evidence:
    if ev.correctness is not None and ev.correctness.missing_metrics:
      out.append({"scope": ev.candidate.candidate_id,
                  "missing": f"correctness:{','.join(ev.correctness.missing_metrics)}"})
    for operand, path in ev.operand_paths:
      for fact in path.classification.missing_discriminators:
        out.append({"scope": f"{ev.candidate.candidate_id}:{operand}", "missing": fact})
  return out


def _site_counts(path) -> str:
  counts: dict[str, int] = {}
  for site in path.static.instruction_sites:
    counts[site.kind] = counts.get(site.kind, 0) + 1
  return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none"


def _operand_rows(ev: KernelEvidence) -> list[str]:
  rows = []
  for operand, path in ev.operand_paths:
    static = path.static
    dynamic = path.dynamic
    static_text = (f"sites={_site_counts(path)}; fragment={static.fragment_bytes_per_wave or '-'}B/wave; "
                   f"LDS={static.operand_lds_bytes if static.operand_lds_bytes is not None else '-'}B; "
                   f"spill={static.spill_bytes if static.spill_bytes is not None else '-'}B; "
                   f"fetch_groups={len(static.compulsory_fetch_groups)}")
    amp = (dynamic.transferred_bytes / dynamic.useful_bytes
           if dynamic.useful_bytes and dynamic.transferred_bytes is not None else None)
    tiers = ", ".join(f"{t.tier}:{t.status}" for t in dynamic.serving_tiers) or "unknown"
    dynamic_text = (f"useful={dynamic.useful_bytes if dynamic.useful_bytes is not None else '-'}B; "
                    f"transferred={dynamic.transferred_bytes if dynamic.transferred_bytes is not None else '-'}B; "
                    f"amp={amp:.3g}" if amp is not None else
                    f"useful={dynamic.useful_bytes if dynamic.useful_bytes is not None else '-'}B; "
                    f"transferred={dynamic.transferred_bytes if dynamic.transferred_bytes is not None else '-'}B; amp=-")
    contradictions = ", ".join(path.classification.contradicting_evidence_ids) or "none"
    missing = ", ".join(path.classification.missing_discriminators) or "none"
    rows.append(f"| `{operand}` | {path.declared_strategy} | {path.classification.strategy} | "
                f"{path.classification.confidence}/{path.classification.status} | {static_text} | "
                f"{dynamic_text} | {tiers} | {contradictions} | {missing} |")
  return rows


def _render(analysis: KernelAnalysis, ingest: IngestResult) -> str:
  L = [f"# Kernel analysis", "", f"**Question:** {analysis.question}", ""]
  # Lead with the conclusion.
  L.append("## Conclusion")
  if analysis.conclusions:
    for c in analysis.conclusions:
      L.append(f"- {c['statement']} _(confidence: {c['confidence']}, {c['truth_status']})_")
  else:
    reason = analysis.eligibility.get("reason") if isinstance(analysis.eligibility, dict) else None
    L.append(f"- No conclusion: {reason or 'no eligible comparison'}.")
  L.append("")

  L.append("## Operand paths")
  any_paths = False
  for ev in analysis.candidates:
    if not ev.operand_paths:
      continue
    any_paths = True
    L.extend((f"### `{ev.candidate.candidate_id}`", "",
              "| Operand | Declared | Classified | Confidence/status | Static | Dynamic | Serving tiers | Contradictions | Missing |",
              "|---|---|---|---|---|---|---|---|---|"))
    L.extend(_operand_rows(ev))
    L.append("")
  if not any_paths:
    L.extend(("- no per-operand evidence recorded", ""))

  # Eligibility.
  L.append("## Candidate eligibility")
  if analysis.candidates:
    for ev in analysis.candidates:
      cid = ev.candidate.candidate_id
      elig = analysis.eligibility.get(cid, {})
      flags = ", ".join(f"{k}={'yes' if v.get('eligible') else 'no'}" for k, v in elig.items())
      L.append(f"- `{cid}`: {flags}")
  else:
    for r in ingest.invalid:
      L.append(f"- invalid: {r.schema or '?'} — {r.message}")
    for r in ingest.unsupported:
      L.append(f"- unsupported: {r.schema or '?'} — {r.message}")
  L.append("")

  # Unknowns.
  L.append("## Unknowns")
  if analysis.unknowns:
    for u in analysis.unknowns:
      L.append(f"- `{u['scope']}`: {u['missing']}")
  else:
    L.append("- none recorded")
  L.append("")

  # Next experiment.
  L.append("## Next experiment")
  if analysis.recommendation:
    r = analysis.recommendation
    var = r.get("variable") or f"two-factor {r.get('hints', {}).get('factorial', [])}"
    L.append(f"- **Hypothesis:** {r['hypothesis']}")
    L.append(f"- **Intended variable:** {var}")
    if r.get("required_artifacts"):
      L.append(f"- **Required:** {'; '.join(r['required_artifacts'])}")
    for outcome in r.get("expected_outcomes", []):
      L.append(f"- Expected: {outcome}")
  else:
    L.append("- none — the question is answered by current evidence")
  L.append("")
  return "\n".join(L)


def write_outputs(investigation: Investigation, out_dir: str | Path) -> dict[str, str]:
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  analysis = investigation.analysis
  files = {}
  ev_path = out / "kernel_evidence.json"
  ev_path.write_text(json.dumps([e.to_json() for e in analysis.candidates], indent=2, sort_keys=True))
  files["kernel_evidence.json"] = str(ev_path)
  an_path = out / "kernel_analysis.json"
  an_path.write_text(json.dumps(analysis.to_json(), indent=2, sort_keys=True))
  files["kernel_analysis.json"] = str(an_path)
  md_path = out / "kernel_analysis.md"
  md_path.write_text(investigation.markdown)
  files["kernel_analysis.md"] = str(md_path)
  rec_path = out / "kernel_recommendation.json"
  rec_path.write_text(json.dumps(analysis.recommendation or {}, indent=2, sort_keys=True))
  files["kernel_recommendation.json"] = str(rec_path)
  return files
