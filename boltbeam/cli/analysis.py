from __future__ import annotations

import json
import pathlib
import sys

from boltbeam.cli._common import _write, _fail, _load_json, _load_evidences
from boltbeam.manifest import candidate_by_id

def _register_diagnose_pathology(sub) -> None:
  diag = sub.add_parser("diagnose", help="compiler pathology diagnostics")
  diagsub = diag.add_subparsers(dest="diagnose_cmd", required=True)
  p = diagsub.add_parser("compiler-pathology", help="classify compiler pathology for a candidate")
  p.add_argument("candidate", help="candidate id (e.g. decode_flash_block_tile_g5_native_context)")
  p.add_argument("--evidence", default=None, help="normalized evidence JSON (compiler_pathology rows)")
  p.add_argument("--out", default=None, help="write pathology report JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="also write a markdown summary (- for stdout)")
  p.set_defaults(fn=cmd_diagnose_pathology)



def cmd_diagnose_pathology(args) -> int:
  from boltbeam.diagnostics.pathology import classify_pathology
  from boltbeam.artifacts.base import read_evidence
  from boltbeam.vocab import RowKind

  evs = _load_evidences(args.evidence) if args.evidence else []
  cp_rows = [r for ev in evs for r in ev.rows if r.kind == RowKind.COMPILER_PATHOLOGY]

  if not cp_rows and not args.evidence:
    # No evidence: produce a timing-only report from candidates.json meta if available.
    cand = candidate_by_id(args.candidate)
    if cand is None:
      return _fail(f"diagnose: unknown candidate id {args.candidate!r} and no --evidence supplied")
    # Build a synthetic timing-only row list from known benchmark data in the description.
    cp_rows = []  # classifier will return NATIVE_ISA_ORACLE_NEEDED

  report = classify_pathology(args.candidate, cp_rows)
  _write(report.to_json(), args.out)
  if args.markdown:
    lines = [
      f"# Compiler Pathology Report — {args.candidate}",
      f"",
      f"**Classification:** `{report.classification}`  ",
      f"**Confidence:** {report.confidence}  ",
      f"",
      f"## Reason",
      f"{report.reason}",
      f"",
      f"## Next action",
      f"{report.next_action}",
      f"",
      f"## Reopen directive",
      f"{report.reopen_directive}",
    ]
    md = "\n".join(lines) + "\n"
    if args.markdown == "-":
      sys.stdout.write(md)
    else:
      pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
      pathlib.Path(args.markdown).write_text(md)
  return 0

def _register_investigate_kernels(sub) -> None:
  p = sub.add_parser("investigate-kernels",
                     help="ingest kernel evidence/fragments -> eligibility, contrast, diagnosis, next experiment")
  p.add_argument("paths", nargs="+", help="kernel evidence/fragment JSON artifacts")
  p.add_argument("--question", default=None, help="the question to answer")
  p.add_argument("--out-dir", default=None, help="write kernel_evidence/analysis/recommendation here")
  p.set_defaults(fn=cmd_investigate_kernels)



def cmd_investigate_kernels(args) -> int:
  from boltbeam.kernel_analysis.report import investigate, write_outputs
  result = investigate(list(args.paths), args.question or "")
  if args.out_dir:
    files = write_outputs(result, args.out_dir)
    sys.stdout.write(json.dumps({"exit_code": result.exit_code, "outputs": files}, indent=2, sort_keys=True) + "\n")
  else:
    sys.stdout.write(result.markdown + "\n")
  return result.exit_code

def _register_compare_kernels(sub) -> None:
  p = sub.add_parser("compare-kernels", help="print contrasts from a kernel_analysis.json (read-only)")
  p.add_argument("analysis", help="kernel_analysis.json")
  p.set_defaults(fn=cmd_compare_kernels)



def cmd_compare_kernels(args) -> int:
  analysis = _load_json(args.analysis)
  sys.stdout.write(json.dumps(analysis.get("contrasts", []), indent=2, sort_keys=True) + "\n")
  return 0

def _register_explain_kernel(sub) -> None:
  p = sub.add_parser("explain-kernel", help="print eligibility + hypotheses for one candidate (read-only)")
  p.add_argument("analysis", help="kernel_analysis.json")
  p.add_argument("--candidate", required=True, help="candidate id")
  p.set_defaults(fn=cmd_explain_kernel)



def cmd_explain_kernel(args) -> int:
  analysis = _load_json(args.analysis)
  hyps = [h for h in analysis.get("hypotheses", [])]
  elig = analysis.get("eligibility", {}).get(args.candidate, {})
  sys.stdout.write(json.dumps({"candidate": args.candidate, "eligibility": elig,
                               "hypotheses": hyps}, indent=2, sort_keys=True) + "\n")
  return 0

def _register_next_experiment(sub) -> None:
  p = sub.add_parser("next-experiment", help="print the recommendation from a kernel_analysis.json (read-only)")
  p.add_argument("analysis", help="kernel_analysis.json")
  p.set_defaults(fn=cmd_next_experiment)



def cmd_next_experiment(args) -> int:
  analysis = _load_json(args.analysis)
  sys.stdout.write(json.dumps(analysis.get("recommendation") or {}, indent=2, sort_keys=True) + "\n")
  return 0

def register(sub) -> None:
  _register_diagnose_pathology(sub)
  _register_investigate_kernels(sub)
  _register_compare_kernels(sub)
  _register_explain_kernel(sub)
  _register_next_experiment(sub)
