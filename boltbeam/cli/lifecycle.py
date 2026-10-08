from __future__ import annotations

import json
import pathlib

from boltbeam.cli._common import _write, _fail


def _register_lifecycle_compare(sub) -> None:
  p = sub.add_parser("lifecycle-compare", help="per workload, ours (tinygrad HCQ graph profile) vs vLLM (nsys sqlite): "
                     "kernel sum / wall / launches per step, gap split into per-role + per-category kernel deltas and "
                     "a lifecycle delta (wall - kernel sum, launch count)")
  p.add_argument("--manifest", default=None, help="manifest JSON {model, pattern, ours_gemm_json, workloads:[{id, kind, "
                 "batch, ours:{jsonl, meta, prompt_len, command}, vllm:{sqlite, prompt_len, command}}]}; default: the "
                 "Nemotron-H 4B / RTX 5090 trace locations")
  p.add_argument("--ours-dir", default=None, help="default manifest only: directory holding ours_b{B}_p{P}.jsonl/_meta.json")
  p.add_argument("--workloads", default=None, help="comma-separated workload ids to run (default: all)")
  p.add_argument("--print-manifest", action="store_true", help="print the effective manifest and exit")
  p.add_argument("--out", default=None, help="write the boltbeam.lifecycle_comparison.v1 JSON here instead of stdout")
  p.add_argument("--markdown", default=None, help="also write the markdown tables here")
  p.set_defaults(fn=cmd_lifecycle_compare)


def cmd_lifecycle_compare(args) -> int:
  from boltbeam.trace.lifecycle_compare import nemotron_default_manifest, render_markdown, run_manifest
  try:
    if args.manifest:
      manifest = json.loads(pathlib.Path(args.manifest).read_text())
    else:
      manifest = nemotron_default_manifest(**({"ours_dir": args.ours_dir} if args.ours_dir else {}))
  except (OSError, json.JSONDecodeError) as exc:
    return _fail(f"lifecycle-compare: {exc}")
  if args.print_manifest:
    _write(manifest, None); return 0
  only = [w for w in args.workloads.split(",") if w] if args.workloads else None
  try:
    doc = run_manifest(manifest, only)
  except (KeyError, ValueError) as exc:
    return _fail(f"lifecycle-compare: {exc}")
  _write(doc, args.out)
  if args.markdown:
    pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.markdown).write_text(render_markdown(doc))
  return 0


def register(sub) -> None:
  _register_lifecycle_compare(sub)
