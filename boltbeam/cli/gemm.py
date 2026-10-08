"""GEMM strategy commands: the derived space for one shape, and the per-shape scan table."""
from __future__ import annotations

import json
import pathlib

from boltbeam.cli._common import _write


def _lowering(path:str):
  from boltbeam.search.gemm_strategy import GemmLowering
  return GemmLowering.from_json(json.loads(pathlib.Path(path).read_text()))


def _jsonl_or_json(path:str | None) -> list[dict]:
  if not path: return []
  text = pathlib.Path(path).read_text()
  if path.endswith(".jsonl"): return [json.loads(line) for line in text.splitlines() if line.strip()]
  data = json.loads(text)
  return data["rows"] if isinstance(data, dict) and "rows" in data else data


def cmd_gemm_space(args) -> int:
  from boltbeam.search.gemm_strategy import GemmMachine, Shape, space_document
  machine = GemmMachine.from_target(args.target)
  _write(space_document(machine, _lowering(args.lowering), Shape(args.m, args.n, args.k), top=args.top), args.out)
  return 0


def cmd_gemm_scan(args) -> int:
  from boltbeam.search.gemm_scan import markdown, scan_table
  from boltbeam.search.gemm_strategy import GemmMachine
  table = scan_table(GemmMachine.from_target(args.target), _lowering(args.lowering), _jsonl_or_json(args.reference),
                     selection=_jsonl_or_json(args.selection), ours=_jsonl_or_json(args.ours), measured=_jsonl_or_json(args.measured),
                     gemm_rows_per_token=args.gemm_rows_per_token, roles=set(args.roles.split(",")) if args.roles else None)
  if args.markdown: pathlib.Path(args.markdown).write_text(markdown(table))
  _write(table, args.out)
  return 0


def cmd_gemm_calibrate(args) -> int:
  from boltbeam.search.gemm_calibration import calibrate
  from boltbeam.search.gemm_strategy import GemmMachine
  rows = [r for path in args.measured for r in _jsonl_or_json(path)]
  _write(calibrate(GemmMachine.from_target(args.target), _lowering(args.lowering), rows), args.out)
  return 0


def register(sub) -> None:
  p = sub.add_parser("gemm-space", help="the tile-GEMM strategy space derived from the target's facts + the compiler's lowering facts, for one shape")
  p.add_argument("--target", default="nvidia_sm120")
  p.add_argument("--lowering", required=True, help="boltbeam.gemm_lowering_facts.v1 JSON (tinygrad: extra/llm_research/gemm_lowering_facts.py)")
  p.add_argument("--m", type=int, required=True); p.add_argument("--n", type=int, required=True); p.add_argument("--k", type=int, required=True)
  p.add_argument("--top", type=int, default=8)
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_gemm_space)
  p = sub.add_parser("gemm-calibrate", help="fit the cost model's term coefficients to measured candidate times; validation held out by shape")
  p.add_argument("--lowering", required=True); p.add_argument("--target", default="nvidia_sm120")
  p.add_argument("--measured", action="append", required=True, help="measured candidate rows (jsonl/json; repeatable)")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_gemm_calibrate)
  p = sub.add_parser("gemm-scan", help="per-shape table: derived candidate vs promoted route vs reference (vLLM) vs roofline, with the limiting factor")
  p.add_argument("--target", default="nvidia_sm120")
  p.add_argument("--lowering", required=True, help="boltbeam.gemm_lowering_facts.v1 JSON")
  p.add_argument("--reference", required=True, help="reference timings per (role, m, n, k) with kernels (kernel audit vllm_gemm.json)")
  p.add_argument("--selection", help="promoted routes (tinygrad dense_bf16_sm120_selection.json)")
  p.add_argument("--ours", help="our production timings per (role, m) (kernel audit ours.json)")
  p.add_argument("--measured", help="derived-space measurements (tinygrad dense_bf16_geometry_search.py --scan JSONL)")
  p.add_argument("--gemm-rows-per-token", type=int, default=2, help="GEMM rows our route runs per token (2 under hi/lo)")
  p.add_argument("--roles", help="comma list of roles")
  p.add_argument("--markdown", help="also write the table as markdown here")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_gemm_scan)
