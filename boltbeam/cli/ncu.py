"""NCU commands: collect any target command under ncu, import a raw export, and audit ours vs a reference."""
from __future__ import annotations

import json
import pathlib

from boltbeam.cli._common import _write


def cmd_ncu_collect(args) -> int:
  from boltbeam.profiler.ncu_counters import collect
  env = dict(kv.split("=", 1) for kv in args.env or [])
  target = args.target[1:] if args.target and args.target[0] == "--" else args.target
  doc = collect(target, out=args.report, side=args.side, ncu=args.ncu, sudo_mem_max=None if args.no_sudo else args.mem_max,
                env=env, nvtx=args.nvtx)
  _write(doc, args.out)
  return 0


def cmd_ncu_import(args) -> int:
  from boltbeam.profiler.ncu_counters import import_raw_csv
  stdout = pathlib.Path(args.target_stdout).read_text() if args.target_stdout else None
  _write(import_raw_csv(pathlib.Path(args.csv).read_text(), side=args.side, target_stdout=stdout, source_report=args.csv), args.out)
  return 0


def cmd_ncu_audit(args) -> int:
  from boltbeam.profiler.ncu_audit import audit, census_from_sass, markdown
  from boltbeam.search.gemm_strategy import GemmLowering, GemmMachine
  counters = [json.loads(pathlib.Path(p).read_text()) for p in args.counters]
  timings = json.loads(pathlib.Path(args.dims_from).read_text())
  timings = timings["rows"] if isinstance(timings, dict) else timings
  dims = {r["role"]: (r["n"], r["k"]) for r in timings}
  census = {}
  for spec in args.sass or []:
    prefix, path = spec.split("=", 1)
    census[prefix] = census_from_sass(pathlib.Path(path).read_text())
  for path in args.census or []:
    census.update(json.loads(pathlib.Path(path).read_text()))
  lowering = GemmLowering.from_json(json.loads(pathlib.Path(args.lowering).read_text())) if args.lowering else None
  doc = audit(counters, dims, machine=GemmMachine.from_target(args.target), lowering=lowering,
              gemm_rows_per_token=args.gemm_rows_per_token, census=census)
  if args.markdown: pathlib.Path(args.markdown).write_text(markdown(doc))
  _write(doc, args.out)
  return 0


def register(sub) -> None:
  p = sub.add_parser("ncu-collect", help="profile a target command under ncu (memory-capped sudo scope), export and import its kernel counters")
  p.add_argument("--report", required=True, help="ncu -o report prefix")
  p.add_argument("--side", required=True, choices=("ours", "reference"))
  p.add_argument("--ncu", default="ncu")
  p.add_argument("--mem-max", default="20G"); p.add_argument("--no-sudo", action="store_true")
  p.add_argument("--nvtx", action="store_true", help="label launches by NVTX push/pop range 'role/M'")
  p.add_argument("--env", action="append", help="KEY=VALUE for the target (repeatable)")
  p.add_argument("--out", default=None)
  p.add_argument("target", nargs="...", help="-- the target command; it labels launches with NVTX 'role/M' ranges or 'SHAPE role M kernels' stdout lines")
  p.set_defaults(fn=cmd_ncu_collect)
  p = sub.add_parser("ncu-import", help="import an 'ncu --page raw --csv' export into boltbeam.ncu_kernel_counters.v1")
  p.add_argument("csv"); p.add_argument("--side", required=True, choices=("ours", "reference"))
  p.add_argument("--target-stdout", help="the target's stdout with 'SHAPE role M kernels' lines (when there is no NVTX)")
  p.add_argument("--out", default=None); p.set_defaults(fn=cmd_ncu_import)
  p = sub.add_parser("ncu-audit", help="ours vs reference per shape: kernel facts, roofline, the gap by known limiter + UNEXPLAINED, research queue")
  p.add_argument("--counters", action="append", required=True, help="ncu counters JSON (ours and reference; repeatable)")
  p.add_argument("--dims-from", required=True, help="JSON rows with role, n, k (e.g. the kernel audit's vllm_gemm.json)")
  p.add_argument("--lowering", help="boltbeam.gemm_lowering_facts.v1 (roofline + rows bucket)")
  p.add_argument("--target", default="nvidia_sm120")
  p.add_argument("--gemm-rows-per-token", type=int, default=2)
  p.add_argument("--sass", action="append", help="KERNEL_PREFIX=path.sass static opcode census (repeatable)")
  p.add_argument("--census", action="append", help="JSON {kernel_prefix: {OPCODE: count}} (repeatable)")
  p.add_argument("--markdown"); p.add_argument("--out", default=None); p.set_defaults(fn=cmd_ncu_audit)
