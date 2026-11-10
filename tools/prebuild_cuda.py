#!/usr/bin/env python3
"""Measure CUDA compiler memory serially, then derive a bounded Ninja build count.

A transient system service running as the calling user enforces a memory bound and samples its live
cgroup peak. Every CUDA object is measured separately. A matching report can
resume interrupted calibration; --measured-target imports an earlier peak.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boltbeam.runtime.prebuild_cuda import prebuild


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('ninja_file', type=Path)
  ap.add_argument('--report', type=Path, required=True)
  ap.add_argument('--ninja', default='ninja')
  ap.add_argument('--include-dir', type=Path, action='append', default=[])
  ap.add_argument('--memory-limit-bytes', type=int,
                  help='optional hard cap; default is available RAM/cgroup headroom minus one measured compiler peak')
  ap.add_argument('--measured-target', action='append', default=[], metavar='OBJECT=BYTES',
                  help='reuse a previously measured completed object')
  ap.add_argument('--enrich-report', action='store_true',
                  help='add current CUDA source/compiler/header identity to a completed live-sampled report without building')
  args = ap.parse_args()
  try:
    prebuild(args.ninja_file, report_path=args.report, ninja=args.ninja, include_dirs=args.include_dir, memory_limit_bytes=args.memory_limit_bytes,
             measured_targets=args.measured_target, enrich=args.enrich_report)
  except ValueError as exc:
    ap.error(str(exc))


if __name__ == '__main__':
  main()
