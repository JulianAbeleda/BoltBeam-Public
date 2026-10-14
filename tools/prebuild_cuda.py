#!/usr/bin/env python3
"""Measure CUDA compiler memory serially, then derive a bounded Ninja build count.

A transient system service running as the calling user enforces a memory bound and samples its live
cgroup peak. Every CUDA object is measured separately. A matching report can
resume interrupted calibration; --measured-target imports an earlier peak.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'boltbeam' / 'runtime'))
import build_jobs


def bounded_build(command: list[str], limit: int, log: Path) -> int:
  from bounded_job import run
  result = run(command, log=log, limit=limit,
               env={k: v for k, v in os.environ.items() if k in
                    ('PATH', 'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'HOME', 'TMPDIR')})
  if result['exit_code'] or result['memory_peak_bytes'] <= 0:
    raise RuntimeError(f'Bounded build failed; inspect {log}')
  return result['memory_peak_bytes']


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
  if args.memory_limit_bytes is not None and args.memory_limit_bytes <= 0:
    ap.error('--memory-limit-bytes must be positive')
  ninja_file = args.ninja_file.resolve()
  ninja_sha256 = hashlib.sha256(ninja_file.read_bytes()).hexdigest()
  source_identity = build_jobs.build_identity(ninja_file, args.ninja, include_deps=False)
  if args.include_dir:
    os.environ['CPATH'] = os.pathsep.join([*(str(p.resolve()) for p in args.include_dir),
                                         *([os.environ['CPATH']] if os.environ.get('CPATH') else [])])
  records = build_jobs.cuda_build_records(ninja_file, args.ninja)
  targets = list(dict.fromkeys(str((Path(r.get('directory') or ninja_file.parent) / r['output']).resolve())
                               for r in records if r.get('output')))
  if not targets:
    ap.error('no CUDA compilation targets found')
  if args.enrich_report:
    previous = json.loads(args.report.read_text())
    measured_targets = previous.get('measured_targets') or {}
    if (previous.get('ninja_file') != str(ninja_file) or previous.get('ninja_sha256') != ninja_sha256 or
        set(measured_targets) != set(targets) or any(int(v) <= 0 for v in measured_targets.values())):
      ap.error('report must contain a positive measured peak for every CUDA target of this Ninja file')
    previous.update(source_identity=source_identity,
                    build_identity=build_jobs.build_identity(ninja_file, args.ninja),
                    measurement_method='live_cgroup_peak', compiler_peak_bytes=max(map(int, measured_targets.values())))
    args.report.write_text(json.dumps(previous, indent=2))
    return
  measured = {}
  if args.report.is_file():
    try:
      previous = json.loads(args.report.read_text())
      if (previous.get('ninja_file') == str(ninja_file) and
          previous.get('ninja_sha256') == ninja_sha256 and
          previous.get('source_identity') == source_identity and
          previous.get('measurement_method') == 'live_cgroup_peak'):
        measured.update({target: int(peak) for target, peak in previous['measured_targets'].items()
                         if target in targets and int(peak) > 0 and Path(target).is_file()})
    except (OSError, ValueError, TypeError, KeyError):
      pass
  for entry in args.measured_target:
    try:
      target, peak = entry.rsplit('=', 1)
      peak = int(peak)
    except ValueError:
      ap.error('--measured-target requires OBJECT=BYTES')
    target = str(Path(target).resolve())
    if target not in targets or peak <= 0:
      ap.error('measured target must be listed in Ninja file with positive peak')
    measured[target] = peak
  args.report.parent.mkdir(parents=True, exist_ok=True)
  def memory_limit() -> int:
    available = build_jobs.available_memory()
    if args.memory_limit_bytes is not None:
      available = min(available, args.memory_limit_bytes)
    if args.memory_limit_bytes is None and measured:
      available -= max(measured.values())
    if available <= 0:
      raise RuntimeError('Insufficient available memory for a bounded compiler')
    return available

  def report() -> dict:
    return {'ninja_file': str(ninja_file), 'ninja_sha256': ninja_sha256,
            'source_identity': source_identity,
            'measurement_method': 'live_cgroup_peak', 'measured_targets': measured,
            'compiler_peak_bytes': max(measured.values())}

  if measured:
    args.report.write_text(json.dumps(report(), indent=2))
  base = [args.ninja, '-C', str(ninja_file.parent), '-f', str(ninja_file)]
  for target in targets:
    if target in measured:
      continue
    dry = subprocess.run([*base, '-n', target], capture_output=True, text=True, check=True)
    if 'ninja: no work to do.' in dry.stdout:
      # The object may have been compiled in an interrupted calibration. Its
      # earlier peak cannot be trusted, so force this output to be rebuilt.
      Path(target).unlink()
    log = args.report.with_name(args.report.stem + '.' + Path(target).name + '.log')
    measured[target] = bounded_build([*base, '-j', '1', target], memory_limit(), log)
    args.report.write_text(json.dumps(report(), indent=2))
  profile = report()
  profile['build_identity'] = build_jobs.build_identity(ninja_file, args.ninja)
  args.report.write_text(json.dumps(profile, indent=2))
  build_env = {**os.environ, 'BOLTBEAM_BUILD_PROFILE': str(args.report)}
  available = build_jobs.available_memory()
  cpus = build_jobs.available_cpus()
  cap = memory_limit()
  jobs = build_jobs.derived_jobs(build_file=ninja_file,
                                 ninja=args.ninja, memory_bytes=cap, cpus=cpus,
                                 env=build_env)
  if jobs is None:
    try:
      jobs = int(os.environ['MAX_JOBS'])
      if jobs <= 0:
        raise ValueError
    except ValueError:
      ap.error('MAX_JOBS must be a positive integer')
  print(json.dumps({'jobs': jobs, 'compiler_peak_bytes': profile['compiler_peak_bytes']}), flush=True)
  profile['parallel_build'] = {'jobs': jobs, 'available_memory_bytes': available,
                               'cpu_count': cpus, 'memory_limit_bytes': cap,
                               'compiler_peak_bytes': profile['compiler_peak_bytes']}
  args.report.write_text(json.dumps(profile, indent=2))
  profile['final_build_peak_bytes'] = bounded_build([*base, '-j', str(jobs)], cap,
                                                   args.report.with_suffix('.build.log'))
  args.report.write_text(json.dumps(profile, indent=2))


if __name__ == '__main__':
  main()
