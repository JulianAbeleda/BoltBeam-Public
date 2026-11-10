"""Build a CUDA extension's Ninja file before a model loads, each CUDA object in a bounded system service.

An engine started under BOLTBEAM_REQUIRE_PREBUILT=1 refuses to compile while it is measured (build_jobs.PrebuildRequired
names the Ninja file); this builds that file outside the measurement. Each CUDA object is measured serially for its
compiler memory peak, then the module is built with the bounded Ninja job count derived from those peaks. A matching
report resumes an interrupted calibration (docs/cuda-build-limits.md)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from boltbeam.runtime import build_jobs


def bounded_build(command:list[str], limit:int, log:Path) -> int:
  from boltbeam.runtime.bounded_job import run
  result = run(command, log=log, limit=limit,
               env={k: v for k, v in os.environ.items() if k in
                    ('PATH', 'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'HOME', 'TMPDIR')})
  if result['exit_code'] or result['memory_peak_bytes'] <= 0:
    raise RuntimeError(f'Bounded build failed; inspect {log}')
  return result['memory_peak_bytes']


def prebuild(ninja_file:Path, *, report_path:Path, ninja:str = 'ninja', include_dirs:list[Path] = (), memory_limit_bytes:int | None = None,
             measured_targets:list[str] = (), enrich:bool = False) -> dict:
  """Build ninja_file's module, bounded; the build profile, also written to report_path."""
  if memory_limit_bytes is not None and memory_limit_bytes <= 0:
    raise ValueError('memory limit must be positive')
  ninja_file = Path(ninja_file).resolve()
  ninja_sha256 = hashlib.sha256(ninja_file.read_bytes()).hexdigest()
  source_identity = build_jobs.build_identity(ninja_file, ninja, include_deps=False)
  if include_dirs:
    os.environ['CPATH'] = os.pathsep.join([*(str(p.resolve()) for p in include_dirs),
                                         *([os.environ['CPATH']] if os.environ.get('CPATH') else [])])
  records = build_jobs.cuda_build_records(ninja_file, ninja)
  targets = list(dict.fromkeys(str((Path(r.get('directory') or ninja_file.parent) / r['output']).resolve())
                               for r in records if r.get('output')))
  if not targets:
    raise ValueError('no CUDA compilation targets found')
  if enrich:
    previous = json.loads(report_path.read_text())
    measured_targets = previous.get('measured_targets') or {}
    if (previous.get('ninja_file') != str(ninja_file) or previous.get('ninja_sha256') != ninja_sha256 or
        set(measured_targets) != set(targets) or any(int(v) <= 0 for v in measured_targets.values())):
      raise ValueError('report must contain a positive measured peak for every CUDA target of this Ninja file')
    previous.update(source_identity=source_identity,
                    build_identity=build_jobs.build_identity(ninja_file, ninja),
                    measurement_method='live_cgroup_peak', compiler_peak_bytes=max(map(int, measured_targets.values())))
    report_path.write_text(json.dumps(previous, indent=2))
    return previous
  measured = {}
  if report_path.is_file():
    try:
      previous = json.loads(report_path.read_text())
      if (previous.get('ninja_file') == str(ninja_file) and
          previous.get('ninja_sha256') == ninja_sha256 and
          previous.get('source_identity') == source_identity and
          previous.get('measurement_method') == 'live_cgroup_peak'):
        measured.update({target: int(peak) for target, peak in previous['measured_targets'].items()
                         if target in targets and int(peak) > 0 and Path(target).is_file()})
    except (OSError, ValueError, TypeError, KeyError):
      pass
  for entry in measured_targets:
    try:
      target, peak = entry.rsplit('=', 1)
      peak = int(peak)
    except ValueError:
      raise ValueError('--measured-target requires OBJECT=BYTES')
    target = str(Path(target).resolve())
    if target not in targets or peak <= 0:
      raise ValueError('measured target must be listed in Ninja file with positive peak')
    measured[target] = peak
  report_path.parent.mkdir(parents=True, exist_ok=True)
  def memory_limit() -> int:
    available = build_jobs.available_memory()
    if memory_limit_bytes is not None:
      available = min(available, memory_limit_bytes)
    if memory_limit_bytes is None and measured:
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
    report_path.write_text(json.dumps(report(), indent=2))
  base = [ninja, '-C', str(ninja_file.parent), '-f', str(ninja_file)]
  for target in targets:
    if target in measured:
      continue
    dry = subprocess.run([*base, '-n', target], capture_output=True, text=True, check=True)
    if 'ninja: no work to do.' in dry.stdout:
      # The object may have been compiled in an interrupted calibration. Its
      # earlier peak cannot be trusted, so force this output to be rebuilt.
      Path(target).unlink()
    log = report_path.with_name(report_path.stem + '.' + Path(target).name + '.log')
    measured[target] = bounded_build([*base, '-j', '1', target], memory_limit(), log)
    report_path.write_text(json.dumps(report(), indent=2))
  profile = report()
  profile['build_identity'] = build_jobs.build_identity(ninja_file, ninja)
  report_path.write_text(json.dumps(profile, indent=2))
  build_env = {**os.environ, 'BOLTBEAM_BUILD_PROFILE': str(report_path)}
  available = build_jobs.available_memory()
  cpus = build_jobs.available_cpus()
  cap = memory_limit()
  jobs = build_jobs.derived_jobs(build_file=ninja_file,
                                 ninja=ninja, memory_bytes=cap, cpus=cpus,
                                 env=build_env)
  if jobs is None:
    try:
      jobs = int(os.environ['MAX_JOBS'])
      if jobs <= 0:
        raise ValueError
    except ValueError:
      raise ValueError('MAX_JOBS must be a positive integer')
  print(json.dumps({'jobs': jobs, 'compiler_peak_bytes': profile['compiler_peak_bytes']}), flush=True)
  profile['parallel_build'] = {'jobs': jobs, 'available_memory_bytes': available,
                               'cpu_count': cpus, 'memory_limit_bytes': cap,
                               'compiler_peak_bytes': profile['compiler_peak_bytes']}
  report_path.write_text(json.dumps(profile, indent=2))
  profile['final_build_peak_bytes'] = bounded_build([*base, '-j', str(jobs)], cap,
                                                   report_path.with_suffix('.build.log'))
  report_path.write_text(json.dumps(profile, indent=2))
  return profile


PREBUILD_TRIES = 8  # modules one engine start may ask for in turn (Nemotron 3.5 on vLLM 0.31.0, RTX 5090: xqa, then trtllm_utils)


def required(text:str) -> Path | None:
  """The Ninja file a refused engine start names (build_jobs.PrebuildRequired), or None."""
  import re
  found = re.search(r"Prebuild required: (\S+?build\.ninja)", text)
  return Path(found.group(1)) if found else None


def ninja_beside(python:str | None) -> str:
  """The ninja an engine's own build uses: its venv's, else the one on PATH."""
  import shutil
  own = Path(python).parent / 'ninja' if python else None
  return str(own) if own and own.is_file() else shutil.which('ninja') or 'ninja'
