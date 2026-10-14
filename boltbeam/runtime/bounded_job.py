"""Run memory-heavy work as the calling user in an isolated system service."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
import subprocess
import time
import uuid
import build_jobs

GIB = 1024 ** 3


def host_memory() -> int:
  return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                  if x.startswith('MemTotal:'))) * 1024


def prepare_slice():
  # The aggregate cap also protects the host when multiple jobs overlap.
  subprocess.run(['sudo', '-n', 'systemctl', 'set-property', '--runtime', 'boltbeam.slice',
                  'MemoryMax=' + str(host_memory() - max(4 * GIB, host_memory() // 8)), 'MemoryHigh=infinity',
                  'MemorySwapMax=0'], check=True, timeout=15)
  subprocess.run(['sudo', '-n', 'systemctl', 'daemon-reload'], check=True, timeout=15)
  subprocess.run(['sudo', '-n', 'systemctl', 'start', 'boltbeam.slice'], check=True, timeout=15)


def memory_budget(requested: int | None = None) -> int:
  available = build_jobs.available_memory()
  total = host_memory()
  safe = min(available, total) - max(4 * GIB, total // 8)
  if requested is not None and requested <= 0:
    raise ValueError('Memory limit must be positive')
  limit = safe if requested is None else min(requested, safe)
  if limit <= 0:
    raise RuntimeError('Insufficient RAM after reserving host headroom')
  return limit


def service_command(command, unit, limit, env, cwd, timeout):
  return ['sudo', '-n', 'systemd-run', '--system', '--wait', '--pipe', '--unit=' + unit,
          '-p', 'User=' + str(os.getuid()), '-p', 'Group=' + str(os.getgid()),
          '-p', 'WorkingDirectory=' + str(Path(cwd).resolve()), '-p', 'Slice=boltbeam.slice',
          '-p', 'MemoryAccounting=yes', '-p', 'MemoryMax=' + str(limit),
          '-p', 'MemoryHigh=infinity', '-p', 'MemorySwapMax=0',
          '-p', 'OOMPolicy=kill', '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=5',
          '-p', 'RuntimeMaxSec=' + str(timeout),
          *['--setenv=' + k + '=' + v for k, v in env.items()], '--', *command]


def stop(unit):
  subprocess.run(['sudo', '-n', 'systemctl', 'stop', unit], capture_output=True, timeout=15)


def run(command, *, log: Path, limit=None, env=None, cwd=None, timeout=1800,
        samples: Path | None = None) -> dict:
  """Fail closed without system-manager access; never fall back to a user unit.

  A hard cap avoids MemoryHigh reclaim throttling. Sustained full-memory stalls
  stop the whole job after five seconds above 20%, before oomd's session policy.
  """
  limit = memory_budget(limit)
  prepare_slice()
  unit = 'boltbeam-job-' + uuid.uuid4().hex + '.service'
  env = dict(os.environ if env is None else env)
  argv = service_command(command, unit, limit, env, cwd or os.getcwd(), timeout)
  started = time.monotonic()
  peak = 0
  reason = None
  previous = None
  stalled = 0.0
  log.parent.mkdir(parents=True, exist_ok=True)
  sample_path = samples or log.with_suffix('.memory.jsonl')
  with log.open('w') as output, sample_path.open('w') as trace:
    process = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT)
    try:
      while process.poll() is None:
        result = subprocess.run(['systemctl', 'show', unit, '-p', 'ControlGroup', '--value'],
                                capture_output=True, text=True, timeout=5)
        group = result.stdout.strip()
        if group:
          if not group.startswith('/boltbeam.slice/'):
            raise RuntimeError('Job escaped the dedicated system slice: ' + group)
          directory = Path('/sys/fs/cgroup') / group.lstrip('/')
          try:
            current = int((directory / 'memory.current').read_text())
            peak = max(peak, int((directory / 'memory.peak').read_text()))
            memory_stat = dict(line.split() for line in (directory / 'memory.stat').read_text().splitlines())
            pressure = next(line for line in (directory / 'memory.pressure').read_text().splitlines()
                            if line.startswith('full '))
            total = int(dict(item.split('=') for item in pressure.split()[1:])['total'])
            now = time.monotonic()
            fraction = 0.0
            if previous:
              elapsed = now - previous[0]
              fraction = (total - previous[1]) / (elapsed * 1e6)
              stalled = stalled + elapsed if fraction >= 0.2 else 0.0
            previous = (now, total)
            trace.write(json.dumps(dict(elapsed_s=now-started, current_bytes=current,
                                        peak_bytes=peak, anon_bytes=int(memory_stat.get('anon', 0)),
                                        file_bytes=int(memory_stat.get('file', 0)),
                                        full_pressure_fraction=fraction)) + '\n')
            trace.flush()
            if stalled >= 5:
              reason = 'memory-pressure: full stalls exceeded 20% for five seconds'
              stop(unit)
              break
          except FileNotFoundError:
            pass  # A completed service may already have released its cgroup.
        time.sleep(0.1)
      rc = process.wait(timeout=15)
      final = subprocess.run(['systemctl', 'show', unit, '-p', 'MemoryPeak', '-p', 'Result',
                              '-p', 'ExecMainCode', '-p', 'ExecMainStatus'],
                             capture_output=True, text=True, timeout=5)
      properties = dict(line.split('=', 1) for line in final.stdout.splitlines() if '=' in line)
      final_peak = properties.get('MemoryPeak', '')
      if final_peak.isdigit() and int(final_peak) < 2**63:
        peak = max(peak, int(final_peak))
      if reason is None and rc:
        reason = properties.get('Result') or 'service failed; inspect log'
      # systemd can report SIGTERM as a successful service stop. It is not a
      # completed benchmark. Transient units may already be gone, so retain
      # the --wait termination record as a fallback to the live properties.
      output.flush()
      killed = re.search(r'^Main processes terminated with: code=(killed|dumped)/status=([^\n]+)',
                         log.read_text(), re.MULTILINE)
      if reason is None and (properties.get('ExecMainCode') in ('2', '3') or killed):
        reason = 'terminated by signal: ' + (killed.group(2) if killed else properties.get('ExecMainStatus', 'unknown'))
    finally:
      # Covers Ctrl-C, sampling failures, and unexpected systemd-run termination.
      stop(unit)
      if process.poll() is None:
        process.wait(timeout=15)
  status = dict(exit_code=rc if reason is None else (rc or 1), seconds=time.monotonic()-started,
                memory_limit_bytes=limit, memory_peak_bytes=peak, unit=unit, failure_reason=reason)
  log.with_suffix('.status.json').write_text(json.dumps(status, indent=2))
  return status
