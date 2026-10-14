#!/usr/bin/env python3
"""Run a benchmark in BoltBeam's isolated system slice; requires sudo -n."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'boltbeam' / 'runtime'))
from bounded_job import run

if __name__ == '__main__':
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--log', required=True, type=Path)
  ap.add_argument('--memory-limit-bytes', type=int)
  ap.add_argument('--timeout', type=int, default=7200)
  ap.add_argument('command', nargs=argparse.REMAINDER)
  args = ap.parse_args()
  command = args.command[1:] if args.command[:1] == ['--'] else args.command
  if not command:
    ap.error('provide a command after --')
  result = run(command, log=args.log, limit=args.memory_limit_bytes, timeout=args.timeout)
  print(result)
  sys.exit(result['exit_code'])
