#!/usr/bin/env python3
"""Rescore saved quality responses without regenerating or modifying model outputs.

Example: --suite suite.json --log awq=awq.log --log nvfp4=nvfp4.log --output comparison.json
Scores assess answer content, not conformance to the requested JSON-only format.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'boltbeam/runtime'))
from quality_checks import grade_response


def source(path):
  return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def regrade(suite, log):
  records = []
  for line in log.read_text().splitlines():
    try:
      row = json.loads(line)
    except ValueError:
      continue
    if isinstance(row, dict) and 'quality' in row:
      records.append(row)
  if len(records) != 1:
    raise ValueError(f'{log}: expected exactly one quality record, found {len(records)}')
  original = records[0]
  responses = original['quality']
  if [r['prompt'] for r in responses] != [c['prompt'] for c in suite]:
    raise ValueError(f'{log}: prompts/order differ from the suite')
  if any(r['expected'] != c['answer'] for r, c in zip(responses, suite)):
    raise ValueError(f'{log}: canonical answers changed')
  checks = [grade_response(case, response['text'], response.get('token_ids'),
                           thinking=response.get('thinking', original.get('thinking', False)),
                           finish_reason=response.get('finish_reason'))
            for case, response in zip(suite, responses)]
  return {'source': source(log), 'original_passed': original['passed'],
          'strict_passed': sum(c['strict_passed'] for c in checks),
          'passed': sum(c['passed'] for c in checks), 'total': len(checks), 'quality': checks}


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--suite', required=True, type=Path)
  ap.add_argument('--log', required=True, action='append', metavar='LABEL=PATH')
  ap.add_argument('--output', required=True, type=Path)
  args = ap.parse_args()
  suite = json.loads(args.suite.read_text())
  runs = {}
  for entry in args.log:
    label, path = entry.split('=', 1)
    if label in runs:
      ap.error('duplicate label: ' + label)
    runs[label] = regrade(suite, Path(path))
  report = {'method': 'rescore_saved_outputs', 'suite': source(args.suite), 'runs': runs,
            'remaining_failures': {label: [c['prompt'] for c in run['quality'] if not c['passed']]
                                   for label, run in runs.items()}}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps({label: {k: run[k] for k in ('original_passed', 'strict_passed', 'passed', 'total')}
                    for label, run in runs.items()}, indent=2))


if __name__ == '__main__':
  main()
