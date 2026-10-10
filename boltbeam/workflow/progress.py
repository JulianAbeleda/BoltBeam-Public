"""How far the pipeline is. A long stage reports `done of total` here; the pipeline prints it as
`stage KEY: progress P/Q`, and a screen reads that line. Each stage's wall time is kept in the run
(stage_times.json) and, per chip and engine, on this machine, so the next run's bar weights stages by time."""
from __future__ import annotations

import json
import pathlib
from typing import Callable

from boltbeam.target.targets import chips_dir

STAGE_TIMES = "stage_times.json"
SCHEMA = "boltbeam.stage_times.v1"

_report: Callable[[int, int], None] = lambda done, total: None  # noqa: E731


def report(done:int, total:int) -> None:
  """Called by a long stage: `done` of `total` parts finished."""
  _report(done, total)


def listen(fn:Callable[[int, int], None]) -> None:
  """The pipeline points report() at the current stage."""
  global _report
  _report = fn


def step_ids(keys:list[str]) -> list[str]:
  """A key that repeats (analyze, output) gets #2, #3, so each step has its own time."""
  seen:dict[str, int] = {}
  out = []
  for key in keys:
    seen[key] = seen.get(key, 0) + 1
    out.append(key if seen[key] == 1 else f"{key}#{seen[key]}")
  return out


def history_path() -> pathlib.Path:
  """Next to this machine's chip profiles: ~/.boltbeam/stage_times.json."""
  return chips_dir().parent / STAGE_TIMES


def _history() -> dict:
  try:
    return json.loads(history_path().read_text())
  except (OSError, ValueError):
    return {}


def expected(where:str, ids:list[str]) -> list[float] | None:
  """The last run's seconds per step for this chip and engine, or None on a first run. A step the last run did
  not have takes the median of the ones it did."""
  last = _history().get(where) or {}
  known = sorted(float(v) for k, v in last.items() if k in ids)
  if not known:
    return None
  mid = known[len(known) // 2]
  return [max(float(last.get(i, mid)), 0.1) for i in ids]


def save(run:pathlib.Path | None, where:str, times:dict[str, float], *, finished:bool) -> None:
  """The run keeps its own times; this machine's history keeps the last finished run's."""
  if run is not None and run.is_dir():
    (run / STAGE_TIMES).write_text(json.dumps({"schema": SCHEMA, "where": where, "seconds": times}, indent=2) + "\n")
  if finished:
    path = history_path()
    all_ = _history()
    all_[where] = times
    try:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(json.dumps(all_, indent=2, sort_keys=True) + "\n")
    except OSError:
      pass
