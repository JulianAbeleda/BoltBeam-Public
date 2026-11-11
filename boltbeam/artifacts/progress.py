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

# A first run has no history: these are the expected seconds per step, from runs on the M3 (2026-10-09), so the bar
# is weighted by time from the start and the percent is marked an estimate. A key may hold one figure or one per
# engine. Anything not listed is quick: DEFAULT_SECONDS.
DEFAULTS:dict[str, float | dict[str, float]] = {
  "measure_timing": {"tinygrad": 130.0, "llama.cpp": 22.0},
  "role_time": {"tinygrad": 40.0, "llama.cpp": 25.0},
  "search": {"tinygrad": 350.0, "llama.cpp": 1.0},  # the kernel search exists for tinygrad on Metal; others skip it
}
DEFAULT_SECONDS = 1.0

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


def engine_of(where:str) -> str:
  """The engine in a history key "chip|engine|batch N"."""
  parts = where.split("|")
  return parts[1] if len(parts) > 1 else ""


def default_seconds(step_id:str, engine:str) -> float:
  row = DEFAULTS.get(step_id.split("#")[0], DEFAULT_SECONDS)
  if isinstance(row, dict):
    return float(row.get(engine, min(row.values())))
  return float(row)


def expected(where:str, ids:list[str]) -> tuple[list[float], str]:
  """Seconds per step for the bar, and where they come from: "history" (the last finished run for this chip and
  engine; a step that run did not have takes the median of the ones it did) or "default" (a first run: the
  DEFAULTS table, so a long stage is never drawn as 1/N of the bar)."""
  last = _history().get(where) or {}
  known = sorted(float(v) for k, v in last.items() if k in ids)
  if not known:
    return [default_seconds(i, engine_of(where)) for i in ids], "default"
  mid = known[len(known) // 2]
  return [max(float(last.get(i, mid)), 0.1) for i in ids], "history"


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
