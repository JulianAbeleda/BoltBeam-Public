"""The engine arguments a run used, written into the run's engine_settings (workload_profile.json, and trace_request.json when it
exists), where the report finds whether the experts were offloaded (workflow/offload.recorded). An engine that is not told
what it used is "placement not recorded" in the report, never all on the GPU."""
from __future__ import annotations

import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from boltbeam.artifacts.run_io import read_json, write_json

TRACE_ENV = "LLAMA_MOE_CACHE_TRACE"  # llama.cpp's routing trace, for a simulator of the cache; recorded when set, never read as a rate
DECODE_STATS = re.compile(r"llama_moe_cache: ubatch <= 8: hits = (\d+), misses = (\d+)")  # the decode batches' line of the stats the cache logs at exit (-lv 4)


def moe_cache_hit_rate(log:str) -> float | None:
  """The expert cache's decode hit rate from llama.cpp's own exit statistics (hits over hits + misses, summed over its lines);
  None when the log has none (no cache, or the log level hides them)."""
  rows = [(int(h), int(m)) for h, m in DECODE_STATS.findall(log)]
  hits = sum(h for h, _ in rows)
  return hits / (hits + sum(m for _, m in rows)) if hits + sum(m for _, m in rows) else None


def record(run:pathlib.Path | str, args:Sequence[str], *, engine:str, log:str | None = None, env:Mapping[str, str] | None = None) -> dict[str, Any]:
  """Merge the arguments the engine ran with (without its executable), the engine's name, the trace path if the environment set one
  and the cache's hit rate from `log` into the run's engine_settings. Returns what was written."""
  run = pathlib.Path(run)
  got: dict[str, Any] = {"engine": engine, "engine_args": [str(a) for a in args]}
  if env and env.get(TRACE_ENV):
    got["moe_cache_trace"] = env[TRACE_ENV]
  if (rate := moe_cache_hit_rate(log or "")) is not None:
    got["moe_cache_hit_rate"] = rate
  for name in ("workload_profile.json", "trace_request.json"):
    if (run / name).is_file():
      doc = read_json(run / name)
      doc["engine_settings"] = {**(doc.get("engine_settings") or {}), **got}
      write_json(run / name, doc)
  return got
