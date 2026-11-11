"""Engine families: a root engine and its forks (data/engine_families.json, the one table). Setup's Engine picker shows
one row per family and moves between its forks; `boltbeam doctor` shows each family with the forks found; a run
records the family and fork it ran on (fork_of).

    fork      a build of the family: upstream (the root) or a fork, with its repo and whether BoltBeam runs it yet
    provider  the adapter that runs a fork (providers.ENGINES); a fork with one is found by the engine scan
    found     a fork with no provider: its env var, then its paths (globs, ~ the home folder), then its binary on PATH
"""
from __future__ import annotations

import functools
import glob
import json
import os
import pathlib
import shutil
from collections.abc import Iterable, Mapping
from typing import Any

TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "engine_families.json"


@functools.lru_cache(maxsize=1)
def families() -> tuple[dict[str, Any], ...]:
  return tuple(json.loads(TABLE.read_text())["families"])


def fork_of(provider:str) -> dict[str, str | None]:
  """The family and fork an adapter runs, as a run records them (engine_family, fork); None for one not listed."""
  for fam in families():
    for f in fam["forks"]:
      if f.get("provider") == provider:
        return {"engine_family": fam["family"], "fork": f["fork"]}
  return {"engine_family": None, "fork": None}


def fork_for(formats:Iterable[str]) -> dict[str, str] | None:
  """The fork that reads a weight format no root reads (its formats), as a screen names it; None when none does."""
  want = set(formats)
  for fam in families():
    for f in fam["forks"]:
      if want & set(f.get("formats") or ()):
        return {"engine_family": fam["family"], "family_label": fam["label"], "fork": f["fork"], "label": f["label"]}
  return None


def repo(family:str, fork:str) -> str:
  return next(f["repo"] for fam in families() if fam["family"] == family for f in fam["forks"] if f["fork"] == fork)


def find(fork:Mapping[str, Any], env:Mapping[str, str] | None = None, home:pathlib.Path | None = None) -> str | None:
  """Where a fork with no provider is on this machine, or None: its env var, its paths, then its binary on PATH."""
  env = os.environ if env is None else env
  if fork.get("env") and env.get(fork["env"]):
    return env[fork["env"]]
  home = home or pathlib.Path.home()
  for pattern in fork.get("paths") or ():
    hits = sorted(glob.glob(str(home / pattern[2:]) if pattern.startswith("~/") else pattern))
    if hits:
      return hits[0]
  return shutil.which(fork["binary"], path=env.get("PATH")) if fork.get("binary") else None


def build_of(binary:str, family:str, env:Mapping[str, str] | None = None, home:pathlib.Path | None = None) -> dict[str, Any] | None:
  """The fork row whose build this program is: a fork with no provider whose found program sits in the same folder
  (llama-bench and llama-server are built side by side), else the family's row with a provider (the root BoltBeam
  runs); None when the family has neither."""
  here = pathlib.Path(binary).expanduser().resolve().parent
  rows = next((fam["forks"] for fam in families() if fam["family"] == family), ())
  for f in rows:
    where = None if f.get("provider") else find(f, env, home)
    if where and pathlib.Path(where).expanduser().resolve().parent == here:
      return f
  return next((f for f in rows if f.get("provider")), None)


def tool_flags(binary:str, tool:str, family:str = "llama.cpp", env:Mapping[str, str] | None = None,
               home:pathlib.Path | None = None) -> tuple[dict[str, Any] | None, tuple[str, ...] | None]:
  """The fork row of this program's build (build_of) and the flags its `tool` has (the row's flags); None for the
  flags when the row lists none, so nothing is checked."""
  row = build_of(binary, family, env, home)
  flags = ((row or {}).get("flags") or {}).get(tool)
  return row, tuple(flags) if flags is not None else None


def _fix(fork:Mapping[str, Any]) -> str:
  """The one thing to do for a fork not here: get it from its repo, and name it by its env var (the engine scan's,
  for a fork with a provider)."""
  from boltbeam.collectors import engine_scan
  env = engine_scan.PRIMARY.get(fork.get("provider") or "") or fork.get("env")
  how = f"export {env}=<its path>" if env else "put it on PATH"
  branch = f" (branch {fork['branch']})" if fork.get("branch") else ""
  return f"get it from {fork['repo']}{branch}, then {how}"


def rows(provider_rows:list[dict[str, Any]], env:Mapping[str, str] | None = None,
         home:pathlib.Path | None = None) -> list[dict[str, Any]]:
  """Every family with its forks as this machine has them. provider_rows are providers.available's rows: a fork with
  a provider is found and runs here as its row says; one without is found by find, and runs only if the table says.
    state   available (picked and run here), found (here, BoltBeam cannot run it yet), not_here, not_compatible"""
  by = {r["provider"]: r for r in provider_rows}
  out = []
  for fam in families():
    forks = []
    for f in fam["forks"]:
      row = by.get(f.get("provider") or "")
      if row is not None:
        state = "available" if row.get("available") else row.get("state") or "not_here"
        why, path = row.get("reason"), None
      elif f.get("provider"):  # an adapter that did not answer for this machine
        state, why, path = "not_here", None, None
      else:
        path = find(f, env, home)
        state = "found" if path else "not_here"
        why = f.get("why") if path else None
      if state == "available" and not f["runs"]:
        state, why = "found", f.get("why")
      forks.append({"fork": f["fork"], "label": f["label"], "repo": f["repo"], "provider": f.get("provider"),
                    "runs": f["runs"], "state": state, "path": path, "reason": why,
                    "fix": _fix(f) if state == "not_here" else None})
    out.append({"family": fam["family"], "label": fam["label"], "forks": forks})
  return out
