"""Which build of each engine runs on this machine, and whether it was upstream's newest: the "current" build.

Two ideas, kept apart. The pin (data/engine_pins.json) is the audited reference and moves only by a justified
engine-upgrade audit (docs/engine-upgrade-audit.md). The current build is per machine: what runs use by default. It
lives in ~/.boltbeam/builds.json (beside the engine scan, engine_scan.engines_file), never in the repo.

    engine row     current {version, commit, folder, paths}   the build runs use
                   pending                                    a newer build installed beside it, not yet compared
                   previous                                   the build current replaced; kept
                   older [...]                                builds before previous; BoltBeam's own are deleted
                                                              after the next comparison that succeeds (deleted [...])
                   checked {latest, commit, checked_at, error} the last look at upstream (data/engine_releases.json)
                   not_installed {version, reason, at}        a release this machine did not install, and why
                   comparisons [...]                          each one-time comparison that moved current

A build BoltBeam found already on the machine has folder None: it is the user's, recorded and never deleted. Only
releases count (no rc, dev, alpha, beta, local or GitHub-prerelease builds). Only `screen engines --update`
(workflow/engine_update.py) installs, sets pending and deletes. The pipeline reads current; on its first --analyze run
after a new build it runs the previous build once, compares the two (workflow/diff) and calls compared(). An env var
the user set always wins over current (as over PATH and the engine scan); no comparison is made with it.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import pathlib
import re
from typing import Any, Callable, Iterator

from boltbeam.vocab import SCHEMA_ENGINE_BUILDS as SCHEMA

TABLE = pathlib.Path(__file__).resolve().parent.parent / "data" / "engine_releases.json"
UPGRADE_FILE = "engine_upgrade.json"  # the one-time comparison, saved in the new build's run
MARK = ".boltbeam-build"  # written into every folder BoltBeam installs; nothing without it is ever deleted
_PRE = {"dev": 0, "a": 1, "b": 2, "rc": 3}


def rows() -> dict[str, dict[str, Any]]:
  return json.loads(TABLE.read_text())["engines"]


def row(engine:str) -> dict[str, Any]:
  got = rows().get(engine)
  if got is None:
    raise ValueError(f"{engine} has no upstream release source in {TABLE.name}; one of {', '.join(rows())}")
  return got


# --- versions ------------------------------------------------------------------------------------------------------

def version_key(version:str | None) -> tuple | None:
  """An ordering key for a release version, or None for one this does not read (a dev build's "0.6.0-dev", a local
  "+cu130" build): 0.33.0, v0.9.0, 1.3.0rc29, 0.9.0.1, 0.8.5.post1; a pre-release sorts below its release."""
  m = re.fullmatch(r"v?(\d+(?:\.\d+)*)(?:\.?(dev|a|b|rc)(\d+))?(?:\.post(\d+))?", str(version or "").strip())
  if not m:
    return None
  nums = tuple(int(x) for x in m.group(1).split("."))
  pre = (_PRE[m.group(2)], int(m.group(3))) if m.group(2) else (9, 0)
  return (*nums, *(0,) * (6 - len(nums)), *pre, int(m.group(4) or 0))


def stable(version:str) -> bool:
  """A release: no rc, dev, alpha or beta, no local build. A .postN fix of a release is a release (PEP 440)."""
  key = version_key(version)
  return key is not None and key[6] == 9


def candidates(r:dict[str, Any], payload:Any) -> list[str]:
  """The versions an upstream answer names, before the stable filter. GitHub: no draft or prerelease (llama.cpp's
  daily bNNNNN builds are prereleases). PyPI: each release with a file not yanked."""
  if r["source"] == "github":
    out = []
    for rel in payload if isinstance(payload, list) else [payload]:
      m = None if rel.get("draft") or rel.get("prerelease") else re.fullmatch(r["tag"], str(rel.get("tag_name") or ""))
      out += [m.group(1)] if m else []
    return out
  if r["source"] == "pypi":
    return [v for v, files in (payload.get("releases") or {}).items() if any(not f.get("yanked") for f in files)]
  if r["source"] == "simple":
    return re.findall(rf">{re.escape(r['dist'])}-([^-<]+?)(?:-[^<]*\.whl|\.tar\.gz)<", str(payload))
  raise ValueError(f"unknown release source {r['source']!r} in {TABLE.name}")


def newest(r:dict[str, Any], payload:Any) -> str | None:
  """The newest release upstream names; None when it names none."""
  got = [v for v in candidates(r, payload) if stable(v)]
  return max(got, key=version_key) if got else None


def newer(a:str | None, b:str | None) -> bool | None:
  """Is a newer than b; None when either is not a release version (unknown is not an answer)."""
  ka, kb = version_key(a), version_key(b)
  return None if ka is None or kb is None else ka > kb


def satisfies(version:str, spec:str) -> bool | None:
  """Does a version meet a requirement such as ">=3.10,<4" or "<=2.10.0a0,>=2.9.1"; None when it cannot be read."""
  import operator
  ops = {">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt, "~=": operator.ge}
  have = version_key(version.split("+")[0])
  for clause in filter(None, (c.strip() for c in spec.split(","))):
    m = re.fullmatch(r"(>=|<=|==|!=|~=|>|<)\s*([\w.]+)", clause)
    want = version_key(m.group(2)) if m else None
    if have is None or want is None:
      return None
    if not ops[m.group(1)](have, want):
      return False
  return True


def requirement(requires_dist:list[str] | None, name:str) -> str | None:
  """The version constraint a package's metadata puts on one dependency ("torch<=2.10.0a0,>=2.9.1" -> the part after
  the name); None when it names none."""
  for req in requires_dist or []:
    if m := re.match(rf"{re.escape(name)}(?![\w.-])\s*(?:\[[^\]]*\])?\s*\(?([<>=!~][^;)]*)", req.strip()):
      return m.group(1).strip()
  return None


def wheels(index:str, dist:str, version:str) -> list[tuple[str, str]]:
  """(python tag, platform) of each wheel a PEP 503 index page lists for that version."""
  return re.findall(rf">{re.escape(dist)}-{re.escape(version)}-(cp\d+)-[^-<]+-([^<]+?)\.whl<", index)


def check(engine:str, read:Callable[[str], Any] | None = None) -> dict[str, Any]:
  """Upstream's newest release, with the source, time and (GitHub) its commit; a failed read is "could not check"."""
  from boltbeam.collectors import providers
  r = row(engine)
  out = {"engine": engine, "source": r["url"], "checked_at": now(), "latest": None, "commit": None, "error": None}
  try:
    payload = (read or providers.upstream)(r["url"])
    out["latest"] = newest(r, payload)
    sha = str(payload.get("target_commitish") or "") if isinstance(payload, dict) else ""
    out["commit"] = sha if re.fullmatch(r"[0-9a-f]{40}", sha) else None
    if out["latest"] is None:
      out["error"] = "upstream named no stable release"
  except providers.upstream_errors() as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
  return out


def fresh(checked:dict[str, Any] | None) -> bool:
  """A check younger than the policy's engine_latest_max_age_h that read a release: upstream is not asked again."""
  from boltbeam.core.policy import value
  at = (checked or {}).get("checked_at")
  if not at or not checked.get("latest") or checked.get("error"):
    return False
  age = _dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(at.replace("Z", "+00:00"))
  return age.total_seconds() / 3600 <= value("engine_latest_max_age_h")


def now() -> str:
  return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the machine's state --------------------------------------------------------------------------------------------

def state_file() -> pathlib.Path:
  from boltbeam.collectors import engine_scan
  return engine_scan.engines_file().with_name("builds.json")


def load() -> dict[str, Any]:
  try:
    data = json.loads(state_file().read_text())
  except (OSError, ValueError):
    data = {}
  return data if isinstance(data, dict) and data.get("schema") == SCHEMA else {"schema": SCHEMA, "engines": {}}


def save(state:dict[str, Any]) -> pathlib.Path:
  path = state_file()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
  return path


def entry(state:dict[str, Any], engine:str) -> dict[str, Any]:
  return state.setdefault("engines", {}).setdefault(engine, {})


def installed(state:dict[str, Any], engine:str, build:dict[str, Any]) -> str:
  """A build is on the machine: current when there is none to compare it with, else pending (a pending build it
  replaces, never compared, joins older). Returns which."""
  e = entry(state, engine)
  e.pop("not_installed", None)
  if not e.get("current"):
    e["current"] = build
    return "current"
  if newer(build.get("version"), e["current"].get("version")) is False:
    raise ValueError(f"{engine} {build.get('version')} is not newer than the current {e['current'].get('version')}")
  if e.get("pending"):
    e.setdefault("older", []).append(e["pending"])
  e["pending"] = build
  return "pending"


def compared(state:dict[str, Any], engine:str, record:dict[str, Any]) -> dict[str, Any]:
  """The one-time comparison is saved: pending becomes current, current becomes previous, previous joins older."""
  e = entry(state, engine)
  if not e.get("pending"):
    raise ValueError(f"{engine} has no pending build to make current")
  e.setdefault("comparisons", []).append({**record, "before": e["current"].get("version"),
                                          "after": e["pending"]["version"], "at": now()})
  if e.get("previous"):
    e.setdefault("older", []).append(e["previous"])
  e["previous"], e["current"], e["pending"] = e["current"], e["pending"], None
  return e


def deletable(build:dict[str, Any], root:str | None) -> str | None:
  """Why this older build must stay, or None when BoltBeam made it and may delete it: a folder it recorded, inside
  its builds root, holding its mark. A build found on the machine (folder None) is the user's."""
  folder = build.get("folder")
  if not folder:
    return "found on this machine, not installed by BoltBeam"
  p, base = pathlib.Path(folder).resolve(), pathlib.Path(root or "/nonexistent").resolve()
  if base not in p.parents or not (p / MARK).is_file():
    return f"{folder} is not a folder BoltBeam installed under {base}"
  return None


def env_for(build:dict[str, Any] | None, env:dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
  """The env vars that select this build, and those the user already set (theirs win, as over PATH)."""
  env = os.environ if env is None else env
  paths = (build or {}).get("paths") or {}
  return ({k: v for k, v in paths.items() if not env.get(k)}, sorted(k for k in paths if env.get(k)))


@contextlib.contextmanager
def using(build:dict[str, Any] | None) -> Iterator[list[str]]:
  """Run with this build selected, in this process and the engines it starts; yields the user's overriding vars."""
  sets, overridden = env_for(build)
  try:
    os.environ.update(sets)
    yield overridden
  finally:
    for k in sets:
      os.environ.pop(k, None)


# --- what a run says ------------------------------------------------------------------------------------------------

def _same(a:str | None, b:str | None) -> bool:
  return bool(a and b) and (a.startswith(b) or b.startswith(a))


def run_version(provider:str, run:pathlib.Path, state:dict[str, Any] | None = None) -> str | None:
  """The release a run measured: the recorded build (llama.cpp's commit, vLLM's version) matched to a build this
  machine recorded or to upstream's latest release; else the recorded build as it is."""
  from boltbeam.collectors import providers
  got = providers.run_build(run)
  e = entry(state if state is not None else load(), provider)
  known = [b for b in (e.get("current"), e.get("pending"), e.get("previous"), *(e.get("older") or [])) if b]
  checked = e.get("checked") or {}
  known.append({"version": checked.get("latest"), "commit": checked.get("commit")})
  return next((b["version"] for b in known if b.get("version") and (_same(got, b.get("commit")) or got == b["version"])), got)


def currency(provider:str, version:str | None, state:dict[str, Any] | None = None) -> dict[str, Any] | None:
  """Was this build the newest upstream release when the run measured it: from the last check (no network here). A
  check older than the policy's engine_latest_max_age_h, a failed one, or none: not checked."""
  if provider not in rows():
    return None
  checked = entry(state if state is not None else load(), provider).get("checked") or {}
  latest, at = checked.get("latest"), checked.get("checked_at")
  ran = f"build {version}" if version else "build not recorded"
  if not fresh(checked):
    why = (f"the last check failed: {checked['error']}" if checked.get("error") else
           f"last checked {at}" if at else "never checked; `screen engines --update` checks")
    return {"latest": latest, "checked_at": at, "is_latest": None, "words": f"{ran}; latest not checked ({why})"}
  order = newer(latest, version)  # a build that is not a release (a commit, a dev build) is not the latest release
  is_latest = None if version is None else version == latest or order is False
  words = f"{ran} (latest at run time)" if is_latest else f"{ran}; latest was {latest}" if is_latest is False else ran
  return {"latest": latest, "checked_at": at, "is_latest": is_latest, "words": f"{words}, checked {at}"}


def standing(engine:str, state:dict[str, Any] | None = None, found:dict[str, Any] | None = None) -> dict[str, Any] | None:
  """Where this machine's build of an engine stands against upstream's newest release, from the last check (no
  network): behind (a newer release is out; `engines --update` installs it), waiting (one is installed and waits for
  its comparison), up_to_date, or unknown (never checked, the check failed, or an equal answer older than
  engine_latest_max_age_h). found: the build here when none is recorded. The update and the TUI's Engine read this."""
  if engine not in rows():
    return None
  e = entry(state if state is not None else load(), engine)
  checked, pend = e.get("checked") or {}, (e.get("pending") or {}).get("version")
  cur, latest, at = (e.get("current") or found or {}).get("version"), checked.get("latest"), checked.get("checked_at")
  out = {"engine": engine, "current": cur, "pending": pend, "latest": latest, "checked_at": at}
  if checked.get("error") or not latest:
    why = f"the last check failed: {checked['error']}" if checked.get("error") else "never checked"
    return {**out, "state": "unknown", "words": f"latest release unknown ({why})"}
  if pend and newer(latest, pend) is False:
    return {**out, "state": "waiting", "words": f"{pend} installed; the next analyzed Run compares it once with {cur}"}
  if cur and (cur == latest or newer(latest, cur) is False):
    return ({**out, "state": "up_to_date", "words": f"{cur} is the latest release"} if fresh(checked) else
            {**out, "state": "unknown", "words": f"latest release unknown (last checked {at})"})
  return {**out, "state": "behind", "words": f"{cur or 'build unknown'} · {latest} is out"}


def comparison(run:pathlib.Path) -> dict[str, Any] | None:
  """The one-time comparison saved with this run, as one line; None when the run made none."""
  p = run / UPGRADE_FILE
  if not p.is_file():
    return None
  d = json.loads(p.read_text())
  how = f"refused: {d['refused']}" if d.get("refused") else f"see {UPGRADE_FILE}"
  return {"file": UPGRADE_FILE, "before_run": d.get("before_run"),
          "words": f"compared once with the previous build {d.get('before')} (run {d.get('before_run')}); {how}"}


def build_lines(engine:dict[str, Any]) -> list[str]:
  """The run's build beside upstream's newest, and the comparison it made: the words the summary, the report and the
  TUI's Run page show (results.results engine.build_lines)."""
  latest, cmp = (engine.get("build") or {}).get("latest") or {}, engine.get("comparison") or {}
  return ([f"Upstream: {latest['words']}."] if latest.get("words") else []) + (
    [f"Upgrade comparison: {cmp['words']}."] if cmp.get("words") else [])
