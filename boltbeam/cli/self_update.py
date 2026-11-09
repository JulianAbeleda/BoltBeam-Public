"""`boltbeam update`: BoltBeam updating itself (not its engines). The one place that knows the installed version, the
newest release, how this copy was installed and the command that upgrades it. The screen asks the same facts at boot
(`python -m boltbeam.workflow.screen update`) and offers Upgrade now / Skip / Skip this version.

    installed   pyproject.toml beside this package (a checkout, or the Homebrew keg, which carries the tree), else the
                installed distribution's metadata (pip); unknown when neither says
    latest      the newest stable vX.Y.Z tag of the public mirror (GitHub tags API); a pre-release tag is no release
    cache       $BOLTBEAM_UPDATE_FILE, else ~/.boltbeam/update.json: the last good check, made again after the policy's
                update_check_hours. A failed check is "unknown" with its error, never "up to date", and runs again next time
    install     homebrew (this package under <prefix>/Cellar/boltbeam): brew update, then brew upgrade boltbeam;
                checkout (a git checkout): pull it yourself, nothing is run; pip (otherwise): pip install -U the tag's tarball
    skip        Skip this version is kept in the cache file; a newer release prompts again
    off         BOLTBEAM_NO_UPDATE_CHECK=1: the screen never checks (`boltbeam update` still does, being asked)
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from boltbeam.core import policy

MIRROR = "JulianAbeleda/BoltBeam-Public"
TAGS_URL = f"https://api.github.com/repos/{MIRROR}/tags?per_page=100"
TARBALL = f"https://github.com/{MIRROR}/archive/refs/tags/v{{}}.tar.gz"
FILE_ENV, OFF_ENV = "BOLTBEAM_UPDATE_FILE", "BOLTBEAM_NO_UPDATE_CHECK"
STABLE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")
ROOT = pathlib.Path(__file__).resolve().parents[2]  # the folder that holds the boltbeam package
DONE, FAILED = "updated:", "update failed:"  # the last line of `update --yes`; the screen reads it


def parse(v:str | None) -> tuple[int, ...] | None:
  m = STABLE.fullmatch(v or "")
  return tuple(map(int, m.groups())) if m else None


def installed(root:pathlib.Path | None = None) -> str | None:
  py = (root or ROOT) / "pyproject.toml"
  if py.is_file() and re.search(r'^name = "boltbeam"$', text := py.read_text(), re.M):
    m = re.search(r'^version = "([^"]+)"$', text, re.M)
    return m.group(1) if m else None
  from importlib import metadata
  try:
    return metadata.version("boltbeam")
  except metadata.PackageNotFoundError:
    return None


def fetch_tags() -> list[dict[str, Any]]:
  import urllib.request
  req = urllib.request.Request(TAGS_URL, headers={"Accept": "application/vnd.github+json", "User-Agent": "boltbeam"})
  with urllib.request.urlopen(req, timeout=policy.value("update_check_timeout_s")) as r:
    return json.loads(r.read())


def latest_of(tags:list[dict[str, Any]]) -> str:
  stable = [t["name"][1:] for t in tags if t["name"].startswith("v") and parse(t["name"][1:])]
  if not stable:
    raise ValueError(f"no vX.Y.Z tag on {MIRROR}")
  return max(stable, key=parse)


def how(latest:str | None, root:pathlib.Path | None = None) -> dict[str, Any]:
  """How this copy was installed and the commands that upgrade it (None: BoltBeam runs nothing, `words` says what to do)."""
  root = root or ROOT
  parts = root.parts
  for i in range(len(parts) - 1):
    if parts[i] == "Cellar" and parts[i + 1] == "boltbeam":
      brew = str(pathlib.Path(*parts[:i]) / "bin" / "brew")
      return {"install": "homebrew", "command": [[brew, "update"], [brew, "upgrade", "boltbeam"]],
              "words": "brew update && brew upgrade boltbeam"}
  if (root / ".git").exists():
    return {"install": "checkout", "command": None,
            "words": f"a git checkout: pull it yourself (git -C {shlex.quote(str(root))} pull); BoltBeam runs no git for you"}
  if latest is None:
    return {"install": "pip", "command": None, "words": "pip install -U from the mirror, once the newest release is known"}
  argv = [sys.executable, "-m", "pip", "install", "-U", TARBALL.format(latest)]
  return {"install": "pip", "command": [argv], "words": shlex.join(argv)}


def state_file(env:Mapping[str, str] = os.environ) -> pathlib.Path:
  return pathlib.Path(env.get(FILE_ENV) or pathlib.Path.home() / ".boltbeam" / "update.json").expanduser()


def _load(path:pathlib.Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    return {}


def _save(path:pathlib.Path, state:dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def check(*, force:bool = False, now:float | None = None, fetch:Callable[[], list] = fetch_tags,
          env:Mapping[str, str] = os.environ) -> dict[str, Any]:
  """The cache, checked upstream when it is older than the policy's hours (or force). A failed check keeps the last
  good one's time, so the next call tries again, and is reported as this call's error."""
  path, now = state_file(env), time.time() if now is None else now
  state = _load(path)
  state.pop("error", None)
  if force or now - float(state.get("checked_at") or 0) >= policy.value("update_check_hours") * 3600:
    try:
      state.update(latest=latest_of(fetch()), checked_at=now)
    except (OSError, ValueError, KeyError, TypeError) as exc:
      state["error"] = f"could not check {MIRROR}: {exc}"
    _save(path, {k: v for k, v in state.items() if k != "error"})
  return state


def payload(state:dict[str, Any], *, have:str | None = None, root:pathlib.Path | None = None, off:bool = False) -> dict[str, Any]:
  """One answer for the screen and the CLI: newer, skipped, current, ahead, unknown or off; prompt only when newer."""
  have = installed(root) if have is None else have
  latest = None if off or state.get("error") else state.get("latest")
  h, l, skipped = parse(have), parse(latest), parse(state.get("skipped"))
  status = ("off" if off else "unknown" if not (h and l) else "current" if l == h else "ahead" if l < h
            else "skipped" if skipped and l <= skipped else "newer")
  return {"kind": "update", "status": status, "prompt": status == "newer",
          "installed": have, "latest": latest, "skipped": state.get("skipped"), "checked_at": state.get("checked_at"),
          "error": None if off else state.get("error"), **how(latest, root)}


def screen(skip_version:str | None = None, env:Mapping[str, str] = os.environ, **kw) -> dict[str, Any]:
  """The seam's `update`: the payload at boot (no network when switched off), after recording a skip when given."""
  if skip_version:  # Skip this version: kept until a newer release
    _save(path := state_file(env), {**_load(path), "skipped": skip_version})
  off = env.get(OFF_ENV, "") not in ("", "0")
  return payload({} if off else check(env=env, **kw), off=off)


def words(p:dict[str, Any]) -> str:
  if p["status"] == "unknown":
    return p["error"] or f"the installed version is unknown ({ROOT}): nothing to compare"
  if p["status"] in ("newer", "skipped"):
    return f"BoltBeam {p['latest']} is available (you have {p['installed']}). Upgrade ({p['install']}): {p['words']}"
  return f"BoltBeam {p['installed']} is " + ("the newest release." if p["status"] == "current" else
                                             f"ahead of the newest release ({p['latest']}).")


def cmd_update(args, out=sys.stdout, run:Callable[..., Any] = subprocess.run, **kw) -> int:
  p = payload(check(force=True, **kw))
  print(words(p), file=out, flush=True)
  if p["status"] == "unknown":
    return 1
  if not args.yes or p["status"] not in ("newer", "skipped"):
    return 0
  if not p["command"]:
    print(f"{FAILED} nothing run: {p['words']}", file=out, flush=True)
    return 2
  for argv in p["command"]:
    print("$ " + shlex.join(argv), file=out, flush=True)
    code = run(argv).returncode
    if code:
      print(f"{FAILED} {shlex.join(argv)} exited {code}", file=out, flush=True)
      return 1
  print(f"{DONE} BoltBeam {p['latest']} is installed; quit and run boltbeam again to use it", file=out, flush=True)
  return 0


def register(sub) -> None:
  p = sub.add_parser("update", help="is a newer BoltBeam released: check the public mirror now and print how to upgrade; "
                                   "--yes runs the upgrade (Homebrew or pip; a checkout is pulled by you)")
  p.add_argument("--yes", action="store_true", help="run the upgrade command for this install")
  p.set_defaults(fn=cmd_update)
