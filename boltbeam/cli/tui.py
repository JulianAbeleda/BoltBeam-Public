"""`boltbeam tui`: the screen (tui/, a Go program), built when it is missing or stale, then run.

    boltbeam tui                    the screens (needs a terminal)
    boltbeam tui --json chips       any boltbeam-tui flag or command passes through

This module is the one place that knows where the checkout with the Go source is, which Go this machine has and
which one tui/go.mod asks for, and where the built `boltbeam-tui` lives. `boltbeam doctor` reads the same facts from
here, so the two never disagree.

    checkout    $BOLTBEAM_REPO, else walk up from the working folder, else the folder this package was imported
                from (an editable install). A checkout holds boltbeam/__init__.py and tui/go.mod.
    binary      $BOLTBEAM_TUI_DIR/boltbeam-tui, else ~/.cache/boltbeam/boltbeam-tui. The build writes the hash of
                the Go sources beside it (boltbeam-tui.sources); the binary is stale when the hash differs. A
                boltbeam-tui next to the Go source or on PATH counts as built for doctor, unchecked (no sidecar).
    Go          found on PATH; it must be at least go.mod's version. Too old or missing: the install line, exit 2.
                Nothing is downloaded: the build runs with GOTOOLCHAIN=local.
    run         exec boltbeam-tui --repo <checkout> --python <this interpreter> <the flags given>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping

BINARY = "boltbeam-tui"
SIDECAR = BINARY + ".sources"
DIR_ENV = "BOLTBEAM_TUI_DIR"
REPO_ENV = "BOLTBEAM_REPO"
GO_SITE = "https://go.dev/dl/"
SOURCES = ("go.mod", "go.sum")  # beside every *.go file under tui/


class BuildError(RuntimeError):
  pass


def is_checkout(path:pathlib.Path) -> bool:
  return (path / "tui" / "go.mod").is_file() and (path / "boltbeam" / "__init__.py").is_file()


def checkout(start:pathlib.Path | None = None, env:Mapping[str, str] | None = None) -> pathlib.Path | None:
  """The BoltBeam checkout that holds the screen's Go source, or None when none is around."""
  env = os.environ if env is None else env
  if env.get(REPO_ENV):
    p = pathlib.Path(env[REPO_ENV]).expanduser()
    return p if is_checkout(p) else None
  here = (start or pathlib.Path.cwd()).resolve()
  for d in (here, *here.parents):
    if is_checkout(d):
      return d
  pkg = pathlib.Path(__file__).resolve().parents[2]
  return pkg if is_checkout(pkg) else None


def required_go(checkout:pathlib.Path) -> str | None:
  """The Go version tui/go.mod names ("1.26.5"), or None when the file has no go line."""
  m = re.search(r"^go\s+(\d+(?:\.\d+)*)\s*$", (checkout / "tui" / "go.mod").read_text(), re.M)
  return m.group(1) if m else None


def go_version(env:Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
  """(the go program on PATH, its version as "1.26.5"). (None, None) when there is no go; (path, None) when it does
  not answer `go version`."""
  env = os.environ if env is None else env
  go = shutil.which("go", path=env.get("PATH"))
  if not go:
    return None, None
  try:
    out = subprocess.run([go, "version"], capture_output=True, text=True, timeout=30).stdout
  except (OSError, subprocess.SubprocessError):
    return go, None
  m = re.search(r"\bgo(\d+(?:\.\d+)*)\b", out)
  return go, m.group(1) if m else None


def version_tuple(v:str) -> tuple[int, ...]:
  return tuple(int(x) for x in v.split("."))


def go_is_enough(have:str | None, need:str | None) -> bool:
  """The Go here can build the screen: it answers with a version, and not one older than go.mod's."""
  return bool(have) and (need is None or version_tuple(have) >= version_tuple(need))


def install_line(need:str | None) -> str:
  return f"install Go {need + ' ' if need else ''}or newer from {GO_SITE} (tui/go.mod names the version)"


def cache_dir(env:Mapping[str, str] | None = None) -> pathlib.Path:
  env = os.environ if env is None else env
  return pathlib.Path(env.get(DIR_ENV) or pathlib.Path.home() / ".cache" / "boltbeam").expanduser()


def source_files(checkout:pathlib.Path) -> list[pathlib.Path]:
  tui = checkout / "tui"
  return sorted([*tui.rglob("*.go"), *(tui / s for s in SOURCES if (tui / s).is_file())])


def source_hash(checkout:pathlib.Path) -> str:
  """One hash over the Go sources' relative paths and bytes: the same tree hashes the same on any machine."""
  h = hashlib.sha256()
  for f in source_files(checkout):
    h.update(str(f.relative_to(checkout)).encode())
    h.update(b"\0")
    h.update(f.read_bytes())
  return h.hexdigest()


def stale(binary:pathlib.Path, checkout:pathlib.Path) -> bool | None:
  """True when the sidecar's hash is not the sources' hash, False when it is, None when there is no sidecar (a
  binary built by hand: not checked)."""
  side = binary.parent / SIDECAR
  if not side.is_file():
    return None
  return side.read_text().strip() != source_hash(checkout)


def built(checkout:pathlib.Path | None, env:Mapping[str, str] | None = None) -> pathlib.Path | None:
  """Where a built boltbeam-tui is: the cache folder, next to the Go source, then PATH. None when it is not built."""
  env = os.environ if env is None else env
  places = [cache_dir(env) / BINARY]
  if checkout is not None:
    places.append(checkout / "tui" / BINARY)
  for p in places:
    if p.is_file() and os.access(p, os.X_OK):
      return p
  on_path = shutil.which(BINARY, path=env.get("PATH"))
  return pathlib.Path(on_path) if on_path else None


def facts(env:Mapping[str, str] | None = None) -> dict[str, Any]:
  """Everything doctor shows about the screen: the checkout, the Go here and the Go needed, the built binary."""
  env = os.environ if env is None else env
  repo = checkout(env=env)
  go, have = go_version(env)
  need = required_go(repo) if repo else None
  binary = built(repo, env)
  return {"checkout": str(repo) if repo else None, "go": go, "go_version": have, "go_required": need,
          "go_ok": go_is_enough(have, need), "binary": str(binary) if binary else None,
          "stale": stale(binary, repo) if binary and repo else None}


def build(checkout:pathlib.Path, go:str, out:pathlib.Path, run:Callable[..., Any] = subprocess.run) -> None:
  """`go build -o out .` in tui/, with GOTOOLCHAIN=local so no Go is downloaded, then the sources' hash beside it."""
  out.parent.mkdir(parents=True, exist_ok=True)
  proc = run([go, "build", "-o", str(out), "."], cwd=str(checkout / "tui"), capture_output=True, text=True,
             env={**os.environ, "GOTOOLCHAIN": "local"})
  if proc.returncode != 0:
    raise BuildError((proc.stderr or proc.stdout or "").strip() or f"go build exited {proc.returncode}")
  (out.parent / SIDECAR).write_text(source_hash(checkout) + "\n")


def run(passthrough:list[str], *, exec:Callable[..., Any] = os.execv, err=sys.stderr,
        env:Mapping[str, str] | None = None, run_build:Callable[..., Any] = subprocess.run) -> int:
  """Build when needed, then exec the screen with the flags given. 2: no checkout or no fit Go; 1: the build failed."""
  env = os.environ if env is None else env
  repo = checkout(env=env)
  if repo is None:
    err.write(f"no BoltBeam checkout with tui/ found: run from the clone, or export {REPO_ENV}=/path/to/BoltBeam\n")
    return 2
  binary = cache_dir(env) / BINARY
  if not binary.is_file() or stale(binary, repo) is not False:
    go, have = go_version(env)
    need = required_go(repo)
    if not go_is_enough(have, need):
      have_words = f"go{have} at {go} is older than {need}" if have else "Go is not on PATH"
      err.write(f"{have_words}: {install_line(need)}\n")
      return 2
    err.write(f"building {binary} from {repo / 'tui'} (once; again when the Go source changes)\n")
    try:
      build(repo, go, binary, run_build)
    except BuildError as exc:
      err.write(f"go build failed:\n{exc}\n")
      return 1
  argv = [str(binary), "--repo", str(repo), "--python", sys.executable, *passthrough]
  exec(str(binary), argv)
  return 0


def cmd_tui(args) -> int:
  return run(list(args.args))


def register(sub) -> None:
  p = sub.add_parser("tui", help="open the screen; builds boltbeam-tui into ~/.cache/boltbeam when it is missing or stale, "
                                "then runs it with the flags given (--json <command> for agents)")
  p.add_argument("args", nargs=argparse.REMAINDER, help="passed through to boltbeam-tui")
  p.set_defaults(fn=cmd_tui)
