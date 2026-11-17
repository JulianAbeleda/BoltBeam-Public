"""Engine mods: stored in the arkey_ie_mods repo, applied here (`boltbeam mods`, `screen mods`). BoltBeam holds no engine patch.

data/engine_patches.json is the pointer: where the mods repo is ($BOLTBEAM_IE_MODS, else mods_repo.default_path), the
commit it pins and BoltBeam's own uses of a mod. data/ie_mods_index.json is that repo's index.json at the pin, shipped
with BoltBeam: every mod fact (switch, check, output label, marker, size) is read from it, so every reader answers the
same with or without the repo (tests/test_engine_patches.py keeps it equal to the pinned index). The repo itself is
read only for a patch's text or a kernel manifest, at the pin.

    plan     the mods that fit a checkout (--src, else the current build's source, engine_patches.json apply): base =
             its commit (HEAD checked against the mods' base and said), or else each in series order applied in a
             scratch clone; a tree without .git cannot say its commit: it is taken, unchecked and said so, as the base
             of the newest released mods version (or --base); the chosen ones (identical by default) checked the
             same way, a mod whose apply dependency is not chosen named and left out, changes in the checkout that a
             clone does not carry named
    apply    a clone of the checkout at its commit (theirs is never changed; a rerun resets the clone), every chosen
             mod applied or none (a tree without .git: a copy, made a repo of its own), built with the engine's
             install recipe (engine_update.install) configured with the base commit (apply row commit_cmake) and
             recorded in builds.json under `modded`, replaced by its key on a rerun (a taken tree's key marked -copy),
             as applied, not verified (a taken tree: base unchecked too)
    verify   after the build: the commit it says (apply row commit_header) is the base, else refused and not recorded;
             with the kernel hook among the mods (install_paths mod), each kernel row's manifest at the pin against
             that commit by the hook's own rule (install_paths commit_match), and its load said as not confirmed with
             the line that confirms it; a taken tree's lines all say its base was told, not checked
"""
from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable

from boltbeam.core.refused import Refused

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


@functools.lru_cache(maxsize=1)
def pointer() -> dict[str, Any]:
  return json.loads((DATA / "engine_patches.json").read_text())


@functools.lru_cache(maxsize=1)
def index() -> dict[str, Any]:
  return json.loads((DATA / "ie_mods_index.json").read_text())


def mods(engine:str | None = None) -> list[dict[str, Any]]:
  return [m for m in index()["mods"] if engine in (None, m["engine"])]


def named(engine:str, name:str) -> list[dict[str, Any]]:
  """A mod by its name, in every version of the engine that has it."""
  return [m for m in mods(engine) if m["name"] == name]


def switch(m:dict[str, Any]) -> str | None:
  return (m.get("switch") or {}).get("name")


def _git(where:pathlib.Path, *argv:str, stdin:str | None = None) -> subprocess.CompletedProcess:
  return subprocess.run(["git", "-C", str(where), *argv], capture_output=True, text=True, input=stdin)


def _repo() -> pathlib.Path:
  repo = pointer()["mods_repo"]
  return pathlib.Path(os.path.expanduser(os.environ.get(repo["env"]) or repo["default_path"]))


def at_pin(rel:str, what:str) -> str:
  """A file of the mods repo at the pin: the one way the repo itself is read (a patch's text, a kernel manifest)."""
  got = _git(_repo(), "show", f"{pointer()['mods_repo']['pin']}:{rel}")
  if got.returncode:
    raise Refused(f"{what}: {rel} is not readable at the pin in {_repo()} ({got.stderr.strip()[-120:]})")
  return got.stdout


def head_of(where:pathlib.Path) -> str | None:
  """The commit of a checkout whose top is where; None for a tree without .git (one inside another repo too)."""
  top = _git(where, "rev-parse", "--show-toplevel")
  if top.returncode or os.path.realpath(top.stdout.strip()) != os.path.realpath(where):
    return None
  return _git(where, "rev-parse", "HEAD").stdout.strip() or None


def same_commit(engine:str, build:str | None, entry:str | None) -> bool:
  """The kernel hook's rule (install_paths commit_match): the build's commit cut at the suffix separator (never the
  manifest's entry), then equal to the entry on their common prefix of min_prefix or more characters."""
  rule = pointer()["install_paths"][engine]["commit_match"]
  build, entry = (build or "").split(rule["suffix_sep"])[0], entry or ""
  n = min(len(build), len(entry))
  return n >= rule["min_prefix"] and build[:n] == entry[:n]


def _base(engine:str, want:str | None) -> str:
  """The base commit a tree without .git is taken at: --base (a mods version or a commit), else the newest released
  version's (its base names a release; the commit-named versions are unreleased lines)."""
  vs = [v for v in index().get("versions", []) if v["engine"] == engine and v["base"].get("commit")]
  if not want:
    return max(vs, key=lambda v: [int(x) for x in re.findall(r"\d+", v["base"].get("release") or "")])["base"]["commit"]
  got = next((v["base"]["commit"] for v in vs if want == v["version"] or same_commit(engine, want, v["base"]["commit"])), None)
  if got is None:
    raise Refused(f"--base {want} is not a {engine} mods version or base commit ({', '.join(v['version'] for v in vs)})")
  return got


def _reset(copy:pathlib.Path, src:pathlib.Path, head:str, skip:tuple[str, ...] | list[str] = ()) -> None:
  """A clone of the checkout at its commit, clean: made, or an earlier one reset (a rerun starts from the same tree).
  A tree without .git: a fresh copy of it (its top-level build directories, skip, left out), a repo of its own."""
  if head_of(src) is None:
    shutil.rmtree(copy, ignore_errors=True)
    top = lambda d, names: [n for n in names if os.path.samefile(d, src) and os.path.isdir(os.path.join(d, n))  # noqa: E731
                            and any(fnmatch.fnmatch(n, p) for p in skip)]
    shutil.copytree(src, copy, symlinks=True, ignore=top)
    for argv in (("init", "-q"), ("add", "-A"), ("-c", "user.name=boltbeam", "-c", "user.email=boltbeam@localhost",
                                                 "-c", "commit.gpgsign=false", "commit", "-qm", f"a copy of {src}")):
      if (got := _git(copy, *argv)).returncode:
        raise Refused(f"the copy of {src} in {copy} could not be made a repo: {got.stderr.strip()[-200:]}")
    return
  if not (copy / ".git").exists():
    got = subprocess.run(["git", "clone", "-q", "--no-checkout", str(src), str(copy)], capture_output=True, text=True)
    if got.returncode:
      raise Refused(f"the checkout {src} could not be cloned: {got.stderr.strip()[-200:]}")
  elif _git(copy, "cat-file", "-e", head).returncode:
    _git(copy, "fetch", "-q", str(src), head)
  for argv in (("checkout", "-q", "-f", head), ("clean", "-fdq")):
    if (got := _git(copy, *argv)).returncode:
      raise Refused(f"the clone in {copy} could not be reset to {head[:9]}: {got.stderr.strip()[-200:]}")


def _series(copy:pathlib.Path, rows:list[dict[str, Any]]) -> list[str]:
  """Apply in order, each on the ones before it that applied; returns the ids that did not apply."""
  return [m["id"] for m in rows if _git(copy, "apply", "-", stdin=at_pin(m["patch"], m["id"])).returncode]


def plan(engine:str, src:pathlib.Path, labels:tuple[str, ...] = ("identical",), only:list[str] | None = None,
         base:str | None = None) -> dict[str, Any]:
  head = head_of(src)
  if head and base and head != _base(engine, base):
    raise Refused(f"the checkout {src} is at {head[:9]}, not --base {base}")
  taken = head is None
  head = head or _base(engine, base)
  row = pointer()["apply"][engine]
  rows = [m for m in mods(engine) if m["applier"] == row["applier"]]
  exact = [m for m in rows if m["base"].get("commit") == head]
  if unknown := sorted(set(only or []) - {m["id"] for m in exact or rows}):
    raise Refused(f"not a mod for this checkout's commit {head[:9]}: {', '.join(unknown)}")
  pick = [m for m in (exact or rows) if (m["id"] in only if only else m["state"] == "series" and m["output"] in labels)]
  ids, missing, conflicts = {m["id"] for m in pick}, {}, []
  with tempfile.TemporaryDirectory(prefix="boltbeam-mods-") as tmp:
    while True:  # to a fixed point: a mod whose apply dependency is not taken, or conflicted, is left out with it
      while miss := {m["id"]: d for m in pick if (d := [x for x in (f"{m['engine']}/{m['version']}/{y}" for y in m["depends_on"]["apply"])
                                                        if x not in {p["id"] for p in pick}])}:
        missing.update(miss)
        pick = [m for m in pick if m["id"] not in miss]
      _reset(pathlib.Path(tmp) / "src", src, head, row.get("copy_skip", []))
      if not (bad := _series(pathlib.Path(tmp) / "src", pick)):
        break
      conflicts += bad
      pick = [m for m in pick if m["id"] not in bad]
  bases = sorted({m["base"].get("commit", "")[:9] for m in rows})
  return {"head": head, "taken": taken,
          "fits": (f"base, taken: the tree has no .git, so it is taken to be the mods' base {head[:9]} (unchecked)" if taken
                   else f"base: HEAD {head[:9]} is the mods' base" if exact else f"another commit: HEAD {head[:9]} is not "
                   f"a mods base ({', '.join(bases)}); applied in series order in a clone"),
          "chosen": [m["id"] for m in pick], "conflicts": conflicts, "missing_dependencies": missing,
          "not_carried": [] if taken else _git(src, "status", "--porcelain").stdout.splitlines(),
          "not_chosen": [f"{m['id']} ({m['output']}, {m['state']})" for m in (exact or rows) if m["id"] not in ids]}


def apply(engine:str, src:pathlib.Path, chosen:list[str], into:pathlib.Path, head:str, vendor, taken:bool = False) -> dict[str, Any]:
  """Every chosen mod applied to the clone, or none: on a failure the clone is reset and the refusal names the mod.
  taken: the tree has no .git, so head is the base it was told, never checked: never recorded as verified."""
  from boltbeam.workflow import engine_update
  row = pointer()["apply"][engine]
  copy = into / row["source"]
  _reset(copy, src, head, row.get("copy_skip", []))
  rows = [m for m in mods(engine) if m["id"] in chosen]
  if failed := _series(copy, rows):
    _reset(copy, src, head, row.get("copy_skip", []))
    raise Refused(f"{', '.join(failed)} did not apply on {head[:9]} in series order; nothing was applied")
  build = engine_update.install(engine, f"{head[:9]}-mods", into, vendor,
                                {"cmake": [f.format(commit=head) for f in row.get("commit_cmake", [])]})
  return {**build, "base_commit": head, "checks": verify(engine, into, head, rows, taken),
          "status": "applied, not verified; base unchecked (no .git)" if taken else "applied, not verified",
          "mods": [{k: m[k] for k in ("id", "switch", "output")} for m in rows]}


def verify(engine:str, folder:pathlib.Path, head:str, rows:list[dict[str, Any]], taken:bool = False) -> dict[str, Any]:
  """The built commit is the base (refused otherwise); each kernel the hook among rows would take, checked against it.
  taken: the base was told to the build, not checked, so the header agrees by construction: every line says so."""
  h, told = pointer()["apply"][engine].get("commit_header"), f"base taken, unchecked: the build was told it is {head[:9]}" if taken else ""
  if not h:
    return {"commit": told or "not checked: the engine's apply row names no commit header"}
  text = (folder / h["path"]).read_text() if (folder / h["path"]).is_file() else ""
  got = m.group(1) if (m := re.search(h["pattern"], text)) else None
  if not same_commit(engine, got, head):
    raise Refused(f"the build says its commit is {got or 'nothing'} ({folder / h['path']}), not the base {head[:9]}: "
                  "a kernel manifest checks the build's commit, so it would not load; the build is not recorded")
  out: dict[str, Any] = {"commit": told or f"{got} ({h['path']}), the base {head[:9]}"}
  ip = pointer()["install_paths"].get(engine) or {}
  if hook := next((m for m in rows if m["name"] == ip.get("mod")), None):
    out["kernels"] = [{**k, "loaded": f"{told}; {k['loaded']}" if told else k["loaded"]}
                      for k in (_kernel(engine, k, got, hook, ip) for k in index().get("kernels", []) if k["hook"] == hook["id"])]
  return out


def _kernel(engine:str, k:dict[str, Any], built:str, hook:dict[str, Any], ip:dict[str, Any]) -> dict[str, Any]:
  """One kernel row's manifest at the pin against the built commit; a manifest that cannot be read says unknown (the
  build is still recorded). The path printed is the working tree's, said when it is not the pinned content checked."""
  rel = k["dir"] + ip["manifest_file"]
  try:
    text = at_pin(rel, k["id"])
    man = json.loads(text)
    builds = [str(b) for b in man["engine"]["builds"]]
    lines = [ip["loaded_line"].format(id=e["id"]) for e in man["entries"]]
  except (Refused, ValueError, KeyError, TypeError) as exc:
    return {"id": k["id"], "builds": [], "loaded": f"unknown: the manifest {rel} at the pin is not readable ({type(exc).__name__}: {exc})"}
  if not any(same_commit(engine, built, b) for b in builds):
    return {"id": k["id"], "builds": builds, "loaded": f"will not load: the manifest was checked on {', '.join(builds)}, "
                                                        f"this build is {built}; the engine's own kernels run"}
  wt = _repo() / rel
  differs = "" if wt.is_file() and wt.read_text() == text else " (not the pinned file checked here: check out the pin first)"
  return {"id": k["id"], "builds": builds, "loaded": f"not confirmed: the manifest matches this build, no load check runs "
          f"here; with {switch(hook)}={wt}{differs} its log must say {'; '.join(lines)}"}


HELP = "the arkey_ie_mods mods that fit the engine build here; --apply builds them into a clone"


def arguments(p) -> None:
  """The flags of `screen mods` and `boltbeam mods` (cli/mods.py), one set."""
  from boltbeam.collectors import engine_builds
  from boltbeam.workflow.context_audit import OUTPUTS
  p.add_argument("--engine", choices=list(engine_builds.rows()), help="default: the one engine with mods for this machine's build")
  p.add_argument("--src", help="the source of the build you run (default: the current build's src)")
  p.add_argument("--base", help="a source without .git: the mods version or base commit it is (default: the newest released one's)")
  p.add_argument("--apply", action="store_true", help="apply the chosen mods to a clone of it and build it (yours is never changed)")
  p.add_argument("--label", action="append", default=[], choices=list(OUTPUTS[1:3]), help="also take mods with this output label")
  p.add_argument("--mod", action="append", help="exactly these mod ids (engine/version/name)")
  p.add_argument("--target", default=None, help="with --apply or no --engine: the chip the build is for (default: this machine's)")


def only_engine(backend:str | None) -> str:
  """The engine `boltbeam mods` takes without --engine: the one with an applier (apply row) this machine's build can run."""
  from boltbeam.collectors import engine_builds as eb
  got = [e for e in pointer()["apply"] if e != "why" and e in eb.rows_for(backend)]
  if len(got) != 1:
    raise Refused(f"pass --engine: {len(got)} engines have mods for this machine ({', '.join(got) or 'none'})")
  return got[0]


def text(res:dict[str, Any]) -> str:
  """`boltbeam mods` in plain lines; --json prints the record itself."""
  out = [f"mods {res.get('engine', '')}".rstrip() + f": {res['refused']}"] if res.get("refused") else []
  if "fits" in res:
    out.append(f"{res['engine']} source: {res['fits']}")
    out += [f"  mods that fit ({len(res['chosen'])}):", *(f"    {m}" for m in res["chosen"] or ["none"])]
    out += [f"  conflicts, left out: {', '.join(res['conflicts'])}"] * bool(res["conflicts"])
    out += [f"  left out, a mod it needs is not taken: {m} needs {', '.join(d)}" for m, d in res["missing_dependencies"].items()]
    out += [f"  not chosen: {m}" for m in res["not_chosen"]]
    out += [f"  not carried into the clone: {len(res['not_carried'])} uncommitted changes in your tree"] * bool(res["not_carried"])
  if b := res.get("build"):
    out += [f"built: {b['folder']} ({b['key']})", f"  commit {b['checks']['commit']}",
            *(f"  kernel {k['id']}: {k['loaded']}" for k in b["checks"].get("kernels", []))]
  elif res.get("chosen") and not res.get("refused"):
    out.append("`boltbeam mods --apply` builds them into a clone (yours is never changed)")
  return "\n".join(out)


def modded(engine:str, path:str | None) -> dict[str, Any] | None:
  """The modded build a path belongs to (builds.json modded), for tune's words."""
  from boltbeam.collectors import engine_builds as eb
  real = os.path.realpath(path) if path else None
  return next((b for b in eb.entry(eb.load(), engine).get("modded") or []
               if real and real in {os.path.realpath(p) for p in (b.get("paths") or {}).values()}), None)


def run(engine:str, *, src:str | None = None, labels:tuple[str, ...] = ("identical",), only:list[str] | None = None,
        base:str | None = None, apply_them:bool = False, backend:str | None = None,
        say:Callable[[str], None] = print) -> dict[str, Any]:
  """`screen mods`: the plan; with apply_them, the modded build, recorded."""
  from boltbeam.collectors import engine_builds as eb
  from boltbeam.target import vendors
  if engine not in pointer()["apply"]:
    return {"refused": f"no applier is wired for {engine} mods yet (engine_patches.json apply)"}
  state = eb.load()
  cur = eb.entry(state, engine).get("current") or {}
  where = pathlib.Path(src).expanduser() if src else (pathlib.Path(cur["folder"]) / pointer()["apply"][engine]["source"] if cur.get("folder") else None)
  if where is None or not where.is_dir():
    return {"refused": f"no source tree for {engine}: pass --src (the source of the build you run)"}
  try:
    got = plan(engine, where, labels, only, base)
    say(f"mods {engine}: {got['fits']}")
    if not apply_them:
      return got
    if not got["chosen"]:
      return {**got, "refused": "no mod to apply on this commit with the labels allowed"}
    # a taken tree's key is marked, so it never replaces a checked checkout's build at the same base
    key = f"{got['head'][:9]}{'-copy' * got['taken']}-mods-{hashlib.sha256(' '.join(sorted(got['chosen'])).encode()).hexdigest()[:8]}"
    into = pathlib.Path(state.get("root") or eb.state_file().with_name("engines")).expanduser() / engine / key
    say(f"mods {engine}: {len(got['chosen'])} mods onto a clone of {where} at {got['head'][:9]}, built in {into}")
    build = {**apply(engine, where, got["chosen"], into, got["head"], vendors.for_backend(backend), got["taken"]), "key": key}
    for line in [f"commit {build['checks']['commit']}", *(f"kernel {x['id']}: {x['loaded']}" for x in build["checks"].get("kernels", []))]:
      say(f"mods {engine}: {line}")
  except (Refused, RuntimeError, OSError, subprocess.SubprocessError) as exc:  # engine_update.CannotInstall is a RuntimeError
    return {"refused": str(exc) if isinstance(exc, Refused) else f"{type(exc).__name__}: {exc}"}
  e = eb.entry(state, engine)
  e["modded"] = [b for b in e.get("modded") or [] if b.get("key") != key] + [build]
  eb.save(state)
  return {**got, "build": build}
