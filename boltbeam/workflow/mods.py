"""Engine mods: stored in the arkey_ie_mods repo, applied here (`boltbeam mods`, `screen mods`). BoltBeam holds no engine patch.

data/engine_patches.json is the pointer: where the mods repo is ($BOLTBEAM_IE_MODS, else mods_repo.default_path), the
commit it pins and BoltBeam's own uses of a mod. data/ie_mods_index.json is that repo's index.json at the pin, shipped
with BoltBeam: every mod fact (switch, check, output label, marker, size) is read from it, so every reader answers the
same with or without the repo (tests/test_engine_patches.py keeps it equal to the pinned index). The repo itself is
read only for a patch's text or a kernel manifest, at the commit read.

    resolve  which commit `mods` reads (run): --pin or $<pin_env> exactly; else the newest commit on origin's default
             branch past the pin whose index schema this BoltBeam reads (SCHEMAS) and whose requires.boltbeam it meets,
             walking back from the newest; else the pin (the floor and the offline default). BoltBeam's own clone is
             fetched first; a configured clone never (its origin as last fetched). Said, and recorded in the plan,
             the build record and a run on the build (mods_commit)

    plan     the mods that fit a checkout (--src, else the current build's source, engine_patches.json apply): base =
             its commit (HEAD checked against the mods' base and said), or else each in series order applied in a
             scratch clone; a tree without .git cannot say its commit: it is taken, unchecked and said so, as the base
             of the newest released mods version (or --base); the chosen ones (by default the series: its identical
             and quality-gated mods, the set the benchmarks measured; parked never) checked the
             same way, a mod whose apply dependency is not chosen named and left out, changes in the checkout that a
             clone does not carry named
    apply    a clone of the checkout at its commit (theirs is never changed; a rerun resets the clone), every chosen
             mod applied or none (a tree without .git: a copy, made a repo of its own), built with the engine's
             install recipe (engine_update.install) configured with the base commit and no fetch (apply row commit_cmake, offline_cmake) and
             recorded in builds.json under `modded`, replaced by its key on a rerun (a taken tree's key marked -copy),
             as applied, not verified (a taken tree: base unchecked too); each kernel the build matches installed in
             its folder (kernels/), its manifest the value a run gives the hook's switch (run_env)
    verify   after the build: the commit it says (apply row commit_header) is the base, else refused and not recorded;
             with the kernel hook among the mods (install_paths mod), each kernel row's manifest at the pin against
             that commit by the hook's own rule (install_paths commit_match), and its load said as not confirmed with
             the line that confirms it; a taken tree's lines all say its base was told, not checked
"""
from __future__ import annotations

import contextlib
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


SCHEMAS = ("arkey_ie_mods.index.v1",)  # the index schemas this BoltBeam reads: resolve takes no commit with another
_AT: dict[str, Any] = {}  # the commit run() reads (resolve, while reading): commit, index, record; empty = the pin


@functools.lru_cache(maxsize=1)
def shipped() -> dict[str, Any]:
  return json.loads((DATA / "ie_mods_index.json").read_text())


def index() -> dict[str, Any]:
  """The mods index at the commit read: the one resolve chose while run() reads it, else the pin's (shipped)."""
  return _AT.get("index") or shipped()


def commit() -> str:
  return _AT.get("commit") or pointer()["mods_repo"]["pin"]


def mods(engine:str | None = None) -> list[dict[str, Any]]:
  return [m for m in index()["mods"] if engine in (None, m["engine"])]


def named(engine:str, name:str) -> list[dict[str, Any]]:
  """A mod by its name, in every version of the engine that has it."""
  return [m for m in mods(engine) if m["name"] == name]


def switch(m:dict[str, Any]) -> str | None:
  return (m.get("switch") or {}).get("name")


def _git(where:pathlib.Path, *argv:str, stdin:str | None = None) -> subprocess.CompletedProcess:
  return subprocess.run(["git", "-C", str(where), *argv], capture_output=True, text=True, input=stdin)


def _own() -> pathlib.Path:  # BoltBeam's own clone, beside its engine scan, when no clone is configured
  from boltbeam.collectors import engine_saved
  return engine_saved.engines_file().parent / "arkey_ie_mods"


def _repo() -> pathlib.Path:
  """The mods repo: $<env>, else the default path when a clone is there, else BoltBeam's own clone (ensure fetches it)."""
  repo = pointer()["mods_repo"]
  given = pathlib.Path(os.path.expanduser(os.environ.get(repo["env"]) or repo["default_path"]))
  return given if os.environ.get(repo["env"]) or given.exists() else _own()


def _get() -> str:  # what fixes a missing mods repo, or one without the pin
  r, w = pointer()["mods_repo"], _repo()
  return f"fetch it: git -C {w} fetch origin" if (w / ".git").exists() else f"clone it: git clone {r['url']} {w} (or set ${r['env']})"


def _has_pin(w:pathlib.Path) -> bool:
  return w.is_dir() and _git(w, "cat-file", "-e", pointer()["mods_repo"]["pin"] + "^{commit}").returncode == 0


def ensure(say:Callable[[str], None] = print) -> pathlib.Path:
  """The mods repo with the pinned commit in it. A configured clone ($<env>, or the default path) is used as it is and
  never fetched into; with none, BoltBeam clones the repo's url into its own folder (~/.boltbeam/arkey_ie_mods) and
  fetches the pin, saying what it fetches and from where first. Refused when the pin is still not there."""
  r, w = pointer()["mods_repo"], _repo()
  if _has_pin(w):
    return w
  if w != _own():
    raise Refused(f"the mods repo {w} does not have the pinned commit {r['pin'][:9]}; {_get()}")
  say(f"mods: fetching arkey_ie_mods at {r['pin'][:9]} from {r['url']} into {w}")
  if not (w / ".git").exists():
    w.parent.mkdir(parents=True, exist_ok=True)
    try:
      got = subprocess.run(["git", "clone", "-q", "--no-checkout", r["url"], str(w)], capture_output=True, text=True,
                           timeout=CLONE_TIMEOUT_S, env={**os.environ, **QUIET})
    except subprocess.TimeoutExpired:
      raise Refused(f"arkey_ie_mods could not be cloned from {r['url']}: {_why('no_answer', s=CLONE_TIMEOUT_S)}") from None
    if got.returncode:
      raise Refused(f"arkey_ie_mods could not be cloned from {r['url']}: {got.stderr.strip()[-200:]}")
  if not _has_pin(w):
    _git(w, "fetch", "-q", "origin", r["pin"])  # a pin no branch holds: by its id (a server that refuses that leaves it missing, said below)
  if not _has_pin(w):
    raise Refused(f"the pinned commit {r['pin'][:9]} is not in {r['url']} (cloned into {w})")
  return w


CLONE_TIMEOUT_S = 120  # the first clone of the mods repo; bounded so an offline machine is refused, not hung
FETCH_TIMEOUT_S = 10  # a fetch of the mods repo (usually a few new commits); past it the pin is read, said as offline
QUIET = {"GIT_TERMINAL_PROMPT": "0"}  # a fetch or clone never waits on a password prompt


def _why(key:str, **kw) -> str:  # resolve's reasons, in plain_words.json mods.resolve_why
  from boltbeam.report import plain
  return plain.words()["mods"]["resolve_why"][key].format(**kw)


def _fetch(w:pathlib.Path, *what:str) -> str | None:
  """git fetch origin [what] into BoltBeam's own clone, bounded (offline must not hang a command); why it failed, or None."""
  try:
    got = subprocess.run(["git", "-C", str(w), "fetch", "-q", "origin", *what], capture_output=True, text=True,
                         timeout=FETCH_TIMEOUT_S, env={**os.environ, **QUIET})
  except subprocess.TimeoutExpired:
    return _why("no_answer", s=FETCH_TIMEOUT_S)
  return (got.stderr.strip()[-160:] or "git fetch failed") if got.returncode else None


def _ver(v:str | None) -> tuple[int, ...]:
  return tuple(int(x) for x in re.findall(r"\d+", v or ""))


def _index_at(w:pathlib.Path, c:str) -> dict[str, Any] | None:
  got = _git(w, "show", f"{c}:index.json")
  try:
    return json.loads(got.stdout) if not got.returncode else None
  except ValueError:
    return None


def unreadable(idx:dict[str, Any] | None) -> str | None:
  """Why this BoltBeam cannot read a commit's index (its schema, its requires.boltbeam), or None."""
  from boltbeam.cli.self_update import installed
  if not isinstance(idx, dict) or idx.get("schema") not in SCHEMAS:
    return _why("schema", schema=idx.get("schema") if isinstance(idx, dict) else None)
  need, mine = (idx.get("requires") or {}).get("boltbeam"), installed()
  return _why("requires", need=need, mine=mine) if need and (not mine or _ver(need) > _ver(mine)) else None


def resolve(say:Callable[[str], None] = print, pin:str | None = None) -> dict[str, Any]:
  """The commit of the mods repo `mods` reads (module doc, resolve): {commit, how, why, pin, words}; refused when a
  forced commit is missing or unreadable. The pin is the floor: a commit is taken only from origin's default branch
  past it; anything that fails (offline, no default branch, a branch without the pin) reads the pin and says why."""
  r = pointer()["mods_repo"]
  floor, env = r["pin"], r["pin_env"]
  forced, by = (pin, "--pin") if pin else (os.environ.get(env), f"${env}")
  w = ensure(say)
  own = w == _own()
  if forced:
    if own and _git(w, "cat-file", "-e", f"{forced}^{{commit}}").returncode:
      _fetch(w, forced)
    full = _git(w, "rev-parse", "--verify", "-q", f"{forced}^{{commit}}").stdout.strip()
    if not full:
      raise Refused(f"{by} {forced}: no such commit in {w}; {_get()}")
    if why := unreadable(idx := _index_at(w, full)):
      raise Refused(f"{by} {full[:9]}: {why}")
    return _at(full, "forced", _why("forced", by=by), idx)
  if own and (err := _fetch(w)):
    return _at(floor, "pin", _why("offline", error=err))
  tip = _git(w, "rev-parse", "--verify", "-q", "refs/remotes/origin/HEAD").stdout.strip()
  if not tip:
    return _at(floor, "pin", _why("no_head", clone=w))
  if _git(w, "merge-base", "--is-ancestor", floor, tip).returncode:
    return _at(floor, "pin", _why("off_branch", tip=tip[:9]))
  skipped = []  # only descendants of the pin: --ancestry-path keeps a first-parent commit older than a merge of the pin out
  for c in _git(w, "rev-list", "--first-parent", "--ancestry-path", f"{floor}..{tip}").stdout.split():
    if why := unreadable(idx := _index_at(w, c)):
      skipped.append((c[:9], why))
      continue
    return _at(c, "walked back" if skipped else "newest compatible",
               _why("newer", commit=skipped[0][0], why=skipped[0][1]) if skipped else _why("newest", tip=tip[:9]), idx)
  return _at(floor, "pin", _why("newer", commit=skipped[0][0], why=skipped[0][1]) if skipped else _why("at_pin"))


def _at(c:str, how:str, why:str, idx:dict[str, Any] | None = None) -> dict[str, Any]:
  from boltbeam.report import plain
  pin = pointer()["mods_repo"]["pin"]
  return {"commit": c, "how": how, "why": why, "pin": pin, "index": None if c == pin else idx,
          "words": plain.words()["mods"]["resolved" if how == "newest compatible" else "resolved_why"].format(commit=c[:9], how=how, why=why)}


@contextlib.contextmanager
def reading(at:dict[str, Any]):
  """Every mod fact read from the resolved commit's index, every file at that commit, for the block (run)."""
  _AT.clear()
  _AT.update({k: at[k] for k in ("commit", "index") if at.get(k)}, record=record(at))
  try:
    yield
  finally:
    _AT.clear()


def record(at:dict[str, Any]) -> dict[str, Any]:  # what a plan, a build and a run on it keep of the commit read
  return {k: at[k] for k in ("commit", "how", "why", "pin", "words")}


def at_pin(rel:str, what:str) -> str:
  """A file of the mods repo at the commit read (the pin unless resolve chose another): the one way the repo itself is
  read (a patch's text, a kernel manifest)."""
  got = _git(_repo(), "show", f"{commit()}:{rel}")
  if got.returncode:
    raise Refused(f"{what}: {rel} is not readable at {commit()[:9]} in {_repo()}; {_get()} ({got.stderr.strip()[-120:]})")
  return got.stdout


def pin_bytes(rel:str, what:str, sha256:str) -> bytes:
  """A file of the mods repo at the commit read as bytes (a kernel's cubin too), refused unless it hashes to the index's sha256."""
  got = subprocess.run(["git", "-C", str(_repo()), "show", f"{commit()}:{rel}"], capture_output=True)
  if got.returncode or hashlib.sha256(got.stdout).hexdigest() != sha256:
    raise Refused(f"{what}: {rel} at {commit()[:9]} in {_repo()} is not the file the index names "
                  f"({got.stderr.decode(errors='replace').strip()[-120:] or 'sha256 differs'})")
  return got.stdout


def kernel_dir(folder:pathlib.Path, k:dict[str, Any]) -> pathlib.Path:
  """Where a modded build keeps a kernel row's files (its manifest names its cubins beside it)."""
  return pathlib.Path(folder) / "kernels" / k["dir"].rstrip("/").rsplit("/", 1)[-1]


def install_kernels(engine:str, folder:pathlib.Path, checks:dict[str, Any]) -> list[str]:
  """Each kernel the build matches (verify said so), its files at the pin copied into the build folder; their manifests.
  A run on the build points the hook's switch at the one manifest (run_env), so the kernel the benchmarks ran runs."""
  ip, rows, out = pointer()["install_paths"][engine], {k["id"]: k for k in index().get("kernels", [])}, []
  for got in checks.get("kernels", []):
    if not got.get("matches"):
      continue
    k = rows[got["id"]]
    dest = kernel_dir(folder, k)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for name, sha in k["files"].items():
      (dest / name).write_bytes(pin_bytes(k["dir"] + name, k["id"], sha))
    out.append(str(dest / ip["manifest_file"]))
  return out


def kernel_manifest(engine:str, checks:dict[str, Any], folder:pathlib.Path) -> str | None:
  """The one manifest a run on the build gives the hook's switch: of the kernel rows installed, the one whose hook no
  other installed row's hook runs on (a region manifest on the window hook, 19, covers the one-op manifest on 09 and
  names its files beside it); None when that is not one row."""
  rows, by = {k["id"]: k for k in index().get("kernels", [])}, {m["id"]: m for m in mods()}
  got = [rows[c["id"]] for c in checks.get("kernels", []) if c.get("matches")]
  under = {f"{h.rsplit('/', 1)[0]}/{d}" for h in {k["hook"] for k in got} for d in (by.get(h) or {}).get("depends_on", {}).get("run", [])}
  top = [k for k in got if k["hook"] not in under]
  return str(kernel_dir(folder, top[0]) / pointer()["install_paths"][engine]["manifest_file"]) if len(top) == 1 else None


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
  """Apply in order, each on the ones before it that applied; returns the ids that did not apply. Rows that name one
  patch are features of one module (the index's `module`): the patch is applied once and stands or fails for all."""
  bad = {p: _git(copy, "apply", "-", stdin=at_pin(p, i)).returncode for p, i in {m["patch"]: m["id"] for m in rows}.items()}
  return [m["id"] for m in rows if bad[m["patch"]]]


def default_labels() -> tuple[str, ...]:  # identical and quality-gated: the series the benchmarks measured
  from boltbeam.workflow.context_audit import GATED, IDENTICAL
  return IDENTICAL, GATED


def evidence(mod_id:str) -> str | None:  # a mod version's benchmark (engine_patches.json benchmarks, keyed engine/version)
  rel = pointer().get("benchmarks", {}).get(mod_id.rsplit("/", 1)[0])
  return f"{pointer()['mods_repo']['url'].replace('https://', '')}/tree/master/{rel}" if rel else None


def verified() -> set[str]:
  """The mods whose version the index marks verified (a tested tree): the only ones applied; the rest are listed."""
  ok = {(v["engine"], v["version"]) for v in index().get("versions", []) if (v.get("verified") or {}).get("state", "none") != "none"}
  return {m["id"] for m in mods() if (m["engine"], m["version"]) in ok}


def plan(engine:str, src:pathlib.Path, labels:tuple[str, ...] | None = None, only:list[str] | None = None,
         base:str | None = None) -> dict[str, Any]:
  if only and labels:
    raise Refused("--label and --mod both choose the mods: give one or the other")
  labels = labels or default_labels()
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
  ok = verified()
  if bad := sorted(set(only or []) - ok):
    from boltbeam.report import plain
    raise Refused(plain.words()["mods"]["unverified_asked"].format(ids=", ".join(bad)))
  pick = [m for m in (exact or rows) if m["id"] in ok and (m["id"] in only if only else m["state"] == "series" and m["output"] in labels)]
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
          "chosen": [m["id"] for m in pick], "conflicts": conflicts, "labels": None if only else list(labels),
          "quality_gated": {m["id"]: [switch(m), evidence(m["id"])] for m in pick if m["output"] == "quality-gated"},
          "marks": {m["id"]: m["output"] if m["state"] == "series" else m["state"] for m in pick
                    if m["output"] != "identical" or m["state"] != "series"}, "missing_dependencies": missing,
          "not_carried": [] if taken else _git(src, "status", "--porcelain").stdout.splitlines(),
          "not_chosen": [f"{m['id']} ({m['output']}, {m['state'] if m['id'] in ok else 'version not verified'})" for m in (exact or rows) if m["id"] not in ids]}


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
                                {"cmake": [f.format(commit=head) for f in row.get("commit_cmake", [])] + row.get("offline_cmake", [])})
  checks = verify(engine, into, head, rows, taken)
  return {**build, "base_commit": head, "checks": checks, "kernel_manifests": install_kernels(engine, into, checks),
          "kernel_manifest": kernel_manifest(engine, checks, into),
          "status": "applied, not verified; base unchecked (no .git)" if taken else "applied, not verified",
          "mods_commit": commit(), "mods": [{k: m.get(k) for k in ("id", "engine", "name", "switch", "output")} for m in rows]}


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
                      for k in (_kernel(engine, k, got, hook, ip, folder) for k in index().get("kernels", []) if k["hook"] in {m["id"] for m in rows})]
  return out


def _kernel(engine:str, k:dict[str, Any], built:str, hook:dict[str, Any], ip:dict[str, Any], folder:pathlib.Path) -> dict[str, Any]:
  """One kernel row's manifest at the pin against the built commit; a manifest that cannot be read says unknown (the
  build is still recorded). matches: the hook would take it on this build, so apply installs it beside the build."""
  rel = k["dir"] + ip["manifest_file"]
  try:
    text = at_pin(rel, k["id"])
    man = json.loads(text)
    builds = [str(b) for b in man["engine"]["builds"]]
    lines = [ip["loaded_line"].format(id=e["id"]) for e in man["entries"]]
  except (Refused, ValueError, KeyError, TypeError) as exc:
    return {"id": k["id"], "builds": [], "matches": False, "loaded": f"unknown: the manifest {rel} at the pin is not readable ({type(exc).__name__}: {exc})"}
  if not any(same_commit(engine, built, b) for b in builds):
    return {"id": k["id"], "builds": builds, "matches": False, "loaded": f"will not load: the manifest was checked on "
            f"{', '.join(builds)}, this build is {built}; the engine's own kernels run"}
  return {"id": k["id"], "builds": builds, "matches": True, "loaded": f"not confirmed: the manifest matches this build and is "
          f"installed with it; a run on the build sets {switch(hook)}={kernel_dir(folder, k) / ip['manifest_file']}, and its "
          f"log must say {'; '.join(lines)}"}


HELP = "the arkey_ie_mods mods that fit the engine build here; apply builds them into a clone"


def arguments(p, hide_apply:bool = False) -> None:
  """The flags of `screen mods` and `boltbeam mods` (cli/mods.py), one set."""
  import argparse
  from boltbeam.collectors import engine_builds
  from boltbeam.workflow.context_audit import OUTPUTS
  p.add_argument("--engine", choices=list(engine_builds.rows()), help="default: the one engine with mods for this machine's build")
  p.add_argument("--src", help="the source of the build you run (default: the current build's src)")
  p.add_argument("--base", help="a source without .git: the mods version or base commit it is (default: the newest released one's)")
  p.add_argument("--apply", action="store_true", help=argparse.SUPPRESS if hide_apply else  # `boltbeam mods apply`
                 "apply the chosen mods to a clone of it and build it (yours is never changed)")
  p.add_argument("--label", action="append", default=[], choices=list(OUTPUTS[:3]),
                 help="only series mods with this output label (default: identical and quality-gated, the benchmarked series)")
  p.add_argument("--mod", action="append", help="exactly these mod ids (engine/version/name)")
  p.add_argument("--target", default=None, help="the chip the build is for (default: this machine's); the mods refuse a GPU backend they are not for")
  p.add_argument("--rebuild", action="store_true", help="with apply: build the chosen mods again even when that build is already on disk")
  p.add_argument("--pin", help="read arkey_ie_mods at exactly this commit (also $BOLTBEAM_IE_MODS_PIN; default: the newest "
                 "compatible commit past BoltBeam's pin, the pin when offline)")


def only_engine(backend:str | None) -> str:
  """The engine `boltbeam mods` takes without --engine: the one with an applier (apply row) this machine's build can run."""
  from boltbeam.collectors import engine_builds as eb
  got = [e for e in pointer()["apply"] if e != "why" and e in eb.rows_for(backend)]
  if len(got) != 1:
    raise Refused(f"pass --engine: {len(got)} engines have mods for this machine ({', '.join(got) or 'none'})")
  return got[0]


def text(res:dict[str, Any]) -> str:
  """`boltbeam mods` in plain lines; --json prints the record itself."""
  out = [(res.get("mods_commit") or {}).get("words")] * bool((res.get("mods_commit") or {}).get("words"))
  out += [f"mods {res.get('engine', '')}".rstrip() + f": {res['refused']}"] if res.get("refused") else []
  if "fits" in res:
    out.append(f"{res['engine']} source: {res['fits']}")
    qg = res.get("quality_gated") or {}
    out += [f"  mods that fit ({len(res['chosen'])}; " + (f"series, output {' or '.join(res['labels'])}):" if res.get("labels") else "named by --mod):"),
            *(f"    {m}" + (f" ({res['marks'][m]})" if m in (res.get("marks") or {}) else "") for m in res["chosen"] or ["none"])]
    out += [f"  quality-gated: {m}, its output may differ from stock with {sw or 'it'} on; evidence: {e or 'none recorded'}"
            for m, (sw, e) in qg.items()]
    out += [f"  conflicts, left out: {', '.join(res['conflicts'])}"] * bool(res["conflicts"])
    out += [f"  left out, a mod it needs is not taken: {m} needs {', '.join(d)}" for m, d in res["missing_dependencies"].items()]
    out += [f"  not chosen: {m}" for m in res["not_chosen"]]
    out += [f"  not carried into the clone: {len(res['not_carried'])} uncommitted changes in your tree"] * bool(res["not_carried"])
  if b := res.get("build"):
    out += [f"{'already built, not rebuilt (--rebuild builds it again)' if res.get('reused') else 'built'}: {b['folder']} ({b['key']})",
            f"  commit {(b.get('checks') or {}).get('commit')}", *(f"  kernel {k['id']}: {k['loaded']}" for k in (b.get("checks") or {}).get("kernels", []))]
  elif res.get("already_built") and not res.get("refused"):
    out.append(f"already built as {res['key']}: `boltbeam mods apply` offers it as it is (--rebuild builds it again)")
  elif res.get("chosen") and not res.get("refused"):
    out.append("`boltbeam mods apply` builds them into a clone (yours is never changed)")
  return "\n".join(out)


def modded(engine:str, path:str | None) -> dict[str, Any] | None:
  """The modded build a path belongs to (builds.json modded), for tune's words."""
  from boltbeam.collectors import engine_builds as eb
  real = os.path.realpath(path) if path else None
  return next((b for b in eb.entry(eb.load(), engine).get("modded") or []
               if real and real in {os.path.realpath(p) for p in (b.get("paths") or {}).values()}), None)


def found(engine:str, path:str | None, features) -> dict[str, Any] | None:
  """The mods a build has, from where it can be read: its build record (modded: a build `screen mods --apply` made, at any
  path the record names), else each mod's marker (index) among the names read from the build's files (features), as a
  record of the same shape marked found "markers" (the version with the most markers; the series' default labels are its
  mods, as an apply would choose; the others are named). None: no record and no marker, which is not proof of a stock
  build (a marker renamed by a later mods version reads as absent), so a reader says it and how to fix it."""
  from boltbeam.report import plain
  if rec := modded(engine, path):
    return rec
  names = set(features or ())
  hit = [m for m in mods(engine) if m.get("marker") and m["marker"] in names]
  if not any(re.fullmatch(r"[A-Z][A-Z0-9_]+", m["marker"]) for m in hit):  # a message alone may be the engine's own text: a switch name is the mod's
    return None
  if not hit:
    return None
  ver = max({m.get("version") for m in hit}, key=lambda v: (sum(m.get("version") == v for m in hit), _ver(v)))
  rows = [m for m in hit if m.get("version") == ver]
  ok = default_labels()
  return {"key": None, "found": "markers", "version": ver, "status": plain.words()["mods"]["found_markers"].format(n=len(rows), total=sum(m.get("version") == ver for m in mods(engine))),
          "mods": [{k: m.get(k) for k in ("id", "engine", "name", "switch", "output")} for m in rows if m.get("output") in ok],
          "other": [m["id"] for m in rows if m.get("output") not in ok]}


def not_here(engine:str, backend:str | None) -> str | None:
  """Why the engine's mods do nothing on this machine's GPU backend (the apply row's backends), or None: they apply, or
  the backend is not known."""
  row = pointer()["apply"].get(engine) or {}
  if not backend or not row.get("backends") or backend.lower() in {b.lower() for b in row["backends"]}:
    return None
  from boltbeam.report import plain
  return plain.words()["mods"]["backend"].format(engine=engine, what=row.get("backends_words") or "/".join(row["backends"]), backend=backend)


def key_of(got:dict[str, Any]) -> str:
  """The modded build's key for a plan's chosen set: its base, -copy for a taken tree (so it never replaces a checked
  checkout's build at the same base), and a hash of the chosen ids with their patches' and kernels' sha256 (index), so
  a commit that changes a patch under the same id builds anew and one that does not reuses the build."""
  by, chosen = {m["id"]: m for m in mods()}, set(got["chosen"])
  what = sorted([f"{i} {by.get(i, {}).get('sha256', '')}" for i in chosen] +
                [f"{k['id']} {sorted(k.get('files', {}).items())}" for k in index().get("kernels", []) if k.get("hook") in chosen])
  return f"{got['head'][:9]}{'-copy' * got['taken']}-mods-{hashlib.sha256(' '.join(what).encode()).hexdigest()[:8]}"


def run(engine:str, *, src:str | None = None, labels:tuple[str, ...] | None = None, only:list[str] | None = None,
        base:str | None = None, apply_them:bool = False, backend:str | None = None, rebuild:bool = False,
        pin:str | None = None, say:Callable[[str], None] = print) -> dict[str, Any]:
  """`screen mods`: the plan at the resolved commit (resolve; pin forces one), with its rows, the chosen set's key and
  whether that build is already on disk; with apply_them, the modded build, recorded (one already on disk is offered
  as it is unless rebuild). The commit read is said and kept as mods_commit."""
  if engine not in pointer()["apply"]:
    return {"refused": f"no applier is wired for {engine} mods yet (engine_patches.json apply)"}
  if why := not_here(engine, backend):
    return {"refused": why}
  try:
    at = resolve(say, pin)
  except (Refused, OSError, subprocess.SubprocessError) as exc:
    return {"refused": str(exc) if isinstance(exc, Refused) else f"{type(exc).__name__}: {exc}"}
  say(at["words"])
  with reading(at):
    got = _run(engine, src=src, labels=labels, only=only, base=base, apply_them=apply_them, backend=backend, rebuild=rebuild, say=say)
    if "fits" in got:
      got["rows"] = rows(engine, got, only)
  return {**got, "mods_commit": record(at)}


def _run(engine:str, *, src:str | None, labels:tuple[str, ...] | None, only:list[str] | None, base:str | None,
         apply_them:bool, backend:str | None, rebuild:bool, say:Callable[[str], None]) -> dict[str, Any]:
  from boltbeam.collectors import engine_builds as eb
  from boltbeam.target import vendors
  state = eb.load()
  cur = eb.entry(state, engine).get("current") or {}
  where = pathlib.Path(src).expanduser() if src else (pathlib.Path(cur["folder"]) / pointer()["apply"][engine]["source"] if cur.get("folder") else None)
  if where is None or not where.is_dir():
    return {"refused": f"no source tree for {engine}: pass --src (the source of the build you run)"}
  try:
    got = plan(engine, where, labels, only, base)
    say(f"mods {engine}: {got['fits']}")
    if got["chosen"]:
      key = key_of(got)
      have = next((b for b in built(engine, state) if b.get("key") == key), None)
      got = {**got, "key": key, "already_built": have is not None}
    if not apply_them:
      return got
    if not got["chosen"]:
      return {**got, "refused": "no mod to apply on this commit with the labels allowed"}
    if have and not rebuild:
      if (r := have.get("mods_resolved") or {}).get("commit"):  # the same patches by key; say the commit it was made from
        say(_why("built_from", commit=r["commit"][:9], how=r.get("how")))
      say(f"mods {engine}: these mods are already built as {key} in {have['folder']}; nothing was rebuilt (--rebuild builds it again)")
      return {**got, "build": have, "reused": True}
    into = pathlib.Path(state.get("root") or eb.state_file().with_name("engines")).expanduser() / engine / key
    say(f"mods {engine}: {len(got['chosen'])} mods onto a clone of {where} at {got['head'][:9]}, built in {into}")
    build = {**apply(engine, where, got["chosen"], into, got["head"], vendors.for_backend(backend), got["taken"]), "key": key,
             "mods_resolved": _AT.get("record")}
    for line in [f"commit {build['checks']['commit']}", *(f"kernel {x['id']}: {x['loaded']}" for x in build["checks"].get("kernels", []))]:
      say(f"mods {engine}: {line}")
  except (Refused, RuntimeError, OSError, subprocess.SubprocessError) as exc:  # engine_update.CannotInstall is a RuntimeError
    return {"refused": str(exc) if isinstance(exc, Refused) else f"{type(exc).__name__}: {exc}"}
  state = eb.load()  # read again: a build takes minutes, and another writer (an engine update, another apply) may have saved
  e = eb.entry(state, engine)
  e["modded"] = [b for b in e.get("modded") or [] if b.get("key") != key] + [build]
  eb.save(state)
  return {**got, "build": build, "already_built": True}


@functools.lru_cache(maxsize=1)
def _shapes() -> dict[str, Any]:
  return json.loads((DATA / "mod_shapes.json").read_text())["engines"]


def shape_why(engine:str | None, name:str | None, shape:dict[str, Any] | None) -> str | None:
  """Why a mod does nothing on this model and command, or None: its `needs` (data/mod_shapes.json) over the shape they
  show. The one reading of a mod's shape: a switch (withheld), a knob row of its group (tune.derive) and a spend row
  (spend.unfit) all ask here. A fact the shape could not read, or a mod with no row, is a reason too. shape None (a
  caller with no model read): nothing is checked."""
  if shape is None:
    return None
  from boltbeam.report import plain
  from boltbeam.workflow import tune
  w, row = plain.words()["mods"], (_shapes().get(engine) or {}).get(name)
  try:
    return w["shape_no_row"] if row is None else None if tune.evaluate(row["needs"], shape, set(shape)) else w["shape_lacks"].format(why=row["why"], needs=row["needs"])
  except tune.Missing as miss:
    return w["shape_unread"].format(fact=miss, needs=row["needs"])


def withheld(build:dict[str, Any] | None, shape:dict[str, Any] | None) -> dict[str, dict[str, str]]:
  """{switch: {mod, why}} for the build's mods with a switch whose shape this model and command do not show (shape_why)."""
  by = {m["id"]: m for m in mods()}
  rows = [{**by.get(b["id"], {}), **{k: v for k, v in b.items() if v is not None}} for b in (build or {}).get("mods") or []]
  return {switch(m): {"mod": m["id"], "why": why} for m in rows if switch(m) and (why := shape_why(m.get("engine"), m.get("name"), shape))}


def with_switches(shown:dict[str, Any], build:dict[str, Any] | None, shape:dict[str, Any] | None = None) -> dict[str, Any]:
  """A rendered command (env, argv, command) on a modded build with its mods' switches (run_env) set, as a run on it
  sets them; a switch the render already sets keeps its value. shape: the model's and the command's (withheld): a mod
  whose shape they do not show is not set, and the command says which and why (`withheld`)."""
  held = withheld(build, shape)
  if held:  # a held switch is on no command
    shown = {**shown, "withheld": held, "env": {k: v for k, v in (shown.get("env") or {}).items() if k not in held}}
  env = {k: v for k, v in run_env(build or {}).items() if k not in (shown.get("env") or {}) and k not in held}
  if "argv" not in shown or not (env or held):
    return shown
  env = {**env, **(shown.get("env") or {})}
  return {**shown, "env": env, "command": " ".join([*(f"{k}={v}" for k, v in env.items()), *map(str, shown["argv"])])}


# --- the screens' rows (screen mods, the Engine picker's modded fork, a run on it) ---------------------------------------

def run_on(m:dict[str, Any]) -> str | None:
  """The value a run on a modded build sets the mod's switch to: its `on` when that is one plain value (1). None when it
  is not one value a run can set alone (a manifest's path, a count to pick, `unset or 1`: on by default)."""
  sw = m.get("switch") or {}
  on = str(sw.get("on") or "")
  return on if sw.get("kind") == "env" and re.fullmatch(r"[A-Za-z0-9_.-]+", on) else None


def run_env(build:dict[str, Any]) -> dict[str, str]:
  """The switches a run on this modded build sets: each of its mods' (index rows by id) that run_on can set, and the
  kernel hook's (install_paths mod) set to the manifest the build picked (kernel_manifest), else its one manifest."""
  by = {m["id"]: m for m in mods()}  # the build's own record wins: it may come from a commit past the pin (resolve)
  rows = [{**by.get(b["id"], {}), **{k: v for k, v in b.items() if v is not None}} for b in build.get("mods") or []]
  plan = pointer().get("plan_only") or {}  # set by a plan that spends what they free, never by default
  out = {switch(m): on for m in rows if (on := run_on(m)) and f"{m.get('engine')}/{m.get('name')}" not in plan}
  hook = next((m for m in rows if m.get("name") and m["name"] == (pointer()["install_paths"].get(m.get("engine")) or {}).get("mod")), None)
  got = [p for p in [build.get("kernel_manifest")] if p and os.path.isfile(p)] or [p for p in build.get("kernel_manifests") or [] if os.path.isfile(p)]
  if hook and switch(hook) and len(got) == 1:  # the build's pick (kernel_manifest), else its one manifest
    out[switch(hook)] = got[0]
  return out


@contextlib.contextmanager
def switched(env:dict[str, str]):
  """These switches set in this process (and the engines it starts) for the block; one the user set stays theirs."""
  sets = {k: v for k, v in env.items() if k not in os.environ}
  try:
    os.environ.update(sets)
    yield
  finally:
    for k in sets:
      os.environ.pop(k, None)


def rows(engine:str, got:dict[str, Any], only:list[str] | None = None) -> list[dict[str, Any]]:
  """Every mod the plan weighed, in series order, as the screens show it: what it changes, its switch, its output label
  in words, and chosen or not with why (data/plain_words.json mods). only: the --mod ids asked, when given."""
  from boltbeam.report import plain
  w = plain.words()["mods"]
  ids, ok = {*got["chosen"], *got["conflicts"], *got["missing_dependencies"], *(n.split(" ")[0] for n in got["not_chosen"])}, verified()
  out = []
  for m in mods(engine):
    if m["id"] not in ids:
      continue
    sw, on, plan = m.get("switch") or {}, run_on(m), f"{engine}/{m['name']}" in (pointer().get("plan_only") or {})
    hooked = m["name"] == (pointer()["install_paths"].get(engine) or {}).get("mod")
    if m["id"] in got["chosen"] and m["id"] in (got.get("quality_gated") or {}):
      why = w["chosen_gated"].format(evidence=got["quality_gated"][m["id"]][1] or w["no_evidence"])
    elif m["id"] in got["chosen"]:
      why = w["chosen"]
    elif m["id"] in got["conflicts"]:
      why = w["conflict"].format(commit=got["head"][:9])
    elif m["id"] in got["missing_dependencies"]:
      why = w["needs"].format(needs=", ".join(got["missing_dependencies"][m["id"]]))
    elif m["id"] not in ok:
      why = w["unverified"].format(engine=m["engine"], version=m["version"])
    elif only and m["id"] not in only:
      why = w["not_asked"]
    elif m["state"] != "series":
      why = w["parked"].format(state=m["state"])
    else:
      why = w["label"].format(label=m["output"], labels=" or ".join(got.get("labels") or default_labels()))
    files, title = m.get("files") or [], m.get("title")
    out.append({"id": m["id"], "name": m["name"], "chosen": m["id"] in got["chosen"], "why": why,
                "about": w["about_module" if m.get("module") and title else "about_titled" if title else "about"].format(title=title, module=m.get("module"), files=", ".join(files) if len(files) <= 2 else f"{len(files)} files ({', '.join(files[:2])}, …)",
                                           added=m["lines"]["added"], removed=m["lines"]["removed"]),
                "switch": f"{sw['name']}={on}" if on and not plan else sw.get("name"),
                "switch_why": m.get("switch_null_why") if not sw else w["switch_plan"].format(switch=sw["name"], on=on) if plan
                              else None if on else w["switch_manifest"] if hooked else
                              w["switch_default_on"] if sw.get("default") == "on" else w["switch_unset"].format(on=sw.get("on")),
                "output": m["output"], "output_words": w["output"].get(m["output"], m["output"])})
  return out


def built(engine:str, state:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """The modded builds recorded for the engine (builds.json modded) whose folder and paths are all on disk."""
  from boltbeam.collectors import engine_builds as eb
  return [b for b in eb.entry(state if state is not None else eb.load(), engine).get("modded") or []
          if b.get("folder") and os.path.isdir(b["folder"]) and all(os.path.exists(p) for p in (b.get("paths") or {}).values())]


def with_forks(families:list[dict[str, Any]], provider_rows:list[dict[str, Any]], state:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """engine_families.rows with each family's root engine's modded builds after its forks (screen providers)."""
  by = {r["provider"]: r for r in provider_rows}
  return [{**fam, "forks": fam["forks"] + fork_rows(fam["family"], by[fam["family"]], state)} if fam["family"] in by else fam
          for fam in families]


def fork_rows(engine:str, provider_row:dict[str, Any] | None, state:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """Each modded build of the engine as a fork of its family: run by the engine's own adapter, its program the path
  tune and the engine picker name; offered only when the engine runs here."""
  from boltbeam.collectors import engine_builds as eb
  from boltbeam.report import plain
  w, got = plain.words()["mods"], built(engine, state)
  try:
    prog = eb.row(engine)["version"]["from"]
  except (ValueError, KeyError):
    prog = None
  if not (provider_row or {}).get("available"):  # the engine does not run here: nothing to offer its builds on
    return []
  return [{"fork": b["key"], "label": w["fork"] + (f" {b['key']}" if len(got) > 1 else ""), "repo": pointer()["mods_repo"]["url"],
           "provider": engine, "runs": True, "state": "available", "fix": None, "modded": b["key"],
           "path": (b.get("paths") or {}).get(prog) or next(iter((b.get("paths") or {}).values()), None),
           "reason": w["fork_reason"].format(n=len(b.get("mods") or []), status=b.get("status") or "applied, not verified",
                                             at=b.get("installed_at") or "?")}
          for b in got]
