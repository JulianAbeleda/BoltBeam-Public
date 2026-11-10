"""`boltbeam fetch`: bring finished runs home from a remote machine (a rented pod, a Linux box) over SSH.

    list        one ssh call runs a small inline python3 on the remote (BoltBeam need not be installed there). It looks
                for run folders (run_manifest.json) in the remote runs root: <root>/<id>, <root>/.work/<id> and
                <root>/saved/<id> (screen.run_places' layout), and reports each one's id, model, chip, GPU, date, size
                and every file's size and sha256. Nothing is written remotely; nothing outside the root is read
                (symlinks are not followed, and a symlinked file is left behind and named).
    root        --root, else the first that exists of REMOTE_ROOTS: tools/pod-setup.sh's runs folder ($WORK/runs, WORK
                /root/boltbeam today, /workspace/boltbeam before), then ~/boltbeam-runs (the Homebrew wrapper's
                default). The answer says which was used and what was tried.
    copy        rsync (when both ends have it and the remote path needs no shell quoting) or tar over ssh, of exactly the
                listed files: raw traces (RAW_TRACES) stay behind unless --with-traces. The copy lands in a hidden
                staging folder beside the destination and moves into place only after every file's size and sha256
                equal the remote listing; any difference refuses the run and removes the local staging folder.
    where       <local runs root>/saved/<id> (BOLTBEAM_SAVED when set and no --into): a fetched run is a finished
                result to keep and compare, which is what a saved run is; .work is the screens' scratch area that
                `clean-work` empties. save.json and results.json are written when the run brought none, as `screen save`
                does, so the Saved runs page lists it with its share of the limit (fetched.json's `added` names them).
    again       a run already here whose files equal the remote's is skipped and said so; a different run under the
                same id is refused, never overwritten (fetch it with --into another runs folder, or move the local one).
    chip        a run's chip profile, when the chip is in neither this machine's registry nor its own profiles (an H100
                autoscan profiled on the pod): read by the same lister from CHIP_PLACES only, the pod's chips folder
                beside its runs root ($WORK/chips, BOLTBEAM_CHIPS_DIR in pod-setup) or ~/.boltbeam/chips, file
                <target_id>.json and nothing else; carried in the listing with its sha256, checked here, and kept
                under <chips>/fetched/<host>/ (targets.save_fetched_profile: namespaced, never over this machine's own
                profile, never matched to this machine's GPU). Then results.json is computed against it.
    provenance  fetched.json (schemas/fetched.schema.json) in each fetched run: where it came from (dest, host, port,
                remote path), when (UTC), its chip and GPU from the run's own facts, this BoltBeam's version, the
                transport, the files checked and the ones left behind.
    safety      only ssh and rsync to DEST; BatchMode=yes and ConnectTimeout (policy fetch_ssh_connect_timeout_s) so a
                missing key fails at once instead of prompting; nothing on the remote is deleted or written, and
                nothing is read outside the runs root but the run's own chip profile file (chip, above).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tarfile
from typing import Any

from boltbeam.core import policy

FETCH_RECORD = "fetched.json"
SCHEMA_FETCHED = "boltbeam.fetched.v1"
REMOTE_ROOTS = ("/root/boltbeam/runs", "/workspace/boltbeam/runs", "~/boltbeam-runs")
RAW_TRACES = ("*.nsys-rep", "*.qdstrm", "*.qdrep", "*.sqlite", "cuda_gpu_trace.csv")  # large raw captures
SAFE_PATH = re.compile(r"[\w./+-]+")  # a remote path rsync can be handed without any shell quoting

# Runs on the remote as `python3 - WITH_TRACES ROOT...`: reads only under the first root that exists, writes nothing.
LISTER = r'''
import fnmatch, hashlib, json, os, shutil, sys, time
traces, roots = sys.argv[1] == "1", sys.argv[2:]
PATTERNS = %(patterns)r
def kept(name): return traces or not any(fnmatch.fnmatch(name, p) for p in PATTERNS)
def sha(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
  return h.hexdigest()
def facts(run):
  out = {}
  for name in ("run_manifest.json", "machine_facts.json"):
    try:
      with open(os.path.join(run, name)) as f: out[name] = json.load(f)
    except (OSError, ValueError): out[name] = {}
  m, gpus = out["run_manifest.json"], out["machine_facts.json"].get("gpus") or [{}]
  return {"model_id": m.get("model_id"), "target_id": m.get("target_id"), "workload": m.get("workload"),
          "gpu": gpus[0].get("name"), "gpus": len(out["machine_facts.json"].get("gpus") or []) or None}
def chip(root, tid):
  if not tid or os.path.basename(tid) != tid or tid.startswith("."): return None
  for d in (os.path.join(os.path.dirname(root), "chips"), os.path.expanduser("~/.boltbeam/chips")):
    p = os.path.join(d, tid + ".json")
    if os.path.isfile(p) and not os.path.islink(p):
      with open(p, "rb") as f: b = f.read()
      return {"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest(), "text": b.decode("utf-8", "replace")}
  return None
def run(root, rel):
  path, files, left = os.path.join(root, rel), [], []
  for d, dirs, names in os.walk(path):
    dirs[:] = sorted(x for x in dirs if not os.path.islink(os.path.join(d, x)))
    for n in sorted(names):
      p = os.path.join(d, n); r = os.path.relpath(p, path)
      if os.path.islink(p) or not os.path.isfile(p): left.append({"path": r, "size": None, "why": "not a regular file"})
      elif not kept(n): left.append({"path": r, "size": os.path.getsize(p), "why": "raw trace"})
      else: files.append({"path": r, "size": os.path.getsize(p), "sha256": sha(p)})
  when = time.strftime("%%Y-%%m-%%dT%%H:%%M:%%SZ", time.gmtime(os.path.getmtime(os.path.join(path, "run_manifest.json"))))
  f = facts(path)
  return {"id": os.path.basename(rel), "where": rel, "date": when, "files": files, "left_behind": left,
          "chip_profile": chip(root, f["target_id"]), **f}
tried, found = [], None
for r in roots:
  a = os.path.abspath(os.path.expanduser(r)); tried.append(a)
  if os.path.isdir(a): found = a; break
runs = []
if found:
  for place in ("", ".work", "saved"):
    base = os.path.join(found, place)
    if not os.path.isdir(base) or os.path.islink(base) and place: continue
    for n in sorted(os.listdir(base)):
      rel = os.path.join(place, n) if place else n
      if not n.startswith(".") and not os.path.islink(os.path.join(base, n)) and os.path.isfile(os.path.join(base, n, "run_manifest.json")):
        runs.append(run(found, rel))
print(json.dumps({"root": found, "tried": tried, "rsync": bool(shutil.which("rsync")), "tar": bool(shutil.which("tar")), "runs": runs}))
''' % {"patterns": RAW_TRACES}


class Refused(Exception):
  """The fetch cannot go on: ssh failed, no runs root, an unknown id, a copy that does not match, a same-id conflict."""


class Ssh:
  """The only way this module reaches DEST: ssh (and rsync over the same ssh options). Tests replace argv building."""

  def __init__(self, dest:str, port:int | None = None):
    if not dest or dest.startswith("-") or any(c.isspace() for c in dest):
      raise Refused(f"DEST must be [user@]host: {dest!r}")
    self.dest, self.port = dest, port

  @property
  def host(self) -> str:
    return self.dest.rsplit("@", 1)[-1]

  def opts(self) -> list[str]:
    return ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(policy.value('fetch_ssh_connect_timeout_s'))}",
            *(["-p", str(self.port)] if self.port else [])]

  def ssh(self, remote:str) -> list[str]:
    return ["ssh", *self.opts(), self.dest, remote]

  def rsync(self, remote_dir:str, into:pathlib.Path) -> list[str]:
    return ["rsync", "-a", "--files-from=-", "-e", shlex.join(["ssh", *self.opts()]), f"{self.dest}:{remote_dir}/", f"{into}/"]

  def call(self, argv:list[str], data:bytes = b"", out=None) -> bytes:
    p = subprocess.run(argv, input=data, stdout=out if out is not None else subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
      err = (p.stderr or b"").decode(errors="replace").strip().splitlines()[-3:]
      raise Refused(f"{argv[0]} to {self.dest} failed (exit {p.returncode}): {' / '.join(err) or 'no message'}")
    return p.stdout or b""


def listing(ssh:Ssh, root:str | None = None, *, with_traces:bool = False) -> dict[str, Any]:
  """The remote runs root and its runs, from one ssh call (LISTER). Refused when no root exists."""
  roots = [root] if root else list(REMOTE_ROOTS)
  cmd = "python3 - " + shlex.join(["1" if with_traces else "0", *roots])
  raw = ssh.call(ssh.ssh(cmd), LISTER.encode())
  try:
    got = json.loads(raw)
  except ValueError as exc:
    raise Refused(f"the listing from {ssh.dest} is not JSON ({exc}): is python3 on the remote?") from exc
  if not got.get("root"):
    raise Refused(f"no runs folder on {ssh.dest}: tried {', '.join(got.get('tried') or roots)} (pass --root)")
  for r in got["runs"]:  # the remote's answer names local paths: a name that climbs out is refused, not followed
    if r["id"] != pathlib.PurePosixPath(r["id"]).name or r["id"].startswith("."):
      raise Refused(f"the listing from {ssh.dest} names the run {r['id']!r}, not a folder name")
    for name in (f["path"] for f in r["files"]):
      if os.path.isabs(name) or ".." in pathlib.PurePosixPath(name).parts:
        raise Refused(f"the listing from {ssh.dest} names {name!r}, not a plain path inside a run")
  kept = {r["id"] for r in got["runs"] if r["where"].startswith("saved/")}  # a saved run's scratch copy in .work is the
  got["runs"] = [r for r in got["runs"] if not (r["where"].startswith(".work/") and r["id"] in kept)]  # same run, unsaved
  return got


def _sha(p:pathlib.Path) -> str:
  h = hashlib.sha256()
  with p.open("rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
      h.update(b)
  return h.hexdigest()


def differs(files:list[dict[str, Any]], here:pathlib.Path) -> list[str]:
  """The listed files whose copy under `here` is missing or not the same size and sha256 (size first, so a cheap
  difference never hashes)."""
  bad = []
  for f in files:
    p = here / f["path"]
    if not p.is_file() or p.is_symlink() or p.stat().st_size != f["size"] or _sha(p) != f["sha256"]:
      bad.append(f["path"])
  return bad


def local_roots(into:str | None, env=os.environ) -> tuple[pathlib.Path, pathlib.Path]:
  """The local runs root and its saved folder, as the screen resolves them: --into, else BOLTBEAM_RUNS, else the
  checkout's runs/ (tui/main.go); saved is BOLTBEAM_SAVED (only without --into), else <root>/saved."""
  from boltbeam.cli.self_update import ROOT
  root = pathlib.Path(into or env.get("BOLTBEAM_RUNS") or ROOT / "runs").expanduser().resolve()
  saved = env.get("BOLTBEAM_SAVED") if not into else None
  return root, pathlib.Path(saved).expanduser().resolve() if saved else root / "saved"


def state(run:dict[str, Any], saved:pathlib.Path) -> str:
  """here (same content already in saved/), conflict (a different run under the id) or new."""
  d = saved / run["id"]
  if not d.exists():
    return "new"
  return "conflict" if differs(run["files"], d) else "here"


def rows(got:dict[str, Any], saved:pathlib.Path) -> list[dict[str, Any]]:
  return [{"id": r["id"], "where": r["where"], "model_id": r["model_id"], "target_id": r["target_id"], "gpu": r["gpu"],
           "date": r["date"], "bytes": sum(f["size"] for f in r["files"]),
           "left_behind_bytes": sum(f["size"] or 0 for f in r["left_behind"]), "state": state(r, saved)}
          for r in got["runs"]]


def _pull(ssh:Ssh, got:dict[str, Any], run:dict[str, Any], into:pathlib.Path) -> str:
  """Copy the run's listed files into `into`: rsync when both ends have it and the path is plain, else tar over ssh,
  read here member by member (only regular files the listing named; nothing is written outside `into`)."""
  remote = f"{got['root']}/{run['where']}"
  names = "\n".join(f["path"] for f in run["files"]).encode() + b"\n"
  if got.get("rsync") and shutil.which("rsync") and SAFE_PATH.fullmatch(remote):
    ssh.call(ssh.rsync(remote, into), names)
    return "rsync"
  if not got.get("tar"):
    raise Refused(f"neither rsync nor tar on {ssh.dest}")
  wanted, tmp = {f["path"] for f in run["files"]}, into.parent / "stream.tar"
  with tmp.open("wb") as out:  # COPYFILE_DISABLE: a Mac's tar would add ._ AppleDouble members
    ssh.call(ssh.ssh("COPYFILE_DISABLE=1 tar -C " + shlex.quote(remote) + " -cf - -T -"), names, out)
  with tarfile.open(tmp) as tf:
    for m in tf:
      name = os.path.normpath(m.name)
      if not m.isfile() or name not in wanted:
        raise Refused(f"the tar stream from {ssh.dest} holds {m.name!r}, which the listing did not name")
      dst = into / name
      dst.parent.mkdir(parents=True, exist_ok=True)
      with tf.extractfile(m) as src, dst.open("wb") as f:
        shutil.copyfileobj(src, f)
  tmp.unlink()
  return "tar"


def chip_profile(ssh:Ssh, run:dict[str, Any]) -> dict[str, Any]:
  """Make the run's chip a target here when this machine does not know it: the remote's profile, checked by sha256,
  kept under the host's own folder. Refused when the profile arrived different from what the remote hashed."""
  from boltbeam.target import targets as reg
  tid, got = run.get("target_id"), run.get("chip_profile")
  if not tid:
    return {"target_id": None, "used": None}
  if tid in reg.TARGETS and not reg.is_fetched(tid):
    return {"target_id": tid, "used": "this machine's" if reg.is_local(tid) else "registry"}
  if not got:
    return {"target_id": tid, "used": "fetched" if tid in reg.TARGETS else None,
            "why": None if tid in reg.TARGETS else f"no {tid}.json on {ssh.dest} beside the runs root or in ~/.boltbeam/chips"}
  raw = got["text"].encode()
  if len(raw) != got["size"] or hashlib.sha256(raw).hexdigest() != got["sha256"]:
    raise Refused(f"the chip profile {got['path']} arrived different from the remote's sha256; not fetched")
  host = ssh.host + (f"_{ssh.port}" if ssh.port else "")
  try:
    path = reg.save_fetched_profile(raw, host)
  except ValueError as exc:
    raise Refused(f"{got['path']} on {ssh.dest} is not a chip profile ({exc}); not fetched") from exc
  kept = _sha(path) == got["sha256"]
  return {"target_id": tid, "used": "fetched", "path": str(path), "remote_path": got["path"], "sha256": got["sha256"],
          **({} if kept else {"why": f"{path} was already here with other content and is kept; this run reads it"})}


def record(ssh:Ssh, got:dict[str, Any], run:dict[str, Any], transport:str, with_traces:bool) -> dict[str, Any]:
  from boltbeam.cli.self_update import installed
  return {"schema": SCHEMA_FETCHED, "id": run["id"],
          "source": {"dest": ssh.dest, "host": ssh.host, "port": ssh.port, "path": f"{got['root']}/{run['where']}"},
          "fetched_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "target_id": run["target_id"], "gpu": run["gpu"], "gpus": run["gpus"], "boltbeam_version": installed(),
          "transport": transport, "traces": "included" if with_traces else "left behind",
          "files": run["files"], "left_behind": run["left_behind"], "chip_profile": chip_profile(ssh, run)}


def fetch_one(ssh:Ssh, got:dict[str, Any], run:dict[str, Any], saved:pathlib.Path, *, with_traces:bool,
              shadows:list[pathlib.Path] = ()) -> dict[str, Any]:
  """One run into saved/<id>: skipped when here, refused on a same-id conflict or any file that does not match."""
  from boltbeam.core.canonical import pretty_json
  from boltbeam.workflow.results import results
  from boltbeam.workflow.run_folder import SAVE_RECORD
  dest, st = saved / run["id"], state(run, saved)
  if st == "here":
    return {"id": run["id"], "status": "skipped", "dir": str(dest), "why": "already here with the same files"}
  if st == "conflict":
    raise Refused(f"{dest} holds a different run with the id {run['id']}; not overwritten: fetch with --into another "
                  "runs folder, or move the local one")
  stage = saved / f".fetching-{run['id']}"
  shutil.rmtree(stage, ignore_errors=True)
  (stage / run["id"]).mkdir(parents=True)
  try:
    transport = _pull(ssh, got, run, stage / run["id"])
    if bad := differs(run["files"], stage / run["id"]):
      raise Refused(f"{run['id']}: {len(bad)} file(s) differ from the remote after the copy ({', '.join(bad[:5])}); "
                    "not fetched")
    rec, here = record(ssh, got, run, transport, with_traces), stage / run["id"]
    before = {p for p in here.rglob("*") if p.is_file()}
    if not (here / SAVE_RECORD).exists():  # the Saved runs page lists runs that carry save.json
      save = {"saved_at": datetime.datetime.now().isoformat(timespec="seconds"), "from": f"{ssh.dest}:{rec['source']['path']}"}
      (here / SAVE_RECORD).write_text(json.dumps(save, indent=2) + "\n")
    if not (here / "results.json").exists():  # what `screen save` writes, so the list shows the share of the limit
      try:
        (here / "results.json").write_text(pretty_json(results(here)))
      except (Exception, SystemExit) as exc:  # a chip not registered here, say: the run is still fetched as it was
        rec["results_not_written"] = str(exc)
    rec["added"] = sorted([FETCH_RECORD, *(str(p.relative_to(here)) for p in here.rglob("*") if p.is_file() and p not in before)])
    (here / FETCH_RECORD).write_text(json.dumps(rec, indent=2) + "\n")
    os.replace(here, dest)
  finally:
    shutil.rmtree(stage, ignore_errors=True)
  first = [str(p) for p in shadows if (p / "run_manifest.json").exists()]
  return {"id": run["id"], "status": "fetched", "dir": str(dest), "transport": transport, "files": len(run["files"]),
          "bytes": sum(f["size"] for f in run["files"]), "left_behind": len(run["left_behind"]), "chip": rec["chip_profile"],
          "results": "results_not_written" not in rec, "warning": f"{', '.join(first)} has the same id and is found first "
          "when the run is opened by id (screen.find_run)" if first else None}


def fetch(ssh:Ssh, ids:list[str], saved:pathlib.Path, *, root:str | None = None, every:bool = False,
          with_traces:bool = False, here:pathlib.Path | None = None) -> dict[str, Any]:
  """List the remote's runs; with ids or every, fetch them. A refused run is reported and the rest go on."""
  got = listing(ssh, root, with_traces=with_traces)
  out = {"schema": SCHEMA_FETCHED, "kind": "fetch", "dest": ssh.dest, "remote_root": got["root"],
         "tried": got["tried"], "saved": str(saved), "runs": rows(got, saved), "fetched": []}
  if not (ids or every):
    return out
  by_id = {r["id"]: r for r in got["runs"]}
  if missing := [i for i in ids if i not in by_id]:
    raise Refused(f"no run {', '.join(missing)} under {got['root']} on {ssh.dest}")
  saved.mkdir(parents=True, exist_ok=True)
  for r in (got["runs"] if every else [by_id[i] for i in ids]):
    try:
      shadows = [here / r["id"], here / ".work" / r["id"]] if here else []  # run_places before saved
      out["fetched"].append(fetch_one(ssh, got, r, saved, with_traces=with_traces, shadows=shadows))
    except Refused as exc:
      out["fetched"].append({"id": r["id"], "status": "refused", "why": str(exc)})
  return out


def provenance(run:pathlib.Path) -> dict[str, Any] | None:
  """What the screen shows of a fetched run: the host (or chip) it came from and when; None for a run made here."""
  try:
    rec = json.loads((run / FETCH_RECORD).read_text())
  except (OSError, ValueError):
    return None
  if rec.get("schema") != SCHEMA_FETCHED:
    return None
  return {"host": (rec.get("source") or {}).get("host"), "chip": rec.get("target_id"), "gpu": rec.get("gpu"),
          "fetched_at": rec.get("fetched_at")}
