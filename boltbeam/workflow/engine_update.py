"""`screen engines --update`: look at each engine's upstream releases and install a newer one beside the builds here.

This is the one place BoltBeam installs or deletes an engine build (coding-principles: contain dangerous power). The
pipeline and the collectors never install; they read the current build (collectors/engine_builds.py). It runs only
when asked: the TUI Engine picker's "Upgrade <engine> to <release>" (Upgrade engines), or `pipeline --upgrade-engines`.

    look      the release table's source (data/engine_releases.json), short timeout, at most once per
              engine_latest_max_age_h (data/policy.json); a failed look is "could not check" and installs nothing
    seed      the first time, the build already on this machine (env var, PATH, the engine scan) is recorded as
              current, with the version and commit it reports; it is the user's and never deleted
    fit       a release that declares requirements (TensorRT-LLM: Python, torch, CUDA) is checked against this machine
              from its published metadata first; one that does not fit, or whose requirement cannot be read, is not
              installed and is recorded as such
    install   a newer release goes into its own folder, <root>/<engine>/<version>, never over a build:
              llama.cpp   git clone at the release tag (full history, no trees: the build number is counted), cmake
                          with the vendor row's flags (target/vendors.py ggml_cmake, ggml_arch), the bench targets
              vllm, tensorrt-llm   a venv named by the version, the table's pip steps
              ollama      the release archive for this platform, unpacked
              and is recorded as pending; the pipeline's next --analyze run compares it once with current
    prune     after a comparison that succeeded: the builds older than the previous one that BoltBeam installed
"""
from __future__ import annotations

import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Callable

from boltbeam.collectors import engine_builds as eb


class CannotInstall(RuntimeError):
  """This machine cannot install that build: the reason, said before anything is downloaded."""


class NotCompatible(CannotInstall):
  """The release's own requirements do not fit this machine, or cannot be read (unknown is not pass)."""


def sh(argv:list[str], log:pathlib.Path, *, env:dict[str, str] | None = None, timeout:float = 7200.0) -> str:
  """Run one install command, its output appended to the build's install.log; raises naming the command and its end."""
  proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open("a") as f:
    f.write(f"$ {' '.join(argv)}\n{proc.stdout}{proc.stderr}\n")
  if proc.returncode != 0:
    raise RuntimeError(f"{pathlib.Path(argv[0]).name} {argv[1] if len(argv) > 1 else ''} exited {proc.returncode}: "
                       f"{(proc.stderr or proc.stdout).strip()[-300:]} (log {log})")
  return proc.stdout


def probe(engine:str, path:str, run:Callable[..., Any] = subprocess.run) -> dict[str, str | None]:
  """What a build says it is: {version, commit}, each None when it does not say."""
  v = eb.row(engine)["version"]
  try:
    proc = run([a.format(path=path) for a in v["argv"]], capture_output=True, text=True, timeout=60)
    text = proc.stdout + proc.stderr
  except (OSError, subprocess.SubprocessError):
    text = ""
  first = lambda pats: next((m.group(1) for p in pats if (m := re.search(p, text))), None)  # noqa: E731
  return {"version": first(v["pattern"]), "commit": first(v.get("commit") or [])}


def here(engine:str, run:Callable[..., Any] = subprocess.run) -> dict[str, Any] | None:
  """The build already on this machine, as the finders see it (env var, then the engine scan), with the version and
  commit it reports; None when the engine is not here. folder None: BoltBeam did not install it."""
  from boltbeam.collectors import engine_scan
  r, found = eb.row(engine), {}
  for var in r["paths"]:
    path = os.environ.get(var) or ((engine_scan.cached_or_scan().get("found") or {}).get(var) or {}).get("path")
    if path:
      found[var] = path
  if r["version"]["from"] not in found:
    return None
  return {**probe(engine, found[r["version"]["from"]], run), "folder": None, "paths": found, "installed_at": None}


def fits(engine:str, version:str, vendor, read:Callable[[str], Any] | None = None, *,
         python:str | None = None, machine:str | None = None, driver:str | None = "read") -> dict[str, str]:
  """Does this release's own published metadata fit this machine: what the install steps fill in ({torch_spec},
  {torch_index}). Raises NotCompatible with the reason; a requirement that cannot be read is not a pass."""
  from boltbeam.collectors import providers, toolchain
  spec = eb.row(engine)["install"]
  req = spec.get("requires")
  if vendor.backend not in spec.get("backends", [vendor.backend]):
    raise NotCompatible(f"BoltBeam installs this engine for {' or '.join(spec['backends'])} GPUs; this one is {vendor.backend}")
  if not req:
    return {}
  read = read or providers.upstream
  python = python or f"{sys.version_info[0]}.{sys.version_info[1]}"
  machine = (machine or platform.machine()).lower()
  try:
    meta = read(req["metadata"].format(version=version))["info"]
    index = str(read(eb.row(engine)["url"]))
  except (*providers.upstream_errors(), AttributeError) as exc:
    raise NotCompatible(f"its requirements could not be read ({type(exc).__name__}: {exc})") from exc
  if not meta.get("requires_python") or eb.satisfies(python, meta["requires_python"]) is None:
    raise NotCompatible(f"its Python requirement could not be read ({meta.get('requires_python')!r})")
  if not eb.satisfies(python, meta["requires_python"]):
    raise NotCompatible(f"it needs Python {meta['requires_python']}; BoltBeam runs on {python}")
  tag = "cp" + python.replace(".", "")
  if not any(t == tag and p.endswith(machine) for t, p in eb.wheels(index, eb.row(engine)["dist"], version)):
    raise NotCompatible(f"{eb.row(engine)['dist']} {version} publishes no wheel for Python {python} on {machine}")
  torch_spec, cuda_spec = (eb.requirement(meta.get("requires_dist"), req[k]) for k in ("torch", "cuda"))
  cuda = re.search(r"(?:>=|==|~=)\s*(\d+)", cuda_spec or "")
  if not torch_spec or not cuda or cuda.group(1) not in req["cuda_driver"]:
    raise NotCompatible(f"its torch ({torch_spec}) or CUDA ({req['cuda']}{cuda_spec or ''}) requirement could not be read")
  need = req["cuda_driver"][cuda.group(1)]
  driver = toolchain.driver_version(vendor.vendor) if driver == "read" else driver
  major = int(driver.split(".")[0]) if driver and driver.split(".")[0].isdigit() else None
  if major is None:
    raise NotCompatible(f"it needs CUDA {cuda.group(1)} (a driver of {need} or newer); this machine's driver could not be read")
  if major < need:
    raise NotCompatible(f"it needs CUDA {cuda.group(1)}, a driver of {need} or newer; this machine has {driver}")
  return {"torch_spec": torch_spec, "torch_index": req["torch_index"][cuda.group(1)]}


def _arch(vendor) -> str:
  if not vendor.ggml_arch:
    return ""
  argv, pattern = vendor.ggml_arch
  m = re.search(pattern, subprocess.run(list(argv), capture_output=True, text=True, timeout=30).stdout)
  if not m:
    raise CannotInstall(f"{argv[0]} did not say this GPU's architecture")
  return "".join(m.groups())


def _llama(spec:dict[str, Any], version:str, folder:pathlib.Path, vendor, log:pathlib.Path, fill:dict[str, str]) -> None:
  from boltbeam.target import toolkits
  if not vendor.ggml_cmake:
    raise CannotInstall(f"BoltBeam has no llama.cpp build recipe for {vendor.vendor} GPUs (target/vendors.py ggml_cmake)")
  env = {}
  if vendor.toolkit and (tool := toolkits.tool(vendor.toolkit.tools[0], shutil.which)):  # cmake finds the compiler on PATH
    env["PATH"] = os.pathsep.join([str(pathlib.Path(tool).parent), os.environ.get("PATH", "")])
  src, build = folder / "src", folder / "build"
  if not (src / ".git").is_dir():
    sh(["git", "clone", *spec["clone"], "--branch", spec["ref"].format(version=version), spec["repo"], str(src)], log)
  flags = [f.format(arch=_arch(vendor)) if "{arch}" in f else f for f in vendor.ggml_cmake]
  sh(["cmake", "-S", str(src), "-B", str(build), *spec["cmake"], *flags], log, env=env)
  sh(["cmake", "--build", str(build), "--config", "Release", "--parallel", str(os.cpu_count() or 1),
      "--target", *spec["targets"]], log, env=env)


def _venv(spec:dict[str, Any], version:str, folder:pathlib.Path, vendor, log:pathlib.Path, fill:dict[str, str]) -> None:
  from boltbeam.collectors import toolchain
  driver = toolchain.driver_version(vendor.vendor)
  major = int(driver.split(".")[0]) if driver and driver.split(".")[0].isdigit() else None
  venv = folder / "venv"
  py = str(venv / "bin" / "python")
  if not pathlib.Path(py).is_file():
    sh([sys.executable, "-m", "venv", str(venv)], log)
  steps = [list(s) for s in spec["steps"]]
  if (older := spec.get("older_driver")) and major is not None and major < older[0]:
    steps[-1] += older[1]
  for step in steps:
    fill = {**fill, "version": version}
    if any("{torch}" in a for a in step):
      fill["torch"] = sh([py, "-c", "import torch; print(torch.__version__)"], log).strip()
    sh([py, "-m", "pip", "install", "-q", "--no-cache-dir", *[a.format(**fill) for a in step]], log)
  sh([py, "-c", spec["check"]], log)


def _archive(spec:dict[str, Any], version:str, folder:pathlib.Path, vendor, log:pathlib.Path, fill:dict[str, str]) -> None:
  import urllib.request
  where = f"{platform.system().lower()}-{platform.machine().lower()}"
  asset = spec["assets"].get(where)
  if asset is None:
    raise CannotInstall(f"no release archive for {where} in data/engine_releases.json")
  folder.mkdir(parents=True, exist_ok=True)
  with urllib.request.urlopen(spec["url"].format(version=version, asset=asset), timeout=60) as got, (folder / asset).open("wb") as f:
    shutil.copyfileobj(got, f)
  sh(["tar", "-xf", str(folder / asset), "-C", str(folder)], log)
  (folder / asset).unlink()


INSTALL = {"llama": _llama, "venv": _venv, "archive": _archive}


def install(engine:str, version:str, folder:pathlib.Path, vendor, fill:dict[str, str] | None = None) -> dict[str, Any]:
  """One release into its own folder (a rerun after a failure reuses what it finished), marked as BoltBeam's; the
  build record: version, commit, folder, and the path each env var selects."""
  r = eb.row(engine)
  log = folder / "install.log"
  folder.mkdir(parents=True, exist_ok=True)
  (folder / eb.MARK).write_text(f"{engine} {version}\n")
  INSTALL[r["install"]["kind"]](r["install"], version, folder, vendor, log, fill or {})
  paths = {}
  for var, rel in r["paths"].items():
    got = next((folder / p for p in ([rel] if isinstance(rel, str) else rel) if (folder / p).exists()), None)
    if got is not None:
      paths[var] = str(got)
  if r["version"]["from"] not in paths:
    raise RuntimeError(f"{engine} {version} installed, but {r['paths'][r['version']['from']]} is not in {folder} (log {log})")
  return {"version": version, "commit": probe(engine, paths[r["version"]["from"]])["commit"], "folder": str(folder),
          "paths": paths, "installed_at": eb.now()}


def prune(state:dict[str, Any], engine:str, rm:Callable[[str], None] = shutil.rmtree) -> list[dict[str, Any]]:
  """After a comparison that succeeded: delete the builds older than the previous one that BoltBeam installed, each
  recorded under deleted; a build that is not BoltBeam's stays in older with the reason (engine_builds.deletable)."""
  e = eb.entry(state, engine)
  keep, gone = [], []
  for b in e.get("older") or []:
    if why := eb.deletable(b, state.get("root")):
      keep.append({**b, "kept": why})
      continue
    rm(b["folder"])
    gone.append({"version": b.get("version"), "folder": b["folder"], "at": eb.now()})
  e["older"] = keep
  e.setdefault("deleted", []).extend(gone)
  return gone


def look(state:dict[str, Any], engine:str, read:Callable[[str], Any] | None = None) -> dict[str, Any]:
  """Upstream's newest release into the machine's state, asked at most once per engine_latest_max_age_h."""
  e = eb.entry(state, engine)
  if not eb.fresh(e.get("checked")):
    got = eb.check(engine, read)
    e["checked"] = {k: got[k] for k in ("latest", "commit", "checked_at", "source", "error")}
  return e["checked"]


def latest(engines:list[str], read:Callable[[str], Any] | None = None,
           seed:Callable[[str], dict | None] = here) -> list[dict[str, Any]]:
  """`engines --latest`, the TUI's look after the engine scan: each engine's standing (engine_builds.standing) after
  a look at upstream when the last one is stale. Nothing is installed; a build here that no update recorded is read
  (seed) for its version only, not recorded."""
  state = eb.load()
  for engine in engines:
    look(state, engine, read)
  eb.save(state)
  return [eb.standing(e, state, None if eb.entry(state, e).get("current") else seed(e)) for e in engines]


def update(engines:list[str], *, backend:str, root:pathlib.Path | None = None, say:Callable[[str], None] = print,
           read:Callable[[str], Any] | None = None, seed:Callable[[str], dict | None] = here,
           put:Callable[..., dict[str, Any]] = install, fit:Callable[..., dict[str, str]] = fits) -> list[dict[str, Any]]:
  """Look, seed, check the fit and install, per engine; the machine's state is saved after each. One row per engine:
  action (could_not_check, up_to_date, waiting, installed, not_compatible, not_installed) and words."""
  from boltbeam.target import vendors
  vendor = vendors.for_backend(backend)
  state = eb.load()
  root = pathlib.Path(root or state.get("root") or eb.state_file().with_name("engines")).expanduser()
  state["root"] = str(root)
  out = []
  for engine in engines:
    e, got = eb.entry(state, engine), look(state, engine, read)
    if not e.get("current") and (found := seed(engine)):
      e["current"] = found
    cur, pend, latest, st = e.get("current"), e.get("pending"), got["latest"], eb.standing(engine, state)["state"]
    if st == "unknown":
      row = {"action": "could_not_check", "words": f"could not check upstream ({got['error']}); nothing installed"}
    elif st == "waiting":
      row = {"action": "waiting", "words": f"{pend['version']} is installed and waits for its one comparison with "
                                           f"{(cur or {}).get('version')} (the next --analyze run)"}
    elif st == "up_to_date":
      row = {"action": "up_to_date", "words": f"{cur['version']} is the latest release"}
    else:
      try:
        fill = fit(engine, latest, vendor, read)
        say(f"engines {engine}: installing {latest} into {root / engine / latest}")
        which = eb.installed(state, engine, put(engine, latest, root / engine / latest, vendor, fill))
        row = {"action": "installed", "words": f"{latest} installed as {which}" + (
          f"; the next --analyze run compares it once with {cur.get('version') or 'the build here'}" if which == "pending" else "")}
      except NotCompatible as exc:
        row = {"action": "not_compatible", "words": f"{latest} not compatible here: {exc}; current stays"}
      except (CannotInstall, RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        row = {"action": "not_installed", "words": f"{latest} not installed: {exc}"}
      if row["action"] != "installed":
        e["not_installed"] = {"version": latest, "reason": row["words"], "at": eb.now()}
    eb.save(state)
    say(f"engines {engine}: {row['words']}")
    out.append({"engine": engine, "latest": latest, "checked_at": got["checked_at"], **row,
                "current": (e.get("current") or {}).get("version"), "pending": (e.get("pending") or {}).get("version")})
  return out
