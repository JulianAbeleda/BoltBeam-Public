"""`boltbeam selfcheck`: run the core path on a tiny synthetic model and check every output.

main carries no test suite, so this is the check a reader can run after install. It needs no GPU, no network
and no model file. It writes a synthetic GGUF, runs inspect, roofline-theoretical, load, autoscan, analyze and
output, then asks the screen seam for the ceiling and the results, the way the TUI does. Each step prints PASS
or FAIL. Any FAIL exits 1.

The schema check is data driven. An output file `<name>.json` must carry `"schema": "boltbeam.<name>.v1"`.
When the checkout's schemas/ folder is present, the file must also have the required keys that
schemas/<name>.schema.json lists.
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
from typing import Any, Callable

TARGET = "amd_gfx1100"  # a fixed registry target, so the arithmetic does not depend on this machine
SCHEMAS = pathlib.Path(__file__).resolve().parents[2] / "schemas"

# step name, the files it must write (relative to the work folder)
STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
  ("inspect", ("model_profile.json",)),
  ("roofline-theoretical", ("theoretical_roofline.json",)),
  ("load", ("run/run_manifest.json", "run/model_profile.json", "run/weight_inventory.json",
            "run/workload_profile.json")),
  ("autoscan", ("run/hardware_profile.json", "run/scan_evidence.json")),
  ("analyze", ("run/search_space.json", "run/route_policy.json", "run/measurement_plan.json",
               "run/analysis_report.json")),
  ("output", ("run/output_manifest.json", "run/provider_plan.json", "run/report.html")),
  ("screen ceiling", ("screen_ceiling.json",)),
  ("screen results", ("screen_results.json",)),
)


def _quiet(fn:Callable[[], Any]) -> Any:
  """Run fn with its stdout captured: the commands print JSON a reader does not need here."""
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    rc = fn()
  return rc, buf.getvalue()


def _screen(argv:list[str], out:pathlib.Path) -> int:
  from boltbeam.workflow import screen
  rc, text = _quiet(lambda: screen.main(argv))
  out.write_text(text)
  return rc


def _not_measured_is_ok(rc:int) -> int:
  return 0 if rc == 3 else rc


def check_file(path:pathlib.Path) -> str | None:
  """Return what is wrong with one output file, or None when it is fine."""
  if not path.exists():
    return f"{path.name} was not written"
  if path.suffix != ".json":
    return None if path.stat().st_size > 0 else f"{path.name} is empty"
  try:
    doc = json.loads(path.read_text())
  except ValueError as exc:
    return f"{path.name} is not JSON: {exc}"
  if path.name.startswith("screen_"):  # the screen seam answers its own schema id with a kind
    from boltbeam.workflow.screen import SCHEMA as SCREEN_SCHEMA
    kind = path.stem.removeprefix("screen_")
    if doc.get("schema") != SCREEN_SCHEMA or doc.get("kind") != kind:
      return f"{path.name}: expected {SCREEN_SCHEMA} kind {kind}, got {doc.get('schema')} kind {doc.get('kind')}"
    return None
  want = f"boltbeam.{path.stem}.v1"
  if doc.get("schema") != want:
    return f"{path.name}: expected schema {want}, got {doc.get('schema')}"
  spec = SCHEMAS / f"{path.stem}.schema.json"
  if spec.exists():
    missing = [k for k in json.loads(spec.read_text()).get("required", []) if k not in doc]
    if missing:
      return f"{path.name}: missing required keys {missing} (schemas/{spec.name})"
  return None


def _actions(work:pathlib.Path, model:str) -> dict[str, Callable[[], int]]:
  from boltbeam.cli import main
  run = str(work / "run")
  return {
    "inspect": lambda: main(["inspect", model, "--target", TARGET, "--out", str(work / "model_profile.json")]),
    "roofline-theoretical": lambda: main(["roofline-theoretical", model, "--target", TARGET,
                                          "--out", str(work / "theoretical_roofline.json")]),
    "load": lambda: main(["load", model, "--run", run, "--target", TARGET, "--id", "selfcheck"]),
    "autoscan": lambda: main(["autoscan", "--run", run]),
    "analyze": lambda: main(["analyze", "--run", run]),
    "output": lambda: main(["output", "--run", run]),
    "screen ceiling": lambda: _screen(["ceiling", model, "--target", TARGET], work / "screen_ceiling.json"),
    # results exits 3 when the run has no measurement yet. Offline, that is the expected answer.
    "screen results": lambda: _not_measured_is_ok(_screen(["results", "--run", run], work / "screen_results.json")),
  }


def selfcheck(work:pathlib.Path, out=sys.stdout, keep:bool = False) -> int:
  from boltbeam.synth.gguf import dense_decoder_kv, dense_decoder_tensors, write_gguf
  model = write_gguf(work / "selfcheck.gguf", dense_decoder_kv(), dense_decoder_tensors())
  actions = _actions(work, model)
  failed = 0
  for name, files in STEPS:
    if failed:  # every step reads the one before it, so a failure stops the path
      out.write(f"SKIP  {name}\n")
      continue
    try:
      rc, _ = _quiet(actions[name])
      problems = [f"exit code {rc}"] if rc else []
    except Exception as exc:  # a crash is a FAIL with its reason, not a traceback
      problems = [f"{type(exc).__name__}: {exc}"]
    problems += [p for p in (check_file(work / f) for f in files) if p] if not problems else []
    if problems:
      failed += 1
      out.write(f"FAIL  {name}: {'; '.join(problems)}\n")
    else:
      out.write(f"PASS  {name}\n")
  level = "schema id and required keys" if SCHEMAS.is_dir() else "schema id (schemas/ is not in this install)"
  out.write(f"checked: {level}" + (f"; outputs in {work}" if keep else "") + "\n")
  out.write("selfcheck: FAIL\n" if failed else "selfcheck: PASS\n")
  return 1 if failed else 0


def cmd_selfcheck(args) -> int:
  if args.dir:
    work = pathlib.Path(args.dir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    return selfcheck(work, keep=True)
  with tempfile.TemporaryDirectory(prefix="boltbeam-selfcheck-") as tmp:
    return selfcheck(pathlib.Path(tmp))


def register(sub) -> None:
  p = sub.add_parser("selfcheck", help="run the core path on a tiny synthetic model and check every output (no GPU, no network)")
  p.add_argument("--dir", default=None, help="keep the outputs in this folder (default: a temporary folder)")
  p.set_defaults(fn=cmd_selfcheck)
