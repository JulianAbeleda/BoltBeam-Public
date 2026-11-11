"""A run folder as facts: where a run id is looked up, its stages and what it still needs, the engine it was
measured with, and the files every reader opens. Nothing here computes a number; the limit is roofline/limit.py and
the results are workflow/results.py. The seam that prints these is `python -m boltbeam.workflow.screen`."""
from __future__ import annotations

import pathlib
import shutil
from typing import Any

from boltbeam.core.refused import Refused
from boltbeam.report.html import STAGES, stage_state
from boltbeam.workflow import fetch
from boltbeam.artifacts.run_io import MEASURE_STATUS, load_manifest, optional
from boltbeam.vocab import SCHEMA_TUI as SCHEMA
from boltbeam.workflow.layout import LAYOUTS

# what a run still needs, read off measurement_plan.json: (plan block, the evidence it asks for, its request file)
NEEDS = (("primitive_profile", "probe_evidence", "probe_request.json"),
         ("timing_profile", "timing_trace", "trace_request.json"))


SCHEMA_MEASURE_STATUS = "boltbeam.measure_status.v1"


# --- run folders -----------------------------------------------------------------------------------------------

# the stage rows a running screen counts as steps (analyze is setup on its first run, so it is not counted here)
COUNTED_STAGES = frozenset({"ingest_probe", "ingest_timing", "output"})


def stage_rows(manifest:dict[str, Any], measure:dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """state: done, not_needed (state_note says why) or open (report/html.py stage_state decides)."""
  stages = manifest.get("stages", {}) or {}
  rows = []
  for key, label, note in STAGES:
    state, why = stage_state(key, stages.get(key), measure)
    rows.append({"key": key, "label": label, "note": note, "counted": key in COUNTED_STAGES, "done": key in stages,
                 "state": state, "state_note": why,
                 "artifacts": list((stages.get(key) or {}).get("artifacts", []) or [])})
  return rows


def blocked(plan:dict[str, Any], run:pathlib.Path | None = None) -> list[dict[str, Any]]:
  """Outstanding requests after accounting for evidence ingested since planning."""
  return [{"need": need, "request": request} for block, need, request in NEEDS
          if (plan.get(block) or {}).get("status") == "requested"
          and not (run is not None and optional(run, block + ".json"))]


def summary(run:pathlib.Path) -> dict[str, Any]:
  manifest = load_manifest(run)
  report = optional(run, "analysis_report.json")
  plan = optional(run, "measurement_plan.json")
  return {
    "id": run.name, "model_id": manifest.get("model_id"), "model_format": manifest.get("model_format"),
    "target_id": manifest.get("target_id"), "workload": manifest.get("workload"),
    "latest_stage": manifest.get("latest_stage"), "status": report.get("status", "not_analyzed"),
    "stages": stage_rows(manifest, optional(run, MEASURE_STATUS)), "blocked": blocked(plan, run),
    "measured": {"probe": (run / "primitive_profile.json").exists(), "timing": (run / "timing_profile.json").exists()},
    "report": "report.html" if (run / "report.html").exists() else None,
    "measure": optional(run, MEASURE_STATUS) or None,
    **({"fetched": f} if (f := fetch.provenance(run)) else {}),  # a run brought home by `boltbeam fetch`: from where
  }


def is_run(path:pathlib.Path) -> bool:
  return path.is_dir() and (path / "run_manifest.json").exists()


def delete(root:pathlib.Path, run:pathlib.Path) -> dict[str, Any]:
  """Remove one run folder. Only a folder holding run_manifest.json directly under the runs root qualifies."""
  root, run = root.resolve(), run.resolve()
  if run.parent != root:
    raise Refused(f"{run} is not directly under the runs folder {root}")
  if not is_run(run):
    raise Refused(f"{run} is not a run folder (no run_manifest.json)")
  shutil.rmtree(run)
  return {"schema": SCHEMA, "kind": "deleted", "root": str(root), "run": run.name}


WORK = ".work"  # where the screens keep a run until it is saved
SAVED = "saved"  # where a saved run lives (the screens' default --saved)


def run_places(root:pathlib.Path, saved:pathlib.Path | None = None) -> list[tuple[str, pathlib.Path]]:
  """Where a run id is looked up, in this order: <root>/<id> (runs made by --json start and older layouts),
  <root>/.work/<id> (the screens' unsaved runs), <saved>/<id> (saved runs; default <root>/saved)."""
  return [("runs", root), ("work", root / WORK), ("saved", saved if saved is not None else root / SAVED)]


def find_run(root:pathlib.Path, run_id:str, saved:pathlib.Path | None = None) -> pathlib.Path:
  """The run folder for an id: the first of run_places that holds it. Refused, naming every place, when none does."""
  if not run_id or run_id != pathlib.Path(run_id).name or run_id.startswith("."):
    raise Refused(f"run id must be a folder name: {run_id!r}")
  places = run_places(root, saved)
  for _, folder in places:
    if is_run(folder / run_id):
      return folder / run_id
  raise Refused(f"no run {run_id} in " + ", ".join(str(f / run_id) for _, f in places))


def runs(root:pathlib.Path, saved:pathlib.Path | None = None, *, all_places:bool = False) -> dict[str, Any]:
  """One summary per run under root. A root that does not exist yet holds no runs: that is a fact, not an error.
  This read never creates it; the first pipeline run does (artifacts/run_io.py run_dir). all_places also lists the
  screens' work area and the saved runs (run_places), each row marked where it lives: runs, work or saved."""
  if root.exists() and not root.is_dir():
    raise Refused(f"{root} is not a folder")
  places = run_places(root, saved) if all_places else [("runs", root)]
  rows = []
  for where, folder in places:
    if folder.is_dir():
      rows += [{**summary(p), "where": where, "dir": str(p)} for p in sorted(folder.iterdir()) if is_run(p)]
  return {"schema": SCHEMA, "kind": "runs", "root": str(root), "runs": rows}

def not_measured(run:pathlib.Path) -> str:
  """Why a run holds no speed, from its measure status: "not measured: <reason>"."""
  status = optional(run, MEASURE_STATUS)
  reason = status.get("reason") or (f"the measure step ended {status['status']}" if status.get("status") else
                                     "the run never reached the measure step")
  return "not measured: " + str(reason)

def measurement_of(run:pathlib.Path) -> dict[str, Any] | None:
  """The run's measurement choice (providers.resolve_measurement) as measure_status.json recorded it; None for a
  run from before the choice existed."""
  return ((optional(run, MEASURE_STATUS).get("capture") or {}).get("measurement")) or None

SAVE_RECORD = "save.json"

def layout_words(run:pathlib.Path) -> str | None:
  """A run's GPU layout as the engine picker words it ("split by layer (2 GPUs)"), from its measure_status.json;
  None for one GPU or a run that recorded none, so a one-GPU list reads as it always has."""
  m = optional(run, MEASURE_STATUS)
  if (lay := m.get("layout")) in (None, "one") or lay not in LAYOUTS:
    return None
  return LAYOUTS[lay] + (f" ({m['gpus']} GPUs)" if m.get("gpus") else "")
