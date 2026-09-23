from __future__ import annotations

import pathlib
from typing import Any

from boltbeam.plan.runner_plan import build_runner_plan
from boltbeam.workflow.common import run_dir, update_manifest, write_json

RUNNER_ARTIFACTS = ("runner_plan.json",)


def runner_plan_run(run:str | pathlib.Path, *, providers:tuple[str, ...] = (),
                    bundle_dir:str | pathlib.Path | None = None) -> dict[str, Any]:
  out = run_dir(run)
  plan = build_runner_plan(out, providers=providers, bundle_dir=bundle_dir)
  write_json(out / "runner_plan.json", plan)
  bundle_path = pathlib.Path(plan["bundle"]["path"])
  write_json(bundle_path / "runner_plan.json", plan)
  update_manifest(out, stage="runner_plan", artifacts=list(RUNNER_ARTIFACTS))
  return plan
