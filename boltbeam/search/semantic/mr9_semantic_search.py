"""Execute finalized MR8 semantic populations and issue bounded MR9 verdicts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import pathlib
import statistics
import re

from boltbeam.core.canonical import sha256_file, sha256_hex
import subprocess
from typing import Any

from boltbeam.search.semantic.mr8_population_selection import SCHEMA as MR8_SCHEMA, build_mr8_population_selection
from boltbeam.search.semantic_campaign_cli import run_request
from boltbeam.search.semantic.semantic_identity import content_sha256
from boltbeam.search.semantic.semantic_population_export import export_population
from boltbeam.search.full_kernel.tinygrad_full_kernel import PersistentJSONLSession


SCHEMA = "boltbeam.mr9_semantic_search.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT = re.compile(r"^[0-9a-f]{40}$")
_SELECTION_FIELDS = {"schema","status","mr7_sha256","mr7_prerequisites","model_sha256","resolved_target_sha256",
  "selected_roles","selection_basis","selected_semantic_identity_count","population_count","candidate_count",
  "ignored_unselected_population_request_count","populations","policy","mr7_ranking","selection_request","selection_sha256"}
_ROW_FIELDS = {"role","semantic_identity_sha256","semantic_identity","population_request_sha256","campaign_request","population"}


def _file_sha256(path: pathlib.Path) -> str:
  return sha256_file(path)


def _provider_command_binding(provider_command: Sequence[str], revision: str) -> dict[str, Any]:
  """Bind the executable and canonical provider script to one clean Tinygrad checkout."""
  if len(provider_command) != 4 or tuple(provider_command[2:]) != ("--backend", "METAL"):
    raise ValueError("MR9 requires the canonical Tinygrad Metal provider command")
  executable, script = pathlib.Path(provider_command[0]).expanduser(), pathlib.Path(provider_command[1]).expanduser()
  if not executable.is_absolute() or not script.is_absolute():
    raise ValueError("MR9 provider executable and script paths must be absolute")
  # Do not resolve the executable symlink before checking containment: a venv
  # interpreter normally resolves to the system Python while its selected
  # environment entry point is still rooted in the Tinygrad checkout.
  executable, script = executable.absolute(), script.resolve()
  if not executable.is_file() or not script.is_file(): raise ValueError("MR9 provider executable or script is unavailable")
  expected = pathlib.Path("extra/llm_research/search_provider.py")
  try: root = script.parents[2]; relative_script = script.relative_to(root)
  except (IndexError, ValueError) as exc: raise ValueError("MR9 provider script is not rooted in a Tinygrad checkout") from exc
  try: relative_executable = executable.relative_to(root)
  except ValueError as exc: raise ValueError("MR9 provider executable is outside the Tinygrad checkout") from exc
  if relative_script != expected or relative_executable != pathlib.Path(".venv/bin/python"):
    raise ValueError("MR9 provider command is not the canonical checkout-local Tinygrad provider")
  try:
    git_root = pathlib.Path(subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], check=True,
      text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()).resolve()
    actual_revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], check=True,
      text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip())
    subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(expected)], check=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pinned_script = subprocess.run(["git", "-C", str(root), "show", f"{revision}:{expected}"], check=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
  except (OSError, subprocess.CalledProcessError) as exc:
    raise ValueError("MR9 cannot verify the Tinygrad provider checkout") from exc
  script_sha256 = _file_sha256(script)
  if git_root != root or actual_revision != revision or dirty or sha256_hex(pinned_script) != script_sha256:
    raise ValueError("MR9 Tinygrad provider checkout is not the clean pinned revision")
  return {"root":str(root), "revision":revision, "dirty":False,
    "executable_path":str(executable), "executable_sha256":_file_sha256(executable.resolve()),
    "provider_script_path":str(script), "provider_script_sha256":script_sha256}


def _validate_selection(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
  if set(selection) != _SELECTION_FIELDS: raise ValueError("MR8 selection has missing or unknown canonical fields")
  selection_payload = {key:value for key,value in selection.items() if key != "selection_sha256"}
  if selection.get("selection_sha256") != content_sha256(selection_payload):
    raise ValueError("MR8 canonical selection hash mismatch")
  if any(_SHA256.fullmatch(selection.get(key, "")) is None for key in ("mr7_sha256","model_sha256","resolved_target_sha256")):
    raise ValueError("MR8 provenance hashes are malformed")
  if not isinstance(selection.get("mr7_prerequisites"), Mapping) or not isinstance(selection.get("selection_basis"), list) or \
      not isinstance(selection.get("ignored_unselected_population_request_count"), int) or selection["ignored_unselected_population_request_count"] < 0:
    raise ValueError("MR8 selection provenance is malformed")
  if selection.get("schema") != MR8_SCHEMA or selection.get("status") != "COMPLETE":
    raise ValueError("MR9 requires a complete MR8 population selection")
  ranking, request = selection.get("mr7_ranking"), selection.get("selection_request")
  if not isinstance(ranking, Mapping) or not isinstance(request, Mapping):
    raise ValueError("MR8 retained MR7 provenance is unavailable")
  from boltbeam.search.semantic.mr8_population_selection import _validate_authoritative_ranking
  _validate_authoritative_ranking(ranking)
  try: rederived = build_mr8_population_selection(ranking, request)
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(f"MR8 retained MR7 provenance cannot be re-derived: {exc}") from exc
  if rederived != selection:
    raise ValueError("MR8 selection differs from retained MR7 provenance re-derivation")
  populations = selection.get("populations")
  if not isinstance(populations, list) or not populations or len(populations) != selection.get("population_count") or \
      len(populations) != selection.get("selected_semantic_identity_count"):
    raise ValueError("MR8 selected population coverage is incomplete")
  selected_roles = selection.get("selected_roles")
  if not isinstance(selected_roles, list) or not selected_roles or len(selected_roles) != len(set(selected_roles)):
    raise ValueError("MR8 selected role coverage is malformed")
  seen = set()
  for row in populations:
    if not isinstance(row, Mapping): raise ValueError("MR8 population row is malformed")
    if set(row) != _ROW_FIELDS: raise ValueError("MR8 population row has missing or unknown canonical fields")
    request, population = row.get("campaign_request"), row.get("population")
    if not isinstance(request, Mapping) or content_sha256(request) != row.get("population_request_sha256"):
      raise ValueError("MR8 campaign request hash mismatch")
    if not isinstance(population, Mapping) or export_population(request) != population:
      raise ValueError("MR8 materialized population differs from its canonical campaign request")
    identity = row.get("semantic_identity_sha256")
    if not isinstance(identity, str) or identity in seen: raise ValueError("MR8 semantic population identity is duplicated")
    workload = request.get("semantic_workload")
    workload_identity = workload.get("semantic_identity") if isinstance(workload, Mapping) else None
    target = workload.get("target") if isinstance(workload, Mapping) else None
    if content_sha256(workload_identity) != identity or workload_identity != row.get("semantic_identity") or \
        row.get("role") not in selected_roles or not isinstance(workload_identity, Mapping) or \
        workload_identity.get("role") != row.get("role"):
      raise ValueError("MR8 semantic population identity binding is inconsistent")
    if workload.get("model_hash") != selection.get("model_sha256") or not isinstance(target, Mapping) or \
        target.get("resolved_target_hash") != selection.get("resolved_target_sha256"):
      raise ValueError("MR8 semantic population model/target binding is inconsistent")
    seen.add(identity)
  if set(selected_roles) != {row["role"] for row in populations}:
    raise ValueError("MR8 selected role coverage is incomplete")
  if sum(len(row["population"]["candidates"]) for row in populations) != selection.get("candidate_count"):
    raise ValueError("MR8 selected candidate population is partial")
  return populations


def _raw_samples(value: Any) -> list[float] | None:
  if not isinstance(value, Mapping): return None
  samples = value.get("samples_ns")
  if not isinstance(samples, list) or len(samples) < 5 or any(not isinstance(sample, (int, float)) or
      isinstance(sample, bool) or not math.isfinite(sample) or sample <= 0 for sample in samples): return None
  return [float(sample) for sample in samples]


def _decision(campaign: Mapping[str, Any], *, minimum_win_fraction: float,
              maximum_relative_mad: float, required_repeats: int,
              semantic_identity: Mapping[str, Any]) -> dict[str, Any]:
  rows = campaign.get("population")
  if campaign.get("status") != "COMPLETE" or not isinstance(rows, list):
    return {"verdict":"BLOCKED_PARTIAL_POPULATION", "reason":"initial finite campaign was not complete"}
  measured = sorted((row for row in rows if row.get("state") == "MEASURED"), key=lambda row:row["rank"])
  if not measured: return {"verdict":"BLOCKED_PARTIAL_POPULATION", "reason":"no correct measured candidates"}
  for row in measured:
    check = row.get("worker", {}).get("correctness", {}).get("evidence")
    oracle = check.get("oracle") if isinstance(check, Mapping) else None
    if oracle is None and isinstance(check, Mapping) and isinstance(check.get("evidence"), Mapping):
      check = check["evidence"]
      oracle = check.get("oracle")
    binding = check.get("exact_gguf_binding") if isinstance(check, Mapping) else None
    identity = row.get("candidate", {}).get("workload", {})
    if not isinstance(oracle, str) or oracle != "exact GGUF tensor + deterministic activation" or not isinstance(binding, Mapping) or \
        set(binding) != {"model_sha256","tensor_name"} or binding.get("model_sha256") != identity.get("model_sha256") or \
        binding.get("tensor_name") != semantic_identity.get("tensor_name"):
      return {"verdict":"BLOCKED_PARTIAL_POPULATION", "reason":"measured candidate lacks the exact real-role GGUF oracle",
              "candidate_hash":row["candidate_hash"]}
  controls = [row for row in measured if row["candidate"].get("schedule", {}).get("plan_kind") == "tinygrad_heuristic.v1"]
  if len(controls) != 1: return {"verdict":"BLOCKED_PARTIAL_POPULATION", "reason":"heuristic control is not exactly measured"}
  repeat_rows = campaign.get("finalist_repetitions")
  if not isinstance(repeat_rows, list): return {"verdict":"BLOCKED_FINALIST_REPEAT", "reason":"finalist repetitions are unavailable"}
  repeats = {row.get("candidate_hash"):row.get("repetitions") for row in repeat_rows if isinstance(row, Mapping)}
  required = {row["candidate_hash"] for row in measured[:2]} | {controls[0]["candidate_hash"]}
  if set(repeats) != required:
    return {"verdict":"BLOCKED_FINALIST_REPEAT", "reason":"finalist/control repeat coverage is incomplete"}
  stability = []
  by_hash = {row["candidate_hash"]:row for row in measured}
  for candidate_hash in sorted(required):
    candidate_repeats = repeats[candidate_hash]
    if not isinstance(candidate_repeats, list) or len(candidate_repeats) != required_repeats:
      return {"verdict":"BLOCKED_FINALIST_REPEAT", "reason":"finalist repeat count is incomplete",
              "candidate_hash":candidate_hash}
    initial = _raw_samples(by_hash[candidate_hash].get("measurement"))
    repeat_samples = [_raw_samples(row.get("performance", {}).get("measurement")) if isinstance(row, Mapping) and
                      row.get("status") == "MEASURED" else None for row in candidate_repeats]
    if initial is None or any(samples is None for samples in repeat_samples):
      return {"verdict":"BLOCKED_FINALIST_REPEAT", "reason":"finalist repeat lacks five raw samples",
              "candidate_hash":candidate_hash}
    run_medians = [statistics.median(initial), *(statistics.median(samples) for samples in repeat_samples if samples is not None)]
    median = statistics.median(run_medians)
    relative_mad = statistics.median(abs(value-median) for value in run_medians) / median
    stability.append({"candidate_hash":candidate_hash, "run_medians_ns":run_medians,
      "aggregate_median_ns":median, "relative_mad":relative_mad,
      "status":"stable" if relative_mad <= maximum_relative_mad else "unstable"})
  if any(row["status"] != "stable" for row in stability):
    return {"verdict":"INCONCLUSIVE_FINALIST_INSTABILITY", "reason":"one or more finalist/control repeats are unstable",
            "stability":stability}
  stable_by_hash = {row["candidate_hash"]:row for row in stability}
  ordered = sorted(stability, key=lambda row:(row["aggregate_median_ns"], row["candidate_hash"]))
  best, control = ordered[0], stable_by_hash[controls[0]["candidate_hash"]]
  improvement = 1 - best["aggregate_median_ns"] / control["aggregate_median_ns"]
  machine = best["candidate_hash"] != control["candidate_hash"]
  win = machine and improvement >= minimum_win_fraction
  return {"verdict":"MACHINE_WINNER" if win else "REFUTED_NO_MATERIAL_WIN",
    "selected_candidate_hash":best["candidate_hash"] if win else control["candidate_hash"],
    "machine_candidate_hash":best["candidate_hash"] if machine else None,
    "control_candidate_hash":control["candidate_hash"], "improvement_fraction":improvement,
    "minimum_win_fraction":minimum_win_fraction, "stability":stability,
    "reason":"stable machine candidate clears the control threshold" if win else "no stable machine candidate clears the control threshold"}


def run_mr9_semantic_search(selection: Mapping[str, Any], *, provider_command: Sequence[str],
                            finalist_repeats: int = 2, minimum_win_fraction: float = 0.03,
                            maximum_relative_mad: float = 0.05,
                            requested_provider_revision: str | None = None,
                            requested_boltbeam_revision: str | None = None) -> dict[str, Any]:
  if not isinstance(provider_command, Sequence) or isinstance(provider_command, (str, bytes)) or not provider_command or \
      any(not isinstance(value, str) or not value for value in provider_command):
    raise ValueError("MR9 provider command must be a non-empty argv sequence")
  if not isinstance(finalist_repeats, int) or finalist_repeats < 2:
    raise ValueError("MR9 requires at least two finalist repetitions")
  if not isinstance(minimum_win_fraction, (int, float)) or not 0 < minimum_win_fraction < 1:
    raise ValueError("MR9 minimum win fraction must be between zero and one")
  if not isinstance(maximum_relative_mad, (int, float)) or not 0 < maximum_relative_mad < 1:
    raise ValueError("MR9 maximum relative MAD must be between zero and one")
  populations = _validate_selection(selection)
  campaigns, decisions = [], []
  if not isinstance(requested_provider_revision, str) or _GIT.fullmatch(requested_provider_revision) is None:
    raise ValueError("MR9 requires a pinned Tinygrad provider revision")
  if not isinstance(requested_boltbeam_revision, str) or _GIT.fullmatch(requested_boltbeam_revision) is None:
    raise ValueError("MR9 requires a clean pinned BoltBeam revision")
  provider_binding = _provider_command_binding(provider_command, requested_provider_revision)
  session = PersistentJSONLSession(tuple(provider_command))
  with session:
    for row in populations:
      campaign = run_request(dict(row["campaign_request"]), provider_command=list(provider_command), provider_session=session,
                             requested_provider_revision=requested_provider_revision,
                             requested_boltbeam_revision=requested_boltbeam_revision,
                             finalist_count=2, finalist_repeats=finalist_repeats)
      campaigns.append({"role":row["role"], "semantic_identity_sha256":row["semantic_identity_sha256"],
                        "campaign":campaign,
                        "terminal_failures":[{"candidate_hash":item["candidate_hash"], "state":item["state"],
                          "reason":item.get("reason"), "worker":item.get("worker")} for item in campaign.get("population", [])
                          if item.get("state") != "MEASURED"]})
      decisions.append({"role":row["role"], "semantic_identity_sha256":row["semantic_identity_sha256"],
                        **_decision(campaign, minimum_win_fraction=float(minimum_win_fraction),
                                    maximum_relative_mad=float(maximum_relative_mad), required_repeats=finalist_repeats,
                                    semantic_identity=row["semantic_identity"])})
  complete = all(row["verdict"] in ("MACHINE_WINNER", "REFUTED_NO_MATERIAL_WIN") for row in decisions)
  return {"schema":SCHEMA, "status":"COMPLETE" if complete else "BLOCKED",
    "mr8_sha256":content_sha256(selection), "model_sha256":selection.get("model_sha256"),
    "resolved_target_sha256":selection.get("resolved_target_sha256"), "provider_command":list(provider_command),
    "provider_revision":requested_provider_revision,
    "provider_binding":provider_binding,
    "boltbeam_revision":requested_boltbeam_revision,
    "policy":{"finalist_count":2, "finalist_repeats":finalist_repeats,
      "minimum_win_fraction":float(minimum_win_fraction), "maximum_relative_mad":float(maximum_relative_mad),
      "minimum_raw_samples_per_run":5}, "campaigns":campaigns, "decisions":decisions}


def closure_disposition(result: Mapping[str, Any]) -> str:
  """Map one complete canonical MR9 result to MR13's conditional branch."""
  if not isinstance(result, Mapping) or result.get("schema") != SCHEMA or result.get("status") != "COMPLETE":
    raise ValueError("MR13 requires a complete canonical MR9 result")
  decisions = result.get("decisions")
  if not isinstance(decisions, list) or not decisions:
    raise ValueError("MR9 result has no complete decisions")
  verdicts = [row.get("verdict") for row in decisions if isinstance(row, Mapping)]
  if len(verdicts) != len(decisions) or any(verdict not in ("MACHINE_WINNER", "REFUTED_NO_MATERIAL_WIN") for verdict in verdicts):
    raise ValueError("MR9 result contains a blocked or unknown decision")
  return "winner" if "MACHINE_WINNER" in verdicts else "refuted"


def main(argv: list[str] | None = None) -> int:
  import argparse
  parser = argparse.ArgumentParser(description="Execute complete MR8 semantic populations through persistent Tinygrad search")
  parser.add_argument("selection", type=pathlib.Path); parser.add_argument("--provider-command", required=True)
  parser.add_argument("--provider-revision", required=True)
  parser.add_argument("--boltbeam-revision", required=True)
  parser.add_argument("--finalist-repeats", type=int, default=2); parser.add_argument("--minimum-win", type=float, default=0.03)
  parser.add_argument("--maximum-relative-mad", type=float, default=0.05); parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args(argv)
  if args.out.exists(): raise FileExistsError(f"MR9 output already exists: {args.out}")
  command = json.loads(args.provider_command)
  result = run_mr9_semantic_search(json.loads(args.selection.read_text()), provider_command=command,
    finalist_repeats=args.finalist_repeats, minimum_win_fraction=args.minimum_win,
    maximum_relative_mad=args.maximum_relative_mad, requested_provider_revision=args.provider_revision,
    requested_boltbeam_revision=args.boltbeam_revision)
  args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
  return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__": raise SystemExit(main())

__all__ = ["SCHEMA", "closure_disposition", "run_mr9_semantic_search", "main"]
