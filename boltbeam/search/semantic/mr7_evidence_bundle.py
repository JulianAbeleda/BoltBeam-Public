"""Plan, collect, and bind exact MR7 evidence without hand-authored JSON.

The provider boundary is deliberately data-only. Complete outer durations are
requested and retained as complete samples; this module never apportions an
outer graph duration among member calls.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
import pathlib
import re

from boltbeam.core.canonical import sha256_file
import select
import statistics
import subprocess
import time
from typing import Any

from boltbeam.search.role_cost_ranking import (ENCLOSURES_SCHEMA, MEASUREMENTS_SCHEMA, MR4_SUMMARY_SCHEMA,
  build_role_cost_ranking, reconcile_material_calls, whole_step_from_mr4_summary)
from boltbeam.search.semantic.semantic_identity import (SEMANTIC_IDENTITY_FIELDS_SHA256, canonical_json, content_sha256,
  require_matching_identity_fields)


PLAN_SCHEMA = "boltbeam.mr7_evidence_plan.v1"
PROVIDER_PROTOCOL = "boltbeam.mr7_provider.v1"
PROVIDER_RESULTS_SCHEMA = "boltbeam.mr7_provider_results.v1"
BUNDLE_SCHEMA = "boltbeam.mr7_evidence_bundle.v1"
AUTHORITY_SCHEMA = "boltbeam.mr7_authority.v1"
MINIMUM_SAMPLES = 5
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _request_id(kind: str, payload: Mapping[str, Any]) -> str:
  return f"{kind}-{content_sha256(payload)[:20]}"


def build_evidence_plan(mr4_summary: Mapping[str, Any], census: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
  """Derive every required provider measurement from immutable MR4/MR5 facts."""
  # This validates status/readiness/order and ensures the selected arm exists.
  whole_step_from_mr4_summary(mr4_summary, arm)
  material = reconcile_material_calls(census)
  isolated: dict[str, dict[str, Any]] = {}
  for call in material:
    identity = call["execution_identity"]
    key = content_sha256(identity)
    if key not in isolated:
      payload = {"kind":"exact_isolated_execution", "execution_identity":identity,
        "shape_mode":"exact_workload", "minimum_samples":MINIMUM_SAMPLES, "correctness_required":True,
        "call_indices":[], "occurrence_count":0}
      isolated[key] = payload
    isolated[key]["call_indices"].append(call["call_index"])
    isolated[key]["occurrence_count"] += 1
  outer_groups: dict[str, list[int]] = {}
  for call in material: outer_groups.setdefault(call["outer_id"], []).append(call["call_index"])
  requests = []
  for payload in isolated.values():
    payload["call_indices"].sort()
    requests.append({"request_id":_request_id("isolated", payload), **payload})
  for outer_id,call_indices in outer_groups.items():
    payload = {"kind":"complete_outer_enclosure", "outer_id":outer_id, "call_indices":sorted(call_indices),
      "shape_mode":"complete_outer", "minimum_samples":MINIMUM_SAMPLES, "no_member_duration_splitting":True}
    requests.append({"request_id":_request_id("outer", payload), **payload})
  requests.sort(key=lambda row:row["request_id"])
  if len({row["request_id"] for row in requests}) != len(requests): raise ValueError("provider request ID collision")
  return {"schema":PLAN_SCHEMA, "mr4_sha256":content_sha256(mr4_summary), "mr5_sha256":content_sha256(census),
          "arm":arm, "minimum_samples":MINIMUM_SAMPLES, "material_call_count":len(material),
          "requests":requests,
          "accounting":"isolated identities are deduplicated; each outer request measures its complete duration"}


def _measurement(value: Any) -> tuple[list[float], float]:
  if not isinstance(value, Mapping): raise ValueError("provider measurement is unavailable")
  samples = value.get("samples_ns")
  if not isinstance(samples, list) or len(samples) < MINIMUM_SAMPLES or any(
      not isinstance(sample, (int, float)) or isinstance(sample, bool) or not math.isfinite(sample) or sample <= 0
      for sample in samples):
    raise ValueError("provider measurement requires at least five positive raw samples")
  normalized = [float(sample) for sample in samples]
  median = statistics.median(normalized)
  if value.get("median_ns") != median: raise ValueError("provider measurement median does not match raw samples")
  return normalized, median


def _run_map(plan: Mapping[str, Any], provider_results: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], str]:
  if plan.get("schema") != PLAN_SCHEMA: raise ValueError("unsupported MR7 evidence plan schema")
  if provider_results.get("schema") != PROVIDER_RESULTS_SCHEMA or provider_results.get("plan_sha256") != content_sha256(plan):
    raise ValueError("provider results do not bind the exact MR7 evidence plan")
  expected_environment = {"METAL_HYBRID_REPLAY":"0" if plan.get("arm") == "control" else "1"}
  if provider_results.get("provider_session") != "persistent_jsonl" or provider_results.get("provider_environment") != expected_environment:
    raise ValueError("provider session/environment does not match the selected MR4 arm")
  command, command_hash = provider_results.get("provider_command"), provider_results.get("provider_command_sha256")
  if not isinstance(command, list) or any(not isinstance(value, str) or not value for value in command) or \
      command_hash != content_sha256(command):
    raise ValueError("provider command identity is unavailable")
  file_rows = provider_results.get("provider_file_identities")
  if not isinstance(file_rows, list) or any(not isinstance(row, Mapping) or
      not isinstance(row.get("argv_index"), int) or row["argv_index"] < 0 or row["argv_index"] >= len(command) or
      not isinstance(row.get("sha256"), str) or _SHA256.fullmatch(row["sha256"]) is None for row in file_rows):
    raise ValueError("provider executable/script file identity is unavailable")
  provider_file_hashes = {row["sha256"] for row in file_rows}
  requests = plan.get("requests"); runs = provider_results.get("runs")
  if not isinstance(requests, list) or not isinstance(runs, list): raise ValueError("provider request/results rows are unavailable")
  expected = {row["request_id"]:row for row in requests}
  observed: dict[str, Mapping[str, Any]] = {}
  provider_identity: str | None = None
  provider_identity_fields_sha256: str | None = None
  for row in runs:
    request_id = row.get("request_id") if isinstance(row, Mapping) else None
    if request_id not in expected or request_id in observed: raise ValueError("provider result is extra or duplicated")
    if row.get("request_sha256") != content_sha256(expected[request_id]):
      raise ValueError("provider result request hash mismatch")
    if row.get("status") != "ok" or not isinstance(row.get("result"), Mapping):
      raise ValueError("provider result is blocked or malformed")
    result = row["result"]
    # The vocabulary declaration is a protocol-level fact on the response envelope, not a measurement field.
    require_matching_identity_fields(row.get("identity_fields"), row.get("identity_fields_sha256"))
    declared_sha256 = row["identity_fields_sha256"]
    if provider_identity_fields_sha256 is None:
      provider_identity_fields_sha256 = declared_sha256
    elif declared_sha256 != provider_identity_fields_sha256:
      raise ValueError("provider identity_fields_sha256 changed during collection")
    identity = result.get("provider_identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("revision"), str) or len(identity["revision"]) != 40 or \
        not isinstance(identity.get("dirty"), bool) or not isinstance(identity.get("script_sha256"), str) or \
        _SHA256.fullmatch(identity["script_sha256"]) is None:
      raise ValueError("provider runtime identity is unavailable")
    if identity["script_sha256"] not in provider_file_hashes:
      raise ValueError("provider runtime script hash does not match the executed command files")
    canonical_identity = canonical_json(identity)
    if provider_identity is not None and canonical_identity != provider_identity:
      raise ValueError("provider runtime identity changed during collection")
    provider_identity = canonical_identity
    observed[request_id] = result
  if set(observed) != set(expected): raise ValueError("provider results have incomplete request coverage")
  if provider_identity_fields_sha256 is None:
    raise ValueError("provider did not declare identity_fields_sha256 in any response")
  return observed, provider_identity_fields_sha256


def adapt_provider_results(plan: Mapping[str, Any], provider_results: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
  """Validate exact provider responses and emit ranker-native measurements/enclosures."""
  observed, provider_identity_fields_sha256 = _run_map(plan, provider_results)
  measurements, enclosures = [], []
  for request in plan["requests"]:
    result = observed[request["request_id"]]
    samples, median = _measurement(result.get("measurement"))
    if request["kind"] == "exact_isolated_execution":
      if result.get("execution_identity") != request["execution_identity"] or result.get("shape_mode") != "exact_workload":
        raise ValueError("isolated provider result changed the exact execution identity")
      correctness, generated = result.get("correctness"), result.get("generated")
      if not isinstance(correctness, Mapping) or correctness.get("passed") is not True:
        raise ValueError("isolated provider correctness failed")
      if not isinstance(generated, Mapping): raise ValueError("isolated generated identity is unavailable")
      program = request["execution_identity"]
      if generated.get("source_hash") != program["source_sha256"] or generated.get("binary_hash") != program["binary_sha256"]:
        raise ValueError("isolated generated source/binary identity mismatch")
      evidence = (correctness.get("evidence_hash"), result.get("performance_evidence_hash"), generated.get("plan_hash"))
      if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in evidence):
        raise ValueError("isolated provider evidence hash is unavailable")
      measurements.append({"execution_identity":request["execution_identity"], "execution":{"shape_mode":"exact_workload"},
        "correctness":dict(correctness), "performance":{"evidence_hash":result["performance_evidence_hash"],
          "measurement":{"samples_ns":samples, "median_ns":median}}, "generated":dict(generated)})
    else:
      if result.get("outer_id") != request["outer_id"] or result.get("call_indices") != request["call_indices"] or \
          result.get("shape_mode") != "complete_outer":
        raise ValueError("outer provider result changed complete enclosure membership")
      if request.get("no_member_duration_splitting") is not True:
        raise ValueError("outer provider request permits member-duration splitting")
      evidence_hash = result.get("performance_evidence_hash")
      if not isinstance(evidence_hash, str) or _SHA256.fullmatch(evidence_hash) is None:
        raise ValueError("outer provider evidence hash is unavailable")
      enclosures.append({"outer_id":request["outer_id"], "call_indices":request["call_indices"],
                         "samples_ns":samples, "median_ns":median, "evidence_hash":evidence_hash})
  return ({"schema":MEASUREMENTS_SCHEMA, "plan_sha256":content_sha256(plan), "measurements":measurements,
           "provider_identity_fields_sha256":provider_identity_fields_sha256},
          {"schema":ENCLOSURES_SCHEMA, "plan_sha256":content_sha256(plan), "rows":enclosures,
           "accounting":"complete outer samples only; no graph-duration/member splitting"})


def collect_provider_results(plan: Mapping[str, Any], command: Sequence[str], *, timeout_s: float = 60.0) -> dict[str, Any]:
  """Run one campaign-scoped JSONL provider so model/capture caches are reused."""
  if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command:
    raise ValueError("provider command must be a non-empty argv sequence")
  if timeout_s <= 0: raise ValueError("provider timeout must be positive")
  plan_sha = content_sha256(plan); runs = []; command_list = [str(value) for value in command]
  def file_identities():
    identities = []
    for index,value in enumerate(command_list):
      path = pathlib.Path(value)
      if not path.is_file(): continue
      identities.append({"argv_index":index, "path":str(path.resolve()), "sha256":sha256_file(path)})
    return identities
  before_files = file_identities()
  arm = plan.get("arm")
  if arm not in ("control", "hybrid"): raise ValueError("MR7 provider plan arm is invalid")
  provider_environment = {"METAL_HYBRID_REPLAY":"0" if arm == "control" else "1"}
  process = subprocess.Popen(command_list, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL, start_new_session=True, env={**os.environ, **provider_environment})
  pending = ""
  try:
    for request in plan.get("requests", []):
      if process.poll() is not None: raise RuntimeError(f"provider session exited {process.returncode}")
      envelope = {"protocol":PROVIDER_PROTOCOL, "plan_sha256":plan_sha,
                  "request_id":request["request_id"], "request":request}
      assert process.stdin is not None and process.stdout is not None
      process.stdin.write(canonical_json(envelope) + "\n"); process.stdin.flush()
      deadline = time.monotonic() + timeout_s
      while "\n" not in pending:
        ready, _, _ = select.select([process.stdout], [], [], max(0, deadline-time.monotonic()))
        if not ready: raise TimeoutError(f"provider {request['request_id']} timed out")
        chunk = os.read(process.stdout.fileno(), 65536)
        if not chunk: raise RuntimeError("provider session closed without a response")
        pending += chunk.decode()
      line, pending = pending.split("\n", 1)
      response = json.loads(line)
      if not isinstance(response, Mapping) or response.get("protocol") != PROVIDER_PROTOCOL or \
          response.get("request_id") != request["request_id"]:
        raise ValueError("provider response identity mismatch")
      runs.append({"request_id":request["request_id"], "request_sha256":content_sha256(request),
                   "status":response.get("status"), "result":response.get("result"),
                   "identity_fields":response.get("identity_fields"),
                   "identity_fields_sha256":response.get("identity_fields_sha256")})
  finally:
    if process.poll() is None:
      process.terminate()
      try: process.wait(timeout=2)
      except subprocess.TimeoutExpired: process.kill(); process.wait()
  after_files = file_identities()
  if after_files != before_files: raise ValueError("provider executable/script identity changed during collection")
  file_identities = []
  file_identities.extend(before_files)
  return {"schema":PROVIDER_RESULTS_SCHEMA, "plan_sha256":plan_sha, "provider_command":command_list,
          "provider_command_sha256":content_sha256(command_list), "provider_file_identities":file_identities,
          "provider_session":"persistent_jsonl", "provider_environment":provider_environment, "runs":runs}


def _write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
  path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def build_authority(*, mr4_summary: Mapping[str, Any], census: Mapping[str, Any], mr0: Mapping[str, Any],
                    mr6: Mapping[str, Any], plan: Mapping[str, Any], provider_results: Mapping[str, Any],
                    classification_evidence: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
  """Retain the exact MR7 producer authority needed to reproduce a ranking."""
  return {"schema":AUTHORITY_SCHEMA, "mr4_summary":dict(mr4_summary), "census":dict(census), "mr0":dict(mr0),
    "mr6":dict(mr6), "plan":dict(plan), "provider_results":dict(provider_results),
    "classification_evidence":[dict(row) for row in classification_evidence]}


def ranking_from_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
  """Validate provider-bound MR7 inputs and reproduce the canonical ranking."""
  required = {"schema","mr4_summary","census","mr0","mr6","plan","provider_results","classification_evidence"}
  if not isinstance(authority, Mapping) or set(authority) != required or authority.get("schema") != AUTHORITY_SCHEMA:
    raise ValueError("MR7 authority has missing or unknown fields")
  mr4, census, mr0, mr6, plan, provider_results = (authority[name] for name in
    ("mr4_summary","census","mr0","mr6","plan","provider_results"))
  for name,artifact in (("mr0",mr0),("mr6",mr6)):
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("schema"), str) or not artifact["schema"]:
      raise ValueError(f"{name} prerequisite artifact requires an explicit schema")
  expected_plan = build_evidence_plan(mr4, census, arm=plan.get("arm") if isinstance(plan, Mapping) else None)
  if plan != expected_plan: raise ValueError("MR7 authority plan differs from MR4/MR5 derivation")
  measurements, enclosures = adapt_provider_results(plan, provider_results)
  whole = whole_step_from_mr4_summary(mr4, plan["arm"])
  prerequisites = {"mr0":content_sha256(mr0), "mr4":content_sha256(mr4),
                   "mr5":content_sha256(census), "mr6":content_sha256(mr6)}
  classifications = authority["classification_evidence"]
  if not isinstance(classifications, list) or any(not isinstance(row, Mapping) for row in classifications):
    raise ValueError("MR7 authority classification evidence is malformed")
  return build_role_cost_ranking(census, measurements["measurements"], whole, prerequisites=prerequisites,
    enclosures=enclosures, classification_evidence=classifications)


def write_evidence_bundle(output_dir: pathlib.Path, *, mr4_summary: Mapping[str, Any], census: Mapping[str, Any],
                          mr0: Mapping[str, Any], mr6: Mapping[str, Any], plan: Mapping[str, Any],
                          provider_results: Mapping[str, Any]) -> dict[str, Any]:
  """Atomically claim a fresh directory and write all ranker inputs plus hashes."""
  if output_dir.exists(): raise FileExistsError(f"MR7 evidence output already exists: {output_dir}")
  for name,artifact in (("mr0", mr0), ("mr6", mr6)):
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("schema"), str) or not artifact["schema"]:
      raise ValueError(f"{name} prerequisite artifact requires an explicit schema")
  if plan.get("mr4_sha256") != content_sha256(mr4_summary) or plan.get("mr5_sha256") != content_sha256(census):
    raise ValueError("MR7 plan input artifact hash mismatch")
  expected_plan = build_evidence_plan(mr4_summary, census, arm=plan.get("arm"))
  if plan != expected_plan: raise ValueError("MR7 evidence plan differs from the plan derived from MR4/MR5")
  whole = whole_step_from_mr4_summary(mr4_summary, plan["arm"])
  measurements, enclosures = adapt_provider_results(plan, provider_results)
  prerequisites = {"mr0":content_sha256(mr0), "mr4":content_sha256(mr4_summary),
                   "mr5":content_sha256(census), "mr6":content_sha256(mr6)}
  artifacts = {"census.json":dict(census), "measurements.json":measurements, "whole-step.json":whole,
               "enclosures.json":enclosures, "prerequisites.json":prerequisites,
               "plan.json":dict(plan), "provider-results.json":dict(provider_results)}
  authority = build_authority(mr4_summary=mr4_summary, census=census, mr0=mr0, mr6=mr6, plan=plan,
                              provider_results=provider_results)
  artifacts["authority.json"] = authority
  output_dir.mkdir(parents=True, exist_ok=False)
  for name,value in artifacts.items(): _write_json(output_dir / name, value)
  manifest = {"schema":BUNDLE_SCHEMA, "status":"COMPLETE", "arm":plan["arm"],
    "artifacts":{name:{"sha256":content_sha256(value)} for name,value in artifacts.items()},
    "ranker_argv":["--authority","authority.json"]}
  _write_json(output_dir / "manifest.json", manifest)
  return manifest


__all__ = ["AUTHORITY_SCHEMA", "BUNDLE_SCHEMA", "PLAN_SCHEMA", "PROVIDER_PROTOCOL", "PROVIDER_RESULTS_SCHEMA",
  "adapt_provider_results", "build_authority", "build_evidence_plan", "collect_provider_results",
  "ranking_from_authority", "write_evidence_bundle"]
