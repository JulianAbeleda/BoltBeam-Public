"""Compare kernels per role on any tinygrad GPU: search, BoltBeam's own check, matched whole-model A/B, decision.

For every role in a run's route_policy.json this module
  1. asks the tinygrad fork's search provider which devices it can run (its capability action) and picks the first
     of the run's backend's devices that opened (boltbeam/data/search_runtime.json, a table, not code),
  2. checks the provider's device identity against BoltBeam's own reading of the GPU (search/provider_check.py),
  3. builds the role's finite space from what the model's decode binds through on that device (search/role_space.py:
     the fork's decode emitters, or Opt sequences on tinygrad's scheduler), with BubbleBeam's proposal and
     FutureSight's static rejections recorded,
  4. runs it through the provider (`search-full-kernel`'s own runner),
  5. rebuilds every candidate the provider measured from the source it returned, on the model's weight bytes, checks
     it against BoltBeam's reference and times it with BoltBeam's kernel timer: BoltBeam's own time decides, the
     provider's only ordered the search; a kernel BoltBeam cannot rebuild, finds incorrect or reads noisily is "not
     reproduced by BoltBeam" and the role is never promoted,
  6. times the winner against the default in a matched whole-model decode A/B in tinygrad's runtime,
  7. turns that into a CandidateDecision with the existing evaluator, appends it to the run's route ledger, and
     writes the route's status into route_policy.json.

Then the fusions (compare_fusions): kernels that replace several adjacent decode-graph nodes with one launch, proposed
by BubbleBeam from the provider's fused emitters and BoltBeam's own decode graph (search/fusion_space.py), run through
the same campaign and checked the same way, each judged on the in-model time of the tie-out rows it replaces.

The provider's numbers are claims. The kernel numbers this module judges by are BoltBeam's own (the kernel timer);
the whole-model numbers are tinygrad runtime times. None of them is a llama.cpp time.
"""
from __future__ import annotations

import datetime as _dt
import functools, hashlib, json, os, pathlib, statistics, subprocess
from collections.abc import Callable, Mapping
from typing import Any

from boltbeam.role_key import role_key, role_label, role_stem
from boltbeam.artifacts.base import EvidenceFlags, EvidenceRow, EvidenceSource, NormalizedEvidence
from boltbeam.eval.evaluator import evaluate
from boltbeam.ledger.model import CandidateDecision, EvidenceRef, status_for_verdict
from boltbeam.ledger.store import LedgerStore
from boltbeam.manifest import Candidate
from boltbeam.plan.resolved_target import candidate_target, resolved_target_document
from boltbeam.search import provider_check, role_space
from boltbeam.vocab import Verdict

SCHEMA = "boltbeam.kernel_compare.v2"
FOLDER = "kernel_compare"
TIMING_SOURCE = "tinygrad runtime (whole model) and BoltBeam's kernel timer (each kernel alone)"
FORK_URL = "https://github.com/JulianAbeleda/tinygrad-arkey"
PROVIDER = "extra/llm_research/search_provider.py"
PROTOCOL = "tinygrad.search_provider.v1"
AB_DRIVER = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "tinygrad_decode_ab.py"
RUNTIME_TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "search_runtime.json"
REOPEN = "reopen when the plan binds by model-graph role identity or the A/B is re-measured at a new revision"
DEFAULT_KIND = role_space.HEURISTIC


@functools.lru_cache(maxsize=1)
def _runtime_table() -> dict[str, Any]:
  return json.loads(RUNTIME_TABLE.read_text())


def runtime_row(backend:str | None) -> dict[str, Any] | None:
  """The search runtime row of a BoltBeam target backend (Metal, CUDA, AMD), or None when the table has none."""
  rows = _runtime_table()["backends"]
  return next((dict(v) for k, v in rows.items() if backend and k.lower() == str(backend).lower()), None)


def decode_env(backend:str | None, device:str) -> dict[str, str]:
  """The whole-model A/B environment: the device the search served on and the backend's recorded switches."""
  return {"DEV": device, **dict((runtime_row(backend) or {}).get("decode_env") or {})}


def _sha256_file(path:pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""): digest.update(chunk)
  return digest.hexdigest()


def _now() -> str:
  return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- readiness: what the screen says before anything starts ---------------------------------------------------

def default_fork_root() -> pathlib.Path:
  """$BOLTBEAM_TINYGRAD_ROOT (or the older TINYGRAD_ROOT; target/tinygrad_root.py reads both), else the fork the
  engine scan saved (collectors/engine_scan.py), else tinygrad-arkey-exp next to this BoltBeam checkout."""
  from boltbeam.collectors import engine_scan
  from boltbeam.target.tinygrad_root import BOLTBEAM_TINYGRAD_ROOT_ENV, configured_root
  if env := configured_root():
    return pathlib.Path(env).expanduser()
  if kept := engine_scan.saved(BOLTBEAM_TINYGRAD_ROOT_ENV):
    return pathlib.Path(kept)
  return pathlib.Path(__file__).resolve().parents[2].parent / "tinygrad-arkey-exp"


PYTHON_ENV = "BOLTBEAM_TINYGRAD_PYTHON"  # the fork's interpreter, outside the checkout
VENV_ENV = "BOLTBEAM_TINYGRAD_VENV"  # or the venv folder that holds it (bin/python)


def fork_python(root:pathlib.Path) -> pathlib.Path:
  """The fork's interpreter: $BOLTBEAM_TINYGRAD_PYTHON, else $BOLTBEAM_TINYGRAD_VENV/bin/python, else the fork's own
  .venv folder when git ignores it. Never a link placed in the checkout: git counts that as an untracked file, the
  provider then reports a dirty tree, and the search refuses every candidate."""
  if env := os.environ.get(PYTHON_ENV):
    return pathlib.Path(env).expanduser()
  if env := os.environ.get(VENV_ENV):
    return pathlib.Path(env).expanduser() / "bin" / "python"
  return root / ".venv" / "bin" / "python"


def _venv_link(root:pathlib.Path) -> bool:
  """The python would come from a .venv link inside the checkout (no env set)."""
  return not (os.environ.get(PYTHON_ENV) or os.environ.get(VENV_ENV)) and (root / ".venv").is_symlink()


def _has_numpy(python:pathlib.Path) -> bool:
  venv = python.parent.parent
  if any((venv / "lib").glob("python3*/site-packages/numpy")):
    return True
  try:
    return subprocess.run([str(python), "-c", "import numpy"], capture_output=True, timeout=60).returncode == 0
  except (OSError, subprocess.TimeoutExpired):
    return False


def readiness(root:pathlib.Path | None = None, model:str | None = None) -> dict[str, Any]:
  """Can this machine compare kernels now? Each missing piece names the one command that fixes it."""
  root = pathlib.Path(root) if root else default_fork_root()
  venv = f"cd {root} && python3.12 -m venv .venv && .venv/bin/pip install numpy"
  out: dict[str, Any] = {"ready": False, "fork": str(root), "missing": None, "message": None, "fix": None}
  if not (root / PROVIDER).is_file():
    return out | {"missing": "fork", "message": f"The tinygrad fork is not at {root}.",
                  "fix": f"git clone -b exp {FORK_URL} {root}"}
  if _venv_link(root):
    return out | {"missing": "venv", "message": f"{root / '.venv'} is a link. Git counts it as a change, so the "
                  "search refuses every kernel.",
                  "fix": f"rm {root / '.venv'} && export {PYTHON_ENV}={(root / '.venv').resolve() / 'bin' / 'python'}"}
  python = fork_python(root)
  if not python.is_file():
    return out | {"missing": "venv", "message": f"The fork's python is not at {python}.", "fix": venv}
  if not _has_numpy(python):
    return out | {"missing": "numpy", "message": "The fork's python has no numpy.",
                  "fix": f"{python} -m pip install numpy"}
  if model is not None and not pathlib.Path(model).expanduser().is_file():
    return out | {"missing": "model", "message": f"The model file is not at {model}.", "fix": None}
  return out | {"ready": True}


def fork_changes(root:pathlib.Path) -> list[str]:
  """The fork's uncommitted paths, as git status shows them. The provider refuses to search a checkout with any."""
  try:
    text = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                          check=True).stdout
  except (OSError, subprocess.CalledProcessError) as exc:
    return [f"git status failed: {exc}"]
  return [line[3:] for line in text.splitlines() if line.strip()]


def clean_refusal(root:pathlib.Path) -> str | None:
  """Why the search cannot run on this fork checkout, or None. Checked before the search, so one sentence says it
  instead of every candidate being refused with provider_describe:dirty_or_missing_clean_evidence."""
  changes = fork_changes(root)
  if not changes:
    return None
  shown = ", ".join(changes[:5]) + (f" and {len(changes) - 5} more" if len(changes) > 5 else "")
  hint = f" Put the venv outside the checkout and set {PYTHON_ENV}." if any(c.rstrip("/") == ".venv" for c in changes) else ""
  return f"the tinygrad fork at {root} has uncommitted changes ({shown}); the search runs on a clean checkout only.{hint}"


# --- the request for one role ---------------------------------------------------------------------------------

def _ask(root:pathlib.Path, action:str, payload:Mapping[str, Any], timeout_s:float = 180.0) -> dict[str, Any]:
  """One request to the fork's provider, in its own process; the result, or RuntimeError with the provider's words."""
  line = json.dumps({"protocol": PROTOCOL, "request_id": action, "action": action, "payload": dict(payload)})
  proc = subprocess.run([str(fork_python(root)), PROVIDER], cwd=root, input=line + "\n", capture_output=True, text=True,
                        timeout=timeout_s, env={**os.environ, "PYTHONPATH": "."})
  reply = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
  if reply.get("status") != "ok":
    why = (reply.get("error") or {}).get("message") or proc.stderr.strip()[-300:] or "no reply"
    raise RuntimeError(f"provider {action} failed: {why}")
  return reply["result"]


def capability(root:pathlib.Path, devices:list[str]) -> dict[str, Any]:
  """The provider's own account of the devices it can run: one row per named device, opened or with its reason."""
  return _ask(root, "capability", {"devices": list(devices)})


def describe(root:pathlib.Path, device:str | None = None, *, shapes:list[dict[str, int]] | None = None) -> dict[str, Any]:
  """The provider's describe; with shapes ([{n, k}]), its fused emitters' validate verdicts at each shape too."""
  return _ask(root, "describe", ({"device": device} if device else {}) | ({"shapes": list(shapes)} if shapes else {}))


def pick_device(backend:str | None, reply:Mapping[str, Any]) -> tuple[str | None, str | None]:
  """(device, None) for the first of the backend's devices the provider opened, else (None, why)."""
  row = runtime_row(backend)
  if row is None:
    return None, f"the search runtime table (data/search_runtime.json) has no row for {backend or 'an unknown backend'}"
  rows = {str(r.get("device")).upper(): r for r in reply.get("devices") or [] if isinstance(r, Mapping)}
  for name in row["tinygrad_devices"]:
    got = rows.get(name.upper())
    if got and got.get("opened"):
      return str(got["device"]), None
  said = "; ".join(f"{name}: {(rows.get(name.upper()) or {}).get('reason') or 'not reported'}" for name in row["tinygrad_devices"])
  return None, f"the provider runs on none of this {backend} machine's devices ({said})"


TOLERANCE = {"atol": 0.001, "rtol": 0.001}


def semantic_workload(route:Mapping[str, Any], *, tensor_name:str, model_sha:str, target:Mapping[str, Any]) -> dict[str, Any]:
  """The exact semantic workload of one role: the recorded tensor, its shape and packed format, the one-token GEMV."""
  role, quant = str(route["role"]), str(route["quant"]).upper()
  n, k = (int(x) for x in route["shape"])
  module = tensor_name[:-len(".weight")] if tensor_name.endswith(".weight") else tensor_name
  identity = {"phase": "decode", "tensor_name": tensor_name, "module_path": module, "role": role, "logical_m": 1,
              "logical_n": n, "logical_k": k, "source_quant_storage": quant, "source_layout": "transposed_row_major",
              "module_representation": "gguf_packed", "input_dtype": "fp16", "output_dtype": "fp16", "accumulator_dtype": "fp32"}
  return {"schema": "tinygrad.semantic_provider_workload.v1", "model_hash": model_sha, "target": dict(target),
          "semantic_identity": identity, "operation": "matmul", "shape": {"m": 1, "n": n, "k": k},
          "operands": {"a": {"dtype": "fp16"}, "b": {"quantization": quant, "layout": "transposed_row_major"}, "c": {"dtype": "fp16"}},
          "tolerance": dict(TOLERANCE), "fixture_shape_substitution": "forbidden"}


def propose(route:Mapping[str, Any], *, tensor_name:str, model:str, model_id:str, target_id:str, describe_result:Mapping[str, Any],
            model_sha:str, run_id:str, timestamp:str, sm_count:int | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
  """BubbleBeam, then the population, then FutureSight, for one role: (campaign request, population, FutureSight
  evidence, the role's proposal inputs). The documented pipeline (docs/semantic-campaign.md) steps 1 to 3."""
  from boltbeam.search.futuresight_adapter import assess_population, propose_request, target_facts
  from boltbeam.search.semantic.semantic_population_export import export_population
  observed = describe_result["target"]
  resolved = resolved_target_document(target_id, observed)
  workload = semantic_workload(route, tensor_name=tensor_name, model_sha=model_sha, target=candidate_target(resolved))
  facts, _missing = target_facts(target_id, describe_result)
  inputs = role_space.proposal_inputs(describe_result, facts, str(route["quant"]), subgroup_size=workload["target"].get("subgroup_size"),
                                      shape=(int(route["shape"][0]), int(route["shape"][1])), sm_count=sm_count)
  if inputs.get("why"):
    raise ValueError(inputs["why"])
  role, quant = str(route["role"]), str(route["quant"])
  spec = {"semantic_workload": workload, "schedule": inputs["schedule"], "resolved_target": resolved, "gguf_path": str(model),
          "execution": {"shape_mode": "exact_workload", "warmups": 2, "samples": 7},
          "request_id": f"{run_id}-{role}-{quant.lower()}", "run_id": run_id, "timestamp": timestamp,
          "budget": {"max_candidates": len(inputs["coupled_rows"]) + 2, "timeout_s": 600.0},
          "axis_choices": inputs["axis_choices"], "coupled_rows": inputs["coupled_rows"]}
  request, _missing = propose_request(spec, describe_result)          # 1. BubbleBeam
  population = export_population(request)                           # 2. the population
  evidence = assess_population(population)                           # 3. FutureSight
  return request, population, evidence, inputs


# --- reading a search result ----------------------------------------------------------------------------------

def plan_text(schedule:Mapping[str, Any]) -> str:
  """One plan in words: the default kernel, an Opt sequence, or an emitter with its schedule values."""
  kind = str(schedule.get("plan_kind") or "")
  if kind == DEFAULT_KIND: return "default kernel"
  if kind == role_space.EMITTER_PLAN_KIND:
    mem = schedule.get("memory") or {}
    return (f"{(schedule.get('compute') or {}).get('family')}: {(schedule.get('launch') or {}).get('threads')} threads, "
            f"{(schedule.get('tile') or {}).get('n')} rows a block, {((mem.get('b') or {}).get('vector_width'))}-byte weight loads, "
            f"activation in {(mem.get('a') or {}).get('space')} memory, {(schedule.get('pipeline') or {}).get('stage_count')} accumulators")
  return _plan_text(list(schedule.get("transforms") or []), kind)


def _plan_text(transforms:list[Mapping[str, Any]], kind:str) -> str:
  if kind == DEFAULT_KIND: return "default kernel"
  if not transforms: return "no opts"
  return " ".join(f"{t['op']} {t['arg']}" + (f" on axis {t['axis']}" if t.get("axis") else "") for t in transforms)


def _median(r:Mapping[str, Any] | None) -> float | None:
  return ((r or {}).get("measurement") or {}).get("median_ns")


def kernel_record(row:Mapping[str, Any] | None) -> dict[str, Any] | None:
  """The kernel the provider says it compiled for a population row: source, function, launch, buffers."""
  compiled = (((row or {}).get("worker") or {}).get("provider") or {}).get("compile") or {}
  record = (compiled.get("result") or {}).get("kernel") if isinstance(compiled, Mapping) else None
  return dict(record) if isinstance(record, Mapping) else None


def _entry(row:Mapping[str, Any]) -> dict[str, Any]:
  sched = row["candidate"]["schedule"]
  generated = (row.get("worker") or {}).get("generated") or {}
  return {"candidate_hash": row["candidate_hash"], "plan_kind": sched["plan_kind"], "schedule": dict(sched),
          "transforms": list(sched.get("transforms", [])), "plan": plan_text(sched), "plan_hash": generated.get("plan_hash"),
          "median_ns": _median(row), "is_default": sched["plan_kind"] == DEFAULT_KIND}


def summarize_search(result:Mapping[str, Any], installed:Mapping[str, Any] | None = None) -> dict[str, Any]:
  """The provider's winner, the default kernel, the model's own kernel (the row the decode installs) and the counts,
  from one full-kernel search result. Ranks and times here are the provider's claims."""
  measured = [r for r in result.get("population", []) if r.get("state") == "MEASURED"]
  measured.sort(key=lambda r: (r.get("rank") or 1 << 30))
  default = next((r for r in measured if r["candidate"]["schedule"]["plan_kind"] == DEFAULT_KIND), None)
  model = next((r for r in measured if role_space.is_row(r["candidate"], installed)), None)
  out = {"status": result.get("status"), "counts": dict(result.get("counts", {})), "winner": None,
         "default_median_ns": _median(default), "model_kernel": _entry(model) if model else None}
  if measured:
    out["winner"] = _entry(measured[0])
    out["winner"]["is_model_kernel"] = model is not None and measured[0]["candidate_hash"] == model["candidate_hash"]
  return out


def time_roles(run:pathlib.Path, *, root:pathlib.Path | None = None) -> dict[str, Any]:
  """Time every role inside a real decode in tinygrad's runtime (collectors/tinygrad_role_time.py)."""
  from boltbeam.collectors import tinygrad_role_time
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest
  root = pathlib.Path(root) if root else default_fork_root()
  manifest = load_manifest(run)
  model = str(manifest.get("model_path") or "")
  ready = readiness(root, model)
  if not ready["ready"]:
    raise RuntimeError(f"{ready['message']} Fix: {ready['fix']}" if ready.get("fix") else ready["message"])
  trace = tinygrad_role_time.collect(run, root=root, python=fork_python(root), model=model,
                                     model_id=str(manifest.get("model_id")), target=get_target(manifest.get("target_id")))
  from boltbeam.workflow.screen import _measured_vs_ceiling, _optional
  ceil = _measured_vs_ceiling(manifest, _optional(run, "model_profile.json"))
  table = tinygrad_role_time.loss(ceil.get("_roles") or [], trace, ceil.get("floor_ms"), ceil.get("band"))
  if table and table["status"] != "measured":  # say it where the job ends, not only on the next screen
    raise RuntimeError(table["reason"])
  return trace


def role_losses(run:pathlib.Path) -> dict[tuple[Any, ...], dict[str, Any]]:
  """The per-role loss rows of a run, by role_key (role, quant, rows, k); empty before the roles were timed."""
  from boltbeam.collectors import tinygrad_role_time
  from boltbeam.workflow.common import read_json
  from boltbeam.workflow.screen import _measured_vs_ceiling, _optional
  from boltbeam.workflow.common import load_manifest
  path = run / tinygrad_role_time.TRACE
  if not path.is_file(): return {}
  ceil = _measured_vs_ceiling(load_manifest(run), _optional(run, "model_profile.json"))
  table = tinygrad_role_time.loss(ceil.get("_roles") or [], read_json(path), ceil.get("floor_ms"), ceil.get("band")) or {"roles": []}
  return {role_key(r): r for r in table["roles"]}  # an incomplete table has no roles


def kernel_numbers(summary:Mapping[str, Any], in_model:Mapping[str, Any] | None = None,
                   verify:Mapping[str, Any] | None = None, tensors:int | None = None) -> dict[str, Any] | None:
  """Number one of two: the winning plan alone, by BoltBeam's kernel timer (less the dispatch floor) when the check
  ran, else the provider's claim, beside the model's own kernel. The model's time is the role's kernels inside the
  running model (tinygrad's role time), per tensor read: a plan reads one of the role's tensors a call, and the model
  may read several in one call (a fused gate and up), so the role's in-model time is divided by its tensors per
  token (the model profile's count) when that is known, else by its calls. Never a stand-in for the whole-model
  number."""
  w = summary.get("winner")
  if not w or w.get("median_ns") is None: return None
  mine = ((verify or {}).get("winner") or {}).get("boltbeam") or {}
  model = ((verify or {}).get("model_kernel") or {}).get("boltbeam") or {}
  plan_us = mine.get("us_less_floor") if mine.get("status") == "measured" else None
  out = {"plan_us": plan_us if plan_us is not None else w["median_ns"] / 1000.0,
         "plan_us_source": "BoltBeam's kernel timer, less the dispatch floor" if plan_us is not None else "the provider's claim",
         "provider_plan_us": w["median_ns"] / 1000.0,
         "reference_us": (summary.get("default_median_ns") or 0) / 1000.0 if summary.get("default_median_ns") is not None else None,
         "reference": "search fixture heuristic (the provider's time)",
         "model_kernel_us": model.get("us_less_floor") if model.get("status") == "measured" else None,
         "model_us_per_call": None, "faster_than_model": None}
  if in_model and in_model.get("calls_per_token"):
    # the baseline: the role's own kernels inside the running model, GPU time per tensor at the same shape
    reads = tensors or in_model["calls_per_token"]
    per_call = in_model["actual_ms"] * 1000.0 / reads
    out.update(model_us_per_call=per_call, faster_than_model=out["plan_us"] < per_call, tensors_per_token=reads,
               model_calls_per_token=in_model["calls_per_token"])
    if tensors and abs(tensors - in_model["calls_per_token"]) > 1e-9:
      out["per_tensor_note"] = (f"the model reads {tensors} tensors of this role per token in {in_model['calls_per_token']:.0f} "
                                "calls; the plan reads one a call, so the model's time is divided per tensor")
  return out


def plan_id(role:str, quant:str, winner:Mapping[str, Any]) -> str:
  return f"tinygrad-plan:{role}:{quant}:{(winner.get('plan_hash') or winner['candidate_hash'])[:12]}"


# --- BoltBeam's own check of the provider's claims ------------------------------------------------------------

def role_bytes(model:str, quant:str, rows:int, cols:int, tensor_name:str | None = None) -> tuple[str, bytes]:
  """The role's own weight bytes from the model file (the tensor of this quant and shape)."""
  from boltbeam.collectors import metal_native as native
  from boltbeam.profile.gguf import read_gguf_layout
  _, tensors, data_start = read_gguf_layout(pathlib.Path(model))
  name, offset = native.tensor_for(quant, rows, cols, tensors, tensor_name)
  size = rows * (cols // native.BLOCK_ELEMS[quant]) * native.BLOCK_BYTES[quant]
  with open(model, "rb") as f:
    f.seek(data_start + offset)
    return name, f.read(size)


class Checker:
  """BoltBeam's bridge on this machine, its flush and dispatch floor, opened once for a run's roles."""

  def __init__(self, backend:str, band:float, bridge=None):
    from boltbeam.collectors import kernel_timer
    self.backend, self.band = backend, band
    self.bridge = bridge or kernel_timer.bridge_for(backend)
    self.flusher = kernel_timer.Flusher(self.bridge, backend)
    self.floor_us = self.flusher.floor_us()
    self.libraries: dict[str, int] = {}

  def facts(self) -> dict[str, Any]:
    return provider_check.bridge_facts(self.bridge)

  def __call__(self, summary:Mapping[str, Any], result:Mapping[str, Any], *, model:str, quant:str, rows:int, cols:int,
               role:str, tensor_name:str | None = None) -> dict[str, Any]:
    """Every candidate the provider measured correct, rebuilt and timed by BoltBeam on the role's own bytes, each
    claim judged; BoltBeam's own winner is the fastest whose claim reproduced."""
    from boltbeam.collectors import boltbeam_gemv
    tensor, weights = role_bytes(model, quant, rows, cols, tensor_name)
    x = boltbeam_gemv.vector(f"{role}:{quant}", cols)
    out: dict[str, Any] = {"tensor": tensor, "floor_us": self.floor_us, "band": self.band, "candidates": []}
    for row in sorted((r for r in result.get("population", []) if r.get("state") == "MEASURED"), key=lambda r: r.get("rank") or 1 << 30):
      entry = _entry(row)
      record = kernel_record(row)
      if record is None:
        mine = {"status": "not_rebuilt", "reason": "the provider returned no kernel source and launch for this candidate"}
      else:
        mine = provider_check.retime(self.bridge, self.flusher, self.floor_us, record, quant=quant, rows=rows, cols=cols,
                                     weights=weights, x=x, label=entry["plan"], libraries=self.libraries)
      out["candidates"].append({"candidate_hash": entry["candidate_hash"], "plan": entry["plan"], "provider_rank": row.get("rank"),
                                "boltbeam": mine, **provider_check.judge(entry.get("median_ns"), mine, self.band)})
    return out


  def fusion(self, result:Mapping[str, Any], *, model:str, proposal:Mapping[str, Any]) -> dict[str, Any]:
    """Every fused candidate the provider measured correct, rebuilt by BoltBeam on the fusion's own tensors, checked
    against the covered nodes computed by BoltBeam and timed; then each covered node's kernel alone (the unfused
    side, from the provider's compile of the fastest candidate), checked and timed the same way."""
    from boltbeam.search import fusion_space
    inst = proposal["instance"]
    rows, cols = (int(x) for x in inst["shape"])
    quant, order = str(proposal["quant"]), fusion_space.adjacency(proposal["covers"])["order"]
    weights = {node: role_bytes(model, quant, rows, cols, name)[1] for node, name in inst["tensors"].items()}
    out: dict[str, Any] = {"tensors": dict(inst["tensors"]), "floor_us": self.floor_us, "band": self.band, "candidates": [], "unfused": []}
    measured = sorted((r for r in result.get("population", []) if r.get("state") == "MEASURED"), key=lambda r: r.get("rank") or 1 << 30)
    for row in measured:
      entry = _entry(row)
      record = kernel_record(row)
      mine = ({"status": "not_rebuilt", "reason": "the provider returned no kernel source and launch for this candidate"} if record is None
              else provider_check.retime_labeled(self.bridge, self.flusher, self.floor_us, record, nodes=order, quant=quant, rows=rows,
                                                 cols=cols, weights=weights, seed=proposal["name"], label=entry["plan"], libraries=self.libraries))
      out["candidates"].append({"candidate_hash": entry["candidate_hash"], "plan": entry["plan"], "provider_rank": row.get("rank"),
                                "boltbeam": mine, **provider_check.judge(entry.get("median_ns"), mine, self.band)})
    timed = [c for c in out["candidates"] if c["boltbeam"].get("status") == "measured"]
    pick = min(timed, key=lambda c: c["boltbeam"]["us_less_floor"], default=None)
    source = next((r for r in measured if pick and r["candidate_hash"] == pick["candidate_hash"]), measured[0] if measured else None)
    compiled = ((((source or {}).get("worker") or {}).get("provider") or {}).get("compile") or {}).get("result") or {}
    for node in (compiled.get("fusion") or {}).get("unfused") or []:
      name = str(node.get("node"))
      if name not in order:
        out["unfused"].append({"node": name, "boltbeam": {"status": "not_rebuilt", "reason": "not a node this fusion covers"}})
        continue
      mine = provider_check.retime_labeled(self.bridge, self.flusher, self.floor_us, node.get("kernel") or {}, nodes=[name], quant=quant,
                                           rows=rows, cols=cols, weights=weights, seed=proposal["name"], label=f"{name} alone",
                                           libraries=self.libraries)
      out["unfused"].append({"node": name, "boltbeam": mine, **provider_check.judge(None, mine, self.band)})
    return out


def checked(summary:Mapping[str, Any], verify:Mapping[str, Any]) -> dict[str, Any]:
  """The provider's winner and the model's own kernel as BoltBeam judged them, and BoltBeam's own winner: the fastest
  candidate BoltBeam measured correct (its time less the floor). Its claim must reproduce for any decision: a slower
  candidate whose claim happened to reproduce never stands in for it, since that would refute plans BoltBeam itself
  found faster."""
  rows = {c["candidate_hash"]: c for c in verify.get("candidates") or []}
  out = {k: rows.get((summary.get(k) or {}).get("candidate_hash")) for k in ("winner", "model_kernel")}
  timed = [c for c in rows.values() if (c.get("boltbeam") or {}).get("status") == "measured"]
  good = [c for c in rows.values() if c.get("verdict") == provider_check.REPRODUCED]
  out["boltbeam_winner"] = min(timed, key=lambda c: c["boltbeam"]["us_less_floor"], default=None)
  out["reproduced"], out["not_reproduced"] = len(good), len(rows) - len(good)
  return out


# --- the matched whole-model A/B ------------------------------------------------------------------------------

def run_ab(root:pathlib.Path, model:str, *, transforms:list[Mapping[str, Any]], shape:tuple[int, int],
           pairs:int = 3, timeout_s:float = 1800.0, env:Mapping[str, str] | None = None) -> dict[str, Any]:
  """The driver loads the model once and alternates default and plan arms in that one process."""
  argv = [str(fork_python(root)), str(AB_DRIVER), "--model", str(model), "--opts", json.dumps(list(transforms)),
          "--shape", f"{shape[0]},{shape[1]}", "--pairs", str(pairs)]
  proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout_s,
                        env={**os.environ, "PYTHONPATH": ".", **(env or {})})
  lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
  if proc.returncode != 0 or not lines:
    raise RuntimeError(f"decode A/B failed: {(proc.stderr or proc.stdout).strip()[-400:]}")
  return json.loads(lines[-1])


def matched_ab(raw:Mapping[str, Any], env:Mapping[str, str] | None = None) -> dict[str, Any]:
  """Medians and noise from the interleaved arms. The plan counts as bound only when it reached decode calls:
  a plan applied to other kernels leaves both arms running the same decode."""
  b = [x for arm in raw["baseline"] for x in arm["tok_s_samples"]]
  c = [x for arm in raw["candidate"] for x in arm["tok_s_samples"]]
  if not c: c = list(b)
  bm, cm = statistics.median(b), statistics.median(c)
  first = raw["baseline"][0]["tokens"]
  binding = dict(raw["binding"])
  return {"timing_source": TIMING_SOURCE, "context": raw["context"], "runtime_env": dict(env or {}),
          "pairs": raw["pairs"], "baseline_tok_s": bm, "candidate_tok_s": cm, "delta_pct": (cm - bm) / bm * 100.0,
          # noise is the wider of the two sides' ranges; overlapping ranges mean no measurable change
          "spread_pct": max(max(b) - min(b), max(c) - min(c)) / bm * 100.0,
          "ranges_overlap": min(c) <= max(b) and min(b) <= max(c),
          "baseline_samples": b, "candidate_samples": c,
          "token_match": all(arm["tokens"] == first for arm in raw["candidate"]),
          "deterministic": all(arm["tokens"] == first for arm in raw["baseline"]),
          "binding": binding, "route_bound": binding.get("calls_reached", 0) > 0 and binding.get("errors", 0) == 0,
          "load_s": raw.get("load_s"), "max_rss_bytes": raw.get("max_rss_bytes")}


# --- the decision ---------------------------------------------------------------------------------------------

def decide(route:Mapping[str, Any], winner:Mapping[str, Any], ab:Mapping[str, Any], *, model_id:str, target_id:str,
           ab_path:str, ab_sha:str) -> CandidateDecision:
  """The existing evaluator judges the A/B: correctness, binding, speed tier, rollback, target completeness."""
  role, quant = str(route["role"]), str(route["quant"])
  cid = plan_id(role, quant, winner)
  candidate = Candidate(candidate_id=cid, workload="decode", quant=(quant,), roles=(role,),
                        required_evidence_kinds=("wd_speed",), rollback={"kernel_plan": "default"},
                        route_family=(route.get("candidates") or [""])[0], origin="search",
                        description=f"{winner['plan']} for {role} {quant}")
  row = EvidenceRow(kind="wd_speed", metric="tok_s", value=float(ab["candidate_tok_s"]), unit="tok/s", role=role,
                    quant=quant, context=int(ab["context"]),
                    extra={"baseline_tok_s": float(ab["baseline_tok_s"]), "spread_pct": float(ab["spread_pct"]),
                           "timing_source": TIMING_SOURCE})
  evidence = NormalizedEvidence(model_id=model_id, target_id=target_id, workload="decode",
                                source=EvidenceSource("tinygrad", "boltbeam/runtime/tinygrad_decode_ab.py", ab_path, f"sha256:{ab_sha}"),
                                rows=(row,), contexts=(int(ab["context"]),),
                                flags=EvidenceFlags(route_bound=bool(ab["route_bound"]), token_match=bool(ab["token_match"]),
                                                    deterministic=bool(ab["deterministic"])))
  return evaluate(candidate, [evidence], model_id=model_id, target_id=target_id, workload="decode")


def default_wins(route:Mapping[str, Any], winner:Mapping[str, Any], *, model_id:str, target_id:str,
                 search_path:str, search_sha:str, model_kernel:bool = False) -> CandidateDecision:
  """The search's fastest kernel is the reference or the model's own kernel: no plan to try, a firm refutation."""
  role, quant = str(route["role"]), str(route["quant"])
  which = "the model's own kernel" if model_kernel else "the search reference kernel"
  ref = EvidenceRef(evidence_id=search_sha, path=search_path, kind="full_kernel_search", fingerprint=f"sha256:{search_sha}",
                    claim=f"{which} is the fastest measured {role} {quant} kernel")
  return CandidateDecision(candidate_id=plan_id(role, quant, winner), model_id=model_id, target_id=target_id,
                           workload="decode", verdict=Verdict.REFUTE.value, evidence=(ref,),
                           reason=f"no searched plan beat {which} alone",
                           next_action="keep the default kernel")


def route_status(verdict:str) -> str:
  """route_policy.json status for a verdict: promote and refute are firm; anything else is not decided yet."""
  return {Verdict.PROMOTE.value: "promoted", Verdict.REFUTE.value: "refuted"}.get(verdict, "blocked")


# --- one run ----------------------------------------------------------------------------------------------------

def _calls(row:Mapping[str, Any] | None) -> str:
  return f"{row['calls_per_token']:.0f}" if row else "unknown number of"


def _git_state(repo:pathlib.Path) -> tuple[str | None, bool]:
  try:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
    return rev, dirty
  except (OSError, subprocess.CalledProcessError):
    return None, True


def _write(path:pathlib.Path, obj:Any) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
  path.write_text(text)
  return hashlib.sha256(text.encode()).hexdigest()


class ProviderRefused(RuntimeError):
  """The provider's device is not the GPU BoltBeam reads, or the provider runs on none of this machine's devices."""


def _target_band(target_id:str) -> float:
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.screen import plausibility_band
  return plausibility_band(get_target(target_id))


def compare_run(run:pathlib.Path, *, root:pathlib.Path | None = None, say:Callable[[str], None] = print,
                search:Callable[..., dict[str, Any]] | None = None, ab_run:Callable[..., dict[str, Any]] | None = None,
                describe_fn:Callable[..., dict[str, Any]] | None = None,
                capability_fn:Callable[[pathlib.Path, list[str]], dict[str, Any]] | None = None,
                checker:Any = None, only:set[str] | None = None, ab_only_if_faster:bool = False,
                step:Callable[[int, int], None] | None = None, fusions:bool = True,
                tensors_of_model:list[tuple[str, tuple[int, ...], int, int]] | None = None) -> dict[str, Any]:
  """Compare kernels for every route of one run and write the outcome into the run folder.

  `search`, `ab_run`, `describe_fn`, `capability_fn` and `checker` default to the real provider, driver and BoltBeam
  bridge; tests pass doubles (and `tensors_of_model`, the GGUF's tensor list, for the fusions). fusions=False skips
  the fusion stage. ab_only_if_faster skips the whole-model A/B for a role whose best plan alone is not
  faster than the model's own kernel per call (Run's search stage). step(done, total) is called after each role.
  """
  from boltbeam.target.targets import get_target
  from boltbeam.workflow.common import load_manifest, read_json, write_json
  root = pathlib.Path(root) if root else default_fork_root()
  manifest = load_manifest(run)
  model = str(manifest.get("model_path") or "")
  ready = readiness(root, model) if search is None else {"ready": True}
  if not ready["ready"]:
    raise RuntimeError(f"{ready['message']} Fix: {ready['fix']}" if ready.get("fix") else ready["message"])
  if search is None and (why := clean_refusal(root)):
    raise RuntimeError(why)
  target_id, model_id = str(manifest.get("target_id")), str(manifest.get("model_id"))
  backend = get_target(target_id).backend
  # the device: the first of this backend's devices the provider says it opened
  row = runtime_row(backend)
  reply = (capability_fn or capability)(root, list((row or {}).get("tinygrad_devices") or []))
  device, why = pick_device(backend, reply)
  if device is None:
    raise ProviderRefused(why)
  facts = (describe_fn or describe)(root, device)
  # the provider's device against BoltBeam's own reading of this machine's GPU
  if checker is None:
    checker = Checker(backend, _target_band(target_id))
  ident = provider_check.identity(facts.get("target") or {}, checker.facts())
  if not ident["passed"]:
    raise ProviderRefused(ident["reason"])
  policy = read_json(run / "route_policy.json")
  routes = [r for r in policy.get("routes", []) if r.get("role") and r.get("quant") and r.get("shape")]
  if only: routes = [r for r in routes if f"{r['role']}:{r['quant']}" in only]
  if search is None and not (run / "tinygrad_timing_trace.json").is_file():
    say("role-time: start")
    time_roles(run, root=root)
  losses = role_losses(run)
  profile = read_json(run / "model_profile.json") if (run / "model_profile.json").is_file() else {}
  tensors = {role_key(r): r.get("tensor_name") for r in profile.get("roles", []) if r.get("tensor_name")}
  counts = {role_key(r): int(r["count"]) for r in profile.get("roles", []) if r.get("count")}
  # the biggest loss first: that is where a better kernel can save the most
  routes.sort(key=lambda r: -losses.get(role_key(r), {}).get("lost_ms", float("-inf")))
  provider_rev = (facts.get("provider_revision") or {}).get("revision")
  bb_rev, bb_dirty = _git_state(pathlib.Path(__file__).resolve().parents[2])
  model_sha = _sha256_file(pathlib.Path(model)) if search is None else "0" * 63 + "1"
  folder = run / FOLDER
  timestamp = _now()
  env = decode_env(backend, device)
  if search is None:
    def search(request, evidence):  # noqa: E306 - the documented campaign, one provider process per role
      from boltbeam.search.full_kernel.tinygrad_full_kernel import PersistentJSONLSession
      from boltbeam.search.semantic_campaign_cli import run_request
      with PersistentJSONLSession((str(fork_python(root)), "-m", "extra.llm_research.search_provider"), cwd=str(root)) as session:
        return run_request(request, provider_command=[], futuresight_evidence=evidence, provider_session=session,
                           requested_provider_revision=provider_rev)
  if ab_run is None:
    def ab_run(transforms, shape):  # noqa: E306
      return run_ab(root, model, transforms=transforms, shape=shape, env=env)
  record = {"schema": SCHEMA, "run": run.name, "model_id": model_id, "target_id": target_id, "model_path": model,
            "model_sha256": model_sha, "provider_revision": provider_rev, "boltbeam_revision": bb_rev,
            "boltbeam_dirty": bb_dirty, "fork": str(root), "timestamp": timestamp, "timing_source": TIMING_SOURCE,
            "device": device, "capability": reply, "identity": ident, "band": checker.band, "runtime_env": env,
            "note": "Kernel times are BoltBeam's kernel timer on the source the provider returned; whole-model times are "
                    "tinygrad runtime times; none is a llama.cpp time. Provider times are claims, kept beside.",
            "roles": []}
  ledger = LedgerStore(run / "route_ledger.jsonl")
  say(f"compare roles: {len(routes)} on {device}")
  for route in routes:
    role, quant = str(route["role"]), str(route["quant"])
    key, stem, rk = role_label(role, quant, route["shape"]), role_stem(role, quant, route["shape"]), role_key(route)
    n, k = (int(x) for x in route["shape"])
    say(f"role {key}: search")
    tensor_name = route.get("tensor_name") or tensors.get(rk)
    try:
      request, population, evidence, inputs = propose(route, tensor_name=str(tensor_name), model=model, model_id=model_id,
                                                      target_id=target_id, describe_result=facts, model_sha=model_sha,
                                                      run_id=f"{run.name}-compare", timestamp=timestamp,
                                                      sm_count=checker.facts().get("sm_count"))  # BoltBeam's bridge, after the identity check
    except Exception as exc:  # no space for this role: say why, the other roles go on
      request, population, evidence, inputs = None, None, None, {"why": str(exc)}
    if request is not None:
      _write(folder / f"{stem}-search-request.json", request | {"futuresight_evidence": evidence})
      _write(folder / f"{stem}-population.json", population)
      try:
        result = search(request, evidence)                           # 4. the campaign measures the survivors
      except Exception as exc:  # a provider failure blocks this role, not the others
        result = {"status": "BLOCKED", "counts": {}, "population": [], "error": str(exc),
                  "futuresight_static_evidence": evidence}
    else:
      result = {"status": "BLOCKED", "counts": {}, "population": [], "error": inputs["why"]}
    search_sha = _write(folder / f"{stem}-search-result.json", result)
    summary = summarize_search(result, inputs.get("installed_row"))
    rejected = [r for r in (evidence or {}).get("rejections") or [] if "candidate_hash" in r]
    row: dict[str, Any] = {"role": role, "quant": quant, "shape": [n, k], "search": summary, "space": {
                             "binds_through": inputs.get("binds_through"), "family": inputs.get("family"), "generator": inputs.get("generator"),
                             "proposed": len((population or {}).get("candidates") or []),
                             "bubblebeam_rejected_rows": len((request or {}).get("rejected_coupled_rows") or []),
                             "futuresight_rejected": len(rejected),
                             "futuresight_reasons": sorted({str(r.get("reason")) for r in rejected})},
                           "search_result": f"{FOLDER}/{stem}-search-result.json", "ab": None, "decision": None}
    winner = summary["winner"]
    decision: CandidateDecision | None = None
    verify, judged = None, None
    if winner is not None:
      say(f"role {key}: check")
      try:
        verify = checker(summary, result, model=model, quant=quant, rows=n, cols=k, role=role, tensor_name=tensor_name)
      except Exception as exc:  # BoltBeam could not run its own check: nothing the provider says is taken
        verify = {"error": f"{type(exc).__name__}: {exc}"[:400]}
      _write(folder / f"{stem}-check.json", verify)
      judged = checked(summary, verify) if "error" not in verify else {}
      short = lambda v: {kk: v[kk] for kk in ("plan", "verdict", "provider_us", "boltbeam_us", "boltbeam_us_less_floor", "ratio", "band", "reason") if kk in v} if isinstance(v, dict) else v  # noqa: E731
      row["check"] = {"winner": short(judged.get("winner")), "model_kernel": short(judged.get("model_kernel")),
                      "boltbeam_winner": short(judged.get("boltbeam_winner")), "reproduced": judged.get("reproduced"),
                      "not_reproduced": judged.get("not_reproduced"), "error": verify.get("error"), "tensor": verify.get("tensor"),
                      "floor_us": verify.get("floor_us"), "chip_band": verify.get("band")}
      row["check_result"] = f"{FOLDER}/{stem}-check.json"
      if judged.get("boltbeam_winner"):  # BoltBeam's pick (its own fastest) replaces the provider's rank
        pick = judged["boltbeam_winner"]["candidate_hash"]
        row["provider_winner"] = winner
        winner = next(_entry(r) for r in result["population"] if r.get("candidate_hash") == pick)
        model_row = summary.get("model_kernel")
        winner["is_model_kernel"] = bool(model_row) and model_row["candidate_hash"] == pick
        summary["winner"] = winner
        verify = {"winner": judged["boltbeam_winner"], "model_kernel": judged.get("model_kernel")}
    reproduced = bool(verify) and (verify.get("winner") or {}).get("verdict") == provider_check.REPRODUCED
    k_numbers = kernel_numbers(summary, losses.get(rk), verify if reproduced else None, counts.get(rk))
    if winner is None:
      reason = result.get("error") or "no candidate compiled and passed the correctness check"
      row.update(status="blocked", reason=f"decided by the search: {reason}", decided_by="search")
    elif not reproduced:
      best = (verify or {}).get("winner") or {}
      why = (verify or {}).get("error") or (f"BoltBeam's fastest candidate, {best.get('plan')}: {best.get('reason')}" if best
                                            else "BoltBeam measured no candidate correct" if verify is not None else "BoltBeam's check did not run")
      row.update(status="blocked", decided_by="check", reason=f"{provider_check.NOT_REPRODUCED}: {why}", not_reproduced=True)
    elif winner["is_default"] or winner.get("is_model_kernel"):
      decision = default_wins(route, winner, model_id=model_id, target_id=target_id,
                              search_path=row["search_result"], search_sha=search_sha,
                              model_kernel=bool(winner.get("is_model_kernel")))
    elif ab_only_if_faster and not (k_numbers or {}).get("faster_than_model"):
      model_us = (k_numbers or {}).get("model_us_per_call")
      row.update(status="blocked", decided_by="kernel alone",
                 reason=f"decided by the kernel alone: the best plan took {(k_numbers or {}).get('plan_us', 0):.1f} µs per call by BoltBeam's timer, "
                        + (f"the model's own kernel {model_us:.1f} µs; no whole-model A/B" if model_us is not None
                           else "with no model time to compare; no whole-model A/B"))
    elif winner["plan_kind"] != "tinygrad_opt_sequence.v1":
      row.update(status="blocked", decided_by="binding",
                 reason="decided by binding: the whole-model A/B binds Opt sequences only; an emitter plan reaches the "
                        "model through the fork's route admission, which the A/B driver does not install yet")
    else:
      say(f"role {key}: ab")
      try:
        ab = matched_ab(ab_run(winner["transforms"], (n, k)), env)
      except Exception as exc:
        ab = None
        row.update(status="blocked", reason=f"decided by the whole model: the A/B did not run: {exc}",
                   decided_by="whole model")
      if ab is not None:
        ab_sha = _write(folder / f"{stem}-ab.json", ab)
        row["ab"] = {key_: ab[key_] for key_ in ("baseline_tok_s", "candidate_tok_s", "delta_pct", "spread_pct", "ranges_overlap", "pairs",
                                                 "token_match", "route_bound", "context", "timing_source")}
        row["ab_result"] = f"{FOLDER}/{stem}-ab.json"
        row["ab"]["binding"] = ab["binding"]
      if ab is not None and not ab["route_bound"]:
        # Both arms ran the same code, so neither number can judge the plan. Not tested is not refuted.
        row.update(status="blocked", decided_by="binding",
                   reason=f"decided by binding: the plan reached {ab['binding'].get('calls_reached', 0)} of the "
                          f"{_calls(losses.get(rk))} {role} {quant} calls per token "
                          f"({ab['binding'].get('decode_graph_calls', 0)} decode calls in all) "
                          f"(applied to {ab['binding'].get('applied', 0)} other kernels); "
                          "the A/B compared the default decode with itself")
      elif ab is not None:
        decision = decide(route, winner, ab, model_id=model_id, target_id=target_id, ab_path=row["ab_result"], ab_sha=ab_sha)
    row["kernel"] = k_numbers
    if row["kernel"] and row["kernel"]["model_us_per_call"] is not None:
      row["role_calls_per_token"] = row["kernel"]["tensors_per_token"]
    if decision is not None:
      by = "kernel alone" if row["ab"] is None else "whole model"
      row.update(status=route_status(decision.verdict), reason=f"decided by the {by}: {decision.reason}",
                 decided_by=by, decision=decision.to_json())
      if status_for_verdict(decision.verdict) is not None:
        ledger.add(decision, scope={"model": model_id, "target": target_id, "workload": "decode", "role": role, "quant": quant},
                   reopen_condition=REOPEN if decision.verdict == Verdict.REFUTE.value else "")
    if winner is not None: row["plan_id"] = plan_id(role, quant, winner)
    record["roles"].append(row)
    _apply(route, row)
    write_json(run / "route_policy.json", policy)
    _write(folder / "compare.json", record)
    say(f"role {key}: done {row['status']}")
    if step: step(len(record["roles"]), len(routes))
  if fusions:
    try:
      found = compare_fusions(run, facts=facts, describe_shapes=lambda shapes: ((describe_fn or describe)(root, device, shapes=shapes)
                                                                                 .get("fusion_shape_verdicts") or []),
                              search=search, checker=checker, model=model, model_sha=model_sha, target_id=target_id, losses=losses,
                              timestamp=timestamp, say=say, tensors=tensors_of_model)
    except Exception as exc:  # the fusions are their own stage: a failure there leaves the roles' results standing
      found = {"schema": FUSIONS_SCHEMA, "run": run.name, "fusions": [], "error": f"{type(exc).__name__}: {exc}"[:400]}
      _write(folder / FUSIONS_FILE, found)
      say(f"compare fusions: failed: {found['error']}")
    record["fusions"] = [{k: f.get(k) for k in ("name", "quant", "verdict", "reason", "in_model_ms", "lost_ms")} | {
      "fused_ms": (f.get("fused") or {}).get("ms_per_token"), "unfused_ms": (f.get("unfused") or {}).get("ms_per_token")}
      for f in found.get("fusions") or []]
    record["fusions_result"] = f"{FOLDER}/{FUSIONS_FILE}"
    _write(folder / "compare.json", record)
  say(f"compare done: {run}")
  return record


def _apply(route:dict[str, Any], row:Mapping[str, Any]) -> None:
  """Write one role's outcome into its route_policy.json row. selected_route is set only for a promoted plan,
  because consumers of route_policy.json run a selected route."""
  route["status"] = row["status"]
  route["selected_route"] = row.get("plan_id") if row["status"] == "promoted" else None
  route["evidence_refs"] = [x for x in (row.get("search_result"), row.get("ab_result")) if x]
  winner = row["search"]["winner"] or {}
  route["compare"] = {"plan_id": row.get("plan_id"), "plan": winner.get("plan"), "plan_hash": winner.get("plan_hash"),
                      "search_median_ns": winner.get("median_ns"), "default_median_ns": row["search"]["default_median_ns"],
                      "kernel": row.get("kernel"), "decided_by": row.get("decided_by"),
                      "role_calls_per_token": row.get("role_calls_per_token"),
                      "measured_correct": row["search"]["counts"].get("measured_correct"),
                      "candidates": row["search"]["counts"].get("total"), "ab": row.get("ab"), "reason": row.get("reason"),
                      "timing_source": TIMING_SOURCE, "check": row.get("check"), "not_reproduced": bool(row.get("not_reproduced")),
                      "space": row.get("space")}



# --- fusions: one launch for several adjacent decode-graph nodes ------------------------------------------------

FUSIONS_FILE = "fusions.json"
FUSIONS_SCHEMA = "boltbeam.kernel_fusions.v1"


def _fusion_stem(proposal:Mapping[str, Any]) -> str:
  return f"fusion-{proposal['name'].replace('+', '-')}-{str(proposal['quant']).lower()}"


def fusion_verdict(proposal:Mapping[str, Any], check:Mapping[str, Any] | None, losses:Mapping[tuple[str, str], Mapping[str, Any]],
                   band:float) -> dict[str, Any]:
  """One fusion's numbers and verdict. The bar is the in-model time of the tie-out rows it replaces (the "where the
  token goes" rows), per token; BoltBeam's fused kernel alone (less the floor) times the launches per token must beat it
  by more than the chip's band. It must also beat BoltBeam's unfused side timed the same way (each covered node's
  kernel alone, less the floor; a lower bound when a node's reading was not reproduced): an isolated time against an
  in-model one alone can show a gain that is only the difference between the two ways of timing. A covered node with
  no row of its own counts 0 ms in the model (fusion_space.NOT_IN_MODEL)."""
  from boltbeam.search import fusion_space
  inst = proposal.get("instance") or {}
  rows = [losses.get(tuple(r)) for r in inst.get("tie_out_rows") or []]
  calls = inst.get("calls_per_token")
  out: dict[str, Any] = {"in_model_ms": None, "lost_ms": None, "fused": None, "unfused": None}
  if rows and all(rows):
    out["in_model_ms"] = sum(r["actual_ms"] for r in rows)
    out["lost_ms"] = sum(r["lost_ms"] for r in rows)
    out["in_model_rows"] = [{"row": f"{r['role']} {r['quant']}", "ms": r["actual_ms"], "lost_ms": r["lost_ms"],
                             "calls_per_token": r["calls_per_token"]} for r in rows]
  weights = set(proposal.get("weights") or [])
  zero = [n for n in proposal.get("covers") or [] if n not in weights]
  if zero:
    out["in_model_note"] = f"{', '.join(zero)} {fusion_space.NOT_IN_MODEL}"
  if check is None or check.get("error"):
    return out | {"verdict": "not_reproduced" if check else "not_searched",
                  "reason": (check or {}).get("error") or "the search measured no fused candidate"}
  good = [c for c in check.get("candidates") or [] if c.get("verdict") == provider_check.REPRODUCED]
  if not good:
    why = "; ".join(sorted({str(c.get("reason")) for c in check.get("candidates") or []})) or "the search measured no fused candidate"
    return out | {"verdict": "not_reproduced", "reason": f"{provider_check.NOT_REPRODUCED}: {why}"}
  best = min(good, key=lambda c: c["boltbeam_us_less_floor"])
  fused_ms = best["boltbeam_us_less_floor"] * calls / 1000.0
  out["fused"] = {"plan": best["plan"], "candidate_hash": best["candidate_hash"], "us_less_floor": best["boltbeam_us_less_floor"],
                  "ms_per_token": fused_ms, "correctness": (best.get("boltbeam") or {}).get("correctness"),
                  "provider_note": best.get("provider_note")}
  alone = check.get("unfused") or []
  good_alone = [u for u in alone if u.get("verdict") == provider_check.REPRODUCED]
  every = bool(alone) and len(good_alone) == len(alone) and {u["node"] for u in alone} == set(proposal.get("covers") or [])
  # the unfused side alone: its sum when BoltBeam reproduced every node; the reproduced nodes' sum is a lower bound
  # otherwise (a node's time is never below 0), and like for like a fused kernel must beat even that
  lower = sum(u["boltbeam_us_less_floor"] for u in good_alone) * calls / 1000.0 if good_alone else None
  out["unfused"] = {"nodes": [{"node": u.get("node"), "verdict": u.get("verdict"), "us_less_floor": u.get("boltbeam_us_less_floor"),
                               "reason": u.get("reason")} for u in alone],
                    "ms_per_token": lower if every else None, "lower_bound_ms": lower}
  if not every:
    out["unfused"]["why"] = ("BoltBeam did not reproduce every covered node's kernel alone; the ones it did sum to a lower bound"
                             if alone else "the provider returned no unfused kernels")
  if out["in_model_ms"] is None:
    return out | {"verdict": "none_faster", "reason": "the tie-out has no in-model time for the rows this fusion replaces"}
  if lower is not None and fused_ms >= lower * (1.0 - band):
    return out | {"verdict": "none_faster", "decided_by": "kernel alone",
                  "reason": (f"decided by the kernel alone: BoltBeam's fastest fused kernel ({best['plan']}) takes {fused_ms:.3f} ms per token, "
                             f"not faster than the nodes it replaces timed alone the same way ({lower:.3f} ms"
                             f"{'' if every else ', a lower bound'}); the model's {out['in_model_ms']:.3f} ms in the model is not a like-for-like bar")}
  if fused_ms < out["in_model_ms"] * (1.0 - band):
    return out | {"verdict": "found_not_applied", "decided_by": "binding",
                  "reason": (f"decided by binding: BoltBeam's fused kernel ({best['plan']}) takes {fused_ms:.3f} ms per token against "
                             f"the model's {out['in_model_ms']:.3f} ms for these rows; the whole-model A/B does not install a fused emitter yet")}
  return out | {"verdict": "none_faster", "decided_by": "kernel alone",
                "reason": (f"decided by the kernel alone: BoltBeam's fastest fused kernel ({best['plan']}) takes {fused_ms:.3f} ms per token "
                           f"against the model's {out['in_model_ms']:.3f} ms for these rows (the chip's band is ±{100 * band:.1f}%)")}


def compare_fusions(run:pathlib.Path, *, facts:Mapping[str, Any], describe_shapes:Callable[[list[dict[str, int]]], list[dict[str, Any]]],
                    search:Callable[..., dict[str, Any]], checker:Any, model:str, model_sha:str, target_id:str,
                    losses:Mapping[tuple[str, str], Mapping[str, Any]], timestamp:str, say:Callable[[str], None],
                    tensors:list[tuple[str, tuple[int, ...], int, int]] | None = None) -> dict[str, Any]:
  """BubbleBeam proposes the fusions (fusion_space.propose), the population is exported, FutureSight assesses it, the
  campaign measures it, BoltBeam checks and times both sides, and fusion_verdict judges. Writes
  kernel_compare/fusion-<nodes>-<quant>-{search-request,population,search-result,check}.json and fusions.json."""
  from boltbeam.search import fusion_space
  from boltbeam.search.futuresight_adapter import assess_population, propose_request, target_facts
  from boltbeam.search.semantic.semantic_population_export import export_population
  folder = run / FOLDER
  families = [f for f in facts.get("decode_fusions") or [] if isinstance(f, Mapping)]
  record: dict[str, Any] = {"schema": FUSIONS_SCHEMA, "run": run.name, "timing_source": TIMING_SOURCE, "fusions": []}
  if not families:
    record["note"] = "the provider describes no fused emitter on this device"
    _write(folder / FUSIONS_FILE, record)
    return record
  if tensors is None:
    from boltbeam.profile.gguf import read_gguf_layout
    tensors = read_gguf_layout(pathlib.Path(model))[1]
  by_layer = fusion_space.layers(tensors)
  first = fusion_space.propose(families, by_layer, set(losses))
  shapes = sorted({tuple(p["instance"]["shape"]) for p in first if p.get("instance")})
  verdicts = describe_shapes([{"n": n, "k": k} for n, k in shapes]) if shapes else []
  proposals = fusion_space.propose(families, by_layer, set(losses), verdicts)
  resolved = resolved_target_document(target_id, facts["target"])
  target = candidate_target(resolved)
  chip, _missing = target_facts(target_id, facts)
  say(f"compare fusions: {len(proposals)} proposed by the provider's fused emitters")
  for proposal in proposals:
    stem = _fusion_stem(proposal)
    row: dict[str, Any] = {"name": proposal["name"], "family": proposal["family"], "covers": proposal["covers"], "quant": proposal["quant"],
                           "instance": proposal.get("instance"), "proposed_rows": len(proposal.get("rows") or []),
                           "rejected_rows": proposal.get("rejected_rows") or [], "stem": stem}
    if proposal.get("why"):
      row.update(verdict="rejected", reason=f"rejected by BubbleBeam: {proposal['why']}", decided_by="proposal")
      record["fusions"].append(row | fusion_verdict(proposal, None, losses, checker.band) | {"verdict": "rejected", "reason": row["reason"]})
      say(f"fusion {proposal['name']} {proposal['quant']}: rejected ({proposal['why']})")
      continue
    say(f"fusion {proposal['name']} {proposal['quant']}: search")
    work = fusion_space.workload(proposal, model_sha=model_sha, target=target, tolerance=TOLERANCE)
    base = {"schedule.plan_kind": role_space.EMITTER_PLAN_KIND, "schedule.compute.family": proposal["family"]}
    spec = {"semantic_workload": work, "schedule": dict(role_space.SEED_SCHEDULE), "resolved_target": resolved, "gguf_path": str(model),
            "execution": {"shape_mode": "exact_workload", "warmups": 2, "samples": 7},
            "request_id": f"{run.name}-{stem}", "run_id": f"{run.name}-compare", "timestamp": timestamp,
            "budget": {"max_candidates": len(proposal["rows"]) + 2, "timeout_s": 600.0},
            "axis_choices": {}, "coupled_rows": [{**base, **r} for r in proposal["rows"]]}
    try:
      request, _missing = propose_request(spec, facts)                                     # 1. BubbleBeam
      request["rejected_coupled_rows"] = list(request["rejected_coupled_rows"]) + [
        {"row": {**base, **r["row"]}, "reason": r["reason"]} for r in proposal.get("rejected_rows") or []]
      population = export_population(request)                                              # 2. the population
      evidence = assess_population(population)                                             # 3. FutureSight
    except Exception as exc:  # no space for this fusion: say why, the others go on
      row.update(verdict="not_searched", reason=f"no space: {exc}")
      record["fusions"].append(row); continue
    _write(folder / f"{stem}-search-request.json", request | {"futuresight_evidence": evidence})
    _write(folder / f"{stem}-population.json", population)
    row["futuresight_rejected"] = [{"candidate_hash": r.get("candidate_hash"), "reason": r.get("reason")}
                                   for r in evidence.get("rejections") or [] if "candidate_hash" in r]
    try:
      result = search(request, evidence)                                                   # 4. the campaign
    except Exception as exc:
      result = {"status": "BLOCKED", "counts": {}, "population": [], "error": str(exc), "futuresight_static_evidence": evidence}
    _write(folder / f"{stem}-search-result.json", result)
    row["search"] = {"status": result.get("status"), "counts": dict(result.get("counts") or {}), "error": result.get("error"),
                     "refused": [{"candidate_hash": r.get("candidate_hash"), "state": r.get("state"), "reason": r.get("reason"),
                                  "plan": plan_text((r.get("candidate") or {}).get("schedule") or {})}
                                 for r in result.get("population") or [] if r.get("state") != "MEASURED"]}
    check = None
    if any(r.get("state") == "MEASURED" for r in result.get("population") or []):
      say(f"fusion {proposal['name']} {proposal['quant']}: check")
      try:
        check = checker.fusion(result, model=model, proposal=proposal)
      except Exception as exc:  # BoltBeam could not run its own check: nothing the provider says is taken
        check = {"error": f"{type(exc).__name__}: {exc}"[:400]}
      _write(folder / f"{stem}-check.json", check)
      row["check_result"] = f"{FOLDER}/{stem}-check.json"
    judged = fusion_verdict(proposal, check, losses, checker.band)
    if check is None:
      judged.update(verdict="not_reproduced" if result.get("population") else "not_searched",
                    reason=f"the provider measured no fused candidate: {result.get('error') or 'every candidate was refused'}")
    row.update(judged, search_result=f"{FOLDER}/{stem}-search-result.json")
    record["fusions"].append(row)
    say(f"fusion {proposal['name']} {proposal['quant']}: done {row['verdict']}")
  _write(folder / FUSIONS_FILE, record)
  return record


# --- the search as a stage of Run, and what the screens show per role -----------------------------------------

STATUS_FILE = "search_status.json"
STATUS_SCHEMA = "boltbeam.kernel_search_status.v1"
VERDICTS = ("applied", "found_not_applied", "none_faster", "not_reproduced", "not_searched")


def detect(backend:str | None) -> str | None:
  """BoltBeam's own detection: can its bridge open a GPU of this backend here? None when it can, else why not."""
  from boltbeam.collectors import kernel_timer
  try:
    bridge = kernel_timer.bridge_for(str(backend))
  except Exception as exc:  # no driver, no device, no compiler: BoltBeam could not check the provider's claims here
    return f"BoltBeam cannot open a {backend} GPU on this machine to check the provider: {exc}"
  close = getattr(bridge, "close", None)
  if close: close()
  return None


def search_applies(backend:str | None, provider:str, *, root:pathlib.Path | None = None,
                   capability_fn:Callable[[pathlib.Path, list[str]], dict[str, Any]] | None = None,
                   detect_fn:Callable[[str | None], str | None] | None = None) -> str | None:
  """Why Run's kernel search does not exist for this engine and chip, or None when it does. Two answers decide it:
  the provider's own capability (the devices it can run, asked with its capability action) and BoltBeam's own
  detection of the GPU (its bridge opens one), never a list of backend names. Without the fork checked out the
  provider cannot be asked; readiness() names the fix for that."""
  if provider != "tinygrad":
    return f"the kernel search runs in tinygrad only; this run used {provider}"
  row = runtime_row(backend)
  if row is None:
    return f"the search runtime table (data/search_runtime.json) has no row for {backend or 'an unknown backend'}"
  if why := (detect_fn or detect)(backend):
    return why
  root = pathlib.Path(root) if root else default_fork_root()
  if capability_fn is None and not (root / PROVIDER).is_file():
    return None
  try:
    reply = (capability_fn or capability)(root, list(row["tinygrad_devices"]))
  except Exception as exc:
    return f"the provider could not say which devices it runs: {exc}"
  return pick_device(backend, reply)[1]


def write_status(run:pathlib.Path, status:str, reason:str | None = None, seconds:float | None = None) -> None:
  _write(run / FOLDER / STATUS_FILE, {"schema": STATUS_SCHEMA, "status": status, "reason": reason, "seconds": seconds})


def read_status(run:pathlib.Path) -> dict[str, Any] | None:
  path = run / FOLDER / STATUS_FILE
  try:
    return json.loads(path.read_text()) if path.is_file() else None
  except (OSError, ValueError):
    return None


def skip(run:pathlib.Path, reason:str, say:Callable[[str], None] = print) -> dict[str, Any]:
  write_status(run, "skipped", reason)
  say(f"search skipped: {reason}")
  return {"status": "skipped", "reason": reason}


def search_stage(run:pathlib.Path, *, backend:str, provider:str, root:pathlib.Path | None = None,
                 say:Callable[[str], None] = print, step:Callable[[int, int], None] | None = None,
                 compare:Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
  """Run's "Search faster kernels" stage: the per-role search where it exists, else a skip with its reason. A
  search that cannot start (no fork, a dirty fork) is a skip too: the run still ends with its results."""
  import time
  if why := search_applies(backend, provider, root=root):
    write_status(run, "skipped", why)
    say(f"search skipped: {why}")
    return {"status": "skipped", "reason": why}
  root = pathlib.Path(root) if root else default_fork_root()
  if compare is None:
    ready = readiness(root, str((json.loads((run / "run_manifest.json").read_text())).get("model_path") or ""))
    why = (f"{ready['message']} Fix: {ready['fix']}" if ready.get("fix") else ready["message"]) if not ready["ready"] \
      else clean_refusal(root)
    if why:
      write_status(run, "skipped", why)
      say(f"search skipped: {why}")
      return {"status": "skipped", "reason": why}
  began = time.monotonic()
  try:
    (compare or compare_run)(run, root=root, say=say, ab_only_if_faster=True, step=step)
  except ProviderRefused as exc:  # BoltBeam refused the provider's device: nothing it would say is taken
    write_status(run, "skipped", f"provider refused: {exc}")
    say(f"search skipped: provider refused: {exc}")
    return {"status": "skipped", "reason": f"provider refused: {exc}"}
  seconds = round(time.monotonic() - began, 1)
  write_status(run, "searched", None, seconds)
  return {"status": "searched", "reason": None, "seconds": seconds}


def _speedup(kernel:Mapping[str, Any] | None) -> float | None:
  if not kernel or not kernel.get("model_us_per_call") or not kernel.get("plan_us"):
    return None
  return float(kernel["model_us_per_call"]) / float(kernel["plan_us"])


def role_verdict(route:Mapping[str, Any] | None, status:Mapping[str, Any] | None) -> dict[str, Any]:
  """best_found and verdict for one role, from its route_policy.json compare record and the stage's status.
  Only measured numbers: the plan alone and the model's own kernel per call, the binding count, the A/B."""
  c = (route or {}).get("compare") if isinstance((route or {}).get("compare"), dict) else None
  if c is None:
    reason = (status or {}).get("reason") or ("the search found no route for this role" if status else
                                              "the kernel search did not run for this run")
    return {"best_found": None, "verdict": "not_searched", "verdict_reason": reason}
  k = c.get("kernel") if isinstance(c.get("kernel"), dict) else None
  x = _speedup(k)
  plan = c.get("plan")
  best = None
  if k is not None and plan:
    best = {"plan": plan, "plan_us": k.get("plan_us"), "model_us": k.get("model_us_per_call"), "speedup": x,
            "calls_per_token": c.get("role_calls_per_token"),
            "text": f"{plan} · {x:.1f}x faster" if x and x > 1 else f"{plan} · not faster" if x else plan}
  searched = {"candidates": c.get("candidates"), "measured_correct": c.get("measured_correct")}
  if c.get("not_reproduced"):  # BoltBeam's own check disagreed with the provider: never a finding
    return {"best_found": None, "verdict": "not_reproduced", "verdict_reason": c.get("reason"), **searched}
  if k is None:
    return {"best_found": None, "verdict": "not_searched", "verdict_reason": c.get("reason") or "the search measured no kernel", **searched}
  if (route or {}).get("status") == "promoted":
    return {"best_found": best, "verdict": "applied", "verdict_reason": c.get("reason"), **searched}
  if not k.get("faster_than_model") or plan == "default kernel":
    return {"best_found": best, "verdict": "none_faster", "verdict_reason": c.get("reason"), **searched}
  return {"best_found": best, "verdict": "found_not_applied", "verdict_reason": c.get("reason"), **searched}


def found_summary(routes:list[Mapping[str, Any]]) -> dict[str, Any] | None:
  """The roles with a faster kernel found but not applied: ms per token saved if applied, the sum over roles of
  (model µs - plan µs) x calls per token, and how many of those calls the plan reached in the whole-model A/B."""
  rows, ms, reached, calls, ab_roles = [], 0.0, 0, 0.0, 0
  for r in routes:
    v = role_verdict(r, {"status": "searched"})
    b = v["best_found"]
    if v["verdict"] != "found_not_applied" or not b or b.get("calls_per_token") is None:
      continue
    ms += (b["model_us"] - b["plan_us"]) * b["calls_per_token"] / 1000.0
    rows.append(role_label(str(r.get('role')), str(r.get('quant')), r.get('shape')))
    ab = (r.get("compare") or {}).get("ab") or {}
    binding = ab.get("binding") if isinstance(ab.get("binding"), dict) else None
    if binding is not None:
      ab_roles += 1
      reached += int(binding.get("calls_reached") or 0)
      calls += float(b["calls_per_token"])
  if not rows:
    return None
  return {"roles": rows, "ms": ms, "calls_reached": reached if ab_roles else None,
          "calls": calls if ab_roles else None}
