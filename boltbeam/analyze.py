from __future__ import annotations

import json
import pathlib
from typing import Any

from boltbeam.policy.emit import emit_seed_policy
from boltbeam.profile.ir import ModelProfile, TargetProfile, TensorRole
from boltbeam.quantization.quant import quant_capability, route_families_for
from boltbeam.search.emit import emit_search_space
from boltbeam.synth.fixtures import emit_fixture_manifest
from boltbeam.target.tinygrad_root import resolve_tinygrad_root
from boltbeam.vocab import role_rank_priority
from boltbeam.core.canonical import pretty_json


ARTIFACTS = (
  "model_profile.json",
  "search_space.json",
  "route_policy.seed.json",
  "fixture_manifest.json",
  "next_measurement_plan.json",
  "tinygrad_commands.md",
  "analysis_manifest.json",
)


def _rank_by_memory_cost(role:TensorRole) -> tuple[float, int, int, str]:
  """Answers: which role is hottest to keep resident, by bytes/elem?

  Quant priority is derived from the registry byte-cost (cheaper bytes/elem = hotter), not a hardcoded
  per-quant table; unknown quants sink last. This reproduces the old Q4_K<Q5_K<Q6_K<Q8_0<F16 ordering:
  Q8_0 (smaller bytes/elem) ranks before F16 here.

  This is a different question from boltbeam/workflow/analyze.py:_rank_by_routability, which answers
  "which role can we act on" (route-family support beats memory cost there: F16 ranks before Q8_0). Both
  are correct; do not merge them. They share only the coarse-role tiebreak table, centralized in
  vocab.role_rank_priority.
  """
  cap = quant_capability(role.quant)
  quant_rank = cap.bytes_per_elem if cap else 9.0
  return (quant_rank, role_rank_priority(role.role), -(role.rows * role.cols * role.count), role.tensor_name)


def _role_summary(role:TensorRole) -> dict[str, Any]:
  return {
    "role": role.role,
    "tensor_name": role.tensor_name,
    "shape": [role.rows, role.cols],
    "quant": role.quant,
    "count": role.count,
    "candidate_families": _candidate_families(role),
  }


def _candidate_families(role:TensorRole) -> list[str]:
  # single source of truth: the quant registry decides route families (no duplicate quant->family table).
  fams = list(route_families_for(role.quant))
  if not fams:
    return ["profile_first"]                       # unsupported or known-no-route quant
  cap = quant_capability(role.quant)
  if cap and cap.dequant_family == "k_quant":      # k-quant roles also get the reduce-source diagnostic
    fams.append("reduce_source_trace")
  return fams


def _tool_status(tinygrad_root:pathlib.Path) -> dict[str, Any]:
  tools = {
    "role_attribution": "extra/qk_decode_role_attribution_modular.py",
    "reduce_source_trace": "extra/qk_decode_reduce_source_trace.py",
    "runtime_overhead": "extra/qk_decode_runtime_overhead.py",
  }
  return {
    name: {"path": rel, "present": (tinygrad_root / rel).exists()}
    for name, rel in tools.items()
  }


# tinygrad DEV backend derived from the target descriptor's backend (not hardcoded to AMD; audit Z4/A9)
_BACKEND_DEV = {"AMD": "AMD", "CUDA": "NV", "Metal": "METAL"}


def _env_prefix(route_flags:dict[str, str], extra:dict[str, str] | None=None, dev:str="AMD") -> str:
  env = {"DEV": dev, "JIT": "1", "PYTHONPATH": "."}
  env.update(route_flags)
  if extra: env.update(extra)
  return " ".join(f"{k}={v}" for k, v in env.items())


def _format_cmd(cmd:str) -> str:
  return f"```bash\n{cmd}\n```"


def build_tinygrad_commands(model_path:str | pathlib.Path, model_id:str, tinygrad_root:str | pathlib.Path,
                            ctxs:tuple[int, ...], max_context:int,
                            route_flags:dict[str, str] | None=None, backend:str="AMD") -> dict[str, Any]:
  tg = pathlib.Path(tinygrad_root).expanduser()
  model = pathlib.Path(model_path).expanduser()
  ctxs_s = ",".join(str(x) for x in ctxs)
  flags = route_flags or {"DECODE_Q4K_G3_ANYSHAPE": "1"}
  dev = _BACKEND_DEV.get(backend, "AMD")
  role_prefix = _env_prefix(flags, dev=dev)
  runtime_prefix = _env_prefix(flags, {"QK_MODEL": str(model), "QK_CKPTS": ctxs_s}, dev=dev)

  commands = {
    "profile_only": (
      f"cd {tg} && PYTHONPATH=. python3 extra/qk_decode_role_attribution_modular.py "
      f"--model {model} --id {model_id} --ctxs {ctxs_s} --max-context {max_context} --profile-only"
    ),
    "capture_roles": (
      f"cd {tg} && {role_prefix} python3 extra/qk_decode_role_attribution_modular.py "
      f"--model {model} --id {model_id}-route --ctxs {ctxs_s} --max-context {max_context} --capture"
    ),
    "reduce_source_trace": (
      f"cd {tg} && {role_prefix} python3 extra/qk_decode_reduce_source_trace.py "
      f"--model {model} --id {model_id}-route --ctx {ctxs[0]}"
    ),
    "runtime_overhead": (
      f"cd {tg} && {runtime_prefix} python3 extra/qk_decode_runtime_overhead.py"
    ),
  }
  return {
    "tinygrad_root": str(tg),
    "route_flags": flags,
    "commands": commands,
    "tools": _tool_status(tg),
  }


def build_measurement_plan(profile:ModelProfile, target:TargetProfile, model_path:str | pathlib.Path,
                           tinygrad_root:str | pathlib.Path, ctxs:tuple[int, ...], max_context:int,
                           route_flags:dict[str, str] | None=None) -> dict[str, Any]:
  commands = build_tinygrad_commands(model_path, profile.model_id, tinygrad_root, ctxs, max_context,
                                     route_flags, backend=target.backend)
  prioritized = [_role_summary(r) for r in sorted(profile.roles, key=_rank_by_memory_cost)]
  # group the focus by ROUTE FAMILY (registry-driven), not by hardcoded quant names (audit Z7/A9)
  by_family:dict[str, list[dict[str, Any]]] = {}
  for r in sorted(profile.roles, key=_rank_by_memory_cost):
    for fam in _candidate_families(r):
      by_family.setdefault(fam, []).append(_role_summary(r))
  phases = [
    {
      "id": "M0_profile",
      "purpose": "Confirm the GGUF-derived profile and route classifier before touching the GPU.",
      "command": commands["commands"]["profile_only"],
      "promotion": "No promotion; this catches profile drift and missing role classification.",
    },
    {
      "id": "M1_role_attribution",
      "purpose": "Measure full decode buckets by role and context with route flags enabled.",
      "command": commands["commands"]["capture_roles"],
      "promotion": "Select the hottest role whose bucket exceeds the configured residual threshold.",
    },
    {
      "id": "M2_reduce_source_trace",
      "purpose": "Resolve r_* reduce kernels by graph position and source role before designing new kernels.",
      "command": commands["commands"]["reduce_source_trace"],
      "promotion": "Only fuse or eliminate a reduce after source resolution is firm.",
    },
    {
      "id": "M3_runtime_overhead",
      "purpose": "Separate host/runtime overhead from GPU work so codegen is not blamed for sync cost.",
      "command": commands["commands"]["runtime_overhead"],
      "promotion": "If host sync dominates, route search pauses and runtime batching/sync becomes the target.",
    },
  ]
  return {
    "schema": "boltbeam.next_measurement_plan.v1",
    "model_id": profile.model_id,
    "model_path": str(pathlib.Path(model_path).expanduser()),
    "target": target.to_json(),
    "contexts": list(ctxs),
    "max_context": max_context,
    "tinygrad": commands,
    "priority_roles": prioritized,
    "phase_order": phases,
    "route_focus": {
      "by_route_family": by_family,
      "notes": [
        "Prefer measured attribution over shape hunches.",
        "Generated routes may replace hand routes only after correctness, route-binding, speed, memory, and rollback gates pass.",
        "If a candidate fails but exposes a missing topology knob, ledger it as search-space-incomplete rather than refuted.",
      ],
    },
  }


def _write_json(path:pathlib.Path, obj:dict[str, Any]) -> None:
  path.write_text(pretty_json(obj))


def _write_commands_md(path:pathlib.Path, plan:dict[str, Any]) -> None:
  commands = plan["tinygrad"]["commands"]
  tools = plan["tinygrad"]["tools"]
  lines = [
    f"# Tinygrad Measurement Commands: {plan['model_id']}",
    "",
    "Run GPU measurements serially. Each command writes tinygrad-side artifacts; BoltBeam only generates the handoff.",
    "",
    "## Tool Presence",
    "",
  ]
  for name, status in tools.items():
    marker = "present" if status["present"] else "missing"
    lines.append(f"- `{name}`: {marker} at `{status['path']}`")
  lines += [
    "",
    "## 1. Profile And Classifier",
    "",
    _format_cmd(commands["profile_only"]),
    "",
    "## 2. Full Decode Role Attribution",
    "",
    _format_cmd(commands["capture_roles"]),
    "",
    "## 3. Reduce Source Trace",
    "",
    _format_cmd(commands["reduce_source_trace"]),
    "",
    "## 4. Runtime Overhead Check",
    "",
    _format_cmd(commands["runtime_overhead"]),
    "",
    "## Decision Rule",
    "",
    "Start from the hottest measured bucket. If it is a route miss, bind the generated route. "
    "If it is an unfused reduce, resolve the source before fusing. If a generated candidate fails because "
    "a topology knob is absent, mark search-space-incomplete and expose the knob instead of hand-writing a one-off route.",
    "",
  ]
  path.write_text("\n".join(lines))


def emit_analysis_bundle(profile:ModelProfile, target:TargetProfile, model_path:str | pathlib.Path,
                         out_dir:str | pathlib.Path, ctxs:tuple[int, ...]=(128, 512),
                         tinygrad_root:str | pathlib.Path | None=None,
                         max_context:int=4608,
                         route_flags:dict[str, str] | None=None) -> dict[str, Any]:
  out = pathlib.Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)

  profile_json = profile.to_json()
  search_json = emit_search_space(profile, target)
  policy_json = emit_seed_policy(profile, target)
  fixtures_json = emit_fixture_manifest(profile)
  plan_json = build_measurement_plan(profile, target, model_path, resolve_tinygrad_root(tinygrad_root),
                                     ctxs, max_context, route_flags)

  _write_json(out / "model_profile.json", profile_json)
  _write_json(out / "search_space.json", search_json)
  _write_json(out / "route_policy.seed.json", policy_json)
  _write_json(out / "fixture_manifest.json", fixtures_json)
  _write_json(out / "next_measurement_plan.json", plan_json)
  _write_commands_md(out / "tinygrad_commands.md", plan_json)

  manifest = {
    "schema": "boltbeam.analysis_manifest.v1",
    "model_id": profile.model_id,
    "target": target.target_id,
    "out_dir": str(out),
    "artifacts": list(ARTIFACTS),
    "summary": {
      "roles": len(profile.roles),
      "architecture_class": profile.architecture_class,
      "supported_roles": sum(1 for r in profile.roles if route_families_for(r.quant)),
      "blocked_roles": sum(1 for r in profile.roles if not route_families_for(r.quant)),
      "contexts": list(ctxs),
    },
  }
  _write_json(out / "analysis_manifest.json", manifest)
  return manifest
