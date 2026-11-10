"""One role's proposal: its exact semantic workload (the recorded tensor, its shape and packed format, the one-token
GEMV), the correctness tolerance, and the documented pipeline's steps 1 to 3 (docs/semantic-campaign.md): BubbleBeam
proposes, the population is exported, FutureSight assesses it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from boltbeam.plan.resolved_target import candidate_target, resolved_target_document
from boltbeam.search import role_space


def tolerances() -> dict[str, float]:
  from boltbeam.math.numerics import TOLERANCE
  return {"atol": TOLERANCE, "rtol": TOLERANCE}


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
          "tolerance": tolerances(), "fixture_shape_substitution": "forbidden"}


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
