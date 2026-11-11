"""The vLLM routing adapter: a thin reader of what BoltBeam's vLLM prefill driver already records. With
--semantic-dispatch, runtime/vllm_decode_drive.py:418-420 wraps each MoE runner's router.select_experts (runtime/
vllm_semantics.py:402-429, install_routing, through LLM.apply_model) and :449-450 writes read_routing's records
(vllm_semantics.py:432-452: {module, operation_id, shape [tokens, top_k], expert_counts}) into its JSON line as `routing`.
This sums them per layer. vLLM's own path, LLM(enable_return_routed_experts=True) (vllm/entrypoints/llm.py:143, :205;
CompletionOutput.routed_experts [seq_len, layer_num, topk], vllm/outputs.py:63; checked in the installed vLLM 0.30.0,
vllm-bench/.venv, not the pinned 0.31.0), is not used: the driver's hook already gives the counts. No capture is started
here: run the driver's real-text prefill with --semantic-dispatch --capture and pass its line (--from)."""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any

LAYER = re.compile(r"\.layers\.(\d+)\.")


def layers(routing:list[dict[str, Any]]) -> list[dict[str, Any]]:
  per: dict[int, dict[str, Any]] = {}
  for r in routing:
    m = LAYER.search(r["module"] + ".")
    if not m:
      raise ValueError(f"no layer index in the module name {r['module']!r}")
    acc = per.setdefault(int(m.group(1)), {"layer": int(m.group(1)), "shape": [0, r["shape"][1]], "expert_counts": {}})
    acc["shape"][0] += r["shape"][0]
    for e, c in r["expert_counts"].items():
      acc["expert_counts"][e] = acc["expert_counts"].get(e, 0) + c
  return [per[k] for k in sorted(per)]


def profile(path:str | pathlib.Path, model:dict[str, Any], *, n_expert:int, corpus:str, tokens:int | None = None) -> dict[str, Any]:
  from boltbeam.roofline import routing_profile as rp
  line = next((x for x in map(json.loads, pathlib.Path(path).read_text().splitlines()) if x.get("routing")), None)
  if line is None:
    raise ValueError(f"{path}: no driver line with routing (run the vLLM prefill driver with --semantic-dispatch --capture)")
  return rp.record(layers(line["routing"]), model=model, n_expert=n_expert,
                   source={"engine": "vllm", "how": "the prefill driver's select_experts hook (runtime/vllm_semantics.install_routing)",
                           "phase": "prefill", "corpus": (line.get("corpus") or {}).get("file") or corpus, "file": str(path)})
