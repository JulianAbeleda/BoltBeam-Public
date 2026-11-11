"""vLLM, AutoAWQ checkpoints (gemm, 4-bit, zero point, group): the output layer (lm_head) as RTN int4 in AWQ's own layout
(qweight, qzeros, scales), with `"lm_head": true` added to quantization_config, which stock vLLM's AWQ config reads to
quantize ParallelLMHead. Plan and write are vllm_ct's (vllm_engine.py `write` with fmt=awq). Keyed by format, not model."""
from __future__ import annotations

import json
import pathlib

from boltbeam.core.refused import Refused
from boltbeam.quantization.head_writers import vllm_ct
from boltbeam.quantization.head_writers.vllm_ct import accuracy, device, find, speed, stock  # noqa: F401


def plan(src:str, python:str, target:str | None = None) -> dict:
  p = pathlib.Path(src)
  q = json.loads((p / "config.json").read_text()).get("quantization_config") or {}
  if q.get("quant_method") != "awq" or q.get("version", "gemm").lower() != "gemm" or q.get("bits") != 4 or q.get("zero_point") is not True:
    raise Refused("only AutoAWQ 4-bit gemm checkpoints with a zero point are handled")
  if q.get("lm_head"):
    raise Refused("the output layer is already quantized (lm_head is true in quantization_config)")
  g = int(q["group_size"])
  shape, dtype, bits_now = vllm_ct.head(p, "lm_head.qweight", g)
  return vllm_ct.plan_row(shape, dtype, bits_now, f"int4 group {g}", 4 + 16 / g + 4 / g, g, 4)  # a 16 bit scale and a nibble zero point per group


def write(src:str, dst:str, python:str, pl:dict) -> dict:
  return vllm_ct.write(src, dst, python, pl, fmt="awq")
