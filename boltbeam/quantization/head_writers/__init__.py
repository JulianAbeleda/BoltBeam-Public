"""One module per engine, named by a row of data/speed_up.json (quantization/speed_up.py is the engine-neutral core).
A module holds the functions its row names: find(explicit), stock(engine), a plan per part (src, engine, target=, part=),
write(src, dst, engine, plan), accuracy(src, dst, engine, texts); with speed with MTP on, mtp(model, engine, out, how)
returning {depths: {depth: {context, base, ks, tokens}}, engine: the build, sweep} from mtp_ks() when how names no ks (how: the part row's measure,
ks or grow, and main for a draft head that is its own file) and device(engine) {target, name, backend, memory_bytes,
budget_bytes}. Every one raises core.refused.Refused to say it cannot, in one sentence. What measures MTP on an engine is
data/engine_knobs.json's (engines.<E>.mtp), called through collectors/mtp.measure."""
from __future__ import annotations

import json
import pathlib


def detect(path:str | pathlib.Path) -> str:
  """A checkpoint's format, as data/speed_up.json's formats name it: gguf, compressed-tensors/<format>,
  tensorrt-llm-engine, else the quantization method or plain safetensors. Never a model name."""
  p = pathlib.Path(path).expanduser()
  if p.is_file():
    return "gguf" if p.suffix == ".gguf" else p.suffix.lstrip(".") or "file"
  try:
    cfg = json.loads((p / "config.json").read_text()) if (p / "config.json").is_file() else {}
  except (ValueError, OSError) as exc:
    from boltbeam.core.refused import Refused
    raise Refused(f"{p.name} could not be read: config.json {type(exc).__name__} {exc}") from exc
  if "build_config" in cfg or any(p.glob("*.engine")):
    return "tensorrt-llm-engine"
  q = cfg.get("quantization_config") or {}
  if q.get("quant_method") == "compressed-tensors":
    return "compressed-tensors/" + str(q.get("format"))
  return str(q.get("quant_method") or "safetensors")

