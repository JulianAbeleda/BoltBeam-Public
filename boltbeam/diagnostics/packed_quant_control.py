from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from boltbeam.core.canonical import sha256_hex

from boltbeam.profile.gguf import GGML_TYPE_NAMES, read_gguf_layout
from boltbeam.quantization.quant import quant_capability


def _split_bits(x:Any, width:int) -> Any:
  """Match ggml's lane-major unpack order: all low lanes, then all high lanes."""
  import numpy as np
  mask = (1 << width) - 1
  return np.stack([(x >> shift) & mask for shift in range(0, 8, width)], axis=-2).reshape(*x.shape[:-1], -1)


def dequant_q4_k(raw:bytes) -> Any:
  import numpy as np
  cap = quant_capability("Q4_K")
  assert cap is not None
  blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, cap.block_bytes)
  d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
  dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
  s = blocks[:, 4:16]
  scales = np.concatenate((s[:, 0:4] & 63, (s[:, 8:12] & 15) | ((s[:, 0:4] >> 6) << 4)), axis=1)
  mins = np.concatenate((s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)), axis=1)
  qs = blocks[:, 16:144].reshape(-1, 4, 32)
  q = np.stack((qs & 15, qs >> 4), axis=2).reshape(-1, 8, 32).astype(np.float32)
  return (d[:, :, None] * scales[:, :, None] * q - dmin[:, :, None] * mins[:, :, None]).reshape(-1, 256)


def dequant_q6_k(raw:bytes) -> Any:
  import numpy as np
  cap = quant_capability("Q6_K")
  assert cap is not None
  blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, cap.block_bytes)
  low = _split_bits(blocks[:, :128].reshape(-1, 2, 64), 4)
  high = _split_bits(blocks[:, 128:192].reshape(-1, 2, 32), 2) << 4
  codes = (low | high).reshape(-1, 256).astype(np.int16) - 32
  scales = blocks[:, 192:208].copy().view(np.int8).astype(np.float32).repeat(16, axis=1)
  d = blocks[:, 208:210].copy().view("<f2").astype(np.float32)
  return d * codes.astype(np.float32) * scales


_DEQUANT = {12: dequant_q4_k, 14: dequant_q6_k}


def packed_tensor_sample(model:pathlib.Path, ggml_type:int, blocks:int=4) -> tuple[dict[str, Any], bytes]:
  _, tensors, data_start = read_gguf_layout(model)
  try:
    name, dims, typ, offset = next(row for row in tensors if row[2] == ggml_type and len(row[1]) == 2)
  except StopIteration as exc:
    raise ValueError(f"{model} has no 2D {GGML_TYPE_NAMES.get(ggml_type, ggml_type)} tensor") from exc
  quant = GGML_TYPE_NAMES[typ]
  cap = quant_capability(quant)
  if cap is None: raise ValueError(f"BoltBeam has no capability record for {quant}")
  nbytes = blocks * cap.block_bytes
  with model.open("rb") as f:
    f.seek(data_start + offset)
    raw = f.read(nbytes)
  if len(raw) != nbytes: raise EOFError(f"short packed sample for {name}: wanted {nbytes}, got {len(raw)}")
  return {"tensor": name, "dims": list(dims), "ggml_type": typ, "quant": quant,
          "absolute_offset": data_start + offset, "blocks": blocks, "packed_bytes": nbytes,
          "packed_sha256": sha256_hex(raw)}, raw


def run_control(model:pathlib.Path, device:str="METAL", blocks:int=4) -> dict[str, Any]:
  import numpy as np
  import tinygrad
  from tinygrad import Tensor, dtypes
  from tinygrad.llm.gguf import ggml_data_to_tensor

  result:dict[str, Any] = {"schema": "boltbeam.packed_quant_control.v1", "model": str(model),
                           "device": device, "runtime_module": str(pathlib.Path(tinygrad.__file__).resolve()),
                           "blocks_per_quant": blocks, "cases": []}
  activation = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(1, 256)
  for ggml_type in (12, 14):
    identity, raw = packed_tensor_sample(model, ggml_type, blocks)
    oracle = _DEQUANT[ggml_type](raw).astype(np.float32)
    packed = Tensor(np.frombuffer(raw, dtype=np.uint8).copy(), dtype=dtypes.uint8, device=device)
    decoded_t = ggml_data_to_tensor(packed, blocks * 256, ggml_type).reshape(blocks, 256)
    decoded = decoded_t.realize().numpy().astype(np.float32)
    matvec = (Tensor(activation, device=device) @ decoded_t.transpose()).realize().numpy().astype(np.float32)
    expected_matvec = activation @ oracle.T
    dequant_err = np.abs(decoded - oracle)
    matvec_err = np.abs(matvec - expected_matvec)
    result["cases"].append({**identity,
      "oracle_sha256": sha256_hex(oracle.tobytes()),
      "runtime_sha256": sha256_hex(decoded.tobytes()),
      "dequant_max_abs_error": float(dequant_err.max()), "dequant_mismatch_count": int(np.count_nonzero(dequant_err)),
      "matvec_oracle": expected_matvec.flatten().tolist(), "matvec_runtime": matvec.flatten().tolist(),
      "matvec_max_abs_error": float(matvec_err.max()),
    })
  result["pass"] = all(c["dequant_mismatch_count"] == 0 and c["matvec_max_abs_error"] <= 1e-4 for c in result["cases"])
  return result


def main() -> int:
  ap = argparse.ArgumentParser(description="Compare real GGUF Q4_K/Q6_K blocks against an independent NumPy oracle")
  ap.add_argument("--model", type=pathlib.Path, required=True)
  ap.add_argument("--device", default="METAL")
  ap.add_argument("--blocks", type=int, default=4)
  ap.add_argument("--out", type=pathlib.Path)
  ns = ap.parse_args()
  result = run_control(ns.model, ns.device, ns.blocks)
  encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if ns.out is not None:
    ns.out.parent.mkdir(parents=True, exist_ok=False)
    ns.out.write_text(encoded)
  print(encoded, end="")
  return 0 if result["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
