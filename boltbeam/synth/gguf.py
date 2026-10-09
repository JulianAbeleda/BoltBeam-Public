"""A minimal GGUF v3 writer: a tiny synthetic model for offline checks (no GPU, no real model).

It mirrors boltbeam.profile.gguf.GGUFReader. `write_gguf` emits a byte-valid GGUF the reader accepts.
Tensor dims are written in GGUF file order (the reader reverses them: profile rows,cols = reversed(dims)),
so pass file_dims = (cols, rows) for a 2D weight, or (n_expert, cols, rows) for a 3D expert stack.
`boltbeam selfcheck` uses it. The dev test kit (tests/ggufkit.py) builds its other models on top of it.
"""
from __future__ import annotations

import pathlib
import struct
from typing import Any

# GGUF metadata value type ids (the subset the reader supports)
_T_U32, _T_I32, _T_F32, _T_BOOL, _T_STRING, _T_ARRAY, _T_U64 = 4, 5, 6, 7, 8, 9, 10


def _string(s:str) -> bytes:
  b = s.encode("utf-8")
  return struct.pack("<Q", len(b)) + b


def _scalar(v:Any) -> bytes:
  if isinstance(v, bool):
    return struct.pack("<i", _T_BOOL) + struct.pack("<?", v)
  if isinstance(v, str):
    return struct.pack("<i", _T_STRING) + _string(v)
  if isinstance(v, float):
    return struct.pack("<i", _T_F32) + struct.pack("<f", v)
  if isinstance(v, int):
    return struct.pack("<i", _T_I32) + struct.pack("<i", v)
  if isinstance(v, (list, tuple)):
    # homogeneous array; only string / int element arrays are needed by the profile reader
    elem_t = _T_STRING if (v and isinstance(v[0], str)) else _T_I32
    out = struct.pack("<i", _T_ARRAY) + struct.pack("<i", elem_t) + struct.pack("<Q", len(v))
    for e in v:
      out += _string(e) if elem_t == _T_STRING else struct.pack("<i", int(e))
    return out
  raise TypeError(f"unsupported metadata value {v!r}")


def write_gguf(path:str | pathlib.Path, kv:dict[str, Any],
               tensors:list[tuple[str, tuple[int, ...], int]]) -> str:
  """Write a GGUF v3 file. `tensors` items are (name, file_dims, ggml_type)."""
  p = pathlib.Path(path)
  buf = bytearray(b"GGUF")
  buf += struct.pack("<i", 3)                                   # version
  buf += struct.pack("<q", len(tensors)) + struct.pack("<q", len(kv))
  for key, val in kv.items():
    buf += _string(key) + _scalar(val)
  for name, dims, ggml_type in tensors:
    buf += _string(name)
    buf += struct.pack("<I", len(dims))
    for d in dims:
      buf += struct.pack("<Q", int(d))
    buf += struct.pack("<i", ggml_type) + struct.pack("<Q", 0)  # offset unused by the profile reader
  p.write_bytes(bytes(buf))
  return str(p)


def dense_decoder_kv(arch:str = "qwenlike", layers:int = 2) -> dict[str, Any]:
  return {"general.architecture": arch, f"{arch}.embedding_length": 512, f"{arch}.feed_forward_length": 1376,
          f"{arch}.block_count": layers, "tokenizer.ggml.tokens": ["a", "b", "c", "d"]}


def dense_decoder_tensors(quant:int = 12, layers:int = 2) -> list[tuple[str, tuple[int, ...], int]]:
  """A tiny dense decoder: q/k/v/o + gate/up/down per layer (file dims = (cols, rows))."""
  ts:list[tuple[str, tuple[int, ...], int]] = []
  for i in range(layers):
    ts += [
      (f"blk.{i}.attn_q.weight", (512, 512), quant),
      (f"blk.{i}.attn_k.weight", (512, 512), quant),
      (f"blk.{i}.attn_v.weight", (512, 512), quant),
      (f"blk.{i}.attn_output.weight", (512, 512), quant),
      (f"blk.{i}.ffn_gate.weight", (512, 1376), quant),
      (f"blk.{i}.ffn_up.weight", (512, 1376), quant),
      (f"blk.{i}.ffn_down.weight", (1376, 512), quant),
      (f"blk.{i}.attn_norm.weight", (512,), 0),
    ]
  ts += [("token_embd.weight", (512, 4), quant), ("output.weight", (512, 4), 14)]
  return ts
