"""The GGUF formats and the pure-Python reference every GEMV is held to (collectors/gemv_reference.py), on any GPU."""
from __future__ import annotations

import random
import struct

import pytest

from boltbeam.collectors import gemv_reference


def _q4_k_block(rng:random.Random) -> bytes:
  return struct.pack("<ee", rng.uniform(0.001, 0.05), rng.uniform(0.0, 0.02)) + bytes(rng.randrange(256) for _ in range(140))


def _q6_k_block(rng:random.Random) -> bytes:
  return bytes(rng.randrange(256) for _ in range(192)) + struct.pack("<16b", *(rng.randrange(-64, 64) for _ in range(16))) \
    + struct.pack("<e", rng.uniform(0.001, 0.05))


BLOCK = {"Q4_K": _q4_k_block, "Q6_K": _q6_k_block}


@pytest.mark.parametrize("quant", ["Q4_K", "Q6_K"])
def test_cpu_reference_matches_the_numpy_oracle(quant):
  np = pytest.importorskip("numpy")
  from boltbeam.diagnostics import packed_quant_control as oracle
  rng = random.Random(quant)
  raw = b"".join(BLOCK[quant](rng) for _ in range(4))
  ours = [v for b in range(4) for v in gemv_reference.DEQUANT[quant](raw[b * gemv_reference.BLOCK_BYTES[quant]:(b + 1) * gemv_reference.BLOCK_BYTES[quant]])]
  theirs = {"Q4_K": oracle.dequant_q4_k, "Q6_K": oracle.dequant_q6_k}[quant](raw).reshape(-1)
  assert np.allclose(np.array(ours, dtype=np.float32), theirs, rtol=1e-5, atol=1e-6)


def test_check_rows_cover_the_first_and_last_row():
  assert gemv_reference.check_rows(1024)[0] == 0 and gemv_reference.check_rows(1024)[-1] == 1023
  assert gemv_reference.check_rows(3) == [0, 1, 2]


def test_q8_0_blocks_dequantize_as_d_times_q_and_the_layouts_agree_with_the_quant_registry():
  from boltbeam.collectors import engine_kernels as ek
  from boltbeam.quantization.quant import quant_capability
  blk = struct.pack("<e", 0.5) + struct.pack("<32b", *range(-16, 16))  # block_q8_0: half d, 32 int8
  assert gemv_reference.dequant_q8_0_block(blk) == [0.5 * q for q in range(-16, 16)]
  assert gemv_reference.reference_row(blk * 4, "Q8_0", 1, 64, [1.0] * 64) == pytest.approx(2 * 0.5 * sum(range(-16, 16)))  # row 1: blocks 2 and 3
  for quant in gemv_reference.DEQUANT:  # one source of block facts: data/quants.json; the reference's layouts repeat it and are pinned here
    cap = quant_capability(quant)
    assert (cap.block_elems, cap.block_bytes, cap.ggml_type) == (gemv_reference.BLOCK_ELEMS[quant], gemv_reference.BLOCK_BYTES[quant], gemv_reference.GGML_TYPES[quant])
  # one ggml type table: the engine probes read the reference's
  assert ek.GGML_TYPE is gemv_reference.GGML_TYPES and set(gemv_reference.BLOCK_BYTES) == set(gemv_reference.BLOCK_ELEMS) == set(gemv_reference.DEQUANT) == set(gemv_reference.METADATA_BYTES)


def test_an_expert_role_binds_the_first_matrix_of_its_stacked_tensor():
  """DeepSeek-Coder-V2-Lite (cross-model check 2026-10-21): a layer's 64 experts are one GGUF tensor (2048, 1408, 64);
  the probe's 1408x2048 Q4_K role found no 2-D tensor of that shape and the decode run stopped before measuring."""
  q4k = gemv_reference.GGML_TYPES["Q4_K"]
  tensors = [("blk.1.ffn_gate_shexp.weight", (2048, 2816), q4k, 0), ("blk.1.ffn_gate_exps.weight", (2048, 1408, 64), q4k, 4096)]
  assert gemv_reference.tensor_for("Q4_K", 1408, 2048, tensors, "blk.1.ffn_gate_exps.weight", 64) == ("blk.1.ffn_gate_exps.weight", 4096)
  with pytest.raises(ValueError, match="no Q4_K tensor of shape 1408x2048"):
    gemv_reference.tensor_for("Q4_K", 1408, 2048, tensors)  # a dense role never binds an expert stack
