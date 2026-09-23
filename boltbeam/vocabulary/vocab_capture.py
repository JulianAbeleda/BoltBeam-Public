"""Ingest an empirical vocab capture (real scheduled ISA of a model) and diff it against the registry.

The declarative extractor says what a model SHOULD need; this says what the codegen ACTUALLY emits. Mapping the
real ISA histogram -> primitive classes gives three ground-truth buckets the declarative view cannot:
  - USED:               primitives that actually appear in the model's real kernels
  - LOWERED_BUT_UNUSED: primitives the substrate can lower but the default route does not emit (e.g. default-off
                        v_dot2/v_dot4 — the capability exists but nothing uses it)
  - BACKLOG:            registry primitives with no lowering (unchanged)
plus UNMAPPED ISA mnemonics (mapping gaps to fix). Pure analysis."""
from __future__ import annotations
from typing import Any

from boltbeam.vocabulary.model_vocab import load_primitive_vocab

# RDNA3 ISA mnemonic prefix -> registry primitive. First matching prefix wins; order matters (specific first).
_ISA_TO_PRIMITIVE: tuple[tuple[str, str], ...] = (
  ("v_dot2_f32_f16", "v_dot2_f32_f16"), ("v_dot2", "v_dot2_f32_f16"),
  ("v_dot4", "v_dot4_i32_i8"), ("v_dot8", "v_dot8_i32_i4"),
  ("ds_bpermute", "xlane_reduce"), ("ds_permute", "xlane_reduce"), ("v_permlane", "xlane_reduce"),
  ("v_bfe", "dequant_bitunpack"), ("v_alignbit", "dequant_bitunpack"),
  ("v_cvt_f32_ubyte", "dequant_convert"),
  ("v_exp", "transcendental"), ("v_rcp", "transcendental"), ("v_rsq", "transcendental"),
  ("v_log", "transcendental"), ("v_sin", "transcendental"), ("v_cos", "transcendental"),
  # float division lowers to a v_div_scale / v_div_fmas / v_div_fixup sequence plus v_ldexp, not one opcode
  ("v_div_scale", "transcendental"), ("v_div_fmas", "transcendental"), ("v_div_fixup", "transcendental"),
  ("v_ldexp", "transcendental"),
  # packed-f16 ALU: two lanes of the same elementwise work
  ("v_pk_add_f16", "elementwise"), ("v_pk_mul_f16", "elementwise"), ("v_pk_fma_f16", "elementwise"),
  ("v_pk_max_f16", "elementwise"), ("v_pk_min_f16", "elementwise"),
  ("v_sub_f", "elementwise"), ("v_dual_sub_f", "elementwise"), ("v_dual_fmaak", "elementwise"),
  ("v_rndne", "elementwise"), ("v_pack_b32_f16", "elementwise"), ("v_cvt_i32_f32", "elementwise"),
  ("v_sqrt", "transcendental"),
  ("global_load", "global_load"),
  ("ds_load", "lds_stage"), ("ds_store", "lds_stage"),
  ("v_fma", "elementwise"), ("v_fmac", "elementwise"), ("v_fmaak", "elementwise"),
  ("v_fmamk", "elementwise"), ("v_mul_f", "elementwise"), ("v_add_f", "elementwise"), ("v_mad", "elementwise"),
  ("v_dual_mul_f", "elementwise"), ("v_dual_add_f", "elementwise"), ("v_dual_fmac", "elementwise"),
  ("v_cvt_f16_f32", "elementwise"), ("v_cvt_f32_f16", "elementwise"), ("v_cvt_f32_i32", "dequant_convert"),
)
# Control / addressing (integer index math) / generic bit-logic / movement — not vocabulary primitives.
# NOTE reduce has NO distinct ISA mnemonic (it is a loop of elementwise + control flow), so ISA-level capture
# cannot detect it directly; that is a known limitation of the empirical layer, not a real "unused".
# Stores are deliberately ignored rather than mapped: the registry has no store primitive, and adding one
# would be foundational — unconditional on every kernel, never a choice, so it could not change any lever
# count. Recorded here as a decision, not an oversight.
_IGNORE_PREFIXES = ("global_store", "buffer_store", "scratch_store", "ds_store_b",
                    "v_pk_lshrrev", "v_pk_ashrrev", "v_pk_lshlrev", "v_pk_add_u", "v_pk_sub_u",
                    "v_or_b16", "v_xad", "v_clz", "v_ctz", "v_bfrev",
                    "v_mul_u32_u24", "v_subrev_nc", "v_add_lshl",
                    "s_", "v_mov", "v_cndmask", "v_readlane", "v_readfirstlane", "v_writelane", "v_accvgpr",
                    "v_add_nc", "v_add_co", "v_sub_nc", "v_sub_co", "v_lshl_add", "v_lshl_or", "v_mad_u64",
                    "v_mul_lo", "v_mul_hi", "v_lshlrev", "v_lshrrev", "v_and", "v_or_b32", "v_xor", "v_not",
                    "v_min", "v_max", "v_cmp", "v_bcnt", "v_ashrrev", "v_dual_mov", "v_add3", "v_perm_b32",
                    "v_dual_cndmask", "v_dual_lshlrev")


def _map_mnemonic(m: str) -> str | None:
  for pref, prim in _ISA_TO_PRIMITIVE:
    if m.startswith(pref): return prim
  return None


def ingest_capture(capture: dict[str, Any], registry: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
  reg = registry if registry is not None else load_primitive_vocab()
  hist: dict[str, int] = capture.get("isa_histogram", {})
  used: dict[str, int] = {}          # primitive -> total ISA count
  unmapped: dict[str, int] = {}
  for mnem, cnt in hist.items():
    prim = _map_mnemonic(mnem)
    if prim is not None: used[prim] = used.get(prim, 0) + cnt
    elif not any(mnem.startswith(p) for p in _IGNORE_PREFIXES): unmapped[mnem] = cnt
  used_set = set(used)
  lowered = {n for n, p in reg.items() if p.get("lowered")}
  backlog = {n for n, p in reg.items() if not p.get("lowered")}
  return {
    "schema": "boltbeam.vocab_capture_report.v1",
    "model": capture.get("model"), "n_kernels": capture.get("n_kernels"),
    "used": dict(sorted(used.items(), key=lambda kv: -kv[1])),
    "lowered_but_unused": sorted(lowered - used_set),
    "backlog": sorted(backlog),
    "unmapped_isa": dict(sorted(unmapped.items(), key=lambda kv: -kv[1])[:20]),
  }
