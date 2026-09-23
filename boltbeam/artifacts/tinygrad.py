"""Tinygrad artifact adapters (audit-brain-build-scope BB2).

Turn raw tinygrad bench JSON into `NormalizedEvidence`. Detection is by JSON *shape* (which keys are present)
and, as a secondary hint, the directory name — never by an absolute producer path. Adapters preserve the raw
artifact's path (as passed in) and byte fingerprint in `EvidenceSource`, and NEVER guess a required field:
a recognized-but-thin artifact raises `AdapterIncomplete`; an unrecognized one raises `UnsupportedArtifact`.

Orthogonality (coding-principles "artifact parsing is separate from route search"): this module only
normalizes. It does not evaluate, promote, or execute anything.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Callable

from boltbeam.vocab import RowKind, Workload, TinygradArtifactKind, SCHEMA_TINYGRAD_KV_CTX_SLOPE, SCHEMA_TINYGRAD_REG_SCALAR_LOWERING
from boltbeam.artifacts.base import (NormalizedEvidence, EvidenceRow, EvidenceSource, EvidenceFlags,
                                     AdapterIncomplete, UnsupportedArtifact, sha256_bytes)

# target normalization: the only capability id BoltBeam v1 emits for AMD RDNA3.
_GFX1100 = "amd_gfx1100"

# producing-tool filenames per family (provenance only; never an absolute path).
_TOOL = {
  TinygradArtifactKind.DECODE_ROLE_ATTRIBUTION: "qk_decode_role_attribution_modular.py",
  TinygradArtifactKind.DECODE_REDUCE_SOURCE_TRACE: "qk_decode_reduce_source_trace.py",
  TinygradArtifactKind.DECODE_RUNTIME_OVERHEAD: "qk_decode_runtime_overhead.py",
  TinygradArtifactKind.PROMOTION_GATE: "amd_isa_g3_weight_promotion_gate.py",
  TinygradArtifactKind.DECODE_WD: "amd_isa_q6k_direct_speed.py",
  TinygradArtifactKind.PREFILL_AUTHORITY: "qk_prefill_authority_refresh.py",
  TinygradArtifactKind.COMPILER_PATHOLOGY: "tinygrad.compiler_pathology.v1",
  TinygradArtifactKind.KV_CTX_SLOPE: "llama_kv_ctx_slope_bench.py",
  TinygradArtifactKind.REG_SCALAR_LOWERING: "qk_tg_p10_reg_scalar_repro.py",
}


# ---- small helpers -----------------------------------------------------------------------------------------
def _load(path:str) -> tuple[bytes, dict[str, Any]]:
  p = pathlib.Path(path)
  raw = p.read_bytes()
  try:
    obj = json.loads(raw)
  except (json.JSONDecodeError, ValueError) as e:
    raise UnsupportedArtifact(f"not valid JSON: {e}", path=path)
  if not isinstance(obj, dict):
    raise UnsupportedArtifact("artifact is not a JSON object", path=path)
  return raw, obj


def _target_from(text:str | None) -> str | None:
  """Map a hardware/path string to a target id. 'gfx1100' anywhere -> amd_gfx1100."""
  if text and "gfx1100" in text.lower():
    return _GFX1100
  return None


def _ctx_int(x:Any) -> int:
  return int(x)


def _source(kind:TinygradArtifactKind, path:str, raw:bytes) -> EvidenceSource:
  return EvidenceSource(producer="tinygrad", tool=_TOOL[kind], path=path, fingerprint=sha256_bytes(raw))


def _per_ctx_has_by_role(d:dict[str, Any]) -> bool:
  pc = d.get("per_ctx")
  if not isinstance(pc, dict) or not pc:
    return False
  first = next(iter(pc.values()))
  return isinstance(first, dict) and "by_role" in first


# ---- detection ---------------------------------------------------------------------------------------------
def _detect(d:dict[str, Any]) -> TinygradArtifactKind | None:
  """Detect the artifact family from JSON shape. Order matters: the most specific signature wins."""
  if d.get("schema") == SCHEMA_TINYGRAD_KV_CTX_SLOPE:
    return TinygradArtifactKind.KV_CTX_SLOPE
  if d.get("schema") == "tinygrad.compiler_pathology.v1":
    return TinygradArtifactKind.COMPILER_PATHOLOGY
  if d.get("schema") == SCHEMA_TINYGRAD_REG_SCALAR_LOWERING:
    return TinygradArtifactKind.REG_SCALAR_LOWERING
  if "reduce_rows" in d:
    return TinygradArtifactKind.DECODE_REDUCE_SOURCE_TRACE
  if "authority_ref" in d and "by_arm" in d:
    return TinygradArtifactKind.PREFILL_AUTHORITY
  if "promotion_contract" in d or "promotion_arm" in d:
    return TinygradArtifactKind.PROMOTION_GATE
  if "wd" in d and ("tier_of_best" in d or "median_delta_pct" in d):
    return TinygradArtifactKind.DECODE_WD
  if "median_host_sync_pct" in d or ("rows" in d and "hardware" in d):
    return TinygradArtifactKind.DECODE_RUNTIME_OVERHEAD
  if "per_ctx" in d and _per_ctx_has_by_role(d):
    return TinygradArtifactKind.DECODE_ROLE_ATTRIBUTION
  return None


def detect_kind(path:str) -> TinygradArtifactKind | None:
  """Detect the tinygrad artifact family at `path`, or None if no adapter recognizes it."""
  _, d = _load(path)
  return _detect(d)


# ---- adapters ----------------------------------------------------------------------------------------------
def _adapt_role_attribution(path:str, raw:bytes, d:dict[str, Any], model_id:str | None = None, target_id:str | None = None) -> NormalizedEvidence:
  kind = TinygradArtifactKind.DECODE_ROLE_ATTRIBUTION
  model_id = model_id or d.get("model_id")
  if not model_id:
    raise AdapterIncomplete("role attribution artifact lacks model_id", path=path)
  per_ctx = d.get("per_ctx")
  if not isinstance(per_ctx, dict) or not per_ctx:
    raise AdapterIncomplete("role attribution artifact has no per_ctx buckets", path=path)

  rows:list[EvidenceRow] = []
  contexts:list[int] = []
  for ctx_key, block in per_ctx.items():
    ctx = _ctx_int(block.get("ctx", ctx_key))
    contexts.append(ctx)
    for role in block.get("by_role", []):
      pct = role.get("pct")
      if pct is None:
        continue
      quants = role.get("quants") or []
      rows.append(EvidenceRow(
        RowKind.ROLE_ATTRIBUTION.value, "wall_share", pct / 100.0, "fraction",
        role=role.get("name"), quant=(quants[0] if quants else None), context=ctx,
        extra={"route_classes": role.get("route_classes", []), "bytes_per_step": role.get("bytes_per_step"),
               "kernels": role.get("kernels")}))
    for kt in block.get("by_kernel_top", []):
      pct = kt.get("pct_of_gpu_compute")
      if pct is None:
        continue
      rows.append(EvidenceRow(
        RowKind.ROLE_ATTRIBUTION.value, "pct_of_gpu_compute", float(pct), "%",
        role=kt.get("role"), quant=kt.get("quant"), shape=kt.get("matdims"), context=ctx,
        extra={"kernel": kt.get("kernel"), "route_class": kt.get("route_class"),
               "reduce_class": kt.get("reduce_class"), "is_weight": kt.get("is_weight"),
               "bytes_per_call": kt.get("bytes_per_call"), "calls_per_step": kt.get("calls_per_step")}))

  return NormalizedEvidence(
    model_id=model_id, target_id=target_id or _GFX1100, workload=Workload.DECODE.value,
    source=_source(kind, path, raw), rows=tuple(rows), contexts=tuple(sorted(set(contexts))))


def _adapt_reduce_source(path:str, raw:bytes, d:dict[str, Any], model_id:str | None = None, target_id:str | None = None) -> NormalizedEvidence:
  kind = TinygradArtifactKind.DECODE_REDUCE_SOURCE_TRACE
  # Attribution must be explicit: use an explicit --model-id override, else a REAL top-level "model_id".
  # The run-label `id` (e.g. "qwen3-14b-promoted") is a bench-run tag, NOT a model id, so never derive from it.
  model_id = model_id or d.get("model_id")
  if not model_id:
    raise AdapterIncomplete("reduce-source trace lacks model_id; pass --model-id", path=path)
  if "ctx" not in d:
    raise AdapterIncomplete("reduce-source trace lacks ctx", path=path)
  ctx = _ctx_int(d["ctx"])

  rows:list[EvidenceRow] = []
  for rr in d.get("reduce_rows", []):
    pct = rr.get("pct_gpu")
    if pct is None:
      continue
    rows.append(EvidenceRow(
      RowKind.REDUCE_SOURCE.value, "pct_gpu", float(pct), "%", context=ctx,
      extra={"kernel": rr.get("kernel"), "calls_in_step": rr.get("calls_in_step"),
             "shape_factors": rr.get("shape_factors", []), "prev": rr.get("prev_nonreduce", []),
             "next": rr.get("next_nonreduce", [])}))

  return NormalizedEvidence(
    model_id=model_id, target_id=target_id or _GFX1100, workload=Workload.DECODE.value,
    source=_source(kind, path, raw), rows=tuple(rows), contexts=(ctx,))


def _adapt_runtime_overhead(path:str, raw:bytes, d:dict[str, Any], model_id:str | None = None, target_id:str | None = None) -> NormalizedEvidence:
  kind = TinygradArtifactKind.DECODE_RUNTIME_OVERHEAD
  model_id = model_id or d.get("model_id")
  if not model_id:
    raise AdapterIncomplete("runtime-overhead artifact lacks model_id", path=path)
  target_id = target_id or _target_from(d.get("hardware")) or _target_from(path) or _GFX1100

  rows:list[EvidenceRow] = []
  contexts:list[int] = []
  for r in d.get("rows", []):
    ctx = _ctx_int(r["ctx"])
    contexts.append(ctx)
    extra = {"flash": r.get("flash")}
    if r.get("tok_s_W") is not None:
      rows.append(EvidenceRow(RowKind.RUNTIME_OVERHEAD.value, "tok_s_W", float(r["tok_s_W"]), "tok/s",
                              context=ctx, extra=dict(extra, tok_s_D_ceiling=r.get("tok_s_D_ceiling"))))
    if r.get("host_sync_pct_of_wall") is not None:
      rows.append(EvidenceRow(RowKind.RUNTIME_OVERHEAD.value, "host_sync_pct",
                              float(r["host_sync_pct_of_wall"]), "%", context=ctx, extra=dict(extra)))
    if r.get("programs_per_token") is not None:
      rows.append(EvidenceRow(RowKind.RUNTIME_OVERHEAD.value, "programs_per_token",
                              float(r["programs_per_token"]), "programs", context=ctx, extra=dict(extra)))

  return NormalizedEvidence(
    model_id=model_id, target_id=target_id, workload=Workload.DECODE.value,
    source=_source(kind, path, raw), rows=tuple(rows), contexts=tuple(sorted(set(contexts))))


def _all_true(vals:list[Any]) -> bool | None:
  """True iff there is at least one value and every present value is truthy; None if none observed."""
  present = [v for v in vals if v is not None]
  if not present:
    return None
  return all(bool(v) for v in present)


def _max_spread(*vals:Any) -> float | None:
  """The MAX of the present measurement-spread values (conservative: the widest spread wins), else None."""
  present = [float(v) for v in vals if v is not None]
  return max(present) if present else None


def _hidden_fallback_from_routes(clean_vals:list[Any], leaked_vals:list[Any]) -> bool | None:
  """Derive hidden_fallback from per-ctx route_clean / leaked_routes signals (mirrors the prefill adapter's
  tensile_fallback extraction, but computed from the promotion-gate route facts).

  None  if no block carries route_clean or leaked_routes at all (unknown, never inferred);
  False if every present route_clean is True and every present leaked_routes is empty (route is clean);
  True  otherwise (a block was not route-clean, or leaked into another route -> a hidden fallback fired).
  """
  has_signal = any(c is not None for c in clean_vals) or any(l is not None for l in leaked_vals)
  if not has_signal:
    return None
  all_clean = all(bool(c) for c in clean_vals if c is not None)
  no_leaks = all(not l for l in leaked_vals if l is not None)
  return False if (all_clean and no_leaks) else True


def _adapt_promotion_gate(path:str, raw:bytes, d:dict[str, Any], model_id:str | None = None, target_id:str | None = None) -> NormalizedEvidence:
  kind = TinygradArtifactKind.PROMOTION_GATE
  model_id = model_id or d.get("model_id")
  if not model_id:
    raise AdapterIncomplete("promotion-gate artifact lacks model_id", path=path)
  per_ctx = d.get("per_ctx")
  if not isinstance(per_ctx, dict) or not per_ctx:
    raise AdapterIncomplete("promotion-gate artifact has no per_ctx buckets", path=path)

  rows:list[EvidenceRow] = []
  contexts:list[int] = []
  tmatch:list[Any] = []
  clean:list[Any] = []
  leaked:list[Any] = []
  for ctx_key, block in per_ctx.items():
    ctx = _ctx_int(block.get("ctx", ctx_key))
    contexts.append(ctx)
    tmatch.append(block.get("token_match"))
    clean.append(block.get("route_clean"))
    leaked.append(block.get("leaked_routes"))
    if block.get("lag_pct") is None:
      continue
    extra:dict[str, Any] = {"owned_tok_s": block.get("owned_tok_s"), "g3_tok_s": block.get("g3_tok_s"),
                            "g3_fired": block.get("g3_fired"), "route_clean": block.get("route_clean")}
    spread = _max_spread(block.get("g3_spread_pct"), block.get("owned_spread_pct"))
    if spread is not None:
      extra["spread_pct"] = spread     # widest measurement spread for this ctx (noise floor for the evaluator)
    rows.append(EvidenceRow(RowKind.WD_SPEED.value, "lag_pct", float(block["lag_pct"]), "%", context=ctx, extra=extra))

  flags = EvidenceFlags(token_match=_all_true(tmatch), route_bound=_all_true(clean),
                        hidden_fallback=_hidden_fallback_from_routes(clean, leaked))
  return NormalizedEvidence(
    model_id=model_id, target_id=target_id or _GFX1100, workload=Workload.DECODE.value,
    source=_source(kind, path, raw), rows=tuple(rows), contexts=tuple(sorted(set(contexts))), flags=flags)


def _adapt_decode_wd(path:str, raw:bytes, d:dict[str, Any], model_id:str | None = None, target_id:str | None = None) -> NormalizedEvidence:
  kind = TinygradArtifactKind.DECODE_WD
  model_id = model_id or d.get("model_id")
  if not model_id:
    raise AdapterIncomplete("decode-wd artifact lacks model_id", path=path)
  wd = d.get("wd")
  if not isinstance(wd, dict) or not wd:
    raise AdapterIncomplete("decode-wd artifact has no wd contexts", path=path)

  rows:list[EvidenceRow] = []
  contexts:list[int] = []
  tmatch:list[Any] = []
  for ctx_key, block in wd.items():
    ctx = _ctx_int(ctx_key)
    contexts.append(ctx)
    tmatch.append(block.get("token_match"))
    if block.get("delta_pct") is None:
      continue
    extra:dict[str, Any] = {"baseline_tok_s": block.get("baseline_tok_s"),
                            "candidate_tok_s": block.get("candidate_tok_s")}
    spread = _max_spread(block.get("candidate_spread_pct"), block.get("baseline_spread_pct"))
    if spread is not None:
      extra["spread_pct"] = spread     # widest measurement spread for this ctx (noise floor for the evaluator)
    rows.append(EvidenceRow(RowKind.WD_SPEED.value, "delta_pct", float(block["delta_pct"]), "%", context=ctx, extra=extra))

  flags = EvidenceFlags(
    token_match=(d.get("token_match_all_ctx") if "token_match_all_ctx" in d else _all_true(tmatch)),
    route_bound=d.get("route_bound_all_ctx"))
  return NormalizedEvidence(
    model_id=model_id, target_id=target_id or _GFX1100, workload=Workload.DECODE.value,
    source=_source(kind, path, raw), rows=tuple(rows), contexts=tuple(sorted(set(contexts))), flags=flags)


def _adapt_prefill_authority(path:str, raw:bytes, d:dict[str, Any], model_id:str | None = None,
                             target_id:str | None = None) -> NormalizedEvidence:
  kind = TinygradArtifactKind.PREFILL_AUTHORITY
  model_id = model_id or d.get("model_id")
  if not model_id:
    raise AdapterIncomplete("prefill authority artifact lacks model_id", path=path)
  by_arm = d.get("by_arm")
  if not isinstance(by_arm, dict) or "current_default" not in by_arm:
    raise AdapterIncomplete("prefill authority artifact lacks a current_default arm", path=path)
  arm = by_arm["current_default"]
  tok = arm.get("whole_prefill_tok_s")
  if not isinstance(tok, dict) or not tok:
    raise AdapterIncomplete("prefill authority current_default arm lacks whole_prefill_tok_s", path=path)
  # authority_ref is the pinned baseline tok/s per ctx; emit tok_s rows WITH a baseline so the evaluator can
  # score a signed delta (without a baseline the row would be un-scorable -> inconclusive).
  authority_ref = d.get("authority_ref") if isinstance(d.get("authority_ref"), dict) else {}
  spread = arm.get("chunk_spread_pct") if isinstance(arm.get("chunk_spread_pct"), dict) else {}

  rows:list[EvidenceRow] = []
  contexts:list[int] = []
  for ctx_key, val in tok.items():
    if val is None:
      continue
    ctx = _ctx_int(ctx_key)
    contexts.append(ctx)
    extra:dict[str, Any] = {"arm": "current_default", "route_kernels": arm.get("route_kernels", [])}
    base = authority_ref.get(ctx_key) or authority_ref.get(str(ctx))
    if base is not None: extra["baseline_tok_s"] = float(base)
    sp = spread.get(ctx_key) or spread.get(str(ctx))
    if sp is not None: extra["spread_pct"] = float(sp)
    rows.append(EvidenceRow(
      RowKind.PREFILL_SPEED.value, "tok_s", float(val), "tok/s", context=ctx, extra=extra))

  flags = EvidenceFlags(hidden_fallback=arm.get("tensile_fallback"))
  return NormalizedEvidence(
    model_id=model_id, target_id=target_id or _GFX1100, workload=Workload.PREFILL.value,
    source=_source(kind, path, raw), rows=tuple(rows), contexts=tuple(sorted(set(contexts))), flags=flags)


# ---- registry ----------------------------------------------------------------------------------------------
def _adapt_compiler_pathology(path:str, raw:bytes, d:dict[str, Any], model_id:str | None = None, target_id:str | None = None) -> NormalizedEvidence:
  """Adapter for tinygrad.compiler_pathology.v1 artifacts.

  Emits compiler_pathology RowKind rows — one per metric per kernel entry. Baseline timing
  is carried in each row's extra so the classifier can compute ratios without cross-row joins.
  Never guesses missing resource counts as zero (raises AdapterIncomplete instead).
  """
  kind = TinygradArtifactKind.COMPILER_PATHOLOGY
  model_id = model_id or d.get("model_id")
  if not model_id:
    raise AdapterIncomplete("compiler_pathology artifact lacks model_id", path=path)
  workload = d.get("workload", "decode")
  candidate_id = d.get("candidate_id")
  if not candidate_id:
    raise AdapterIncomplete("compiler_pathology artifact lacks candidate_id", path=path)

  baseline = d.get("baseline", {})
  baseline_kernel = baseline.get("kernel")
  baseline_us = baseline.get("time_us_per_workgroup")
  baseline_lds = baseline.get("lds_bytes")

  kernels = d.get("kernels", [])
  if not kernels:
    raise AdapterIncomplete("compiler_pathology artifact has no kernels list", path=path)

  rows: list[EvidenceRow] = []
  contexts: list[int] = []

  for kentry in kernels:
    kname = kentry.get("kernel")
    role = kentry.get("role")
    ctx = kentry.get("context")
    if ctx is not None:
      contexts.append(int(ctx))
    base_extra = {
      "kernel": kname,
      "candidate_id": candidate_id,
      "baseline_kernel": baseline_kernel,
      "baseline_time_us_per_workgroup": baseline_us,
      "sources": kentry.get("sources", {}),
    }

    def _row(metric: str, value: float | int, unit: str, extra: dict | None = None) -> EvidenceRow:
      return EvidenceRow(RowKind.COMPILER_PATHOLOGY.value, metric, float(value), unit,
                         role=role, context=int(ctx) if ctx is not None else None,
                         extra={**base_extra, **(extra or {})})

    # Timing rows — always emit if present.
    cand_us = kentry.get("time_us_per_workgroup")
    if cand_us is not None:
      rows.append(_row("time_us_per_workgroup", cand_us, "us/workgroup"))
      rows.append(_row("baseline_time_us_per_workgroup", baseline_us or 0.0, "us/workgroup"))
      if baseline_us and baseline_us > 0:
        rows.append(_row("time_ratio_vs_baseline", cand_us / baseline_us, "ratio"))

    # Resource rows — only emit when present; never default missing to zero.
    _RESOURCE_METRICS: list[tuple[str, str]] = [
      ("vgpr", "registers"), ("sgpr", "registers"),
      ("scratch_bytes", "bytes"), ("lds_bytes", "bytes"),
      ("vector_load_bits", "bits"), ("occupancy_waves_per_cu", "waves/cu"),
    ]
    for field_name, unit in _RESOURCE_METRICS:
      v = kentry.get(field_name)
      if v is not None:
        rows.append(_row(field_name, v, unit))

    # Baseline LDS — emit as expected_lds_bytes so classifier can compute ratio.
    if baseline_lds is not None:
      rows.append(_row("expected_lds_bytes", baseline_lds, "bytes"))

    # Instruction counts — from nested static_counts dict.
    sc = kentry.get("static_counts", {})
    _INSTR_MAP: list[tuple[str, str]] = [
      ("total", "static_instr"), ("valu", "valu_instr"), ("salu", "salu_instr"),
      ("vmem", "vmem_instr"), ("lds", "lds_instr"),
      ("barrier", "barrier_count"), ("waitcnt", "waitcnt_count"),
    ]
    for sc_key, metric_name in _INSTR_MAP:
      v = sc.get(sc_key)
      if v is not None:
        rows.append(_row(metric_name, v, "instructions"))

  source = EvidenceSource(
    producer="tinygrad",
    tool="tinygrad.compiler_pathology.v1",
    path=path,
    fingerprint=sha256_bytes(raw),
    raw_summary={"candidate_id": candidate_id, "kernel_count": len(kernels)},
  )
  return NormalizedEvidence(
    model_id=model_id,
    target_id=target_id or _target_from(d.get("target_id")) or _GFX1100,
    workload=workload,
    source=source,
    rows=tuple(rows),
    contexts=tuple(sorted(set(contexts))),
  )


def _adapt_kv_ctx_slope(path:str, raw:bytes, d:dict[str, Any], model_id:str | None = None,
                        target_id:str | None = None) -> NormalizedEvidence:
  kind = TinygradArtifactKind.KV_CTX_SLOPE
  model_id = model_id or d.get("model_id")
  if not model_id:
    raise AdapterIncomplete("KV context-slope artifact lacks model_id", path=path)
  meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
  fit = d.get("fit") if isinstance(d.get("fit"), dict) else {}
  config = d.get("config") if isinstance(d.get("config"), dict) else {}
  rows_in = d.get("rows") if isinstance(d.get("rows"), list) else []
  b = fit.get("b_ms_per_ctx")
  if b is None:
    raise AdapterIncomplete("KV context-slope artifact lacks fit.b_ms_per_ctx", path=path)
  kv_bytes = meta.get("kv_bytes_per_ctx_token")
  if kv_bytes is None:
    raise AdapterIncomplete("KV context-slope artifact lacks meta.kv_bytes_per_ctx_token", path=path)

  contexts = sorted({int(r["ctx"]) for r in rows_in if isinstance(r, dict) and r.get("ok") and r.get("ctx") is not None})
  extra = {
    "cache_type_k": meta.get("cache_type_k") or config.get("cache_type_k"),
    "cache_type_v": meta.get("cache_type_v") or config.get("cache_type_v"),
    "a_ms": fit.get("a_ms"),
    "r2": fit.get("r2"),
    "kv_bytes_per_ctx_token": kv_bytes,
    "implied_kv_bandwidth_gb_s": fit.get("implied_kv_bandwidth_gb_s"),
    "depths": contexts or config.get("depths", []),
    "producer_schema": d.get("schema"),
  }
  for key in ("storage_only_b_ms_per_ctx", "quant_residual_b_ms_per_ctx",
              "baseline_b_ms_per_ctx", "baseline_kv_bytes_per_ctx_token",
              "baseline_cache_type_k", "baseline_cache_type_v"):
    if key in d:
      extra[key] = d[key]
    elif isinstance(d.get("residual"), dict) and key in d["residual"]:
      extra[key] = d["residual"][key]

  ev_row = EvidenceRow(RowKind.KV_CTX_SLOPE.value, "b_ms_per_ctx", float(b), "ms/context-token", extra=extra)
  return NormalizedEvidence(
    model_id=model_id,
    target_id=target_id or d.get("target_id") or _target_from(d.get("hardware")) or _GFX1100,
    workload=Workload.DECODE.value,
    source=_source(kind, path, raw),
    rows=(ev_row,),
    contexts=tuple(contexts))


def _adapt_reg_scalar_lowering(path:str, raw:bytes, d:dict[str, Any], model_id:str | None = None,
                               target_id:str | None = None) -> NormalizedEvidence:
  """Adapter for tinygrad.reg_scalar_lowering.v1: a generated-UOp REG/reduction-accumulator lowering repro.

  Emits one COMPILER_LOWERING row per case (metric=case_id; value=1 iff the case is numerically correct) carrying the
  per-case failure signature in `extra` (compile_ok / numeric_ok / error_class / reg_accumulator_observed). The
  classifier (boltbeam/diagnostics/reg_lowering.py) reads these to decide EMITTER_BLOCKED vs reachable/reopened.
  """
  mid = model_id or d.get("model_id")
  if not mid: raise AdapterIncomplete("reg_scalar_lowering artifact lacks model_id", path=path)
  cases = d.get("cases")
  if not isinstance(cases, list) or not cases:
    raise AdapterIncomplete("reg_scalar_lowering artifact has no cases list", path=path)
  rows = []
  for c in cases:
    cid = c.get("case_id")
    if not cid: raise AdapterIncomplete("reg_scalar_lowering case lacks case_id", path=path)
    rows.append(EvidenceRow(
      RowKind.COMPILER_LOWERING.value, cid, 1.0 if c.get("numeric_ok") else 0.0, "bool",
      extra={"compile_ok": bool(c.get("compile_ok")), "runtime_ok": bool(c.get("runtime_ok")),
             "numeric_ok": bool(c.get("numeric_ok")), "error_class": str(c.get("error_class", "")),
             "reg_accumulator_expected": str(c.get("reg_accumulator_expected", "scalar")),
             "reg_accumulator_observed": str(c.get("reg_accumulator_observed", "unknown")),
             "uses_reg_store_devec": bool(c.get("uses_reg_store_devec")),
             "generated_uop_only": bool(c.get("generated_uop_only", True)),
             "uses_external_kernel": bool(c.get("uses_external_kernel", False))}))
  return NormalizedEvidence(
    model_id=mid,
    target_id=target_id or d.get("target_id") or _GFX1100,
    workload=Workload.DECODE.value,
    source=_source(TinygradArtifactKind.REG_SCALAR_LOWERING, path, raw),
    rows=tuple(rows))


_REGISTRY:dict[TinygradArtifactKind, Callable[[str, bytes, dict[str, Any], "str | None", "str | None"], NormalizedEvidence]] = {
  TinygradArtifactKind.DECODE_ROLE_ATTRIBUTION: _adapt_role_attribution,
  TinygradArtifactKind.DECODE_REDUCE_SOURCE_TRACE: _adapt_reduce_source,
  TinygradArtifactKind.DECODE_RUNTIME_OVERHEAD: _adapt_runtime_overhead,
  TinygradArtifactKind.PROMOTION_GATE: _adapt_promotion_gate,
  TinygradArtifactKind.DECODE_WD: _adapt_decode_wd,
  TinygradArtifactKind.PREFILL_AUTHORITY: _adapt_prefill_authority,
  TinygradArtifactKind.COMPILER_PATHOLOGY: _adapt_compiler_pathology,
  TinygradArtifactKind.KV_CTX_SLOPE: _adapt_kv_ctx_slope,
  TinygradArtifactKind.REG_SCALAR_LOWERING: _adapt_reg_scalar_lowering,
}


def normalize(path:str, *, model_id:str | None = None, target_id:str | None = None) -> NormalizedEvidence:
  """Normalize the tinygrad artifact at `path` into NormalizedEvidence.

  `model_id`/`target_id`, when provided, override/fill those ids for any family. `model_id` is required for
  artifacts (like reduce-source traces) that do not carry a real top-level model_id. `target_id` lets a
  non-gfx1100 artifact be labeled correctly instead of silently defaulting to amd_gfx1100.

  Raises `UnsupportedArtifact` if no adapter recognizes the shape, `AdapterIncomplete` if a recognized
  artifact is missing a required field.
  """
  raw, d = _load(path)
  kind = _detect(d)
  if kind is None or kind not in _REGISTRY:
    raise UnsupportedArtifact(f"no tinygrad adapter recognizes artifact at {path}", path=path)
  return _REGISTRY[kind](path, raw, d, model_id, target_id)


def normalize_dir(directory:str, *, model_id:str | None = None, target_id:str | None = None) -> list[NormalizedEvidence]:
  """Best-effort normalize every recognizable JSON artifact under `directory`.

  Unrecognized or incomplete artifacts are skipped (their reasons collected on `normalize_dir.last_skipped`,
  not raised); only genuinely normalizable artifacts are returned. `model_id` is threaded to every adapter.
  """
  out:list[NormalizedEvidence] = []
  skipped:list[tuple[str, str]] = []
  for p in sorted(pathlib.Path(directory).rglob("*.json")):
    sp = str(p)
    try:
      out.append(normalize(sp, model_id=model_id, target_id=target_id))
    except (UnsupportedArtifact, AdapterIncomplete) as e:
      skipped.append((sp, getattr(e, "reason", str(e))))
  normalize_dir.last_skipped = skipped  # type: ignore[attr-defined]
  return out
