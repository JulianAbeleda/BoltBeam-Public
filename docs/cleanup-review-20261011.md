# Cleanup review, 2026-10-11

The review was checked against BoltBeam `cbf3d09` and read-only tinygrad metadata at
`f9e245c4e809e95282431130e2848c33c85b4ee3`. No GPU or remote machine was used. Fork development is outside this cleanup.

## Verdicts and timing authorities

- A1–A8: BoltBeam's own reproduced timings decide. Missing or partial unfused measurements cannot pass a fusion;
  the role comparison uses the installed kernel timed alone and the chip band. The fastest own measurement must
  reproduce on every search path; the installed candidate wins a tie inside the band. Provider timings retain an
  explicit ordering-only label. Refutations cite the check artifact, not the provider's search artifact.
- A13–A15, A17, A21, C10, C11: untimed weights are included in the remainder's description; an unidentified attention
  ideal has a KV label; an isolated whole-step rate describes its kernel sum and keeps the untraced rate separately;
  a shapeless aggregate cannot be assigned to two shaped roles. KV and flash share the profile's attention-layer
  count. The timer owns the noisy rule; attribution uses the same run ceiling peak, and explicit peak input wins.
- A16 is mitigated, not fully fixed: the executable hash is labelled `benchmark_executable`, and loaded kernel-library
  identity is explicitly `not_recorded`. A capture of the actual loaded library is still needed; nearby libraries
  or the benchmark executable cannot prove which kernel binary ran.
- C1/C2: removed BoltBeam's copy of block-quant installed launch geometry. Candidate generation remains independent
  of installed-route identity. A9/A10 remain blocked on the provider's actual model-binding metadata. The fork's
  `decode_emitter_space.installed_row` returns family defaults, and `FlashDecodeProvider.describe` calls
  `flash_decode_route_for` without model binding or wide-substrate selection. BoltBeam now refuses to call these
  the model's kernel. It accepts a shape/quant-scoped `model_binding.installed_row` only with `model_bound: true`;
  fusion's unfused side and flash require the same provenance assertion. The current fork does not supply it.
- A11: every installed flash dimension must be supplied; a missing dimension is refused, and a 32-lane installed
  tile remains 32 lanes even when the device subgroup is wider. Completing A9/A10 requires a fork metadata change
  and is deferred, rather than substituting another kernel.
- C3 and BoltBeam's C12: the type IDs live in `profile/gguf.py`, physical layouts in `profile/decode_roles.py`;
  capability data and adapters derive them. Tolerances import the owner in `collectors/metal_native.py`.
  The Q1_0 physical row is 128 elements / 18 bytes (two scale bytes plus 16 payload bytes), confirmed
  against the fork’s `tinygrad/llm/gguf.py` and generated `block_q1_0` struct. The former 32 / 4 row
  omitted scale storage; this was corrected in a separate functional follow-up after the NFC sweep.
  `sz.py` has no authority exceptions. Fork-side tables, legality rules and tolerance ownership remain outside scope.

## Records and boundary findings

| IDs | Disposition and evidence |
| --- | --- |
| R1 | Corrected `kernel-timer.md` and the timer docstring. Its own tables give 54.3/66.0 = 0.82 of the M3 token and 3.38/3.98 = 0.85 of the 5090 token; 3.38 ms is within 3% of the 3.29 ms in-model weight sum. |
| R2 | Corrected the 0.13 ms fusion claim to an estimate, not an upper bound or measured gain. The record's Table 1 has 1.09–1.30 isolated/in-model ratios and 16–30% spread on the small roles, one model at context 128. |
| R3 | Not fixed: fork `test/unit/test_block_quant_q5_k.py:30` still uses `rtol=2e-4` plus `atol=2e-3*max(abs(want))`; its result record omits the dominating absolute term. |
| R4 | The fail-open implication is stale: fork `model_route_plan.py` explicitly separates `measured_shapes` from `MEASURED_SHAPE_FIELDS` and rejects absent/unknown gate fields. The evidence-list name and repeated performance prose remain; fork record edits are deferred. |
| R5 | Not fixed: fork `decode_kernels.py` still says “exact NVIDIA decode FFN shape”, and `decode_routes.py` still prints the old flash shape in a refusal. |
| R6 | Not fixed: `flash_decode_attention.py` claims no production dependence on research modules while `tinygrad/llm/boltbeam_authority.py` re-exports from `extra/llm_research/boltbeam_authority.py`. |
| R7 | Corrected the supported-backend, BubbleBeam ownership, supported-format and interpreter-scan descriptions. |
| R8 | Not fixed: fork `block_quant_microgate.py` still calls max absolute error / max reference magnitude `max_rel_err` and supplies `max_threads=1024`. |
| E9 | Unknown block layouts now refuse the weight-ledger check, instead of accepting an understated limit. |
| E10 | Engine scan reports permission-skipped paths and propagates unexpected filesystem faults; missing paths remain ordinary. |
| E11 | TUI refuses an unknown `--role-time` before starting a job. |
| E12 | CUDA symbols match normalized complete kernel names; ambiguous template prefixes refuse. Missing warp-size metadata raises an error. |
| E13 | Pod setup requires a full model digest, downloads to a fresh staging file, verifies it before replacement, and no longer resumes arbitrary partial data. The default digest was checked against upstream `x-linked-etag` at model revision `7c41481f57cb95916b40956ab2f0b139b296d974`. |
| P1 | Not fixed: the fork's `decode_q8_0_gemv` route and disable switch still name one format for a multi-format implementation. |
| P2 | Not fixed: `fp16_fma` is type-validated but does not select the Q6 FFN emitter's computation. |
| P3 | Not fixed: fork `BlockQuantPrimitiveLinear._ggml_type` still stores a format object unlike sibling integer fields. |
| P4 | Not fixed: the fork retains separate repeated `load_*_promotion` bodies. |
| P5 | Not fixed: the flash provider imports BoltBeam's candidate schema through `BOLTBEAM_ROOT` without a revision pin. Resolving this interface cycle needs fork work. |
| P6 | Not fixed: fork `EmitterFamily.to_json` still branches on family names and imports constants. |
| P7 | Pod setup passes operator values as shell arguments, refuses a dirty checkout and checks out detached instead of resetting a `pod` branch. |
| P8 | Removed the dead second docstring; within-noise rows retain their signed difference so the loss column adds up. |

## Corrected shape-pins route counts

The fork's historical `docs/task_workflow/output/nv-shape-pins-to-records-result-20261010.md` says 5 admitted
routes for the 27B and 3 for the 4B. At the revision above, a read-only CPU metadata call to
`model._decode_route_promotions(config, ("NV", "sm_120"), getenv_fn=lambda *a, **k: 0)` gives **4 and 2**:

- 27B: `decode_q4k_epilogue_resadd`, `decode_q4k_w1w3_fusion`, `decode_q4k_w1w3_fp16_store`, `decode_q8_0_gemv`.
- 4B: `decode_q4k_epilogue_resadd`, `decode_q8_0_gemv`.

Calling every shape probe with `promoted=True` instead gives 5 and 3, because it also counts `decode_norm_fusion`;
that is not production admission. This correction is recorded here because the fork is read-only for this task.
