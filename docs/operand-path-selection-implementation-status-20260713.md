# Operand-path selection implementation status

Date: 2026-07-13

Scope authority: `docs/operand-path-selection-e2e-scope-20260713.md`

Implementation continuation: `docs/qwen3-8b-p9-completion-handoff-20260713.md`

## Outcome

> **CLOSED 2026-07-31 — P9 did not materialise.**
>
> The blockers named below were never resolved: the shipping final-ISA semantic
> map and the exact decode kernel input/reference authority were not produced,
> and `p9_complete` was never set. No further work happened after 2026-07-14.
>
> The P9 assembly code (`kernel_analysis/p9_assemble.py`, `p9_bundle.py` and
> their tests, 734 lines) has been pruned from the trunk on that basis. It
> remains on `dev` and `exp` per `docs/branch-flow.md`; nothing is destroyed.
>
> This note is the verdict record that authorises the prune. The line of work is
> recorded as *attempted and not completed*, which is a real result: P0–P8 are
> implemented and contract-tested, and the operand-path work stopped one gate
> short. Anyone resuming it starts from this document and the scope it cites,
> and recovers the code from `dev`.

P0-P8 are implemented and contract-tested across BoltBeam and the tinygrad provider. P9 is intentionally incomplete:
the current 8B routes preserve correctness and throughput, but the shipping AMD/HIP binaries do not yet carry a
source-to-final-instruction semantic operand map. Decode also lacks an immutable packed-weight, activation, and
reference-output artifact for isolated kernel-level correctness. The P9 bundle therefore remains fail-closed and
cannot claim `p9_complete`.

## Real-hardware proof

Target: RX 7900 XTX / gfx1100. Model: `Qwen3-8B-Q4_K_M.gguf`.

- Promoted prefill `attn_qo` `(512,4096,4096)`: exact shipping AMD binary
  `ad3a947d218302443523eb03b622302b2f7c90706fcf83b1a59666a61e17cf3e`.
- Full guarded comparison: 2,097,152 fp16 outputs, `max_abs_error=0`, no NaN/Inf, inputs unchanged, guards intact,
  preflight/postflight health passed.
- Final code-object resources: 188 compiler-reported VGPRs, 248 descriptor-allocated VGPRs, 18 SGPRs, 40,960 bytes
  LDS, zero scratch, zero VGPR/SGPR spills, workgroup `(32,4,2)`.
- Final disassembly: 574 instructions, 8 global loads, 48 DS loads, 8 DS stores, 29 waits, 2 barriers, 32 WMMAs.
- Current model smoke after normal warmup: prefill pp512 `3860 tok/s`; decode ctx512 `116.9 tok/s`. These agree with
  the recent authorities (`3881` and `117.1 tok/s`) and show no throughput regression.
- GPU was idle and healthy after all checks.

An ISA-only `AMD:ISA:gfx1100` compile was explicitly rejected as production evidence: it produced a different binary
from the shipping `AMD` compiler and failed numerical correctness. The adapter now compiles the exact shipping target
on both sides of the parent/child binary hash contract.

## Acceptance audit

| Acceptance item | Status | Evidence or blocker |
|---|---|---|
| Exclusive strategy model and compatibility | pass | Contradictory legacy declarations fail closed; MALL is not a transport. |
| Orthogonal serving tiers | pass | Register/scratch/LDS/L0/L1/L2/LLC/DRAM/host/unknown vocabulary is normalized. |
| Resources and final ISA join by binary | pass | Shipping code-object hash binds descriptor, metadata, disassembly, execution, and correctness. |
| Semantic instruction attribution or explicit unknown | pass | AMD/HIP rows remain unknown; no route-name or alternate-binary inference is admitted. |
| Static classifier fixture matrix | pass | Register/LDS/cache/reload/ambiguity/spill fixtures pass. |
| External isolated hierarchy sweep | partial | Typed injected provider and isolation contract pass; a fresh real gfx1100 sweep was not run in this change. |
| Counters supported/unsupported typing | pass | gfx1100 blocked PMCs remain typed unsupported and do not become zero. |
| Dynamic attribution respects scope | pass | Kernel-wide/proxy evidence cannot silently become per-operand evidence. |
| Profiler operand rendering | pass | Static, dynamic, classification, confidence, truth, and missing evidence are rendered. |
| Provider/model-agnostic candidate matrix | pass | Core feasibility and candidate construction have no tinygrad/model branches. |
| Prediction cannot promote | pass | Only correctness-eligible comparable measured matrices select. |
| Concrete external runner loop | pass | BoltBeam emits a stdin/stdout provider command; tinygrad lazily registers the exact adapter. |
| Full correctness before timing | pass | Guarded worker blocks timing on any execution/correctness failure. |
| Same-session randomized ranking | pass | Session coordinator and selector retain raw samples/order and return typed non-decisions. |
| Selection/mechanism confidence split | pass | Missing PMCs reduce mechanism confidence without blocking timing selection. |
| Reversible ledgered handoff | pass | Recommendation context and append-only ledger are implemented. |
| Current 8B prefill and decode complete P9 | blocked | Shipping final-ISA semantic map and exact decode kernel input/reference authority are missing. |
| Second-provider neutrality | pass | CUDA/SASS fixture traverses the same core rules. |
| Full suites and structural goldens | pass | BoltBeam and tinygrad suites pass; P9 golden is explicitly synthetic/non-production. |

## Remaining bounded work to P9

1. Emit or derive compiler-owned semantic ownership for the shipping AMD/HIP final instructions, joined to that exact
   code object. A/B global-load to LDS-stage to WMMA flow must be attributable or remain unknown per row.
2. Produce an immutable decode artifact containing the exact packed Q4_K model bytes, fp16 activation, and float32
   reference output; then run it through the same guarded isolated worker.
3. Build the production prefill/decode operand reports, use a candidate matrix or typed infeasibility for each important
   role, and join the existing raw/practical rooflines and model benchmark authorities.
4. Only after those gates pass may the P9 bundle set `p9_complete=true`. gfx1100 cache-tier PMCs can remain unsupported;
   they block mechanism specificity, not a measured winner.
