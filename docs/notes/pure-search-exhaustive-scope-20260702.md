# Pure Machine Search — Exhaustive Scope

The north star: the machine search generates **every kernel of every model** on a target, by composing a
**complete** primitive vocabulary. No hand-written kernels — only vocab and codegen substrate.

## Philosophy (read first)

- **The goal is learning, so the scope is exhaustive.** Everything is eventually in scope. Nothing is "closed"
  or "refuted." Items are only ordered by *value* and *effort*. A low-value item is *deferred*, never deleted.
- **Every dead-end is a retained learning, not a closure.** When a lever doesn't pay, we keep the precise
  mechanistic *why* (`nontranslation_reason`) as an asset. Understanding why something is low-value is itself
  coverage of the space.
- **Capture-driven, measurement-gated.** Use the empirical vocab capture to see what the real codegen emits;
  build the primitive *system* that changes it; re-capture to confirm it is actually used and the number moves.
  Never declare a win from a kernel-level number.
- **A primitive is a system, not an instruction.** A dot without its operand-prep (pack/convert/fuse) and its
  scheduling (occupancy/register pressure) washes. Scope each primitive with its supporting vocab.

Exhaustiveness has three axes: **Vocabulary** (every primitive), **Coverage** (every kernel of the forward),
**Generality** (every arch / quant / target). All three must eventually be complete.

---

## Axis 1 — The complete primitive vocabulary

Status: `lowered` (substrate can generate) · `used` (default route emits it) · `backlog` (no lowering yet) ·
`learned` (explored, precise why retained). Value = near-term roofline leverage. Learning = coverage value.

### Dots (VOP3P family)
| primitive | status | value | learning | note |
|---|---|---|---|---|
| v_dot2_f32_f16 | lowered, **unused** | low | done | MAC not the bottleneck → Amdahl-tiny (+0.6%); vgpr↑ occupancy cost |
| v_dot4_i32_i8 | lowered, unused | **high** | — | integer dot avoids the convert bottleneck; −10% at kernel level |
| v_dot8_i32_i4 | backlog | **high** | — | 8×4-bit dot = Q4_K nibble width exactly; needs i4 packing vocab |
| v_dot4_u32_u8 | backlog | low | — | unsigned variant; coverage |
| v_dot8_u32_u4 | backlog | low | — | unsigned 8×4-bit; coverage |
| v_dot2_f32_bf16 | backlog | low | — | no bf16 activations in qwen3; coverage/other models |
| v_dot2_i32_i16 / u32_u16 | backlog | low | — | niche int16 dots; coverage |

### Dequant
| dequant_bitunpack | lowered, used | — | done | v_bfe/v_alignbit; inherent to Q4_K/Q6_K |
| dequant_convert | lowered, used | — | **key learning** | **THE decode bottleneck**; irreducible in float (int→float has no cheaper form); only avoided by moving to integer dot |

### Reduce / norm
| reduce | lowered, used | — | done | native loop; unfused-reduce overhead is *scheduling*, not a missing primitive |
| fused_reduce_scale | backlog | low (~2.8%) | done | RMSNorm/qk_norm single-load reduce+scale; needs a dedicated fused-LOAD kernel |

### Cross-lane
| ds_bpermute (xlane_reduce) | lowered, used | — | done | the combine/reduce primitive (M5) |
| ds_permute / v_permlane / ds_swizzle | backlog | low | — | alternative cross-lane topologies; coverage |

### Memory
| global_load | lowered, used | — | done | vectorizable loads |
| lds_stage | lowered, used | — | done | LDS + barrier |
| coalesced_int8x4_load | backlog | med | — | packed int8×4 load so int-dot operands arrive packed (no pack-from-scalar ALU) |
| vec_store_to_reg | backlog | **high** | — | **vectorized-V PV primitive** — flash-attention decisive-win blocker; the last thing keeping a hand-written attention kernel; UOp-spec walled |

### Math
| elementwise | lowered, used | — | done | add/mul/sub/select |
| transcendental | lowered, used | — | done | exp/sigmoid/sin/cos/rcp/rsqrt |

### Fusion / scheduling
| scheduler_fuse_elementwise_into_gemv | lowered, demonstrated | high | done | fuses a Tensor-op quantize into a generated GEMV; the blocker was custom_kernel opacity |
| custom_kernel_fusion (meta) | backlog | — | **key learning** | custom_kernels are opaque fusion boundaries; the fix is to express kernels as Tensor ops so the scheduler fuses natively, not to make custom_kernels fusable |
| pm_intdot_tiled_match | backlog | med | — | harden pm_intdot to emit v_dot4 on a real tiled/derived-operand reduction (round/select→char→int), not just the clean 4-term direct-load idiom |

### Coordination (high-effort, high-learning — NOT closed)
| cross_workgroup_atomic_float | backlog | low-now / **high-learning** | partial | would unlock the ~13.6% attention-combine bucket; genuinely-new coordination primitive; high effort |
| grid_sync | backlog | low-now / high-learning | partial | cross-workgroup barrier; same bucket |

### Quant schemes (learning-rich, deferred)
| sub_4bit (Q3/Q2 weights) | deferred | low-now | done (naive) | naive per-role demotion fails dNLL; **learning open**: importance-aware / mixed sub-4-bit is unexplored and eventually in scope |
| f16 dequant | learned | none | **done** | strictly worse: int→f16 doubles converts. Retained as a hardware rule. |

### Other-arch primitives (Generality axis)
| ssm_scan | backlog | — (hybrid) | — | selective-scan for qwen3.5-hybrid/mamba |
| ssm_conv | backlog | — (hybrid) | — | short causal conv1d for SSM blocks |
| moe_expert_batched_gemv / router / dispatch | backlog | — (MoE) | — | for MoE arches |

---

## Axis 2 — Coverage: every kernel of the forward

Each kernel class must eventually be **generated** (not custom_kernel) and **optimal**. Current state:

**Decode:**
- Q4_K GEMV (gate/up 12288/17408/25600, q/o, kv 1024) — generated (G3) but a **custom_kernel**, scalar FMA, dot-free. → int-dot-as-Tensor-ops adoption.
- Q6_K GEMV (ffn_down long-K, lm_head) — generated coop; shipped.
- Flash-decode attention — generated near-parity (98.3–98.8%), **hand-written oracle still exists** → PV primitive.
- RMSNorm / qk_norm — generated; unfused-reduce overhead → fused_reduce_scale (low value).
- RoPE — generated (transcendental + rotate).
- SiLU/SwiGLU — generated.
- Residual adds — generated (elementwise).
- KV cache read/write — generated.
- Sampling/argmax — generated.

**Prefill:**
- Dense GEMM (gate/up/down/qkv/o) — generated via TC/WMMA schedule; ~parity+.
- Attention prefill (SDPA/flash) — generated.
- lm_head — Q6_K GEMV.

The two non-generated / sub-optimal spots: **the Q4_K decode GEMV** (custom_kernel, dot-free) and **flash-decode attention PV** (hand-written oracle). Everything else is generated.

---

## Axis 3 — Generality: every arch / quant / target

- **qwen3 dense** 0.6B / 1.7B / 4B / 8B / 14B / 32B — same primitive vocabulary, dim-scaled. (0.6B/1.7B are Q8_0 → exercise the q8 dequant path.)
- **qwen3.5-27B hybrid** — SSM + attention → needs ssm_scan / ssm_conv (new vocabulary; the declarative extractor's decoder-op table doesn't cover it yet — empirical capture does).
- **MoE arches** — expert batched-GEMV / router / dispatch families (BoltBeam already models these).
- **Quants** — Q4_K, Q6_K (covered), Q8_0 (partial), Q5_K / Q3_K / Q2_K (coverage TODO).
- **Targets** — amd_gfx1100 (primary, complete); nvidia_sm89 / apple_metal (descriptor-only; capability data present, no evaluator). Eventual: the dot family maps differently per target (mma/dp4a on nvidia, simdgroup on apple).

---

## The composition systems (primitives → kernels)

- **Int-dot GEMV system** (the roofline system): x→q8_1 quantize · coalesced int8×4 load · integer dot (v_dot4 / v_dot8_i4) · per-group scale · fuse-into-generated-GEMV · shape-selective routing. Components: quantize-fuse ✓ demonstrated, v_dot4 ✓ lowered, pm_intdot_tiled_match ○, coalesced-load ○, routing ○, adoption ○.
- **Attention system**: flash split (Hq×S) · score dot (v_dot2 ✓) · online softmax (transcendental ✓) · **PV (vec_store_to_reg ○)** · combine (cross-lane ✓ / atomics ○). Blocker: PV.
- **Norm system**: fused reduce+scale (○, low value).

---

## Priority tiers (value ordering — but everything is eventually done)

- **Tier 1 — Roofline: adopt the integer-dot GEMV.** Harden pm_intdot → express the production Q4_K GEMV as fused Tensor ops → re-capture (confirm v_dot4 in ISA, convert gone) → W==D. Closes the adoption gap *and* attacks the convert bottleneck. Only path with real ~2× headroom.
- **Tier 2 — Last hand-written kernel: attention PV (`vec_store_to_reg`).** Unblocks the generated-attention decisive win.
- **Tier 3 — Completeness: the rest of the dot family** (v_dot8_i4 first), fused_reduce_scale, cross-lane alternatives, coalesced-load, shape-selective routing.
- **Tier 4 — Generality:** SSM (hybrid), MoE, remaining quants (Q5/Q3/Q2_K, Q8_0), other targets.
- **Tier 5 — High-effort / high-learning:** cross-workgroup atomics + grid-sync (attention combine); importance-aware sub-4-bit schemes. Low near-term value, high learning — genuinely in scope, not closed.

Every tier is on the exhaustive path. Tier ordering is value/effort, not open/closed.

---

## The method (the loop)

1. **Capture** the target model's real scheduled ISA (`boltbeam vocab-capture`), find the dominant instruction class.
2. **Extract** the model's required vocabulary (`boltbeam vocab`), diff against the registry → used / lowered-but-unused / backlog.
3. **Build** the smallest primitive *system* (vocab + substrate, no kernels) that changes the dominant class.
4. **Re-capture + measure** (W==D, NLL) to confirm the primitive is actually used and the number moves.
5. **Record the why** either way — a win or a retained learning. Both are coverage.

## The learning ledger (retained "why"s — assets, not closures)

- v_dot2 MAC → Amdahl-tiny (MAC is ~1/3 of a dequant-dominated kernel; +vgpr occupancy cost).
- f16 dequant → int→f16 doubles converts (hardware rule).
- int-dot naive whole-route → washed by un-fused quantize + wrong-shape routing (vocab gap, not the primitive).
- attention combine fusion → sacrifices Hq×S occupancy (needs a *new* coordination primitive, not a topology tweak).
- sub-4-bit naive → dNLL fails for high-byte roles (better schemes unexplored).
- convert bottleneck → irreducible in float; only avoided by integer dot (the unifying insight).

---

## Goal: fully decouple upstream BEAM (2026-07-02)

**Upstream tinygrad `BEAM` is scaffolding, not the plan — decouple from it entirely.** It was only ever a
convenient off-the-shelf way to *search* a schedule and prove one exists. It is NOT the in-house path and must
not be a runtime dependency.

**Why the BEAM-based "proof" is misleading.** The result "int-dot GEMV → 0.19ms with BEAM = matches G3" reads
like a win, but it is misleading: it depends on upstream BEAM, which (a) is an external timing search we do not
own, (b) hangs on gfx1100 for the whole-model forward (the int-dot kernel re-search dead-locks the GPU), and
(c) isn't how the fast kernels actually work. The correct reading is the *other* number in that experiment:
**G3 hits 0.18ms with `opts_to_apply=()` — zero BEAM.** Its speed is STRUCTURAL (the lane-partition thread
map), not searched. So schedule quality does not require a runtime search; it can be predicted and applied.

**The in-house path is BubbleBeam + Futuresight.** Futuresight already IS the anti-BEAM: static schedule
prediction — it ranks candidate thread maps by a layout predicate (unit-stride packed-word access) *before*
timing, to make the hand-proven lane partition discoverable without search. The plan is to **extend
BubbleBeam/Futuresight's static schedule prediction to cover the int-dot GEMV** so it is fast by construction
(G3-style structural efficiency, `opts_to_apply=()`), never invoking upstream BEAM.

**Consequence:** the occupancy-starved diagnosis (schedule-trace) is correct and stands, but its FIX is
"Futuresight predicts the tiling/lane-partition," not "run BEAM." Decoupling upstream BEAM also removes the
gfx1100 BEAM-hang blocker, because we never call it. Any candidate/doc that cites BEAM as the fix is to be
re-read as "needs BubbleBeam+Futuresight scheduling."
