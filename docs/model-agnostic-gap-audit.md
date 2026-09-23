# Model-Agnostic Gap Audit (A0)

Goal: make BoltBeam handle new dense/MoE transformer GGUFs, quants, and targets
by *profile facts* instead of Qwen/AMD/Q4_K-specific assumptions.

This document is the A0 deliverable: every Qwen / AMD / Q4_K / Q6_K / dense-only
assumption in the four scanned files is listed with a **classification**, an
**owner**, and a **fix path** (the later A-phase that removes it). Scanned:
`profile/gguf.py`, `search/emit.py`, `data/candidates.json`, `analyze.py`
(plus the adjacent authorities they lean on: `profile/ir.py`, `targets.py`,
`policy/emit.py`, `synth/fixtures.py`, `vocab.py`).

Classification legend:

- **name-pattern** — behavior keyed off a tensor/model name string.
- **quant-family-specific** — behavior hardcoded per quant (Q4_K/Q6_K/…).
- **dense-only** — assumes a dense decoder; MoE/encoder-decoder unrepresentable.
- **target-specific** — behavior hardcoded for one target/backend (AMD/gfx1100).
- **candidate-manifest-specific** — a shipped model-era decision baked in as a
  universal candidate rather than data/ledger.

---

## profile/gguf.py

| # | Location | Assumption | Classification | Fix path |
|---|----------|-----------|----------------|----------|
| G1 | `role_from_tensor_name` L74–82 | Roles are the *grouped* Qwen set (`ffn_gate`/`ffn_up`→`ffn_gate_up`, `attn_q`/`attn_output`→`attn_qo`, `attn_k`/`attn_v`→`attn_kv`). q/o and k/v are collapsed; no fine-grained roles. | name-pattern, dense-only | **A1** role taxonomy IR (fine roles + derived grouped views) |
| G2 | `role_from_tensor_name` L74–82 | No MoE tensor names: `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` (expert stacks), `ffn_gate_inp` (router), shared-expert variants all fall through to `other`. | name-pattern, dense-only | **A1/A5** MoE roles + MoE search semantics |
| G3 | `profile_from_gguf` L96 | `len(dims) != 2` tensors are skipped. MoE expert weights are 3D `[n_expert, out, in]`, so **every expert weight is dropped** → a MoE model profiles as if it had no FFN. | dense-only | **A2/A5** architecture classifier + MoE role grouping |
| G4 | `profile_from_gguf` L98 | `embedding`, `norm`, `other` roles are dropped from `roles`. Fine for GEMV search, but means embedding/lm_head/norm are invisible to any future target that cares. | dense-only (scope) | **A1** taxonomy keeps them as first-class (derived views can still filter) |
| G5 | `GGML_TYPE_NAMES` L10–15 | The GGML type→name table lives here as a private dict; quant *behavior* (block size, packing, dequant family, route support) exists nowhere. | quant-family-specific | **A3** quant capability registry (this table moves into data) |
| G6 | `profile_from_gguf` L88–91 | Architecture is just the raw `general.architecture` string; there is no derived architecture *class* (dense vs MoE vs enc-dec vs unknown). | dense-only | **A2** architecture classifier |
| G7 | `profile_from_gguf` L92,101 | `vocab`/`lm_head` handling assumes a single tied output; encoder-decoder / multi-head outputs unmodeled. | dense-only | **A2** (unknown_transformer emits an honest incomplete profile, no crash) |

## search/emit.py

| # | Location | Assumption | Classification | Fix path |
|---|----------|-----------|----------------|----------|
| S1 | `_route_families` L10,20,30 | Primary mechanism is `if role.quant in ("Q4_K","Q5_K")` / `== "Q6_K"` / `in ("F16","BF16")`. Quant → route family is hardcoded code, not data. | quant-family-specific | **A3/A4** quant capability registry drives emission |
| S2 | `_route_families` L8 | `target` is a parameter but **never read** — target capability (wave size, dot support, LDS, backend status) does not shape the space. | target-specific (absent) | **A4/A6** target capability descriptors feed emission |
| S3 | `_route_families` fallthrough L36 | An unknown quant yields `families=[]` — a **silent omission**, not an explicit `unsupported_quant` / search-space-incomplete row. | quant-family-specific | **A3/A4** unknown quant → explicit blocked row |
| S4 | whole file | No MoE route families (expert GEMV/GEMM batching, router/top-k, expert layout, active-expert dispatch). A MoE model would emit dense FFN families over its (currently missing) expert roles. | dense-only | **A5** MoE search semantics |

## data/candidates.json

| # | Location | Assumption | Classification | Fix path |
|---|----------|-----------|----------------|----------|
| C1 | `decode_q4k_lanemap_gemv`, `decode_q4k_g3_anyshape`, `decode_q4k_g3_anyshape_attn_k` | Q4_K shipped-8B decisions encoded as universal candidates ("for the shipped 8B shapes"). Every new Q4_K profile inherits them whether or not they apply. | candidate-manifest-specific, quant-family-specific | **A7** route-family templates + ledger entries for shipped decisions |
| C2 | `decode_q6k_coop_shipped`, `decode_q6k_direct_halfwarp_refuted` | Q6_K shipped/refuted history baked into the manifest as universal truth (incl. the refuted axis). | candidate-manifest-specific, quant-family-specific | **A7** keep as ledger history, template the live family |
| C3 | grouped `roles` values (`attn_kv`, `attn_qo`, `ffn_gate_up`) | Candidates are written against the grouped Qwen roles only; fine-grained or MoE roles cannot be targeted. | dense-only | **A1/A7** roles reference the taxonomy; grouped views stay valid |
| C4 | `rollback` env-var names (`DECODE_Q4K_G3_ANYSHAPE`, `Q6K_PRIMITIVE`, `BUBBLEBEAM_FUTURESIGHT`, …) | Rollback knobs are tinygrad-AMD-specific env vars embedded in the model-agnostic manifest. | target-specific | **A7** (documented as target-scoped; templates carry target applicability) |

## analyze.py

| # | Location | Assumption | Classification | Fix path |
|---|----------|-----------|----------------|----------|
| Z1 | `_role_rank` L25 | `quant_rank` dict hardcodes Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16 priority; unknown quants sink to rank 9. | quant-family-specific | **A3/A4** priority derived from quant registry (byte-cost) |
| Z2 | `_role_rank` L26–32 | `role_rank` hardcodes grouped Qwen roles; MoE/fine roles get default rank 8. | name-pattern, dense-only | **A1** rank keyed off taxonomy role class |
| Z3 | `_candidate_families` L47–51 | **Duplicate** of `emit._route_families`' quant→family knowledge (Q4_K→lanemap, Q6_K→q6k_route, F16→matmul). Two sources of truth for the same rule. | quant-family-specific | **A4** single capability source; analyze reuses emit |
| Z4 | `_env_prefix` L67 | `DEV=AMD` hardcoded into every emitted command. | target-specific | **A6** command prefix derived from target backend |
| Z5 | `build_tinygrad_commands` L54–59,83,88–102 | Tool paths `qk_decode_*` (Qwen-decode-named tinygrad scripts) and default route flag `DECODE_Q4K_G3_ANYSHAPE=1` hardcoded. | name-pattern, target-specific | **A6/A7** target/route descriptors supply the flag; tools stay tinygrad-scoped data |
| Z6 | `emit_analysis_bundle` L215–216 | `tinygrad_root` default `/home/ubuntu/tinygrad-arkey` and `max_context=4608` (Qwen ctx) baked as defaults. | target-specific, name-pattern | **A6** (kept as caller-supplied; no host path in agnostic core) |
| Z7 | `build_measurement_plan` L117–118,155–162 | `route_focus` splits roles into `q4k_like_roles` / `q6k_roles`; summary counts `q4k_like_roles`/`q6k_roles`. | quant-family-specific | **A4** focus/summary keyed off registry route families, not named quants |

## Adjacent authorities (context, not in the four files)

| # | Location | Assumption | Classification | Fix path |
|---|----------|-----------|----------------|----------|
| V1 | `vocab.py` | Roles and quant families are **not** centralized — roles are free strings scattered across gguf/emit/analyze/candidates. Violates "Centralize authority". | name-pattern | **A1** role taxonomy in vocab; **A3** quant families in registry |
| T1 | `targets.py` L5–10 | `nvidia_sm89`/`apple_metal` are copies of AMD (`wave_size=32`, no LDS/dot/vector/backend-status data). No `descriptor_only` vs `complete` status. | target-specific | **A6** target capability expansion + backend status |
| P1 | `profile/ir.py` `TargetProfile` L27–33 | No capability fields (subgroup, vector width, LDS, dot/dequant primitives, backend status). | target-specific | **A6** |
| P2 | `policy/emit.py` L12–19 | Seed policy iterates `route_families` and lists `family` names; inherits every emit assumption (S1–S4) transitively. | quant/dense (transitive) | fixed once **A4/A5** land |

---

## Acceptance (A0)

Every Qwen / AMD / Q4_K / Q6_K / dense-only assumption in the four scanned files
is listed above with an owner and a fix path. The fix paths partition cleanly
onto the later phases:

- **A1** role taxonomy: G1, G2, G4, Z2, C3, V1
- **A2** architecture classifier: G3, G6, G7
- **A3** quant capability registry: G5, S1, S3, Z1, V1
- **A4** search emission from capabilities: S1, S2, S3, Z3, Z7, P2
- **A5** MoE search semantics: G2, G3, S4
- **A6** target capability expansion: S2, Z4, Z5, Z6, T1, P1
- **A7** candidate manifest generalization: C1, C2, C3, C4, Z5

Definition-of-done guardrail for every phase below: the existing 132 tests stay
green and the current Qwen-dense/Q4_K/Q6_K search space is byte-stable unless a
fixture is intentionally updated in the same commit.
