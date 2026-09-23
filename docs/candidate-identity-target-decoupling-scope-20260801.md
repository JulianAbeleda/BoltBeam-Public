# Candidate identity: model/target decoupling — scope

Date: 2026-08-01

Repo: BoltBeam, branch `exp`. Status: CI-A/CI-B/CI-C implemented 2026-08-01
(`c5c529d`), CI-D outcome recorded in §8. Does not authorize promotion to `main`.

---

## 1. The defect in one sentence

**Two designs for the same concept coexist in this repository.** The analysis, trace, and quantization
paths model a workload as `(ModelProfile, TargetProfile)` — two peer types. The candidate and
qualification path models it as **one fused string**, `"qwen3_8b_q4k_m_gfx1100"`.

The fused string is the older design. It is the one that gates qualification, and it is why four of the
five registered targets cannot qualify anything.

---

## 2. Evidence

### 2.1 The correct design already exists and is already used

`boltbeam/profile/ir.py` defines `ModelProfile` and `TargetProfile` as separate types. Consumers take
them as separate parameters:

```python
boltbeam/analyze.py:129        def build_measurement_plan(profile:ModelProfile, target:TargetProfile, ...)
boltbeam/analyze.py:233        def emit_analysis_bundle(profile:ModelProfile, target:TargetProfile, ...)
boltbeam/trace/timing.py:21    def build_trace_request(profile:ModelProfile, target:TargetProfile, ...)
boltbeam/quantization/quant_gemv.py:8   imports both
```

And a real target registry backs it — `boltbeam/data/targets.json`, schema
`boltbeam.target_registry.v1`, loaded by `boltbeam/target/targets.py::load_target_registry`:

| target_id | backend | wave | backend_status |
| --- | --- | ---: | --- |
| `amd_gfx1100` | AMD | 32 | **complete** |
| `nvidia_sm89` | CUDA | 32 | descriptor_only |
| `nvidia_sm120` | CUDA | 32 | descriptor_only |
| `apple_metal` | Metal | 32 | descriptor_only |
| `apple_m4_10c` | Metal | 32 | descriptor_only |

`TargetProfile` carries `wave_size`, `lds_bytes_per_cu`, `vram_bytes`, `subgroup_size`,
`vector_load_bits`, `dot_primitives`, `dequant_primitives`, `compute_units`, `memory_bandwidth_gbs`,
`peak_tflops`, `backend_status`, `capabilities`. **This is the right abstraction and it is already
declared.**

### 2.2 The candidate path ignores it

`boltbeam/full_kernel_candidate_set.py`:

```python
:11    QWEN3_8B_PROFILE = "qwen3_8b_q4k_m_gfx1100"
:133   if seed_workload["profile"] != QWEN3_8B_PROFILE:
:134     raise ValueError(f"8B candidate-set seed profile must be {QWEN3_8B_PROFILE}")
:143   workload["profile"], workload["role"] = QWEN3_8B_PROFILE, role
:146   payload["applicability"] = {"exact_shape":True, "profiles":[QWEN3_8B_PROFILE], "roles":[role], ...}
```

It does not default to gfx1100 — **it raises on anything else**, and it stamps the fused string into
`applicability.profiles`, so every candidate it emits is target-locked by construction.

### 2.3 The pattern is narrow — which makes this tractable

Exhaustive grep for the fused `model_quant_target` identifier shape across `boltbeam/`:

```
"qwen3_8b_q4k_m_gfx1100"
"qwen3_8b_q4_k_m_gfx1100_prefill"
"qwen3_8b_q4_k_m_gfx1100_decode"
"qwen3_14b_32b_q4_k_m_gfx1100_decode"
```

**Four identifiers.** This is not a diffuse refactor; it is a small, well-bounded surface using the
wrong representation.

### 2.4 What the function actually needs

`build_qwen3_8b_buffer2_candidate_set` uses exactly two things beyond the seed:

```python
QWEN3_8B_ROLE_SHAPES = (("attn_kv",512,1024,4096), ("attn_qo",512,4096,4096), ...)
if (pipeline["buffer_count"], pipeline["stage_count"]) != (2,1): raise
```

Role shapes and a pipeline-structure check. **Neither is a GPU fact.** An 8B Q4_K_M model has
`attn_qo` at `(512,4096,4096)` on gfx1100, on an M4, and on an H100 alike. The `gfx1100` in the
identifier carries no information the function consumes — it is a string equality check against the
machine the code was written on.

**The test:** if removing a target's name from an identifier loses no information, it was never
identity. This fails that test.

---

## 3. What it blocks

- **Qualification for every non-AMD target.** In the seven-step lifecycle
  (SEARCH → QUALIFY → PROMOTE → POLICY → ADMIT → LOWER → EXECUTE), QUALIFY needs a seed candidate. The
  generator raises for any profile that is not gfx1100.
- **Four of five registered targets.** All the CUDA and Metal entries are `descriptor_only`, and this
  gate is one reason they cannot advance.
- **Concretely, today:** the tinygrad-side Metal precontract kernel is *correct and measured*
  (`max_abs_error` 0.0, 3.4× over control, 87/87 candidates correct in a BoltBeam campaign) and cannot
  be promoted, because nothing can mint it a qualification artifact.

---

## 4. Target design — use the types that already exist

**Do not invent an abstraction. Adopt the one in `profile/ir.py`.**

- **Model identity** is `(family, size, quant)` — the thing that determines role shapes. It is
  target-independent.
- **Target identity** is a `target_id` resolvable in the registry. It carries the declared facts.
- **A candidate's applicability** is a pair, not a concatenation.

The precedent is exact and already in this codebase: `build_measurement_plan(profile, target, ...)`.
The candidate path should look the same.

**Backward compatibility is a hard requirement.** `amd_gfx1100` promoted artifacts, their identities,
and their hashes must remain valid and resolvable. This is a representation change, not a re-promotion.

---

## 5. Work packages

**Revised 2026-08-01 after finding prior art.** An earlier draft of this section specified an
exhaustive census, a hash-coverage investigation, and an additive-vs-breaking migration decision. All
three are unnecessary: **tinygrad solved this exact problem on 2026-07-31 (`c9e3b9bd1`, M1b), and the
solution is additive by construction.** The packages below adopt it.

### The prior art — copy this shape

`extra/llm_research/model_profiles.py:100-121` in tinygrad-arkey-exp:

```python
def _device_neutral_id(profile: ModelProfile) -> str:
  """Strip the trailing device-profile label a model profile's `id` happens to carry.
  A `ModelProfile` describes the MODEL (family/size/quant/roles/attention) -- `device_profile`
  is a caller-supplied label, not a fact this profile derives anything from ... The `_gfx1100`
  suffix on today's ids is therefore a naming artifact, not part of the model's identity."""
  suffix = f"_{profile.device_profile}"
  ...

_PROFILE_ALIASES = {
  **{_device_neutral_id(profile): profile.id for profile in MODEL_PROFILES},
  ...
}
```

**Aliases, not renames.** The neutral form is *derived from each profile's own fields* — not
hand-written per profile, which is what keeps it from drifting out of sync with `id`. Existing ids keep
resolving unchanged.

**Why this removes the hard parts:**

| earlier concern | resolved by |
| --- | --- |
| does a promoted hash cover the fused string? | **irrelevant** — `qwen3_8b_q4k_m_gfx1100` is untouched |
| additive or breaking migration? | **additive**, by construction |
| exhaustive census for breakage | **nothing existing changes**, so nothing to census for |
| cross-repo coordination with tinygrad | tinygrad reads the same strings it always did |

### CI-A — Derive and alias the neutral profile form

Add the `_device_neutral_id` equivalent. Derive the neutral id from the profile's own fields; register
it as an alias to the existing id. **Do not hand-write a second literal per profile** — that is the
drift the tinygrad docstring calls out.

Acceptance: every existing identifier still resolves to exactly what it resolves to today, and the
neutral form resolves to the same value.

### CI-B — Let the seed generator take a target

`build_qwen3_8b_buffer2_candidate_set(seed, target_id="amd_gfx1100")` — a parameter with a default
preserving today's behaviour byte-for-byte. Resolve the target through `load_target_registry`
(`boltbeam/target/targets.py`) rather than a module constant, and stamp `applicability.profiles` from
the resolved pair instead of `QWEN3_8B_PROFILE`.

Fail closed on an unresolvable `target_id`; do not silently fall back to gfx1100.

Acceptance: called with no target argument, output is **byte-identical** to today. Called with
`apple_m4_10c`, it produces a valid candidate set with no `gfx1100` in any identity.

### CI-C — Prove a second target can qualify

Unchanged from the earlier draft, and the reason to do any of this. Mint a seed candidate set for a
`descriptor_only` target — Metal is the immediate consumer, since a correct, measured kernel
(`max_abs_error` 0.0, 3.4x over control, 87/87 candidates correct) is waiting on it — and take a
candidate to QUALIFY without a gfx1100 string anywhere in its identity.

**CI-A and CI-B are refactoring. CI-C is the deliverable.**

### CI-D — Delete what the simplification made dead

**A simplification that only adds is not a simplification.** Once CI-A/CI-B land, the older
representation has consumers that exist solely to work around it. Remove them in the same arc, not
"later":

Candidates for deletion, each to be confirmed dead by CI-A/CI-B rather than assumed:

- **`QWEN3_8B_PROFILE` as a gate.** After CI-B, `full_kernel_candidate_set.py:133-134`'s equality
  check and raise have no reason to exist — the target is a parameter, not a constant to match against.
  The constant may survive as a default value; the *check* should not.
- **Any hand-written second literal per profile.** If CI-A is implemented correctly the neutral form is
  derived, so any literal spelling of it is a duplicate that can drift. tinygrad's
  `_PROFILE_ALIASES` keeps two hand-written legacy spellings (`qwen3_8b_q4_k_m_gfx1100`,
  `qwen3_14b_q4_k_m_gfx1100`) for underscore variants — check whether BoltBeam's equivalents are
  genuinely needed for back-compat or are just duplicates.
- **Per-target branches or special cases** introduced anywhere to work around the fused string.
- **Dead helpers** whose only caller was the gate being removed.

Method — do not delete on inspection:

1. Land CI-A/CI-B.
2. For each candidate, remove it and prove nothing depends on it: the AMD identity/hash controls
   unchanged, the cross-repo consumer still loads, the test suite's failing-id **set** unchanged.
3. If something does depend on it, that dependency is the finding — record it rather than keeping the
   code silently.

This repo already has the discipline: `1017135a4 [refactor] delete dead resident_fp16_admit` and
`b149e77`/`5c64ade`/`fdb2844`'s artifact pruning are the precedent, and `be2327a` removed 18,516 lines
in one move. **Deletion is a first-class deliverable here, not cleanup.**

**Acceptance for CI-D:** net lines removed is negative, or an explicit statement of what could not be
removed and why. A decoupling that leaves both representations standing has doubled the surface rather
than simplified it.

## 6. Evidence contract

1. **AMD non-regression is mandatory.** Promoted identities, hashes, and route manifests unchanged; the
   cross-repo consumer still loads. Any movement stops work and is reported with the diff.
2. **Diff test-id sets, never counts** — and state which branch, given the trunk/dev split.
3. **One concern per commit**, with the NFC claim stated where it applies (this repo's own convention,
   per its recent `[nn] NFC -` commits).
4. Every number from a command actually run.
5. **Coordinate the cross-repo change.** tinygrad consumes these artifacts; a representation change is
   a two-repo commit pair, not a BoltBeam-local refactor.

---

## 7. Non-goals

- **Not** re-promoting or re-qualifying any AMD candidate. Existing results stand.
- **Not** changing candidate *schema* semantics, `candidate_hash` derivation, or the v1/v2 split.
- **Not** the tinygrad-side seed constants (`packed_wmma_production_canary.py:16` and friends) — those
  are a coordinated follow-on, tracked in CI0 item 4.
- **Not** a new registry, env var, or control plane. §4 uses what exists.
- **Not** the remaining `gfx1100` references in ROCm-specific tooling (`*_rocprof.py`) — those are
  legitimately AMD-scoped and are not part of this defect.

---

## 8. Known limitations

- **No AMD or NVIDIA hardware here.** AMD non-regression is artifact-identity and load-compatibility
  only, not execution.
- CI-D retention, 2026-08-01: the remaining fused-spelling lines are load-bearing, not dead.
  tinygrad's two legacy alias entries in `_PROFILE_ALIASES` are pinned by
  `test/unit/test_model_profiles.py:46` (the underscore spelling must resolve to the canonical
  profile); BoltBeam's `route_manifest.py` profile constants feed `to_manifest_dict()`, which
  p9 artifact bundles fingerprint and `mr12_static_audit` byte-compares against the tinygrad
  snapshot, and the manifest asset itself is hash-pinned by its detached SHA-256 sidecar.
  Deleting any of them would break the evidence contract, so they stay as recorded findings.
- `backend_status: descriptor_only` on four of five targets means this scope removes *one* gate. It does
  not establish that those targets can complete the remaining lifecycle steps.
