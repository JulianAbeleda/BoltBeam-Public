# Candidate identity model/target decoupling — CI-A/CI-B/CI-C closeout

Date: 2026-08-01. Repo: BoltBeam, branch `exp`. Scope:
`docs/candidate-identity-target-decoupling-scope-20260801.md` (revised 2026-08-01).

## What landed

- CI-A: `QWEN3_8B_NEUTRAL_ID = "qwen3_8b_q4k_m"` is derived from the profile's own
  fields plus the registry arch spelling (`_device_neutral_id` + `_arch_of`), never
  hand-written. `PROFILE_ALIASES` maps the neutral form to the canonical
  `qwen3_8b_q4k_m_gfx1100`; `resolve_profile_id` leaves every existing identifier
  unchanged.
- CI-B: `build_qwen3_8b_buffer2_candidate_set(seed, target_id="amd_gfx1100")`
  resolves the target through `load_target_registry` and fails closed on an
  unknown target. The seed gate is now model-level (the seed must denote the 8B
  Q4_K_M model in canonical or neutral spelling), not a constant equality check.
  Stamped identities come from the resolved pair: profile id is
  `{neutral}_{arch}`, workload target and applicability targets come from the
  registry row.
- CI-C: minted
  `bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-m4-10c-v1/candidate-set.json`
  in tinygrad for `apple_m4_10c` — four roles, profile `qwen3_8b_q4k_m_m4_10c`,
  target `Metal:m4_10c:wave32`, zero `gfx1100` occurrences in the artifact bytes.
- CI-D: the constant-equality gate was replaced (CI-B); no hand-written neutral
  literal, per-target branch, or dead helper was introduced. The route manifest's
  `profile_id` values are retained: they are hash-pinned by the detached SHA-256
  sidecar and byte-compared against the tinygrad snapshot by `mr12_static_audit`,
  so they are back-compat identifiers, not duplicates.

## CI-D deletion audit — what could not be removed, and why (2026-08-01, verified)

The two candidate deletions outside the builder were each probed and found load-bearing:

- tinygrad `extra/llm_research/model_profiles.py` `_PROFILE_ALIASES` legacy entries
  (`qwen3_8b_q4_k_m_gfx1100` / `qwen3_14b_q4_k_m_gfx1100`): pinned by
  `test/unit/test_model_profiles.py:46`, which asserts the underscore spelling resolves
  to the canonical profile (suite: 7 passed). Removing them would break a documented
  back-compat contract, so they stay.
- BoltBeam `boltbeam/policy/route_manifest.py` `PROFILE_DECODE` / `PROFILE_PREFILL`: consumed
  by `to_manifest_dict()` (`tests/test_boundary.py`,
  `tests/kernel_analysis/test_p9_assemble.py`, `tests/kernel_analysis/test_p9_bundle.py`),
  which p9 artifact bundles fingerprint; the manifest asset values they mirror are
  hash-pinned by `route_manifest.v1.json.sha256` and byte-compared by `mr12_static_audit`.
  Removing them would change exported bytes and invalidate the pinned asset.

Per the scope's CI-D method, these dependencies are the finding and are recorded here
rather than removed. Net lines: the decoupling commit removed 10 lines (the old gate and
stamping) against 98 added. One genuinely dead line was found and removed in both repos:
`PROFILE_DECODE_LARGE` (BoltBeam `route_manifest.py` and tinygrad
`extra/llm_research/route_manifest.py`) had zero references outside its definition and no
exported consumer; the full BoltBeam suite (1226 passed) and the tinygrad route-manifest
tests confirm the deletion changes nothing.

## Evidence (every number from a command actually run)

- Default output is byte-identical to the promoted AMD artifact: deterministic
  JSON sha256 `61b22f8797ece839a2c8a624dad2811437e6760c6e6ab2867b79c8aede8574e0`;
  `set_hash` `e9839825993c70876a45b43db2a19098e35218c3a4a4f2f53c9d359ee9133d43`;
  first entry identity `5585ac260bd84f780aee4a390a1e2318952197e8ef1b9054febd0e9c9e16a27b`.
- Metal mint: `set_hash`
  `f28071738ab95045cae85e36982177d5ac11dd83d2f462e409ee65cbad5bf747`, four
  entries, `canonical_identity` prefix `3dadb9ace8440743`/`96089dde39fb3e05`/
  `42793843577f8803`/`bcfb2d2add30d032`, `rg -c gfx1100` = 0.
- BoltBeam tests: 73 passed on `exp` (candidate-set, full-kernel emit,
  search-spec, targets, policy-guard, promote-routes suites).
- tinygrad consumer: `load_candidate_payloads` + `find_role_template` load both
  the AMD artifact (profile `qwen3_8b_q4k_m_gfx1100`) and the Metal mint (profile
  `qwen3_8b_q4k_m_m4_10c`) unchanged.

## Not done here (scope says so)

- Hardware qualification of the Metal mint (M1e lane is the consumer).
- tinygrad-side seed constants (`packed_wmma_production_canary.py` and friends).
