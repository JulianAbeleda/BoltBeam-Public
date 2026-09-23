# BoltBeam Audit Brain Build Scope

Status: implementation scope for Claude.

## Goal

Move the relevant audit brain out of tinygrad and into BoltBeam without moving
GPU execution out of tinygrad.

BoltBeam should become the place that knows:

- what model/profile is being evaluated;
- what route families are legal;
- what measurements are required;
- how tinygrad result artifacts are normalized;
- whether a candidate is promoted, refuted, deferred, or search-space-incomplete;
- which evidence and rollback policy support that decision.

tinygrad should remain the place that:

- loads models;
- compiles/runs kernels;
- captures GPU timings and counters;
- emits raw benchmark/profile artifacts;
- owns runtime route implementation.

The desired split is:

```text
BoltBeam:
  profile -> search space -> measurement plan -> artifact ingestion
  -> normalized evidence -> evaluator -> ledger/policy/report

tinygrad:
  generated commands -> model execution -> GPU profiling
  -> raw artifacts for BoltBeam to ingest
```

This is a decoupling project. Do not copy every tinygrad one-off phase tool into
BoltBeam. Preserve tinygrad as an executor and make BoltBeam the durable audit
and decision layer.

## Source Citations

This scope is not standalone opinion. It is grounded in the current tinygrad
audit artifacts and the BoltBeam principles docs below. Claude should read these
before implementing, and every imported historical ledger row must cite one of
these or a more specific successor artifact.

### Principle And Boundary Sources

| claim | citation |
|---|---|
| Core coding principles: centralize authority, modularize execution, encode invariants, avoid re-sprawl | `/home/ubuntu/tinygrad-arkey/structure/Development/coding-principles.md`; BoltBeam copy: `docs/coding-principles.md` |
| Performance primitives must be measured as whole primitives, not isolated wins | `/home/ubuntu/tinygrad-arkey/structure/Development/performance-primitive-research-principles.md` |
| Current active-work state and agnostic-search direction | `/home/ubuntu/tinygrad-arkey/docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md` |
| BoltBeam currently emits profile/search/policy/measurement handoff only | `docs/analyze-command-scope.md`; `boltbeam/analyze.py` |

### Current Result Sources To Preserve

| result | citation |
|---|---|
| Q4_K G3 generated route is speed-equivalent and promoted | `/home/ubuntu/tinygrad-arkey/bench/amd-isa-backend-g3-weight-promotion/latest.json`; `/home/ubuntu/tinygrad-arkey/bench/qk-candidate-evaluator/decode_q4k_g3_generated/ledger_update.json`; `/home/ubuntu/tinygrad-arkey/docs/amd-isa-g3-weight-promotion-hardening-scope-20260629.md` |
| Q6_K direct half-warp route is token-correct but speed-refuted | `/home/ubuntu/tinygrad-arkey/bench/amd-isa-backend-q6k-direct-speed/latest.json`; `/home/ubuntu/tinygrad-arkey/bench/qk-candidate-evaluator/decode_q6k_direct_refuted/ledger_update.json`; `/home/ubuntu/tinygrad-arkey/docs/amd-isa-q6k-direct-route-full-scope-20260629.md` |
| Prefill role-selective pipe is promoted and supersedes global pipe where eligible | `/home/ubuntu/tinygrad-arkey/bench/qk-prefill-pipe-role-selective/latest.json`; `/home/ubuntu/tinygrad-arkey/bench/qk-candidate-evaluator/prefill_pipe_role_selective_default/ledger_update.json` |
| Prefill whole-role attribution and authority harness are the prefill measurement sources | `/home/ubuntu/tinygrad-arkey/bench/qk-prefill-whole-role-attribution/latest.json`; `/home/ubuntu/tinygrad-arkey/bench/qk-prefill-whole-role-attribution/summary.md`; `/home/ubuntu/tinygrad-arkey/extra/qk_prefill_whole_synced.py` |
| Runtime overhead must be separated from GPU/codegen work | `/home/ubuntu/tinygrad-arkey/bench/qk-decode-runtime-overhead/result.json`; `/home/ubuntu/tinygrad-arkey/extra/qk_decode_runtime_overhead.py` |
| 14B/32B Q4_K route miss, anyshape binding, and topology-space stop | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-truegen-q1432-result-20260630.md`; `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-shape-tuned-topology-kt-result-20260630.md` |
| 14B split-K was measured/refuted for FFN; gap redirected away from Q4_K FFN | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-split-k-sk-result-20260630.md` |
| 14B/32B model-driven role attribution redirected target to reduce/source resolution | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-ldr-attribution-result-20260630.md`; `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-model-driven-decode-route-continuation-scope-20260630.md` |
| Reduce-source tracing identified `attn_k` route miss as the major 14B issue | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-attn-k-route-miss-result-20260630.md`; `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-reduce-source-resolution-scope-20260630.md`; `/home/ubuntu/tinygrad-arkey/extra/qk_decode_reduce_source_trace.py` |
| Decode attention native route is correct but low-leverage / not the active max-out target | `/home/ubuntu/tinygrad-arkey/bench/amd-isa-backend-decode-attention-ceiling/latest.json`; `/home/ubuntu/tinygrad-arkey/docs/amd-isa-system-residual-to-bandwidth-ceiling-scope-20260629.md` |
| Candidate evaluator already exists in tinygrad and should be moved conceptually, not copied wholesale | `/home/ubuntu/tinygrad-arkey/extra/qk_candidate_evaluator.py`; `/home/ubuntu/tinygrad-arkey/bench/qk-candidate-evaluator/` |

### Citation Rule For New Work

Every BoltBeam ledger row, promoted/refuted decision, and historical seed entry
must include:

- `evidence.path`: repo-relative if the artifact is in BoltBeam, absolute only
  for external tinygrad provenance during migration;
- `evidence.kind`: normalized artifact kind;
- `evidence.fingerprint`: byte fingerprint of the cited file;
- `claim`: one sentence saying exactly what the citation proves.

Do not cite a scope doc as speed evidence if a result artifact exists. Scope docs
may justify why a phase exists; result JSON/summary artifacts justify verdicts.

## Design Principles That Must Hold

This build must follow [coding-principles.md](coding-principles.md):

- **Centralize authority:** schemas, result kinds, verdict enums, thresholds,
  route families, and ledger states live in one place.
- **Modularize execution:** readers, cache, evaluator, ledger, and CLI are
  separate modules with narrow interfaces.
- **Abstract for simplicity:** public CLI stays boring; internal adapters can
  handle messy tinygrad artifact details.
- **Keep orthogonal:** tinygrad execution, artifact normalization, evaluation,
  and policy update are independent.
- **Encode invariants:** use typed result objects and schemas, not loose
  stringly-typed dictionaries at boundaries.
- **No re-sprawl:** new artifact kind = adapter row/schema, not a cloned script.
- **Audit before build:** if an artifact cannot prove a decision, verdict is
  `inconclusive` or `adapter-incomplete`, not a guessed promotion.

## Current BoltBeam State

Already present:

- `profile/`: GGUF reader and `ModelProfile`.
- `search/`: route-family search-space emission.
- `policy/`: seed route-policy emission.
- `synth/`: shape fixture manifest.
- `analyze`: handoff bundle that writes profile/search/policy/fixtures,
  measurement plan, and tinygrad commands.
- schemas for model/search/analysis/measurement-plan artifacts.
- coding principles and commit-message hook.

Missing:

- result artifact ingestion;
- normalized evidence model;
- artifact cache/fingerprints;
- route/candidate manifest;
- candidate evaluator;
- refuted/deferred ledger;
- policy consistency guard;
- ceiling/roofline math helpers;
- compact report generation;
- reusable audit CLI.

## What To Port From tinygrad

Port concepts and result contracts, not GPU code.

Relevant tinygrad tools and what BoltBeam should do with them:

| tinygrad tool/artifact family | tinygrad role | BoltBeam role | source |
|---|---|---|---|
| `qk_decode_role_attribution_modular.py` | run model/profile/capture and emit role bucket artifacts | ingest role attribution and normalize per-role wall share | `/home/ubuntu/tinygrad-arkey/extra/qk_decode_role_attribution_modular.py`; 14B use: `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-ldr-attribution-result-20260630.md` |
| `qk_decode_reduce_source_trace.py` | trace reduce kernels to source positions | ingest reduce-source rows and attach them to candidate decisions | `/home/ubuntu/tinygrad-arkey/extra/qk_decode_reduce_source_trace.py`; attn_k result: `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-attn-k-route-miss-result-20260630.md` |
| `qk_decode_runtime_overhead.py` | separate host sync from GPU work | ingest runtime-overhead rows and prevent false codegen blame | `/home/ubuntu/tinygrad-arkey/extra/qk_decode_runtime_overhead.py`; `/home/ubuntu/tinygrad-arkey/bench/qk-decode-runtime-overhead/result.json` |
| prefill authority artifacts | measure synced whole-prefill throughput | ingest as authority gate evidence | `/home/ubuntu/tinygrad-arkey/extra/qk_prefill_whole_synced.py`; `/home/ubuntu/tinygrad-arkey/bench/qk-prefill-whole-role-attribution/latest.json` |
| decode W==D artifacts | measure token/s and route binding | ingest as authority gate evidence | current candidate examples: `/home/ubuntu/tinygrad-arkey/bench/amd-isa-backend-g3-weight-promotion/latest.json`; `/home/ubuntu/tinygrad-arkey/bench/amd-isa-backend-q6k-direct-speed/latest.json` |
| policy consistency checks | catch stale docs/policies | reimplement as BoltBeam policy guard over ledgers/manifests | `/home/ubuntu/tinygrad-arkey/extra/qk_policy_consistency_check.py`; principles: `/home/ubuntu/tinygrad-arkey/docs/repo-principles-audit-remediation-scope-20260630.md` |
| artifact cache | fingerprint results and source state | implement in BoltBeam over normalized evidence | `/home/ubuntu/tinygrad-arkey/extra/qk_artifact_cache.py`; consolidation note: `/home/ubuntu/tinygrad-arkey/docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md` |
| candidate evaluator | classify promote/refute/defer | implement in BoltBeam as the main decision engine | `/home/ubuntu/tinygrad-arkey/extra/qk_candidate_evaluator.py`; `/home/ubuntu/tinygrad-arkey/bench/qk-candidate-evaluator/` |
| refutation ledgers | prevent re-chasing closed axes | implement as durable ledger JSONL/JSON | examples: `/home/ubuntu/tinygrad-arkey/bench/qk-candidate-evaluator/decode_q6k_direct_refuted/ledger_update.json`; `/home/ubuntu/tinygrad-arkey/docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md` |
| roofline/ceiling audits | calculate theoretical/practical headroom | implement in BoltBeam as pure math helpers | `/home/ubuntu/tinygrad-arkey/extra/amd_isa_system_residual_ceiling_audit.py`; `/home/ubuntu/tinygrad-arkey/docs/amd-isa-system-residual-to-bandwidth-ceiling-scope-20260629.md` |

Do **not** port:

- AMD ISA phase gates;
- kernel microgates;
- SQTT/PMC capture code;
- model-loading code;
- raw GPU benchmark loops;
- one-off archived probes;
- tinygrad runtime flags as BoltBeam defaults.

## Target Package Layout

Add these modules only as their phase requires them:

```text
boltbeam/
  artifacts/
    __init__.py
    base.py              # ArtifactKind, ArtifactEnvelope, ResultReader
    tinygrad.py          # tinygrad artifact adapters
  cache/
    __init__.py
    fingerprint.py       # source/artifact/profile fingerprints
    store.py             # small local cache index
  ledger/
    __init__.py
    model.py             # LedgerEntry, Verdict, EvidenceRef
    store.py             # JSONL/JSON ledger IO
  eval/
    contracts.py         # existing promotion requirements
    evaluator.py         # CandidateDecision engine
    thresholds.py        # TIER_A/B/C and regression guards
  report/
    __init__.py
    markdown.py          # human summary reports
  math/
    __init__.py
    roofline.py          # pure ceiling/Amdahl helpers
```

Add schemas as the durable boundary:

```text
schemas/normalized_evidence.schema.json
schemas/candidate_decision.schema.json
schemas/route_ledger.schema.json
schemas/artifact_cache.schema.json
schemas/tinygrad_artifact_index.schema.json
schemas/ceiling_report.schema.json
```

Public CLI additions should be few and stable:

```bash
boltbeam ingest ARTIFACT_OR_DIR --kind auto --out evidence.json
boltbeam evaluate --profile model_profile.json --search search_space.json --evidence evidence.json --out decision.json
boltbeam ledger add decision.json --ledger route_ledger.jsonl
boltbeam ledger report route_ledger.jsonl --out summary.md
boltbeam cache status
boltbeam check-policy
```

Avoid adding one CLI command per tinygrad phase.

## Core Data Model

### Normalized Evidence

Every ingested result becomes:

```json
{
  "schema": "boltbeam.normalized_evidence.v1",
  "evidence_id": "sha256:...",
  "source": {
    "producer": "tinygrad",
    "tool": "qk_decode_role_attribution_modular.py",
    "path": "bench/...",
    "fingerprint": "sha256:..."
  },
  "model_id": "qwen3-14b",
  "target_id": "amd_gfx1100",
  "workload": "decode",
  "contexts": [128, 512],
  "rows": [
    {
      "kind": "role_attribution",
      "role": "attn_kv",
      "quant": "Q4_K",
      "shape": [1024, 5120],
      "metric": "wall_share",
      "value": 0.38,
      "unit": "fraction",
      "confidence": "measured"
    }
  ],
  "flags": {
    "route_bound": true,
    "token_match": true,
    "deterministic": true,
    "hidden_fallback": false
  }
}
```

Rules:

- raw artifact fields stay in `source.raw_summary` only when needed;
- evaluator reads normalized rows, not tinygrad-specific shapes;
- unknown fields are preserved but not used for verdicts;
- missing required fields produce `adapter-incomplete`, not a crash.

### Candidate Decision

```json
{
  "schema": "boltbeam.candidate_decision.v1",
  "candidate_id": "decode_q4k_g3_anyshape_attn_k",
  "model_id": "qwen3-14b",
  "target_id": "amd_gfx1100",
  "workload": "decode",
  "verdict": "promote|refute|defer|inconclusive|search-space-incomplete",
  "tier": "A|B|C|none",
  "reason": "Q4_K attn_k route miss dominates decode and G3 route is token-identical",
  "evidence": ["sha256:..."],
  "rollback": {"DECODE_ROUTE_ATTN_K": "0"},
  "next_action": "profile-scoped promotion gate",
  "guardrails": {
    "correctness": "pass",
    "route_bound": "pass",
    "speed": "pass",
    "memory_fit": "pass",
    "fallback": "pass"
  }
}
```

### Ledger Entry

```json
{
  "schema": "boltbeam.route_ledger_entry.v1",
  "candidate_id": "decode_q6k_direct_halfwarp",
  "status": "refuted",
  "scope": {
    "model_family": "qwen3",
    "quant": "Q6_K",
    "target": "amd_gfx1100"
  },
  "evidence": ["sha256:..."],
  "reopen_condition": "new row grouping or reduction primitive not present in tested route",
  "do_not_retry": true
}
```

## Verdict Vocabulary

Use one enum everywhere:

| verdict | meaning |
|---|---|
| `diagnostic` | explains a bottleneck, not a candidate |
| `candidate` | worth measuring, not promoted |
| `promote` | passes correctness, route, speed, memory, rollback |
| `refute` | failed the relevant gate with firm evidence |
| `defer` | promising but blocked by a named capability |
| `inconclusive` | evidence insufficient or noisy |
| `search-space-incomplete` | candidate failed because BoltBeam cannot express needed knobs yet |
| `adapter-incomplete` | BoltBeam cannot safely read the artifact yet |

No other free-form verdict strings in evaluator output.

## Phase Plan

### BB0 - Inventory And Boundary Lock

Goal: make the split explicit before writing evaluator code.

Work:

- Add a tinygrad artifact inventory fixture under `tests/fixtures/tinygrad/`.
- Document which tinygrad outputs are supported in v1.
- Add a `TinygradArtifactKind` table:
  - `decode_role_attribution`;
  - `decode_reduce_source_trace`;
  - `decode_runtime_overhead`;
  - `decode_wd`;
  - `prefill_authority`;
  - `promotion_gate`;
  - `ceiling_report`.

Acceptance:

- no GPU execution from BoltBeam;
- `boltbeam analyze` remains unchanged except for referencing supported artifact
  kinds;
- tests prove unsupported artifact kinds return `adapter-incomplete`.

### BB1 - Normalized Evidence Model And Schemas

Goal: create the one data shape all evaluators consume.

Work:

- Add `boltbeam/artifacts/base.py`.
- Add `NormalizedEvidence`, `EvidenceRow`, `EvidenceFlags`,
  `EvidenceSource`.
- Add `schemas/normalized_evidence.schema.json`.
- Add JSON writer/reader with schema id checks.

Acceptance:

- unit tests create evidence rows for role attribution and reduce trace;
- missing `model_id`, `target_id`, or `workload` fails validation;
- no evaluator imports raw tinygrad artifact fields.

### BB2 - Tinygrad Artifact Adapters

Goal: ingest raw tinygrad artifacts into normalized evidence.

Work:

- Add `boltbeam/artifacts/tinygrad.py`.
- Implement adapter registry keyed by tool name / schema / path hints.
- Support at least:
  - role attribution rows;
  - reduce source rows;
  - runtime overhead rows;
  - W==D/token-speed rows if present;
  - prefill authority rows if present.
- Preserve raw artifact path and fingerprint.

Acceptance:

- fixture artifacts from tinygrad-style JSON normalize into evidence;
- malformed but recognizable artifacts return `adapter-incomplete` with reason;
- unknown artifacts return `unsupported-artifact-kind`;
- no hardcoded `/home/ubuntu/tinygrad-arkey` paths in normalized evidence.

### BB3 - Artifact Cache And Fingerprints

Goal: prevent re-evaluating stale or duplicated evidence.

Work:

- Add `boltbeam/cache/fingerprint.py` and `store.py`.
- Fingerprint:
  - artifact bytes;
  - model profile JSON;
  - search-space JSON;
  - target profile;
  - optional tinygrad commit hash if present in artifact metadata.
- Add `artifact_cache.json` schema.

Acceptance:

- same evidence gets same fingerprint across checkouts;
- absolute paths are excluded from stable ids;
- cache status CLI reports entries, producers, model ids, and stale/missing
  artifact paths;
- tests prove path relocation does not change semantic fingerprint.

### BB4 - Route/Candidate Manifest

Goal: candidate definitions are data, not scattered evaluator branches.

Work:

- Add `schemas/candidate_manifest.schema.json`.
- Add default manifest data under `boltbeam/data/candidates.json`.
- Initial candidate rows:
  - `decode_q4k_lanemap_gemv`;
  - `decode_q4k_g3_anyshape`;
  - `decode_q6k_coop_shipped`;
  - `decode_q6k_direct_halfwarp_refuted`;
  - `decode_attention_native_correct_not_fast`;
  - `prefill_pipe_global`;
  - `prefill_pipe_role_selective`;
  - `runtime_host_sync`;
  - `reduce_elimination`.
- Include:
  - workload;
  - quant/role applicability;
  - required evidence kinds;
  - thresholds;
  - rollback env;
  - refuted-axis tags.

Acceptance:

- evaluator discovers candidates from manifest;
- adding a candidate row needs no top-level evaluator edit if evidence kinds are
  already supported;
- route families in `search_space.json` can map to candidate ids.

### BB5 - Candidate Evaluator

Goal: classify evidence into promote/refute/defer/inconclusive decisions.

Work:

- Add `boltbeam/eval/evaluator.py`.
- Add `boltbeam/eval/thresholds.py`.
- Implement tier policy:
  - TIER_A: broad win, default-promotable if guardrails pass;
  - TIER_B: residual win, promotable with clean rollback and no protected
    regression;
  - TIER_C: diagnostic only unless strategic;
  - exact thresholds live in one table.
- Guardrails:
  - correctness;
  - route-bound;
  - hidden fallback absent;
  - deterministic or noise-qualified;
  - speed tier;
  - memory fit if evidence exists;
  - rollback specified.

Acceptance:

- fixtures reproduce these decision shapes:
  - Q4_K G3 speed-equivalent -> `promote`;
  - Q6_K direct regression -> `refute`;
  - native attention correct-not-fast -> `defer` or `refute` with low-leverage;
  - missing topology knob -> `search-space-incomplete`;
  - missing correctness flag -> `inconclusive`.
- no candidate can promote without rollback.

### BB6 - Ledger Store

Goal: durable memory for promoted/refuted/deferred/search-space-incomplete
routes.

Work:

- Add `boltbeam/ledger/model.py` and `store.py`.
- Add JSONL store for append-only evidence history.
- Add compact JSON summary for current state.
- Ledger entries include reopen condition.

Acceptance:

- `boltbeam ledger add decision.json --ledger route_ledger.jsonl`;
- duplicate decision id updates summary but preserves history;
- `do_not_retry` rows suppress candidate recommendations unless reopen
  condition is explicitly met;
- tests cover promote, refute, defer, and search-space-incomplete.

### BB7 - Policy Consistency Guard

Goal: stop docs/manifests/policies from drifting.

Work:

- Add `boltbeam/policy/check.py`.
- Check:
  - every promoted candidate has rollback;
  - every refuted candidate has evidence and reopen condition;
  - search-space-incomplete candidates are not listed as refuted;
  - route policy does not select a refuted candidate;
  - docs mention current route status consistently when docs are configured as
    checked inputs.
- Add `boltbeam check-policy` CLI.

Acceptance:

- guard passes on seeded fixtures;
- a temporary bad fixture selecting a refuted route fails;
- a promoted route without rollback fails;
- no brittle grep over archive docs unless explicitly configured.

### BB8 - CLI Workflow

Goal: one boring CLI workflow for the full audit loop.

Work:

Add:

```bash
boltbeam ingest PATH_OR_DIR --kind auto --out evidence.json
boltbeam evaluate --profile model_profile.json --search search_space.json --evidence evidence.json --out decision.json
boltbeam ledger add decision.json --ledger route_ledger.jsonl
boltbeam ledger report route_ledger.jsonl --out summary.md
boltbeam cache status
boltbeam check-policy
```

Rules:

- CLI should call module functions only;
- no GPU execution;
- all commands return nonzero on malformed machine contracts;
- human reports are derived from machine-readable decisions.

Acceptance:

- end-to-end test with fixtures:

```text
analyze fixture profile
ingest fixture tinygrad results
evaluate candidate
ledger add
ledger report
check-policy
```

### BB9 - Ceiling And Amdahl Math

Goal: keep theoretical ceilings in BoltBeam, not as one-off markdown math.

Work:

- Add `boltbeam/math/roofline.py`.
- Inputs:
  - bytes read;
  - measured bandwidth;
  - kernel/share fraction;
  - candidate speedup;
  - achievable bandwidth;
  - context rows where relevant.
- Outputs:
  - floor ms/token;
  - ceiling tok/s;
  - Amdahl projected whole-model gain;
  - confidence/assumptions.
- Add `ceiling_report` schema.

Acceptance:

- tests reproduce simple known Amdahl cases;
- report distinguishes memcpy peak from dequant-achievable bandwidth;
- no model-specific constants outside fixture/profile data.

### BB10 - Search-Space Reachability Audit

Goal: determine whether a missed win is due to evidence, implementation, or
unexposed knobs.

Work:

- Add `boltbeam/search/reachability.py`.
- Given profile + target + manifest:
  - enumerate legal route families;
  - identify knobs present in grammar but not emitter;
  - identify candidates blocked by target capabilities;
  - classify as:
    - `reachable`;
    - `emitter-blocked`;
    - `primitive-missing`;
    - `target-incomplete`;
    - `refuted-by-ledger`.

Acceptance:

- fixture can represent the known large-shape case:
  - Q4_K G3 route reachable;
  - non-exposed topology knob -> `emitter-blocked`;
  - split-K if not represented -> `search-space-incomplete` or
    `emitter-blocked`, not refuted.
- result feeds evaluator/ledger.

### BB11 - Report Generation

Goal: make Claude/user handoffs reproducible from ledgers and evidence.

Work:

- Add `boltbeam/report/markdown.py`.
- Generate:
  - current promoted/refuted/deferred table;
  - hottest measured buckets;
  - next actions;
  - missing evidence;
  - search-space-incomplete knobs;
  - rollback commands.

Acceptance:

- report uses decisions/ledger only, not raw ad hoc artifact parsing;
- every claim links to evidence id/path;
- report is deterministic.

### BB12 - Legacy Evidence Seed

Goal: import the most important existing conclusions as seed ledger entries
without porting archived scripts.

Work:

Create a seed ledger fixture with these rows, each citing tinygrad artifact paths
as provenance when available:

| seed row | required citation |
|---|---|
| Q4_K G3 generated route speed-equivalent on 8B | `/home/ubuntu/tinygrad-arkey/bench/amd-isa-backend-g3-weight-promotion/latest.json`; `/home/ubuntu/tinygrad-arkey/bench/qk-candidate-evaluator/decode_q4k_g3_generated/ledger_update.json` |
| Q6_K direct half-warp refuted | `/home/ubuntu/tinygrad-arkey/bench/amd-isa-backend-q6k-direct-speed/latest.json`; `/home/ubuntu/tinygrad-arkey/bench/qk-candidate-evaluator/decode_q6k_direct_refuted/ledger_update.json` |
| native attention correct-route-bound but not fast enough | `/home/ubuntu/tinygrad-arkey/bench/amd-isa-backend-decode-attention-ceiling/latest.json`; `/home/ubuntu/tinygrad-arkey/docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md` |
| prefill pipe global promoted | `/home/ubuntu/tinygrad-arkey/bench/qk-prefill-pipe-promotion/latest.json`; successor context in `/home/ubuntu/tinygrad-arkey/bench/qk-prefill-pipe-role-selective/latest.json` |
| prefill role-selective promoted | `/home/ubuntu/tinygrad-arkey/bench/qk-prefill-pipe-role-selective/latest.json`; `/home/ubuntu/tinygrad-arkey/bench/qk-candidate-evaluator/prefill_pipe_role_selective_default/ledger_update.json` |
| Q4_K large-shape route miss fixed by anyshape binding | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-truegen-q1432-result-20260630.md` |
| topology tuning stopped for 14B/32B because legal alternatives are more serial | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-shape-tuned-topology-kt-result-20260630.md` |
| split-K for 14B FFN refuted as role-local no-win | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-split-k-sk-result-20260630.md` |
| reduce-source trace identified attn_k route miss vs pure reduce | `/home/ubuntu/tinygrad-arkey/docs/qwen-14b-32b-attn-k-route-miss-result-20260630.md` |

Acceptance:

- seed ledger imports with `boltbeam ledger import`;
- every seed row has `status`, `evidence`, and `reopen_condition`;
- seed rows are clearly marked as imported historical evidence, not freshly
  measured by BoltBeam.

## End-To-End Acceptance Gate

The build is complete when this works without running the GPU:

```bash
python3 -m unittest discover -s tests -v

PYTHONPATH=. python3 -m boltbeam.cli analyze \
  /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --target amd_gfx1100 \
  --id qwen3-14b \
  --out-dir outputs/qwen3-14b-analysis

PYTHONPATH=. python3 -m boltbeam.cli ingest \
  tests/fixtures/tinygrad/qwen3_14b_role_attribution.json \
  --out outputs/qwen3-14b-analysis/evidence.role.json

PYTHONPATH=. python3 -m boltbeam.cli evaluate \
  --profile outputs/qwen3-14b-analysis/model_profile.json \
  --search outputs/qwen3-14b-analysis/search_space.json \
  --evidence outputs/qwen3-14b-analysis/evidence.role.json \
  --out outputs/qwen3-14b-analysis/decision.json

PYTHONPATH=. python3 -m boltbeam.cli ledger add \
  outputs/qwen3-14b-analysis/decision.json \
  --ledger outputs/qwen3-14b-analysis/route_ledger.jsonl

PYTHONPATH=. python3 -m boltbeam.cli ledger report \
  outputs/qwen3-14b-analysis/route_ledger.jsonl \
  --out outputs/qwen3-14b-analysis/summary.md

PYTHONPATH=. python3 -m boltbeam.cli check-policy \
  --ledger outputs/qwen3-14b-analysis/route_ledger.jsonl
```

Expected verdict:

```text
BOLTBEAM_AUDIT_BRAIN_PASS
```

## Non-Goals

- No kernel implementation.
- No AMD ISA backend changes.
- No tinygrad runtime route changes.
- No GPU benchmark execution in BoltBeam.
- No direct dependency on tinygrad imports.
- No wholesale import of archived tinygrad probes.
- No one-off command per historical phase.
- No default model/runtime policy changes.

## Claude Build Instructions

1. Start with BB1/BB2, not the evaluator. The evaluator must consume normalized
   evidence, not raw tinygrad artifacts.
2. Keep each phase separately committed with BoltBeam prefixes:
   - `[schema]` for schemas;
   - `[eval]` for evaluator/thresholds;
   - `[policy]` for ledgers/policy checks;
   - `[cli]` for CLI wiring;
   - `[test]` for fixtures/tests;
   - `[docs]` for documentation only.
3. Run `python3 -m unittest discover -s tests -v` after every phase.
4. Do not add a new file for each new model or route. Add rows to manifests or
   fixtures.
5. If a tinygrad artifact shape is unclear, write an adapter-incomplete fixture
   and stop with a precise reason. Do not guess.
6. If a candidate lacks the needed topology knobs, return
   `search-space-incomplete` and add the knob to the reachability audit. Do not
   hand-write a special route in BoltBeam.
7. Keep generated output under `outputs/` ignored unless the artifact is a small
   durable fixture or schema.
8. Push after each completed phase.
