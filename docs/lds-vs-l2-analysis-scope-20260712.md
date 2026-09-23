# LDS-vs-L2 Analysis Scope (2026-07-12)

Exhaustive scope for giving BoltBeam the ability to **decide, per operand, whether to stage a
tile through LDS or rely on the cache (L2 / MALL)** — grounded in the interfaces that exist
today, and honest about the one resource we do **not** have (L2 hardware counters are blocked on
gfx11).

Status: scope only. No code in this document promotes anything; every number the pipeline
produces is a *candidate* until measured on the real machine.

---

## 1. The decision, stated precisely

"LDS vs L2" is **not** a mutually-exclusive kernel switch. L2 is always in the path
(every global load traverses it); LDS is an *additional* explicitly-managed buffer you may stage
into **on top of** the cache. The real question is **per access-stream (per operand / role)**:

> For *this* operand's reuse, is manually staging its tile in LDS worth the cost
> (a DRAM→LDS copy + a barrier + lost occupancy + bank-conflict risk), or does the cache
> (L2, then MALL) already capture the reuse well enough?

A single kernel can answer this differently for different operands (weights vs activations vs KV).
The answer is a **surface over the workload point** (shape, ctx, batch, dtype), not a constant:
the same kernel can want LDS at ctx=4096 and L2 at ctx=512. See
`dictionary/01-gpu-memory.md#lds-vs-l2--the-one-line-contrast` for the plain-language version.

Deliverable: an oracle that, given (target facts, operand access-stream, workload point), emits a
**per-operand staging recommendation with an interval and a truth-status**, expressed as a
BoltBeam candidate that measurement promotes or refutes.

---

## 2. Constraints imposed by reality (these shape everything below)

1. **L2 attribution counters are blocked.** `data/targets.json` gfx1100
   `capabilities.counter_status` = `{"SQ_BUSY_CYCLES": "working", "others": "blocked_gfx11_pmc"}`.
   `GL2C_HIT` / `GL2C_MISS` / `GL2C_EA_RDREQ_*` are mapped but dead. **We cannot directly measure
   L2 hit-rate or bytes-at-L2.** The only trustworthy signal is **end-to-end timing**
   (`SQ_BUSY_CYCLES` + wall-clock). Everything here is founded on timing, not internal attribution.
2. **BoltBeam does not run the GPU.** Per README + `runner_plan.py`: BoltBeam emits
   `trace_request.json` inside a `runner_bundle`, an external (tinygrad-side) runner executes, and
   BoltBeam ingests. The sweep must be a **request → external run → ingest → classify** flow.
3. **RDNA3 MALL (Infinity Cache) is adaptive + prefetched.** The capacity "cliffs" a sweep sees
   are **soft bands, not sharp knees**. We emit intervals and DERIVED bands, and never claim exact
   cache bytes.
4. **Clock ramp / thermal drift** (already handled in `roofline_ceiling.py`): cold ≠ sustained.
   The sweep must run sustained and reuse the cold-vs-sustained regime logic.
5. **Measured facts are machine-state-specific.** They must be keyed to a `system_snapshot_id`
   (the `CalibrationProfile` pattern), not baked into the static target registry as if universal.

---

## 3. What exists today (inventory — reuse, don't rebuild)

| Asset | File | What it gives us |
|---|---|---|
| Streaming-ceiling resolver | `boltbeam/roofline_ceiling.py` | `resolve_achieved_ceiling(cold_samples, sustained_samples, raw_peak_gbs, reference_kernel_gbs) -> CeilingResolution`. Pure fn; **currently uncalled**. Handles clock-ramp / weak-copy. This is 80% of the per-size sweep classifier. |
| Analytical cycle model | `boltbeam/perf/cycle_model.py` | `predict_cycles(graph, resources, launch, calibration, …)`. Already has instruction classes `lds_load`/`lds_store`/`global_load`/`global_store`, a `memory_completion` floor, and occupancy from `lds_bytes`. **No cache tier** — knows DRAM bandwidth only. |
| Occupancy bound | `boltbeam/perf/occupancy.py` | `derive_occupancy(resources, calibration)` — the LDS/register→resident-waves penalty the decision trades against. |
| Truth model | `boltbeam/core/facts.py` | `TruthStatus` = measured / derived / modeled / assumed / imported / unknown; `Fact` with status. |
| Calibration artifact | `boltbeam/perf/calibration.py` | `CalibrationProfile(system_snapshot_id, facts, …)`, `.interval(name)`, `.fact(name)`, `calibration_id` = `sha256_json`. The shape our measured tier-facts belong in. |
| Provider-neutral handoff | `boltbeam/runner_plan.py` | emits `trace_request.json` + `runner_bundle_manifest.json`. The channel the sweep request rides. |
| Normalized ingest | `boltbeam/artifacts/base.py` | `NormalizedEvidence`, `AdapterIncomplete` (fail-closed on missing fields), `sha256_*`. The ingest discipline. |
| Target registry | `boltbeam/data/targets.json`, `targets.py`, `profile/ir.py::TargetProfile` | gfx1100: `lds_bytes_per_cu=65536`, `memory_bandwidth_gbs=960`, `compute_units=96`. **Missing: L2 size, MALL size, L2 bw, LDS bw.** |
| Promote/refute loop | `cli.py::cmd_evaluate`, `cmd_reopen_check`, ledger | candidates promote/refute by evidence; refuted can `reopen-check`. The A/B confirm rides this. |

---

## 4. Gaps (the actual build)

- **G1 — No cache-tier facts.** Nothing carries L2 size, MALL size, L2 bandwidth, or LDS
  bandwidth. Without them neither the model nor the calculator can distinguish "hits L2 (fast)"
  from "misses to DRAM (slow)".
- **G2 — No sweep collector.** `resolve_achieved_ceiling` exists but there's no working-set
  *sweep* request, no ingest, no knee/band classifier.
- **G3 — cycle_model has no tier resolution.** `memory_completion` is single-bandwidth; a
  `global_load`'s effective bandwidth must become a function of working-set residency.
- **G4 — No per-operand staging candidate.** No route-family template that expresses
  "operand X: LDS-staged" vs "operand X: cache-streamed", and no crossover evaluator.
- **G5 — No CLI surface / report integration** for any of the above.

---

## 5. Architecture & data flow

```
        ┌─────────────────────────── BoltBeam (no GPU) ───────────────────────────┐
target  │  W1 emit sweep request ─▶ trace_request.json (in runner_bundle)         │
facts   │                                    │                                    │
        │                                    ▼   (external tinygrad-side runner)  │
        │                          W2 runner executes copy sweep ────────────┐    │
        │                                    │ ws_bandwidth_samples.json      │    │
        │  W3 ingest + band-classify ◀───────┘                               │    │
        │        │  memory_hierarchy_profile.json  (Facts, MEASURED/DERIVED,  │    │
        │        │   keyed to system_snapshot_id; L2 hit-rate = UNKNOWN)      │    │
        │        ▼                                                            │    │
        │  W4 reconcile with targets.json spec priors (ASSUMED)              │    │
        │        │                                                            │    │
        │        ▼                                                            │    │
        │  W5 cycle_model tier-aware   ──▶  W6 per-operand LDS/L2 candidate    │    │
        │      (predict staged vs           (crossover math; emits candidates │    │
        │       unstaged, per operand)       + interval + truth-status)       │    │
        │        │                                                            │    │
        │        ▼                                                            │    │
        │  W7 A/B confirm: emit staged & streamed candidates ─▶ trace_request ─┘    │
        │      ingest timing ─▶ evaluate/promote-or-refute (existing loop)          │
        └───────────────────────────────────────────────────────────────────────┘
```

Two measurement round-trips: **W1–W3** (once per machine snapshot — recover the tiers) and
**W7** (per operand family — confirm the prediction). W5/W6 are offline pruners between them.

---

## 6. Work items (component-by-component)

### W0 — Vocabulary & schema
- Add schema constants (in `boltbeam/vocab.py`): `SCHEMA_WS_SWEEP_REQUEST`,
  `SCHEMA_WS_SWEEP_SAMPLES`, `SCHEMA_MEMORY_HIERARCHY_PROFILE`, `SCHEMA_LDS_L2_CANDIDATE`.
- Add fact-name vocabulary (namespaced): `mem.tier.l2.bytes`, `mem.tier.mall.bytes`,
  `mem.tier.l2.bandwidth_gbs`, `mem.tier.mall.bandwidth_gbs`, `mem.tier.dram.bandwidth_gbs`,
  `mem.tier.lds.bandwidth_gbs`, `mem.tier.lds.bytes_per_cu` (mirror of registry, MEASURED echo).

### W1 — Working-set sweep **request** (BoltBeam side)
New `boltbeam/mem_sweep.py::build_ws_sweep_request(target, *, sizes=None) -> dict`.
- Default sizes: geometric ladder from **4 KB → 256 MB** (spanning below L2, across MALL, into
  DRAM), e.g. ~28 points at √2 spacing; caller-overridable.
- For each size: request a **sustained** streaming-copy loop (read+write counted) *and* a short
  **cold** burst (feeds `roofline_ceiling`'s regime logic), plus repeat count for medians.
- Also request one **LDS-resident** micro-loop (fixed ≤64 KB buffer, reused N×) to measure
  `mem.tier.lds.bandwidth_gbs` directly — the one LDS number a copy sweep can't get.
- Emit as `trace_request.json` content via the existing `runner_plan` bundle mechanism
  (extend `runner_plan.py` request list, or add `--include ws-sweep`).
- **No execution here.** Output is a request artifact only.

### W2 — Runner-side executor (external contract, tinygrad-side — *spec only, not built here*)
Define the contract the tinygrad-side runner must satisfy; it lives in the fork's `extra/qk`
(alongside the existing decode harnesses), **not** in BoltBeam:
- Input: `ws_sweep_request.json`. Output: `ws_bandwidth_samples.json` conforming to
  `SCHEMA_WS_SWEEP_SAMPLES`: `[{bytes, cold_gbs[], sustained_gbs[], kind: copy|lds_resident,
  clock_state?, system_snapshot_id}]`.
- Must pin clocks / run sustained; must record `system_snapshot_id` (driver+clock+board identity)
  so the profile is machine-stamped. Missing fields → `AdapterIncomplete` on ingest (fail-closed).

### W3 — Ingest + band classifier (BoltBeam side) → `memory_hierarchy_profile.json`
New `boltbeam/mem_hierarchy.py`:
- For each sweep size, call `resolve_achieved_ceiling(cold_samples=…, sustained_samples=…)` to get
  a clean per-size achieved GB/s (reusing its clock-ramp / weak-copy guards).
- **Band detection** over the size→GB/s curve: locate the descending plateaus
  (L2-speed → MALL-speed → DRAM-speed) and the transition bands between them. Emit:
  - `mem.tier.l2.bandwidth_gbs`, `mem.tier.mall.bandwidth_gbs`, `mem.tier.dram.bandwidth_gbs`
    from plateau medians → **`TruthStatus.MEASURED`** (they're timed).
  - `mem.tier.l2.bytes`, `mem.tier.mall.bytes` from the transition-band midpoints →
    **`TruthStatus.DERIVED`** with an explicit `uncertainty` interval (soft bands per §2.3).
  - `mem.tier.lds.bandwidth_gbs` from the LDS-resident loop → **MEASURED**.
  - `mem.l2.hit_rate` → **`TruthStatus.UNKNOWN`** with reason `blocked_gfx11_pmc` (recorded, not
    guessed — the honesty marker).
- Package as a `CalibrationProfile`-shaped artifact (`system_snapshot_id` + Facts), id via
  `sha256_json`. Fewer than 2 plateaus detected → emit `blockers=["sweep-underspecified: <n> "
  "plateaus"]` and promote nothing (fail-closed).

### W4 — Spec priors + reconciliation
- Add **optional** spec-sheet priors to `targets.json` gfx1100 under a new `memory_tiers` block
  (`l2_bytes`, `mall_bytes`, and any vendor bandwidths) with an inline note that these are
  **priors only** (`TruthStatus.ASSUMED`/`IMPORTED`), extend `TargetProfile` + `_from_row` in
  `targets.py`/`profile/ir.py` to carry it.
- `boltbeam/reconciliation.py` (exists) gets a check: measured (W3) vs prior (W4); on large
  disagreement, **measured wins**, prior flagged. Never let a spec number silently override a
  measurement (mirrors `roofline_ceiling`'s reference-floor discipline).

### W5 — Tier-aware cycle model
- Extend `predict_cycles` memory accounting: a `global_load`'s effective service bandwidth is
  resolved from the **residency of its access-stream working set** against the W3 tiers
  (`W_concurrent ≤ l2.bytes → l2.bw; ≤ mall.bytes → mall.bw; else dram.bw`), and an `lds_load` uses
  `lds.bandwidth_gbs`. Keep it interval-valued (optimistic/median/pessimistic already exist).
- Requires threading the operand's `working_set_bytes` + `reuse` into `resources`/`launch`
  (new fields; absent → blocker, not a guess).
- Gate: if the tier facts' truth-status is worse than MEASURED/DERIVED, the prediction inherits an
  `assumptions`/`blockers` entry — never silently "modeled as fact".

### W6 — Per-operand LDS/L2 candidate generator (the calculator)
New `boltbeam/lds_l2.py`. For a role's access-stream + workload point, encode the crossover (§7),
call W5 for both variants, and emit **two candidates** (`operand=X: lds_staged`,
`operand=X: cache_streamed`) into the search space, each with:
- predicted-cycle interval, the assumed workload point, the deciding term (which condition fired),
  and truth-status inherited from the weakest input fact.
- A `recommendation` field = cheaper predicted variant, explicitly marked **prediction, not
  verdict** (promotion requires W7).
- Guardrails: never recommend LDS if tile > `lds_bytes_per_cu`, or if the occupancy penalty from
  `derive_occupancy` erases the modeled gain, or if `reuse ≤ 1` (decode weights → cache is
  correct by construction; emit `cache_streamed` only). MoE/hybrid roles scoped like any other
  operand (arch-scoped families, per README).

### W7 — A/B confirm (reuse existing loop)
- The two W6 candidates become a normal `trace_request` (via `runner_plan`); the runner times
  both; `ingest-timing` → `evaluate` promotes the winner / refutes the loser. `reopen-check`
  already handles "re-time a previously-refuted staging choice under a new workload point."
- The confirmed winner + its workload point is what actually lands in `route_policy.seed.json`.

### W8 — CLI surface
Add subparsers + `cmd_*` (mirroring `cmd_roofline_plan` wiring, `main()` `sub.add_parser`):
- `boltbeam mem-sweep-request --target amd_gfx1100 --run runs/X` (W1)
- `boltbeam ingest-mem-sweep ws_bandwidth_samples.json --run runs/X` (W3)
- `boltbeam lds-l2 --run runs/X --role ffn_down --ctx 512 --batch 1` (W6; prints the per-operand
  prediction + condition that fired)

### W9 — Report/output integration
- `boltbeam output` includes `memory_hierarchy_profile.json` + any LDS/L2 candidates and their
  promote/refute state in the bundle + report, with the truth-status and the
  `blocked_gfx11_pmc` honesty note surfaced (not hidden).

---

## 7. The crossover model (what W6 encodes)

For one access-stream of an operand at a workload point, define: working set `W` (bytes touched in
the reuse window), reuse `R` (reads per byte by the workgroup), tier bandwidths `B_l2/B_mall/B_dram`
and `B_lds`, LDS capacity `C_lds = lds_bytes_per_cu`, and occupancy factor `ρ = occ_staged/occ_streamed ≤ 1`.

- **Cache-streamed cost** (rely on L2/MALL): served at the bandwidth of the smallest tier that
  holds the *concurrently-live* working set `W*` (summed across resident workgroups):
  `T_stream ≈ (R · traffic) / B(W*)`, where `B(W*) = B_l2 if W* ≤ l2.bytes else B_mall if W* ≤
  mall.bytes else B_dram`. Reuse is captured only to the extent the tier holds `W*`.
- **LDS-staged cost**: one DRAM fill of the tile + reuse from LDS + sync + occupancy tax:
  `T_lds ≈ W/B_dram + (R · traffic)/B_lds + t_barrier + occupancy_penalty(ρ)`.
- **Stage LDS iff** `T_lds < T_stream` **and** `W ≤ C_lds` **and** `ρ`-loss doesn't erase the gain.
  Intuition: LDS wins when the cache tier serving `W*` is slow (capacity thrash to DRAM) **or** the
  reuse bandwidth demand exceeds the cache tier — the two dominant conditions. `R ≤ 1` ⇒ never
  stage (decode weights).

The model is interval-valued (soft MALL bands → wide `B(W*)` near transitions) and returns *which
condition fired*, so the recommendation is explainable, not a black-box threshold.

---

## 8. Truth-status & guardrails (the honesty spine)

- Timed plateau bandwidths → **MEASURED**. Capacity bands → **DERIVED** + interval. L2 hit-rate →
  **UNKNOWN (`blocked_gfx11_pmc`)**, recorded explicitly. Spec priors → **ASSUMED/IMPORTED**.
- W6 recommendation inherits the **weakest** input truth-status; a recommendation built on ASSUMED
  tiers is labelled as such and cannot promote.
- **Prediction ≠ promotion.** W5/W6 prune; only W7 (real A/B timing) promotes. This is the same
  discipline as `evaluate`/`reopen-check` and the roofline ceiling's reference-floor guard.
- **Fail closed** everywhere: missing working-set/reuse inputs, <2 detected plateaus, or a stale
  `system_snapshot_id` → blocker + refusal, never a guessed number.
- No model-name / role special-cases in the decision path (policy-guard style); the crossover reads
  only facts + workload point.

---

## 9. Test plan

- **Unit (proving grounds, not authority):** synthetic size→GB/s curves with known plateaus →
  assert W3 recovers the right bands + truth-status; degenerate curves (1 plateau, noisy) → assert
  blockers. Crossover unit: hand-built (W, R, tiers) → assert the expected condition fires and the
  guardrails (tile>LDS, R≤1, occupancy erasure) veto correctly.
- **Contract:** `ws_bandwidth_samples.json` missing a field → `AdapterIncomplete`, not a crash.
- **Real-machine gate (mandatory for any promotion):** run W1→W3 on the gfx1100, sanity-check the
  DRAM plateau against the known ~960 GB/s / `roofline_ceiling` sustained number; then W6→W7 on a
  known operand (e.g. `ffn_down` GEMV decode) and confirm it recommends `cache_streamed` (R≈1) and
  that A/B agrees. A prefill GEMM operand with real reuse is the positive LDS case.

---

## 10. Risks / known-soft-spots (state them, don't hide them)

- **Adaptive MALL + prefetch** blur the capacity knees → bands, not bytes. We ship intervals and
  DERIVED status; we do **not** claim exact cache sizes.
- **Blocked L2 counters** mean we can never *attribute* a win to "L2 hits" — only *observe* the
  end-to-end delta. Any "why" is inference, labelled as such.
- **Measurement noise / clock ramp** can flip a close A/B; W7 must use medians + the intervals, and
  a near-tie is reported as a tie (no false precision).
- **Sweep cost:** ~28 sizes × (cold+sustained+repeats) is a few seconds of runner time — cheap,
  run once per machine snapshot, cached by `system_snapshot_id`.

## 11. Sequencing (minimal path to first real verdict)

1. **W0 + W1 + W2-contract + W3** → recover the tiers, land `memory_hierarchy_profile.json`.
   *(This alone is independently useful: it's the missing hardware facts, measured.)*
2. **W5 + W6** → offline per-operand prediction (the "calculator" the request asked for).
3. **W7 + W8 + W9** → confirmation loop + CLI + report.
4. **W4** anytime (spec priors are a nicety, not on the critical path).

## 12. Out of scope / deferred

- NVIDIA/Metal tiers (targets are `descriptor_only`); the schema is target-generic but only
  gfx1100 is populated.
- Unblocking gfx11 PMC (`blocked_gfx11_pmc`) — if those counters ever open, W3 gains a direct
  hit-rate cross-check, but the timing path stands on its own and is the authority regardless.
- Multi-operand joint LDS budgeting (two operands competing for the same 64 KB) — v1 is
  per-operand; joint allocation is a follow-up once single-operand verdicts are trusted.

---

# Appendix A — Agent Task Cards (executable)

This appendix is the **build instruction for implementing agents** and supersedes the prose in §6
where they differ. Each card is self-contained: exact file, exact signatures, exact JSON shapes,
and a copy-pasteable acceptance test. Do only your card. Do **not** run `git` — the lead commits.

## Ground rules (every card)
- Python, no type-annotation syntax that won't parse under the repo's style; match surrounding code.
- `from __future__ import annotations` at top of every new module.
- Every emitted dict has a `"schema"` key from `boltbeam/vocab.py` (constants already added: W0).
- **Fail closed**: missing/short inputs → return a dict with a non-empty `"blockers": [...]` list and
  emit no numbers; never guess. Raise `ValueError` only for programmer error, not data gaps.
- Put tests in `tests/` mirroring existing test files; run **only your file's tests** to verify:
  `python -m pytest tests/test_<yourfile>.py -q` must pass.
- Do not edit files owned by another card (see "Files" — they are disjoint by design).

## Mechanical patterns (verified 2026-07-12 — copy these)
- **Fact** (`from boltbeam.core.facts import Fact`; `from boltbeam.artifacts.base import EvidenceSource`):
  - MEASURED: `Fact(name, value, "measured", unit=..., sources=(EvidenceSource("boltbeam", tool, "", fingerprint),))`
    — MEASURED **requires** ≥1 source and **forbids** `derivation`/`input_fact_ids`.
  - DERIVED: `Fact(name, value, "derived", unit=..., derivation="<how>", input_fact_ids=(...), uncertainty={"low":lo,"high":hi})`
    — DERIVED **requires** `derivation` **and** (sources or input_fact_ids).
  - UNKNOWN: `Fact(name, None, "unknown", derivation="blocked_gfx11_pmc")` — value **must** be None.
  - `name` **must** contain a `.` (namespaced) or `__post_init__` raises. Use `f.to_json()` to serialize.
- **Interval** (`from boltbeam.perf.calibration import Interval`): `Interval(low, median, high)`, requires
  `0 <= low <= median <= high`; `.to_json()` → `{"low","median","high"}`.
- **CLI wiring** (`boltbeam/cli.py`): add `def cmd_x(args) -> int:` that calls `_write(result_dict, args.out)`
  and `return 0`; load inputs with `_load_json(path)`. In `main()` add
  `p = sub.add_parser("name", help="..."); p.add_argument("--foo"); p.add_argument("--out", default=None); p.set_defaults(fn=cmd_x)`.
  Dispatch is automatic via `args.fn`.
- **Fact names** to use verbatim: `mem.tier.l2.bytes`, `mem.tier.mall.bytes`, `mem.tier.l2.bandwidth_gbs`,
  `mem.tier.mall.bandwidth_gbs`, `mem.tier.dram.bandwidth_gbs`, `mem.tier.lds.bandwidth_gbs`,
  `mem.l2.hit_rate` (UNKNOWN).

---

## TC-1 — Working-set sweep request  ·  Wave 1  ·  no deps
**Files:** create `boltbeam/mem_sweep.py`, create `tests/test_mem_sweep.py`, edit `boltbeam/runner_plan.py`
(add `"ws_sweep_request.json"` to the `_OPTIONAL_INPUTS` tuple — one line, nothing else).
**Interface:**
```python
def build_ws_sweep_request(target_id: str, sizes: list[int] | None = None) -> dict
```
- If `sizes` is None, use a geometric ladder 4096 → 268435456 bytes at ~√2 spacing (compute with a loop
  doubling every 2 steps; ~28 entries). Round each to an int.
- Return:
```python
{"schema": SCHEMA_WS_SWEEP_REQUEST,           # from boltbeam.vocab
 "target_id": target_id,
 "note": "sustained+cold streaming copy per size to locate cache tiers; plus one lds_resident loop",
 "points": [{"bytes": n, "kind": "copy", "repeats": 5, "sustained": True, "cold": True} for n in sizes],
 "lds_probe": {"kind": "lds_resident", "bytes": 32768, "reuse": 256, "repeats": 5}}
```
- No hardware, no execution. Pure function of inputs.
**Acceptance test:** default call returns ≥20 points, all `bytes` ints strictly increasing, smallest ≤ 4096,
largest ≥ 2**28, schema equals `SCHEMA_WS_SWEEP_REQUEST`, and `"ws_sweep_request.json"` is present in
`boltbeam.runner_plan._OPTIONAL_INPUTS`.

## TC-3 — Sweep ingest + band classifier  ·  Wave 1  ·  no deps
**Files:** create `boltbeam/mem_hierarchy.py`, create `tests/test_mem_hierarchy.py`.
**Interface:**
```python
def classify_ws_sweep(samples: dict, *, system_snapshot_id: str, tool: str = "tinygrad_ws_sweep") -> dict
```
- `samples` shape = the runner output: `{"schema": SCHEMA_WS_SWEEP_SAMPLES, "system_snapshot_id": str,
  "points": [{"bytes": int, "cold_gbs": [..], "sustained_gbs": [..], "kind": "copy"}, ...],
  "lds_probe": {"sustained_gbs": [..]}}`.
- For each `copy` point call
  `boltbeam.roofline_ceiling.resolve_achieved_ceiling(cold_samples=p["cold_gbs"], sustained_samples=p["sustained_gbs"])`
  and take `.achieved_gbs` as that size's bandwidth. Build the (bytes → gbs) curve sorted by bytes.
- **Band detection (simple + deterministic):** walk the curve; a "plateau" is a maximal run where
  consecutive gbs stay within ±8% of the run's median. Keep plateaus with ≥2 points. Expect
  descending plateaus = [L2, MALL, DRAM] (top-bandwidth first). If fewer than 2 plateaus → return
  `{"schema": ..., "blockers": ["sweep-underspecified: <n> plateaus"], "facts": []}`.
- Emit **Facts** (`.to_json()` each):
  - `mem.tier.l2.bandwidth_gbs`, `mem.tier.mall.bandwidth_gbs`, `mem.tier.dram.bandwidth_gbs` from the
    plateau medians (highest-bw plateau = l2; lowest = dram; middle = mall if 3 plateaus, else omit mall)
    → status `"measured"`, unit `"GB/s"`, one `EvidenceSource("boltbeam", tool, "", system_snapshot_id)`.
  - `mem.tier.l2.bytes`, `mem.tier.mall.bytes` from the byte-midpoint between adjacent plateaus (geometric
    mean of the last point of the faster plateau and first point of the slower) → status `"derived"`,
    unit `"bytes"`, `derivation="sweep transition-band midpoint"`, `input_fact_ids=()`,
    `uncertainty={"low": <faster plateau last bytes>, "high": <slower plateau first bytes>}`, and pass a
    dummy source too (derived allows sources). Use the bandwidth fact ids as `input_fact_ids` if easy;
    otherwise attach `sources=(EvidenceSource("boltbeam", tool, "", system_snapshot_id),)`.
  - `mem.tier.lds.bandwidth_gbs` from `samples["lds_probe"]["sustained_gbs"]` median → `"measured"`.
  - `mem.l2.hit_rate` → `Fact("mem.l2.hit_rate", None, "unknown", derivation="blocked_gfx11_pmc")`.
- Return `{"schema": SCHEMA_MEMORY_HIERARCHY_PROFILE, "system_snapshot_id": system_snapshot_id,
  "facts": [f.to_json() for f in facts], "blockers": []}`.
**Acceptance test:** feed a synthetic 3-plateau curve (e.g. 900→900→... then 500→500 then 250→250 GB/s
across increasing sizes) + an lds_probe of [2600,2650]; assert 7 facts emitted (l2/mall/dram bandwidth +
l2/mall bytes + lds bandwidth + hit_rate), l2 bw ≈ top plateau,
dram bw ≈ bottom plateau, `mem.l2.hit_rate` has status "unknown"/value None, `blockers == []`. Feed a
flat single-plateau curve → assert non-empty `blockers` and `facts == []`.

## TC-5 — Tier crossover primitives  ·  Wave 1  ·  no deps
**Files:** create `boltbeam/perf/mem_tier.py`, create `tests/test_mem_tier.py`. (Does NOT touch cycle_model.)
**Interface (all pure functions):**
```python
def effective_bandwidth_gbs(working_set_bytes: float, tiers: dict) -> float
def stream_cost_s(traffic_bytes: float, reuse: float, working_set_bytes: float, tiers: dict) -> float
def lds_cost_s(tile_bytes: float, traffic_bytes: float, reuse: float, tiers: dict,
               *, barrier_s: float = 0.0, occupancy_factor: float = 1.0) -> float
def decide(*, tile_bytes: float, traffic_bytes: float, reuse: float, tiers: dict,
           lds_capacity_bytes: float, barrier_s: float = 0.0, occupancy_factor: float = 1.0) -> dict
```
- `tiers` = `{"l2_bytes","mall_bytes","l2_gbs","mall_gbs","dram_gbs","lds_gbs"}` (floats; mall_* optional/None).
- `effective_bandwidth_gbs`: return l2_gbs if ws ≤ l2_bytes; elif mall present and ws ≤ mall_bytes → mall_gbs;
  else dram_gbs. (GB/s = 1e9 bytes/s.)
- `stream_cost_s = (reuse * traffic_bytes) / (effective_bandwidth_gbs(ws) * 1e9)`.
- `lds_cost_s = tile_bytes/(dram_gbs*1e9) + (reuse*traffic_bytes)/(lds_gbs*1e9) + barrier_s`, then divide the
  reuse term's benefit by `occupancy_factor` (occupancy_factor ≤ 1 makes LDS worse): implement as
  `lds_reuse_term / occupancy_factor`.
- `decide`: compute both; `stage_lds = (lds < stream) and (tile_bytes <= lds_capacity_bytes) and (reuse > 1)`.
  Return `{"schema": SCHEMA_LDS_L2_CANDIDATE, "recommendation": "lds_staged" if stage_lds else "cache_streamed",
  "stream_cost_s": stream, "lds_cost_s": lds, "reason": "<which condition decided>", "prediction_only": True}`.
  `reason` must name the deciding factor: `"reuse<=1"`, `"tile_exceeds_lds"`, `"lds_faster"`, or `"cache_faster"`.
**Acceptance test:** (a) reuse=1 → always `cache_streamed`, reason `"reuse<=1"`. (b) tile 128KB > 64KB cap →
`cache_streamed`, reason `"tile_exceeds_lds"`. (c) big reuse, small tile, ws that lands in DRAM tier vs lds_gbs
much higher → `lds_staged`, reason `"lds_faster"`. (d) `effective_bandwidth_gbs` returns the right tier at each
boundary.

## TC-6 — Per-operand calculator  ·  Wave 2  ·  needs TC-3 + TC-5
**Files:** create `boltbeam/lds_l2.py`, create `tests/test_lds_l2.py`.
**Interface:**
```python
def tiers_from_profile(profile: dict) -> dict          # memory_hierarchy_profile.json -> tiers dict for mem_tier
def lds_l2_for_operand(profile: dict, *, tile_bytes: float, traffic_bytes: float, reuse: float,
                       lds_capacity_bytes: float = 65536.0, role: str = "") -> dict
```
- `tiers_from_profile`: read the facts list; map fact `name`→`value` for the 6 `mem.tier.*` names into the
  `tiers` dict TC-5 expects (`mem.tier.l2.bytes`→`l2_bytes`, etc.); mall_* → None if absent. If a required
  bw fact (l2/dram/lds) is missing → return `{"blockers": ["missing tier fact: <name>"]}`.
- `lds_l2_for_operand`: call `tiers_from_profile`; if it has blockers, propagate them (fail closed). Else call
  `boltbeam.perf.mem_tier.decide(...)`, attach `"role": role`, `"tile_bytes"`, `"reuse"`, and the
  weakest input truth-status (scan the profile facts used; if any is `"derived"`/`"assumed"`, set
  `"truth_status": <weakest>`, else `"measured"`). Return that dict.
**Acceptance test:** build a `memory_hierarchy_profile`-shaped dict (reuse TC-3's output or hand-write facts);
assert an R=1 operand → `cache_streamed`; assert a high-reuse small-tile operand → `lds_staged`; assert a
profile missing `mem.tier.dram.bandwidth_gbs` → non-empty blockers.

## TC-8 — CLI surface  ·  Wave 3  ·  needs TC-1 + TC-3 + TC-6
**Files:** edit `boltbeam/cli.py` only (add 3 `cmd_*` fns + 3 `add_parser` blocks), create `tests/test_cli_lds_l2.py`.
- `boltbeam mem-sweep-request --target <id> [--out F]` → `_write(build_ws_sweep_request(args.target), args.out)`.
- `boltbeam ingest-mem-sweep <samples.json> --snapshot <id> [--out F]` →
  `_write(classify_ws_sweep(_load_json(args.samples), system_snapshot_id=args.snapshot), args.out)`.
- `boltbeam lds-l2 --profile <memory_hierarchy_profile.json> --tile-bytes N --traffic-bytes N --reuse R
  [--role NAME] [--out F]` → `_write(lds_l2_for_operand(_load_json(args.profile), tile_bytes=args.tile_bytes,
  traffic_bytes=args.traffic_bytes, reuse=args.reuse, role=args.role or ""), args.out)`. Use
  `type=float`/`type=int` on the numeric args.
**Acceptance test:** call `main(["mem-sweep-request","--target","amd_gfx1100"])` returns 0; call the ingest and
lds-l2 commands with tmp files written from TC-1/TC-3 outputs and assert exit 0 and a schema key in the output.

## Sequencing for the lead
- **Wave 1 (parallel):** TC-1, TC-3, TC-5. **Wave 2:** TC-6. **Wave 3:** TC-8.
- After all cards: run full `python -m pytest -q`; then commit with `[schema]`/`[cli]`/`[eval]` prefixes
  (commit-msg hook requires a subsystem prefix) and push.
