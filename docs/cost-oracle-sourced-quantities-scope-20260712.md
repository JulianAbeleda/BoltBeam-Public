# Cost Oracle: Sourced-Quantity Route Decision — Scope

Date: 2026-07-12

## Objective

Make BoltBeam's route/schedule decision (e.g. LDS-staged vs register-resident per
role) a **sourced-quantity decision**: every quantity the decision needs is pulled
from the source that is *authoritative and cheapest* for that quantity — computed
where computable, measured where not — and the decision escalates to a more
expensive source only when the cheaper one is not decisive.

This does not make BoltBeam run kernels. BoltBeam owns the decision and the
evidence classification; tinygrad remains the compiler/runtime that produces
compile and timing artifacts BoltBeam consumes.

## The principle

"Static vs dynamic" is a false choice. Both are **sources for the same quantities.**
The discipline is a *sourcing map*: route each quantity to its authoritative source,
and never model what you can measure or measure what you can compute.

Refine "two sources" into a **three-tier cost ladder** (climb only as far as the
decision requires):

| tier | source | needs | example quantities | cost |
|---|---|---|---|---|
| T1 | **analytical** | shapes + schedule params only | arithmetic intensity, bytes moved, working-set, tile geometry, L2-fit | free |
| T2 | **compile-derived** | compile the kernel (no run) | VGPR, LDS bytes, spills, ISA inst/wait/ds_load counts → **occupancy** | one compile |
| T3 | **run-derived** | execute on device (offline) | achieved throughput, latency-hiding efficiency, cache hit-rate | one pinned run |

## Quantity → authoritative source map

| quantity | source tier | provider | error | hard-constraint? |
|---|---|---|---|---|
| arithmetic intensity (FLOP/byte) | T1 | roofline | exact | — |
| bytes moved / global traffic | T1 | roofline | exact | — |
| working-set vs L2 fit | T1 | new (needs L2 size) | exact | — |
| occupancy (waves/SIMD) | T2 | new occupancy calc | exact | **yes** (min threshold) |
| register spills / scratch | T2 | compile-evidence | exact | **yes** (must be 0) |
| inst/WMMA, waits/WMMA, ds_load/WMMA | T2 | ISA static counts | exact | — |
| latency-hiding efficiency | T3 | timing | ±20–50% if modeled | — |
| cache hit-rate | T3 | PMC counters | measured | — |
| achieved throughput (tok/s, TFLOPS) | T3 | timing authority | measured | — (final tiebreak) |

Model-only quantities (latency hiding, throughput) must **not** be predicted from a
T1/T2 model as if authoritative — that is exactly the silent-misprediction failure
(uniform-LDS mistake). They are T3 or nothing.

## What exists vs what is needed

Already in BoltBeam:
- **Analytical (T1):** `roofline_ceiling.py`, `roofline_plan.py`, `prefill_roofline*.py`.
- **Measurement (T3) ingestion:** `timing.py`, `timing_compare.py`, `substrate_compare.py`,
  `hw_trace.py`, `ingest-timing`; PMC counter registry in `targets.json`
  (`GL2C_HIT/MISS`, `SQ_BUSY_CYCLES`, `bytes_moved`).
- **Cross-check:** `reconciliation.py`.
- **HW registry:** `targets.json` (has `lds_bytes_per_cu: 65536`, `compute_units: 96`,
  `memory_bandwidth_gbs: 960`, `peak_tflops`, `wave_size: 32`).

Gaps to fill:
1. **T2 compile-evidence ingestion** — a classifier that consumes tinygrad's compiled
   resource artifact (VGPR/SGPR/LDS/spills/ISA counts; produced by the
   `mmq_compile_evidence`-style path) keyed by exact candidate identity. Analogous to
   `ingest-probe`/`ingest-timing` but for the compile tier.
2. **Occupancy calculator (T1/T2)** — `waves = min(vgpr_file/vgpr_per_wave capped at
   max_waves, lds_per_cu/lds_per_wg, ...)`. Pure function of compile-evidence + target
   constants.
3. **`targets.json` constants** — add the occupancy-model inputs currently missing:
   `vgpr_per_simd`, `max_waves_per_simd`, `simds_per_cu`, and `l2_bytes` (+ Infinity
   Cache) for L2-fit. gfx1100: vgpr_per_simd=1536, max_waves_per_simd=16, simds_per_cu=2,
   l2_bytes≈6 MiB.
4. **Cost-quantity schema** — the sourcing map as data (`data/cost_quantities.json`):
   each quantity → {tier, provider, error_bound, hard_constraint}. This makes the map
   auditable and target-portable, not hardcoded.

## Components to build

### C1. Quantity schema + providers
`data/cost_quantities.json` (the map above) + a provider registry: each quantity name
resolves to a callable that returns `(value, source_tier, error)`. T1 providers are
pure functions; T2/T3 providers read ingested artifacts.

### C2. Occupancy + L2-fit analytical providers
`boltbeam/occupancy.py`: occupancy from (VGPR, LDS, target) and working-set-vs-L2 from
(shapes, tile, target). Both exact given inputs.

### C3. Compile-evidence ingestion
`boltbeam/compile_evidence.py` + `ingest-compile` CLI: classify tinygrad's resource
artifact into the T2 quantities, keyed by candidate identity (reuse the candidate-set
identity model).

### C4. The decision oracle
`boltbeam/cost_oracle.py`: for a candidate set,
1. **Prune (T1+T2 only):** drop candidates failing hard constraints (spills>0,
   occupancy<min, bandwidth-bound below a floor). Cheap — no runs.
2. **Rank (T1+T2):** score survivors by the analytical/compile model.
3. **Escalate (T3) only at ties:** where two survivors are within the *combined error
   budget* of the cheaper tiers, require a measured throughput to break the tie. Only
   these candidates cost a run.
Output: a decision per role with the winning candidate + the quantity vector + the
source and confidence of each quantity.

### C5. Calibration ledger (cross-check)
`boltbeam/calibration.py` (extends `reconciliation.py`): where a quantity is available
from both a computed and a measured source, record `measured/computed`. Agreement
validates the model; disagreement localizes the missing term and is surfaced as the
next model-improvement target. Over time this shrinks the set of quantities that must
be measured (the model earns trust per-regime).

## Decision algorithm (summary)

```
for role in roles:
  cands = candidate_schedules(role)             # e.g. {lds, register}
  q = {c: quantities(c, up_to_tier=T2) for c in cands}   # analytical + compile, cheap
  cands = [c for c in cands if passes_hard_constraints(q[c])]   # prune (spills/occupancy)
  ranked = sort(cands, key=analytic_cost(q))
  if separation(ranked[0], ranked[1]) > error_budget(T2):
     pick ranked[0]                              # math decided it; no run
  else:
     measure(ranked[:k]); pick by measured throughput   # escalate to T3 only here
  record decision + per-quantity source + calibration deltas
```

## Output contract

Extend `route_policy.seed.json`: each role's chosen schedule carries a
`decision_provenance` block — the quantity vector, the source tier of each quantity,
which quantities were decisive, whether T3 was needed, and the calibration deltas.
tinygrad consumes the policy and executes; it never re-decides.

## Division of labor (unchanged)

- **BoltBeam:** owns the sourcing map, the analytical providers, the decision oracle,
  and evidence classification. Computes T1 itself; consumes T2/T3 artifacts.
- **tinygrad:** compiles (produces T2 resource artifacts) and runs (produces T3 timing),
  and is the sole legality/runtime authority. BoltBeam does not compile or run.

## Completion

Complete when: (1) LDS-vs-register per-role decisions are produced by the oracle with
each quantity sourced per the map; (2) hard-constraint pruning and analytic ranking run
with **zero** device runs; (3) T3 runs occur only for candidates within the error budget
of the cheaper tiers, and the decision records which; (4) the calibration ledger reports
computed-vs-measured deltas per quantity, and at least occupancy is validated
(computed == PMC-measured) on one real capture; (5) the emitted route policy carries full
per-quantity decision provenance.

Not complete if any T3-only quantity (throughput, latency-hiding) is modeled and treated
as authoritative, or if a run is spent on a decision the cheaper tiers already settled.
