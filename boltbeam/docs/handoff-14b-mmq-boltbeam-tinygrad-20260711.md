# 14B MMQ Pause Handoff: BoltBeam Side

Date: 2026-07-11

BoltBeam repo: `/home/ubuntu/BoltBeam/boltbeam`

BoltBeam head at pause:

```text
c8b87cc [search] join mmq r4 evidence
```

Tinygrad repo: `/home/ubuntu/tinygrad-arkey`

Tinygrad head pushed to master:

```text
2c7a4eb45 [mmq] add bounded coop numeric atom
```

## Mission

BoltBeam's job is to turn the 14B MMQ llama-kernel knowledge into structured
answers that can drive tinygrad atom reductions.

The project is not blocked on knowing that llama has a fast kernel. We have the
source and an oracle. The active gap is this:

```text
llama kernel structure is known
tinygrad emitted coop atom is correct but slow
BoltBeam must explain which structural facts are missing or wrong in the
tinygrad atom, using evidence rather than guesses
```

## Current Tinygrad State

Tinygrad now has an emitted bounded coop numeric atom:

```text
backend: q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0
shape:   M=16, N=16, K=256
layout:  mmq_ds4
role:    ffn_gate_up
```

Tinygrad status:

```text
numeric correctness: PASS
production dispatch: unchanged
default route:       direct_packed
promotion:           blocked
```

Important caveat:

```text
store_owner metadata is not attached to the emitted Tensor custom_kernel graph.
R4 owner proof remains a separate lowered AMD ISA trace.
```

## Current R5 Evidence

Command used in tinygrad:

```text
PYTHONPATH=. python3 extra/qk/mmq_machine_search.py \
  --r5-geometry-search --run --warmups 0 --rounds 1 --out /tmp/r5.json
```

Observed result:

```text
status:            PASS_NON_PROMOTABLE
promotion_verdict: NO_PROMOTION_WITHOUT_BOUNDED_COOP_WIN
blocker:           no emitted cooperative MMQ tile candidate has a bounded
                   same-session win
```

Measured candidate facts:

| candidate | status | speedup vs direct_packed | note |
|---|---:|---:|---|
| `r5_llama_coop_oracle_16x16` | PASS | 34.19 | oracle only, not promotable |
| `r5_ds4_warp_4x5` | PASS | 1.40 | existing direct atom |
| `r5_ds4_dot4x4_8x7` | PASS | 1.25 | existing direct atom |
| `r5_ds4_coop_tile_16x16` | PASS | 0.38 | emitted coop atom, too slow |
| `r5_ds4_lds_skeleton_4x5` | PASS | 0.18 | structural skeleton |

BoltBeam should treat this as:

```text
correctness barrier crossed
performance barrier not crossed
promotion still illegal
```

## Existing BoltBeam Surfaces

Relevant files:

```text
search/research_categories.py
search/r4_evidence_join.py
search/epoch_join.py
search/resource_join.py
search/timing_join.py
search/mmq_epoch_oracle.py
search/epoch_model.py
```

The 18 research categories already exist in `search/research_categories.py`.
For this MMQ phase, BoltBeam must be able to answer each category for the
emitted coop atom, not just for the llama oracle.

Priority categories for the next step:

| category | why it matters now |
|---|---|
| `work_ownership` | prove or explain the gap between separate R4 owner proof and emitted numeric graph |
| `store_path` | current atom likely loses through naive 256 gated stores |
| `resource_model` | need VGPR/SGPR/LDS/scratch/occupancy for the emitted atom |
| `sync_cadence` | need waitcnt/barrier cost and visibility proof |
| `k_loop_cadence` | need compare tinygrad K loop against llama MMQ cadence |
| `accumulator_mapping` | need know whether accumulator identity matches llama-style fragments |
| `baseline_comparator` | same-session direct_packed is the promotion denominator |
| `promotion_gates` | R6 must stay blocked until same-session emitted coop win |
| `stop_criteria` | stop only on a real lowering/resource wall |

## Required BoltBeam Answers

For `q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0`, BoltBeam should produce or join
evidence answering:

```text
1. Correctness contract:
   M=16, N=16, K=256, Q4_K weights, Q8_1 DS4 activations, fp32 output,
   tolerance 1e-3.

2. Numeric oracle:
   tinygrad DS4 reference and llama oracle are comparators; llama oracle is not
   a route candidate.

3. Data layout:
   Q4_K block format, DS4 q8 value/scale/sum indexing, LDS local index mapping.

4. Tile geometry:
   emitted atom's workgroup/wave/lane geometry vs llama MMQ geometry.

5. Work ownership:
   emitted store owners, duplicate/missing stores, owner proof join status.

6. Accumulator mapping:
   lane/register/fragment/output identity for the current atom.

7. Shared memory lifecycle:
   q8 DS4 stage events, LDS bytes, barriers, reuse count.

8. K-loop cadence:
   panel step, reload count, reuse per load, K carry behavior.

9. Dot primitive:
   instruction family actually emitted and whether it matches expected dot/MMQ
   structure.

10. Resource model:
    VGPR, SGPR, LDS, scratch, occupancy, waves per CU.

11. Sync cadence:
    barriers, waitcnts, dependency edges, unnecessary waits.

12. Store path:
    final store count, coalescing, vectorization, gated-store overhead.

13. Baseline comparator:
    same-session direct_packed timing and llama oracle timing.

14. Roofline target:
    whether current emitted atom is memory, store, sync, occupancy, or compute
    limited.

15. Search-space knobs:
    legal writeback, tiling, staging, unroll, and placement knobs that preserve
    semantics.

16. Route binding rules:
    Q4_K + ffn_gate_up + 14B shape facts, not model-name branches.

17. Promotion gates:
    correctness PASS, same-session speed win, no production dispatch change
    before gate, rollback path.

18. Stop criteria:
    exact missing primitive/API/resource wall if tinygrad cannot express the
    needed structure.
```

## Tinygrad Artifacts BoltBeam Should Consume

From tinygrad:

```text
extra/qk/mmq_machine_search.py --r5-geometry-search --run
extra/qk/mmq_machine_search.py --boltbeam-oracle-trace
build_r4_evidence_artifacts()
coop_tile_blocked_translation_evidence()
```

Useful schemas already emitted or joinable:

```text
q4k-q8-1-mmq-r5-geometry-search.v1
q4k-q8-1-mmq-machine-search.v1
boltbeam.hw_trace.v1
tinygrad.mmq_owner_coverage.v1
tinygrad.mmq_staging_evidence.v1
tinygrad.kernel_resource_trace.v1
```

## What Not To Do

Do not mark the llama oracle as a promotable route.

Do not treat numeric PASS as route readiness.

Do not let model names like 8B/14B/32B become dispatch logic. Model profiles
are data; route selection should depend on quant, role, shape, layout, and
resource facts.

Do not add a third parallel authority. Tinygrad's route/harness facts and
BoltBeam's joins must agree on candidate ids, backend ids, and promotion gates.

## Next BoltBeam Work

1. Ingest the current R5 report and classify the emitted coop atom across the
   18 research categories.
2. Join R4 owner evidence with R5 timing/correctness and resource traces.
3. Produce a concise diagnosis for why `r5_ds4_coop_tile_16x16` is slower than
   direct_packed.
4. Emit the next tinygrad atom-edit recommendation as a structural delta, for
   example:

```text
replace 256 gated stores with owner-shaped writeback
change LDS layout/reuse cadence
change accumulator fragment mapping
remove unnecessary barrier/wait path
reduce VGPR pressure or scratch
```

5. Stop only if the evidence shows a real hard wall in tinygrad lowering or the
   current GPU substrate.

## Resume Command Sketch

Tinygrad should produce fresh artifacts:

```text
cd /home/ubuntu/tinygrad-arkey
PYTHONPATH=. python3 extra/qk/mmq_machine_search.py \
  --r5-geometry-search --run --warmups 0 --rounds 1 \
  --out bench/prefill-14b-mmq-machine-search/r5-current.json

PYTHONPATH=. python3 extra/qk/mmq_machine_search.py \
  --boltbeam-oracle-trace \
  --out bench/prefill-14b-mmq-machine-search/boltbeam-oracle-trace.json
```

BoltBeam should then join and report:

```text
cd /home/ubuntu/BoltBeam/boltbeam
python3 -m pytest -q
```

If no CLI exists yet for this exact join, the next BoltBeam task is to add that
thin CLI around the existing pure join modules rather than manually inspecting
JSON.
