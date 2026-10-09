# MMQ Writeback Closed-Loop Result

Date: 2026-07-11

## Implementation

BoltBeam commits:

```text
bb79cd3 [docs] scope MMQ closed-system loop
e8cf0f6 [schema] add truth and experiment contracts
7a1905e [profile] capture exact execution snapshot
4a37400 [search] validate MMQ experiment bundles
bea0a4a [search] drive and diagnose MMQ writeback search
```

Tinygrad commits:

```text
0aa4c8485 [mmq] add canonical writeback candidates
725c02521 [mmq] emit atomic experiment bundles
e6f97442f [mmq] align experiment evidence contract
```

## Closed System

Captured snapshot:

```text
system_snapshot_id: sha256:dd9459855728450ea37dd31a5deb03ee9a023b5dee08af91a487bae4eace9ed2
```

The snapshot preserves the live gfx1100 XTX/GRE marketing-identity conflict as
measured facts instead of silently selecting one label.

## Controlled Candidates

```text
mmq.wb.gated_matrix.m16.n16.k256.v1
mmq.wb.direct_owner.m16.n16.k256.v1
```

Both candidates use backend
`q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0`. Numeric body, 16x16x256
geometry, DS4 staging, synchronization, K-loop identity, generated inputs, and
same-session `direct_packed` comparator policy are held constant. Only writeback
mode changes.

Protocol per bundle:

```text
warmups: 3
measured rounds: 10
correctness atol: 1e-3
production dispatch changed: false
```

## Results

First order: gated, then direct owner.

| candidate | median ms | min ms | comparator median ms | speedup vs comparator | max abs error |
|---|---:|---:|---:|---:|---:|
| gated matrix | 17.9549 | 17.4889 | 8.9808 | 0.5002 | 0.00012207 |
| direct owner | 7.0178 | 6.8288 | 8.7519 | 1.2471 | 0.00012207 |

Reverse order: direct owner, then gated.

| candidate | median ms | min ms | comparator median ms | speedup vs comparator | max abs error |
|---|---:|---:|---:|---:|---:|
| direct owner | 6.9845 | 6.8182 | 8.9972 | 1.2882 | 0.00012207 |
| gated matrix | 18.2655 | 18.0459 | 8.9271 | 0.4887 | 0.00012207 |

The direction and approximate magnitude survive reversal. Direct-owner
writeback is about 2.56-2.62x faster than gated writeback in the bounded atom and
beats its same-session `direct_packed` comparator in both runs.

## BoltBeam Verdict

The experimental result strongly supports the immediate writeback hypothesis.
The formal `boltbeam.mmq_diagnosis.v1` conclusion remains `inconclusive` because
the real bundles are intentionally `INCOMPLETE_EVIDENCE`.

Exact missing evidence:

```text
actual compiled binary SHA-256
VGPR count
SGPR count
LDS bytes
scratch bytes
workgroup resource evidence
actual ISA-derived store-count evidence
```

The current harness exposes bounded execution and source/UOp-derived structure,
but not the COMGR binary bytes or a complete compiled resource record. BoltBeam
does not fabricate those facts and therefore refuses the stronger causal claim.
Dynamic counters remain recommended/unsupported and block broader GPU-level
diagnosis, not this controlled timing comparison.

## Verification

```text
BoltBeam: 534 passed
Tinygrad focused MMQ suite: 46 passed in root integration run
Tinygrad WP-B focused suite: 82 passed
direct-owner real AMD correctness: PASS
gated real AMD correctness: PASS
production dispatch changed: false
```

## Next Work

Expose the actual compiled-code object and static resource metadata from the AMD
compile path. Feed those facts into the existing bundle without changing the
numeric experiment. Once both real bundles validate as evidence-complete, rerun
the same pair and allow BoltBeam to issue the formal writeback diagnosis.

Do not widen geometry, staging, synchronization, or K-loop search until that
provenance gap is closed or explicitly shown to be unavailable through the
current lowering path.
