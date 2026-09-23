# MMQ Complete-Knowledge Result

Date: 2026-07-11

Scope:

```text
GPU: live gfx1100, GPU-db466b901e983568, 96 CUs
kernel: bounded Q4_K x Q8_1 DS4 cooperative atom
shape: M=16, N=16, K=256
candidates: gated_matrix_v0 and direct_owner_v0
```

## Result

The bounded operational model is validated from semantic contract through
Tinygrad construction, loaded AMD binary, final ISA, static resources, dynamic
GPU evidence, isolated device execution, and full invocation wall time.

No production route changed.

## Exact Compiler Evidence

| fact | gated | direct owner |
|---|---:|---:|
| HSACO bytes | 14,600 | 6,792 |
| final ISA instructions | 2,131 | 465 |
| final b32 store sites | 256 | 1 |
| VGPR | 26 | 27 |
| SGPR | 26 | 29 |
| LDS | 256 B | 256 B |
| scratch/spills | 0 | 0 |

Loaded library bytes equal the analyzed program bytes. Rendered source, binary,
ISA, resources, launch geometry, and candidate/system identities are bound and
hashed. Final instructions carry structured classes, domains, dependencies,
epochs, and exec-mask bounds.

## Numeric And Ownership Contract

Both candidates pass bounded correctness with maximum absolute error
`0.00012207` at tolerance `1e-3`.

Both own 256 outputs uniquely. Every reached output store executes with a
full-wave active-lane bound. The two candidates produce the same 16 calibrated
64-byte output-line write requests.

## Dynamic Execution

Both candidates launch 256 waves. The gated candidate has materially more
instruction and wave activity:

```text
VALU: gated 75,008; direct 43,008
SALU: gated 147,712; direct 17,664
wave cycles: gated about 5.83-5.94M; direct about 1.80-2.02M
SQ busy: gated about 1.10M; direct about 0.34-0.38M
```

Output traffic is identical. Input request evidence is not identical:

```text
semantic input floor: 58 unique 128-byte lines
gated requests: 137-138
direct requests: 75-76
gated excess over direct: 61-63 requests
```

The calibrated load evidence is a scoped request proxy, not a universal DRAM
byte counter.

## Kernel Device Model

Isolated compile-once GPU timestamps:

```text
gated measured: 15.800 us
direct measured: 8.960 us
measured ratio: 1.7634x
```

The frozen v7 device model predicts:

```text
gated predicted: 15.530 us, error 1.71%
direct predicted: 8.568 us, error 4.37%
predicted ratio: 1.8125x, ratio error 2.79%
```

All device-model validation gates pass.

Gated device decomposition:

```text
shared device baseline: 8.304 us
shared LDS offset: 0.264 us
false-site/LDS interaction: 4.801 us
exact final-ISA false-site residual: 2.161 us
total: 15.530 us
```

An independent exact-ISA probe reproduced the missing residual with an
`8.473 ns` cost per false site. A grid-scaling hypothesis was tested and
rejected; false-site cost amortizes rather than amplifies at larger grids.

## Harness Scope Correction

The original harness timing is end-to-end CPU wall time, not isolated kernel
time. Each sample includes Tensor/UOp construction, input realization/transfer,
schedule creation, warmed compile/cache lookup, launch/synchronization, lifecycle
work, and NumPy readback.

Exact interleaved phase medians:

| phase | gated ms | direct ms |
|---|---:|---:|
| validation/views | 0.001 | 0.001 |
| Q4 + DS4 construction/realization | 2.254 | 2.264 |
| output allocation | 0.079 | 0.078 |
| UOp construction | 11.747 | 2.776 |
| schedule creation | 2.495 | 1.105 |
| warmed cache lookup | 0.024 | 0.023 |
| enqueue + synchronization | 0.369 | 0.356 |
| lifecycle/report | 0.014 | 0.014 |
| readback/cast | 0.092 | 0.091 |
| host total | 17.072 | 6.707 |

The 10.371 ms host gap is dominated by 8.971 ms of UOp construction and
1.389 ms of schedule construction. Other phases match within measurement noise.

## Hidden Python Construction Axis

Final graph counts initially failed to explain UOp construction because
canonicalization hides attempted Python work.

```text
direct attempted equality/store calls: 8 / 4
gated attempted equality/store calls: 520 / 259
final gated CMPNE nodes: only 66
```

Depth, fanout, total UOps, opcode histogram, and standalone CMPLT scaffolding
were independently varied and rejected as primary causes.

The decisive independent probe held the final canonical graph at 1,246 UOps and
varied attempted writeback operand topology. Candidate-shaped shared-special,
2D-index, and deep-value equality/store attempts add a 7.893 ms interaction
across the 256-site form. This is Python construction/canonicalization cost that
cannot be recovered from the final DAG alone.

## Invocation Model

The frozen v8 host model predicts:

```text
direct host predicted: 6.579 ms; measured: 6.707 ms; error: 1.91%
gated host predicted: 18.295 ms; measured: 17.072 ms; error: 7.17%
```

Primary phase gates pass. Host intervals are wider than device intervals because
rare Python/runtime outliers are preserved rather than discarded.

Combined host v8 plus device v7:

```text
direct predicted: 6.587 ms; measured: 6.716 ms; error: 1.92%
gated predicted: 18.311 ms; measured: 17.088 ms; error: 7.16%
predicted ratio: 2.7796x; measured ratio: 2.5442x; error: 9.25%
```

All declared combined operational gates pass.

## Causal Answer

The gated atom is slower for two separate reasons:

1. On the GPU, 255 false writeback sites expand into branch, predicate, SALU,
   VALU, wait, and extra input-request work even though output transactions are
   unchanged.
2. On the host, constructing those 256 candidate-shaped equality/store paths in
   Python is expensive before canonicalization collapses them into the final DAG.

Direct-owner writeback removes both costs while preserving tested numerics,
ownership, staging, geometry, K-loop behavior, LDS use, and output traffic.

## Remaining Bounded Uncertainty

```text
rare host timing tails widen invocation intervals
warmed-cache conclusions do not automatically transfer to cold compilation
some candidate SQ instruction/wait counters remain zero-suspect
the request proxy is scoped to this allocation/layout family
no claim transfers beyond this 16x16x256 atom and live gfx1100 snapshot
production-shaped/full-route behavior still requires a separate bound experiment
```

These uncertainties do not block the bounded causal or predictive conclusion.
They define the boundary of that conclusion.

## Verification

```text
BoltBeam: 617 passed
Tinygrad: complete test/unit/test_mmq*.py suite exits successfully
production dispatch changed: false
```
