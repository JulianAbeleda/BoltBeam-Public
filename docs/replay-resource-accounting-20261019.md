# Full-replay resource accounting

The resource report accounts for required semantic arithmetic on every observed
native launch, then estimates the demand on shared compute and memory resources
over the full worker replay window. Per-operation DRAM attribution remains
unavailable. Full work coverage does not mean every issued machine instruction
has been counted, or that individual-kernel rooflines are available.

Native identity is joined by observed submission and graph-construction order,
with exact binary, entry, geometry, scheduler phase and operand witnesses.
Primary attention requires complete layer, operator and tensor-shape coverage.
Attention core work and separately derived auxiliary work are disjoint.
Grouped launch counts and durations reconcile with the native stream and turns.

## Temporal memory evidence

PM samples describe device traffic, including inter-request, background and
collector-control demand. They do not establish which kernel caused a write.
Contained bins give the lower temporal demand endpoint; intersecting bins give
the upper endpoint. Pre/post-window traffic is reported separately. All bins
must be contiguous, valid and complete, with no overflow and a recorded
end-of-records decode result.

A collector may preserve a completed startup prefix before work begins. The
reader reconstructs records from that immutable prefix and newly completed
indices, checking the early clock barrier, sample counts, metric values and end
timestamps against the final query. Late start-timestamp changes remain explicit
evidence; raw files are never repaired. A saved merged CSV or a passing flag is
not sufficient. Every input belongs to the same complete capture manifest.

Lossless compressed activity streams retain both raw and compressed hashes and
the original manifest. The reader verifies the decompressed bytes before use.

## Empirical resource estimates

Each nonzero work group needs an exact binary/entry execution-path binding and
an applicable independent rate measurement. Shared resource budgets add;
independent resource budgets combine by maximum. Summing per-kernel maxima can
overstate a resource bound when independent engines overlap.

Rates must match the captured device and driver. Repeat spread, traffic-mix
sensitivity and sample-edge intervals are empirical uncertainty, not statistical
confidence intervals or proven physical capacity bounds. Unknown rates or
execution paths prevent a complete estimate. The observed DRAM write fraction
must fit the declared calibration mix span; unknown L2 read/write mix remains
an explicit sensitivity assumption.
Compute rates use FLOPs/ms, DRAM uses bytes/ms, and L2 request rates use sectors/ms;
an L2 byte rate must be divided by the 32-byte sector width before admission.

The resource percentage uses the same containing worker interval as the PM
demand, not a smaller sum of kernel windows. Estimates exceeding the observed
window are displayed without clamping. Run and Emit retain the existing kernel
time reconciliation and separately show the full-replay resource estimate,
capture digest, rate/binding digest and qualification protocol digest.

## Timing admission

The prospective session comparison uses four fixed native/profiled pairs in
NP, NP, PN, PN order. Native means untraced forced-token replay with the standard
worker. Common host telemetry does not enable custom GPU observers.

The original prospective protocol uses a 5% ceiling. For paired excess
`D = profiled_seconds - 1.05 * native_seconds`, the one-sided 95% Student-t upper
bound on mean excess must be nonpositive, and every observed pair must be within
5%. The inference is conditional on independent approximately normal paired
excesses; four observations cannot establish those assumptions or guarantee a
population maximum. Turn-level differences remain diagnostics.

Raw artifact integrity, request/token/cache equality, observer absence, observed
GPU exclusivity, complete activity/PM data and numeric thermal balance are
separate validity gates. Missing or invalid evidence cannot become a qualified
result. Pair one is selected prospectively as the resource-report source; failed
pairs are neither replaced nor extended until a favorable result appears.

This integration does not itself qualify an existing capture. Production
calibration, final full-window admission and prospective timing remain separate
evidence requirements.

## Retrospective observed acceptance

An explicitly supplied user policy can accept the observed overhead range of the
retained four pairs. It is bound to the complete experiment and is recomputed
when Run and Emit validate the report. It never changes the original prospective
5% verdict, excuses missing capture evidence, or claims a population bound.
Clock variation remains disclosed: the observed differences do not isolate
profiler cost. The report leads with retrospective acceptance and separately
shows the original qualification and empirical resource estimate.

## Verified production replay

The selected Qwen3-8B replay reconciles all 60 turns, 30,634 scheduler steps and
12,767,680 observed launches with required work and 42 exact execution bindings.
Its empirical resource estimate is 320.905–322.598 seconds over a 420.698-second
worker window (76.279–76.682%), dominated by DRAM demand. This is a resource
estimate, not an individual-kernel roofline or guaranteed physical limit.

All four retained comparisons observed 2.744–6.580% elapsed overhead, with a
4.217% mean ratio. The user accepted a retrospective 7% observed ceiling with
clock variation disclosed. The original prospective 5% verdict remains
unavailable; the observations do not isolate profiler cost or guarantee future
overhead. Across the four captures, the full DRAM-demand interval varies by
0.00685%, and L1TEX-to-L2 request demand by 0.05638%.

Selected capture digest:
`b9d4cdc144c13b5173cdb473c8c5a13a6e0ff5f4b79e45c409f0b6342fef433f`.
Independent rates and exact execution-binding digest:
`bdf2038c2c7b315810b18f6da1b0d088453e2d86d6a0afe602ce412582ade4e9`.
