# Remaining accuracy causes, 2026-10-12

This record follows [the accuracy fixes](accuracy-fixes-20261012.md) and the handoff
`handoff-accuracy-next-20261012.md`. It takes the handoff's sections in the order 4, 3, 2, 1, 5. The evidence is
private on the GPU box: the folder `accuracy-next-20261012` and the progress file `next-progress.md` beside the
earlier evidence. One GPU batch ran: three vLLM 8B decode repeats (section 4). Everything else used saved captures.

## 4. Role ambiguity and repeat variability

### The 8B Q/O rows

The 8B Q4_K_M profile has one `attn_qo` row of 72 calls per token: the query (before attention) and the output
(after attention) of 36 layers. The launch order separates them. In every layer the query is the 4096-row launch
just before K and V; the output is the first launch after attention, fused with the residual add in all but each
token's last layer. Timing played no part.

The row went unpaired for a structural reason, not a lack of evidence. The layer walk reserves each gap's first
launches for the output side, one per output row. `ffn_down` is Q4_K in 18 layers and Q6_K in the other 18: two rows,
one launch per layer. The walk counted four output launches where there are three. The query is the fourth launch of
each gap, so every query after the first layer was refused as an input launch.

Fix (main `80685ba`): a row whose format runs in only some layers counts toward that reservation only together with
the other formats of its role, once their layers cover every layer. All three llama.cpp 8B captures and the Ollama
8B capture now pair every role, with no exchange left open: `attn_qo` takes 1,116 input and 1,116 output launches
over 31 tokens. Every recorded pairing fixture is unchanged. The capture is a new replay fixture on dev.
The private repeat bands and holdouts for the 8B were made before this fix; they need re-derivation from the same
three saved captures before they are quoted again.

### Why holdouts exceed the two-run spread

A two-run band is the spread `|x2 - x1|` of one row. The holdout asks whether `|x3 - x1|` is larger. When three
runs are exchangeable (independent, same distribution), exchanging runs 2 and 3 shows that this happens with
probability exactly one half for every row. No hardware cause is needed.

| Setup | Rows outside | One-sided p under 1/2 | Run-3 time-weighted shift |
| --- | ---: | ---: | ---: |
| llama.cpp 8B | 5 / 13 | 0.87 | -0.02% |
| llama.cpp 27B | 15 / 19 | 0.010 | -0.01% |
| vLLM 8B | 6 / 9 | 0.25 | -0.03% |
| vLLM 27B | 7 / 14 | 0.61 | -0.13% |
| tinygrad 8B | 2 / 12 | 0.019 (low side) | -0.01% |
| tinygrad 27B | 5 / 14 | 0.91 | +0.02% |
| All | 40 / 81 = 49.4% | | |

The shifts are row-specific, not common-mode: no run moved all rows together by more than 0.13%.

Within one run, each weight row has a per-token series (31 tokens for llama.cpp, 63 for vLLM). Its lag-one
autocorrelation is small (-0.30 to 0.28), so the standard error of the run's mean is estimable from the run itself.
Comparing run differences with that error (`z = difference / combined standard error`):

| Setup | RMS z (pairs 1-2, 1-3, 2-3) | abs(z) > 3 |
| --- | --- | ---: |
| llama.cpp 8B | 1.32, 1.09, 0.85 | 0 / 18 |
| llama.cpp 27B | 0.69, 1.19, 0.64 | 0 / 30 |
| vLLM 8B (battery) | 0.99, 2.30, 2.18 | 2 / 15 |
| vLLM 27B (battery) | 1.60, 2.12, 2.88 | 6 / 24 |
| vLLM 8B (clock batch) | 1.05, 0.67, 1.34 | 0 / 15 |

llama.cpp run differences are token sampling. The narrow bands are pairs that landed close by chance: for example
the 27B `ssm_out` band is 0.00003 ms against a sampling error of 0.00037 ms for the difference of two runs.

vLLM showed extra run-level variation in the battery. Kernels, grids and calls were identical in all runs.
Decode captures recorded no clock, so the clock could not be checked. Captures now record it (main `74f9043`:
SM and memory clock, power, temperature and throttle reasons, beside each CUDA capture).

The clock batch: three back-to-back vLLM 8B decode repeats, exact battery setup. The prediction and its
decision rule were written first. The memory clock was 13,801 MHz in every busy sample of every capture; the SM
clock averaged 2,796 to 2,811 MHz; no throttle reason was set. The rows stayed within token sampling (RMS z up to
1.34). By the rule, the clock is excluded for this setup. The earlier vLLM excess did not recur; its source stays
unattributed. Allocation and memory placement are not recorded.

### Sampling rule, specified before any more repeats

Two runs cannot support a per-row moved or held verdict: by construction a fresh run falls outside a two-run
spread half of the time. No repeats were collected under the rule below. The rule was adopted afterwards; see "Measured uncertainty" below.

- Decision: a row of an exact setup moved when the after value lies outside the range of n independent before
  runs of that setup.
- Under exchangeable runs, the chance of that with no change is 2 / (n + 1) per row: 2/3 at n = 2, 1/5 at n = 9,
  1/10 at n = 19.
- n and the accepted rate are fixed before collection. Runs are back to back, same request and settings,
  each a full pipeline run. A row's work must be equal in every run.
- The within-run standard error is recorded beside the range. It explains llama.cpp; it can understate vLLM.

## 3. Small-kernel timer precision

The direct event-minus-empty estimate was compared with isolated Nsight Compute duration for ten roles over three
counterbalanced trials. Its error runs from -1.33 to +0.77 µs, for kernels from 2.66 µs to 615 µs. The error is
an absolute quantity, not a share of the kernel.

Held-out check: trials 1 and 2 give a largest error of 1.33 µs; every trial-3 error is within it (largest 1.26 µs).

The limit, stated: on this chip and timer method, an isolated row's floor-subtracted time is known to about
±1.3 µs. Its relative precision is about 1.3 / t: ±50% at 2.7 µs, ±6% at 22 µs, ±0.2% at 615 µs. The recurrent
projection's sign reversal (-0.83 to +0.70 µs) is inside that band. Graph replay is biased low by 0.7 to 2.5 µs and
stays rejected. No estimator constant was changed. Runs with the Nsight oracle already carry per-row
`timer_comparison` (estimate minus kernel).

The llama.cpp 27B `lm_head` reading: 0.6143 ms against a modeled floor of 0.6164 ms, 0.34% below. The floor uses the
read probe's best launch shape, 1,692.7 GB/s; `lm_head` reads at about 1,698 GB/s. The probe's own launch shapes
spread 0.9%, and its cold to sustained difference is 0.15%. The refusal compares the gap with the measured row's
repeat spread (0.00007 ms) only; the floor's own measured uncertainty is not part of the comparison. That is where
the refusal comes from. Whether the probe's spread belongs in the floor check is a policy decision; it was not
changed here. It was decided and changed afterwards: see "Measured uncertainty" below.

## 2. Profiling overhead

Per token, traced wall minus untraced wall, split into kernels and time outside them (first run of each setup):

| Setup | Untraced ms | Captured ms | Kernels ms | Captured - untraced | Kernels - untraced |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp 8B | 4.026 | 4.446 | 3.880 | +0.420 | -0.146 |
| llama.cpp 27B | 13.871 | 16.821 | 13.598 | +2.950 | -0.273 |
| Ollama 8B | 4.093 | 4.346 | 3.853 | +0.253 | -0.241 |
| tinygrad 8B | 4.109 | 4.477 | 3.912 | +0.368 | -0.196 |
| tinygrad 27B | 277.094 | 279.008 | 278.149 | +1.913 | +1.055 |
| vLLM 8B | 10.013 | 10.042 | 9.927 | +0.029 | -0.086 |
| vLLM 27B | 11.656 | 11.627 | 11.448 | -0.029 | -0.208 |

Where it sits: for llama.cpp, Ollama and tinygrad 8B the kernels sum to less than the untraced wall, so the whole
excess is time outside GPU kernels. Inside the llama.cpp capture window it is the gap after the one head launch per
token (27B 0.414 of 0.437 ms; 8B 0.205 of 0.213 ms). The rest of the 27B excess, about 2.8 ms, is outside the
kernel window, in the engine's own timing. tinygrad 27B is the one case where kernel time itself grows. vLLM shows
none. The reports already keep kernels, residual and untraced wall apart, so no BoltBeam change is needed.
Causes inside the engines are out of scope.

## 1. The tinygrad patch boundary

Reproduced in an isolated clone: the dev patch applies to `caa752ac` and gives tree `d5dd40e1`, the same tree as
the validated commit `928aed07` in the bundle. The 13 focused contracts pass. The legacy program-construction file
has the same five failures and 22 passes on the unpatched protected source as on the patched tree. They are
baseline fixture failures of the fork, not caused by the patch. Fixing them is fork work and was not done. The
patch was not deployed; the protected checkout is unchanged.

Scope: the patch carries role labels for attribution (call metadata, Q8 identities, observed extents) and four
fixes that let the models run (recurrent prompt stepping, epilogue admission, CUDA vector alignment, input layout).
None is a speed change. BoltBeam needs the labels to attribute tinygrad matrix work.

## 5. TensorRT

Unchanged and explicit. The TensorRT-engine 8B first build exceeds this host's memory (31 GiB, no swap); no cheap,
clear build-resource change was found, so it was not retried. The 27B checkpoint's asymmetric integer-4 format with
zero points has no correct loader in the installed converter; a new loader is engine work. The PyTorch backend
remains the supported 8B route, with one run and no repeat band.

## Remaining review findings

Both are resolved in "Measured uncertainty" below.

- `attribution.check_window` refused a capture at the larger of the chip band and 1%. It now uses the bandwidth's
  measured uncertainty, and says "not checked" without one.
- Pairing rejected an assignment above 105% of the measured bandwidth or peak. It now uses each figure times
  (1 + its measured uncertainty), and applies no bound without one.
- Without a repeat band, a capture whose kernels exceed its token is no longer refused; it reads "not checked".

## Measured uncertainty

Each measured bandwidth and peak in `boltbeam/data/targets.json` now carries `uncertainty` beside its value: the
probe's spread over its launch shapes and whole repeated runs, (max - min) / max, with no floor. The probes write it
(`--runs N`; `cuda_bandwidth.repeated`). `targets.uncertainty` is the one reader. Every guard reads it:

| Guard | Before | Now |
| --- | --- | --- |
| captured kernels against the captured token (`check_window`) | larger of chip band and 1% | bandwidth uncertainty |
| pairing: a launch faster than the bandwidth or peak | 105% | figure x (1 + its uncertainty) |
| prefill ceiling above the time a row ran in | 105% | the binding figure's uncertainty |
| role or token below its floor (`tinygrad_role_time.refusal`) | the row's repeat spread | repeat spread + floor x uncertainty |

A figure with no measured uncertainty is "not checked" in every guard. No constant replaces it. The documented
run-to-run range (tok/s) is kept as a record; it is no longer part of any guard.

The definition below (launch shapes and repeats together) was replaced the same day by the repeat spread at the
best launch shape: see "Best-shape uncertainty" at the end.

### Measured, 2026-10-13

Four whole runs of each probe. Four is the n the repeat rule asks for a pin. One GPU batch of about 90 seconds;
the box's model server was stopped and restored (health, one completion, build b11571).

| Chip and figure | Value | Uncertainty | Notes |
| --- | ---: | ---: | --- |
| RTX 5090 read | 1,701.4 GB/s | 0.37% | 7 shapes x 4 runs, 1,695.5 to 1,701.4; replaces 1,693.3 |
| RTX 5090 fp16 / bf16 / tf32 | 251.4 / 251.5 / 125.7 | 1.81% / 2.46% / 1.93% | registered rates inside the range |
| RTX 5090 fp16 accumulate / int8 / fp8 / fp4 | 497.7 / 995.8 / 502.8 / 1,988.9 | 2.47% / 2.54% / 2.23% / 2.63% | fp4 throttled |
| RTX 5090 fp32 FMA | 119.7 | 7.67% | one launch shape reads 114.9 |
| M3 read | 97.2 GB/s | 7.56% | 6 shapes x 4 runs; runs 95.2 to 97.1 |
| M3 fp16 matrix | 3.152 | 8.64% | this probe's best is 3.138; 3.152 is inside its range |

The 5090 read value changed. The old 1,693.3 came from a different probe and lies 0.48% below this probe's whole
range. A value and its uncertainty must come from one probe, so the registry now holds this probe's 1,701.4. The two
probes differ by more than either one's spread: the spread does not cover a change of method.

On the M3 the width comes from the weaker launch shapes (fewer threadgroups), not from repeats: the per-run best
moves 2.0% (read) and 0.5% (matrix). The definition follows the request ("its launch shapes and repeats"). It makes
the M3 guards loose. Whether only the near-best shapes should count is open.

### Verdicts that changed

The saved decode captures (llama.cpp 8B and 27B x3, vLLM 8B and 27B x3, corrected Ollama 8B and 27B x1) were
re-attributed from their raw nsys traces with the old and the new code, on private copies. No GPU.

- llama.cpp 27B, all three runs: the table was refused for `lm_head` below its floor. It is now accepted. `lm_head`
  reads "at the limit (within measured repeat spread and floor uncertainty ±0.005615 ms)". That run's own read probe
  spread 0.9%; the reading is 0.34% below the floor. Every other 27B role is checked.
- Pairings: unchanged in every capture. No role became unpaired and none became paired.
- `check_window`: every capture is checked against its run's own read uncertainty (0.90%) and passes.

### The 8B bands, re-derived

The bands were re-derived from the same saved captures after the Q/O pairing fix: runs 1 and 2 make the band, run 3
is the holdout. n = 2, so every verdict is labelled weak, with a false-move rate of 2/3 per row.

| Setup | Holdout rows outside the two-run range |
| --- | ---: |
| llama.cpp 8B | 7 / 13 |
| llama.cpp 27B | 16 / 19 |
| vLLM 8B | 9 / 10 |
| vLLM 27B | 9 / 15 |
| All | 41 / 57 = 72% (2/3 expected by chance) |

The earlier "half outside" table used |x3 - x1| > |x2 - x1|; the range rule counts a row outside when x3 leaves
[min, max] of runs 1 and 2, which happens 2/3 of the time by chance. The bands and verdicts are private evidence
beside the earlier folders.

### The repeat rule, adopted

A row moved when the after value lies outside the range of n before runs. The band records n. Every verdict prints
n and 2/(n+1). Fewer than 4 runs is weak. An engine pin needs n >= 4 fixed by `screen repeat-plan` before
collection; the audit refuses to justify a pin otherwise and says why. A run is never judged by a band it made.
See [the engine-upgrade audit](engine-upgrade-audit.md).

### Literal tolerances still in the code

The list that stood here was classified and resolved: see "The policy table" below.

## Best-shape uncertainty, 2026-10-13

Classify evidence before fixing mechanisms. A slow launch shape is not uncertainty in the peak. It is a separate
fact: how much the shape matters. The probes now write two fields:

- `uncertainty`: the spread of the whole runs' figures, each at its run's best launch shape, (max - min) / max.
  One run has none.
- `shape_spread`: the spread over every run's launch shapes. Data only; no guard reads it.

`targets.uncertainty` stays the one reader and reads only `uncertainty`. A saved read's old `spread` field is no
longer taken as an uncertainty. A run's own read is one probe run, so its guards use the registry's figure for
its chip.

Re-measured with n = 4 per probe: the M3 on the Mac (Metal, no model server running), and the RTX 5090 in one batch
of about 90 seconds. The box's model server was stopped and restored: health ok, one completion, build b11571.

| Chip and figure | Value | Old uncertainty | New uncertainty | Shape spread |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 read | 1,701.4 GB/s | 0.37% | 0.02% | 0.33% |
| RTX 5090 fp16 / bf16 / tf32 | 251.4 / 251.5 / 125.7 | 1.81% / 2.46% / 1.93% | 0.04% / 0.20% / 1.02% | 0.8% / 2.3% / 2.2% |
| RTX 5090 fp16 accumulate / int8 / fp8 / fp4 | 497.7 / 995.8 / 502.8 / 1,988.9 | 2.47% / 2.54% / 2.23% / 2.63% | 0.14% / 0.54% / 0.04% / 0.12% | 2.4% / 2.6% / 2.2% / 2.0% |
| RTX 5090 fp32 FMA | 119.7 | 7.67% | 0.58% | 6.5% |
| M3 read | 97.2 GB/s | 7.56% | 2.06% | 14.0% |
| M3 fp16 matrix | 3.121 (was 3.152) | 8.64% | 0.29% | 8.2% |

Every 5090 rate in the registry is within that batch's best x (1 + its uncertainty), so the values stay. The M3
read's 97.2 is within 97.1 x 1.0206. The M3 fp16 matrix value changed. The old 3.152 came from a different probe.
It lies 1.0% above this probe's whole range (3.112 to 3.121). A value and its uncertainty come from one probe, so the
registry now holds 3.121.

### Verdicts that changed

The same saved decode captures were re-attributed from their raw traces with the old and the new code, on private
copies. No GPU.

- llama.cpp 27B, all three runs: the role table is refused again: "roles below their floor beyond measured repeat
  spread and the floor's uncertainty: lm_head". The floor uses that run's own read, 1,692.7 GB/s. lm_head reads about
  1,698 GB/s, 0.34% faster. The read probe's repeat spread at its best shape is 0.02%. So the gap is not noise: in
  that run a real kernel read faster than the probe's best shape. The shape spread had hidden this.
- `check_window`: every capture is now checked at ±0.02% instead of ±0.90%. All 14 still pass.
- Pairings: unchanged in every capture.
- A pairing test fixture that used the old 1,693.3 now uses the registry's 1,701.4. With 1,693.3 and the new bound,
  lm_head (1,699 GB/s) is refused.

## The policy table

Prefer data over code, and centralize. Each literal tolerance was classified:
(a) policy, a stated choice; (b) a noise tolerance that should be measured; (c) arithmetic.
Every (a) moved to one table, `boltbeam/data/policy.json`, with a one-line reason each. It is read only through
`core.policy.value`. A missing entry raises; no code keeps a default. The old constants are removed.

| Literal | Where | Class | What happened |
| --- | --- | --- | --- |
| equal speed 1% | `eval/thresholds.py` | a | `equal_speed_pct` |
| tier A 5%, tier B 2% | `eval/thresholds.py` | a | `tier_a_min_pct`, `tier_b_min_pct` |
| protected regression 1% | `eval/thresholds.py` | a | `protected_regression_max_pct` |
| ramp 3%, throttle 5% | `roofline/roofline_ceiling.py` | b | the samples' own repeat spread; one sample each: "not checked", sustained used |
| plateau 8% | `memory/mem_hierarchy.py` | a | `cache_tier_plateau_share` |
| DRAM slack 2% | `collectors/kernel_oracle.py` | a | `dram_full_read_share` |
| `GAP_RATIO` 1.3 | `collectors/pairing.py` | a | `same_row_work_ratio`: a ratio of work that sets the report's grain, not a time |
| `NOISY_RULE` 10% and floor multiple 3 | `kernel_timer.py`, `tie_out.py` | a | `noisy_spread_pct`, `noisy_floor_multiple` |
| at ceiling 80%, headroom 50% | `kernel_analysis/theoretical_roofline.py` | a | `at_ceiling_pct`, `headroom_pct` |
| mostly zero 0.9, low correlation 0.9 | `kernel_analysis/diagnosis.py` | a | `mostly_zero_share`, `low_correlation` |
| GPU busy 40% | `collectors/providers.py` | a | `gpu_busy_pct` |
| noise spread multiplier 1.0 | `eval/thresholds.py` | c | left: it multiplies a measured spread by one |
| tie residual 0.001 ms, Metal tolerance 1e-3 | `diff.py`, `metal_native.py` | c | left |

### Open

- The probes keep one cold and one sustained figure per run, so a probe's regime now reads "not checked". Repeated
  cold and sustained samples would measure it. Both measured chips read the same figure as before.
- `same_row_work_ratio` is a stated choice. A measured test of whether two rows' times can be told apart (from the
  capture's own per-launch spread) does not exist.
- `evidence.THROTTLE_SHARE`, `LAUNCH_HEAVY` and `KV_SHARE` were not in the list and were not classified.
- The lm_head refusal: whether a floor should be the larger of the probe and the fastest measured kernel is a
  policy decision. It was not changed. Resolved without that policy: see "The probe's missing shape" below.

## The probe's missing shape, 2026-10-13

Classify evidence before fixing mechanisms. The 27B `lm_head` read about 1,698 GB/s. The read probe's best in that
run was 1,692.7 GB/s, with a repeat spread of 0.02%. So a real kernel beat the probe. That is evidence about the
probe, not about the kernel. The floor was not raised to the fastest kernel: that would be circular.

### Where the gap was

The `lm_head` launch, from the saved capture and the engine's CUDA source (read only): `mul_mat_vec_q` on Q6_K,
a grid of 248,320 blocks (one block per matrix row), 128 threads per block, about 4.2 KB per block, 4-byte loads,
no grid-stride loop. Every probe shape was a grid-stride loop of 16-byte loads with at most 16 blocks per SM.

### The fix

`collectors/cuda_bandwidth.py` adds one launch family: one block per contiguous chunk, a grid that covers the buffer
once. Six shapes: threads, bytes per block and load width of (128, 1 KiB, 16), (128, 4 KiB, 4), (128, 4 KiB, 16),
(256, 4 KiB, 4), (256, 16 KiB, 16), (512, 64 KiB, 16). It is a launch shape, not a model rule.

Measured on the RTX 5090, n = 4, one GPU batch of about 30 seconds. The box's model server was stopped and restored:
health ok, one completion, build b11571.

| Figure | Before | Now |
| --- | ---: | ---: |
| read | 1,701.4 GB/s | 1,707.4 GB/s |
| uncertainty (best-shape run spread) | 0.02% | 0.01% |
| shape spread | 0.33% | 0.67% |
| best shape | grid-stride, 512 threads, 8 blocks per SM | 128 threads, 4 KiB per block, 4-byte loads |

The four runs read 1,707.3, 1,707.2, 1,707.4 and 1,707.3. A wider exploratory sweep (64 to 512 threads, 512 B to
64 KiB per block, both load widths) found nothing faster than 1,706.9 in its single pass. The grid-stride shapes
read 1,695.6 to 1,699.9 in the same batch.

Against the new figure, `lm_head` sits under the ceiling. Its floor becomes 0.6111 ms; it reads 0.6143 ms, 0.5%
above. The saved 27B runs carry their own read (1,692.7 GB/s) from the old probe, so they keep their own floor.
A new run's read uses the new probe.

### The M3

The Metal probe has the same kind of grid-stride kernel, but its largest grid (16,384 threadgroups over 1 GiB,
64 KiB each) is already close to one group per chunk. No saved M3 role reads above the registry's 97.2 GB/s: the
fastest is the llama.cpp 8B `lm_head` at 93.8 GB/s, then the 8B `ffn_down` rows at about 80. There is no gap to
explain, so the Metal probe was not changed.

## Refuse the row, not the table

Unknown is not pass, for the row whose ceiling is in doubt. Before, one role below its floor beyond its repeat
spread and the floor's uncertainty refused the whole role table. Now:

- That row's ceiling is refused, with the reason: how far below its floor, and the tolerance it broke.
- Its time stays measured. It counts in not attributed and in the tie-out, on its own line
  ("ceiling refused: ..."), so the tie-out still adds up.
- The other rows keep their ceilings and verdicts. The losses and shares leave the refused row out.
- The summary and the TUI say it plainly: "Totals exclude 1 row whose ceiling is refused: ... The other rows keep
  their ceilings."
- A whole token below its floor beyond noise still refuses the table, as before.

### Verdicts that changed

The saved decode captures (llama.cpp 8B and 27B x3, vLLM 8B and 27B x3, Ollama 8B and 27B x1) were re-attributed
from their raw traces with the old and the new code, on private copies. No GPU.

- llama.cpp 27B, all three runs: the role table was refused. It is now accepted, with nine roles checked:
  six at the limit and three too small to fill memory. `lm_head` alone has its ceiling refused: 0.0019 to
  0.0021 ms below its floor, against a tolerance of ±0.00013 ms.
- `check_window`: every capture is checked at ±0.01% (the new uncertainty) instead of ±0.02%. All 14 still pass.
- Pairings, unpaired roles and holdouts: unchanged in every capture.
- The 8B captures: no verdict changed.
