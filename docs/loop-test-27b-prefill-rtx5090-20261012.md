# One loop test: 27B prefill, llama.cpp, RTX 5090 (2026-10-12)

The strict loop did not close. The selected GDN kernel became about 43% faster,
but the full upstream engine checkout also changed other rows and the FFN work.
BoltBeam located the recurrent gain and reconciled the captured time. It could
not predict the gain quantitatively or attribute all of the new FFN launches.

## What ran

- BoltBeam: `15f7125`, pulled with `git pull --ff-only` in `~/storage/ubuntu-cmp/BoltBeam`.
- Model: the 27B hybrid Q4_K_M GGUF that the 10-11 audit used.
- Baseline: existing `~/env/llama.cpp/build-cuda/bin/llama-bench`, read-only;
  its timing output reports `ac4cddeb0` (upstream tag b9592).
- Changed engine: fresh `~/storage/loop27-llama-upstream` clone, detached at
  `24e41838e0c834f67db214e6d280af51f266056e`,
  "ggml-cuda: assign four GDN state columns per warp (#30087)".
  This was the newest commit touching CUDA GDN or ssm-conv in upstream history
  after b9592 when inspected. It changes GDN, not ssm-conv.
- Build: Release, CUDA enabled, architecture 120, `/usr/local/cuda/bin/nvcc`,
  `LLAMA_CURL=OFF`; CUDA flash attention and graphs enabled, neither cuBLAS nor
  MMQ forced. The CUDA cache settings match the existing build.
- Both configurations: provider `llama.cpp`, workload `prefill`, lengths 512 and
  2048, `--analyze --measure auto --no-search`; recorded ubatch 512 and batch 2048.
- One pipeline per configuration, no retries. Baseline exit 0 in 44 seconds;
  changed audit exit 0 in 30 seconds. Each pipeline includes its normal untraced
  repetitions and one captured prefill per length after warm-up.
- GPU compute processes were empty before each audit and afterward.
  The box's own model server stayed stopped. No kernels or BoltBeam code were edited.

The referenced `steps-vx27.sh` was absent on this box. The documented pipeline
flags and the storage environment from the earlier decode capture were used.

Environment for both runs:

```sh
W=$HOME/storage/ubuntu-cmp
B=$HOME/storage/bbeng
export TMPDIR=$W/tmp BOLTBEAM_CUDA_CACHE=$W/tmp/boltbeam-cuda CUDA_CACHE_PATH=$W/tmp/nv
export PATH=/usr/local/cuda/bin:$PATH HF_HOME=$B/tmp/hf
# Set BOLTBEAM_LLAMA_BENCH to the baseline or fresh engine path above.
cd "$W/BoltBeam"
# RUN and ID distinguish loop27-baseline from loop27-upstream.
$W/.venv/bin/python -m boltbeam.workflow.screen pipeline --run "$RUN" \
  --target nvidia_sm120 --id "$ID" --provider llama.cpp --workload prefill \
  --ctxs 512,2048 --analyze --measure auto --no-search \
  "$MODEL"   # the 27B Q4_K_M GGUF
$W/.venv/bin/python -m boltbeam.workflow.screen run --run "$RUN" > "$RUN-summary.txt"
```

## Prediction, recorded before the changed audit

The following is the saved prediction, including the baseline template detail
appended before the changed GPU run:

```text
UTC recorded: 2026-10-12T12:49:12.248405+00:00

Prediction recorded BEFORE the changed-engine GPU audit.
BoltBeam commit: 15f7125. Baseline binary reports ac4cddeb0 (b9592).
Change: upstream llama.cpp 24e41838e0c834f67db214e6d280af51f266056e,
"ggml-cuda: assign four GDN state columns per warp (#30087)".
Keep ubatch 512, batch 2048, lengths 512 and 2048, model and audit flags identical.

Predicted target: recurrent, specifically gated_delta_net_cuda. Expect lower time.
ssm_conv is unchanged by this commit and should stay within +/-3%.
Baseline recurrent: 18.773977 ms at 512; 76.596276 ms at 2048.
BoltBeam ceiling: 2.323496 ms at 512; 9.293985 ms at 2048.
The ceiling bounds the modeled maximum recoverable loss: 16.450481 and
67.302291 ms respectively. It is NOT a speedup prediction for this warp change.
BoltBeam has no measured adapter or scheduling model for the new GDN kernel,
so it cannot predict a tighter ms amount. Four columns per warp does not imply
four times faster. Treat any recurrent improvement greater than 3% as directional
confirmation only, not confirmation of a quantitative speedup prediction.

Null prediction: every matrix row and attention row remains within +/-3% of
its baseline captured ms. Unchanged auxiliary kernels also stay within +/-3%;
new/removed auxiliary rows are a failure of the 'nowhere else' condition.
Gaps are reported separately, not silently assigned to recurrent.
Each row's modeled FLOPs, bytes, calls and chunk widths should be unchanged;
new kernel scheduling may change actual traffic without changing modeled work.
The captured wall-time change must equal the sum of all kernel-row changes
plus the measured gap change, within 0.001 ms rounding.
The untraced wall-time change should approximately follow the captured change;
report disagreements rather than attributing them to GDN.

Confound declared in advance: the requested full upstream checkout includes
intervening changes since b9592 outside GDN. No cherry-pick or kernel edits
are being made. Thus the null prediction is a stringent test, not guaranteed
by the source diff. Any movement outside the target means the strict loop
did not close. Only one audit per configuration; no post-result retries.

Baseline template detail read before changed audit:
GDN alone: 18.209403 ms at 512; 74.197830 ms at 2048.
ssm_conv alone: 0.564574 ms at 512; 2.398446 ms at 2048.
Apply the +/-3% unchanged test to the ssm_conv template separately.
```

## Every row

Times are overlap-adjusted captured kernel ms. The two time columns are
baseline -> changed (delta). "Miss" means the time exceeded the predeclared
3% band, a row appeared/disappeared, or its modeled work/call count changed.
A time within 3% does not pass the unchanged-work condition on its own.
For auxiliary rows whose FLOPs and bytes are unknown, equal null values do not
establish equal work; the available call counts are compared instead.

| Row | 512 tokens: ms | 2048 tokens: ms | Prediction |
|---|---:|---:|---|
| ffn_gate_up Q4_K | 36.863 -> 36.147 (-0.715) | 153.358 -> 148.942 (-4.416) | Miss 512, 2048 |
| ffn_down Q4_K | 19.251 -> 18.933 (-0.317) | 80.559 -> 78.426 (-2.133) | Miss 512, 2048 |
| recurrent | 18.774 -> 10.897 (-7.877) | 76.596 -> 44.073 (-32.523) | Direction matched |
| attn_qkv Q8_0 | 8.143 -> 8.140 (-0.003) | 33.963 -> 33.693 (-0.270) | Within band; recorded work/calls unchanged |
| attn_gate Q8_0 | 4.974 -> 4.903 (-0.071) | 20.139 -> 19.717 (-0.421) | Within band; recorded work/calls unchanged |
| ssm_out Q8_0 | 4.947 -> 4.916 (-0.031) | 20.320 -> 20.026 (-0.294) | Within band; recorded work/calls unchanged |
| mul_mat_q_stream_k_fixup | 3.997 -> 3.967 (-0.029) | 17.009 -> 16.474 (-0.535) | Miss 512, 2048 |
| unary_gated_op_kernel | 3.432 -> 3.447 (+0.016) | 14.159 -> 13.984 (-0.176) | Miss 512, 2048 |
| attn_qo Q8_0 | 3.140 -> 3.142 (+0.002) | 13.106 -> 12.979 (-0.127) | Within band; recorded work/calls unchanged |
| rms_norm_f32 | 2.687 -> 3.415 (+0.728) | 11.080 -> 14.096 (+3.016) | Miss 512, 2048 |
| quantize_mmq_q8_1 | 2.592 -> 2.519 (-0.073) | 10.784 -> 10.524 (-0.261) | Miss 512, 2048 |
| concat_f32_non_cont | 2.405 -> 0.000 (-2.405) | 9.952 -> 0.000 (-9.952) | Miss 512, 2048 |
| attn_qo Q6_K | 2.277 -> 2.223 (-0.054) | 9.301 -> 9.019 (-0.282) | Miss 2048 |
| k_bin_bcast | 1.528 -> 1.500 (-0.028) | 6.302 -> 6.114 (-0.188) | Miss 512, 2048 |
| ssm_alpha_beta Q8_0 | 1.052 -> 1.042 (-0.011) | 4.274 -> 4.146 (-0.129) | Miss 2048 |
| attention | 0.768 -> 0.795 (+0.027) | 6.469 -> 6.519 (+0.050) | Miss 512 |
| cpy_scalar | 0.763 -> 0.765 (+0.002) | 3.226 -> 3.212 (-0.015) | Within band; recorded work/calls unchanged |
| l2_norm_f32 | 0.747 -> 0.000 (-0.747) | 3.180 -> 0.000 (-3.180) | Miss 512, 2048 |
| attn_kv Q8_0 | 0.717 -> 0.721 (+0.003) | 2.975 -> 2.971 (-0.004) | Within band; recorded work/calls unchanged |
| lm_head Q6_K | 0.633 -> 0.627 (-0.006) | 0.636 -> 0.625 (-0.011) | Within band; recorded work/calls unchanged |
| flash_attn_stream_k_fixup_general | 0.318 -> 0.316 (-0.001) | 1.370 -> 1.342 (-0.028) | Within band; recorded work/calls unchanged |
| k_get_rows_float | 0.269 -> 0.089 (-0.179) | 1.445 -> 0.453 (-0.992) | Miss 512, 2048 |
| scale_f32 | 0.261 -> 0.258 (-0.003) | 0.264 -> 0.262 (-0.002) | Within band; recorded work/calls unchanged |
| rope_multi | 0.201 -> 0.185 (-0.015) | 0.836 -> 0.761 (-0.075) | Miss 512, 2048 |
| k_set_rows | 0.074 -> 0.077 (+0.003) | 0.316 -> 0.322 (+0.006) | Miss 512 |
| unary_op_kernel | 0.068 -> 0.062 (-0.006) | 0.274 -> 0.271 (-0.003) | Miss 512 |
| quantize_q8_1 | 0.001 -> 0.004 (+0.003) | 0.001 -> 0.004 (+0.003) | Miss 512, 2048 |
| concat_non_cont | 0.000 -> 2.385 (+2.385) | 0.000 -> 9.960 (+9.960) | Miss 512, 2048 |
| mul_mat_vec_q | 0.000 -> 0.107 (+0.107) | 0.000 -> 0.108 (+0.108) | Miss 512, 2048 |
| k_get_rows_float_vec | 0.000 -> 0.099 (+0.099) | 0.000 -> 0.726 (+0.726) | Miss 512, 2048 |

Zero in the table means the named row is absent from that capture.

GDN alone fell 18.209403 -> 10.325311 ms at 512 and
74.197830 -> 41.692541 ms at 2048. ssm-conv stayed within the 3% band:
0.564574 -> 0.571410 ms and 2.398446 -> 2.380354 ms.
The combined recurrent row fell 41.96% and 42.46% respectively.

## Work and attribution

Chunk widths stayed `[512]` and `[512, 512, 512, 512]`. Recurrent and attention
modeled FLOPs and bytes stayed equal. Every non-FFN matrix row's recorded work
and calls stayed equal.

Both FFN rows lost 1/64 of their recorded batched work. Gate/up calls changed
128 -> 126 at 512 and 512 -> 504 at 2048; down calls changed 64 -> 63 and
256 -> 252. Their summed recorded operations changed:

| Tokens | Baseline FFN operations | New attributed FFN operations |
|---|---:|---:|
| 512 | 17,523,466,567,680 | 17,249,662,402,560 |
| 2048 | 70,093,866,270,720 | 68,998,649,610,240 |

The new capture contains two Q4_K `mul_mat_vec_q` launches with no role,
0.106682 ms at 512 and 0.107668 ms at 2048. They have no modeled work in that
row. Source inspection shows changed last-layer output cropping in
`src/models/qwen35.cpp`, consistent with FFN work being reduced to selected
output tokens. This is an explanation supported by the graph and capture,
not a complete role attribution of those two GEMVs.

The full checkout also replaces recurrent L2 normalization with a different
graph construction. `l2_norm_f32` disappears and `rms_norm_f32` grows; concat
and get-rows templates also change. Therefore this is not an isolated GDN
intervention. This confound was recorded before the changed audit.

Both captures have `ambiguous: null`. That does not imply complete work
coverage: the new Q4_K GEMVs remain in an auxiliary row. Attention fixup has
no ceiling because the adapter names no stream-k layout. Many auxiliary rows
have no FLOP or byte model. BoltBeam has no re-performance adapter or scheduling
model for the new GDN kernel, so there was no exact predicted speedup to test.

## Tie-out

The row sums equal the raw capture's union of GPU kernel intervals; overlapping
kernel durations are shared rather than counted twice. The changes reconcile:

| Tokens | Sum of kernel-row changes | Gap change | Captured wall change |
|---|---:|---:|---:|
| 512 | -9.196509 ms | -0.109815 ms | -9.306324 ms |
| 2048 | -42.148504 ms | +36.137928 ms | -6.010576 ms |

Residuals are below 0.001 ms. Captured wall times are
121.363652 -> 112.057328 ms and 507.050358 -> 501.039782 ms.
The 2048 capture's gaps grew from 5.152922 to 41.290850 ms; their cause was not
established and the increase is not assigned to GDN.

Untraced medians improved 124.545817 -> 115.813977 ms and
501.438372 -> 468.192594 ms. At 2048 the untraced improvement is 33.245778 ms,
but captured wall improvement is only 6.010576 ms. This disagreement blocks
an exact end-to-end confirmation from the capture.

## Result and retained evidence

Directional localization succeeded: GDN improved and convolution stayed within
noise. The strict loop failed: unrelated rows moved, FFN work changed, two
GEMVs were not assigned to roles, and the exact gain could not be predicted.
This run does not establish a BoltBeam-validated isolated engine improvement.
No additional GPU run was made to repair the result.

All paths below are on the Ubuntu box; raw captures are retained there rather
than committed to this documentation-only change:

- `~/storage/bbeng/tmp/loop27-baseline/`
- `~/storage/bbeng/tmp/loop27-upstream/`
- `~/storage/bbeng/tmp/loop27-baseline-summary.txt`
- `~/storage/bbeng/tmp/loop27-upstream-summary.txt`
- `~/storage/bbeng/tmp/loop27-prediction.txt`
- `~/storage/bbeng/tmp/loop27-comparison.csv` (exact times, percentages, FLOPs, bytes, calls)
- `~/storage/bbeng/tmp/loop27-checks.json` (sum checks and recorded total operations)
- Each run's `timing_trace.json`, `prefill_trace.json`, and
  `kernel_compare/prefill_capture/{512,2048}/` (raw nsys captures and CSVs).

Related audits: [27B baseline](prefill-audit-27b-rtx5090-20261011.md) and
[vLLM cross-check](prefill-vllm-crosscheck-27b-rtx5090-20261011.md).
