# Accuracy battery: RTX 5090, 2026-10-12

The criteria in `predictions.txt` were written before the battery (main e7e68fa, dev 1f37973).
This report uses saved traces, raw launch CSVs, logs, model profiles, measured peaks and instruction paths.
No GPU work and no code fixes were performed for part 2. Analysis uses main 359e337.
Evidence is kept outside the repository in `acc-20261012`; derived evidence and the append-only
`analysis-progress.md` are in the analysis copy. Quant formats differ across engines; operation counts must agree.

Re-attribution under current main covers repA/repB decode and prefill, the 8B decode and prefill,
ub2048, vLLM decode and both prefills, and every replay capture. Fix 6d288bb repairs the false decode twin;
08b51f5 repairs mixed-format slots and last-layer pruning; d68c7e8 restores packed parts' shapes.
86aa20d changes which repetition future captures collect. Old one-repetition captures cannot acquire a
missing second repetition by re-attribution; their original timing failure remains. The saved fix3-prefill
already collected that second repetition. Original evidence is retained beside the derived files.

## 1. Repeatability — FAIL overall

Criterion: every row and gaps within max(3% of A, 0.05 ms); identical calls, operations, bytes and limit;
each capture within 6% of the untraced median. Prediction: rows pass, medians within 1%, 2048 end-to-end fails.

| workload | rows | rows outside band/work change | largest row spread | gap change / band ms | capture error A / B | median change | verdict |
|---|---|---|---|---|---|---|---|
| prefill 512 | 29 | 0 | 0.175× band | -0.000157 / 0.050000 | -3.397% / -3.182% | -0.048% | PASS |
| prefill 2048 | 29 | 0 | 0.071× band | +0.789862 / 1.154801 | +6.510% / +6.679% | +0.070% | FAIL |
| decode 161.0 | 19 | 0 | 0.087× band | -0.009068 / 0.050000 | +4.996% / +4.861% | +0.024% | PASS |

All individual kernel and gap spreads pass. Eight 2048 ceilings change because measured clock-scaled peaks change; operations and bytes do not. Thus the literal identical-limit criterion also fails. Keep the 3%/0.05 ms row band: no widening is supported. A 6.679% end-to-end envelope would only fit these two original captures; it is not a qualified new band. Decode captures run at initial depth 128 (mean context 161 for llama.cpp), while the untraced benchmark requested 512: the comparison is not a same-context proof.

| workload | row | A ms | B ms | absolute spread ms | band ms | verdict |
|---|---|---|---|---|---|---|
| prefill 512 | attention | 0.791455 | 0.796639 | 0.005184 | 0.050000 | PASS |
| prefill 512 | attn_gate Q8_0 | 4.893407 | 4.898334 | 0.004927 | 0.146802 | PASS |
| prefill 512 | attn_kv Q8_0 | 0.721660 | 0.724061 | 0.002401 | 0.050000 | PASS |
| prefill 512 | attn_qkv Q8_0 | 8.140282 | 8.146460 | 0.006178 | 0.244208 | PASS |
| prefill 512 | attn_qo Q6_K | 2.220118 | 2.224633 | 0.004515 | 0.066604 | PASS |
| prefill 512 | attn_qo Q8_0 | 3.133815 | 3.150266 | 0.016451 | 0.094014 | PASS |
| prefill 512 | concat_non_cont | 2.390395 | 2.392925 | 0.002530 | 0.071712 | PASS |
| prefill 512 | cpy_scalar | 0.765919 | 0.767357 | 0.001438 | 0.050000 | PASS |
| prefill 512 | ffn_down Q4_K | 18.915969 | 18.960482 | 0.044513 | 0.567479 | PASS |
| prefill 512 | ffn_down Q4_K, last layer at output tokens | 0.037617 | 0.037025 | 0.000592 | 0.050000 | PASS |
| prefill 512 | ffn_gate_up Q4_K | 36.090493 | 36.182466 | 0.091973 | 1.082715 | PASS |
| prefill 512 | ffn_gate_up Q4_K, last layer at output tokens | 0.070107 | 0.069719 | 0.000388 | 0.050000 | PASS |
| prefill 512 | flash_attn_stream_k_fixup_general | 0.316719 | 0.316908 | 0.000189 | 0.050000 | PASS |
| prefill 512 | k_bin_bcast | 1.498173 | 1.497136 | 0.001037 | 0.050000 | PASS |
| prefill 512 | k_get_rows_float | 0.092525 | 0.093658 | 0.001134 | 0.050000 | PASS |
| prefill 512 | k_get_rows_float_vec | 0.100116 | 0.099030 | 0.001086 | 0.050000 | PASS |
| prefill 512 | k_set_rows | 0.076766 | 0.076926 | 0.000160 | 0.050000 | PASS |
| prefill 512 | lm_head Q6_K | 0.626484 | 0.626504 | 0.000020 | 0.050000 | PASS |
| prefill 512 | mul_mat_q_stream_k_fixup | 3.971048 | 3.975408 | 0.004360 | 0.119131 | PASS |
| prefill 512 | quantize_mmq_q8_1 | 2.516920 | 2.517472 | 0.000552 | 0.075508 | PASS |
| prefill 512 | quantize_q8_1 | 0.003939 | 0.004086 | 0.000147 | 0.050000 | PASS |
| prefill 512 | recurrent | 10.896704 | 10.898568 | 0.001864 | 0.326901 | PASS |
| prefill 512 | rms_norm_f32 | 3.599225 | 3.602877 | 0.003652 | 0.107977 | PASS |
| prefill 512 | rope_multi | 0.187211 | 0.187834 | 0.000623 | 0.050000 | PASS |
| prefill 512 | scale_f32 | 0.262000 | 0.263801 | 0.001802 | 0.050000 | PASS |
| prefill 512 | ssm_alpha_beta Q8_0 | 1.043562 | 1.046058 | 0.002496 | 0.050000 | PASS |
| prefill 512 | ssm_out Q8_0 | 4.912511 | 4.912832 | 0.000321 | 0.147375 | PASS |
| prefill 512 | unary_gated_op_kernel | 3.446699 | 3.448350 | 0.001651 | 0.103401 | PASS |
| prefill 512 | unary_op_kernel | 0.064039 | 0.063655 | 0.000383 | 0.050000 | PASS |
| prefill 2048 | attention | 6.515239 | 6.522873 | 0.007634 | 0.195457 | PASS |
| prefill 2048 | attn_gate Q8_0 | 19.714303 | 19.717726 | 0.003423 | 0.591429 | PASS |
| prefill 2048 | attn_kv Q8_0 | 2.977112 | 2.983445 | 0.006333 | 0.089313 | PASS |
| prefill 2048 | attn_qkv Q8_0 | 33.716903 | 33.778090 | 0.061187 | 1.011507 | PASS |
| prefill 2048 | attn_qo Q6_K | 9.030269 | 9.032669 | 0.002400 | 0.270908 | PASS |
| prefill 2048 | attn_qo Q8_0 | 13.007771 | 13.022171 | 0.014400 | 0.390233 | PASS |
| prefill 2048 | concat_non_cont | 9.974587 | 9.982944 | 0.008357 | 0.299238 | PASS |
| prefill 2048 | cpy_scalar | 3.213519 | 3.220379 | 0.006860 | 0.096406 | PASS |
| prefill 2048 | ffn_down Q4_K | 78.558809 | 78.648854 | 0.090045 | 2.356764 | PASS |
| prefill 2048 | ffn_down Q4_K, last layer at output tokens | 0.037026 | 0.037009 | 0.000017 | 0.050000 | PASS |
| prefill 2048 | ffn_gate_up Q4_K | 149.098724 | 149.255482 | 0.156758 | 4.472962 | PASS |
| prefill 2048 | ffn_gate_up Q4_K, last layer at output tokens | 0.069896 | 0.070042 | 0.000147 | 0.050000 | PASS |
| prefill 2048 | flash_attn_stream_k_fixup_general | 1.342553 | 1.344634 | 0.002081 | 0.050000 | PASS |
| prefill 2048 | k_bin_bcast | 6.110075 | 6.113954 | 0.003879 | 0.183302 | PASS |
| prefill 2048 | k_get_rows_float | 0.477117 | 0.478900 | 0.001782 | 0.050000 | PASS |
| prefill 2048 | k_get_rows_float_vec | 0.723547 | 0.721286 | 0.002261 | 0.050000 | PASS |
| prefill 2048 | k_set_rows | 0.320891 | 0.321484 | 0.000593 | 0.050000 | PASS |
| prefill 2048 | lm_head Q6_K | 0.625971 | 0.626056 | 0.000085 | 0.050000 | PASS |
| prefill 2048 | mul_mat_q_stream_k_fixup | 16.490357 | 16.499482 | 0.009125 | 0.494711 | PASS |
| prefill 2048 | quantize_mmq_q8_1 | 10.546149 | 10.543432 | 0.002717 | 0.316384 | PASS |
| prefill 2048 | quantize_q8_1 | 0.004083 | 0.003968 | 0.000115 | 0.050000 | PASS |
| prefill 2048 | recurrent | 44.115470 | 44.098420 | 0.017051 | 1.323464 | PASS |
| prefill 2048 | rms_norm_f32 | 14.900812 | 14.891196 | 0.009616 | 0.447024 | PASS |
| prefill 2048 | rope_multi | 0.781897 | 0.782750 | 0.000852 | 0.050000 | PASS |
| prefill 2048 | scale_f32 | 0.267795 | 0.267804 | 0.000010 | 0.050000 | PASS |
| prefill 2048 | ssm_alpha_beta Q8_0 | 4.148637 | 4.147557 | 0.001080 | 0.124459 | PASS |
| prefill 2048 | ssm_out Q8_0 | 20.011236 | 20.019506 | 0.008270 | 0.600337 | PASS |
| prefill 2048 | unary_gated_op_kernel | 13.995754 | 13.994766 | 0.000988 | 0.419873 | PASS |
| prefill 2048 | unary_op_kernel | 0.274706 | 0.275455 | 0.000749 | 0.050000 | PASS |
| decode 161.0 | KV cache write | 0.024401 | 0.024401 | 0.000001 | 0.050000 | PASS |
| decode 161.0 | attention | 0.111892 | 0.111886 | 0.000006 | 0.050000 | PASS |
| decode 161.0 | attn_gate Q8_0 | 1.031411 | 1.027041 | 0.004371 | 0.050000 | PASS |
| decode 161.0 | attn_kv Q8_0 | 0.150406 | 0.150644 | 0.000238 | 0.050000 | PASS |
| decode 161.0 | attn_qkv Q8_0 | 1.693498 | 1.693156 | 0.000342 | 0.050805 | PASS |
| decode 161.0 | attn_qo Q6_K | 0.290704 | 0.290750 | 0.000046 | 0.050000 | PASS |
| decode 161.0 | attn_qo Q8_0 | 0.667684 | 0.667440 | 0.000244 | 0.050000 | PASS |
| decode 161.0 | copy | 0.048061 | 0.048051 | 0.000010 | 0.050000 | PASS |
| decode 161.0 | elementwise | 0.109305 | 0.109189 | 0.000115 | 0.050000 | PASS |
| decode 161.0 | ffn_down Q4_K | 2.090099 | 2.090436 | 0.000337 | 0.062703 | PASS |
| decode 161.0 | ffn_gate_up Q4_K | 4.273834 | 4.273603 | 0.000232 | 0.128215 | PASS |
| decode 161.0 | lm_head Q6_K | 0.614282 | 0.614296 | 0.000013 | 0.050000 | PASS |
| decode 161.0 | norm | 0.366048 | 0.365544 | 0.000504 | 0.050000 | PASS |
| decode 161.0 | other | 0.240841 | 0.240686 | 0.000154 | 0.050000 | PASS |
| decode 161.0 | quantize | 0.407338 | 0.407178 | 0.000160 | 0.050000 | PASS |
| decode 161.0 | recurrent | 0.168583 | 0.168562 | 0.000020 | 0.050000 | PASS |
| decode 161.0 | rope | 0.045321 | 0.045265 | 0.000056 | 0.050000 | PASS |
| decode 161.0 | ssm_alpha_beta Q8_0 | 0.216790 | 0.216160 | 0.000630 | 0.050000 | PASS |
| decode 161.0 | ssm_out Q8_0 | 1.022729 | 1.022772 | 0.000043 | 0.050000 | PASS |

## 2. Gaps at 2048 — location prediction PASS; capture-only explanation unproved

Criterion/prediction: a few large gaps at fixed positions; one chunk removes the boundary gaps and confirms its own untraced time.

| run | launches in window | gaps ms | three largest intervals (zero-based) | capture error | 6% verdict |
|---|---|---|---|---|---|
| repA-prefill | 9915 | 38.493 | 33.074 ms before #7452; 1.629 ms before #5000; 1.627 ms before #2548 | +6.510% | FAIL |
| repB-prefill | 9915 | 39.283 | 33.655 ms before #7452; 1.745 ms before #2548; 1.727 ms before #5000 | +6.679% | FAIL |
| ub2048 | 2385 | 0.324 | 0.001 ms before #250; 0.001 ms before #929; 0.001 ms before #1814 | -1.923% | PASS |
| fix3-prefill | 9915 | 8.359 | 2.690 ms before #7452; 1.737 ms before #5000; 1.737 ms before #2548 | +0.746% | PASS |

The three large gaps in both four-chunk captures are before launches 2548, 5000 and 7452: after the chunk-ending Q6_K `mul_mat_q` and before the next chunk's `rms_norm_f32`. The dominant 33.074/33.655 ms interval is the third boundary, immediately before the chunk containing the output token. It is not an interval distributed across a kernel family. One chunk has 0.324 ms total idle and no interval above 1 ms. The corrected repetition has 8.359 ms idle and only 2.690 ms at that final boundary. A GPU window beginning at the first launch cannot measure idle before that first launch. Saved untraced first repetitions also carried extra host time; these data do not prove the original gaps are exclusively profiler overhead.

## 3. vLLM against llama.cpp — PASS modeled work; context scope FAIL

Criterion: equal operations for projections, attention, recurrent and head; only FFN's last-layer pruning may differ. Decode has equal shapes and calls; bytes may differ by format.

| length | row set | llama.cpp GFLOP | vLLM GFLOP | difference GFLOP | allowed GFLOP | verdict |
|---|---|---|---|---|---|---|
| 512 | projections | 7411.502940 | 7411.502940 | 0.000000 | 0.000000 | PASS |
| 512 | FFN | 17250.197176 | 17523.466568 | 273.269391 | 273.269391 | PASS |
| 512 | attention | 51.640271 | 51.640271 | 0.000000 | 0.000000 | PASS |
| 512 | recurrent | 137.304736 | 137.304736 | 0.000000 | 0.000000 | PASS |
| 512 | head | 2.542797 | 2.542797 | 0.000000 | 0.000000 | PASS |
| 2048 | projections | 29646.011761 | 29646.011761 | 0.000000 | 0.000000 | PASS |
| 2048 | FFN | 68999.184384 | 70093.866271 | 1094.681887 | 1094.681887 | PASS |
| 2048 | attention | 825.036374 | 825.036374 | 0.000000 | 0.000000 | PASS |
| 2048 | recurrent | 549.218943 | 549.218943 | 0.000000 | 0.000000 | PASS |
| 2048 | head | 2.542797 | 2.542797 | 0.000000 | 0.000000 | PASS |

Decode profiles have 12 equal role/shape/count entries and 51.247850 GFLOP per token. All packed parts retain shapes and no role is unpaired after re-attribution. Captures begin at 128 for both engines; llama.cpp samples 63 tokens and vLLM 31, so mean contexts are 161 and 145, not the requested 512. Equal projection work is established; equal attention work at the requested context is not.

## 4. Timer against Nsight Compute — FAIL prediction

Criterion: all cache checks within 2%; per-role errors within 2 percentage points of the prior after column; mean absolute error 6.3% ±2 points; fused errors −4% to −10%.

| role | generic less floor µs | oracle µs | error | prior after | change pp | 2-point verdict |
|---|---|---|---|---|---|---|
| attn_gate Q8_0 | 22.576 | 22.528 | +0.213% | not in prior table | — | unqualified |
| attn_kv Q8_0 | 5.616 | 5.536 | +1.445% | +6.2% | -4.755 | FAIL |
| attn_qkv Q8_0 | 34.880 | 35.776 | -2.504% | not in prior table | — | unqualified |
| attn_qo Q6_K | 18.208 | 19.552 | -6.874% | -8.3% | +1.426 | PASS |
| attn_qo Q8_0 | 41.072 | 42.288 | -2.876% | -4.1% | +1.224 | PASS |
| ffn_down Q4_K | 32.624 | 34.400 | -5.163% | -8.2% | +3.037 | FAIL |
| ffn_gate_up Q4_K | 31.616 | 34.160 | -7.447% | -9.3% | +1.853 | PASS |
| lm_head Q6_K | 614.384 | 615.120 | -0.120% | -0.3% | +0.180 | PASS |
| ssm_alpha_beta Q8_0 | 2.992 | 2.656 | +12.651% | -10.7% | +23.351 | FAIL |
| ssm_out Q8_0 | 22.432 | 22.528 | -0.426% | -3.1% | +2.674 | FAIL |

Cache checks: 10/10 pass; maximum DRAM/weight deviation 0.0460%. Mean absolute error is 3.972% over all ten rows, and 4.625% over the eight roles in the prior table. The ten-row mean is below 4.3–8.3%; the comparable eight-row mean passes. The expanded role set makes the aggregate comparison scope-dependent. The small recurrent row changes sign to +12.651%; fused FFN down remains −5.163%, gate/up −7.447%, attention Q6_K −6.874%. The Q8_0 attention output is −2.876%, outside the predicted fused range. No timer change was made.

## 5. The 8B — FAIL end-to-end; pairing/work and arithmetic PASS

Criterion: every role pairs, no ambiguity, kernels plus gaps equals captured within 0.001 ms, every capture within 6%, work equals the profile including last-layer FFN pruning.

| workload | paired row groups | ambiguity | tie-out residual ms | capture error | verdict |
|---|---|---|---|---|---|
| decode 512 requested | 8 | none | 0.000000 | +4.379% | PASS numerically; context 161 capture / 512 untraced |
| prefill 2048 | 9 | none | 0.000000 | +7.198% | FAIL |
| prefill 512 | 9 | none | 0.000000 | -6.646% | FAIL |

Mixed Q4_K/Q6_K projection rows now match the profile. At 512/2048, FFN work is 5,411.960783 / 21,646.937162 GFLOP; projections are 1,546.188227 / 6,184.752906 GFLOP; head 1.244660 GFLOP. The last layer is accounted at one output token; the Q6_K FFN-down row carries its pruning. There is no matrix-launch row without a role. Exact tie-out uses profiled_wall_ms (the attributed launch window), not captured_ms (the program-reported repetition time, 49.228 ms at 512). The two fields have different scope; treating them interchangeably would fail arithmetic.

## 6. Replay agent-60 — FAIL overall; efficiency prediction PASS

Criterion: finishes, captured turns within 10% of server prompt_ms except possibly early; no unpaired decode role and no mul_mat_q launch; tables include chip best. Prediction: E within five points of 62.5%.

| turn | capture | bench ms | server ms | error | pair ambiguity | verdict |
|---|---|---|---|---|---|---|
| 2 | early | 391.014 | 500.535 | -21.881% | no | FAIL (allowed early) |
| 31 | mid | 1210.670 | 1309.841 | -7.571% | yes | PASS |
| 60 | deep | 1463.736 | 1570.996 | -6.827% | no | PASS |
| 17 | biggest | 1053.559 | 1183.974 | -11.015% | no | FAIL |

All 60 turns finish. Warm E = 65.029% (change +2.529 points); with cold 65.168%. Decode has no unpaired role or ambiguity. The biggest turn misses the 10% gate at −11.015%. The mid capture retains an FFN-down twin ambiguity: this is an open accuracy bug. Literal “no mul_mat_q launch” fails: all four prefill captures use these kernels. They are the engine's real matrix path; absence of unpaired launches is the meaningful pairing check, but it does not replace the written criterion.

| measure | agent-60 b11571 |
|---|---|
| warm turns | 59 |
| deepest prefix | 131075 |
| TTFT median / p95 ms | 1057.1 / 2080.6 |
| decode tok/s median / deepest | 57.56 / 47.76 |
| roofline / chip best s | 379.719 / 374.250 |
| prefill / decode / overhead s | 61.794 / 516.213 / 5.917 |
| lost prefill / recompute / decode / overhead s | 47.174 / 1.041 / 150.073 / 5.917 |
| cache reuse | 97.398% |
| load s | 4.45 |
| throttled warm turns | 56 |

| turn | prefix | cache | new | recomputed | TTFT ms | prompt ms | overhead ms | reply | decode tok/s | ceiling ms | chip best ms | % of roofline | clock | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4,495 | 0 | 4,495 | 0 | 1210 | 1198 | 12 | 791 | 69.6 | 8994 | 8986 | 72% | 2890 | cold THROTTLED |
| 2 | 6,657 | 5,285 | 1,371 | 1 | 517 | 501 | 16 | 695 | 69.3 | 7792 | 7785 | 74% | 2886 | captured THROTTLED |
| 17 | 37,184 | 34,188 | 2,394 | 602 | 1252 | 1184 | 68 | 372 | 63.2 | 4589 | 4519 | 64% | 2826 | captured THROTTLED |
| 31 | 68,766 | 66,446 | 2,319 | 1 | 1424 | 1310 | 115 | 371 | 57.6 | 4900 | 4772 | 62% | 2811 | captured THROTTLED |
| 60 | 131,075 | 129,347 | 1,727 | 1 | 1752 | 1571 | 181 | 681 | 47.8 | 9684 | 9489 | 61% | 2641 | captured THROTTLED |
| warm |  | 3,893,946 | 96,854 | 7,160 | 67,684 | 61,794 | 5,890 | 29,616 |  | 379,719 | 374,250 | 65% |  |  |

The selected-turn table redraws the finished agent-60 run with chip best. The older chat-300 evidence is absent, as its original document records; its rows cannot be recomputed from this agent-60 session.

## Accuracy problems and open work

- Requested decode context is not used by vendor capture: both engines start at 128; their token-window means differ. The untraced 512 median therefore does not certify same-context capture timing.
- The replay mid FFN-down pairing remains ambiguous after all four fixes. It cannot support a fully attributed role verdict.
- Original 2048 prefill and both 8B prefill captures fail the 6% end-to-end gate. The second-repetition fix cannot recover a repetition never saved; this report retains those failures.
- Timer error for the smallest recurrent row changes sign and misses the per-role prediction by 23.351 points. Fusion and dispatch-floor uncertainty remain; the cache checks establish eviction only.
- Identical operations do not imply identical clock-scaled ceilings; the repeated 2048 captures change eight limits. A diff must distinguish work changes from measured peak changes.
- Prefill captured_ms and profiled_wall_ms name different time scopes. Kernel plus idle reconciles only to the latter; consumers must not interpret the former as that launch window.
- The replay biggest turn fails the 10% server gate, and the literal prediction of no mul_mat_q launches was wrong. No code was changed to force these results to pass.

Part 3 still needs the missing engines and model runs. Unknown or refused capture is not a pass.
