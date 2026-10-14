# BoltBeam docs

There are about 100 files here. Most are dated records of one measurement or one change, kept so their numbers
can be checked. A newcomer needs only the first section. The records never move: articles, code and other
records link to them by path.

## Start here

Read these in order. They describe BoltBeam as it is today.

- [the top-level README](../README.md): what BoltBeam answers, install, and the five-minute first run (doctor, one audit, reading it, running it twice)
- [guide-time-your-kernel.md](guide-time-your-kernel.md): time one hand-written kernel in BoltBeam's kernel timer and read it against the measured roofline; a Metal GEMV worked on the M3; the timer's measured limits
- [guide-add-an-engine.md](guide-add-an-engine.md): what an engine adapter must return (truth timing, capture, kernel paths, formats, chunk record, version for the pin table, toolchain), grounded in the five existing adapters
- [CLI.md](CLI.md): the command line end to end, from reading a model file to the ledger.
- [tui/README.md](../tui/README.md): the screen (Setup, Run, Saved runs, Compare) and its JSON mode for agents.
- [engine-upgrade-audit.md](engine-upgrade-audit.md): how to audit an engine upgrade: the pin table, two builds on one setup, repeat bands, what it refuses and what it reports
- [linux.md](linux.md): Linux with an NVIDIA GPU, from a bare Ubuntu to the first report: CUDA on PATH, nsys, llama.cpp with CUDA, the model, the optional engines, `boltbeam doctor`, `boltbeam tui`.
- [h100-pod.md](h100-pod.md): a rented H100 (RunPod) from the bare image to the first report with one command, `tools/pod-setup.sh`; what to rent, what comes back, the cost, what the multi-GPU session adds.

- [short-context-quants-rtx5090-20261014.md](short-context-quants-rtx5090-20261014.md): Qwen3.8 short-context graphs-on comparison and same-GGUF engine control.

## How a part works

Undated reference pages. Each describes one part of the tool and is kept current with it.

- [architecture.md](architecture.md): how the pieces fit.
- [kernel-timer.md](kernel-timer.md): the one kernel timing loop, its adapters (BoltBeam's GEMV, llama.cpp's Metal and CUDA kernels) and the bridges; the Air proof and the Ubuntu commands.
- [dispatch-reconciliation.md](dispatch-reconciliation.md): Dispatch reconciliation
- [portable-trace-profiles.md](portable-trace-profiles.md): Portable trace profiles
- [llama-rocprof-tracing.md](llama-rocprof-tracing.md): llama.cpp Hardware Trace Import
- [full-kernel-candidate-set.md](full-kernel-candidate-set.md): Full-kernel candidate-set persistence
- [semantic-campaign.md](semantic-campaign.md): Semantic campaign
- [mr7-evidence-pipeline.md](mr7-evidence-pipeline.md): MR7 evidence pipeline
- [mr9-semantic-search.md](mr9-semantic-search.md): MR9 semantic search
- [mr13-closure-manifest.md](mr13-closure-manifest.md): MR13 closure manifest
- [exp-metal-mr0-mr13-runbook.md](exp-metal-mr0-mr13-runbook.md): EXP Metal replay and generated-search runbook
- [tinygrad-decode-authority-adapter.md](tinygrad-decode-authority-adapter.md): Tinygrad decode authority adapter
- [tinygrad-prefill-adapter.md](tinygrad-prefill-adapter.md): Tinygrad prefill adapter
- [tinygrad-kfd-direct-bridge.md](tinygrad-kfd-direct-bridge.md): Direct KFD bridge boundary
- [tinygrad-kfd-sidecar-seam.md](tinygrad-kfd-sidecar-seam.md): Tinygrad direct-KFD launch sidecar seam

## Working on BoltBeam

- [maintaining.md](maintaining.md): for people who work on BoltBeam itself: install for development, scripts, the public mirror.
- [release-runbook.md](release-runbook.md): a new version on Homebrew, from the version bump to the user's `brew upgrade`.
- [branch-flow.md](branch-flow.md): what each branch holds and how the trunk reaches the others.
- [coding-principles.md](coding-principles.md): BoltBeam Coding Principles
- [commit-discipline.md](commit-discipline.md): BoltBeam Commit Discipline

## Audit records, October 2026

- [Q4_K_M versus AWQ and NVFP4](quant-comparison-rtx5090-20261014.md): matched replay, engine versus format distinctions, tuning evidence and FP8-cache capacity.

The current measurement work, newest first. Each is a dated record: what was run, the numbers, what passed and
what failed. Later records can correct earlier ones; the correction says so and the earlier record stays as written.

- [loop-test-27b-prefill-rtx5090-20261012.md](loop-test-27b-prefill-rtx5090-20261012.md): one loop test: a GDN kernel change in llama.cpp, the 27B prefill on the RTX 5090; the recurrent row moved, the rest held
- [handoff-accuracy-next-20261012.md](handoff-accuracy-next-20261012.md): handoff: the remaining accuracy work
- [handoff-accuracy-20261012.md](handoff-accuracy-20261012.md): handoff: engine versions, the accuracy battery and the missing engine runs
- [engine-upgrade-llama-b11571-rtx5090-20261012.md](engine-upgrade-llama-b11571-rtx5090-20261012.md): the first engine-upgrade audit: llama.cpp b9592 to b11571 on the 27B, RTX 5090
- [accuracy-next-20261012.md](accuracy-next-20261012.md): the remaining accuracy causes: Q/O pairing, repeat sampling, the timer limit, where the overhead sits
- [accuracy-fixes-20261012.md](accuracy-fixes-20261012.md): the corrections after the battery: pairing without timing twins, tinygrad launch semantics, what the kernel-timer discrepancy measures, fresh repeat bands per engine
- [accuracy-battery-rtx5090-20261012.md](accuracy-battery-rtx5090-20261012.md): the accuracy battery on the RTX 5090: predictions written first, then pairing, work, tie-out and end-to-end gates per engine and model; the timer against Nsight Compute; failures kept as failures
- [session-27b-rtx5090-20261011.md](session-27b-rtx5090-20261011.md): a chat session at depth for the 27B hybrid on the RTX 5090; the model cap 262,144 (qwen35.context_length), the f16 memory cap 197,660 binds, q8_0 binds at the model cap; llama.cpp turn of 512 tokens 125 / 129 / 143 / 260 / 607 ms and decode 72.0 to 45.5 tok/s from 1K to 197,084; the fork 25.4 s per turn (token at a time); compaction 520 ms, under one turn at the cap; where the turn goes at 16K; the power cap throttles every llama.cpp depth
- [replay-27b-rtx5090-20261011.md](replay-27b-rtx5090-20261011.md): the replay workload: a recorded session through the user's own llama-server, per turn and per session against the roofline; the 27B on the RTX 5090
- [prefill-vllm-crosscheck-8b-rtx5090-20261011.md](prefill-vllm-crosscheck-8b-rtx5090-20261011.md): prefill cross-check: vLLM against llama.cpp on the 8B, RTX 5090
- [prefill-vllm-crosscheck-27b-rtx5090-20261011.md](prefill-vllm-crosscheck-27b-rtx5090-20261011.md): cross-check: vLLM against llama.cpp on the 27B hybrid prefill, RTX 5090; same work, same ceilings
- [prefill-audit-27b-rtx5090-20261011.md](prefill-audit-27b-rtx5090-20261011.md): the 27B hybrid on llama.cpp, prompt lengths from its own cap (model 262,144, memory 224,256 binds and fails to create a context); 127.8 / 515 / 2,112 / 9,540 / 67,672 ms at 512 / 2K / 8K / 32K / 131K, MFU 20 to 24%, throttled at 600 W; attention 1% to 15% of the time by 32K, takes over by 131K
- [pairing-layer-walk-20261011.md](pairing-layer-walk-20261011.md): pairing kernels with roles: the layer walk, one rule set for prefill and decode (collectors/pairing.py)
- [ncu-instruction-mix-27b-rtx5090-20261011.md](ncu-instruction-mix-27b-rtx5090-20261011.md): what Nsight Compute showed closed kernels execute (27B vLLM prefill at 512); folded back into the name rules
- [handoff-cleanup-20261011.md](handoff-cleanup-20261011.md): handoff: main's Go tests, then cleanup items 3 to 7; its Rules section is the working rules for this repo
- [dead-code-sweep-20261011.md](dead-code-sweep-20261011.md): dead-code sweep after the cleanup
- [cleanup-review-20261011.md](cleanup-review-20261011.md): cleanup review: verdicts and timing authorities
- [apparatus-boundary-20261011.md](apparatus-boundary-20261011.md): apparatus boundary follow-up: which modules are product and which are dev apparatus, by current callers
- [small-roles-rtx5090-20261010.md](small-roles-rtx5090-20261010.md): what the small roles could recover on the RTX 5090; the engine's own kernel at the fused q/k/v shapes (q+k+v Q4_K 9.9 µs less floor at 85% of peak): fusing recovers at most 0.13 ms of the 3.98 ms token (8 tok/s); batch 2 halves the small roles' cost per token (0.81 to 0.43 ms) and the step is 4.63 ms for two tokens (432 tok/s); the 2048-row Q6_K shape exists in no layer of this model
- [prefill-8b-rtx5090-20261010.md](prefill-8b-rtx5090-20261010.md): prefill measured faithfully on the RTX 5090; measured peaks per matrix path (fp16 251.4 TF to fp4 1,988.9 at about 2.88 GHz, 99.5 to 99.8% of the architectural rate per clock); llama.cpp 34.6 / 141.6 / 631.4 / 3,622 ms at 512 / 2K / 8K / 32K, MFU 21 to 47% (GEMMs flat near 30% of int8, attention rising to 84%); the fork 61.4 ms at 512 and out of GPU memory from 2K; where the prefill time goes at 2K
- [kernel-search-generic-20261010.md](kernel-search-generic-20261010.md): the kernel search on any GPU, and BoltBeam's check of the provider
- [kernel-search-fusion-20261010.md](kernel-search-fusion-20261010.md): fusion candidates in the kernel search, and BoltBeam's own timing
- [in-model-vs-generic-rtx5090-20261010.md](in-model-vs-generic-rtx5090-20261010.md): the same comparison on the RTX 5090; the store flush drains there too (7 to 12%, one rule: read sweep); the 4 µs event floor must come off; read sweep less floor matches nsys within 4% on 6 of 7 roles; the CUDA adapter needed a prefix symbol match and a q8_1 reference
- [in-model-vs-generic-m4-20261010.md](in-model-vs-generic-m4-20261010.md): in-model vs generic timing of the same llama.cpp kernels on the Apple M4; the kernel timer's store flush adds ~60 µs per launch (cold sum 1.24× the token, warm 0.96×); xctrace cannot split llama.cpp on Metal; ggml 0.26 embeds 35 metallibs
- [27b-hybrid-attribution-rtx5090-20261010.md](27b-hybrid-attribution-rtx5090-20261010.md): the first 27B (hybrid, 48 delta-net layers) on the 5090 broke both paths; attribution now pairs every role at once by bytes per launch (gate+up fused and ffn_down both 64 per token), a GEMV no role took is a weight kernel on its own line, a role with no adapter is not timed and out of the floor rule; the Q8_0 adapter on both backends; the profile misses the delta-net in-projections (4.2 GB per token), found and not fixed
- [tinygrad-vs-llama-kernels-m4-20261009.md](tinygrad-vs-llama-kernels-m4-20261009.md): tinygrad's emitted kernels read memory 14 to 30% slower than llama.cpp's on the Apple M4; the rest of the gap is idle time between kernels

## Records the articles cite

The write-ups on [research.arkey.ai](https://research.arkey.ai) link to these files by their path in the
public mirror. Never move or rename them. The articles also cite files outside docs/:
`boltbeam/data/targets.json`, `boltbeam/data/candidates.json`, `boltbeam/data/quants.json`,
`boltbeam/quantization/quant_gemv.py`, `bench/nvidia-sm120-facts-20260923/README.md`,
`bench/nvidia-sm120-prefill-20260923/README.md` and the top-level README.

- [14b-aggregate-fusion-closeout-20260702.md](14b-aggregate-fusion-closeout-20260702.md): 14B decode: aggregate elementwise fusion result + route-search closeout (2026-07-02). Cited by: how-boltbeam-works, the-roofline-holds-on-a-second-gpu.
- [14b-decode-roofline-attribution-20260702.md](14b-decode-roofline-attribution-20260702.md): 14B decode: roofline attribution & in-scope convergence (2026-07-02). Cited by: what-makes-a-kernel-fast.
- [14b-decode-vdot2-gemv-result-20260702.md](14b-decode-vdot2-gemv-result-20260702.md): 14B decode: v_dot2 wired into the Q4_K GEMV (codegen improvement): 2026-07-02. Cited by: what-makes-a-kernel-fast, how-boltbeam-works.
- [14b-practical-roofline-audit-20260702.md](14b-practical-roofline-audit-20260702.md): 14B Practical Roofline Audit. Cited by: how-boltbeam-works.
- [architecture.md](architecture.md): how the pieces fit.
- [decode-loss-stack-reachability-20260701.md](decode-loss-stack-reachability-20260701.md): 14B/32B Decode Loss Stack + Reachability (post-L3, post-attention-closure). Cited by: what-makes-a-kernel-fast.
- [g5-block-tile-result-20260701.md](g5-block-tile-result-20260701.md): G=5 Block Tile: BoltBeam Ledger Update. Cited by: what-makes-a-kernel-fast.

## Earlier records and scopes, July and August 2026

Scopes, results and runbooks from the earlier route-search work, newest first. Each stays at its path because
something links to it: another doc, the code, a branch, or the tinygrad fork's docs.

- [candidate-identity-target-decoupling-scope-20260801.md](candidate-identity-target-decoupling-scope-20260801.md): Candidate identity: model/target decoupling: scope
- [organization-scope-qwen-20260730.md](organization-scope-qwen-20260730.md): BoltBeam organization scope: Qwen assessment
- [organization-scope-claude-20260730.md](organization-scope-claude-20260730.md): BoltBeam organization scope
- [qwen3-8b-p9-completion-handoff-20260713.md](qwen3-8b-p9-completion-handoff-20260713.md): Qwen3-8B P9 completion handoff
- [qwen3-8b-current-dual-roofline-20260713.md](qwen3-8b-current-dual-roofline-20260713.md): Qwen3 8B current dual roofline
- [provider-neutral-kernel-analysis-completion-scope-20260713.md](provider-neutral-kernel-analysis-completion-scope-20260713.md): Provider-Neutral Kernel Analysis Completion Scope
- [operand-path-selection-implementation-status-20260713.md](operand-path-selection-implementation-status-20260713.md): Operand-path selection implementation status
- [operand-path-selection-e2e-scope-20260713.md](operand-path-selection-e2e-scope-20260713.md): End-to-End Operand Path Selection Scope
- [prefill-policy-benchmark-protocol-20260712.md](prefill-policy-benchmark-protocol-20260712.md): Pure Prefill Policy Benchmark Protocol
- [lds-vs-l2-analysis-scope-20260712.md](lds-vs-l2-analysis-scope-20260712.md): LDS-vs-L2 Analysis Scope (2026-07-12)
- [mmq-epoch-model-exhaustive-scope-20260710.md](mmq-epoch-model-exhaustive-scope-20260710.md): MMQ Epoch Model Exhaustive Scope
- [machine-search-research-categories-20260710.md](machine-search-research-categories-20260710.md): Machine Search Research Categories
- [profiler-ontology-architecture-20260704.md](profiler-ontology-architecture-20260704.md): Profiler Portable Performance Ontology & Pipeline
- [profiler-goal-alignment-mvp-20260704.md](profiler-goal-alignment-mvp-20260704.md): Profiler Goal, Alignment, and MVP
- [native-pmc-sampler-and-graph-pmc-20260704.md](native-pmc-sampler-and-graph-pmc-20260704.md): Native-PMC sampler + graph-PMC attempt (2026-07-04)
- [hwtrace-diagnostics-feature-scope-20260704.md](hwtrace-diagnostics-feature-scope-20260704.md): Scope: hw-trace diagnostics: rocprofv3-blind preflight (A) + PMC graph-bypass reason (B)
- [gpu-health-trace-results-20260704.md](gpu-health-trace-results-20260704.md): GPU Health / 14B Trace: Run Results (2026-07-04)
- [gpu-health-trace-handoff-prompt-20260704.md](gpu-health-trace-handoff-prompt-20260704.md): GPU Health / 14B Trace Handoff Prompt
- [gpu-agnostic-profiler-build-scope-20260704.md](gpu-agnostic-profiler-build-scope-20260704.md): GPU-Agnostic Profiler Build Scope
- [tinygrad-boundary-plan-20260702.md](tinygrad-boundary-plan-20260702.md): Tinygrad / BoltBeam Boundary Plan
- [recommendation_14b_20260702.md](recommendation_14b_20260702.md): 14B Frontier Recommendation: 2026-07-02
- [practical-roofline-promotion-scope-20260702.md](practical-roofline-promotion-scope-20260702.md): Practical Roofline Promotion Scope
- [principles-audit-20260701.md](principles-audit-20260701.md): BoltBeam Principles Audit: 2026-07-01
- [lds5-prologue-range-update-20260701.md](lds5-prologue-range-update-20260701.md): LDS5 / PR5: BoltBeam Ledger Update: Prologue Range LDS Staging Track
- [lds5-ledger-update-20260701.md](lds5-ledger-update-20260701.md): LDS5: BoltBeam Ledger Update
- [g5-resource-oracle-and-konly-scope-20260701.md](g5-resource-oracle-and-konly-scope-20260701.md): G=5 Resource Oracle + K-Only Variant Scope
- [g5-generated-isa-primitive-route-scope-20260701.md](g5-generated-isa-primitive-route-scope-20260701.md): G=5 Generated-ISA Primitive Route Scope
- [eb5-ingestion-report-20260701.md](eb5-ingestion-report-20260701.md): EB5: BoltBeam Ingestion Report
- [default-path-handwritten-kernel-audit-20260701.md](default-path-handwritten-kernel-audit-20260701.md): Default-Path Handwritten Kernel Audit
- [attention-combine-reachability-audit-20260701.md](attention-combine-reachability-audit-20260701.md): Attention-Combine Reachability Audit (falsification pass 2)
- [attention-combine-closure-stress-test-20260701.md](attention-combine-closure-stress-test-20260701.md): Attention Combine Closure Stress Test
- [system-fusion-sf4-ledger-report.md](system-fusion-sf4-ledger-report.md): System Fusion SF4: Ledger Report (BoltBeam)
- [roofline-audit-system-scope.md](roofline-audit-system-scope.md): BoltBeam Roofline Audit System Scope
- [parity-14b-32b-scope.md](parity-14b-32b-scope.md): Using BoltBeam to Close 14B/32B Decode Parity (Scope)
- [model-agnostic-search-roadmap.md](model-agnostic-search-roadmap.md): Model-Agnostic Search Roadmap
- [model-agnostic-gap-audit.md](model-agnostic-gap-audit.md): Model-Agnostic Gap Audit (A0)
- [low-level-gpu-sampler-scope.md](low-level-gpu-sampler-scope.md): Low-Level GPU Sampler Scope
- [kv-cache-quantization-policy-scope.md](kv-cache-quantization-policy-scope.md): KV Cache Quantization Policy Scope
- [compiler-pathology-diagnostics-scope.md](compiler-pathology-diagnostics-scope.md): Compiler pathology diagnostics scope
- [audit-brain-build-scope.md](audit-brain-build-scope.md): BoltBeam Audit Brain Build Scope
- [analyze-command-scope.md](analyze-command-scope.md): BoltBeam Analyze Command Scope

Folders:

- [lifecycle/](lifecycle/): lifecycle comparison records (one model on one GPU, JSON and its write-up).
- [graveyard/](graveyard/): one file per symptom that was found and fixed, with what hid it.
- [task_workflow/](task_workflow/README.md): how delegated work is specified (input/) and recorded (output/).

## Older working notes

[notes/](notes/) holds dated working notes that nothing links to: no file in the repo on any branch, no
published article, and no doc in the tinygrad fork. They moved here on 2026-10-09 to keep this folder
readable. They are kept as written.

- [artifact-cache-inventory-20260702.md](notes/artifact-cache-inventory-20260702.md): Artifact Cache Inventory
- [boltbeam-tinygrad-pure-pipe-integration-scope-20260712.md](notes/boltbeam-tinygrad-pure-pipe-integration-scope-20260712.md): BoltBeam-to-tinygrad pure pipe integration scope
- [cooperative-mmq-validation-matrix-20260715.md](notes/cooperative-mmq-validation-matrix-20260715.md): Cooperative-MMQ validation matrix (14B)
- [cost-oracle-sourced-quantities-scope-20260712.md](notes/cost-oracle-sourced-quantities-scope-20260712.md): Cost Oracle: Sourced-Quantity Route Decision: Scope
- [derived-practical-roofline-14b-20260702.md](notes/derived-practical-roofline-14b-20260702.md): 14B Derived Practical Roofline
- [eb0-boundary-contract-20260701.md](notes/eb0-boundary-contract-20260701.md): EB0: Emitter Blocked Boundary Contract
- [fable_handoff_14b_20260702.md](notes/fable_handoff_14b_20260702.md): Fable 5 Handoff: 14B Aggregate System Fusion Candidate
- [gp5-ledger-update-20260701.md](notes/gp5-ledger-update-20260701.md): GP5 Ledger Update: 2026-07-01
- [kv-cache-policy-principles-audit-20260701.md](notes/kv-cache-policy-principles-audit-20260701.md): KV Cache Policy Principles Audit
- [mmq-closed-system-loop-implementation-scope-20260711.md](notes/mmq-closed-system-loop-implementation-scope-20260711.md): MMQ Closed-System Loop Implementation Scope
- [mmq-complete-knowledge-implementation-scope-20260711.md](notes/mmq-complete-knowledge-implementation-scope-20260711.md): MMQ Complete-Knowledge Implementation Scope
- [mmq-complete-knowledge-result-20260711.md](notes/mmq-complete-knowledge-result-20260711.md): MMQ Complete-Knowledge Result
- [mmq-machine-search-done-criteria-20260710.md](notes/mmq-machine-search-done-criteria-20260710.md): MMQ Machine Search Done Criteria
- [mmq-writeback-closed-loop-result-20260711.md](notes/mmq-writeback-closed-loop-result-20260711.md): MMQ Writeback Closed-Loop Result
- [model-format-adapters-scope-result-20260701.md](notes/model-format-adapters-scope-result-20260701.md): Model Format Adapters Scope + Result
- [prefill-why-trace-workflow-20260704.md](notes/prefill-why-trace-workflow-20260704.md): 14B Prefill "Why" Trace Workflow
- [pure-machine-search-default-resolution-scope-20260701.md](notes/pure-machine-search-default-resolution-scope-20260701.md): Pure Machine Search Default Resolution Scope
- [pure-search-exhaustive-scope-20260702.md](notes/pure-search-exhaustive-scope-20260702.md): Pure Machine Search: Exhaustive Scope
- [q4-q6-prefill-hardcoded-assumption-audit-20260715.md](notes/q4-q6-prefill-hardcoded-assumption-audit-20260715.md): Q4/Q6 prefill hard-coded-assumption audit
- [qwen3-14b-authority-roofline-bounded-20260715.md](notes/qwen3-14b-authority-roofline-bounded-20260715.md): Qwen3-14B authority-aligned prefill roofline: bounded capture
- [qwen3-14b-generated-vs-llama-parity-plan-20260714.md](notes/qwen3-14b-generated-vs-llama-parity-plan-20260714.md): Qwen3-14B generated-prefill parity plan
- [qwen35-hybrid-support-result-20260701.md](notes/qwen35-hybrid-support-result-20260701.md): Qwen3.5 Hybrid Support Result
- [real-8b-format-validation-20260701.md](notes/real-8b-format-validation-20260701.md): Real 8B Format Validation
- [refreshed_loss_stack_14b_20260702.md](notes/refreshed_loss_stack_14b_20260702.md): 14B Decode Loss Stack: Refreshed Under Current Promoted Defaults
- [tg-p10-reg-scalar-combine-lowering-scope-20260701.md](notes/tg-p10-reg-scalar-combine-lowering-scope-20260701.md): TG-P10 Scope: REG Scalar Combine Lowering For Pure Attention
- [tinygrad-decouple-runner-contracts-20260702.md](notes/tinygrad-decouple-runner-contracts-20260702.md): Tinygrad Decouple Runner Contracts
