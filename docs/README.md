# BoltBeam docs

This is the index of every file in docs/. It has four parts: where to start, the records the published
articles cite, everything else, and the older working notes.

## Start here

- [CLI.md](CLI.md): the command line end to end, from reading a model file to the ledger.
- [architecture.md](architecture.md): how the pieces fit.
- [../tui/README.md](../tui/README.md): the screen (Setup, Run, Saved runs) and its JSON mode for agents.
- [kernel-timer.md](kernel-timer.md): the one kernel timing loop, its adapters (BoltBeam's GEMV, llama.cpp's Metal and CUDA kernels) and the bridges; the Air proof and the Ubuntu commands.
- [maintaining.md](maintaining.md): for people who work on BoltBeam itself: install for development, scripts,
  the public mirror.
- [branch-flow.md](branch-flow.md): what each branch holds and how the trunk reaches the others.

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
- [architecture.md](architecture.md): BoltBeam Architecture. Cited by: how-boltbeam-works.
- [decode-loss-stack-reachability-20260701.md](decode-loss-stack-reachability-20260701.md): 14B/32B Decode Loss Stack + Reachability (post-L3, post-attention-closure). Cited by: what-makes-a-kernel-fast.
- [g5-block-tile-result-20260701.md](g5-block-tile-result-20260701.md): G=5 Block Tile: BoltBeam Ledger Update. Cited by: what-makes-a-kernel-fast.

## Everything else

Scopes, results and runbooks. Each one stays at its path. A dated doc is here because something links to it:
another doc, the code, a branch, or the tinygrad fork's docs. An undated doc describes a part of the tool.

- [small-roles-rtx5090-20261010.md](small-roles-rtx5090-20261010.md): what the small roles could recover on the RTX 5090; the engine's own kernel at the fused q/k/v shapes (q+k+v Q4_K 9.9 µs less floor at 85% of peak): fusing recovers at most 0.13 ms of the 3.98 ms token (8 tok/s); batch 2 halves the small roles' cost per token (0.81 to 0.43 ms) and the step is 4.63 ms for two tokens (432 tok/s); the 2048-row Q6_K shape exists in no layer of this model
- [in-model-vs-generic-rtx5090-20261010.md](in-model-vs-generic-rtx5090-20261010.md): the same comparison on the RTX 5090; the store flush drains there too (7 to 12%, one rule: read sweep); the 4 µs event floor must come off; read sweep less floor matches nsys within 4% on 6 of 7 roles; the CUDA adapter needed a prefix symbol match and a q8_1 reference
- [in-model-vs-generic-m4-20261010.md](in-model-vs-generic-m4-20261010.md): in-model vs generic timing of the same llama.cpp kernels on the Apple M4; the kernel timer's store flush adds ~60 µs per launch (cold sum 1.24× the token, warm 0.96×); xctrace cannot split llama.cpp on Metal; ggml 0.26 embeds 35 metallibs
- [tinygrad-vs-llama-kernels-m4-20261009.md](tinygrad-vs-llama-kernels-m4-20261009.md): tinygrad's emitted kernels read memory 14 to 30% slower than llama.cpp's on the Apple M4; the rest of the gap is idle time between kernels
- [analyze-command-scope.md](analyze-command-scope.md): BoltBeam Analyze Command Scope
- [attention-combine-closure-stress-test-20260701.md](attention-combine-closure-stress-test-20260701.md): Attention Combine Closure Stress Test
- [attention-combine-reachability-audit-20260701.md](attention-combine-reachability-audit-20260701.md): Attention-Combine Reachability Audit (falsification pass 2)
- [audit-brain-build-scope.md](audit-brain-build-scope.md): BoltBeam Audit Brain Build Scope
- [candidate-identity-target-decoupling-scope-20260801.md](candidate-identity-target-decoupling-scope-20260801.md): Candidate identity: model/target decoupling: scope
- [coding-principles.md](coding-principles.md): BoltBeam Coding Principles
- [commit-discipline.md](commit-discipline.md): BoltBeam Commit Discipline
- [compiler-pathology-diagnostics-scope.md](compiler-pathology-diagnostics-scope.md): Compiler pathology diagnostics scope
- [default-path-handwritten-kernel-audit-20260701.md](default-path-handwritten-kernel-audit-20260701.md): Default-Path Handwritten Kernel Audit
- [dispatch-reconciliation.md](dispatch-reconciliation.md): Dispatch reconciliation
- [eb5-ingestion-report-20260701.md](eb5-ingestion-report-20260701.md): EB5: BoltBeam Ingestion Report
- [exp-metal-mr0-mr13-runbook.md](exp-metal-mr0-mr13-runbook.md): EXP Metal replay and generated-search runbook
- [full-kernel-candidate-set.md](full-kernel-candidate-set.md): Full-kernel candidate-set persistence
- [g5-generated-isa-primitive-route-scope-20260701.md](g5-generated-isa-primitive-route-scope-20260701.md): G=5 Generated-ISA Primitive Route Scope
- [g5-resource-oracle-and-konly-scope-20260701.md](g5-resource-oracle-and-konly-scope-20260701.md): G=5 Resource Oracle + K-Only Variant Scope
- [gpu-agnostic-profiler-build-scope-20260704.md](gpu-agnostic-profiler-build-scope-20260704.md): GPU-Agnostic Profiler Build Scope
- [gpu-health-trace-handoff-prompt-20260704.md](gpu-health-trace-handoff-prompt-20260704.md): GPU Health / 14B Trace Handoff Prompt
- [gpu-health-trace-results-20260704.md](gpu-health-trace-results-20260704.md): GPU Health / 14B Trace: Run Results (2026-07-04)
- [hwtrace-diagnostics-feature-scope-20260704.md](hwtrace-diagnostics-feature-scope-20260704.md): Scope: hw-trace diagnostics: rocprofv3-blind preflight (A) + PMC graph-bypass reason (B)
- [kv-cache-quantization-policy-scope.md](kv-cache-quantization-policy-scope.md): KV Cache Quantization Policy Scope
- [lds-vs-l2-analysis-scope-20260712.md](lds-vs-l2-analysis-scope-20260712.md): LDS-vs-L2 Analysis Scope (2026-07-12)
- [lds5-ledger-update-20260701.md](lds5-ledger-update-20260701.md): LDS5: BoltBeam Ledger Update
- [lds5-prologue-range-update-20260701.md](lds5-prologue-range-update-20260701.md): LDS5 / PR5: BoltBeam Ledger Update: Prologue Range LDS Staging Track
- [llama-rocprof-tracing.md](llama-rocprof-tracing.md): llama.cpp Hardware Trace Import
- [low-level-gpu-sampler-scope.md](low-level-gpu-sampler-scope.md): Low-Level GPU Sampler Scope
- [machine-search-research-categories-20260710.md](machine-search-research-categories-20260710.md): Machine Search Research Categories
- [mmq-epoch-model-exhaustive-scope-20260710.md](mmq-epoch-model-exhaustive-scope-20260710.md): MMQ Epoch Model Exhaustive Scope
- [model-agnostic-gap-audit.md](model-agnostic-gap-audit.md): Model-Agnostic Gap Audit (A0)
- [model-agnostic-search-roadmap.md](model-agnostic-search-roadmap.md): Model-Agnostic Search Roadmap
- [mr13-closure-manifest.md](mr13-closure-manifest.md): MR13 closure manifest
- [mr7-evidence-pipeline.md](mr7-evidence-pipeline.md): MR7 evidence pipeline
- [mr9-semantic-search.md](mr9-semantic-search.md): MR9 semantic search
- [native-pmc-sampler-and-graph-pmc-20260704.md](native-pmc-sampler-and-graph-pmc-20260704.md): Native-PMC sampler + graph-PMC attempt (2026-07-04)
- [operand-path-selection-e2e-scope-20260713.md](operand-path-selection-e2e-scope-20260713.md): End-to-End Operand Path Selection Scope
- [operand-path-selection-implementation-status-20260713.md](operand-path-selection-implementation-status-20260713.md): Operand-path selection implementation status
- [organization-scope-claude-20260730.md](organization-scope-claude-20260730.md): BoltBeam organization scope
- [organization-scope-qwen-20260730.md](organization-scope-qwen-20260730.md): BoltBeam organization scope: Qwen assessment
- [parity-14b-32b-scope.md](parity-14b-32b-scope.md): Using BoltBeam to Close 14B/32B Decode Parity (Scope)
- [portable-trace-profiles.md](portable-trace-profiles.md): Portable trace profiles
- [practical-roofline-promotion-scope-20260702.md](practical-roofline-promotion-scope-20260702.md): Practical Roofline Promotion Scope
- [prefill-policy-benchmark-protocol-20260712.md](prefill-policy-benchmark-protocol-20260712.md): Pure Prefill Policy Benchmark Protocol
- [principles-audit-20260701.md](principles-audit-20260701.md): BoltBeam Principles Audit: 2026-07-01
- [profiler-goal-alignment-mvp-20260704.md](profiler-goal-alignment-mvp-20260704.md): Profiler Goal, Alignment, and MVP
- [profiler-ontology-architecture-20260704.md](profiler-ontology-architecture-20260704.md): Profiler Portable Performance Ontology & Pipeline
- [provider-neutral-kernel-analysis-completion-scope-20260713.md](provider-neutral-kernel-analysis-completion-scope-20260713.md): Provider-Neutral Kernel Analysis Completion Scope
- [qwen3-8b-current-dual-roofline-20260713.md](qwen3-8b-current-dual-roofline-20260713.md): Qwen3 8B current dual roofline
- [qwen3-8b-p9-completion-handoff-20260713.md](qwen3-8b-p9-completion-handoff-20260713.md): Qwen3-8B P9 completion handoff
- [recommendation_14b_20260702.md](recommendation_14b_20260702.md): 14B Frontier Recommendation: 2026-07-02
- [roofline-audit-system-scope.md](roofline-audit-system-scope.md): BoltBeam Roofline Audit System Scope
- [semantic-campaign.md](semantic-campaign.md): Semantic campaign
- [system-fusion-sf4-ledger-report.md](system-fusion-sf4-ledger-report.md): System Fusion SF4: Ledger Report (BoltBeam)
- [tinygrad-boundary-plan-20260702.md](tinygrad-boundary-plan-20260702.md): Tinygrad / BoltBeam Boundary Plan
- [tinygrad-decode-authority-adapter.md](tinygrad-decode-authority-adapter.md): Tinygrad decode authority adapter
- [tinygrad-kfd-direct-bridge.md](tinygrad-kfd-direct-bridge.md): Direct KFD bridge boundary
- [tinygrad-kfd-sidecar-seam.md](tinygrad-kfd-sidecar-seam.md): Tinygrad direct-KFD launch sidecar seam
- [tinygrad-prefill-adapter.md](tinygrad-prefill-adapter.md): Tinygrad prefill adapter

Folders:

- [lifecycle/](lifecycle/): lifecycle comparison records (one model on one GPU, JSON and its write-up).
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
