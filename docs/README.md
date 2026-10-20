# BoltBeam docs

These pages describe BoltBeam as it is today. The dated measurement records, scopes and handoffs behind them
are kept with the development branch, not here; a page that cites one names it by file.

## Start here

Read these in order.

- [the top-level README](../README.md): what BoltBeam answers, install, and the five-minute first run (doctor, one audit, reading it, running it twice)
- [guide-time-your-kernel.md](guide-time-your-kernel.md): time one hand-written kernel in BoltBeam's kernel timer and read it against the measured roofline; a Metal GEMV worked on the M3; the timer's measured limits
- [guide-add-an-engine.md](guide-add-an-engine.md): what an engine adapter must return (truth timing, capture, kernel paths, formats, chunk record, version for the pin table, toolchain), grounded in the five existing adapters
- [CLI.md](CLI.md): the command line end to end, from reading a model file to the ledger.
- [tui/README.md](../tui/README.md): the screen (Setup, Run, Saved runs, Compare) and its JSON mode for agents.
- [engine-upgrade-audit.md](engine-upgrade-audit.md): how to audit an engine upgrade: the pin table, two builds on one setup, repeat bands, what it refuses and what it reports
- [linux.md](linux.md): Linux with an NVIDIA GPU, from a bare Ubuntu to the first report: CUDA on PATH, nsys, llama.cpp with CUDA, the model, the optional engines, `boltbeam doctor`, `boltbeam tui`.
- [h100-pod.md](h100-pod.md): a rented H100 (RunPod) from the bare image to the first report with one command, `tools/pod-setup.sh`; what to rent, what comes back, the cost, what the multi-GPU session adds.

## How a part works

Each describes one part of the tool and is kept current with it.

- [architecture.md](architecture.md): how the pieces fit.
- [kernel-timer.md](kernel-timer.md): the one kernel timing loop, its adapters (BoltBeam's GEMV, llama.cpp's Metal and CUDA kernels) and the bridges; the Air proof and the Ubuntu commands.
- [pairing-layer-walk-20261011.md](pairing-layer-walk-20261011.md): pairing kernels with roles: the layer walk, one rule set for prefill and decode (collectors/pairing.py)
- [replay-resource-accounting-20261019.md](replay-resource-accounting-20261019.md): the full-replay resource report: what it accounts for on every native launch, and what it does not claim
- [llama-rocprof-tracing.md](llama-rocprof-tracing.md): llama.cpp hardware trace import
- [full-kernel-candidate-set.md](full-kernel-candidate-set.md): full-kernel candidate-set persistence
- [semantic-campaign.md](semantic-campaign.md): the semantic campaign
- [mr7-evidence-pipeline.md](mr7-evidence-pipeline.md): the MR7 evidence pipeline
- [mr9-semantic-search.md](mr9-semantic-search.md): MR9 semantic search
- [mr13-closure-manifest.md](mr13-closure-manifest.md): the MR13 closure manifest
- [tinygrad-decode-authority-adapter.md](tinygrad-decode-authority-adapter.md): the tinygrad decode authority adapter
- [cuda-build-limits.md](cuda-build-limits.md): bounded CUDA builds (tools/prebuild_cuda.py, tools/run_bounded.py)

## Working on BoltBeam

- [maintaining.md](maintaining.md): for people who work on BoltBeam itself: install for development, scripts, the public mirror.
- [release-runbook.md](release-runbook.md): a new version on Homebrew, from the version bump to the user's `brew upgrade`.
- [branch-flow.md](branch-flow.md): what each branch holds and how the trunk reaches the others.
- [coding-principles.md](coding-principles.md): BoltBeam coding principles
- [commit-discipline.md](commit-discipline.md): BoltBeam commit discipline

## Records the articles cite

The write-ups on [research.arkey.ai](https://research.arkey.ai) link to these files by their path in the
public mirror, so they stay here as written. Never move or rename them. The articles also cite files outside
docs/: `boltbeam/data/targets.json`, `boltbeam/data/candidates.json`, `boltbeam/data/quants.json`,
`boltbeam/quantization/quant_gemv.py`, `bench/nvidia-sm120-facts-20260923/README.md`,
`bench/nvidia-sm120-prefill-20260923/README.md`, the top-level README and [architecture.md](architecture.md).

- [14b-aggregate-fusion-closeout-20260702.md](14b-aggregate-fusion-closeout-20260702.md): 14B decode: aggregate elementwise fusion result. Cited by: how-boltbeam-works, the-roofline-holds-on-a-second-gpu.
- [14b-decode-roofline-attribution-20260702.md](14b-decode-roofline-attribution-20260702.md): 14B decode: roofline attribution. Cited by: what-makes-a-kernel-fast.
- [14b-decode-vdot2-gemv-result-20260702.md](14b-decode-vdot2-gemv-result-20260702.md): 14B decode: v_dot2 wired into the Q4_K GEMV. Cited by: what-makes-a-kernel-fast, how-boltbeam-works.
- [14b-practical-roofline-audit-20260702.md](14b-practical-roofline-audit-20260702.md): 14B practical roofline audit. Cited by: how-boltbeam-works.
- [decode-loss-stack-reachability-20260701.md](decode-loss-stack-reachability-20260701.md): 14B/32B decode loss stack and reachability. Cited by: what-makes-a-kernel-fast.
- [g5-block-tile-result-20260701.md](g5-block-tile-result-20260701.md): G=5 block tile: ledger update. Cited by: what-makes-a-kernel-fast.
