# The engine-upgrade audit: run it before any engine upgrade

An engine upgrade changes what BoltBeam measures. Before an engine's pin moves, audit the old build and the new
build of the same engine on the same model, chip and workload, and compare them row by row.

## The pin

Every engine BoltBeam measures has one pinned version, in one table:
`boltbeam/data/engine_pins.json`.

- llama.cpp: a release build tag (`bNNNNN`) and its full commit sha. Never moving master.
- vLLM and TensorRT-LLM: their release versions.
- The tinygrad fork: its checked-out commit.
- Ollama: its version, or `unknown` with the reason when the binary reports none.

Each entry keeps a `history` of measured versions, with date and evidence document.
An installed engine that has not been measured has an empty history. The pin file also
holds the current toolchain: driver, CUDA toolkit, torch, Nsight Systems and Nsight Compute.
Pins describe the reference installation; they never supply missing runtime observations.

Every new run manifest records `toolchain` with a UTC timestamp and observed versions.
Measurement steps refresh it before running, using the selected engine’s Python for torch.
Missing tools record `unknown`. The audit reports each tool’s before and after versions
and current pin; older manifests with no toolchain remain explicitly unknown.

Every run records the build it measured. llama.cpp reports `build_commit`; vLLM reports its version. The run's
manifest carries it under `engine`, with the pin check (`providers.record_build`). When the build differs from the
pin, the manifest and the audit say so plainly.

Changing a pin is its own `[target]` or `[repo]` commit. The commit cites the upgrade audit that justified it.
That audit must say "pin change: this audit can justify it". It says so only when the before build's band has at
least 4 runs and that n was fixed by a plan written before the runs were collected (see below). With fewer runs,
or no plan, the audit says "pin change: NOT justified" and names why.

## How to run it

1. Run the old build's audit, unless a saved run of that exact build exists:
   `python -m boltbeam.workflow.screen pipeline --run OLD --provider llama.cpp --workload prefill --ctxs 512,2048 --analyze --measure auto --no-search MODEL`
   with `BOLTBEAM_LLAMA_BENCH` pointing at the old build. Then the same with `--workload decode --ctxs 512`.
2. Run the same for the new build, into NEW.
3. Compare: `python -m boltbeam.workflow.screen diff --before OLD --after NEW`.
   (`replay-diff` is the same command. Two replays compare turn by turn; two prefill or decode audits compare
   row by row.)

Repeat calibration is separate from the upgrade run. Fix n first, then collect n independent runs of the old
build's exact setup, back to back, then derive the band without a GPU:

```sh
python -m boltbeam.workflow.screen repeat-plan --n 4 --out plan.json        # before any run is collected
python -m boltbeam.workflow.screen repeat-bands --runs R1 R2 R3 R4 --plan plan.json --out bands.json
python -m boltbeam.workflow.screen diff --before OLD --after NEW --bands bands.json
```

The rule: a row moved when the after value lies outside the range of the n before runs. With no real change and
exchangeable runs, that happens with probability 2/(n+1) per row: 67% at n = 2, 50% at n = 3, 40% at n = 4,
20% at n = 9. The band records n beside its rows, and every verdict prints n and that rate. A band with fewer
than 4 runs is labelled weak. n = 2 is allowed for a first look; it cannot justify a pin.

The plan fixes n before collection. A band built with it refuses a different number of runs (no stopping early
or late) and any run dated before the plan. Only the before build's band is needed: the after value is judged
against it. A run that is one of the band's own runs is refused as the after side: a holdout is never judged by
a band it made. Bands saved as pairs before n was recorded read as n = 2.

The artifact retains calibration history, source dates, engine builds and evidence hashes. A new pipeline can
load it with `--bands bands.json`. A missing matching calibration leaves the numerical comparison visible and
marks its judgment "not checked". No extra GPU repeats are automatic.

## What it refuses

It refuses, naming the field, when the two runs differ in workload, model file, target or engine. It refuses when
either run recorded no engine build: an unknown build is never compared. These are replay-diff's refusal rules,
from one place (`workflow/diff.refuse_unless_same`).

## What it reports

- Its header names the pin and each run's build against it.
- Toolchain changes, including driver, CUDA and torch, are named alongside the engine build.
- Each row, per prompt length (prefill) or per context (decode):
  - before and after ms, the change, and its band: the range of that exact row over the n before runs;
    relative widths follow the observations, so small rows can have wider percentages than large rows.
  - whether the after value lies outside that range, with n and the false-move rate 2/(n+1);
  - work changes: calls, operations, bytes (prefill), the limit (decode);
  - rows only before, or only after.
- The gaps as a row of their own, with the same band.
- The tie-out on both sides:
  - the kernel changes plus the gap change equal the captured wall change (within 0.001 ms);
  - each side's captured wall time against its untraced median, using the larger of their two measured repeat spreads.
    No matching repeat evidence means "not checked", for both prefill and decode.
- The untraced change.

A part reads "unchanged outside moved rows" only when no row moved, no work changed, no row came or went, the gaps
held, and both sides confirm end to end. An upgrade is expected to move rows; the audit says which.

## Chip figures and stated choices

The audit's noise is measured, never a constant:
- A row's band is the range of its n before runs.
- A chip figure (read bandwidth, a matrix peak) carries `uncertainty` in `data/targets.json`. It is the spread of
  the probe's repeated whole runs at its best launch shape. A run's own read is one probe run, so it uses the
  registry's figure for its chip. A figure without one is "not checked".
- `shape_spread` beside it records how much the launch shape matters. It is data only; no guard reads it.

The stated choices are in one table, `boltbeam/data/policy.json`, each with its reason, read through
`core.policy.value`. Examples: a 1% change counts as equal speed; a protected context may not slow by more than 1%.
A missing entry is an error, not a default.

## Worked example

The earlier clean loop test used the former fixed bands ([loop-test-27b-prefill-rtx5090-20261012.md](loop-test-27b-prefill-rtx5090-20261012.md)):
b9592 against b9592 plus one GDN commit. Only the recurrent row moved; the 2048 gaps moved 0.417 ms against a
0.155 ms band; everything else held, and both sides confirm end to end.

## Exact setup and capture settings

Bands match the target and device, engine build, model path and profile, workload, requested contexts and settings,
GPU layout, capture window and method, and observed toolchain. They are never borrowed from another setup. A pair
whose decode capture used a different context than requested cannot calibrate the requested setup.

Capture commands use the persisted request and the settings observed during its untraced measurement, including
decode token count, engine context allocation and chunk size. `--chunk-size` records an explicit request; unsupported
adapters refuse it. Missing allocation or token evidence requires a fresh measurement rather than a guessed capture.
The torch build string and its runtime CUDA are recorded separately from the CUDA compiler version.
