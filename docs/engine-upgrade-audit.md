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

## Current build versus pin

The pin is the audited reference and moves only as above. Each machine also has a **current** build per engine:
what its runs use by default. It lives in `~/.boltbeam/builds.json` (beside the engine scan; never in the repo),
read by `collectors/engine_builds.py`. Upstream release sources are one table, `boltbeam/data/engine_releases.json`.

- `python -m boltbeam.workflow.screen engines --update [--engine E] [--into DIR]` looks at each engine's newest
  release, at most once per `engine_latest_max_age_h` (24 h, `data/policy.json`; three-second timeout; a failed look
  is "could not check", never "up to date", and is retried next time). Releases only: llama.cpp's `v0.x` line (its
  daily `bNNNNN` builds are GitHub prereleases, nightlies), vLLM's PyPI releases and TensorRT-LLM's on NVIDIA's
  index, never rc, dev, alpha, beta or local builds. A newer one is installed beside the current build, never over
  it: llama.cpp from its tag into its own folder (full commit history so the run records its build number; cmake
  flags and architecture from the vendor row), vLLM and TensorRT-LLM each into a venv named by the version, Ollama
  from its release archive. It is recorded as **pending**. The first time, the build already on the machine is
  recorded as current; it is the user's. This command is the only installer; the pipeline and collectors never
  install.
- TensorRT-LLM is checked against the machine before anything downloads, from the release's own metadata: its
  `requires_python`, a wheel for this Python and CPU on NVIDIA's index, its `torch` constraint (installed as
  declared) and its `cuda-python` major, which sets the least driver. A release that does not fit, or whose
  requirement cannot be read, is not installed: the output and `builds.json` say "not compatible here: <reason>",
  and current stays.
- Each engine keeps its current and previous build. After a comparison that succeeded, builds older than the
  previous one are deleted, only when BoltBeam installed them (its folder under the builds root, holding its mark);
  each deletion is recorded under `deleted`. Nothing is deleted after a refused or failed comparison, and a build
  found on the machine is never deleted.
- The next `pipeline --analyze` of that engine runs the previous current build once into `RUN-was-<version>`,
  then the new build into `RUN`, compares them with this audit, saves `RUN/engine_upgrade.json`, and makes the new
  build current. Without bands the comparison says "not checked"; it does not justify a pin. A failed new-build
  run leaves current unchanged. An engine env var the user set (`BOLTBEAM_LLAMA_BENCH`, ...) wins; no comparison
  is made with it.
- **Upgrade engines** happens only when asked. In the TUI, the Engine picker offers "Upgrade <engine> to <release>"
  for an engine whose newer release is known; it runs `engines --update --engine E` as a job and shows what it did
  (installed as pending, not compatible and why, or not installed and why). `pipeline --upgrade-engines`
  (`--update-engines` is the old name) runs the update first. A Run never installs by itself.
- `engines --latest` says where each engine's build stands against upstream's newest release
  (`engine_builds.standing`): behind, waiting for its comparison, up to date, or unknown (never checked, the check
  failed, or an "up to date" older than `engine_latest_max_age_h`). It installs nothing; it looks at upstream only
  when the last look is stale. The TUI runs it after the engine scan at boot and after an upgrade, and Setup says
  under the Engine line when the picked engine is behind ("0.6.0 · 0.9.0 is out: Upgrade in Engine"), waiting, or
  unknown; an engine on the latest release adds nothing.
- Every run records its build against the last check: "build X (latest at run time)" or "build X; latest was Y",
  or "latest not checked" when the check failed or is older than `engine_latest_max_age_h` (`data/policy.json`).
  The summary, the report and the TUI's Run page show it, with the comparison's line when the run made one.
- A llama.cpp run records a commit and build number, not a release: it is named by the release whose commit it
  matches (a build recorded here, or upstream's latest release). A build that is no release (a `0.6.0-dev` source
  build) is not the latest release.

## How to run it

1. Run the old build's audit, unless a saved run of that exact build exists:
   `python -m boltbeam.workflow.screen pipeline --run OLD --provider llama.cpp --workload prefill --ctxs 512,2048 --analyze --measure auto --no-search MODEL`
   with `BOLTBEAM_LLAMA_BENCH` pointing at the old build. Then the same with `--workload decode --ctxs 512`.
2. Run the same for the new build, into NEW.
3. Compare: `python -m boltbeam.workflow.screen diff --before OLD --after NEW`.
   (`replay-diff` is the same command. Two replays compare turn by turn; two prefill or decode audits compare
   row by row.)
4. Review the kernel classification before the pin moves. `kind_of` names a captured kernel by words in its name
   (`boltbeam/data/kernel_kinds.json`, first match wins); a kernel the new build renames drops the pairing walk to the
   flat pairing with no error. Record a capture on the new version (step 2's) and add its pairing window under
   `tests/fixtures/pairing` (`tests/test_pairing_replay.py` says how). Move the pin, then record the classification:
   `BOLTBEAM_RECORD_KERNEL_KINDS=1 python -m pytest tests/test_kernel_kinds.py` writes
   `tests/fixtures/kernel_kinds/<engine>@<new pin>.json`. Diff it against the old pin's file and review every name that
   changed kind or is new: an attention or recurrent kernel now `other` is a rename, and its word goes into
   `kernel_kinds.json` in the commit that moves the pin; the old pin's file goes in the same commit.
   `tests/test_kernel_kinds.py` fails while the pinned version has no recorded file, and when a capture of a model with
   attention or recurrent layers classifies none of its kernels as that block.

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

The earlier clean loop test used the former fixed bands (`loop-test-27b-prefill-rtx5090-20261012.md`):
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

## Latest llama.cpp reminder in reports

`boltbeam output` compares the commit recorded by the run with upstream llama.cpp
`master`, using GitHub's commits API. A different commit adds a reminder at the end
of `report.html` and `summary.md`: keep the pinned result, then rerun on the latest
version before drawing conclusions about bugs or performance. A matching commit
is stated without the rerun nudge. A missing commit or failed lookup is reported as
unverified, never as an outdated build. This checks upstream source, not the newest
release tag, and does not install a binary or change the pin.

The check has a three-second network timeout. Its source, UTC check time, run commit,
upstream commit and message are saved in `output_manifest.json` as `engine_update`.
Regenerating the report reuses that snapshot for the same run commit on the same UTC
day; the next day's generation refreshes it. Rendering saved facts makes no network
request. The comparison is commit equality, not an ancestry test for custom forks.
