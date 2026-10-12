# The engine-upgrade audit: run it before any engine upgrade

An engine upgrade changes what BoltBeam measures. Before an engine's pin moves, audit the old build and the new
build of the same engine on the same model, chip and workload, and compare them row by row.

## The pin

Every engine BoltBeam measures has one pinned version, in one table:
`boltbeam/data/engine_pins.json`.

- llama.cpp: a release build tag (`bNNNNN`) and its full commit sha. Never moving master.
- vLLM: its release version.
- The tinygrad fork has no pin: no BoltBeam code references a fork revision.

Every run records the build it measured. llama.cpp reports `build_commit`; vLLM reports its version. The run's
manifest carries it under `engine`, with the pin check (`providers.record_build`). When the build differs from the
pin, the manifest and the audit say so plainly.

Changing a pin is its own `[target]` or `[repo]` commit. The commit cites the upgrade audit that justified it.

## How to run it

1. Run the old build's audit, unless a saved run of that exact build exists:
   `python -m boltbeam.workflow.screen pipeline --run OLD --provider llama.cpp --workload prefill --ctxs 512,2048 --analyze --measure auto --no-search MODEL`
   with `BOLTBEAM_LLAMA_BENCH` pointing at the old build. Then the same with `--workload decode --ctxs 512`.
2. Run the same for the new build, into NEW.
3. Compare: `python -m boltbeam.workflow.screen diff --before OLD --after NEW`.
   (`replay-diff` is the same command. Two replays compare turn by turn; two prefill or decode audits compare
   row by row.)

One run per build and workload. No retries.

## What it refuses

It refuses, naming the field, when the two runs differ in workload, model file, target or engine. It refuses when
either run recorded no engine build: an unknown build is never compared. These are replay-diff's refusal rules,
from one place (`workflow/diff.refuse_unless_same`).

## What it reports

- Its header names the pin and each run's build against it.
- Each row, per prompt length (prefill) or per context (decode):
  - before and after ms, the change, and its band: max(3% of the before time, 0.05 ms). The floor keeps tiny
    rows from flagging on noise.
  - whether it moved beyond the band;
  - work changes: calls, operations, bytes (prefill), the limit (decode);
  - rows only before, or only after.
- The gaps as a row of their own, with the same band.
- The tie-out on both sides:
  - the kernel changes plus the gap change equal the captured wall change (within 0.001 ms);
  - each side's captured wall time against its untraced median, within the 6% end-to-end band
    (`tie_out.END_TO_END_BAND`). That band was derived from prefill captures. Decode uses the same band.
- The untraced change.

A part reads "unchanged outside moved rows" only when no row moved, no work changed, no row came or went, the gaps
held, and both sides confirm end to end. An upgrade is expected to move rows; the audit says which.

## Worked example

The clean loop test ([loop-test-27b-prefill-rtx5090-20261012.md](loop-test-27b-prefill-rtx5090-20261012.md)):
b9592 against b9592 plus one GDN commit. Only the recurrent row moved; the 2048 gaps moved 0.417 ms against a
0.155 ms band; everything else held, and both sides confirm end to end.
