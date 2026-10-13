# BoltBeam

<!-- mirror-note: the public mirror inserts its note on the line below this one. Keep this marker. -->

> This is the public mirror of BoltBeam, the tool behind the write-ups on
> [research.arkey.ai](https://research.arkey.ai). It follows the project's `main` branch, which is the
> runnable product and the records of what was measured; tests, experiments and raw campaign evidence
> live on the inner branches and stay in the private repository (see `docs/branch-flow.md`).
> Published so the numbers in those articles can be checked. MIT licensed.


**Where does your model's inference time go, and how far is each part from what the chip can do?**

BoltBeam audits inference time against a measured roofline. It runs your model in a real engine, captures every
kernel, pairs each kernel with the model role it serves, and puts each one beside the fastest that role could run
on this chip. The chip's limits are measured on the chip, not read off a data sheet.

It answers, per role and per kernel:

- **How fast could it go?** The ceiling for each row: its bytes over the measured bandwidth, or its operations over
  the measured peak of the matrix path its instructions use, whichever binds. A second ceiling, "chip best", is the
  same work on the fastest path this chip has for those weights.
- **Where does the time go?** Every millisecond of a decode token, a prefill, a turn at session depth, or a replayed
  chat session, in ranked rows: ms now, ms at the limit, ms lost, and tokens per second if that row alone were fixed.
- **Does it add up?** The rows plus the idle gaps equal the captured time, and the captured time is checked against
  the untraced time inside a measured repeat band.
- **Does a second engine agree?** The same model in another engine must show the same work and the same ceilings.
- **Did an engine upgrade change anything?** The engine-upgrade audit compares two builds row by row and names what
  moved beyond run-to-run noise ([docs/engine-upgrade-audit.md](docs/engine-upgrade-audit.md)).

Engines: llama.cpp, vLLM, Ollama, TensorRT-LLM and a tinygrad fork. Chips: Apple Metal and NVIDIA CUDA, measured
the same way. BoltBeam says where the time goes. Why a kernel is slow is the engine developer's question.

## Install

On a Mac with Homebrew:

```bash
brew install JulianAbeleda/tap/boltbeam
```

`boltbeam` opens the screen; `brew upgrade boltbeam` takes a new version. Runs are kept in `~/boltbeam-runs`.

From source, with Python 3.10 or newer. No dependencies.

```bash
git clone https://github.com/JulianAbeleda/BoltBeam-Public.git
cd BoltBeam-Public
python3 -m pip install .
```

A stock macOS names Python 3.9 as `python3`. BoltBeam refuses it and says so. Name a newer one, for example
`python3.12 -m pip install .`. Without installing, `python3 -m boltbeam.cli <command>` does the same as
`boltbeam <command>`. `boltbeam selfcheck` runs the core path on a tiny synthetic model, with no GPU and no model
file, and prints PASS or FAIL per step.

On Linux with an NVIDIA GPU, [docs/linux.md](docs/linux.md) goes from a bare Ubuntu to the first report. On a
rented H100, `tools/pod-setup.sh` does it in one command ([docs/h100-pod.md](docs/h100-pod.md)).

## Your first run, in five minutes

You need one model file the engine reads (a GGUF for llama.cpp) and the engine. These steps ran on a MacBook Air
(M3) with Homebrew's llama.cpp and a 270M-parameter Q8_0 GGUF.

**1. Check the machine.**

```bash
boltbeam doctor
```

```text
python         found      3.12.13 (/opt/homebrew/opt/python@3.12/bin/python3.12)
GPU            found      Apple M3
chip profile   found      apple_m3_10c (built in)
llama.cpp      found      /opt/homebrew/bin/llama-bench (path); per-role time: boltbeam-kernel-timer-isolated
vllm           n/a        BoltBeam runs vLLM on NVIDIA GPUs only; apple_m3_10c is Metal
capture tool   not found  metal-system-trace is not installed
                          -> install Xcode (xctrace comes with it)
Go             found      go1.26.5 at /opt/homebrew/bin/go
...
```

One row per thing BoltBeam can use, and one fix per gap. It always exits 0. What matters for a first run:

- **chip profile.** If it says `not found`, run the command it prints (`python -m boltbeam.workflow.screen
  autoscan`). That measures this GPU's bandwidth and compute peak once, in about a minute. A run does not do it for
  you, and without it every ceiling is refused.
- **an engine** that says `found`.
- **capture tool.** Without it, per-role time on a Mac uses BoltBeam's kernel timer: each of the engine's own
  kernels timed alone ("isolated"). With Xcode it is a capture of the real token ("in-model").

**2. Run one audit.** Either on the screen:

```bash
boltbeam tui
```

Setup opens with this machine's chip and the first engine found. Pick the model (Setup lists the files in
`~/models`, or type a path), leave the workload on `decode`, and press **Run**. On the M3 the run took about 30 seconds. Or from the command line, with the chip id doctor printed:

```bash
python -m boltbeam.workflow.screen pipeline model.gguf --run runs/first --target apple_m3_10c \
  --provider llama.cpp --workload decode --analyze --no-search
```

This one took 11 seconds. It writes `runs/first/summary.md` and `runs/first/report.html`.

**3. Read the result.** The top of the page, from the M3 run:

```text
⚡ 215.4 tokens per second measured. The limit is 342.2.
█████████████░░░░░░░ 63% of the limit
⚠ Engine build: llama.cpp build 48d22e295 is NOT the pin b11571 (097cf33d9): this run measured another build.
Measured with llama.cpp · isolated, timed by BoltBeam's kernel timer
Where the token goes (estimate)
ROW               MS NOW  MS AT LIMIT  MS AT CHIP BEST  MS LOST  SHARE  TOK/S IF FIXED  % OF PEAK
ffn_gate_up Q8_0  1.44    0.51         0.51             0.92     54%    268.8           23.3%
attn_qo Q8_0      0.38    0.13         0.13             0.25     15%    227.7           22.0%
attn_kv Q8_0      0.25    0.06         0.06             0.19     11%    224.5           16.6%
...
= token           4.64    2.95         2.95             1.70     100%
Per-role repeat check: not checked: no repeat band matches this setup.
```

- The headline is the untraced decode against the roofline: 63% of what this chip allows for this model.
- Each row is one role. **MS LOST** is how far it is from its limit. **TOK/S IF FIXED** is what fixing that row
  alone is worth. Here the feed-forward input projection loses the most: 0.92 ms of 1.70.
- "(estimate)" means the kernels were timed alone, not inside the token. A capture tool removes the estimate.
- The engine build is checked against BoltBeam's pinned version. A different build is measured and said, not refused.

**4. Run it twice.** A first run always says **not checked**. That is expected, not a fault. A verdict ("moved",
"confirmed end to end") needs to know how much the same setup moves between two runs. Run the identical setup
again, derive the repeat bands, and give them to later runs:

```bash
python -m boltbeam.workflow.screen pipeline model.gguf --run runs/second --target apple_m3_10c \
  --provider llama.cpp --workload decode --analyze --no-search
python -m boltbeam.workflow.screen repeat-bands --before runs/first --after runs/second --out bands.json
python -m boltbeam.workflow.screen pipeline model.gguf --run runs/third --target apple_m3_10c \
  --provider llama.cpp --workload decode --analyze --no-search --bands bands.json
```

The third run reads `Per-role repeat check: 5 of 5 roles checked against the measured repeat band of this setup.`
Bands belong to one exact setup: the same chip, engine build, model, workload, contexts and settings. They are never
borrowed from another.

## Next steps

| you want to | read |
|---|---|
| prefill, session depth, or a replayed chat session | the Workload line in Setup; `--workload prefill`, `session`, `replay` ([tui/README.md](tui/README.md)) |
| compare two engines, or two builds of one engine | [docs/engine-upgrade-audit.md](docs/engine-upgrade-audit.md); on the screen, Saved runs, then Compare |
| time your own kernel against the roofline | [docs/guide-time-your-kernel.md](docs/guide-time-your-kernel.md) |
| add an engine | [docs/guide-add-an-engine.md](docs/guide-add-an-engine.md) |
| use the screen and its JSON mode for agents | [tui/README.md](tui/README.md) |
| the command line end to end | [docs/CLI.md](docs/CLI.md) |
| how the timer and its limits were measured | [docs/kernel-timer.md](docs/kernel-timer.md) |
| everything else, and the dated audit records | [docs/README.md](docs/README.md) |
| work on BoltBeam itself | [docs/maintaining.md](docs/maintaining.md) |

## Check the article numbers

The write-ups on [research.arkey.ai](https://research.arkey.ai) cite files in this repository by path.
Each cited number lives in one of these places:

| what the article says | where it comes from |
|---|---|
| a chip's memory speed, matrix peak, SM count | `boltbeam/data/targets.json`, with the measurement in `bench/<target>-facts-<date>/README.md` |
| how a weight format packs, and its refuted routes | `boltbeam/data/quants.json` |
| a route was promoted or refuted, and why | `boltbeam/data/candidates.json` (the candidate registry) |
| a closed experiment and its numbers | the dated record in `docs/`, for example `docs/14b-aggregate-fusion-closeout-20260702.md` |
| the speed limit for a model on a chip | rerun it: `boltbeam roofline-theoretical MODEL.gguf --target TARGET --context 1` |

`bench/` holds the raw measurement records. [docs/README.md](docs/README.md) lists the records the
articles cite.
