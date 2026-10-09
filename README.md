# BoltBeam

<!-- mirror-note: the public mirror inserts its note on the line below this one. Keep this marker. -->

> This is the public mirror of BoltBeam, the tool behind the write-ups on
> [research.arkey.ai](https://research.arkey.ai). It follows the project's `main` branch, which is the
> runnable product and the records of what was measured; tests, experiments and raw campaign evidence
> live on the inner branches and stay in the private repository (see `docs/branch-flow.md`).
> Published so the numbers in those articles can be checked. MIT licensed.


**Turn model weights into a codegen search problem.**

BoltBeam reads a model file. It works out which operations decide the speed. It computes the fastest the
model could ever run on a given chip. Then it emits a bounded search space and a route policy for a compiler
to explore. It measures. It does not guess.

It works the same way on AMD, NVIDIA and Apple Metal. It reads rocprof, Nsight Compute, Metal and llama.cpp
timings into one schema, so their numbers can be compared.

```text
model weights
  -> tensor census            what tensors exist, what shape, what quant
  -> role profile             which are attention, FFN, SSM, embedding
  -> speed limit              the roofline: the fastest each role could run on this chip
  -> candidate search space   the bounded set worth measuring
  -> measured route policy    what actually won, and the evidence
```

The facts drive the search. Nobody writes "if 14B, try this."

More reading: [docs/CLI.md](docs/CLI.md) walks the command line end to end.
[docs/README.md](docs/README.md) is the index of every doc.
[docs/maintaining.md](docs/maintaining.md) is for people who work on BoltBeam itself.

## Install

Python 3.10 or newer. No dependencies.

```bash
git clone https://github.com/JulianAbeleda/BoltBeam-Public.git
cd BoltBeam-Public
python3 -m pip install .
boltbeam selfcheck
```

`boltbeam selfcheck` builds a tiny synthetic model and runs the core path on it. It needs no GPU, no network
and no model file. It prints PASS or FAIL for each step and exits 1 on any failure:

```text
PASS  inspect
PASS  roofline-theoretical
PASS  load
PASS  autoscan
PASS  analyze
PASS  output
PASS  screen ceiling
PASS  screen results
checked: schema id and required keys
selfcheck: PASS
```

Without installing, `python3 -m boltbeam.cli <command>` does the same as `boltbeam <command>`.
A stock macOS names Python 3.9 as `python3`. BoltBeam refuses it and says so. Name a newer one, for example
`python3.12`.

## The screen (TUI)

The screen is a Go program in [tui/](tui/README.md). It reads BoltBeam through the same commands as below.

```bash
cd tui && go build -o boltbeam-tui .
./boltbeam-tui
```

It has three screens:

1. **Setup.** Pick a model, a chip and an engine (llama.cpp or tinygrad). The chip starts as this machine.
   Setup shows the speed limit for that model on that chip. Press **Run**.
2. **Run.** Run finds the speed limit, checks the GPU is free, measures with the engine, and times each
   role. When it ends, it shows the measured tokens per second against the limit, and where the time went.
   **Save run** keeps it.
3. **Saved runs.** Every run you saved. Open one to see its results again.

On a chip that is not this machine, Run gives the speed limit only. Measuring needs the chip itself.
[tui/README.md](tui/README.md) has the keys and the JSON mode for agents.

## The core commands

`boltbeam --help` lists the core group first. Every other command is listed after it, under "Advanced and
campaign commands". Those still run. Scripts and the records call them by name.

```bash
boltbeam inspect model.gguf --target nvidia_sm120                       # 1. what is in the model
boltbeam roofline-theoretical model.gguf --target nvidia_sm120 --context 1   # 2. the speed limit
boltbeam load model.gguf --run runs/model --target nvidia_sm120         # 3. stage a run folder
boltbeam analyze --run runs/model                                       # 4. search space and measurement plan
boltbeam output --run runs/model                                        # 5. policy, provider plan, report.html
```

1. `inspect` reads GGUF, Safetensors, AWQ, GPTQ, ONNX and MLX. It lists the layers, each role's shape and
   quant, and the architecture class. It needs no GPU.
2. `roofline-theoretical` gives the fastest each role could run, from the bytes it must read and the chip's
   measured memory speed and matrix peak. It needs no GPU.
3. `load` writes the model facts into a run folder. `boltbeam autoscan --run runs/model` then records this
   machine's facts in the same folder.
4. `analyze` emits the search space and says exactly which measurements to take next.
5. `output` packages the run: the route policy, the provider plan and a readable report.

Measurements come back with `ingest-timing`, `ingest-probe`, `import-hw-trace` or `metal-measure`.
`evaluate` and `ledger` record what won and what lost. A refuted route drops out of the search, so nobody
pays for it twice.

Targets: `boltbeam/data/targets.json` lists the chips BoltBeam knows, with the source of every number.

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

## Outputs

Every stage writes a versioned JSON contract into the run folder. The JSON Schemas are in `schemas/`.

| file | what it holds |
|---|---|
| `model_profile.json` | tensor census, roles, shapes, quant types |
| `search_space.json` | legal route families and candidate knobs per role |
| `route_policy.json` | the seed or selected route policy |
| `measurement_plan.json` | what to measure next, and why |
| `hw_trace.json` | provider-neutral kernel trace (rocprof, Nsight Compute, llama.cpp) |
| `timing_profile.json` | classified timing buckets and role attribution |
| `kernel_analysis.json` | eligibility, contrasts, hypotheses, next experiment |
| `provider_plan.json` | the handoff to a provider, written by `output` |
| `report.html` | the human-readable run report |
