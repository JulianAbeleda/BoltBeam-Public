# BoltBeam

> This is the public mirror of BoltBeam, the tool behind the write-ups on
> [research.arkey.ai](https://research.arkey.ai). It follows the project's `main` branch, which is the
> runnable product and the records of what was measured; tests, experiments and raw campaign evidence
> live on the inner branches and stay in the private repository (see `docs/branch-flow.md`).
> Published so the numbers in those articles can be checked. MIT licensed.

**Turn model weights into a codegen search problem.**

New here? **[docs/CLI.md](docs/CLI.md)** walks the command line end to end, from reading a model file to the ledger.

BoltBeam reads a model file, works out which operations actually matter for
speed, and emits a bounded search space plus a route policy for a downstream
compiler to explore. It measures; it does not guess.

It is provider-neutral by design: the same pipeline profiles AMD, NVIDIA and
Apple Metal targets, and normalises rocprof, Nsight Compute and llama.cpp output
into one schema so their numbers can be compared.

```text
model weights
  -> tensor census            what tensors exist, what shape, what quant
  -> role profile             which are attention, FFN, SSM, embedding
  -> legal route families     what codegen strategies each role permits
  -> candidate search space   the bounded set worth measuring
  -> measured route policy    what actually won, and the evidence
```

The point is that the facts drive the search. Nobody writes "if 14B, try this."

## Install

```bash
python -m pip install -e '.[dev]'
python3 tools/install_hooks.py     # commit-message checker
boltbeam --help
```

The trunk carries no tests — the suite lives on `dev`, which is where correctness
is answered. See [Branches](#branches).

## Apply it

### 1. Look at a model — no GPU needed

```bash
boltbeam inspect model.gguf --target nvidia_sm120
```

Reads GGUF, Safetensors, AWQ/GPTQ, ONNX and MLX. Emits layer count, per-role
shapes and quants, and the architecture class. Hybrid SSM/attention models are
detected from GGUF metadata rather than by model name, so it works on
architectures released after it was written.

Targets: `amd_gfx1100`, `nvidia_sm89`, `nvidia_sm120`, `apple_metal`,
`apple_m4_10c`.

Produces facts like:

```json
{"roles": [
  {"role": "ffn_gate_up", "shape": [17408, 5120], "quant": "Q4_K"},
  {"role": "lm_head",     "shape": [151936, 5120], "quant": "Q6_K"}
]}
```

### 2. Find the ceiling before optimising anything

```bash
boltbeam roofline-plan plan.json --out roofline.json
```

**Start here.** It derives the achievable ceiling, places the model against it,
and works down to a per-role verdict: what is memory-bound, what is
compute-bound, and how much time is actually reclaimable.

`roofline-theoretical` does the same from a GGUF with no GPU at all.

### 3. Run the measurement loop

```bash
boltbeam load model.gguf --run runs/model        # stage the facts
boltbeam autoscan --run runs/model               # record machine/provider facts
boltbeam analyze --run runs/model                # emit search + measurement bundle
boltbeam import-ncu profile.csv --run runs/model # or import-hw-trace
boltbeam output --run runs/model                 # package policy + report.html
```

Each step is a typed JSON contract, so any one can be re-run or replaced without
redoing the rest.

### 4. Diagnose a slow kernel

```bash
boltbeam investigate-kernels evidence.json --out analysis.json
boltbeam explain-kernel  analysis.json    # why is it slow
boltbeam next-experiment analysis.json    # what to measure next
```

### 5. Record what you learned

```bash
boltbeam evaluate evidence.json --out decision.json
boltbeam ledger add decision.json --ledger route_ledger.jsonl
boltbeam ledger report --ledger route_ledger.jsonl
```

The ledger is durable. A refuted candidate drops out of the search space, so
`evaluate` will no longer score it. When you have fresh numbers for a route you
gave up on, use `reopen-check` — it audits that exact candidate under the same
guardrails without silently promoting it:

```bash
boltbeam reopen-check --candidate decode_attention_g5_8b_refuted \
  --evidence evidence.json --profile model_profile.json
```

This is what stops the same dead end being explored twice.

`boltbeam --help` lists all 60 commands.

## Outputs

Every stage emits a versioned JSON contract:

| file | what it holds |
|---|---|
| `model_profile.json` | tensor census, roles, shapes, quant types |
| `search_space.json` | legal route families and candidate knobs per role |
| `route_policy.seed.json` | seed or selected route policy |
| `hw_trace.json` | provider-neutral kernel trace (rocprof / NCU / llama.cpp) |
| `timing_profile.json` | classified timing buckets and role attribution |
| `kernel_analysis.json` | eligibility, contrasts, hypotheses, next experiment |
| `provider_plan.json` | adapter handoff produced by `output` |
| `report.html` | the human-readable run report |

## Scripts

| script | what it does |
|---|---|
| `tools/install_hooks.py` | installs the commit-message checker |
| `tools/check_commit_msg.py` | enforces `[subsystem]` prefixes and `NFC` marking |
| `tools/sync_branches.sh` | propagates the trunk outward to `dev` and `exp` |
| `tools/classify_bench.py` | classifies `bench/` artifacts as verdict or measurement |
| `tools/export_route_manifest_snapshot.py` | sole exporter for the tinygrad EXP snapshot |

That is the whole of `tools/` on the trunk: repo infrastructure and one exporter.

Do not pipe `sync_branches.sh` into `head` — closing its stdout kills it before
it merges.

### Campaign gates live on `dev`

The same-run measurement gates — `promote_routes.py`, `prepare_same_run.py`,
`measure_same_run.py`, `compare_same_run.py`, `prepare_quant_comparison.py`,
`validate_fused_route_promotion.py`, `account_prefill_roles.py` and their shared
`same_run.py` — are campaign tooling, which `docs/branch-flow.md` excludes from
the trunk by category. They and their runbooks are on `dev`, with their tests:

```bash
git checkout dev
python tools/promote_routes.py --model <model-id> \
  --evidence evidence.json --roles roles/ --run-id clean-1 --out report.json
```

They take the model, target, quant set and context ladder as arguments, so a
second model is a different invocation rather than a second copy of the script.
Their verdicts are records, so those stay here: see `bench/` and
[`docs/task_workflow/output/model-agnostic-same-run-tools-20260731.md`](docs/task_workflow/output/model-agnostic-same-run-tools-20260731.md).

## Scale

Tracked Python on `main` — the product surface only:

| area | lines | files |
|---|---:|---:|
| `boltbeam/` — the product | 33,546 | 216 |
| `tools/` — repo infrastructure | 234 | 4 |
| **total** | **33,780** | **220** |

`dev` adds the suite, the campaign gates and the retained apparatus on top:
53,703 lines across 434 files, of which 17,654 are `tests/`.

```bash
git ls-files '*.py' | xargs wc -l | tail -1     # reproduce, on any branch
```

Line count is not the metric — duplicated knowledge is — so this is here to be
reproduced rather than optimised, and it drifts with every commit.

## Branches

`master` ⊂ `dev` ⊂ `exp`, split by content category rather than maturity. The
trunk holds the product; `dev` adds the verification apparatus — the test suite
and the campaign gates; `exp` holds active experimentation.

So the trunk is what ships and does not carry its own proof. Correctness is
answered on `dev`:

```bash
git checkout dev && pytest
```

`tools/sync_branches.sh` gates each hop accordingly: `dev` is hard-gated on the
suite, so a trunk commit that breaks the product cannot reach `exp`; `exp` is
report-only, because it admits broken intermediate states by design. See
[`docs/branch-flow.md`](docs/branch-flow.md), which also covers the one case
needing a human: when the trunk prunes something the outer branches retain.

## Route-manifest authority

`boltbeam/policy/assets/route_manifest.v1.json` and its SHA-256 sidecar are the
sole canonical route-selection policy. `boltbeam.policy.route_manifest` validates
the asset before exposing it. Refresh the tinygrad EXP snapshot with the exporter
rather than copying the file:

```bash
python tools/export_route_manifest_snapshot.py \
  /path/to/tinygrad-arkey-exp/extra/llm_research/generated/boltbeam_route_manifest.v1.json
```

Missing, stale-schema, noncanonical or hash-mismatched input fails closed.

## Handing a model to an agent

```bash
boltbeam analyze /path/to/model.gguf --target amd_gfx1100 \
  --id qwen3-14b --out-dir outputs/qwen3-14b-analysis
```

The bundle is audit-first: it emits the profile, search and policy files, then
states exactly which provider commands to run next. If a generated candidate
fails because a topology knob is missing, the expected outcome is
`search-space-incomplete` — expose the knob rather than hand-writing a one-off
route.

## Design principles

Measured, not asserted. Every claim in an artifact cites the evidence that
produced it, and a gate that cannot verify its input fails closed rather than
guessing.

Provider-neutral. Backend differences live inside adapters; shared policy lives
above them. A capability a backend lacks is stated explicitly, never faked.

The short version of the house rules: centralize authority, modularize
execution, abstract for simplicity, keep concerns orthogonal — and line count is
not the metric, duplicated knowledge is.

## Further reading

- [`docs/branch-flow.md`](docs/branch-flow.md) — branch layout and sync procedure
- [`docs/task_workflow/`](docs/task_workflow/) — how delegated work is specified and recorded
- [`docs/exp-metal-mr0-mr13-runbook.md`](docs/exp-metal-mr0-mr13-runbook.md) — the Apple Metal MR0-MR13 workflow
- [`INTEGRATION_TINYGRAD.md`](INTEGRATION_TINYGRAD.md) — the tinygrad integration boundary
