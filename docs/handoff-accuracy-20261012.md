# Handoff: engine versions, the accuracy battery, and the missing engine runs (2026-10-12)

Three parts, in order. Each part is small, stands alone, and lands its own work. Do not start a part until the one
before it has landed. If a session crashes, read the part's progress file and resume from the last line.

The goal of parts 2 and 3 is to find where BoltBeam is wrong, not to confirm it. Report failures plainly.

## Rules (read first)

- Read docs/coding-principles.md, docs/branch-flow.md, docs/commit-discipline.md and docs/maintaining.md, and the
  Rules section of docs/handoff-cleanup-20261011.md.
- Branches:
  - main is the product, dev = main + tests, exp = dev + experiments.
  - Code goes on main and tests on dev.
  - At the end of each part, only main, dev and exp exist, each equal to origin, with main ⊂ dev ⊂ exp.
    Remove your worktrees.
- Never use --no-verify. The sz.py cap is 52,000; do not raise it.
- Everything is generic: no model names, model shapes or chip names in code.
- Docs name the model and server generically: "the 27B Q4_K_M GGUF", "the 27B AWQ checkpoint", "the box's own
  model server". The mirror's leak check refuses the real model file name, the server's name and home-directory
  paths.
- Landing:
  1. Check main == origin/main and dev == origin/dev.
  2. Run one `&&` chain, with no pipes:
     `git merge --ff-only X && git update-ref refs/heads/dev X-dev <old dev> && tools/sync_branches.sh && git push origin main dev exp`
  3. tools/sync_branches.sh and tools/publish_mirror.sh need a venv with `pip install -e '.[dev]'` first on PATH.
     Make it in a scratch directory and delete it afterwards.
  4. Publish the mirror from a worktree of origin/dev with tools/publish_mirror.sh.

## Part 1: track every engine version (code only, no GPU)

Today `boltbeam/data/engine_pins.json` pins only llama.cpp b11571 and vLLM 0.31.0. Change it so that:

1. Every engine BoltBeam has an adapter for has an entry. These are the versions installed on the GPU box:

   | engine | version |
   |---|---|
   | llama.cpp | b11571, 097cf33d9499b1eade67d51e16d7c75427e7b267 |
   | vLLM | 0.31.0 |
   | TensorRT-LLM | 1.2.1 |
   | tinygrad fork | caa752ac4 (checked out detached) |
   | Ollama | unknown: `ollama --version` printed nothing. Record "unknown" and the reason. |

2. Each engine keeps a history, not only its current pin: every version BoltBeam has measured, with the date,
   the version or commit, and the doc that used it. The known history:
   - llama.cpp b9592 (ac4cddeb0dbd778f650bf568f6f08344a06abe3a), 2026-10-11, docs/prefill-audit-27b-rtx5090-20261011.md
   - llama.cpp b11571, 2026-10-12, docs/engine-upgrade-llama-b11571-rtx5090-20261012.md
   - vLLM 0.31.0, 2026-10-11, docs/prefill-vllm-crosscheck-27b-rtx5090-20261011.md
3. The toolchain a measurement depends on is recorded too. On the box: NVIDIA driver 595.99.02, CUDA 13.2,
   torch 2.13.0+cu130 (the vLLM venv), Nsight Systems 2026.1.3, Nsight Compute 2026.2.1.
   - Every run's manifest records the toolchain it ran with, read at run time.
   - The pin file holds the current toolchain.
4. The engine-upgrade audit (boltbeam/workflow/diff.py) reports a changed driver, CUDA or torch, the same way it
   reports a changed engine build.

There is one authority, the pin file. Add tests on dev. Land it and publish the mirror.
Report the SHAs and what the pin file now holds.

## Part 2: analyse the finished runs (no GPU)

An earlier agent ran most of an accuracy battery, then was stopped before writing it up.

- The evidence is on the Mac at `~/env/boltbeam-runs/.work/acc-20261012/`, and on the box at
  `~/storage/bbeng/tmp/acc-20261012/`.
- The pass criteria were written BEFORE the runs, in `predictions.txt`. Judge against them, not against what the
  results look like.
- The runs:

  | run folder | what |
  |---|---|
  | repA-prefill, repB-prefill, repA-decode, repB-decode | the 27B Q4_K_M on llama.cpp b11571, identical pipelines back to back: prefill 512 and 2048, decode 512 |
  | ub2048 | the same prefill at -ub 2048 (one chunk) |
  | oracle-generic, oracle-cmp.json | the generic timer under Nsight Compute, 27B decode |
  | l8-decode, l8-prefill | the 8B Q4_K_M on llama.cpp b11571 |
  | replay-a60 | the agent-60 replay on llama.cpp b11571 |
  | vl-prefill, vl-decode | the 27B AWQ on vLLM 0.31.0 |
  | vlB-prefill | the vLLM prefill repeat |
  | fix3-prefill | a re-check run taken while fixing |

- Four fixes landed after some of these runs: 6d288bb, 08b51f5, d68c7e8 and 86aa20d. When a fix affects a saved
  run, re-attribute that run under current main instead of re-running it, and say which ones you re-attributed.
- Progress file: `~/env/boltbeam-runs/.work/acc-20261012/analysis-progress.md`. Append one line after each test.

For each test, give the criterion from predictions.txt, the result with its key numbers, and pass or fail:

1. **Repeatability.** The per-row spread between repA and repB. Do the bands max(3%, 0.05 ms) per row and 6% end to
   end hold between two identical runs? If not, propose a band derived from the data.
2. **The gaps at 2048.** Both b11571 captures at 2048 showed 30 to 40 ms of idle gaps that the untraced run does
   not have. Locate them in the launch sequence: chunk boundaries, the start, or around a specific kernel. Use
   repA, repB and ub2048. "Where", not "why".
3. **vLLM against llama.cpp.** The work per row set must match. FFN differs only by the pruned last layer, which
   b11571 does. Any other difference is a bug.
4. **Timer against Nsight Compute.** The error per role, against the "after" column in docs/kernel-timer.md.
5. **The 8B.** Every role pairs, the tie-out holds, and the work equals the model profile.
6. **Replay agent-60.** Redraw the tables in docs/replay-27b-rtx5090-20261011.md with the chip-best column.

Write `docs/accuracy-battery-rtx5090-20261012.md` with one table per test and a list of the accuracy problems found.
Do not fix code in this part; list the bugs. Land the doc and publish the mirror.

## Part 3: the missing engine runs (GPU)

Run these on the Ubuntu RTX 5090 box, one engine at a time.

### Box rules

- Work under ~/storage only. No sudo.
- Do not edit ~/env/llama.cpp (its build-cuda links to the b11571 build) or ~/tinygrad-arkey.
- Leave boltbeam-tui alone. Kill only processes you started.
- Use the clone ~/storage/ubuntu-cmp/BoltBeam, pulled to origin/main.
- Write each run's prediction to a file before the run.
- Put runs in `~/storage/bbeng/tmp/acc2-20261012/`, and write `progress.md` there after every run.
- GPU runs are lean: one per engine, model and workload. If you repeat one, say why.

### The box's model server

It holds the GPU while it runs.
1. Find its unit: `systemctl --user list-units --type=service | grep -i llama`. Call it `$UNIT`.
2. Before you stop it, save its exact command line:
   `journalctl --user -u "$UNIT" -o cat | grep "Started" | tail -1`. The command follows "Started ... - ".
   It is a transient unit: `systemd-run --user --unit="$UNIT" --collect --property=Restart=on-failure <command>`.
3. Stop it with `systemctl --user stop "$UNIT"`.
4. At the end, restart it with that same systemd-run line, even if a run failed.
5. Then check that `/health` is ok on port 8080, one chat to the server's alias finishes, and `/props` reports b11571.

### One pipeline run

```sh
W=~/storage/ubuntu-cmp; B=~/storage/bbeng
export TMPDIR=$W/tmp BOLTBEAM_CUDA_CACHE=$W/tmp/boltbeam-cuda CUDA_CACHE_PATH=$W/tmp/nv
export PATH=/usr/local/cuda/bin:$PATH HF_HOME=$B/tmp/hf
export BOLTBEAM_VLLM_PYTHON=$B/vllm-venv/bin/python BOLTBEAM_TRTLLM_PYTHON=$B/trtllm-venv/bin/python
export BOLTBEAM_OLLAMA=$B/ollama/bin/ollama
cd $W/BoltBeam
$W/.venv/bin/python -m boltbeam.workflow.screen pipeline --run "$RUN" --target nvidia_sm120 --id "$ID" \
  --provider "$ENGINE" --workload "$WORKLOAD" --ctxs "$CTXS" --analyze --measure auto --no-search "$MODEL"
$W/.venv/bin/python -m boltbeam.workflow.screen run --run "$RUN" > "$RUN-summary.txt"
```

### The models on the box

- The 27B AWQ checkpoint: `~/storage/bbeng/models/` (the W4A16 AWQ folder).
- The 8B BF16 safetensors: `~/storage/bbeng/models/Qwen3-8B`.
- The 27B Q4_K_M and 8B Q4_K_M GGUFs: `~/storage/models/`.
- The 8B F16 GGUF was deleted. Do not recreate it unless an engine needs it, and say so if one does.

### Runs, in this order

1. vLLM, the 27B AWQ: decode 512. This is the repeat that was interrupted.
2. vLLM, the 8B BF16: decode 512, then prefill 512 and 2048.
3. TensorRT-LLM, the 8B: decode 512.
4. Ollama, the 8B: decode 512. Ollama may need the model imported first. If it can't run, give the reason in one line.
5. The tinygrad fork, the 8B: decode 512. Read-only; no fork development.
6. For TensorRT-LLM, Ollama and tinygrad, also try the 27B if the engine loads it. If it can't, give the reason in one line.

### For each run, report

- whether every role pairs
- whether the tie-out holds
- whether the captured time is within 6% of the untraced time
- whether the work per row matches the llama.cpp and vLLM runs of the same model
- % of roofline

Do not build new adapters. Do not change BoltBeam code; list the bugs. Add each engine's version to the pin file's
history (part 1) in one [target] commit.

At the end:
- Append a per-engine table to docs/accuracy-battery-rtx5090-20261012.md with the columns engine, version, model,
  workload, work match, roles paired, tie-out, and % of roofline.
- Land it and publish the mirror.
- Copy the run folders, without the .nsys-rep files, to `~/env/boltbeam-runs/.work/acc2-20261012/` on the Mac.

## Report back (each part)

Use short plain sentences. Report:
- each test's or run's pass or fail, with its key number
- the bugs found, and what stays open
- the SHAs for main, dev, exp and the mirror
- the pytest count
- the sz.py total
- the model server's final status (part 3)
