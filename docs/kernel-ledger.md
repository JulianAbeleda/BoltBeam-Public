# Kernel ledger

`boltbeam kernel-ledger` lists every kernel a decode step launches, grouped by name and grid. For each group it reports launches per step, exclusive time, bytes, the limit and the distance from it. It also says whether the search can propose a candidate for the group and whether the engine can install one. The full list feeds the kernel search, so the search no longer starts from hand-picked rows. Code: `boltbeam/workflow/kernel_ledger.py`. Record: `boltbeam.kernel_ledger.v1`.

```
boltbeam kernel-ledger CAP@10013 CAP@102621 --model main.json --draft draft.json --engine llama.cpp --vendor nvidia \
  --target nvidia_sm120 --k 3 --kv-type Q8_0 --selection 8192 --untraced-ms 15.995 --untraced-ms 18.875 \
  [--counters turns.json ...] [--lowering lowering.json] [--out ledger.json]
```

- **Inputs**
  - A capture that holds prefills (a window of a session's turns) is read with `--session RECORD --window W` (once per capture): only each turn's decode window is kept, placed by the driver's span on the host clock and the engine's own prompt and decode times (`time_budget.turn_windows`), the turns joined end to end. The counters are then the turns'. Without it a prefill's forwards set the per-layer maximum and every verify is read as a draft.
  - `--manifest` is the kernel manifest the engine ran with (`boltbeam.kernel_manifest.v1`). Its kernels are joined by what each entry computes (its matmul's weight format and output rows), so a generated kernel is not left unattributed.
  - `--routing` is the model's measured routing profile. Expert bytes are then the experts a verify of k + 1 columns reaches under it (derived), not the experts_used x columns upper estimate.
  - A capture is an nsys sqlite export, or an extract of one in JSON. The extract carries each launch's grid and the device copies. It may also carry the engine's counters for its turns. The fixtures are in `tests/fixtures/kernel_ledger/`.
  - `--model` and `--draft` are header tables (`{meta, tensors: [name, dims, type id]}`), the same tables `min_memory` reads.
  - Two depths give each row a slope in ms per 1k tokens.
- **Steps**
  - The output layer is the matmul launch whose output rows equal the vocabulary. A forward that runs at least half the most per-layer launches is a verify.
  - A step runs from the end of one verify's output layer to the end of the next.
  - The draft passes the cut finds are checked against the engine's counters (`decode_steps`, the decode row's `draft_forwards`).
- **Join**
  - Launch groups are tied to the header's matrices by launches per forward. This uses the engine's `shapes` rows in `data/kernel_families.json`: rows per block, format families, `min_k`, and `tensors_per_launch`. Chip facts such as `min_cols` sit under `by_target`.
  - Launches per forward are the most common count a step, so a step that a cut shortens (the one across two joined turns) does not unjoin a group.
  - When no single set of matrices fits a group, the group is marked "unattributed", by name. The join never guesses.
- **Two rankings**
  - **Byte-limited.** The bytes are known, so distance = time − bytes / measured bandwidth.
    - Dense matmuls are derived.
    - Experts are an estimate: the number of distinct experts a launch reads lies between `experts_used` and `experts_used × columns`, and the estimate uses the high end.
    - Attention is an estimate: cells read (the selection if given, otherwise the depth) × bytes per cell.
  - **Bound unknown.** No bytes are known, so the ranking value is time − launch floor. The floor is the cheapest lone launch, so this value is an upper bound on what the row can give.
  - The family roll-up and the missing-piece sums keep the two apart. A row with no bound is never sorted as 0.
- **Coverage, computed per depth**
  - `search_can_propose`: whether the region search (the only search since 2026-11-21) covers the row
    (`kernel_ledger.region_covers`, from `data/region.json` alone).
    - The engine that ran has an `engines` row, a generator row has a `generate` action, and the row's kernel is one of
      the engine's graph-op kernels (`engines.<E>.graph_ops.kernels`).
    - `formats.refusal("search")` for matmul formats.
    - `data/decode_graph.json` `kinds` for the other kernels, with `fusion_space.REFERENCE_OPS`.
  - `check`: whether a reference exists for the row's formats.
  - `install_path`: from `data/engine_patches.json` `install_paths`. For llama.cpp this is the external-kernel hook proven in ext-kernel-hook-20261115.
- **Checks**
  - Join coverage: matmul time that no row joined, and main-file matrices that were never joined.
  - No row runs below its derived limit.
  - Overlap across streams.
  - Draft passes against the counters, and against k.
  - Traced step against the untraced step, with the overhead disclosed.
  - Every figure labelled.
- **Conventions**
  - Exclusive time goes to the launch that started first, so a downstream PDL wait is charged to the launch upstream.
  - Every time is a traced time.
  - Idle covers the steps in the capture, not the whole session.
