# MR9 semantic search

MR9 executes the complete population already selected by MR8. The MR8 handoff
retains each canonical Tinygrad campaign request, so the operator does not
reconstruct a request or candidate-population JSON file before the run.

From the BoltBeam checkout, the current EXP control command is:

```sh
cd /Users/julianabeleda/env/BoltBeam
.venv/bin/python -m boltbeam.cli mr9-semantic-search mr8-selection.json \
  --provider-command '["/Users/julianabeleda/env/tinygrad-arkey-exp/.venv/bin/python","/Users/julianabeleda/env/tinygrad-arkey-exp/extra/llm_research/search_provider.py","--backend","METAL"]' \
  --provider-revision "$(git -C /Users/julianabeleda/env/tinygrad-arkey-exp rev-parse HEAD)" \
  --boltbeam-revision "$(git rev-parse HEAD)" \
  --finalist-repeats 2 \
  --minimum-win 0.03 \
  --maximum-relative-mad 0.05 \
  --out mr9-semantic-search.json
```

`mr8-selection.json` must be the unedited output of
`boltbeam mr8-population-selection`; the command refuses a partial population,
a changed retained request, or a population that no longer reproduces from
that request. The MR7 ranking retains the canonical authority packet containing
its exact prerequisite artifacts, derived plan, and raw provider results. MR8
retains that complete ranking and its selection request. MR9 first reconstructs
MR7 from that authority, then re-derives the entire MR8 selection before
launching the provider; a recomputed self-hash on hand-built JSON is not
sufficient. The output path
must not already exist.

One persistent Tinygrad provider process serves every selected population in
the complete MR8 handoff. Both repositories must match the explicitly pinned
revisions and be clean; Tinygrad enforces its pin on every executable stage
and BoltBeam rejects a mismatched or dirty search revision. The controller
also requires the canonical checkout-local provider path and records SHA-256
identities for its script and selected Python executable. Every
candidate goes through target description, admission, compilation, the exact
GGUF tensor plus deterministic-activation oracle, and measurement with at
least five raw samples. The provider cache means each candidate is compiled
once. BoltBeam retains every measured, rejected, and blocked terminal row.

After the finite population completes, the same provider session remeasures
the top two measured rows and the Tinygrad heuristic control twice. Excessive
relative median absolute deviation produces an inconclusive instability
verdict. Otherwise the artifact records either `MACHINE_WINNER` when the best
machine candidate clears the three-percent control threshold, or
`REFUTED_NO_MATERIAL_WIN`. No Metal work is performed while producing or
validating the MR8/MR9 artifacts; Metal runs only when the command above is
explicitly invoked on the target machine.

Each correctness packet binds the exact model SHA-256 and GGUF tensor name.
Every finalist repeat must return the same source, plan, and binary hashes as
its single compile stage; artifact drift blocks the verdict.
