# Apple M4 Metal Qwen3-8B evidence

This directory is the replayable evidence bundle for the EXP-only BoltBeam/Metal compatibility lane. It does not install a runtime route or promote anything to dev/master.

## Outcome

The finite exact-shape search completed: 13 candidates were enumerated, 12 compiled and passed the packed-reference correctness check, one heuristic candidate was rejected as unsupported, and none were blocked. The official selected isolated-role plan was `LOCAL axis=0 arg=128` at a 885,749 ns median for the representative `ffn_gate_up` shape.

The matched full-model diagnostic refuted promotion. Generic EXP decoded at 11.2767 tok/s; applying the selected plan decoded at 10.6452 tok/s, a 5.60% regression. The runtime remains unchanged.

## Replay

For an exact replay, run BoltBeam at the `requested_boltbeam_revision` in the request and tinygrad at its
`requested_provider_revision`. Keep this evidence directory available through an absolute `BUNDLE` path (for
example from the current checkout while the pinned code runs in a worktree):

```sh
export BUNDLE=/path/to/current/BoltBeam/bench/metal-qwen3-8b-20260729
export TINYGRAD_ROOT=/path/to/tinygrad-arkey-exp

.venv/bin/python -m boltbeam.cli search-full-kernel \
  "$BUNDLE/ffn-gate-up-search-request.json" \
  --worker-command '[".venv/bin/python","-m","extra.llm_research.search_provider"]' \
  --tinygrad-root "$TINYGRAD_ROOT" \
  --out "$BUNDLE/ffn-gate-up-search-result.replay.json"

.venv/bin/python -m boltbeam.cli roofline-plan \
  "$BUNDLE/decode-roofline-input.json" \
  --out "$BUNDLE/decode-roofline.replay.json"
```

The request pins clean BoltBeam and provider revisions. Replaying at another revision requires deliberately updating those pins.
For the roofline replay, update the `gguf` field in `decode-roofline-input.json` to the local copy with the recorded
SHA-256. Checkout location is execution configuration and is not part of candidate identity.

## Roofline boundary

The GGUF inventory is exact for packed active weight bytes and counted dense-matrix FLOPs. The 120 GB/s raw bandwidth is the Apple M4 vendor specification; no independently sourced compute peak is claimed. The 86.893 GB/s stream result counts aggregate read plus write traffic, so it is retained as a proxy and is never used as the model's practical bandwidth denominator.

See `whole-model-ab-refutation.json` for raw samples, evidence boundaries, and reopen conditions.
