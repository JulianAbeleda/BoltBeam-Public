# Handoff: remaining accuracy work, 2026-10-12

Read this document and `docs/accuracy-fixes-20261012.md` before doing more work. The investigation, fixes and fresh
validation are landed. The next task is to address the remaining causes, not rerun the completed battery or
claim all accuracy problems are fixed.

## Starting state

The completed investigation landed at:

- main: `c1222449e45f7ecaa678b5549062edead311a8ad`
- dev: `5f38685bc901310a4f5bc97f8843c62cffb5cc3f`
- exp: `b22bc2b57d5acb98c6e394e64c69364474866e10`
- public mirror: `250b2aca62cd0cecbdb7a4d5cb69367f7b9c5033`

These identify the investigation baseline, not the later commit adding this handoff. Fetch and inspect current
refs before editing. Only main/dev/exp remained locally, clean, equal to origin and contained. The resident
model server was restored: active/running/transient, health and completion successful, build b11571-097cf33d9.
Check its current state rather than assuming the saved verification is still current.

Full Python suites on dev and exp and Go vet/tests passed. The last explicitly logged full run had 2,139 tests
plus 27 subtests; an additional repeat-reader regression passed the final full branch gates. Size was
46,884/52,000. Mirror leak scan, fresh install and selfcheck passed.

## Preserve the completed fixes

- Repeat bands come from two independent runs of the exact setup, one absolute spread per row, with source
  dates/builds and evidence hashes. They are observations, not confidence intervals. Holdouts do not widen them.
- No matching band means `not checked`. Never borrow a chip-wide percentage or millisecond floor.
- Capture follows the request and observed untraced settings: context, allocation, token window, JIT/graph mode,
  worker mode, chunking and backend. Historical context-128 captures cannot certify a context-512 request.
- Pairing uses logical dimensions, semantic identities, data-flow order and source-derived launch constraints.
  The replay-middle ambiguity is resolved. Do not restore either literal 1% timing shortcut.
- Every relevant matrix launch must be accounted for, and captured logical work must match the independent
  model inventory. An empty unpaired-role list alone is not evidence of completeness.
- A fused launch retains one joint measured duration and all its logical work. Do not split its time by bytes,
  or compare a partial fused duration against a full role floor. Standalone quantization is not matrix work.
- Repeat observations are read independently of calibrated report verdicts. Attaching bands must not alter
  the observations or the calibration derived from them.

## Evidence and resume points

Public records:

- `docs/accuracy-battery-rtx5090-20261012.md`: original predictions, results and the first follow-up.
- `docs/accuracy-fixes-20261012.md`: measured causes, landed repairs and remaining limitations.
- `docs/engine-upgrade-audit.md`: repeat-band commands and exact setup rules.

Private evidence on the GPU box is under `~/storage/accuracy-fixes-20261012/`. Read `progress.md`,
`final-state.json`, `final-validation-summary.json` and `final-evidence-index.json` first. Original battery
copies remain under `~/storage/accuracy-analysis/` and the earlier battery locations in the prior handoff.

Important evidence in the final folder:

- `repeat-bands.json` and the `*-holdout.json` files: six independently calibrated setups, third run held out.
- `tinygrad-factorial-summary.json`: JIT/profiling mode experiment.
- `timer-decomposition.json`, `timer-recurrent-summary.json`, `timer-direct-*`, `timer-graph-*`: six timer trials.
- `tg8-readback-final/`: diagnostic prediction, output and status.
- `tg8-readback-rerun/readback-summary.json`: direct timestamp-collection measurements.
- `tinygrad8-final-decode-*`, `tinygrad27-final-decode-*`: three complete context-512 captures per size.
- `trt8-metadata-final-decode-1/`: final supported backend, matching settings and complete matrix inventory.
- `trt27-compatibility.json`, `trt27-current-decode-1/`, `trt-current-quant_config_utils.py`: loader failure evidence.
- `ollama8-fixed-rerun-decode-1/`, `ollama27-fixed-rerun-decode-1/`: corrected allocation validation.
- `tinygrad-runtime-fixes.bundle`: reproducible isolated runtime commits.
- `server-restart-private.json`, `server-restored.json`: private lifecycle evidence. Never publish their contents.

Before new work, create `next-progress.md` in that private evidence folder. Append a dated line after each
investigation, test and GPU run, including revision, result, next action and server state. On a crash, resume
from the last completed entry; do not restart the whole battery.

## 1. External tinygrad deployment boundary — no GPU initially

The protected installed runtime was deliberately not changed. The validated runtime was an isolated patched
checkout at `928aed074a6bc2f3caabfc12ee72424e06a8c200`. It has been removed; its patch and bundle remain.
On dev, read `tools/patches/README.md` and `tools/patches/tinygrad-capture-support.patch`. The patch applies to
`caa752ac45b4793f0991436f29a5d783c4277e22`.

The patch repairs CALL metadata propagation, generic Q8 projection identities and observed input extents,
recurrent prompt stepping, epilogue capability admission, CUDA vector alignment and prompt/decode input views.
The 13 focused external contracts pass. The broader legacy program-construction suite has 35 passes and five
failures because its fixtures omit required promotion tickets. Do not weaken runtime admission to pass them.

Reconstruct an isolated checkout, verify patch applicability and exact source identity, and resolve those
fixture contracts. Deployment to the protected installation is a separate action; preserve that installation
until deployment is explicitly requested. If the source identity changes, previous repeat bands do not apply.

Acceptance: reproducible patch/source identity; focused and applicable broader checks pass; any deployment
records the actual imported source. Do not describe the protected default runtime as already repaired.

## 2. Profiling overhead — targeted GPU work

The primary historical distortion was a comparison of different modes. For the 8B model, JIT1 unprofiled was
4.0875 ms/token, JIT1 profiled 4.4958, JIT2 unprofiled 37.9338 and JIT2 profiled 60.7566. Matching modes fixes
that comparison, but leaves real profiling overhead: final JIT1 runs are still about 9% slower for 8B.

The runtime synchronously reads graph timestamps before another replay and inserts GPU timestamp signals.
The direct collection diagnostic measured 0.945146 ms/token of inclusive host collection work. That can overlap
other segments' device execution: it is not an additive amount to subtract from wall time. Separate Python
submission explains much of the JIT2 slowdown.

Investigate timestamp collection/draining and instrumentation with matched JIT, allocation, token window,
source and compiled-program census. Change one mechanism at a time in an isolated runtime. Write a prediction
before each diagnostic. Retain the measured wall, GPU intervals and residual separately.

The 27B tinygrad original 6% prediction passes (+0.691% first, +0.407% holdout), but its narrower empirical
end-to-end band still fails. llama.cpp and corrected Ollama also fail the original 6% prediction; their exact
remaining overhead causes are not established by the tinygrad diagnostic.

Acceptance: demonstrate a causal improvement without dropping work, tokens or timestamp records; preserve
numerical correctness and complete matrix inventory. Report any remaining perturbation as failure.

## 3. Small-kernel timer precision and isolated routes — targeted GPU work

Direct event-minus-empty estimates for the recurrent projection varied from 1.824 to 3.360 microseconds while
isolated Nsight duration stayed around 2.656 microseconds. Empty-launch subtraction is not an additive
correction to actual kernel execution. Graph timing reduced the raw interval but subtraction overcorrected
further, so that experiment was rejected as a product fix.

Keep raw event intervals, empty-launch readings and isolated Nsight duration separate. The current report
splits discrepancy into estimator error and isolated-versus-in-model route difference. FFN kernels have real
route differences even with hardware timing; a cold-cache check does not prove equivalence to a fused route.

Before changing the estimator, establish the same compiled program, launch geometry, operands, memory regime
and measurement scope. Use independent counterbalanced trials and a held-out check. Do not tune a constant to
make the historical table pass. Investigate the llama.cpp 27B lm_head observation below its modeled floor:
request-matched reports currently refuse that floor comparison beyond the measured repeat spread.

Acceptance: independently validated scope and estimator precision, or an explicit limit on what the reading
can establish. Preserve failed original per-role predictions.

## 4. Remaining role ambiguity and repeat variability

8B llama.cpp and Ollama Q/O remains unresolved after standalone quantization launches were excluded. Obtain
additional trustworthy semantic/launch evidence; dimensions, byte counts or near-equal times alone are not
sufficient where multiple assignments remain possible. Unknown must stay unknown. Joint QKV timing in vLLM
and TensorRT is valid joint evidence, not independent per-role timing.

Two repeats can produce a narrow spread by chance. Holdouts exceeded it for multiple rows across all six
calibrated setups. The source of every change is not yet established. Investigate recorded clocks, workload,
allocation, execution and capture conditions before attributing variation to hardware or changing a policy.
Any proposal requiring more repeats must specify the decision and sampling rule before collecting them.

Acceptance: resolve an ambiguity only with discriminating evidence; explain repeat variation with controlled
observations. Do not absorb a holdout into its own calibration or invent statistical confidence from a pair.

## 5. TensorRT resource and quantization support

The TensorRT-engine 8B first build consumed the host budget and was stopped after 92.6 seconds with 2.44 GB
available on a 31 GiB/no-swap host. Request bounds and caching do not remove the first-build requirement.
Do not retry the same build without a concrete resource/build change.

The current PyTorch backend supports unchanged 8B weights. Final single-process context-512 validation
accounts for 15,136,194,560 FLOP/token, with 9.905436 ms unprofiled versus 9.862798 profiled and ample host
memory. This is a different backend setup, not a repaired TensorRT-engine build. Its default IPC worker lacks
loaded inventory exposure; metadata-complete validation uses an explicitly recorded single-process worker.
Only one final run exists, so it has no independent repeat band.

TensorRT-LLM 1.3.0rc29 / Transformers 5.5.4 recognizes the 27B architecture but its converter fails on null
activation quantization for asymmetric integer-4/group-128 weights with zero points. Existing loader paths do
not implement this format correctly. Relabeling the checkpoint or discarding zero points changes the model.

Acceptance: a supported backend with exact workload/settings and complete attribution; a new loader requires
numerical validation against unchanged weights. A supported equivalent checkpoint is a distinct experiment,
not evidence that the original checkpoint works. Resource/support limits may remain explicit external work.

## Execution, safety and landing

Read `docs/coding-principles.md`, `docs/branch-flow.md`, `docs/commit-discipline.md`, `docs/maintaining.md` and
the rules in the earlier handoffs. Work in small increments; land each tested increment independently.
Product belongs on main, Python tests and external patch apparatus on dev, experiments on exp. No model/chip
exceptions, no `--no-verify`, no raised size cap. Use one agent if delegation is requested; do not add agents
under the previous single-agent assignment.

For GPU work:

1. Inspect current evidence and write the run prediction first. Use isolated work under `~/storage`.
2. Save the resident server's exact current start command privately before stopping it. Use the discovery
   procedure in `docs/handoff-accuracy-20261012.md`; keep its unit/model names in private evidence.
3. Run one engine at a time. Guard host and device memory, kill only owned process groups, and wait for resource
   recovery after a guard stop. A pre-start resource skip is not an engine compatibility result.
4. Restore the exact saved server command in `finally`, including after a failed or cancelled run. Verify health,
   one completion and expected build. Leave protected engine checkouts and the TUI untouched.
5. Append the result and restoration status to `next-progress.md`.

Before landing, verify main/dev equal origin and follow the prior handoff's guarded merge/sync/push procedure.
Run appropriate focused tests, full branch gates and Go checks. Publish from an origin/dev worktree with the
mirror script; its leak scan, fresh install and selfcheck are mandatory. Remove task worktrees and scratch
venvs. End with only clean main/dev/exp, equal to origin and main contained in dev contained in exp.

A completed increment means the fix is tested and its evidence is recorded. A completed investigation means
all requested experiments are accounted for and external limitations are explicitly open. Neither means all
engines pass every prediction. Finish by reporting fixed mechanisms, actual GPU results, unresolved causes,
external deployment/support work, exact refs and server state.
