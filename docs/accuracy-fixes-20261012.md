# Accuracy corrections and validation, 2026-10-12

This record follows the [accuracy battery](accuracy-battery-rtx5090-20261012.md). Historical predictions and results remain unchanged. A code correction does not repair a historical capture made at the wrong context. New runs must establish the requested settings and, for an accuracy verdict, supply independent repeat calibration for the exact setup.

## Pairing without timing twins

The replay-middle ambiguity had a structural cause: the solver admitted a fused gate/up interpretation that could not produce the observed launch grid. The matching engine source configures a 128-row tile for that template. With the captured dimensions and device compute-unit count, the separate gate/up products can launch 170 blocks and down 160; the alleged fused gate/up product would launch 1088. That alternative is impossible under every matching source configuration.

The adapter reads the installed engine's launch formula and every matching configuration, retaining a conservative union instead of choosing a chip table. Automatic use requires the captured build revision to match HEAD and the source files to match that revision's committed bytes. Unknown templates or changed launch contracts provide no constraint. The evidence includes source hashes. This establishes consistency with the engine-reported build; it does not independently attest a loaded binary's provenance.

The solver also applies explicit logical dimensions and multiplicity when supplied, respects data-flow ordering when testing unused alternatives, and no longer uses either literal 1% timing shortcut. Equal bytes and near-equal times do not identify two different roles. Unsupported alternatives stay ambiguous. The saved replay now pairs without ambiguity; its measured kernel sum remains 1179.299979 ms. The regression corpus includes that capture and a deliberately unresolved equal-byte/count case.

## Tinygrad launch semantics and timing

Three shared-activation matrix programs lost their logical identities across the opaque CALL boundary. An isolated runtime patch attaches the actual linears' tensor facts before activation packing and carries those identities on the CALL. The normalizer deduplicates repeated metadata within one call, retains distinct fused tensors, and refuses conflicting same-program descriptions. It checks every unresolved matrix or unknown launch, independently of the list of unpaired model roles. A separate inventory check sums the expected projection work by role, format and dimensions and compares it with the observed programs. Thus a generated reduction name cannot hide missing matrix work behind an apparently complete role list.

Fresh 8B decode contains 418 launches/token and 12 semantic matrix-program rows. All three formerly unknown programs have identities. Captured matrix work is **15,136,194,560 FLOP/token**, exactly the independently loaded tensor inventory, including two gate/up products and mixed-format K/V. All three final context-512 captures retain the original compiled-binary multiset and cover all 6,688 launches. A fused launch with different logical role keys retains a joint duration; no byte-weighted split or partial-time/full-floor comparison is fabricated.

The four-mode experiment used context 512, 16 measured tokens, three warm tokens and the same allocation. Every mode has the same 418-call compiled-binary census.

| JIT | Profiling | Wall ms/token | Kernel interval sum ms/token |
| --- | --- | ---: | ---: |
| 1, graphs | off | 4.087512 | not captured |
| 1, graphs | on | 4.495759 | 3.909844 |
| 2, separate dispatches | off | 37.933828 | not captured |
| 2, separate dispatches | on | 60.756552 | 4.217094 |

The old approximately 60 ms versus 4 ms comparison changed both graph mode and profiling. In this controlled experiment, profiling adds 9.99% within JIT1 and 60.16% within JIT2. The original 6% perturbation prediction still fails. Kernel execution alone does not explain Python dispatch and instrumentation time. The runtime source shows that JIT1 reads every graph timestamp synchronously before the next replay and inserts device timestamp signals; JIT2 additionally submits each program through Python. These are different costs from the measured kernel intervals. The final three JIT1 repeats measure 4.476759, 4.593038 and 4.469386 ms profiled wall per token, versus 4.108561 ms unprofiled in the first run and 4.103004 ms in the holdout. Binary identities remain unchanged. A separate direct readback diagnostic measures 0.945146 ms/token inside graph timestamp collection (64 calls for 16 tokens, four graph segments and 836 timestamp reads per token), with a total wall of 4.473708 ms/token. This is inclusive host work that can overlap device execution of other segments. It is not an additive correction, and is not subtracted from the measured wall.

Capture now reuses the untraced JIT mode, checks the token window/allocation/build, records source changes in the build identity, and drains delayed HCQ graph timestamps without including a warm or extra token. Reports reconcile kernel intervals to the captured wall and show the difference from unprofiled wall separately. The residual is labelled as time outside GPU kernels in the profiled wall, including host and profiler overhead.

## What the kernel-timer discrepancy measures

Three counterbalanced repeats of direct event timing and three of event/kernel/event graph replay each passed numerical and cache checks. The small recurrent projection demonstrates that empty-launch subtraction is not an additive correction to kernel execution:

| Direct-event trial | Event interval µs | Empty-launch interval µs | Difference estimate µs | Isolated Nsight duration µs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6.240 | 4.416 | 1.824 | 2.656 |
| 2 | 6.704 | 3.984 | 2.720 | 2.656 |
| 3 | 7.376 | 4.016 | 3.360 | 2.656 |

Historical arithmetic was 6.896 − 3.904 = 2.992 µs with isolated Nsight 2.672 µs. Thus the recurrent sign reversal is in the event/floor estimator while the measured kernel duration stays stable. CUDA events measure an asynchronous interval around submitted work, not an isolated hardware kernel duration; NVIDIA documents the distinction and approximately 0.5 µs resolution in [the Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html).

Graph replay lowers the raw small-kernel interval to 3.232–3.392 µs, but subtracting its empty launch yields only 0.928–1.248 µs against Nsight 2.656–2.688 µs. That experiment is rejected as a correction. No constant is retuned. The product preserves raw intervals and difference estimates and, when Nsight is collected, explicitly reports their discrepancy and the measurement scopes.

The comparison now decomposes total error into `(event − empty − isolated Nsight) + (isolated Nsight − in-model Nsight)`. Across the direct repeats, median second terms for FFN down and gate/up are −1.536 and −1.680 µs. Those differences remain even with hardware timing: the isolated source adapter and fused in-model launches execute different routes. The cache check establishes DRAM eviction, not in-model equivalence. The original per-role prediction remains failed; the event-floor estimate cannot support that precision for the smallest kernels.

## Runtime support and fresh request validation

The protected engine installations are unchanged. The isolated tinygrad patch also honors single-token recurrent prefill, checks a promoted epilogue against its own dimension contract, fixes non-power-of-two CUDA vector alignment without changing packed size, and normalizes prompt input views to the decode feedback layout.

The first successful full-context 27B run exposed a second metadata gap: generic Q8 projections contributed only 37,774,950,400 of the inventory's 51,243,909,120 FLOP/token. The original timing remains saved; its attribution now explicitly fails the independent inventory check. One recurrent QKV program accepts both an activation and a larger state buffer. Buffer dimensions alone cannot identify the matrix input extent. The runtime now constrains candidates with the actual observed module input extent; multiple remaining candidates stay unknown. A short diagnostic recovers the complete inventory, but is not counted as validation at context 512. Three fresh context-512 runs each cover 27,984 launches (1,749/token), 50 kernel rows and 12 matrix-program rows, with the full **51,243,909,120 FLOP/token** and no unresolved launches or unpaired logical roles. The first run measures 277.094235 ms/token unprofiled, 279.007593 ms/token profiled and 278.149438 ms/token in device intervals. All three compiled-binary multisets match the original metadata-incomplete full-context run: the attribution repair did not change the timed programs. The fallback kernels are slow; attribution does not imply an optimized route. These runs use the final isolated runtime commit `928aed074a6bc2f3caabfc12ee72424e06a8c200`. Its patch and focused contract tests are preserved on dev under `tools/patches/`; applying that external runtime patch is a separate deployment step. The 13 focused contract/model-fact tests pass. Including the broader legacy program-construction tests gives 35 passes and five failures: those fixtures do not supply the promoted runtime ticket now required by the runtime. That suite is not claimed green, and its promotion checks were not weakened.

TensorRT capture now records and reuses scheduler capacity/backend, derives capacity from the request, enables its engine build cache, and brackets only the measured request with the profiler API. The original TensorRT-engine path still cannot complete the 8B build within this host budget: the guard stopped it after 92.6 seconds when available host memory reached 2.44 GB (31 GiB RAM, no swap). The build cache cannot repair the first build. No kernel OOM was logged and no timing from the interrupted build is accepted.

An isolated TensorRT-LLM 1.3.0rc29 / Transformers 5.5.4 runtime recognizes the 27B architecture, but its compressed-tensors converter fails on null activation quantization (`inputs_quant_config["strategy"]`, `TypeError`). The checkpoint uses asymmetric integer-4 weights with group size 128 and zero points. The existing loader paths do not implement that format; the AWQ linear path does not retain its zero points. Relabelling or silently changing weights would not be a valid compatibility fix. A new numerically validated loader or a supported equivalent checkpoint is required.

The current PyTorch backend does successfully run the unchanged 8B weights: 9.932713 ms/token unprofiled versus 9.931911 ms profiled (−0.0081%), with at least 21.15 GiB host memory available. This is a distinct backend setup, not a repaired TensorRT-engine build. Its default IPC worker does not expose the loaded model inventory to the caller, so the initial capture lacks packed-QKV identities. A final explicitly recorded single-process worker run validates the loaded Linear dimensions and packing metadata: exactly 15,136,194,560 FLOP/token, no missing matrix work, 9.905436 ms unprofiled and 9.862798 ms profiled (−0.4305%). Kernel intervals total 9.794720 ms with a 0.068079 ms residual outside kernels and zero arithmetic residual. Both modes record the worker setting, capacity 512 and maximum sequence length 584. At least 23.01 GiB host memory remained available. This single run passes the original 6% prediction but has no independent repeat band, so repeat confirmation stays `notchecked`. The default IPC run remains a separate, metadata-incomplete setup.


Fresh decode repeats use initial context 512. The first two independent runs calibrate the exact setup; the third is held out. Their kernel-plus-residual arithmetic reconciles. The comparison reads observed times independently of a report’s calibrated floor verdict, avoiding circular recalibration. Attaching a band does not change the observations or the band derived from them. Normal reports continue to enforce their floor checks. The observed two-run spread is not widened to include the holdout, and is not a confidence interval.

| Runtime / weights | Rows outside observed spread in holdout | Profile wall error, first / holdout | Original 6% wall criterion |
| --- | ---: | ---: | --- |
| llama.cpp 8B GGUF | 5 / 14 | +10.437% / +10.221% | FAIL |
| llama.cpp 27B GGUF | 15 / 19 | +21.267% / +8.833% | FAIL |
| vLLM 8B BF16 | 6 / 10 | +0.288% / +0.252% | PASS |
| vLLM 27B AWQ | 7 / 15 | −0.251% / −0.366% | PASS |
| tinygrad 27B GGUF, isolated runtime | 5 / 15 | +0.691% / +0.407% | PASS |
| tinygrad 8B GGUF, isolated runtime | 2 / 13 | +8.962% / +8.930% | FAIL |

All three llama.cpp 27B captures and all three final 27B tinygrad captures pair their modeled roles. The six vLLM captures account for the complete matrix inventory, but packed QKV timing remains joint. Separate per-role precision is unavailable for that joint duration. The three llama.cpp 8B captures leave Q/O unpaired after known standalone quantization kernels are excluded from projection candidates; the available launch structure does not establish the missing role with certainty. The llama.cpp 27B calibrated reports refuse their role-loss table because the measured `lm_head Q6_K 248320x5120` time falls below its modeled floor by more than its repeat spread. The raw repeat comparison remains available as observations, not a validated floor claim. The 27B tinygrad run passes the original 6% criterion, but fails confirmation within the narrower measured end-to-end repeat band of 1.013410 ms; the holdout wall discrepancy is 1.132835 ms. llama.cpp uses a 32-token request and a 31-step long-minus-short capture; vLLM uses 64 and 63 respectively. The untraced and captured allocations/settings agree. These fresh results do not repair any historical context-128 capture.

The first new Ollama runs revealed an omitted allocation argument: the collector read the untraced `num_ctx` but did not forward it, so the short run chose a different allocation. The omission is corrected and both capture commands are covered by a regression. Those first new captures retain their failed matching-settings status. Corrected reruns verify an actual 512-token prompt and `num_ctx=640` in both the long and short commands. Ollama 8B measures 4.093391 ms unprofiled versus 4.346095 ms profiled (+6.1735%); 27B measures 13.731078 versus 14.591286 ms (+6.2647%). Both fail the original 6% criterion. The 27B roles pair; 8B Q/O remains unresolved for the same structural reason as llama.cpp. A quantization launch can no longer masquerade as that projection. Neither single Ollama run has an independent repeat band, so repeat confirmation is `notchecked`.


## Evidence and limits

The retained evidence set is `accuracy-fixes-20261012`: `progress.md` is the resumable chronology; each run keeps its prediction, command log, raw capture, status and timing artifacts. `repeat-bands.json` uses only the first two independent runs of each exact setup; the `*-holdout.json` files compare the third without widening those bands. Re-attribution preserves original traces, bands and audits with `before_final_semantics` backups and leaves GPU times unchanged. Timer evidence includes `timer-decomposition.json`, `timer-recurrent-summary.json` and the direct/graph counterbalanced captures; the runtime patch is preserved as both a git bundle and the dev patch. The timestamp diagnostic retains `readback-summary.json` and its event/binary census.

The pre-start resource skips immediately following the guarded TensorRT build are not architecture or accuracy outcomes. They were rerun after resource recovery. All GPU runners restore the saved serving command in `finally`; health, a completion request and the expected build were verified after each batch. Private serving commands and model locations stay outside the published tree.

The remaining limitations are measured failures of the original profiling/role-timer predictions, insufficient launch metadata for 8B llama.cpp/Ollama Q/O, shared durations that cannot be split into independent role times, the TensorRT-engine build's host budget, and the 27B checkpoint format in the current TensorRT loader. The isolated tinygrad repair still requires a separate deployment to the protected runtime. Historical captures made at context 128 remain `notchecked` for a context-512 request. No failure is converted into a pass by a code change, timing subtraction, wider band or invented identity.

Follow-up, 2026-10-13: the chip-wide bands and the 105% pairing slack are replaced by each figure's measured uncertainty, and the repeat rule now states n and its false-move rate. See "Measured uncertainty" in [the remaining accuracy causes](accuracy-next-20261012.md).
