# CUPTI hardware event timing

`BOLTBEAM_CUDA_CAPTURE=cupti-hes` selects a process-lifetime kernel timing collector
instead of Nsight Systems. It requires hardware event tracing support in the NVIDIA
GPU/driver, a CUDA 13 CUPTI SDK, CUDA headers, and `g++`:

```bash
export BOLTBEAM_CUDA_CAPTURE=cupti-hes
export BOLTBEAM_CUPTI_ROOT=/path/to/cupti-sdk # contains include/ and lib/
```

Use the normal llama.cpp decode, prefill, or session capture command. Each capture
builds the packaged helper and retains its source, library, raw mangled-symbol CSV,
clock anchor, completion count, and stable library/source hashes in `capture_identity.json`.
Symbols are decoded after capture with `cu++filt`; integer timestamps are made relative
to the CUPTI clock anchor before conversion to floating point. Only CUPTI and its profiler dependencies are added to the
loader path; the engine keeps its own CUDA runtime and math libraries. The helper
enables hardware tracing before creating a CUDA context and collects concurrent
kernel activities without CUDA API tracing. It refuses failed initialization,
missing completion, dropped records, or a disagreement between the completion
count and CSV records. Its reported capture method is `cupti-hes`.

This option supports complete processes. Server termination, profiler API ranges,
and semantic/API replay captures require their existing collectors. Nsight Systems
remains the default when the option is absent.

Selecting a collector does not qualify its overhead. A capture qualifies only when
its native and captured timing windows match and its measured host-timer overhead is
at most the 5% capture-overhead ceiling, and a capture more than 5% faster than native is refused as not
comparable; Run and Emit disclose that overhead.
Qualification under controlled GPU clocks describes that clock policy only; it
cannot establish timing neutrality at automatic clocks. The collector never changes
GPU clocks or power settings.

For a clock controlled externally, set `BOLTBEAM_TIMING_SM_MHZ` to the requested
SM frequency before running the pipeline. Both native and captured stages then
retain NVML samples in `measurement_clock.json`. Repeat bands require complete
observations with both peak clocks within 2% of that request and each other;
they are scoped to that frequency and the retained collector identities. These
observations identify a frequency regime, not the driver's lock state or a
substitute for qualification's interval-level clock balance checks and thermal disclosure.
Missing or invalid observations cannot borrow a controlled-clock band.
