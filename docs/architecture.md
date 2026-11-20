# BoltBeam Architecture

BoltBeam audits where inference time goes against a measured roofline
(`boltbeam --help`). One run is one folder; every stage reads the folder and
writes its artifacts back into it, and the report is rendered from the folder
alone, so a run can be fetched from a remote GPU machine and read anywhere.

## The pipeline the CLI runs

```text
inspect / load    profile/     the model file -> ModelProfile -> model_profile.json, weight_inventory.json
autoscan          workflow/    this machine: GPUs, engines, toolchains -> hardware_profile.json, the chip profile
                  target/      the target registry, the vendor rows, the GPU a run uses
measure           collectors/  an engine run and its outside capture (nsys, rocprofv3, Metal): the untraced
                               token, the kernels, the pairing of kernels with weight roles (collectors/pairing.py)
tie-out           workflow/    the measured token against the limit: roofline/ gives each role's floor from the
                  roofline/    bytes it reads and the measured bandwidth and matrix rates; tie_out.py says where
                               the token loses time, and the capture-overhead ceiling
search            search/      per role, where the engine and chip have a search provider: the kernel lifecycle
                               below (search/search_stage.py; skipped with --no-search, and on a replay)
report / output   report/      results.json, report.html, summary.md from the folder (workflow/output.py)
```

Package by package:

```text
profile/        artifact readers (GGUF, safetensors, AWQ, GPTQ, ONNX, MLX), ProfileIR, the role taxonomy, the
                architecture classifier (from tensor facts, not model names)
target/         target capability data (data/targets.json), the vendor rows (target/vendors.py), the machine's
                GPU scan (target/hardware.py), the GPU a run uses (target/gpu.py)
artifacts/      artifact readers for outside evidence, and the run folder's own I/O: read_json, write_json,
                the run manifest (artifacts/run_io.py)
collectors/     one adapter per engine and per capture tool; the pairing walk; what a capture may claim
workflow/       the stages in order (pipeline.py), the layout of a run on its GPUs, the tie-out, the run
                folder's status, the repeat bands, the screen's text
roofline/       the limits: memory (weights, KV, recurrent state), compute paths, MoE and prefill work; a run's
                ceiling and the one memory speed its numbers use (roofline/limit.py)
report/         the HTML report and the Markdown summary
search/         the kernel lifecycle: proposal and legality (BubbleBeam, FutureSight), the provider and BoltBeam's own
                check of it, the whole-model A/B and the decision (section below)
runtime/        the drivers BoltBeam runs on a GPU: the vLLM, TensorRT-LLM and Ollama decode drivers, the
                CUDA, HIP and Metal device bridges, the bounded build jobs
kernel_analysis/ kernel evidence: ingestion, classification, diagnosis, recommendation
core/           shared types, canonical JSON, the policy accessor (core/policy.py reads data/policy.json)
data/           the authorities: quants, targets, candidates, engine pins, the policy table
vocab.py        the vocabularies: roles, architectures, schema ids
cli/            command wiring only; CORE in cli/__init__.py is the grouped list `boltbeam --help` shows
```

Every stated choice (a tolerance that is not a measured uncertainty, a search
limit, a ceiling) is a row of `data/policy.json` with its reason, read through
`core.policy.value()`. A missing row is an error, never a default in code.

## Search and validate: the kernel lifecycle

The kernel lifecycle is propose -> check -> time -> gate -> install -> retire. It is one stage of the optimization
cycle: audit (the session map's per-role audit, docs/session-map.md) -> search -> gate -> ship, then audit again. That
is the APOD cycle (Assess, Parallelize, Optimize, Deploy) of NVIDIA's CUDA C++ Best Practices Guide
(https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), guided by the roofline: the audit's floors say where
the time is, and the ranked queue says what to do next.

The audit names the rows that lose time; the search stage works on them, one
role at a time (`search/role_compare.py`):

```text
propose    role_space.py, bubblebeam.py   the role's finite space from what the decode binds through on this device;
           futuresight.py                 BubbleBeam proposes values legal for the chip's facts, FutureSight rejects
                                          the illegal ones statically, both recorded
search     tinygrad_fork.py               the provider (the tinygrad fork) runs the space; its times only order it
check      provider_check.py              the provider is not trusted: its device identity against BoltBeam's own
           role_check.py                  reading; every candidate rebuilt from the source it returned, run on the
                                          role's own weight bytes, checked against BoltBeam's reference, timed with
                                          BoltBeam's kernel timer; not rebuilt, incorrect or noisy = not reproduced,
                                          never promoted
validate   role_decision.py               the winner against the default in a matched whole-model decode A/B
                                          (runtime/tinygrad_decode_ab.py), with a binding check that the plan
                                          reached the calls it claims
decide     role_decision.py               the evaluator's CandidateDecision (correctness, binding, speed tier, and
                                          its rollback to the default plan), appended to the route ledger; the
                                          route's status written into route_policy.json
```

Fusions (several adjacent decode-graph nodes in one launch) run the same path
(`search/fusion_compare.py`), each judged on the in-model time of the rows it
replaces. A kernel that wins its own timing but loses the whole-model A/B is
recorded with that result and not promoted.

Scope today: the whole-model A/B installs tinygrad Opt-sequence plans only. An
emitter plan is searched and checked, then stops as "decided by binding": it
reaches the model through the fork's route admission, which the A/B driver
does not install yet. The kernel numbers are BoltBeam's own; the whole-model
numbers are tinygrad runtime times, never a llama.cpp time.

## Import direction

The two largest packages import one way: `workflow -> collectors`. A collector
never imports `boltbeam.workflow`, at module level or inside a function.
`policy/guards.py` scans for it (`collector_workflow_import_failures`) and is
the one authority on exceptions; its allowlist is empty. Below both sit
`profile/`, `target/`, `artifacts/`, `roofline/` and `core/`, which import
neither.

## The emit path: search space, policy, synth, eval, ledger

BoltBeam began as a profile -> search -> codegen frontend, and that path is
still live: `cli/workflow.py` and `analyze.py` emit the search space
(`search/emit.py`), the seed route policy (`policy/emit.py`) and the synthetic
fixtures (`synth/fixtures.py`); `report/markdown.py` reads the ledger
(`ledger/`), which the kernel lifecycle above also writes. They are not retired.
Their contracts:

- `data/formats.json` — every weight, KV-cache and activation format (docs/formats.md): names, block size, bits, each component's status, and the quant capability (dequant family, route
  families, refuted/pending routes). Unknown format -> refused in words; no capability -> `unsupported_quant`.
- `data/targets.json` — target capability + `backend_status`. Only `complete`
  backends promote; `descriptor_only` targets defer.
- `data/candidates.json` — route-family templates (arch-scoped, origin-tagged).

The decision path (search emission, route families, seed policy, evaluator) is
model/quant/target-agnostic and is source-scanned on dev by `policy/guards.py`.
An `unknown_transformer` profile fails closed: no authorized search space, all
routes blocked. See `model-agnostic-search-roadmap.md`.

## Hybrid and MoE profiles

Hybrid SSM/MoE profiles are classified from facts, not model names. SSM metadata
(`<arch>.ssm.*`) or tensor names (`ssm`, `delta`, `conv1d`, `dt_proj`, `a_log`)
produce `hybrid_decoder`; the same signal plus MoE expert/router metadata
produces `hybrid_moe_decoder`. Hybrid profiles inherit the dense route surface
for ordinary attention/FFN roles, hybrid-MoE profiles also inherit MoE expert
families, and SSM tensors get their own `ssm_*` route families.

Every weight the decode reads once per token is a role of the roofline limit,
named by what it is (`vocab.WEIGHT_GEMV_ROLES`): the dense roles, a fused
`attn_qkv` projection, the `attn_gate` beside it, and the SSM matrices split by
tensor (`ssm_out`; `ssm_alpha` and `ssm_beta` grouped as `ssm_alpha_beta`, as k
and v are). The per-role tables key on (role, quant), so two different tensors
of one quant are two names, never one row. These SSM matrices are quant GEMVs
and take the quant route families; conv, state and scan tensors keep the
`ssm_*` families. An SSM matrix the patterns do not name (`ssm_in`) stays
`ssm_projection`, outside the limit, and the tie-out names its GEMV after it in
other kernels.

Kernel classification (which capture-tool kernel names are the attention or
recurrent block, which are weight GEMVs) is a table the pairing walk reads
(`collectors/kernel_kind.kind_of`, beside the pairing). Its words are data,
`data/kernel_kinds.json`, first match wins. An engine that renames a kernel
drops the walk to the flat pairing; the engine-upgrade audit reviews that diff
(`tests/test_kernel_kinds.py` pins the classification per engine version).

## Why Not Generate A Full Synthetic GGUF First?

Synthetic GGUFs are useful for loader and route-policy testing, but the fastest
codegen loop is synthetic shape graphs:

```text
real GGUF -> ProfileIR
ProfileIR -> synthetic graph fixture
ProfileIR -> search candidates
ProfileIR -> runtime route policy
```

Real GGUFs remain the authority for promotion.
