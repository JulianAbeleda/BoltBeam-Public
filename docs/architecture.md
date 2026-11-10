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
report / output   report/      results.json, report.html, summary.md from the folder (workflow/output.py)
```

Package by package:

```text
profile/        artifact readers (GGUF, safetensors, AWQ, GPTQ, ONNX, MLX), ProfileIR, the role taxonomy, the
                architecture classifier (from tensor facts, not model names)
target/         target capability data (data/targets.json), the vendor rows (target/vendors.py), the GPU a
                run uses (target/gpu.py)
artifacts/      artifact readers for outside evidence, and the run folder's own I/O: read_json, write_json,
                the run manifest (artifacts/run_io.py)
collectors/     one adapter per engine and per capture tool; the pairing walk; what a capture may claim
workflow/       the stages in order (pipeline.py), the layout of a run on its GPUs, the tie-out, the run
                folder's status, the repeat bands, the screen's text
roofline/       the limits: memory (weights, KV, recurrent state), compute paths, MoE and prefill work
report/         the HTML report and the Markdown summary
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

## Import direction

The two largest packages import one way: `workflow -> collectors`. A collector
never imports `boltbeam.workflow`, at module level or inside a function.
`policy/guards.py` scans for it (`collector_workflow_import_failures`) and is
the one authority on the exceptions left from before the rule; its allowlist
ends empty. Below both sit `profile/`, `target/`, `artifacts/`, `roofline/` and
`core/`, which import neither.

## The older modules: search, policy, synth, eval, ledger

BoltBeam began as a profile -> search -> codegen frontend, and that path is
still live: `cli/workflow.py` and `analyze.py` emit the search space
(`search/emit.py`), the seed route policy (`policy/emit.py`) and the synthetic
fixtures (`synth/fixtures.py`); `report/markdown.py` reads the ledger
(`ledger/`); `search/role_*` compare measured roles. They are not retired.
Their contracts:

- `data/quants.json` — quant behavior (block size, dequant family, route
  families, refuted/pending routes). Unknown quant -> `unsupported_quant`.
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
