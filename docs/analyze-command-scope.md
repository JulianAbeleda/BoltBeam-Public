# BoltBeam Analyze Command Scope

## Purpose

`boltbeam analyze` is the Claude handoff path. It turns a model artifact into a
complete, reproducible route-search starting point:

```text
GGUF
  -> model_profile.json
  -> search_space.json
  -> route_policy.seed.json
  -> fixture_manifest.json
  -> next_measurement_plan.json
  -> tinygrad_commands.md
```

The command does not promote routes and does not run the GPU. It prepares the
facts, candidate families, and tinygrad measurement commands so an agent can
start from measured evidence instead of redoing shape archaeology.

## Contract

- Model facts come from the GGUF profile reader.
- Search candidates stay profile-scoped and target-scoped.
- Measurement starts from role attribution, then reduce-source resolution, then
  runtime-overhead separation.
- Generated routes may replace hand routes only after correctness,
  route-binding, speed, memory, and rollback gates pass.
- A failing candidate that exposes a missing topology knob is recorded as
  `search-space-incomplete`, not as a refuted route family.

## Artifacts

| artifact | role |
|---|---|
| `model_profile.json` | Tensor census with role, shape, quant, and count. |
| `search_space.json` | Legal route families and candidate knobs per role. |
| `route_policy.seed.json` | Unmeasured seed policy for downstream promotion. |
| `fixture_manifest.json` | Synthetic shape probes for codegen tests. |
| `next_measurement_plan.json` | Ordered measurement phases and role priorities. |
| `tinygrad_commands.md` | Copy-ready tinygrad commands for role attribution and tracing. |
| `analysis_manifest.json` | Bundle index and summary counts. |

## Non-Goals

- It does not synthesize a GGUF.
- It does not compile kernels.
- It does not benchmark or promote a candidate.
- It does not hide missing tinygrad tools; tool presence is recorded in the
  measurement plan and command document.

