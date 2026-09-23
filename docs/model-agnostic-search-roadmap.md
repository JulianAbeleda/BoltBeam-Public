# Model-Agnostic Search Roadmap

BoltBeam started as a Qwen/AMD/Q4_K-shaped route-search frontend. The A0–A10
work turned it into a model-agnostic brain that reads dense and MoE transformer
GGUFs, reasons about quants and targets from data, and refuses to pretend it
knows more than the profile tells it. This doc is the map: what shipped, and
where it goes next.

## What shipped (A0–A10)

| phase | capability | where |
|-------|-----------|-------|
| A0 | assumption audit (every Qwen/AMD/Q4_K/dense hardcode classified with a fix path) | `docs/model-agnostic-gap-audit.md` |
| A1 | fine role taxonomy (`RoleClass`) + derived coarse groups (`RoleGroup`); dense parity preserved | `vocab.py`, `profile/roles.py` |
| A2 | architecture classifier: dense / MoE / encoder-decoder / unknown (metadata-first) | `profile/architecture.py` |
| A3 | quant capability registry (block size, dequant family, route families) as DATA | `data/quants.json`, `quant.py` |
| A4 | search emission driven by quant/target capabilities, not `if quant == "Q6_K"` | `search/emit.py`, `search/families.py` |
| A5 | MoE search semantics: 3D expert stacks profiled; router/batched-GEMV/layout/dispatch families | `profile/gguf.py`, `search/families.py` |
| A6 | target capability registry + backend status; descriptor-only backends never promote | `data/targets.json`, `targets.py` |
| A7 | candidate manifest as route-family templates (arch-scoped, origin-tagged); history reproducible | `data/candidates.json`, `manifest.py` |
| A8 | golden fixtures: dense-Qwen / dense-Llama / MoE / unknown-quant / descriptor-only through the full CLI | `tests/ggufkit.py`, `tests/test_golden_pipeline.py` |
| A9 | policy guard: decision path registry-driven (source-scanned); unknown architecture fails closed | `policy/guards.py` |
| A10 | handoff docs (this file + README/architecture/principles) | `docs/` |

Core invariant across all of it: **a new dense/MoE/unknown profile emits an
honest search space, and no candidate promotes unless model, target, quant, role,
context, and search-space applicability all line up.**

## The data authorities (extend by editing data, not code)

- `data/quants.json` — quant behavior. A quant absent here is `unsupported_quant`
  (search-space-incomplete), never silently dropped.
- `data/targets.json` — target capability + backend status. Only `complete`
  backends promote; `descriptor_only` (nvidia_sm89, apple_metal) defer.
- `data/candidates.json` — route-family templates keyed on (route_family, quant,
  roles, architectures); shipped/refuted decisions kept for reproducibility, the
  durable truth being the ledger.

## Where it goes next: parity, then past it

The reason the brain exists is to drive real codegen. The live target is Qwen3
**14B and 32B decode**, using both models to iterate one search space:

- Parity/exceed plan: `docs/parity-14b-32b-scope.md`.
- The objective is the HBM roofline fraction (960 GB/s peak), not llama; llama
  reaches ~62% of peak, so beating it and climbing to the roofline are the same
  instruction. 8B already runs at 109% of llama as the existence proof.
- Three pinpointed levers (reduce elimination ~52%, split-K decode for
  occupancy-starved KV, Q6_K ffn_down) feed the loop as candidates; the refuted
  axes (topology tuning, ffn_down split-K, sub-4-bit) stay walled in the ledger.
- "Use both to iterate both": a lever promotes generally only when it clears the
  bar on 14B AND 32B evidence; divergence shows where a shape-specific route is
  genuinely needed.

## Still open (honest backlog)

- Encoder-decoder profiling is classified but not yet given route families.
- Target capability pruning of knob grammars (A6 data exists; `family_knobs`
  accepts target but does not yet prune by wave/LDS).
- The MoE and SSM route families are templates; no hybrid SSM/MoE model has
  been promoted yet. Qwen3.5-style profiles are now classified and emitted as
  search spaces, but speed/correctness promotion still requires real tinygrad
  evidence.
- The parity levers (L1–L3) are scoped, not yet executed on-GPU.
