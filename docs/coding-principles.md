# BoltBeam Coding Principles

These rules are the BoltBeam copy of the project coding principles. They are
build rules, not decorative style preferences.

## Core Rule

Favor:

- centralization of authority;
- modularization of execution;
- abstraction for simplicity;
- orthogonality for independence.

Centralize what defines the system. Modularize what carries the system out.
Abstract what should stay simple at the interface. Keep independent concerns
orthogonal so change stays local.

## Centralize

Keep one clear authority point for:

- model profiles and durable schema shapes;
- target capabilities;
- route-family rules;
- route-policy and promotion contracts;
- generated handoff artifact names.

Do not duplicate load-bearing rules across scripts, commands, and docs. If
multiple parts need the same rule, move the rule into one explicit source of
truth.

## Human-Facing And Machine-Enforced

When a useful human rule is cheap to enforce, do both:

- document the rule plainly;
- add a checker, schema, test, or hook.

Do not leave load-bearing rules as prose when a small script can enforce them.
Do not over-enforce judgment-heavy rules whose checker would create more drift
than it prevents.

## Modularize

BoltBeam modules own narrow execution surfaces:

```text
profile/  artifact readers and ProfileIR
search/   candidate-space emission
policy/   seed and measured route policies
synth/    shape-only proving fixtures
eval/     promotion/evaluation contracts
cli.py    command wiring only
```

New behavior should extend the owning module instead of adding an unrelated
one-off script.

## Abstract

Hide implementation complexity behind stable, legible interfaces. Prefer a
small command or function whose output is a durable contract over a loose helper
that returns ad hoc dictionaries.

Good BoltBeam abstractions:

- `ModelProfile` and `TensorRole`;
- stable JSON schemas;
- `emit_search_space(profile, target)`;
- `emit_analysis_bundle(...)`.

Bad abstractions:

- wrappers that silently redefine route policy;
- helpers that skip profile/schema validation;
- model-name branches where shape/quant facts should drive behavior.

## Orthogonalize

Keep distinct concerns independent:

- artifact parsing is separate from route search;
- candidate search is separate from promotion;
- synthetic fixtures are separate from real model gates;
- tinygrad measurement commands are emitted, not executed, by BoltBeam;
- target capability data is separate from model profile data.

## Encode Invariants

Make invalid states hard to represent. Use dataclasses, schemas, constructors,
and validation boundaries for durable facts. Avoid relying on caller discipline
for rules that can be encoded.

Examples:

- profile objects carry role, shape, quant, and count together;
- output artifacts use named schemas;
- commit messages use an enforced subsystem prefix;
- target capability differences are explicit data.

## Keep Public Surfaces Boring

The implementation may do complex GGUF parsing or profile ranking. The caller
should see predictable commands:

```bash
boltbeam inspect ...
boltbeam emit-search ...
boltbeam emit-policy ...
boltbeam synth ...
boltbeam analyze ...
```

Do not expose internal route-search machinery as user-facing flags unless the
flag represents a real, stable choice.

## Separate Ergonomics From Semantics

Convenience should preserve the same failure modes and authority boundaries as
the lower-level API. A bundled command such as `analyze` can write many files,
but it must not silently promote a route or run a benchmark.

## Treat Errors As System Information

Errors at artifact, schema, CLI, and integration boundaries should answer:

1. What failed?
2. Where did it fail?
3. Is the failure recoverable?
4. Does a caller need a typed distinction?
5. What context would help debugging?

Broad errors are acceptable only where the caller cannot take a more specific
action.

## Contain Dangerous Power

BoltBeam should not mutate runtime defaults, launch GPUs, or edit tinygrad as a
side effect of profile analysis. Those operations belong behind explicit
downstream commands or evaluators.

## Design For Replacement

Model artifact readers adapt into `ModelProfile`; search and policy code should
not care whether the source was GGUF, Safetensors, AWQ, GPTQ, ONNX, MLX, or a
future tinygrad graph capture. A new reader is acceptable when it preserves the
same failure mode: either a complete profile with scoped roles, or an explicit
unsupported/incomplete result.

Targets are replaceable too: `amd_gfx1100`, CUDA, and Metal should differ
through `TargetProfile` capability data, not through scattered branches.

## Test Behavior At The Boundary

Use the cheapest test that catches the real failure mode:

- unit tests for pure profile/search/analysis logic;
- schema checks for durable JSON contracts;
- CLI smoke tests for command wiring;
- real model gates only for correctness and speed claims.

## Classify Evidence Before Fixing Mechanisms

When a bug crosses compiler, runtime, hardware, benchmark, or policy boundaries,
classify what each evidence row is allowed to prove before changing code or
changing candidate state.

Trust invariants over symptoms. A microgate may prove a local invariant, but it
does not prove a full route is safe. A W==D or model artifact may expose a real
regression, but it is not trustworthy until the harness is known to populate the
right state and measure the intended path.

Separate:

- compile failure from numeric failure;
- local microgate evidence from integration evidence;
- correctness evidence from speed evidence;
- route binding from route promotion;
- harness failure from system failure;
- fix-off bug pins from fix-on success criteria.

Then fix or classify the smallest violated invariant. Do not let a passing
narrow gate authorize a wider claim, and do not let a broken harness turn noise
into a route or compiler verdict.

## Reducing Code The Right Way

Tiny means understandable, not smallest possible line count. Optimize for one
authoritative source per piece of knowledge.

DRY means duplicated knowledge, not duplicated shape. Similar-looking code is
not a merge target unless it represents the same rule. Prefer accidental
duplication over a leaky abstraction.

Use the rule of three: abstract only after a pattern repeats and is genuinely
identical. When near-duplicates exist because inputs differ, canonicalize the
inputs first, then collapse the code.

Prefer data over code: a new quant, target, route family, or candidate should
usually be a table row or profile fact, not a copied `main()`.

## Performance Search Rules

BoltBeam is a search frontend. A search result is not a win until downstream
measurement proves it.

Label states explicitly:

- `diagnostic`: explains a bottleneck;
- `candidate`: worth measuring;
- `shipped`: passes the real gate with rollback;
- `refuted`: failed the relevant gate with a recorded reason;
- `deferred`: blocked by a named missing capability.

If a generated candidate fails because a topology knob is missing, record
`search-space-incomplete`. Do not hand-write a one-off route and call the search
complete.

## Audit-Brain Safety Rules

These rules are specific to BoltBeam's role as the route-search audit brain.
They are the local port of the tinygrad-side audit discipline: never credit a
candidate unless the evidence is scoped to the same model, target, workload,
quant, role, and context that the candidate claims.

- Evidence rows are scoped facts, not a shared pool. A candidate may use a row
  only when that row's workload, quant, role, context, and target apply to the
  candidate or are explicitly global.
- Multi-context evidence must be reduced conservatively. A protected-context
  regression in any measured context blocks promotion, even if another context
  is a tier-A win.
- Search space is an authority input. `evaluate --profile --search --evidence`
  must not promote candidates that are absent from, unreachable in, or
  inconsistent with the supplied profile/search-space pair.
- Target identity must be explicit or derived from artifact facts. Do not stamp
  `amd_gfx1100` on evidence whose producer did not actually identify gfx1100.
- Adapter output must be evaluator-ready. If an adapter emits a speed row, it
  must carry either a signed delta or enough baseline/candidate data for the
  evaluator to compute one.
- Promotion recommendations that cite a practical roofline must report the
  formula outputs from `practical-roofline-promotion-scope-20260702.md`:
  `P_worst`, `G_worst`, `A_worst`, ceiling basis, and action. A route inside
  the closeout band may stop further search, but it promotes only with explicit
  replacement intent plus the normal correctness/route/rollback guardrails.
- Unknown is not pass. Absence of a fallback, determinism, memory, or route
  signal cannot become proof by convenience; the evaluator may tolerate unknowns
  only where the promotion contract explicitly says they are conditional.

## Model-Agnostic Rules

BoltBeam handles new dense/MoE GGUFs, quants, and targets by profile facts, not
by model-specific assumptions (audit A0–A10). The load-bearing rules:

- Quant behavior is DATA (`data/quants.json`), not `if quant == "Q6_K"` branches.
  A quant absent from the registry is `unsupported_quant` (search-space-incomplete),
  never silently omitted.
- Target capability is DATA (`data/targets.json`) with a `backend_status`. Only a
  `complete` backend may promote; `descriptor_only` targets defer.
- Candidates are route-family templates, arch-scoped and origin-tagged. A
  dense-only template is not offered to a MoE profile and vice versa.
- Roles use the centralized taxonomy (fine `RoleClass`, derived `RoleGroup`);
  MoE expert stacks and routers are first-class, not dense FFN.
- An `unknown_transformer` profile fails closed: no authorized search space, all
  routes blocked. Classify the architecture before authorizing routes.
- The decision path (search emission, families, seed policy, evaluator) must not
  hardcode model/quant/target names; `policy/guards.py` source-scans it.

## Commit Discipline

Every commit uses exactly one owning subsystem prefix. BoltBeam allowed
prefixes are defined in [commit-discipline.md](commit-discipline.md) and
enforced by `.githooks/commit-msg`.

Non-functional commits use:

```text
[repo] NFC - extract hook checker
```

Do not mix NFC refactors with behavior changes.

## Practical Test

Before merging, ask:

1. Where is the single source of truth?
2. What module boundary owns execution?
3. Did the abstraction simplify the interface?
4. Did this preserve orthogonality?
5. Did this create a duplicate rule?
6. Is there one place to update if the integration changes?
7. Is the commit message prefixed with the owning subsystem?
8. If this is NFC, is it behavior-preserving?
9. Are invariants encoded instead of implied?
10. Are dangerous operations contained?
11. Does the error shape match what callers need?
12. Is the behavior tested at the boundary where it can fail?
13. Has each piece of evidence been classified by what it can and cannot prove?
14. If a promotion cites practical roofline, does it report `P_worst`, `G_worst`,
    `A_worst`, the ceiling basis, and whether the result is promote vs closeout?
