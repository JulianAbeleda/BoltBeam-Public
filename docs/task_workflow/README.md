# Task workflow

Where delegated work is specified and where its results land.

```
docs/task_workflow/
  input/    the task as given to the executing agent
  output/   what the execution produced
```

## The two folders

`input/` holds the **specification** — the scope an agent is handed and works
from. It states the objective, the evidence required, the acceptance criteria,
and what is explicitly out of scope. It is written before the work starts and is
not edited to match what happened; if reality contradicts it, that contradiction
is a finding and belongs in `output/`.

`output/` holds **what the work produced** — audit reports, consolidation
scopes, findings. An output document is the durable record of an execution: what
was examined, what was found, what was decided, and what was deliberately not
done.

The split exists so the instruction and the result stay separable. A scope that
was quietly rewritten to match its own outcome cannot be audited, and an outcome
with no recorded instruction cannot be reviewed against intent.

## Naming

`<topic>-<kind>-<YYYYMMDD>.md`, matching the convention already in `docs/`:

```
organization-and-pruning-scope-20260730.md
boltbeam-tinygrad-integration-consolidation-scope-20260727.md
```

Dates are the date the document was written, not the date the work finished.

## What an input document must carry

A specification an agent can execute without asking follow-up questions:

- **Objective** — one paragraph, what done looks like
- **Current state** — measured, with the commands that produced the numbers
- **Method** — how to find what it is looking for, mechanically
- **Evidence format** — what a finding must include to be accepted
- **Acceptance criteria** — checkable, not aspirational
- **Risks and controls** — what could go wrong and what catches it
- **Sequence** — ordered, each step independently revertable
- **Out of scope** — stated explicitly, so silence is not read as permission

The evidence-format requirement is load-bearing. An agent asked to audit will
otherwise produce plausible assertions, and a plausible assertion about code
that has not been read is worse than no finding at all — it gets acted on.

## What an output document must carry

- what was actually examined, and the commands used
- findings, each with its evidence
- anything in the input that the evidence **contradicted**
- what was left undone, and why

The third item matters most. An execution that silently drops an unachievable
part of its scope leaves the next reader believing it was done.

## Rules

1. Inputs are immutable once execution starts. Corrections are new documents
   that supersede, with the supersession stated in both.
2. Findings and fixes are separate commits. An audit phase produces a report;
   remediation happens after review.
3. Contradicting evidence stops the phase and gets written down. The scope is a
   hypothesis, not an instruction to make reality match it.
4. Documentation-only changes here use `[docs]` unless the document is part of a
   subsystem contract, in which case use that subsystem
   (`coding-principles.md`, "Commit Discipline").

## Current contents

| Folder | Document | State |
| --- | --- | --- |
| `input/` | `organization-and-pruning-scope-20260730.md` | ready to execute |
| `output/` | `boltbeam-tinygrad-integration-consolidation-scope-20260727.md` | complete |
| `input/` | `leanness-consolidation-scope-20260830.md` | ready to execute |
| `input/` | `leanness-execution-plan-20260830.md` | packet map, tracked in `leanness-task-state-20260830.json` |
| `output/` | `leanness-close-report-20260830.md` | complete — leanness scope closed |
