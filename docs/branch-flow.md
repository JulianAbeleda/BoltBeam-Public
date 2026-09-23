# Branch layout: `master` ⊂ `dev` ⊂ `exp`

Three branches. The relationship is **containment by content category**, not a
promotion pipeline. Work does not graduate from `exp` to `dev` to `master`;
rather, each branch admits categories of content that the one inside it refuses.

This mirrors `tinygrad-arkey`, whose README states the convention as:

> - `master`: runnable promoted product surface; no handwritten specialized
>   fallback or research archive.
> - `dev`: master plus qualification oracles, development tooling, and durable
>   handoffs.
> - `exp`: active machine-search and hardware experimentation.

The consequence worth stating plainly: **master is lean because of what it
excludes, not because of what has been approved.** Leanness is a property of the
admission rule, not of a review gate.

## What each branch admits

### `master` — the runnable product surface

What the project *is*, when someone runs it.

- The pipeline, the CLI, the schemas, the adapters, the analysis it performs.
- Nothing whose purpose is to *produce knowledge* rather than *be the product*.

Excluded by category, regardless of quality:

- **the test suite** — see below
- research archives and historical evidence
- qualification oracles and measurement scaffolding
- one-off scripts, probes, and campaign tooling
- models or modules with no live caller

### Why the tests are not on `master`

A test's purpose is to *establish that the product is correct*. That is producing
knowledge about the product, not being the product — the same category as a
qualification oracle, which `dev` has always owned. `dev` is defined as "master
plus what you need to verify it"; the suite is precisely that, so an earlier
version of this document contradicted its own definition by admitting tests to
the trunk.

The practical statement: **nothing in `boltbeam/` imports from `tests/`**, and the
full `inspect -> load -> analyze -> output` pipeline runs on a real GGUF with
`tests/` absent. The trunk is what ships; it does not carry its own proof.

This is not "tests don't matter". It relocates where correctness is answered, and
`tools/sync_branches.sh` enforces it: the trunk's first verification is the merge
into `dev`, which is hard-gated on the suite. A trunk commit that breaks the
product cannot reach `exp` without failing that gate.

A thing can be correct, tested and useful and still not belong on `master`. The
question is never "is this good" but "is this the product".

### `dev` — master plus the apparatus

Everything on `master`, plus what you need to *verify* and *develop* it.

- **the test suite** — `tests/`, and `pytest` is run from here
- qualification oracles and correctness harnesses
- development and diagnostic tooling
- durable handoffs and finished investigation records

The admission table below already said "how you check that the tool is correct ->
`dev`". The suite is the largest instance of that, and it now lives where the rule
always put it.

This is where BoltBeam's own audit tooling belongs — the meta-development side
of the boundary the orthogonality audit found is already cleanly separated in
the code.

### `exp` — active experimentation

Machine search, hardware experimentation, probes in flight.

- Broken intermediate states are fine; the suite need not pass.
- Most of what lands here should never move inward. Its value is the answer it
  produces, not the code.
- Once a probe's verdict is recorded, the code can go; the record stays.

## The admission test

Before adding a file to `master`, ask which of these it is:

| If it is… | It belongs on |
| --- | --- |
| part of what the tool does when run | `master` |
| how you check that the tool is correct | `dev` |
| how you found out what to build | `exp` |
| a record of what you learned | `docs/` on `master` |

That last row matters. A **finding** is product — it is how the project knows
what it knows. The **probe that produced it** is not. Record the verdict on
`master`, keep the apparatus on `dev`, delete the probe from `exp`.

The 13 versioned `perf/` models are the worked example: authored in one burst,
unreachable from any live path, verdict recorded nowhere. By this rule they were
never `master` material — they are `exp` apparatus whose finding was never
written down. See
`task_workflow/output/perf-model-verdict-record-20260731.md`.

## Keeping them in sync

Because the relationship is containment, **`master` flows outward** — every
change to the product surface must reach `dev` and `exp`, or they stop being
supersets and start being forks:

```
./tools/sync_branches.sh --dry-run   # report what each merge would delete
./tools/sync_branches.sh             # merge outward, suite-gated at each hop
git push origin dev exp
```

Do this on every trunk commit, or close to it. The failure mode is visible in
tinygrad-arkey today: `dev` is 88 commits behind `master`, meaning it is no
longer "master plus tooling" — it is a fork that happens to share a name. At
that distance the merge is a project rather than a habit.

### The prune case

One situation needs a human, and it is why the script reports before it merges.
When the trunk **prunes** something the outer branches retain, an ordinary merge
deletes it there too — the exact opposite of what the layout is for.

The mechanism is an **apparatus commit**: after the prune merges outward,
restore the retained paths on the outer branch and commit them. `dev` is then
"trunk plus apparatus" by construction, and the restoration sits above the prune
in history, so later merges do not re-delete it.

```
git checkout dev
git merge main                              # brings the prune; deletes the files
git checkout <pre-prune-sha> -- <paths>     # put back what dev retains
git commit -m "[repo] retain <what> as dev apparatus"
```

**Restore the constraining tests too, not just the code.** The first apparatus
commit here restored six models but not the test pinning the allowed set, so
`dev` pinned seven files while holding thirteen and the suite failed. An
apparatus layer that restores code without the test that constrains it leaves
the branch self-inconsistent. The sync script fails loudly on exactly that and
says to fix it on the outer branch, never on the trunk.

Movement inward is not a merge. Promoting something from `exp` to `master`
means deciding it was product all along, and it lands as an ordinary commit on
`master` that then flows outward like any other.

## What this is not

Not a review process, an approval gate, or a release train. There is one person
here. It is a rule about **what kind of thing** each branch holds, so that the
trunk stays the product and the apparatus has somewhere legitimate to live.

If a change is obviously part of the product — commit it to `master` and merge
outward. The layout exists so that the *other* categories have a home, not to
slow down the ones that don't.
