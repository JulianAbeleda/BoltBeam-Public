# GUI Explorations

Design mockups for a BoltBeam interface. Open [run-graph.html](run-graph.html) directly in a browser — it is
self-contained, makes no network requests, and needs no build step.

This is a mockup, not an implementation: nothing here reads a real run directory. The shipped counterpart is
`report.html`, emitted by `boltbeam output` from actual staged artifacts (see
[`boltbeam/report/html.py`](../boltbeam/report/html.py)).

## [run-graph.html](run-graph.html) — the run as a node graph

A BoltBeam run already *is* a DAG — `load → autoscan → analyze → probe / timing → ledger → output`, with typed
artifacts flowing between stages. So the node editor is not a metaphor laid over the data; it is the data
model drawn directly. Borrowed vocabulary: ComfyUI / litegraph — dot-grid canvas, draggable nodes, typed ports,
bezier links.

Drag nodes, drag the canvas to pan, click a node for its artifacts.

**Two readings of the same graph.** A Plain / Technical switch, because "what is this and is it going well" and
"what did the pipeline do" are different questions from different people:

| | Technical | Plain |
|---|---|---|
| node | `Probe Runner` · `ingest-probe` | **Test the building blocks** ③ |
| port | `probe_evidence` | *test results* |
| value | `dominant: gemv_codegen_capped` | *the slow part is how it reads memory* |
| status | `blocked` | *Can't finish — 2 measurements missing* |

Plain adds a briefing bar (*what this is / how it's going / what's needed*), numbers the stages — the pipeline
genuinely is a sequence, so the numbering carries information — and inverts the inspector to lead with **what
this step does → why it matters → result** instead of a file list.

Plain hides no nodes and simplifies no links. Both audiences look at the same picture; only the description
changes. Hiding structure would make the page easier to look at and harder to understand.

## The load-bearing detail

The link from **Probe Runner** to **Output** is drawn as a dashed red line into an empty port, because that
evidence was never produced. The graph shows you *why* Output is blocked rather than just labelling it.

That is the same rule `report.html` follows — absence is rendered, not omitted — expressed visually instead of
as a section heading. A run missing evidence should look missing.

## Relationship to `report.html`

Same palette (Tokyo Night), same accent semantics, so a colour learned in one reads the same in the other:

| artifact type | colour |
|---|---|
| `model_profile` | purple `#bb9af7` |
| `capabilities` | yellow `#e0af68` |
| `measurement_plan` | blue `#7aa2f7` |
| `probe_evidence` | green `#9ece6a` |
| `hw_trace` | orange `#ff9e64` |
| `route_policy` | pink `#f7768e` |
| missing | red `#db4b4b` |

Missing evidence uses the scheme's error red rather than the policy pink, so a broken link reads as a failure
and not merely as another type.

The two are meant as views of one run: the graph is the interactive reading, `report.html` the portable one you
can attach to a handoff.
