# The engine said its graph failed; no screen said so

Symptom (cold review, 2026-10-10). A tinygrad run on the M3 wrote `timing_trace.json` rows with
`source: "tinygrad decode on METAL, JIT with no graphs (graph replay failed), no profiling"`. No screen, report or
summary showed that sentence. The Launch-heavy lever then said "run the token as one graph": the action the engine
had just failed at.

Cause. `collectors/providers.py measure_tinygrad` kept the engine's account in the trace row's `source` and nowhere
else. The seam (`workflow/screen.py`) read the row's `tok_s` only.

Fix. The row carries `jit` and `graph_error` (the last GraphException line the engine printed);
`screen.step_facts` puts the measured token's source and graph state in `results.loss.step`; `evidence.items` turns a
recorded graph failure into the `graph_failed` finding, or rewrites the launch-heavy item ("graph replay failed on
this run (...), so the token ran as N launches") with the lever "Fix the graph path first" and a pointer at the trace
row; the Run screen, report Facts and summary.txt print the sentence. Test: `tests/test_one_measured_token.py
test_graph_failure_is_on_screen_and_in_the_lever`.

Rule. A fact the engine states about its own run is shown before any lever is derived from the numbers that fact
explains.
