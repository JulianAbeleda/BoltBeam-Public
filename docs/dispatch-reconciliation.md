# Dispatch reconciliation

`boltbeam.reconciliation.reconcile_dispatches` consumes semantic HCQGraph dispatch
records and returns counts plus device-duration totals by category. Records with
missing labels or invalid durations are retained under `unknown`; they are never
silently assigned to a bucket.

Device timeline is the sum of dispatch durations and may double-count overlapping
work. Host wall time is end-to-end elapsed time, including overlap and host work.
The report explicitly marks these domains as non-additive; a ratio is diagnostic,
not a claim that device totals equal wall time.
