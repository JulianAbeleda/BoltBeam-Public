package seam

// Where the time went: a prefill's time budget (collectors/time_budget.py, boltbeam.time_budget.v1) as report/plain.py
// lays it out, in `screen plain` (layer 2) and `screen head-to-head` (both runs side by side). The rows add up to the
// wall as the record states it; Warn says when they do not (the trace's GPU work is longer than the wall).

// TimeBudget is one run's budget: Line is the one sentence, Rows the parts (Depth 0) with the rows under each, Lines the
// sentences under the table, Warn the record's own caveat (nil when it has none), Refused why there is no budget.
type TimeBudget struct {
	Title   string          `json:"title"`
	Of      *string         `json:"of"`
	Line    string          `json:"line"`
	Rows    []TimeBudgetRow `json:"rows"`
	Lines   []string        `json:"lines"`
	Warn    *string         `json:"warn"`
	Unit    string          `json:"unit"`
	Refused *string         `json:"refused"`
}

// TimeBudgetRow is a part (Part names it: kernels, exposed, idle, host, unattributed, wall) or a row under one (Depth 1).
// Ms is the figure as shown, Value the number.
type TimeBudgetRow struct {
	Label string   `json:"label"`
	Ms    string   `json:"ms"`
	Value *float64 `json:"value"`
	Depth int      `json:"depth"`
	Part  *string  `json:"part"`
	Key   string   `json:"key"`
}

// TimeBudgetSides is two runs' budgets side by side: a row per part and per row under it in either run, a cell per run
// ("-" where that run has none); Notes are each run's caveat or that it has no budget, "A: " or "B: " first.
type TimeBudgetSides struct {
	Title   string   `json:"title"`
	Unit    string   `json:"unit"`
	Columns []string `json:"columns"`
	Rows    []struct {
		Label string   `json:"label"`
		Depth int      `json:"depth"`
		Part  *string  `json:"part"`
		Cells []string `json:"cells"`
	} `json:"rows"`
	Notes []string `json:"notes"`
}
