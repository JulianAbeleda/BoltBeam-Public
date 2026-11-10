package ui

// The pages after setup: emit, results, saved runs and comparisons.

import (
	"cmp"
	"fmt"
	"github.com/charmbracelet/x/ansi"
	"strings"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// emitAction is the Setup row under Run. Emit turns the run on screen into a gameplan (what kernel to emit per role,
// worst first); it needs a finished run with a per-role table, so before one the row is muted with "Run first".
func emitAction(f Facts) action {
	switch {
	case f.Emitting:
		return action{stMuted.Render("[ Emit ] writing the gameplan…"), "", ""}
	case f.failedStage() != "":
		return action{stMuted.Render("[ Emit ] run failed; see Run for details"), "", ""}
	case f.canEmit():
		label := "[ Emit ]"
		if f.Run.Results.Loss.Replay != nil && f.Run.Results.Loss.Replay.Forced {
			label = "[ Emit replay report ]"
		} else if len(f.Run.Results.Loss.Roles) == 0 && f.hasOptimizationTables() {
			label = "[ Emit gap report ]"
		}
		if f.ReadOnly {
			label += " saved run " + f.Run.ID
		}
		return action{stAccent.Render(label), "emit", ""}
	}
	if f.Run != nil && f.outputDone() {
		return action{stMuted.Render("[ Emit ] no measured work breakdown; see report for " + f.Run.ID), "", ""}
	}
	return action{stMuted.Render("[ Emit ] Run first"), "", ""}
}

// canEmit requires a completed run with kernel-specific evidence, a measured breakdown, or a forced replay report.
func (f Facts) canEmit() bool {
	return f.Run != nil && f.outputDone() && !f.alive() && f.failedStage() == "" && (len(f.Run.Results.Loss.Roles) > 0 || f.hasOptimizationTables() || (f.Run.Results.Loss.Replay != nil && f.Run.Results.Loss.Replay.Forced))
}

func (f Facts) hasOptimizationTables() bool {
	if f.Run == nil {
		return false
	}
	loss := f.Run.Results.Loss
	if loss.WhereTokenGoes != nil && len(loss.WhereTokenGoes.Rows) > 0 {
		return true
	}
	if loss.Replay != nil {
		for _, table := range loss.Replay.Wheres {
			if (table.Status == "measured" || table.Status == "benchmark_only") && len(table.Rows) > 0 {
				return true
			}
		}
	}
	return false
}

// --- Gameplan (Emit) --------------------------------------------------------------------------------------------

func emitLine(f Facts) (string, string) {
	if f.Plan == nil {
		return "open", "no gameplan yet · Emit on Setup writes one from the run on screen"
	}
	if f.Plan.Plan.Kind == "gap_report" {
		return "pass", "gap report written · gameplan.md and gameplan.json"
	}
	if f.Plan.Plan.Kind == "replay_report" {
		return "pass", "checked replay report emitted; timing qualification remains disclosed"
	}
	return "pass", fmt.Sprintf("%d roles, worst first · gameplan.md and gameplan.json written to the run", len(f.Plan.Plan.Roles))
}

func timingScopeLine(scope *string, width int) string {
	if scope == nil || *scope == "" {
		return ""
	}
	return stMuted.Render(ansi.Wordwrap(*scope, width, " ")) + "\n\n"
}

// emitBody is the gameplan as Python wrote it: the same markdown, drawn in the screen's styles (renderMarkdown).
func emitBody(f Facts, width int) string {
	if f.Plan == nil {
		return stMuted.Render("No gameplan yet. Emit on Setup writes a gap report from the measured work, or kernel implementation instructions when supported.")
	}
	if f.Plan.Plan.Kind == "gap_report" && len(f.Plan.Plan.Tables) > 0 && f.Plan.Plan.Presentation.Summary != "" {
		var b strings.Builder
		b.WriteString(renderMarkdown(f.Plan.Plan.Presentation.Summary) + "\n")
		for _, table := range f.Plan.Plan.Tables {
			b.WriteString(whereBody(&table))
		}
		b.WriteString(renderMarkdown(strings.TrimRight(f.Plan.Plan.Presentation.Details, "\n")))
		return timingScopeLine(f.Plan.Plan.TimingScope, width) + b.String()
	}
	md := strings.TrimRight(f.Plan.Markdown, "\n")
	if scope := f.Plan.Plan.TimingScope; scope != nil {
		md = strings.TrimPrefix(md, *scope+"\n\n")
	}
	return timingScopeLine(f.Plan.Plan.TimingScope, width) + renderMarkdown(md)
}

func emitActions(f Facts) []action {
	out := []action{}
	if f.Plan != nil {
		out = append(out, action{"Open the gameplan (gameplan.md)", "gameplan", ""})
	}
	return append(out, action{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)})
}

// cleanAction offers to delete the unsaved runs from earlier sessions; enter asks twice.
func cleanAction(f Facts) []action {
	if f.OldWork == 0 {
		return nil
	}
	label := fmt.Sprintf("Delete the unsaved runs from earlier sessions (%d)", f.OldWork)
	if f.Confirm == "clean" {
		label = stBad.Render("Press enter again to delete them · any other key keeps them")
	}
	return []action{{label, "clean", ""}}
}

// savedAs is where a run was saved: from this session's save, or from the saved-runs list.
func (f Facts) savedAs(id string) (string, bool) {
	if f.SavedID == id && f.SavedDir != "" {
		return f.SavedDir, true
	}
	if f.Saved != nil {
		for _, r := range f.Saved.Runs {
			if r.ID == id {
				return r.Dir, true
			}
		}
	}
	return "", false
}

func resultsBody(f Facts, width int) string {
	if f.Run == nil {
		return stMuted.Render("No run yet. Run makes one.")
	}
	if f.alive() {
		return runningBody(f, width)
	}
	if f.failedStage() != "" {
		return failedBody(f, width)
	}
	var b strings.Builder
	b.WriteString(timingScopeLine(f.Run.Results.Loss.TimingScope, width))
	if f.Run.Measure != nil && f.Run.Measure.Status != "measured" {
		b.WriteString(measureBody(f, width) + "\n\n")
	}
	if f.Run.Results.Loss.Layout != nil {
		b.WriteString(layoutBody(*f.Run.Results.Loss.Layout) + "\n")
	}
	b.WriteString(resultBody(f, width))
	return b.String()
}

func runActions(f Facts) []action {
	out := []action{}
	switch {
	case f.alive():
		return []action{{"Stop the run", "stop", ""}, {"Back to setup", "page", fmt.Sprint(pageSetup)},
			{stMuted.Render("The run keeps going after you leave this screen."), "note", ""}}
	case f.failedStage() != "":
		return []action{{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)}}
	case f.ReadOnly:
		out = append(out, action{"Back to saved runs", "page", fmt.Sprint(pageSaved)},
			action{"Compare with another saved run (this one is before)", "comparefrom", f.Run.ID})
	case f.Run != nil && f.outputDone():
		if dir, ok := f.savedAs(f.Run.ID); ok {
			out = append(out, action{mark("pass") + " Saved to " + dir, "", ""})
		} else {
			out = append(out, action{stAccent.Render("[ Save run ]"), "save", ""})
		}
	}
	if f.Run != nil && f.Run.Report != nil && !f.alive() {
		out = append(out, action{"Open the full report (" + *f.Run.Report + ")", "report", ""})
	}
	return append(out, action{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)})
}

// --- Saved runs -------------------------------------------------------------------------------------------------

func savedLine(f Facts) (string, string) {
	if f.Saved == nil {
		return "run", "reading the saved runs…"
	}
	noun := "saved runs"
	if len(f.Saved.Runs) == 1 {
		noun = "saved run"
	}
	return "pass", fmt.Sprintf("%d %s in %s", len(f.Saved.Runs), noun, f.Saved.Root)
}

func savedBody(f Facts, width int) string {
	var b strings.Builder
	b.WriteString("A run is temporary until you save it. Saving keeps the run, its report, results.json and summary.txt.\n")
	b.WriteString("Enter opens a saved run. Press d twice to delete one.\n")
	return b.String()
}

// --- Compare: two saved runs, as the seam words them -------------------------------------------------------------

func compareLine(f Facts) (string, string) {
	switch {
	case f.Compared == nil:
		return "run", "comparing…"
	case f.Compared.Err != "":
		return "fail", "not compared"
	}
	return "pass", f.Compared.Before + " (before) and " + f.Compared.After + " (after)"
}

func compareRunsBody(f Facts, width int) string {
	c := f.Compared
	switch {
	case c == nil:
		return stMuted.Render("Comparing…")
	case c.Err != "":
		return stWarn.Render(glyphWarn+" Not compared: "+c.Err+".") + "\n" +
			stMuted.Render("Compare two runs of the same model, chip, engine and workload; two replays of one trace may differ in engine.")
	}
	return strings.Join(c.Lines, "\n")
}

func compareActions(f Facts) []action {
	return []action{{"Back to saved runs", "page", fmt.Sprint(pageSaved)}, {"[ Back to setup ]", "page", fmt.Sprint(pageSetup)}}
}

// crossEngines says the two saved runs are replays from different engines: replay-cross, not diff.
func (f Facts) crossEngines(a, b string) bool {
	if f.Saved == nil {
		return false
	}
	engine := map[string]string{}
	for _, r := range f.Saved.Runs {
		if r.Workload != nil && *r.Workload == "replay" {
			engine[r.ID] = r.Provider
		}
	}
	ea, oka := engine[a]
	eb, okb := engine[b]
	return oka && okb && ea != eb
}

func savedActions(f Facts) []action {
	out := []action{}
	if f.CompareFrom != "" {
		out = append(out, action{stInfo.Render("Pick the after run to compare with " + f.CompareFrom + ". The same model, chip and " +
			"engine compare row by row; two replays of one trace may differ in engine."), "note", ""})
	}
	if f.Saved != nil {
		for _, r := range f.Saved.Runs {
			pct := "not measured"
			switch {
			case r.ValidationFailed:
				pct = "FAILED validation"
			case r.PctOfLimit != nil:
				pct = fmt.Sprintf("%.0f%% of limit", *r.PctOfLimit)
			case r.RooflinePct != nil: // a replay: the session against its roofline
				pct = fmt.Sprintf("%.0f%% of roofline", *r.RooflinePct)
			case r.Measured:
				pct = "measured"
			}
			when := strings.Replace(r.SavedAt, "T", " ", 1)
			if len(when) > 16 {
				when = when[:16] // to the minute
			}
			engine := r.Provider
			if r.Layout != nil {
				engine += " · " + *r.Layout
			}
			if r.Workload != nil {
				engine += " " + *r.Workload
			}
			label := fmt.Sprintf("%s · %s · %s · %s · %s", deref(r.ModelID), deref(r.TargetID), engine, when, pct)
			if r.Fetched != nil {
				label += " · " + fetchedWords(r.Fetched)
			}
			if f.Confirm == r.ID {
				label = stBad.Render("Press d again to delete saved run " + r.ID + " · any other key keeps it")
			}
			if f.CompareFrom != "" {
				if r.ID != f.CompareFrom {
					out = append(out, action{label, "compare", r.ID})
				}
				continue
			}
			out = append(out, action{label, "opensaved", r.ID})
		}
	}
	if f.CompareFrom != "" {
		return append(out, action{"[ Cancel compare ]", "comparecancel", ""})
	}
	out = append(out, cleanAction(f)...)
	return append(out, action{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)})
}

func (f Facts) comparing() bool { return f.CJob != nil && f.CJob.Alive }

// compareID is the job id of a run's kernel comparison.
func compareID(run string) string { return run + "-compare" }

func (f Facts) outputDone() bool {
	if f.Run == nil {
		return false
	}
	for _, st := range f.Run.Stages {
		if st.Key == "output" {
			return st.Done
		}
	}
	return false
}

// failedStage is the stage the live log says failed, or "".
func (f Facts) failedStage() string {
	if f.alive() {
		return ""
	}
	return seam.ReadProgress(f.Tail).Failed
}

func aboutRun(f Facts) string {
	if f.Run != nil && f.Run.ID != "" {
		return f.Run.ID
	}
	if f.Job != nil {
		return f.Job.ID
	}
	if f.Run != nil {
		return f.Run.ID
	}
	return ""
}

// aboutRunFrom is the Run page's header: the run, and where it came from when `boltbeam fetch` brought it home.
func aboutRunFrom(f Facts) string {
	about := aboutRun(f)
	if f.Run != nil && f.Run.ID == about && f.Run.Fetched != nil {
		about += " · " + fetchedWords(f.Run.Fetched)
	}
	return about
}

// fetchedWords is "from HOST, fetched DATE": the host it was fetched from, else its chip.
func fetchedWords(x *seam.Fetched) string {
	from := "another machine"
	if x.Host != nil && *x.Host != "" {
		from = *x.Host
	} else if x.Chip != nil && *x.Chip != "" {
		from = *x.Chip
	}
	return "from " + from + ", fetched " + cmp.Or(x.FetchedAt[:min(len(x.FetchedAt), 10)], "at an unknown time")
}

// --- 1 Model -----------------------------------------------------------------------------------------------

// shortRun is the run's number, the part after the model and chip.
func shortRun(id string) string {
	if i := strings.LastIndex(id, "-"); i >= 0 {
		return id[i+1:]
	}
	return id
}

// --- 5 Result ----------------------------------------------------------------------------------------------

func resultLine(f Facts) (string, string) {
	if f.alive() || f.failedStage() != "" {
		return "", "" // the box above holds the bar, or what failed
	}
	if !f.outputDone() {
		return "open", "not run yet"
	}
	if f.comparing() {
		p := seam.ReadCompare(f.CTail)
		now := ""
		if p.Now != "" {
			now = " · " + p.Now + " " + compareDoing[p.Doing] + "…"
		}
		if p.Total == 0 {
			return "run", "working" + now
		}
		return "run", fmt.Sprintf("comparing kernels %s %d of %d%s", bar(float64(len(p.Done))/float64(p.Total), 14), len(p.Done), p.Total, now)
	}
	if p := seam.ReadCompare(f.CTail); p.Failed != "" {
		return "fail", "comparing kernels failed: " + p.Failed
	}
	res := f.Run.Results
	if s := res.Loss.Session; res.Measured && s != nil && s.Focus != nil && len(res.Loss.Runtimes) > 0 && res.Ceiling.TokS != nil {
		r := res.Loss.Runtimes[0]
		return "pass", fmt.Sprintf("%.1f prompt tokens per second at depth %s · %.0f%% of the limit", r.TokS, count(s.Focus), r.TokS / *res.Ceiling.TokS * 100)
	}
	if res.Measured && res.Timing.TokS != nil {
		if res.Ceiling.TokS != nil && *res.Ceiling.TokS > 0 {
			return "pass", fmt.Sprintf("%.1f tokens per second · %.0f%% of the limit", *res.Timing.TokS, *res.Timing.TokS / *res.Ceiling.TokS * 100)
		}
		return "pass", fmt.Sprintf("%.1f tokens per second measured", *res.Timing.TokS)
	}
	if rp := res.Loss.Replay; res.Measured && rp != nil && len(rp.Rows) > 0 {
		return "pass", fmt.Sprintf("replay measured · %d turns", len(rp.Rows))
	}
	if f.Run.Report != nil {
		return "open", "nothing measured yet · enter opens " + *f.Run.Report
	}
	return "open", "nothing measured yet"
}
