package ui

// The pages after setup: emit, results, saved runs and comparisons.

import (
	"cmp"
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// emitAction is the Setup row under Run. Emit turns the run on screen into a gameplan (what kernel to emit per role,
// worst first); it needs a finished run with a per-role table, so before one the row is muted with "Run first".
// Each state says, after the button, what it does or why it cannot (w is Setup's label column).
func emitAction(f Facts, w int) action {
	off := func(why string) action { return action{button(f, w, "[ Emit ]", why, false), "", ""} }
	switch {
	case f.Emitting:
		return off("Writing the gameplan…")
	case f.failedStage() != "":
		return off("The last run failed: see Run for why")
	case f.canEmit():
		label, does := "[ Emit ]", "Writes the kernel gameplan from the last run"
		if f.Run.Results.Loss.Replay != nil && f.Run.Results.Loss.Replay.Forced {
			label, does = "[ Emit replay report ]", "Writes the replay's report from the last run"
		} else if len(f.Run.Results.Loss.Roles) == 0 && f.hasOptimizationTables() {
			label, does = "[ Emit gap report ]", "Writes the gap report from the last run"
		}
		if f.ReadOnly {
			does = strings.Replace(does, "the last run", "saved run "+f.Run.ID, 1)
		}
		return action{button(f, w, label, does, true), "emit", ""}
	}
	if f.Run != nil && f.outputDone() {
		return off("The last run has no per-role breakdown to plan from: its report has what it measured")
	}
	return off("No run yet: Run first")
}

// button is a Setup button and, in the value column, what pressing it does (or why it is off), muted.
func button(f Facts, w int, label, help string, on bool) string {
	text := stMuted.Render(label)
	if on {
		text = stAccent.Render(label)
	}
	pad := max(w-ansi.StringWidth(label), 0)
	room := 1 << 10 // outside a frame (a test of the rows): one line
	if f.cols > 0 {
		room = max(f.cols-w-2, 20)
	}
	lines := strings.Split(ansi.Wordwrap(help, room, " "), "\n") // whole, in the value column
	for i, l := range lines {
		lines[i] = stMuted.Render(l)
	}
	return text + strings.Repeat(" ", pad+2) + strings.Join(lines, "\n"+strings.Repeat(" ", w+4))
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
			b.WriteString(whereBody(&table, width))
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
	if f.plainShown() { // layer 1, then layer 2's sections (render_plain.go)
		return plainBody(f, width)
	}
	return fullDetail(f, width) // the plain block not in (yet): the page as it always was
}

// runActions is What next. After a finished run, in this order: the report, Run again with this setup, Saved runs,
// Compare with a saved run; then Save (or where it was saved), and back to Setup.
func runActions(f Facts) []action {
	switch {
	case f.alive():
		return []action{{"Stop the run", "stop", ""}, {"Back to setup", "page", fmt.Sprint(pageSetup)},
			{stMuted.Render("The run keeps going after you leave this screen."), "note", ""}}
	case f.failedStage() != "":
		return []action{{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)}}
	}
	out := []action{}
	if f.Run != nil && f.Run.Report != nil {
		out = append(out, action{"Open the report (" + *f.Run.Report + ")", "report", ""})
	}
	if f.Run == nil || !(f.outputDone() || f.ReadOnly) {
		return append(out, action{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)})
	}
	dir, saved := f.savedAs(f.Run.ID)
	n := 0
	if f.Saved != nil {
		n = len(f.Saved.Runs)
	}
	out = append(out, action{"Run again with this setup", "rerun", ""}, action{fmt.Sprintf("Saved runs (%d)\t›", n), "page", fmt.Sprint(pageSaved)})
	switch { // Compare opens Saved runs' picker with this run already chosen
	case f.ReadOnly || saved:
		out = append(out, action{"Compare with a saved run", "comparefrom", f.Run.ID})
	default: // only saved runs compare: Compare saves this one first
		out = append(out, action{"Compare with a saved run (saves this one first)", "savecompare", ""})
	}
	switch {
	case f.ReadOnly:
	case saved:
		out = append(out, action{mark("pass") + " Saved to " + dir, "", ""})
	default:
		out = append(out, action{stAccent.Render("[ Save run ]"), "save", ""})
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
	b.WriteString("A run is temporary until you save it.\nSaving keeps its report and its measurements.\n\n")
	b.WriteString("Enter opens a run. c compares two side by side.\nd twice deletes one.\n")
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
	return "pass", f.Compared.Before + " and " + f.Compared.After + ", head to head"
}

// compareRunsBody is two runs head to head: which is A and which B, what differs, their headline figures side by side
// with the better one marked; then their parts, or the one line that says why the parts are not compared.
func compareRunsBody(f Facts, width int) string {
	c := f.Compared
	switch {
	case c == nil:
		return stMuted.Render("Comparing…")
	case c.Err != "":
		return stWarn.Render(glyphWarn + " Not compared: " + c.Err + ".")
	}
	var b strings.Builder
	if h := c.Head; h != nil {
		b.WriteString(headToHead(*h, width))
		if h.Refused != nil {
			return strings.TrimRight(b.String(), "\n")
		}
		if c.PartsNote != nil {
			return b.String() + "\n" + paint(stMuted, hang(*c.PartsNote, 0, width))
		}
		b.WriteString("\n")
	}
	if c.Plain == nil {
		return b.String() + strings.Join(c.Lines, "\n")
	}
	b.WriteString(partsBody(c, width))
	return b.String()
}

// headToHead is the two runs' figures in three columns: the row's label, run A's figure, run B's; the better one in
// the OK colour with a mark after it. A run's title names it once, above.
func headToHead(h seam.HeadToHead, width int) string {
	var b strings.Builder
	for i, r := range h.Runs {
		b.WriteString(stHeader.Render(string(rune('A'+i))) + "  " + strings.TrimLeft(hang(r.ID, 3, width), " ") + "\n")
		b.WriteString(paint(stMuted, hang(r.Title, 3, width)) + "\n")
	}
	if h.Refused != nil {
		return b.String() + "\n" + paint(stWarn, hang(glyphWarn+" "+*h.Refused, 0, width)) + "\n"
	}
	if h.Differs != nil {
		b.WriteString("\n" + hang(*h.Differs, 0, width) + "\n")
	}
	mark := " " + glyphPass
	lw, cw := 0, 0
	for _, r := range h.Rows {
		lw = max(lw, lipgloss.Width(r.Label))
		for _, c := range r.Cells {
			cw = max(cw, lipgloss.Width(c)+lipgloss.Width(mark))
		}
	}
	lw = min(lw, max(width-2*(cw+2)-2, 12))
	b.WriteString(stHeader.Render("\n  "+strings.Repeat(" ", lw)+strings.TrimRight(fmt.Sprintf("  %-*s  %-*s", cw, "A", cw, "B"), " ")) + "\n")
	for _, r := range h.Rows {
		label := truncate(r.Label, lw)
		line := "  " + stMuted.Render(label) + strings.Repeat(" ", lw-lipgloss.Width(label))
		for i, cell := range r.Cells {
			text := cell
			if text == "" {
				text = "–"
			}
			if r.Better != nil && *r.Better == i {
				line += "  " + stOK.Bold(true).Render(text+mark) + strings.Repeat(" ", cw-lipgloss.Width(text+mark))
				continue
			}
			line += "  " + stFigure.Render(text) + strings.Repeat(" ", cw-lipgloss.Width(text))
		}
		b.WriteString(strings.TrimRight(line, " ") + "\n")
	}
	return b.String()
}

// partsBody is the parts of two runs of one model compared (report/plain.compare): what changed, the parts that moved
// most, the trust line; then the seam's lines in full, under Full detail as the Results screen has it.
func partsBody(c *Compared, width int) string {
	l := c.Plain.Layer1
	var b strings.Builder
	b.WriteString(paint(stHeader, hang(l.Fast, 0, width)) + "\n")
	for _, line := range l.Lines {
		b.WriteString(hang(line, 0, width) + "\n")
	}
	for _, p := range l.Top {
		b.WriteString(paint(stAccent, hang(p.Text, 2, width)) + "\n")
	}
	if tr := l.Trust; tr != nil {
		style := stWarn
		if tr.Checked {
			style = stOK
		}
		b.WriteString(paint(style, hang(tr.Text, 0, width)) + "\n")
	}
	return b.String() + "\n" + stHeader.Render(titleFull) + "\n" + strings.Join(c.Lines, "\n")
}

func compareActions(f Facts) []action {
	return []action{{"Back to saved runs", "page", fmt.Sprint(pageSaved)}, {"[ Back to setup ]", "page", fmt.Sprint(pageSetup)}}
}

// savedActions is Saved runs: each run opens; Compare two saved runs picks one, then the one to set beside it.
func savedActions(f Facts) []action {
	out := []action{}
	picking := f.ComparePick || f.CompareFrom != ""
	switch {
	case f.CompareFrom != "":
		out = append(out, action{stInfo.Render("Pick the run to set beside " + f.CompareFrom + ". Any two measured runs compare " +
			"head to head; two runs of one model compare part by part too."), "note", ""})
	case f.ComparePick:
		out = append(out, action{stInfo.Render("Pick the first of the two runs to compare."), "note", ""})
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
			switch {
			case f.CompareFrom != "":
				if r.ID != f.CompareFrom {
					out = append(out, action{label, "compare", r.ID})
				}
			case f.ComparePick:
				out = append(out, action{label, "comparefrom", r.ID})
			default:
				out = append(out, action{label, "opensaved", r.ID})
			}
		}
	}
	if picking {
		return append(out, action{"[ Cancel compare ]", "comparecancel", ""})
	}
	out = append(out, head(""))
	if f.Saved != nil && len(f.Saved.Runs) > 1 {
		out = append(out, action{"Compare two saved runs\t›", "comparepick", ""})
	}
	out = append(out, action{"Fetch runs from another machine\t›", "page", fmt.Sprint(pageFetch)})
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
	if f.alive() || f.failedStage() != "" || f.plainShown() {
		return "", "" // the box above holds the bar, what failed, or layer 1's answers
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
