// Package ui owns the screen. Every view here is a pure function of seam facts and a size, so a test can
// render a screen to a string and pin it; model.go wires the same functions to Bubble Tea.
package ui

import (
	"fmt"
	"github.com/charmbracelet/bubbles/viewport"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// BoltBeam's own plain vocabulary (gui/README.md and gui/run-graph.html). Main lines use the plain word only;
// detail views show the plain word with the record's own name beside it, muted.
var plainStage = map[string]string{
	"load": "Read the model", "autoscan": "Check the machine", "analyze": "Plan what to try",
	"runner_plan": "Prepare the handoff", "ingest_probe": "Test the building blocks", "ingest_timing": "Time the real run",
	"output": "Package the result",
}

var plainStatus = map[string]string{
	"not_analyzed": "not planned yet", "needs_measurement": "needs measuring", "policy_seeded": "plan ready",
}

var plainRoute = map[string]string{
	"promoted": "kept", "refuted": "ruled out", "blocked": "undecided", "unmeasured": "default kernel, none compared yet", "candidate": "to try",
}

var routeMark = map[string]string{"promoted": "kept", "refuted": "fail", "blocked": "crossed", "unmeasured": "wait"}

var plainBucket = map[string]string{
	"at_peak": "at the speed limit", "gemv_codegen_capped": "reads memory slower than it could",
	"latency_bound": "waiting on memory", "elementwise_dilution": "small kernel, dead time",
	"activation_bound": "real activation work", "timing_inconclusive": "not clear yet",
}

var plainRegime = map[string]string{
	"streaming_bound": "reads memory at full speed", "occupancy_starved": "not enough work in flight",
	"dequant_bound": "slowed by unpacking the numbers", "metadata_bound": "slowed by the scale tables",
	"latency_bound": "waiting on memory", "compute_bound": "limited by the sums", "inconclusive": "not clear yet",
}

var plainRole = map[string]string{
	"attn_kv": "attention keys and values", "attn_qo": "attention query and output", "ffn_gate_up": "feed-forward in",
	"ffn_down": "feed-forward out", "lm_head": "vocabulary output", "embed": "token embedding",
}

var plainLimit = map[string]string{"memory": "reading weights", "compute": "doing sums"}

// plainMeasureStage words the pipeline's measuring stages. They are not report stages (report/html.py STAGES), so
// they live apart from plainStage, which tests/test_report_html.py pins to the report's own table.
var plainMeasureStage = map[string]string{
	"measure": "Can this Mac measure", "measure_probe": "Test blocks on the GPU",
	"measure_timing": "Time the real decode",
}

// stageWord is the plain word for any pipeline stage key.
func stageWord(key string) string {
	if plain, ok := plainMeasureStage[key]; ok {
		return plain
	}
	return word(plainStage, key)
}

var plainNeed = map[string]string{"probe_evidence": "the building-block tests", "timing_trace": "a timing trace"}

// word is the plain word for a record key, or the key itself when no plain word exists.
func word(table map[string]string, key string) string {
	if plain, ok := table[key]; ok {
		return plain
	}
	return key
}

// named is the plain word with the record's own name beside it, muted: the detail views' wording.
func named(table map[string]string, key string) string {
	if plain, ok := table[key]; ok {
		return plain + "  " + stMuted.Render(key)
	}
	return key
}

func num(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.4g", *v)
}

func gb(bytes float64) string { return fmt.Sprintf("%.2f GB", bytes/1e9) }

// us prints microseconds as a whole number; kernels are never sub-microsecond on a GPU timeline.
func us(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.0f", *v)
}

func count(v *int) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%d", *v)
}

func deref(s *string) string {
	if s == nil {
		return "-"
	}
	return *s
}

// table lays out rows two spaces apart; the first row is the header.
func table(rows [][]string) string {
	widths := []int{}
	for _, row := range rows {
		for i, cell := range row {
			if i >= len(widths) {
				widths = append(widths, 0)
			}
			widths[i] = max(widths[i], lipgloss.Width(cell))
		}
	}
	var b strings.Builder
	for r, row := range rows {
		for i, cell := range row {
			if r == 0 {
				cell = stHeader.Render(cell)
			}
			b.WriteString(cell)
			if i < len(row)-1 {
				b.WriteString(strings.Repeat(" ", widths[i]-lipgloss.Width(cell)) + "  ")
			}
		}
		b.WriteString("\n")
	}
	return b.String()
}

// runStatus is a run's plain status. A run holding both measurements reads "measured": analyze keeps saying
// needs_measurement until per-role kernel choices are compared, which is a later step than measuring the speed.
func runStatus(status string, measured bool) string {
	if measured {
		return "measured"
	}
	return word(plainStatus, status)
}

func measuredMarks(m seam.Measured) string {
	parts := []string{}
	for _, item := range []struct {
		ok   bool
		name string
	}{{m.Probe, "probe"}, {m.Timing, "timing"}} {
		if item.ok {
			parts = append(parts, item.name+" "+mark("pass"))
		} else {
			parts = append(parts, item.name+" "+mark("wait"))
		}
	}
	return strings.Join(parts, " · ")
}

func pctBar(v *float64, width int) string {
	if v == nil {
		return strings.Repeat(" ", width+5)
	}
	return bar(*v/100, width) + fmt.Sprintf(" %3.0f%%", *v)
}

// --- the five bodies -----------------------------------------------------------------------------------------

func modelBody(f Facts, width int) string {
	p := f.Profile
	switch {
	case f.ModelErr != "":
		return stBad.Render(glyphFail+" ") + f.ModelErr
	case p == nil:
		return stMuted.Render("Nothing read yet. The model is never executed; only its shape is read.")
	}
	var b strings.Builder
	facts := []string{fmt.Sprintf("%s layers", count(p.LayerCount)), fmt.Sprintf("hidden %s", count(p.HiddenSize)),
		fmt.Sprintf("feed-forward %s", count(p.FFNSize)), fmt.Sprintf("vocabulary %s", count(p.VocabSize))}
	if a := p.Metadata.Attention; a.HeadCount != nil && a.HeadCountKV != nil {
		facts = append(facts, fmt.Sprintf("%d heads, %d kv heads", *a.HeadCount, *a.HeadCountKV))
	}
	b.WriteString(strings.Join(facts, " · ") + "\n")
	if !p.Complete {
		b.WriteString(stWarn.Render(glyphWarn+" The profile is incomplete: some roles could not be read.") + "\n")
	}
	rows := [][]string{{"ROLE", "SHAPE", "QUANT", "TENSORS"}}
	for _, r := range p.Roles {
		rows = append(rows, []string{named(plainRole, r.Role), fmt.Sprintf("%dx%d", r.Rows, r.Cols), r.Quant, fmt.Sprint(r.Count)})
	}
	b.WriteString(table(rows))
	b.WriteString(stMuted.Render(fmt.Sprintf("%s · %s · %s", deref(p.Architecture), p.ArchitectureClass, p.Source)))
	return b.String()
}

func chipBody(f Facts, width int) string {
	t := f.target()
	if t == nil {
		return stMuted.Render("Reading the chip list…")
	}
	var b strings.Builder
	if t.Scope != nil {
		b.WriteString(*t.Scope + "\n")
	}
	fmt.Fprintf(&b, "%s · %s\n", gbs(t.MemoryBandwidthGBs), tflops(*t))
	b.WriteString(stMuted.Render(fmt.Sprintf("%s · %s · memory speed from %s", t.Backend, t.BackendStatus, t.FactStatus.MemoryBandwidthGBs)))
	return b.String()
}

func tflops(t seam.Target) string {
	parts := []string{}
	for k, v := range t.MatrixTFLOPS {
		parts = append(parts, fmt.Sprintf("%s %.4g TFLOP/s matrix", k, v))
	}
	if len(parts) == 0 {
		for k, v := range t.PeakTFLOPS {
			parts = append(parts, fmt.Sprintf("%s %.4g TFLOP/s", k, v))
		}
	}
	if len(parts) == 0 {
		return "compute speed unknown"
	}
	return strings.Join(parts, ", ")
}

func roleRows(roles []seam.CeilingRole) string {
	rows := [][]string{{"ROLE", "QUANT", "SHARE", "FLOOR ms"}}
	for _, r := range roles {
		rows = append(rows, []string{named(plainRole, r.Role), r.Quant, bar(r.Share, 10) + fmt.Sprintf(" %2.0f%%", r.Share*100),
			fmt.Sprintf("%.2f", r.FloorMs)})
	}
	return table(rows)
}

func limitBody(f Facts, width int) string {
	c := f.Ceiling
	switch {
	case f.CeilBusy:
		return stMuted.Render("Working out the speed limit…")
	case c == nil && f.CeilErr != "":
		return stWarn.Render(glyphWarn+" No speed limit. ") + f.CeilErr
	case c == nil:
		return stMuted.Render("Read a model in step 1 and pick a chip in step 2 first.")
	}
	d, p := c.Decode, c.Prefill
	var b strings.Builder
	if d.TokS != nil {
		fmt.Fprintf(&b, "%s %s\n", stAccent.Render(glyphBolt), stHeader.Render(fmt.Sprintf("%.1f tokens per second, at best.", *d.TokS)))
	}
	fmt.Fprintf(&b, "Each token reads %s of weights.\n", gb(d.BytesMoved))
	fmt.Fprintf(&b, "The memory moves %.1f GB/s, so one token takes at least %.1f ms.\n", c.PeakBandwidthGBs, d.FloorMs)
	fmt.Fprintf(&b, "What limits it: %s\n", named(plainLimit, d.Regime))
	if p.TokS != nil {
		fmt.Fprintf(&b, "A prompt of %d tokens: %.0f ms at best, %.0f tokens per second.\n", p.Context, p.FloorMs, *p.TokS)
	}
	note := "this is arithmetic, not a measurement"
	if c.TruthStatus != "modeled" {
		note = c.TruthStatus
	}
	b.WriteString(stMuted.Render(fmt.Sprintf("memory speed from %s · compute from %s · %s", c.Target.FactStatus.MemoryBandwidthGBs,
		c.Target.FactStatus.PeakTFLOPS, note)) + "\n\n")
	b.WriteString(stHeader.Render("One token, per role") + "\n" + roleRows(d.Roles) + "\n")
	b.WriteString(stHeader.Render(fmt.Sprintf("A prompt of %d tokens, per role", p.Context)) + "\n" + roleRows(p.Roles) + "\n")
	b.WriteString(stHeader.Render("Assumptions") + "\n" + stMuted.Render(strings.Join(c.Assumptions, "\n")))
	return b.String()
}

// stageState: the manifest says done; the live log says running or failed; an open need says waiting.
func stageState(st seam.Stage, events map[string]string, alive bool, blocked []seam.Need) (string, string) {
	switch events[st.Key] {
	case "failed":
		return "fail", "failed; see the log below"
	case "running":
		if alive {
			return "run", "running"
		}
	}
	if st.Done {
		if len(st.Artifacts) == 1 {
			return "pass", "1 file"
		}
		return "pass", fmt.Sprintf("%d files", len(st.Artifacts))
	}
	for _, n := range blocked {
		if (st.Key == "ingest_probe" && n.Need == "probe_evidence") || (st.Key == "ingest_timing" && n.Need == "timing_trace") {
			return "wait", fmt.Sprintf("needs %s (%s)", word(plainNeed, n.Need), n.Request)
		}
	}
	return "open", "not run"
}

// measureRows are the measuring stages under "Plan what to try". measure_status.json is the record once it exists;
// the live log fills in while the job runs or when it stopped before writing one. Each row is mark, key, detail.
func measureRows(ms *seam.MeasureStatus, events map[string]string, alive bool) [][3]string {
	rows := [][3]string{}
	for _, key := range []string{"measure", "measure_probe", "measure_timing"} {
		switch {
		case ms != nil && !alive && key == "measure" && ms.Status == "skipped":
			rows = append(rows, [3]string{"crossed", key, "not here: " + deref(ms.Reason)})
		case ms != nil && !alive && key != "measure" && ms.Status == "measured":
			rows = append(rows, [3]string{"pass", key, "done"})
		case ms != nil && !alive && key == "measure_probe" && ms.Status == "failed":
			rows = append(rows, [3]string{"fail", key, deref(ms.Reason)})
		case events[key] == "running" && alive:
			rows = append(rows, [3]string{"run", key, "running"})
		case events[key] == "failed":
			rows = append(rows, [3]string{"fail", key, "failed; see the log below"})
		case events[key] == "done" && (ms == nil || alive):
			rows = append(rows, [3]string{"pass", key, "done"})
		}
	}
	return rows
}

func measureBody(f Facts, width int) string {
	r := f.Run
	var b strings.Builder
	switch {
	case (r == nil || r.Stages == nil) && f.alive():
		b.WriteString(stMuted.Render("The run folder appears when the first stage finishes.") + "\n")
	case r == nil:
		return stMuted.Render("No run yet. Plan and measure plans the run, then measures on this GPU when this machine can.")
	default:
		fmt.Fprintf(&b, "%s on %s · %s\n", r.ModelID, r.TargetID, runStatus(r.Status, r.Results.Measured))
		events := seam.StageEvents(f.Tail)
		n := 0
		for _, st := range r.Stages {
			state, detail := stageState(st, events, f.alive(), r.Blocked)
			n++
			fmt.Fprintf(&b, "%s %d. %-26s %s\n", mark(state), n, plainStage[st.Key], stMuted.Render(detail))
			if st.Key == "analyze" {
				for _, row := range measureRows(r.Measure, events, f.alive()) {
					n++
					fmt.Fprintf(&b, "%s %d. %-26s %s\n", mark(row[0]), n, stageWord(row[1]), stMuted.Render(row[2]))
				}
			}
		}
		if ms := r.Measure; ms != nil && ms.Status != "measured" && ms.Command != nil {
			fmt.Fprintf(&b, "%s %s\n", stInfo.Render("to measure"), *ms.Command)
		}
		fmt.Fprintf(&b, "%s %s\n", stInfo.Render("next"), r.NextStep)
	}
	if f.Job == nil {
		return b.String() + stMuted.Render("No process was started for this run from this machine.")
	}
	state := mark("pass") + " finished"
	if f.alive() {
		state = mark("run") + " running " + f.Spin
	}
	fmt.Fprintf(&b, "\n%s  %s\n", state, stMuted.Render(fmt.Sprintf("pid %d · log %s", f.Job.PID, f.Job.LogPath)))
	tail := f.Tail
	if len(tail) > 12 {
		tail = tail[len(tail)-12:]
	}
	b.WriteString(stMuted.Render(strings.Join(tail, "\n")))
	return b.String()
}

func resultBody(f Facts, width int) string {
	if f.Run == nil {
		return stMuted.Render("No run yet. Step 4 makes one.")
	}
	res := f.Run.Results
	var b strings.Builder
	switch {
	case res.Timing.TokS != nil && res.Ceiling.TokS != nil && *res.Ceiling.TokS > 0:
		ratio := *res.Timing.TokS / *res.Ceiling.TokS
		fmt.Fprintf(&b, "%s %s measured. The limit is %.1f.\n", stAccent.Render(glyphBolt),
			stHeader.Render(fmt.Sprintf("%.1f tokens per second", *res.Timing.TokS)), *res.Ceiling.TokS)
		fmt.Fprintf(&b, "%s %.0f%% of the limit\n", bar(ratio, 20), ratio*100)
	case res.Ceiling.TokS != nil:
		fmt.Fprintf(&b, "%s %.1f tokens per second at best. Nothing measured yet.\n", stAccent.Render(glyphBolt), *res.Ceiling.TokS)
	default:
		fmt.Fprintf(&b, "%s\n", stMuted.Render("No speed limit for this chip: "+res.Ceiling.Reason))
	}
	if res.Timing.DominantBucket != nil {
		fmt.Fprintf(&b, "Where the time goes: %s\n", named(plainBucket, *res.Timing.DominantBucket))
	}
	for _, n := range res.Blocked {
		fmt.Fprintf(&b, "%s %s (%s)\n", stInfo.Render(glyphWait+" still needed:"), word(plainNeed, n.Need), n.Request)
	}
	if len(res.Routes) > 0 {
		rows := [][]string{{" ", "ROLE", "QUANT", "KERNEL CHOICE"}}
		for _, rt := range res.Routes {
			won := word(plainRoute, rt.Status)
			if rt.SelectedRoute != nil {
				won += "  " + stMuted.Render(*rt.SelectedRoute)
			}
			rows = append(rows, []string{mark(routeMark[rt.Status]), word(plainRole, rt.Role), rt.Quant, won})
		}
		b.WriteString("\n" + table(rows))
	}
	if res.Timing.Status == "classified" {
		rows := [][]string{{" ", "KERNEL", "µs", "OF STEP", "OF PEAK"}}
		for _, k := range res.Timing.Kernels {
			m := "open"
			switch k.Bucket {
			case "at_peak":
				m = "pass"
			case "gemv_codegen_capped", "latency_bound":
				m = "crossed"
			}
			rows = append(rows, []string{mark(m), k.Name, us(k.Us), pctBar(k.PctStep, 6), pctBar(k.PhysUtilPct, 6)})
		}
		b.WriteString("\n" + table(rows))
	}
	if len(res.Regimes) > 0 {
		rows := [][]string{{"ROLE", "QUANT", "WHAT THE TEST SAW"}}
		for _, g := range res.Regimes {
			rows = append(rows, []string{word(plainRole, g.Role), g.Quant, named(plainRegime, g.Classification)})
		}
		b.WriteString("\n" + table(rows))
	}
	return strings.TrimRight(b.String(), "\n")
}

// --- the checklist and the full view ----------------------------------------------------------------------

func listLine(i int, f Facts, cursor bool) string {
	m, text := steps[i].line(f)
	title := fmt.Sprintf("%d  %-12s", i+1, steps[i].title)
	if cursor {
		return stCursor.Render("▸ ") + mark(m) + " " + stCursor.Render(title) + " " + text
	}
	return "  " + mark(m) + " " + title + " " + text
}

// ChecklistView is the main screen: the five steps, then the chosen step's summary cut to fit `height` lines.
// ChecklistView fills the screen: the chosen step's results on top, stretched to the height, and the five steps
// pinned at the bottom where the keys act.
func ChecklistView(f Facts, cursor, width, height int) string {
	lines := make([]string, len(steps))
	for i := range steps {
		lines[i] = listLine(i, f, i == cursor)
	}
	list := stBox.Width(width - 2).Render(clip(lines, width-4))
	s := steps[cursor]
	hint := s.hint
	if f.alive() && cursor == 3 {
		hint = "x stop · " + hint
	}
	room := max(height-lipgloss.Height(list)-3, 2) // the results box: two borders and its title
	body := fill(strings.Split(s.body(f, width-4), "\n"), room-1)
	return box(title(cursor, f), strings.Join(body, "\n")+"\n"+stMuted.Render(hint), width, false) + "\n" + list
}

// fill cuts or pads lines to exactly n, ending a cut with "…", so the boxes always span the screen.
func fill(lines []string, n int) []string {
	if len(lines) > n {
		return append(lines[:max(n-1, 0)], stMuted.Render("…"))
	}
	return append(lines, make([]string, n-len(lines))...)
}

func title(i int, f Facts) string {
	t := fmt.Sprintf("%d  %s", i+1, steps[i].title)
	if about := steps[i].about(f); about != "" {
		t += " · " + about
	}
	return t
}

// DetailActions is the bottom of the open view: the step's line, then its action rows.
func DetailActions(f Facts, i, row, width int) string {
	var b strings.Builder
	_, text := steps[i].line(f)
	b.WriteString(text)
	if steps[i].actions != nil {
		for j, a := range steps[i].actions(f) {
			if j == row {
				b.WriteString("\n" + stCursor.Render("▸ ") + a.label)
			} else {
				b.WriteString("\n  " + a.label)
			}
		}
	}
	return box("What next", b.String(), width, false)
}

// DetailView is the open view at a given height: the step's whole result scrolls on top, actions at the bottom.
func DetailView(f Facts, i, row, width, height int, scroll *viewport.Model) string {
	actions := DetailActions(f, i, row, width)
	scroll.Width, scroll.Height = width-4, max(height-lipgloss.Height(actions)-3, 1)
	scroll.SetContent(steps[i].body(f, width-4))
	return box(title(i, f), scroll.View(), width, false) + "\n" + actions
}

func clip(lines []string, width int) string {
	for i, l := range lines {
		lines[i] = truncate(l, width)
	}
	return strings.Join(lines, "\n")
}
