// Package ui owns the screen. Every view here is a pure function of seam facts and a size, so a test can
// render a screen to a string and pin it; model.go wires the same functions to Bubble Tea.
package ui

import (
	"fmt"
	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/x/ansi"
	"slices"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
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
	"measure": "Can %s measure", "measure_probe": "Test blocks on the GPU",
	"measure_timing": "Time the real decode",
	"role_time":      "Time each role", "search": "Search faster kernels",
}

// plainVerdict words a role's kernel search verdict; report/html.py PLAIN_VERDICT says the same.
var plainVerdict = map[string]string{"applied": "applied", "found_not_applied": "found, not applied",
	"none_faster": "none faster", "not_searched": "not searched"}

// found is a role's best kernel found and its verdict, as two cells.
func found(r seam.RoleLoss) (string, string) {
	best, verdict := "", ""
	if r.BestFound != nil {
		best = r.BestFound.Text
	}
	if r.Verdict != nil {
		verdict = word(plainVerdict, *r.Verdict)
	}
	return best, verdict
}

// stageWord is the plain word for any pipeline stage key.
func stageWord(key string) string {
	if plain, ok := plainMeasureStage[key]; ok {
		if strings.Contains(plain, "%s") {
			return fmt.Sprintf(plain, here())
		}
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
		// the scope is the machine the facts were measured on, as recorded; it is never this machine read live
		scope := *t.Scope
		if t.ScopeObservedAt != nil {
			scope += stMuted.Render(" · recorded " + *t.ScopeObservedAt)
		}
		b.WriteString(scope + "\n")
	}
	if t.ID == f.ThisMachine && f.Driver != "" {
		b.WriteString(stMuted.Render(capFirst(here())+" now: driver "+f.Driver+", read live") + "\n")
	}
	fmt.Fprintf(&b, "%s · %s\n", memoryLine(f, *t), tflops(*t))
	b.WriteString(stMuted.Render(fmt.Sprintf("%s · %s · memory from %s · compute from %s", t.Backend, t.BackendStatus,
		t.FactStatus.MemoryBandwidthGBs, t.FactStatus.PeakTFLOPS)))
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
	memory := c.Target.FactStatus.MemoryBandwidthGBs
	if c.BandwidthSource != nil {
		memory = *c.BandwidthSource
	}
	b.WriteString(stMuted.Render(fmt.Sprintf("memory speed %s · compute from %s · %s", memoryFrom(memory),
		c.Target.FactStatus.PeakTFLOPS, note)) + "\n\n")
	b.WriteString(stHeader.Render("One token, per role") + "\n" + roleRows(d.Roles) + "\n")
	b.WriteString(stHeader.Render(fmt.Sprintf("A prompt of %d tokens, per role", p.Context)) + "\n" + roleRows(p.Roles) + "\n")
	b.WriteString(stHeader.Render("Assumptions") + "\n" + stMuted.Render(strings.Join(c.Assumptions, "\n")))
	b.WriteString(otherChips(f, c.Target.ID))
	return b.String()
}

// otherChips is read-only what-if: this model's limit on every other registered chip. Nothing here is chosen.
func otherChips(f Facts, current string) string {
	if f.Others == nil {
		return ""
	}
	rows := [][]string{}
	for _, c := range f.Others.Chips {
		if c.ID == current || c.TokS == nil {
			continue
		}
		rows = append(rows, []string{c.ID, fmt.Sprintf("%.1f tokens per second", *c.TokS), fmt.Sprintf("%.1f GB/s", c.PeakBandwidthGBs)})
	}
	if len(rows) == 0 {
		return ""
	}
	return "\n\n" + stHeader.Render("On other chips") + "\n" + table(rows)
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
	if st.State == "not_needed" {
		return "skip", deref(st.StateNote)
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
		case ms != nil && !alive && key == "measure_probe" && deref(ms.Probe) == "absent":
			rows = append(rows, [3]string{"crossed", key, "not here: " + deref(ms.ProbeReason)})
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
		return stMuted.Render("No run yet. Measure with llama.cpp or tinygrad: it plans the run, then measures on this GPU when this machine can.")
	default:
		fmt.Fprintf(&b, "%s on %s · %s\n", r.ModelID, r.TargetID, runStatus(r.Status, r.Results.Measured))
		events := seam.StageEvents(f.Tail)
		n := 0
		for _, st := range r.Stages {
			if st.State == "not_needed" { // Python said this run did not need it: not a step of this run
				continue
			}
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
		if ms := r.Measure; ms != nil && (ms.Status == "skipped" || ms.Status == "failed") { // a measured run's next is its results
			fmt.Fprintf(&b, "%s %s\n", stInfo.Render("next"), r.NextStep)
		}
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

// captureWords names how per-role time was taken, in the screen's words.
var captureWords = map[string]string{
	"tinygrad-profile-events": "tinygrad's own timing",
	"nsys":                    "captured with nsys",
	"rocprofv3":               "captured with rocprofv3",
	"metal-system-trace":      "captured with Metal System Trace",
	// collectors/engine_kernels.py METHOD and WORDS: the engine's own kernels, timed alone by BoltBeam's kernel timer
	"boltbeam-kernel-timer-isolated": "isolated, timed by BoltBeam's kernel timer",
}

// lossTitle is step 5's heading: the provider and how its roles were timed.
func lossTitle(provider string, c *seam.Capture) string {
	how := "roles not timed"
	if c != nil && c.Method != nil {
		how = word(captureWords, *c.Method)
	}
	return provider + " · " + how
}

// tableWidth is the page width roleTable fits; resultBody sets it from the screen before drawing.
var tableWidth = 200

// roleTable is one provider's per-role table: the time, the share of peak, the time per call and why. The reason
// word is the most useful column, so when the page is narrow the share bar goes first, then IDEAL; only when the
// table still does not fit do the time per call and the reason move to a second table, so no row is cut.
func roleTable(roles []seam.RoleLoss, notAttributed *float64) string {
	for _, drop := range [][]string{nil, {"SHARE OF LOSS"}, {"SHARE OF LOSS", "IDEAL ms"}} {
		if t := roleTableWithout(roles, notAttributed, drop); maxWidth(t) <= tableWidth {
			return t
		}
	}
	return roleTableNarrow(roles, notAttributed)
}

func maxWidth(s string) int {
	w := 0
	for _, l := range strings.Split(s, "\n") {
		w = max(w, lipgloss.Width(l))
	}
	return w
}

// roleTableWithout is the full table without the named columns.
func roleTableWithout(roles []seam.RoleLoss, notAttributed *float64, drop []string) string {
	rows := [][]string{{"ROLE", "QUANT", "IDEAL ms", "ACTUAL ms", "LOST ms", "SHARE OF LOSS", "% PEAK", "µs/CALL", "WHY", "BEST FOUND", "VERDICT"}}
	for _, r := range roles {
		pct, us := "", ""
		if r.PctPeak != nil {
			pct = fmt.Sprintf("%.1f%%", *r.PctPeak)
		}
		if r.UsPerCall != nil {
			us = fmt.Sprintf("%.1f", *r.UsPerCall)
		}
		best, verdict := found(r)
		rows = append(rows, []string{word(plainRole, r.Role), r.Quant, fmt.Sprintf("%.2f", r.IdealMs),
			fmt.Sprintf("%.2f", r.ActualMs), fmt.Sprintf("%.2f", r.LostMs), bar(r.Share, 5) + fmt.Sprintf(" %.0f%%", r.Share*100), pct, us, r.Reason, best, verdict})
	}
	if notAttributed != nil {
		rows = append(rows, []string{"not attributed", "", "", fmt.Sprintf("%.2f", *notAttributed), "", "", "", "", "", "", ""})
	}
	if !searched(roles) {
		drop = append(drop, "BEST FOUND", "VERDICT")
	}
	keep := []int{}
	for i, h := range rows[0] {
		if !slices.Contains(drop, h) {
			keep = append(keep, i)
		}
	}
	for j, row := range rows {
		cut := []string{}
		for _, i := range keep {
			cut = append(cut, row[i])
		}
		rows[j] = cut
	}
	return table(rows)
}

func roleTableNarrow(roles []seam.RoleLoss, notAttributed *float64) string {
	main := [][]string{{"ROLE", "QUANT", "IDEAL", "ACTUAL", "LOST", "% PEAK"}}
	why := [][]string{{"ROLE", "QUANT", "µs/CALL", "WHY"}}
	search := [][]string{{"ROLE", "QUANT", "BEST FOUND", "VERDICT"}}
	for _, r := range roles {
		pct, us := "", ""
		if r.PctPeak != nil {
			pct = fmt.Sprintf("%.1f%%", *r.PctPeak)
		}
		if r.UsPerCall != nil {
			us = fmt.Sprintf("%.1f", *r.UsPerCall)
		}
		name := word(plainRole, r.Role)
		main = append(main, []string{name, r.Quant, fmt.Sprintf("%.2f", r.IdealMs), fmt.Sprintf("%.2f", r.ActualMs),
			fmt.Sprintf("%.2f", r.LostMs), pct})
		why = append(why, []string{name, r.Quant, us, r.Reason})
		best, verdict := found(r)
		search = append(search, []string{name, r.Quant, best, verdict})
	}
	if notAttributed != nil {
		main = append(main, []string{"not attributed", "", "", fmt.Sprintf("%.2f", *notAttributed), "", ""})
	}
	out := table(main) + "\n" + table(why)
	if searched(roles) {
		out += "\n" + table(search)
	}
	return out
}

// searched says whether any role carries a kernel search verdict (older results carry none).
func searched(roles []seam.RoleLoss) bool {
	for _, r := range roles {
		if r.Verdict != nil {
			return true
		}
	}
	return false
}

// files is the file names a list of pointers names, once each, in order.
func files(ptrs []seam.Pointer) string {
	seen, out := map[string]bool{}, []string{}
	for _, p := range ptrs {
		if !seen[p.File] {
			seen[p.File] = true
			out = append(out, p.File)
		}
	}
	return strings.Join(out, ", ")
}

// findingsBody is the verdicts read off the run's facts, one line each with its lever, and the files behind the
// per-role table. The report links each pointer; the screen names the files only.
func findingsBody(roles []seam.RoleLoss, findings []seam.Finding) string {
	var b strings.Builder
	all := []seam.Pointer{}
	for _, r := range roles {
		all = append(all, r.Evidence...)
	}
	if len(all) > 0 {
		b.WriteString(stMuted.Render("Evidence: "+files(all)) + "\n")
	}
	for _, f := range findings {
		fmt.Fprintf(&b, "%s %s · %.2f ms\n%s\n", stInfo.Render("next"), f.What, f.Ms, stMuted.Render(f.Do+" Evidence: "+files(f.Evidence)))
	}
	return b.String()
}

// searchBody is Run's kernel search under the per-role table: why it was skipped, or the next step it found.
func searchBody(s *seam.Search) string {
	if s == nil {
		return ""
	}
	switch {
	case s.Status == "skipped":
		return stMuted.Render("Kernel search: skipped. "+deref(s.Reason)) + "\n"
	case s.Next != nil:
		return fmt.Sprintf("%s %s.\n%s\n", stInfo.Render("next"), s.Next.What, stMuted.Render(s.Next.Do))
	}
	return ""
}

// tieOutBody is the measured token line by line against the limit. The last line is the difference.
func tieOutBody(t *seam.TieOut) string {
	if t == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString(stHeader.Render(fmt.Sprintf("Tie-out, ms per token at context %.0f", t.Context)) + "\n")
	if t.Refused != nil {
		b.WriteString(stWarn.Render(glyphWarn+" Not tied out. ") + *t.Refused + "\n")
		return b.String()
	}
	if t.TokenMs == nil {
		return b.String()
	}
	if t.Estimate != nil {
		return b.String() + isolatedTieOut(t)
	}
	rows := [][]string{}
	for i, l := range t.Lines {
		sign := "+ "
		if i == 0 {
			sign = "  "
		}
		rows = append(rows, []string{sign + l.Label, fmt.Sprintf("%.3f", l.Ms), l.How})
	}
	rows = append(rows, []string{"= measured token", fmt.Sprintf("%.3f", *t.TokenMs), deref(t.TokenSource)})
	b.WriteString(table(rows))
	for _, l := range t.Lines {
		if len(l.Parts) > 0 {
			parts := []string{}
			for _, p := range l.Parts {
				parts = append(parts, fmt.Sprintf("%s %.3f", p.Kind, p.Ms))
			}
			b.WriteString(stMuted.Render("  other kernels: "+strings.Join(parts, ", ")) + "\n")
		}
	}
	if t.BusyMs != nil && t.Isolated {
		fmt.Fprintf(&b, "The weight kernels, timed alone, sum to %.3f ms of the %.3f ms token; the rest is attention, norms, the KV read and idle time, not timed.\n", *t.BusyMs, *t.TokenMs)
	} else if t.BusyMs != nil {
		fmt.Fprintf(&b, "All kernels sum to %.3f ms against the real token of %.3f ms; the gap is their difference.\n", *t.BusyMs, *t.TokenMs)
	}
	if t.ShowBoth {
		fmt.Fprintf(&b, "The limit is %.3f ms at context 1 and %.3f ms at context %.0f.\n", t.LimitMsCtx1, t.LimitMs, t.Context)
	}
	if t.UntracedMs != nil && t.TokenSource != nil && strings.HasPrefix(*t.TokenSource, "the captured run") {
		fmt.Fprintf(&b, "Tracing slowed the token: %.3f ms here, %.3f ms untraced.\n", *t.TokenMs, *t.UntracedMs)
	}
	if t.Missing != nil {
		b.WriteString(stMuted.Render("Missing: "+*t.Missing) + "\n")
	}
	return b.String()
}

// isolatedTieOut is the tie-out of a run whose kernels were timed alone: only what is measured, the ideal at
// context 1 and at the attended context, the token, and their difference. The split is one estimate line.
func isolatedTieOut(t *seam.TieOut) string {
	var b strings.Builder
	rows := [][]string{{"  limit at context 1 (ideal)", fmt.Sprintf("%.3f", t.LimitMsCtx1), "derived"}}
	for i, l := range t.Lines {
		sign := "+ "
		if i == 0 {
			sign = "  "
		}
		rows = append(rows, []string{sign + l.Label, fmt.Sprintf("%.3f", l.Ms), l.How})
	}
	rows = append(rows, []string{"= measured token", fmt.Sprintf("%.3f", *t.TokenMs), deref(t.TokenSource)})
	b.WriteString(table(rows))
	e := t.Estimate
	up, least := "", ""
	if e.Scaled {
		up, least = "up to ", "at least "
	}
	fmt.Fprintf(&b, "Estimated split (isolated): weight kernels %s%.1f ms, other and gaps %s%.1f ms.\n", up, e.WeightMs, least, e.OtherMs)
	if e.OtherHow != "" {
		fmt.Fprintf(&b, "  (%s)\n", e.OtherHow)
	}
	return b.String()
}

// estimateHow is how the estimate's numbers were made from the isolated times (workflow/screen.py estimate_how,
// the same words): the scale and its inputs, or "not scaled" with the sum that fits; the sum is the one less the
// dispatch floor when the rows carry one, said so.
func estimateHow(e *seam.Estimate) string {
	base, floor := e.IsolatedSumMs, ""
	if e.FloorUs != nil && e.IsolatedSumLessFloorMs != nil {
		base = *e.IsolatedSumLessFloorMs
		if e.FloorWords != nil {
			floor = " " + *e.FloorWords
		}
	}
	if e.Scaled {
		return fmt.Sprintf("scale %.3f = (%.1f - %.1f) / %.1f%s", e.Scale, e.TokenMs, e.OtherMs, base, floor)
	}
	return fmt.Sprintf("not scaled: %.1f ms alone%s fits the %.1f ms token", base, floor, e.TokenMs)
}

// isoRoleTable is an isolated run's per-role table: what was measured alone (µs per call, GB/s, the share of
// peak) and the estimated split of the token. When the rows carry a dispatch floor (lessFloor) the measured µs
// column is labelled so and the time less the floor stands beside it; GB/s and % PEAK are on that time, as the
// seam's columns_words says once. At a narrow page BYTES/CALL goes first, then IDEAL.
func isoRoleTable(roles []seam.RoleLoss, lessFloor bool) string {
	head := []string{"ROLE", "QUANT", "BYTES/CALL", "µs/CALL", "GB/s", "% PEAK", "IDEAL", "EST. ms IN TOKEN", "EST. LOST", "WHY"}
	if lessFloor {
		head = []string{"ROLE", "QUANT", "BYTES/CALL", "µs/CALL MEASURED", "LESS FLOOR", "GB/s", "% PEAK", "IDEAL", "EST. ms IN TOKEN", "EST. LOST", "WHY"}
	}
	f := func(v *float64, format string) string {
		if v == nil {
			return ""
		}
		return fmt.Sprintf(format, *v)
	}
	all := [][]string{head}
	for _, r := range roles {
		row := []string{word(plainRole, r.Role), r.Quant, f(r.MbPerCall, "%.1f MB"), f(r.UsPerCall, "%.1f")}
		if lessFloor {
			row = append(row, f(r.UsPerCallLessFloor, "%.1f"))
		}
		all = append(all, append(row, f(r.Gbs, "%.1f"), f(r.PctPeak, "%.1f%%"), fmt.Sprintf("%.2f", r.IdealMs), f(r.EstMs, "%.2f"),
			f(r.EstLostMs, "%.2f"), r.Reason))
	}
	only := func(keep ...string) string {
		rows := [][]string{}
		for _, row := range all {
			cut := []string{}
			for i, h := range head {
				if slices.Contains(keep, h) {
					cut = append(cut, row[i])
				}
			}
			rows = append(rows, cut)
		}
		return table(rows)
	}
	without := func(drop ...string) []string {
		return slices.DeleteFunc(slices.Clone(head), func(h string) bool { return slices.Contains(drop, h) })
	}
	for _, drop := range [][]string{nil, {"BYTES/CALL"}, {"BYTES/CALL", "IDEAL"}} {
		if t := only(without(drop...)...); maxWidth(t) <= tableWidth {
			return t
		}
	}
	// still too wide: the estimate in one table, what was measured alone and why in the next, so no row is cut.
	// With a floor the two µs columns take GB/s's place: % PEAK in the first table already carries the rate.
	measured := []string{"ROLE", "QUANT", "µs/CALL", "GB/s", "WHY"}
	if lessFloor {
		head[3] = "µs/CALL"
		measured = []string{"ROLE", "QUANT", "µs/CALL", "LESS FLOOR", "WHY"}
	}
	return only("ROLE", "QUANT", "% PEAK", "EST. ms IN TOKEN", "EST. LOST") + "\n" + only(measured...)
}

// lossBody is the end result for the run's provider: its speed against the limit and where it loses time.
// Another provider's numbers for the same run are shown after it, each labelled with its provider.
func lossBody(l seam.Loss) string { return lossBodyAt(l, 1) }

// lossBodyAt is lossBody for a run that also timed batch > 1: its heading names the batch.
func lossBodyAt(l seam.Loss, batch int) string {
	if l.Status != "modeled" || l.LimitMs == nil {
		return ""
	}
	var b strings.Builder
	provider := l.Provider
	if provider == "" {
		provider = "llama.cpp"
	}
	title := "Measured with " + lossTitle(provider, l.Capture)
	if batch > 1 {
		title += fmt.Sprintf(" · batch %d", batch)
	}
	b.WriteString(stHeader.Render(title) + "\n")
	tie := l.TieOut
	if tie != nil && tie.Missing != nil && l.Missing != nil { // said once, by the per-role line below
		t := *tie
		t.Missing = nil
		tie = &t
	}
	b.WriteString(tieOutBody(tie))
	fmt.Fprintf(&b, "The limit is %.1f ms per token at context 1.\n", *l.LimitMs)
	for _, r := range l.Runtimes {
		what := fmt.Sprintf("%.1f tokens per second, %.1f ms per token", r.TokS, r.Ms)
		if r.Isolated { // a sum of kernels timed alone is not a token: it has no "lost"
			fmt.Fprintf(&b, "%s %s per role: %.1f ms, the kernels timed alone and summed.\n", mark("open"), r.Provider, r.Ms)
			b.WriteString("  " + stMuted.Render("Not a token: attention, norms and gaps are not in it.") + "\n")
			continue
		}
		if r.PerRole {
			what = fmt.Sprintf("%.1f ms of GPU time per token", r.Ms)
		}
		label := r.Provider
		if r.PerRole {
			label += " per role"
		}
		fmt.Fprintf(&b, "%s %s: %s, %s lost.\n", mark("open"), label, what, stHeader.Render(fmt.Sprintf("%.1f ms", r.LostMs)))
		if !r.PerRole {
			b.WriteString("  " + stMuted.Render(r.Note) + "\n")
		}
	}
	if l.Refused != nil {
		b.WriteString(stWarn.Render(glyphWarn+" Per role: not shown. ") + *l.Refused + "\n")
		return b.String()
	}
	if l.Missing != nil {
		b.WriteString(stMuted.Render("Per role: "+*l.Missing) + "\n")
		b.WriteString(findingsBody(nil, l.Findings)) // what the run's own facts say (a failed graph replay) stands without roles
		return b.String() + othersBody(l.Others, provider)
	}
	if len(l.Roles) == 0 {
		return b.String() + othersBody(l.Others, provider)
	}
	shown := provider
	if l.RolesProvider != nil {
		shown = *l.RolesProvider
	}
	if shown != provider && l.ProviderMissing != nil {
		b.WriteString(stMuted.Render("Per role for "+provider+": "+*l.ProviderMissing) + "\n")
	}
	if e := l.Estimate; e != nil {
		b.WriteString("\n" + stHeader.Render("Per role, estimated from isolated kernel times") + "\n" + e.Method + "\n")
		b.WriteString(stMuted.Render(fmt.Sprintf("EST. columns %s: %s.", e.Label, estimateHow(e))) + "\n")
		if e.ColumnsWords != nil {
			b.WriteString(stMuted.Render(*e.ColumnsWords+".") + "\n")
		}
		b.WriteString(isoRoleTable(l.Roles, e.ColumnsWords != nil))
	} else {
		b.WriteString("\nWhere " + shown + " loses time, " + deref(l.Source) + ":\n" + roleTable(l.Roles, l.NotAttributedMs))
	}
	if l.RoleRule != nil {
		b.WriteString(stMuted.Render("Why: "+*l.RoleRule) + "\n")
	}
	b.WriteString(searchBody(l.Search))
	b.WriteString(findingsBody(l.Roles, l.Findings))
	if len(l.UnpairedRoles) > 0 {
		names := []string{}
		for _, u := range l.UnpairedRoles {
			names = append(names, word(plainRole, u.Role)+" "+u.Quant)
		}
		b.WriteString(stMuted.Render("Not split out, so in not attributed: "+strings.Join(names, ", ")+".") + "\n")
	}
	if l.RoleSourceWords != nil {
		b.WriteString(stMuted.Render("Per-role source: "+*l.RoleSourceWords) + "\n")
	}
	if cc := l.CrossCheck; cc != nil && len(cc.Rows) > 0 {
		b.WriteString("\n" + stHeader.Render("Cross-check: "+cc.Words) + "\n")
		if cc.ColumnsWords != nil {
			b.WriteString(stMuted.Render(*cc.ColumnsWords+".") + "\n")
		}
		b.WriteString(rateTable(cc.Rows))
	}
	b.WriteString(othersBody(l.Others, shown))
	return b.String()
}

// rateTable is a labelled row group of read rates per role: µs per call, GB/s and the share of peak. Rows that
// carry a time less the dispatch floor get that column beside the measured one; GB/s and OF PEAK are then on it.
func rateTable(rows []seam.RateRow) string {
	lessFloor := slices.ContainsFunc(rows, func(r seam.RateRow) bool { return r.UsPerCallLessFloor != nil })
	t := [][]string{{"ROLE", "QUANT", "µs/CALL", "GB/s", "OF PEAK"}}
	if lessFloor {
		t = [][]string{{"ROLE", "QUANT", "µs/CALL MEASURED", "LESS FLOOR", "GB/s", "OF PEAK"}}
	}
	for _, r := range rows {
		pct := num(r.PctPeak)
		if r.PctPeak != nil {
			pct = fmt.Sprintf("%.0f%%", *r.PctPeak)
		}
		row := []string{word(plainRole, r.Role), r.Quant, us(r.UsPerCall)}
		if lessFloor {
			row = append(row, us(r.UsPerCallLessFloor))
		}
		t = append(t, append(row, num(r.Gbs), pct))
	}
	return table(t)
}

// othersBody is the other provider's numbers, each labelled with its provider and, for another run, its run.
func othersBody(others []seam.OtherLoss, shown string) string {
	var b strings.Builder
	for _, o := range others {
		if o.Provider == shown && o.Run == nil {
			continue
		}
		label := o.Provider
		if o.Run != nil {
			label += " (run " + shortRun(*o.Run) + ")"
		}
		speed := "not measured" // never 0.0 for a speed that was not taken
		if o.TokS != nil && *o.TokS > 0 {
			speed = fmt.Sprintf("%.1f tokens per second", *o.TokS)
		} else if o.Missing != nil {
			speed = *o.Missing
		}
		title := lossTitle(label, &o.Capture)
		if (o.TokS == nil || *o.TokS <= 0) && len(o.Roles) == 0 {
			title = label // the reason says why; "roles not timed" would only repeat it
		}
		fmt.Fprintf(&b, "\nBeside it, %s: %s.\n", title, speed)
		if len(o.Roles) > 0 {
			b.WriteString(roleTable(o.Roles, o.NotAttributedMs))
		}
	}
	return b.String()
}

// compareNote says where every role and compare time comes from. Python names the runtime (compare-ready
// "runtime", from tinygrad_role_time.runtime_name); the screen never names a backend itself.
func compareNote(f Facts) string {
	runtime := "tinygrad's runtime"
	if f.Ready != nil && f.Ready.Runtime != nil {
		runtime = *f.Ready.Runtime
	}
	return "Times are from " + runtime + ", not llama.cpp."
}

// compareBody is step 5's kernel choice per role: progress while the compare job runs, what is missing when
// this machine cannot compare, and the per-role table once any role has been compared.
func compareBody(f Facts) string {
	res := f.Run.Results
	var b strings.Builder
	if f.comparing() {
		p := seam.ReadCompare(f.CTail)
		rows := [][]string{{" ", "ROLE", "QUANT", "COMPARING"}}
		for _, rt := range res.Routes {
			key, state, m := rt.Role+" "+rt.Quant, "waiting", "open"
			if status, ok := p.Done[key]; ok {
				state, m = word(plainRoute, status), routeMark[status]
			} else if key == p.Now {
				state, m = compareDoing[p.Doing]+"…", "run"
			}
			rows = append(rows, []string{mark(m), word(plainRole, rt.Role), rt.Quant, state})
		}
		return "\n" + table(rows) + stMuted.Render(compareNote(f)) + "\n"
	}
	compared := false
	for _, rt := range res.Routes {
		compared = compared || rt.Status != "unmeasured"
	}
	if r := f.Ready; r != nil && !compared {
		switch {
		case !r.Ready:
			fmt.Fprintf(&b, "\n%s %s\n", stInfo.Render(glyphWait+" cannot time or compare kernels here:"), deref(r.Message))
			if r.Fix != nil {
				fmt.Fprintf(&b, "Fix: %s\n", stAccent.Render(*r.Fix))
			}
		case !r.Applies:
			fmt.Fprintf(&b, "\n%s\n", stMuted.Render(deref(r.CompareMessage)))
		}
	}
	if len(res.Routes) > 0 && !compared {
		b.WriteString("\n" + stMuted.Render(noKernelChoice) + "\n")
		return b.String()
	}
	if !compared {
		return b.String()
	}
	rows := [][]string{{" ", "ROLE", "QUANT", "KERNEL CHOICE", "KERNEL ALONE", "WHOLE MODEL tok/s"}}
	reasons := []string{}
	for _, rt := range res.Routes {
		choice, kernel, whole := word(plainRoute, rt.Status), "-", "-"
		if c := rt.Compare; c != nil {
			if c.Plan != nil {
				choice += " · " + *c.Plan
			}
			if k := c.Kernel; k != nil {
				kernel = fmt.Sprintf("plan %.0f µs alone", k.PlanUs)
				if k.ModelUsPerCall != nil && k.FasterThanModel != nil {
					word := "slower"
					if *k.FasterThanModel {
						word = "faster"
					}
					kernel = fmt.Sprintf("model %.0f µs to plan %.0f µs (%s)", *k.ModelUsPerCall, k.PlanUs, word)
				}
			}
			if ab := c.AB; ab != nil && ab.BaselineTokS != nil && ab.CandidateTokS != nil && ab.DeltaPct != nil {
				whole = fmt.Sprintf("%.2f to %.2f (%+.1f%%)", *ab.BaselineTokS, *ab.CandidateTokS, *ab.DeltaPct)
			}
			if c.Reason != nil {
				reasons = append(reasons, word(plainRole, rt.Role)+" "+rt.Quant+": "+*c.Reason)
			}
		}
		rows = append(rows, []string{mark(routeMark[rt.Status]), word(plainRole, rt.Role), rt.Quant, choice, kernel, whole})
	}
	b.WriteString("\n" + table(rows))
	b.WriteString(stMuted.Render("Kernel: the role's own kernel per call in the model, to the plan alone. Whole model: default to plan.") + "\n")
	b.WriteString(stMuted.Render(compareNote(f)) + "\n")
	for _, r := range reasons {
		b.WriteString(stMuted.Render(r) + "\n")
	}
	return b.String()
}

// noKernelChoice replaces the per-role table while no role has had kernels compared; report/html.py says the same.
const noKernelChoice = "No kernels compared yet. Every role runs the default kernel."

// hereLoss says, in place of "pick Time each role", why this machine cannot time the run's provider per role.
// The run's results never hold machine facts; this machine's provider row does.
func hereLoss(f Facts, l seam.Loss) seam.Loss {
	p := f.provider(l.Provider)
	if p == nil || p.Capture.Method != nil {
		return l
	}
	why := "Not possible on " + here() + ": " + deref(p.Capture.Reason) + "."
	generic := func(s *string) bool { return s != nil && strings.Contains(*s, "Run with") } // the run recorded no reason of its own
	if generic(l.Missing) {
		l.Missing = &why
	}
	if generic(l.ProviderMissing) {
		l.ProviderMissing = &why
	}
	if l.TieOut != nil && generic(l.TieOut.Missing) {
		t := *l.TieOut
		t.Missing = &why
		l.TieOut = &t
	}
	return l
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
		for _, a := range res.Timing.Also { // one headline: the other contexts are listed, labelled, never mixed in
			fmt.Fprintf(&b, "%s\n", stMuted.Render(fmt.Sprintf("Also measured at context %s: %.1f tokens per second (not the headline).", count(a.Context), a.TokS)))
		}
		if s := res.Loss.Step; s != nil && s.GraphFailed {
			why := ""
			if s.GraphError != nil {
				why = " (" + *s.GraphError + ")"
			}
			fmt.Fprintf(&b, "%s\n", stWarn.Render(glyphWarn+" Graph replay failed on this run"+why+": the token ran without graphs."))
		}
	case res.Ceiling.TokS != nil:
		fmt.Fprintf(&b, "%s %.1f tokens per second at best. Nothing measured yet.\n", stAccent.Render(glyphBolt), *res.Ceiling.TokS)
	default:
		fmt.Fprintf(&b, "%s\n", stMuted.Render("No speed limit for this chip: "+res.Ceiling.Reason))
	}
	batch := 1
	if ms := f.Run.Measure; ms != nil {
		batch = slices.Max(append([]int{1}, ms.Batches...))
	}
	if s := measurementSentence(res.Measurement); s != "" && res.Loss.Estimate == nil { // the method line says it otherwise
		b.WriteString(stMuted.Render(s) + "\n")
	}
	b.WriteString(lossBodyAt(hereLoss(f, res.Loss), batch))
	b.WriteString(batchTable(res.Batches))
	if res.Timing.DominantBucket != nil {
		fmt.Fprintf(&b, "Where the time goes: %s\n", named(plainBucket, *res.Timing.DominantBucket))
	}
	for _, n := range res.Blocked {
		fmt.Fprintf(&b, "%s %s (%s)\n", stInfo.Render(glyphWait+" still needed:"), word(plainNeed, n.Need), n.Request)
	}
	b.WriteString(compareBody(f))
	if res.Timing.Status == "classified" && len(res.Timing.Kernels) > 0 {
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
	clear := false
	for _, g := range res.Regimes {
		clear = clear || g.Classification != "inconclusive"
	}
	if p := res.Probe; p != nil && p.Status == "measured" && len(p.Rows) > 0 {
		// the probe's measured rates stand on their own: BoltBeam's reference kernel per role, labelled so
		fmt.Fprintf(&b, "\n%s\n%s", stHeader.Render(p.Label), rateTable(p.Rows))
		if p.DispatchFloorUs != nil {
			fmt.Fprintf(&b, "%s\n", stMuted.Render(fmt.Sprintf("Dispatch floor measured here: %.1f µs per launch.", *p.DispatchFloorUs)))
		}
	} else if len(res.Regimes) > 0 && !clear {
		fmt.Fprintf(&b, "\n%s\n", stMuted.Render(fmt.Sprintf("The building-block tests ran on %d roles. None could be classified:\n"+
			"this GPU does not report the counters the test needs.", len(res.Regimes))))
	}
	if clear {
		rows := [][]string{{"ROLE", "QUANT", "WHAT THE TEST SAW"}}
		for _, g := range res.Regimes {
			rows = append(rows, []string{word(plainRole, g.Role), g.Quant, named(plainRegime, g.Classification)})
		}
		b.WriteString("\n" + table(rows))
	}
	return strings.TrimRight(b.String(), "\n")
}

// --- the checklist and the full view ----------------------------------------------------------------------

func title(i int, f Facts) string {
	t := steps[i].title
	if about := steps[i].about(f); about != "" {
		t += " · " + about
	}
	return t
}

// DetailActions is the bottom of a screen: the page's line, then its rows. Headings are drawn as headings; when
// the rows do not fit maxRows, a window around the cursor is shown with what is above and below counted.
func DetailActions(f Facts, i, row, width, maxRows int) string {
	var b strings.Builder
	_, text := steps[i].line(f)
	b.WriteString(text)
	acts := []action{}
	if steps[i].actions != nil {
		acts = steps[i].actions(f)
	}
	start, end := 0, len(acts)
	if maxRows > 2 && len(acts) > maxRows {
		start = max(0, min(row-maxRows/2, len(acts)-maxRows+2))
		end = min(len(acts), start+maxRows-2)
		if start > 0 {
			b.WriteString("\n" + stMuted.Render(fmt.Sprintf("  ↑ %d more", start)))
		}
	}
	for j := start; j < end; j++ {
		a := acts[j]
		switch {
		case a.do == "head" && a.label == "":
			b.WriteString("\n")
		case a.do == "head" && j > start:
			b.WriteString("\n\n" + stHeader.Render(a.label)) // a blank line sets each section apart
		case a.do == "head":
			b.WriteString("\n" + stHeader.Render(a.label))
		case a.do == "note":
			b.WriteString("\n  " + a.label)
		case j == row:
			b.WriteString("\n" + stCursor.Render("▸ ") + spread(a.label, width-6))
		default:
			b.WriteString("\n  " + spread(a.label, width-6))
		}
	}
	if end < len(acts) {
		b.WriteString("\n" + stMuted.Render(fmt.Sprintf("  ↓ %d more", len(acts)-end)))
	}
	title := "What next"
	switch {
	case i == pageSetup:
		title = "Setup"
	case i == pageSaved:
		title = "Saved runs"
	case picker(i):
		title = "Choose"
	}
	body := b.String()
	if text == "" { // no line: the first row starts the box
		body = strings.TrimPrefix(body, "\n")
	}
	return box(title, body, width, false)
}

// spread puts a row's right part (after a tab) at the right edge, so the "›" marks line up.
func spread(label string, width int) string {
	left, right, ok := strings.Cut(label, "\t")
	if !ok {
		return label
	}
	left = truncate(left, width-lipgloss.Width(right)-2) // a long value is cut; the mark stays
	gap := max(width-lipgloss.Width(right)-lipgloss.Width(left), 2)
	return left + strings.Repeat(" ", gap) + right
}

// ListRows is how many rows a picker or the Saved list shows at once, whatever the terminal height: a long list
// scrolls inside that window (DetailActions counts what is above and below) instead of filling the screen.
const ListRows = 10

// listWindow is the DetailActions budget for a scrolling list: ListRows plus the two count lines, never more than
// the room on screen.
func listWindow(room int) int { return min(max(room, 6), ListRows+2) }

// DetailView is the open view at a given height: the step's whole result scrolls on top, actions at the bottom.
func DetailView(f Facts, i, row, width, height int, scroll *viewport.Model) string {
	if i == pageSetup { // Setup is its rows only: one box, padded to the screen
		actions := DetailActions(f, i, row, width, height-2)
		return actions + strings.Repeat("\n", max(height-lipgloss.Height(actions), 0))
	}
	if i == pageSaved { // the list is the page: the rows scroll in the box on top, the hint sits below
		hint := box("What next", steps[i].body(f, width-4), width, false)
		actions := DetailActions(f, i, row, width, listWindow(height-lipgloss.Height(hint)-3))
		return actions + strings.Repeat("\n", max(height-lipgloss.Height(actions)-lipgloss.Height(hint), 0)) + "\n" + hint
	}
	rows := max(height*2/3-3, 6) // the rows take up to two thirds of the screen
	if picker(i) {
		rows = listWindow(rows) // a picker is a scrolling list of ListRows
	}
	actions := DetailActions(f, i, row, width, rows)
	scroll.Width, scroll.Height = width-4, max(height-lipgloss.Height(actions)-3, 1)
	tableWidth = width - 4
	f.logRoom = scroll.Height - 6 // the log fits under the bar, so the bar never scrolls away
	scroll.SetContent(wrapText(steps[i].body(f, width-4), width-4))
	return box(title(i, f), scroll.View(), width, false) + "\n" + actions
}

// wrapText wraps sentences to the width; a table row (cells apart by two spaces) is cut instead, so columns stay.
func wrapText(text string, width int) string {
	lines := strings.Split(text, "\n")
	for i, l := range lines {
		if strings.Contains(strings.TrimLeft(ansi.Strip(l), " "), "  ") {
			lines[i] = truncate(l, width)
		} else {
			lines[i] = ansi.Wordwrap(l, width, " ")
		}
	}
	return strings.Join(lines, "\n")
}

func clip(lines []string, width int) string {
	for i, l := range lines {
		lines[i] = truncate(l, width)
	}
	return strings.Join(lines, "\n")
}

// capFirst starts a sentence with a phrase like "this machine".
func capFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// --- the running screen ----------------------------------------------------------------------------------------

// runningBody is the whole top box while a run is going: what runs, the bar, the step and the time. l adds the log.
func runningBody(f Facts, width int) string {
	p := seam.ReadProgress(f.Tail)
	var b strings.Builder
	b.WriteString(runHeadline(f) + "\n\n")
	count := ""
	if p.Total > 0 && !p.Setup { // the setup stages are not steps: never "0 of N"
		count = fmt.Sprintf("%d of %d", min(p.Done+1, p.Total), p.Total)
	}
	pct := fmt.Sprintf("%3.0f%%", f.Frac*100)
	if p.Estimate { // a first run: the weights are the pipeline's default table, so the percent is an estimate
		pct = fmt.Sprintf("~%.0f%%", f.Frac*100)
	}
	barW := max(width-4-lipgloss.Width(pct)-lipgloss.Width(count)-4, 10)
	fmt.Fprintf(&b, "  %s  %s  %s\n", bar(f.Frac, barW), pct, count)
	now := "Starting"
	if p.Current != "" && !p.Setup {
		now = stageWord(p.Current)
		if p.SubOf > 0 {
			now += fmt.Sprintf(" (%d of %d)", min(p.Sub+1, p.SubOf), p.SubOf)
		}
	}
	took := elapsed(f.Job)
	if !f.stageAt.IsZero() && p.Over(clock().Sub(f.stageAt).Seconds()) {
		took += " (longer than last time)"
	}
	fmt.Fprintf(&b, "  %s…   %s\n", now, stMuted.Render(took))
	if f.ShowLog {
		tail, room := p.Lines, 20
		if f.logRoom > 0 {
			room = max(min(room, f.logRoom), 3)
		}
		if len(tail) > room {
			tail = tail[len(tail)-room:]
		}
		for i, l := range tail {
			tail[i] = truncate(l, width) // one log line, one screen line
		}
		b.WriteString("\n" + stHeader.Render("Log") + " " + stMuted.Render(truncate(f.Job.LogPath, width-4)) + "\n")
		b.WriteString(stMuted.Render(strings.Join(tail, "\n")))
	}
	return b.String()
}

// runHeadline is "MODEL on CHIP with ENGINE".
func runHeadline(f Facts) string {
	model, chip := "", ""
	if f.Run != nil {
		model, chip = f.Run.ModelID, f.Run.TargetID
	}
	if model == "" && f.Profile != nil {
		model = f.Profile.ModelID
	}
	if chip == "" {
		chip = aboutChip(f)
	}
	line := model + " on " + chip
	if engine := jobEngine(f); engine != "" {
		line += " with " + engine
	}
	if m := jobMeasurement(f); m != "" {
		line += " · " + m
	}
	return line
}

// jobMeasurement is how the job times roles: its --role-time, else (auto) what Setup resolves to, as the picker words it.
func jobMeasurement(f Facts) string {
	if f.Job != nil {
		for i, a := range f.Job.Argv {
			if a == "--role-time" && i+1 < len(f.Job.Argv) {
				return map[string]string{"in-model": "in-model", "generic": "generic (kernel timer)"}[f.Job.Argv[i+1]]
			}
		}
	}
	if o := f.measurementOption(); o != nil {
		return o.Label
	}
	return ""
}

// measurementSentence is the results line for how the run timed its roles, from the run's own record.
func measurementSentence(m *seam.Measurement) string {
	if m == nil || m.Label == nil {
		return ""
	}
	if m.Fallback != nil {
		return "Measurement: " + *m.Label + " (in-model not available here: " + *m.Fallback + ")."
	}
	if m.Choice != "auto" {
		return "Measurement: " + *m.Label + ", chosen in Setup."
	}
	return "Measurement: " + *m.Label + "."
}

// jobEngine is the engine the job was started with (--provider), else the chosen one.
func jobEngine(f Facts) string {
	if f.Job != nil {
		for i, a := range f.Job.Argv {
			if a == "--provider" && i+1 < len(f.Job.Argv) {
				return f.Job.Argv[i+1]
			}
		}
	}
	return f.Engine
}

// elapsed is the time since the job started, as mm:ss (h:mm:ss past an hour).
func elapsed(j *jobs.Job) string {
	if j == nil {
		return ""
	}
	t, err := time.Parse(time.RFC3339, j.StartedAt)
	if err != nil {
		return ""
	}
	d := max(clock().Sub(t), 0)
	h, m, s := int(d.Hours()), int(d.Minutes())%60, int(d.Seconds())%60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, m, s)
	}
	return fmt.Sprintf("%02d:%02d", m, s)
}

// failedBody is a run that stopped: the step that failed, why, and the last lines of its log.
func failedBody(f Facts) string {
	p := seam.ReadProgress(f.Tail)
	var b strings.Builder
	b.WriteString(runHeadline(f) + "\n\n")
	b.WriteString(mark("fail") + " " + stageWord(p.Failed) + " failed\n")
	if p.Reason != "" {
		b.WriteString(p.Reason + "\n")
	}
	tail := p.Lines
	if len(tail) > 10 {
		tail = tail[len(tail)-10:]
	}
	b.WriteString("\n" + stHeader.Render("Last lines of the log") + "\n")
	b.WriteString(stMuted.Render(strings.Join(tail, "\n")))
	return b.String()
}

// batchTable is step 4's points beside their own limit, when more than batch 1 was timed: tokens per second per
// stream and in total, the limit's total, and the share of it reached.
func batchTable(rows []seam.BatchRow) string {
	wide := false
	for _, r := range rows {
		wide = wide || r.Batch > 1
	}
	if !wide {
		return ""
	}
	t := [][]string{{"BATCH", "CONTEXT", "PER STREAM", "TOTAL tok/s", "LIMIT tok/s", "OF LIMIT", "BOUND BY"}}
	for _, r := range rows {
		pct := ""
		if r.PctOfLimit != nil {
			pct = fmt.Sprintf("%.0f%%", *r.PctOfLimit)
		}
		t = append(t, []string{fmt.Sprint(r.Batch), fmt.Sprintf("%.0f", r.Context), fmt.Sprintf("%.1f", r.TokSStream),
			fmt.Sprintf("%.1f", r.TokSTotal), fmt.Sprintf("%.1f", r.Limit.TokSTotal), pct, word(plainLimit, r.Limit.Bound)})
	}
	return "\n" + table(t) + "\n"
}
