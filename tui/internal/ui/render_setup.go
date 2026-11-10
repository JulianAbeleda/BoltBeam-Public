package ui

// The setup pages' bodies: the model, the chip and its limit, and what was measured.

import (
	"fmt"
	"slices"
	"strings"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

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
	slices.Sort(parts)
	return strings.Join(parts, ", ")
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
