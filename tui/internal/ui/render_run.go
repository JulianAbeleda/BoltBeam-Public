package ui

// A run while it runs: its stages, its job, and why it failed.

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// runningBody is the whole top box while a run is going: what runs, the bar, the step and the time. l adds the log.
func runningBody(f Facts, width int) string {
	p := seam.ReadProgress(f.Tail)
	var b strings.Builder
	b.WriteString(runHeadline(f) + "\n\n")
	count := ""
	if p.Total > 0 && !p.Setup { // the setup stages are not steps: never "0 of N"
		count = fmt.Sprintf("%d of %d", min(p.Done+1, p.Total), p.Total)
	}
	if p.Repeats > 1 { // a repeated run: which of its runs this is
		count = strings.TrimSpace(fmt.Sprintf("run %d of %d  %s", p.Repeat, p.Repeats, count))
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
		line += " with " + engine + jobLayout(f)
	}
	if m := jobMeasurement(f); m != "" {
		line += " · " + m
	}
	return line
}

// jobLayout is the job's GPU layout after its engine: its --layout when the job has one, else Setup's; nothing for one
// GPU, so a one-GPU run reads as it always has.
func jobLayout(f Facts) string {
	if f.Job != nil {
		for i, a := range f.Job.Argv {
			if a == "--layout" && i+1 < len(f.Job.Argv) {
				if f.Job.Argv[i+1] == "one" {
					return ""
				}
				g := f
				g.Layout = f.Job.Argv[i+1]
				return " · " + layoutLabel(g)
			}
		}
	}
	return layoutSuffix(f)
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

// failedBody is a run that stopped: the step that failed and why, in one sentence. What stands behind the
// sentence (the engine's own traceback, then the last lines of the log) waits behind l, "Show details".
func failedBody(f Facts, width int) string {
	p := seam.ReadProgress(f.Tail)
	var b strings.Builder
	b.WriteString(runHeadline(f) + "\n\n")
	b.WriteString(mark("fail") + " " + stageWord(p.Failed) + " failed\n")
	if p.Reason != "" {
		b.WriteString(p.Reason + "\n")
	}
	if p.Failed == "role_time" {
		b.WriteString("Report incomplete. Measurements and capture logs are kept.\n")
	}
	if !f.ShowLog {
		b.WriteString("\n" + stMuted.Render("Show details: l") + "\n")
		return b.String()
	}
	if len(p.Details) > 0 {
		b.WriteString("\n" + stHeader.Render("Details") + "\n")
		b.WriteString(stMuted.Render(strings.Join(fit(p.Details, width), "\n")) + "\n")
	}
	tail := p.Lines
	if len(tail) > 10 {
		tail = tail[len(tail)-10:]
	}
	b.WriteString("\n" + stHeader.Render("Last lines of the log") + "\n")
	b.WriteString(stMuted.Render(strings.Join(fit(tail, width), "\n")))
	return b.String()
}
