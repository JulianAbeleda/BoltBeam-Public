package ui

// Tuned for this machine, on the Recommended setup page: the chosen engine's knob rows for this model and build, each
// with its value, where the value came from and why (`screen tune`, facts only, loaded the way Offload is). Measure on
// this machine runs the sweeps as a job, with progress and Stop as the other jobs have; Use these copies the values
// into Engine settings, so the next run passes them. Python owns every row and sentence; nothing here knows an engine.

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// TuneState is the section: the page for TuneKey, why it could not be read, the measure job and its log.
type TuneState struct {
	Key    string // what Page was asked for (tuneKey)
	Page   *seam.Tune
	Err    string
	GPUJob // the measure job and its log (gpujob.go)
	at     time.Time
	done   int
}

type tuneMsg struct {
	key string
	t   *seam.Tune
	err error
}
type tuneJobMsg struct {
	id   string
	job  *jobs.Job
	tail []string
}
type tuneStartedMsg struct{ job *jobs.Job }

// engineBuild is the build of the chosen engine that the Engine picker rests on: its program or python, "" when the
// picker has none (Python then finds one, as tune does).
func (f Facts) engineBuild() string {
	fam, _ := f.familyOf(f.Engine)
	if fam == nil || len(fam.Forks) == 0 {
		return ""
	}
	k := fam.Forks[f.highlighted(*fam)]
	if k.Provider != nil && *k.Provider == f.Engine && k.Path != nil {
		return *k.Path
	}
	return ""
}

// tuneAsk is what Setup says to Tuned for this machine: the context, the workload (session and replay serve follow-ups,
// so chat; decode and prefill time single prompts, so one-shot), the chip, the experts in RAM that Settings asks for
// and every other setting by its name (--setting NAME=VALUE: the screen turns it into the engine's flag, and leaves out
// one this page tunes). The experts go as --experts-in-ram, which the plan reads, so it hears them once.
func (f Facts) tuneAsk() []string {
	workload := "one-shot"
	if w := f.workload(); w == "session" || w == "replay" {
		workload = "chat"
	}
	ask := append(f.Context.tuneArgs(), "--workload", workload, "--target", f.target().ID)
	for _, s := range f.settingsOffered() {
		switch v := f.Settings[s.Name]; {
		case v == "":
		case s.Name == "n_cpu_moe":
			ask = append(ask, "--experts-in-ram", v)
		case s.Name == "cpu_moe":
			ask = append(ask, "--experts-in-ram", "all")
		default:
			ask = append(ask, "--setting", s.Name+"="+v)
		}
	}
	return ask
}

// tuneKey names what the section is for: the model, chip and engine (recommendKey), the build and what Setup asks.
func (f Facts) tuneKey() string {
	if k := f.recommendKey(); k != "" {
		return k + "|" + f.engineBuild() + "|" + strings.Join(f.tuneAsk(), " ")
	}
	return ""
}

// tuneID is the measure job of the model and engine: its log in the TUI's state folder.
func (f Facts) tuneID() string { return RunStem(f.Path, "tune-"+f.Engine) }

func (f Facts) tuning() bool {
	return f.Tune.Job != nil && f.Tune.Job.Alive && f.Tune.Job.ID == f.tuneID()
}

// tunePage is the page for what is chosen now, or nil.
func (f Facts) tunePage() *seam.TunePage {
	if f.Tune.Page == nil || f.Tune.Key != f.tuneKey() {
		return nil
	}
	return f.Tune.Page.Page
}

// loadTune reads the facts-only page once per model, chip, engine and build: a few seconds of CPU, no engine starts.
func (m *Model) loadTune() tea.Cmd {
	key := m.f.tuneKey()
	if key == "" || key == m.f.Tune.Key {
		return nil
	}
	m.f.Tune = TuneState{Key: key} // another model, engine or build: another page and another job
	path, engine, build, ask, client := m.f.Path, m.f.Engine, m.f.engineBuild(), m.f.tuneAsk(), m.client
	return tea.Batch(func() tea.Msg {
		t, err := client.Tune(path, engine, build, ask...)
		return tuneMsg{key, t, err}
	}, m.readTune())
}

// readTune reads the measure job and its whole log: the answer is one JSON object after the progress lines.
func (m Model) readTune() tea.Cmd {
	id := m.f.tuneID()
	return m.readGPUJob(id, func(j *jobs.Job, tail []string) tea.Msg { return tuneJobMsg{id, j, tail} })
}

// startTune starts the measure job on a free GPU: every measured row, or just the knob named.
func (m Model) startTune(knob string) tea.Cmd {
	if m.f.tuning() || m.f.tunePage() == nil || m.f.tunePage().Measure.None != nil {
		return nil
	}
	ask := m.f.tuneAsk()
	if knob != "" {
		ask = append(ask, "--knob", knob)
	}
	argv := m.client.TuneArgv(m.f.Path, m.f.Engine, m.f.engineBuild(), ask...)
	return m.startGPUJob(m.f.tuneID(), argv, func(j *jobs.Job) tea.Msg { return tuneStartedMsg{j} })
}

// advance times the knob in flight, as Speed up times its stage.
func (s *TuneState) advance(now time.Time) {
	if s.Job == nil {
		return
	}
	p := seam.ReadProgress(s.Tail)
	if s.done != p.Done || s.at.IsZero() {
		s.done, s.at = p.Done, now
		if t, err := time.Parse(time.RFC3339, s.Job.StartedAt); err == nil && p.Done == 0 {
			s.at = t
		}
	}
}

// tuneMsg handles the section's messages; ok says it was one.
func (m Model) tuneMsg(msg tea.Msg) (Model, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tuneMsg:
		if msg.key != m.f.Tune.Key {
			break
		}
		if msg.err != nil {
			m.f.Tune.Err = msg.err.Error()
			break
		}
		m.f.Tune.Page = msg.t
	case tuneStartedMsg:
		m.f.Tune.GPUJob = GPUJob{Job: msg.job, Seen: msg.job.StartedAt}
		m.f.Tune.done, m.f.Tune.at = -1, time.Time{}
		m.f.Tune.advance(clock())
		m.note = ""
		return m, m.tickOn(), true
	case tuneJobMsg:
		if msg.id != m.f.tuneID() {
			break
		}
		running, ended := m.f.Tune.take(msg.id, msg.job, msg.tail)
		m.f.Tune.advance(clock())
		switch t, err := seam.ReadTune(msg.tail); {
		case running:
			return m, m.tickOn(), true
		case ended && m.f.Tune.Stopped: // Stop was sent: the values read from facts stand, and the note says so
		case ended && err == nil && t.Page != nil:
			m.f.Tune.Page, m.f.Tune.Err = t, ""
			m.note = t.Page.Says
		case ended:
			if p := m.f.tunePage(); p != nil {
				m.note = strings.ReplaceAll(p.Measure.Failed, "{log}", msg.job.LogPath)
			}
		}
	case gpuStoppedMsg:
		if msg.id == m.f.tuneID() {
			m.f.Tune.Stopped, m.note = true, msg.note
		}
	default:
		return m, nil, false
	}
	return m, nil, true
}

// doTune is one row of the section.
func (m Model) doTune(a action) (Model, tea.Cmd) {
	switch a.do {
	case "tunemeasure":
		return m, m.startTune(a.arg)
	case "tunestop":
		if p := m.f.tunePage(); p != nil {
			return m, m.stopGPUJob(m.f.tuneID(), p.Measure.Stopped)
		}
	case "tuneuse":
		p := m.f.tunePage()
		if p == nil {
			return m, nil
		}
		n := 0
		// Use these replaces what the last one copied: its settings are cleared first, and the environment is exactly
		// this page's set. Settings typed by hand are not in TunedSettings and stay.
		for _, name := range m.f.TunedSettings {
			m.f.setSetting(name, "")
		}
		m.f.TunedSettings = nil
		for _, name := range slices.Sorted(maps.Keys(p.Use.Settings)) {
			if s := m.f.setting(name); s != nil { // a setting this engine's runs take, checked as the form checks it
				if v, ok := settingTyped(*s, p.Use.Settings[name]); ok && v != "" {
					m.f.setSetting(name, v)
					m.f.TunedSettings = append(m.f.TunedSettings, name)
					n++
				}
			}
		}
		m.f.EnvFor = m.f.envKey()
		m.f.Env = nil
		for _, name := range slices.Sorted(maps.Keys(p.Use.Env)) {
			m.f.setEnv(name, p.Use.Env[name])
			n++
		}
		m.f.Tune.Key = m.f.tuneKey() // the copied values are now Settings, so asked of this page again: it stands, not re-read
		m.note = strings.ReplaceAll(p.Use.Used, "{n}", fmt.Sprint(n))
		if p.Use.NotPassed != nil {
			m.note += " " + *p.Use.NotPassed
		}
	}
	return m, nil
}

// tuneActions are the section's rows: Measure on this machine (or Stop while it runs), and Use these.
func tuneActions(f Facts) []action {
	p := f.tunePage()
	if p == nil {
		return nil
	}
	out := []action{}
	switch ms := p.Measure; {
	case f.tuning():
		out = append(out, action{"[ " + ms.Stop + " ] " + ms.StopHelper, "tunestop", ""})
	case ms.None != nil:
		out = append(out, note("  [ "+ms.Label+" ] "+*ms.None))
	default:
		out = append(out, action{"[ " + ms.Label + " ] " + ms.Helper, "tunemeasure", ""})
		for _, k := range ms.Knobs { // one measured row on its own
			out = append(out, action{"[ " + ms.One + " ] " + k, "tunemeasure", k})
		}
	}
	if u := p.Use; u.None != nil {
		out = append(out, note("  [ "+u.Label+" ] "+*u.None))
	} else {
		out = append(out, action{"[ " + u.Label + " ] " + u.Helper, "tuneuse", ""})
	}
	return out
}

// tuneSetting is a row's value as the engine takes it: NAME=VALUE for the environment or a config key, else the
// flag and its value.
func tuneSetting(r seam.TuneRow) string {
	if r.Kind == "env" || r.Kind == "config" {
		return r.Name + "=" + r.Value
	}
	return r.Name + " " + r.Value
}

// tuneBody is the section: its title and what it is for, then each row's value with its source and tags (a tag is
// never cut), its reason under it, the rows not used, and the command. While Measure runs, its bar leads.
func tuneBody(f Facts, width int) string {
	t := f.Tune
	switch {
	case f.tuneKey() == "":
		return ""
	case t.Err != "" && t.Key == f.tuneKey():
		return "\n" + paint(stMuted, hangHard("Tuned for this machine could not be read: "+t.Err, 0, width)) + "\n"
	case f.tunePage() == nil && t.Page != nil && t.Key == f.tuneKey() && t.Page.Refused != nil:
		return "\n" + paint(stMuted, hangHard(*t.Page.Refused, 0, width)) + "\n"
	case f.tunePage() == nil:
		return "\n" + stMuted.Render(strings.TrimSpace(f.Spin+" reading this machine and the build…")) + "\n"
	}
	p := f.tunePage()
	var b strings.Builder
	b.WriteString("\n" + stHeader.Render(p.Title) + "\n")
	if f.tuning() {
		b.WriteString(tuneProgress(f, p, width))
	}
	b.WriteString(hangHard(p.Head, 2, width) + "\n")
	for _, n := range p.Notes {
		b.WriteString(paint(stMuted, hangHard(n, 2, width)) + "\n")
	}
	b.WriteString(paint(stMuted, hangHard(p.Says, 2, width)) + "\n\n")
	for _, r := range p.Rows {
		line := tuneSetting(r) + " · " + r.SourceWords
		b.WriteString(hangHard(line, 4, width)[2:] + "\n") // the row at 2, what wraps of it at 4
		for _, tag := range r.Tags {
			b.WriteString(paint(stWarn, hangHard(glyphWarn+" "+tag, 4, width)) + "\n")
		}
		b.WriteString(paint(stMuted, hangHard(r.Reason, 4, width)) + "\n")
		if r.Confirm != nil {
			b.WriteString(paint(stMuted, hangHard(*r.Confirm, 4, width)) + "\n")
		}
	}
	if len(p.NotUsed) > 0 {
		b.WriteString("\n  " + stHeader.Render(p.NotUsedTitle) + "\n")
		for _, r := range p.NotUsed {
			b.WriteString(paint(stMuted, hangHard(r.Name+": "+r.Reason, 6, width)[2:]) + "\n")
		}
	}
	for _, a := range p.Absent {
		b.WriteString("\n" + paint(stMuted, hangHard(a, 2, width)) + "\n")
	}
	if p.Use.NotPassed != nil {
		b.WriteString("\n" + paint(stMuted, hangHard(*p.Use.NotPassed, 2, width)) + "\n")
	}
	b.WriteString("\n" + paint(stMuted, hangHard(p.Command, 4, width)[2:]) + "\n")
	return b.String()
}

// hangHard is hang that also breaks a word longer than the line (a path, a command): nothing runs past the width.
func hangHard(text string, indent, width int) string {
	lines := strings.Split(ansi.Wrap(text, max(width-indent, 10), " "), "\n")
	pad := strings.Repeat(" ", indent)
	for i, l := range lines {
		lines[i] = pad + strings.TrimRight(l, " ")
	}
	return strings.Join(lines, "\n")
}

// tuneProgress is Measure's bar: the knobs swept of all, the point in flight, the time taken.
func tuneProgress(f Facts, p *seam.TunePage, width int) string {
	pr := seam.ReadProgress(f.Tune.Tail)
	frac := pr.Fraction(clock().Sub(f.Tune.at).Seconds())
	count := ""
	if pr.Total > 0 {
		count = fmt.Sprintf("%d of %d", min(pr.Done+1, pr.Total), pr.Total)
	}
	now := p.Measure.Starting
	if pr.Current != "" {
		now = pr.Current
		if pr.SubOf > 0 {
			now = strings.NewReplacer("{knob}", pr.Current, "{i}", fmt.Sprint(pr.Sub+1), "{n}", fmt.Sprint(pr.SubOf)).Replace(p.Measure.Progress)
		}
	}
	barW := max(width-4-4-lipgloss.Width(count)-6, 10)
	return fmt.Sprintf("  %s  %3.0f%%  %s\n%s\n", bar(frac, barW), frac*100, count,
		hangHard(f.Spin+" "+p.Measure.Running+": "+now+"…   "+stMuted.Render(elapsed(f.Tune.Job)), 2, width))
}
