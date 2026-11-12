package ui

// Measure routing (Offload) and Measure again (This machine): one short GPU job beside a page (gpujob.go). It counted as
// measured only if its log says so; Stop, a crash, a refusal and a kept old figure leave the old figures. When it ends
// the pages that read what it measured are asked again. The words are Python's (plain_words.json measure_again).

import (
	"cmp"
	"strings"

	tea "charm.land/bubbletea/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// SideJob is the measure job last started here: which (routing or host) and its id.
type SideJob struct {
	GPUJob
	Kind, ID string
}

type sideStartedMsg struct {
	kind, id string
	job      *jobs.Job
}
type sideJobMsg struct {
	id   string
	job  *jobs.Job
	tail []string
}

func (f Facts) siding() bool {
	return f.Side.Job != nil && f.Side.Job.Alive && f.Side.Job.ID == f.Side.ID
}

func sideWords(kind string) seam.MeasureWord {
	if kind == "routing" {
		return vocab.Measure.Routing
	}
	return vocab.Measure.Host
}

func (m Model) readSide() tea.Cmd {
	id := m.f.Side.ID
	return m.readGPUJob(id, func(j *jobs.Job, tail []string) tea.Msg { return sideJobMsg{id, j, tail} })
}

func (m Model) startSide(kind string) tea.Cmd {
	id, argv := "measure-host", m.client.HostArgv()
	if kind == "routing" {
		id, argv = RunStem(m.f.Path, "routing-"+m.f.Engine), m.client.RoutingArgv(m.f.Path, m.f.Engine, m.f.engineBuild())
	}
	if m.f.siding() {
		return nil
	}
	return m.startGPUJob(id, argv, func(j *jobs.Job) tea.Msg { return sideStartedMsg{kind, id, j} })
}

// sideDone: routing prints "routing profile: PATH" when it saved one and "routing: WHY" when it refused (why); the host
// measurement prints its rows as JSON, remeasure_failed false when every figure was taken. Only the log of this start counts.
func sideDone(kind string, tail []string) (ok bool, why string) {
	for _, l := range tail {
		if strings.HasPrefix(l, "=== ") {
			ok, why = false, ""
			continue
		}
		switch {
		case strings.HasPrefix(l, "routing: "):
			why = strings.TrimPrefix(l, "routing: ")
		case kind == "routing" && strings.HasPrefix(l, "routing profile: "), kind == "host" && strings.Contains(l, `"remeasure_failed": false`):
			ok = true
		}
	}
	return ok, why
}

// sideMsg handles the job's messages; ok says it was one.
func (m Model) sideMsg(msg tea.Msg) (Model, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case sideStartedMsg:
		m.f.Side = SideJob{GPUJob: GPUJob{Job: msg.job, Seen: msg.job.StartedAt}, Kind: msg.kind, ID: msg.id}
		m.note = ""
		return m, m.tickOn(), true
	case sideJobMsg:
		if msg.id != m.f.Side.ID {
			break
		}
		running, ended := m.f.Side.take(msg.id, msg.job, msg.tail)
		if running {
			return m, m.tickOn(), true
		} else if !ended {
			break
		}
		// it ended: say how (a stopped job may still have saved a figure, so what reads them is asked again either way)
		ok, why := sideDone(m.f.Side.Kind, msg.tail)
		switch w := sideWords(m.f.Side.Kind); {
		case m.f.Side.Stopped:
			m.note = vocab.Measure.Stopped
		case ok:
			m.note = w.Done
		default:
			m.note = cmp.Or(why, strings.ReplaceAll(w.Failed, "{log}", msg.job.LogPath))
		}
		if m.f.Side.Kind == "routing" { // the offload page and Tuned for this machine read the routing profile
			m.f.OffKey, m.f.Tune.Key = "", ""
			return m, m.loadOffload(), true
		}
		m.f.RecKey, m.f.OffKey, m.f.Tune.Key = "", "", "" // and these read the copy rate
		return m, m.checkMachine(), true
	case gpuStoppedMsg:
		if msg.id == m.f.Side.ID {
			m.f.Side.Stopped, m.note = true, msg.note
		}
	default:
		return m, nil, false
	}
	return m, nil, true
}

func (m Model) doSide(a action) (Model, tea.Cmd) {
	if a.do == "sidestop" {
		return m, m.stopGPUJob(m.f.Side.ID, vocab.Measure.Stopped)
	}
	return m, m.startSide(strings.TrimPrefix(a.do, "measure"))
}

// sideActions are the measure rows of a page: the action with what it does, or while it runs its progress and Stop.
func sideActions(f Facts, kind string) []action {
	w := sideWords(kind)
	if f.siding() && f.Side.Kind == kind {
		return []action{note("  " + f.Spin + " " + w.Running + "…   " + elapsed(f.Side.Job)), {"[ " + vocab.Measure.Stop + " ]", "sidestop", ""}}
	}
	return []action{{"[ " + w.Label + " ] " + w.Helper, "measure" + kind, ""}}
}
