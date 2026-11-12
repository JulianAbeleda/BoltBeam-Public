package ui

// A measurement that holds the GPU, started from an action and read as the other jobs are: Tuned for this machine's
// sweeps, Measure routing and Measure again. One start (a free GPU first), one read (status, then the whole log), one
// account of the job this screen saw running and its end, and one Stop that is remembered.

import (
	"strings"

	tea "charm.land/bubbletea/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
)

// GPUJob is the last read of such a job: the job, its log, the start this screen saw running (only its end is read) and
// the start whose end was read; Stopped says Stop was sent, so the end is not reported as a failure.
type GPUJob struct {
	Job         *jobs.Job
	Tail        []string
	Seen, Ended string
	Stopped     bool
}

type gpuStoppedMsg struct{ id, note string }

// startGPUJob starts argv as job id on a free GPU (two measurements on one GPU would both be wrong); started is the
// message that says it began.
func (m Model) startGPUJob(id string, argv []string, started func(*jobs.Job) tea.Msg) tea.Cmd {
	store, repo := m.store, m.client.Repo
	return func() tea.Msg {
		if live := gpuJobs(store); len(live) > 0 {
			return noteMsg(strings.ReplaceAll(vocab.Measure.Busy, "{id}", live[0].ID))
		}
		job, err := store.Start(id, repo, argv)
		if err != nil {
			return noteMsg(strings.ReplaceAll(vocab.Measure.StartFailed, "{why}", err.Error()))
		}
		return started(&job)
	}
}

// readGPUJob reads job id and its log; read is the message that carries them (a nil job: it never ran from here).
func (m Model) readGPUJob(id string, read func(*jobs.Job, []string) tea.Msg) tea.Cmd {
	store := m.store
	return func() tea.Msg {
		job, err := store.Status(id)
		if err != nil {
			return read(nil, nil)
		}
		lines, _ := store.Tail(id, 0)
		return read(&job, lines)
	}
}

// stopGPUJob sends Stop to job id; said is what the page says once it is sent.
func (m Model) stopGPUJob(id, said string) tea.Cmd {
	store := m.store
	return func() tea.Msg {
		if _, err := store.Stop(id); err != nil {
			return noteMsg(strings.ReplaceAll(vocab.Measure.StopFailed, "{why}", err.Error()))
		}
		return gpuStoppedMsg{id, said}
	}
}

// take keeps a read of job id and says whether it is running and whether the job this screen saw running has ended.
func (g *GPUJob) take(id string, job *jobs.Job, tail []string) (running, ended bool) {
	g.Job, g.Tail = job, tail
	switch {
	case job != nil && job.Alive && job.ID == id:
		g.Seen = job.StartedAt
		return true, false
	case job != nil && job.StartedAt == g.Seen && g.Ended != job.StartedAt:
		g.Ended = job.StartedAt
		return false, true
	}
	return false, false
}

// tickOn starts the clock when it is not ticking.
func (m *Model) tickOn() tea.Cmd {
	if m.ticking {
		return nil
	}
	m.ticking = true
	return tick()
}
