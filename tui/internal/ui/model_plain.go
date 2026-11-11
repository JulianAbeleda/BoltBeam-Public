package ui

// The Results screen's commands: the plain words and a run's plain block through the seam, its sections opening and
// closing, and Run again with this setup. (Save as image with freeze is not offered: freeze v0.2.2 draws ANSI input
// in a proportional font, so the columns and the curve drift.)

import (
	"fmt"
	"maps"
	"os"
	"slices"
	"strconv"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

type wordsMsg struct {
	words *seam.Words
	err   error
}

type plainMsg struct {
	id    string
	plain *seam.Plain
	err   error
}

// loadWords reads the plain vocabulary once: stage, role and part names, the labels, the glossary.
func (m Model) loadWords() tea.Cmd {
	return func() tea.Msg { w, err := m.client.Words(); return wordsMsg{w, err} }
}

// loadPlain asks for the plain block of the run on screen once per finished state: the run's id, saved or not, and
// its last stage. A run still going, or one that failed, has none to show.
func (m *Model) loadPlain() tea.Cmd {
	f := m.f
	if f.Run == nil || f.alive() || f.failedStage() != "" || !(f.outputDone() || f.ReadOnly) {
		return nil
	}
	ask := fmt.Sprintf("%s|%t|%s|%t", f.Run.ID, f.ReadOnly, deref(f.Run.LatestStage), f.Run.Results.Measured)
	if m.plainAsked == ask {
		return nil
	}
	m.plainAsked = ask
	id, saved, client := f.Run.ID, f.ReadOnly, m.client
	return func() tea.Msg { p, err := client.Plain(id, saved); return plainMsg{id, p, err} }
}

// plainAnswer keeps the block for the run it was asked for. A run Python cannot word yet keeps the full page.
func (m Model) plainAnswer(msg plainMsg) Model {
	if msg.id != m.runID() {
		return m
	}
	if msg.err != nil {
		m.f.Plain = nil
		return m
	}
	m.f.Plain = msg.plain
	return m
}

// toggleSection opens or closes the section a key names, and scrolls the page to its heading; ok is false for a key
// that names none here.
func (m Model) toggleSection(msg tea.KeyPressMsg) (Model, bool) {
	if m.cursor != pageRun || !m.f.plainShown() {
		return m, false
	}
	open, k, ok := m.f.Open.toggle(msg)
	if !ok {
		return m, false
	}
	title := sectionTitle(m.f.Plain, k)
	if title == "" {
		return m, false
	}
	m.f.Open = open
	DetailView(m.f, m.cursor, m.row, m.width, m.detailHeight(), &m.view)
	body := wrapText(steps[pageRun].body(m.f, m.width-4), m.width-4)
	for i, l := range strings.Split(body, "\n") {
		if strings.Contains(ansi.Strip(l), " "+title) && strings.ContainsAny(ansi.Strip(l), "▸▾") {
			m.view.SetYOffset(i)
			break
		}
	}
	return m, true
}

// rerun starts a run with the setup the run on screen used: the command its job ran when this machine started it (the
// jobs store keeps it), else what the run folder recorded (model, chip, workload, engine, batch sizes, layout and
// measurement; the engine settings are kept in words only, so Setup's are not added).
func (m Model) rerun() tea.Cmd {
	run, store, runs, client := m.f.Run, m.store, m.f.Runs, m.client
	return func() tea.Msg {
		if run == nil {
			return noteMsg("No run on screen to run again.")
		}
		if live := gpuJobs(store); len(live) > 0 { // two measurements on one GPU would both be wrong
			return noteMsg("Run " + live[0].ID + " is still going. Stop it or wait, then run again.")
		}
		argv, model, target, err := againArgv(client, store, run)
		if err != nil {
			return noteMsg("Run again: " + err.Error())
		}
		id := seam.NextName(runs, RunStem(model, target))
		dir, err := client.RunDir(id)
		if err != nil {
			return noteMsg(err.Error())
		}
		for i := range argv {
			switch {
			case i > 0 && argv[i-1] == "--run":
				argv[i] = dir
			case i > 0 && argv[i-1] == "--id":
				argv[i] = id
			}
		}
		if _, err := store.Start(id, client.Repo, argv); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return startedMsg(id)
	}
}

// againArgv is the pipeline command of a run, with its model and chip: its own job's when the store has it, else one
// built from the run's record.
func againArgv(c seam.Client, store jobs.Store, run *seam.Run) (argv []string, model, target string, err error) {
	if job, err := store.Status(run.ID); err == nil {
		if i := slices.Index(job.Argv, "pipeline"); i >= 0 && i+1 < len(job.Argv) {
			argv = slices.Clone(job.Argv)
			if t := slices.Index(argv, "--target"); t >= 0 && t+1 < len(argv) {
				return argv, argv[i+1], argv[t+1], nil
			}
		}
	}
	model, target = run.ModelPath, run.TargetID
	if model == "" || target == "" {
		return nil, "", "", fmt.Errorf("run %s does not record its model and chip", run.ID)
	}
	if _, err := os.Stat(model); err != nil {
		return nil, "", "", fmt.Errorf("the model of run %s is not on this machine: %s", run.ID, model)
	}
	if run.Workload == "replay" {
		return nil, "", "", fmt.Errorf("a replay needs its server; pick the trace and the server in Setup")
	}
	p := seam.Pipeline{Model: model, Target: target, Workload: run.Workload, Measure: "auto", Analyze: true, RunDir: "-"}
	if ms := run.Measure; ms != nil {
		p.Provider = deref(ms.Provider)
		batches := []string{}
		for _, b := range ms.Batches {
			if b > 1 {
				batches = append(batches, strconv.Itoa(b))
			}
		}
		p.Batch = strings.Join(batches, ",")
		if c := ms.Capture; c != nil && c.Measurement != nil {
			p.RoleTime = map[string]string{"in_model": "in-model", "generic": "generic"}[c.Measurement.Choice]
		}
	}
	if l := run.Results.Loss.Layout; l != nil && l.Layout != "one" {
		p.Layout = l.Layout
	}
	argv = c.PipelineArgv(p)
	for _, name := range slices.Sorted(maps.Keys(run.EngineEnv)) { // the engine's environment is part of the run
		argv = append(argv, "--engine-env", name+"="+run.EngineEnv[name])
	}
	return argv, model, target, nil
}
