package ui

// The commands that read facts for the model: engines, chips, models, runs, jobs and comparisons.

import (
	"os"
	"path/filepath"

	tea "github.com/charmbracelet/bubbletea"
)

// scanEngines looks for every engine in the usual folders and saves what was found (Python: screen engines). It
// reuses cached discovery at boot; Scan again forces discovery. Opening Engine never starts a scan.
func (m Model) scanEngines(cached bool) tea.Cmd {
	return func() tea.Msg { e, _, err := m.client.Engines(cached); return enginesMsg{e, err} }
}

// loadLatest asks Python where each engine's build stands against upstream's newest release. It runs after the engine
// scan (at boot, on Scan again) and after an Upgrade; Python looks upstream at most once per engine_latest_max_age_h.
// Drawing only reads the answer.
func (m Model) loadLatest() tea.Cmd {
	return func() tea.Msg { rows, err := m.client.EngineRows("--latest"); return latestMsg{rows, err, ""} }
}

// loadProviders asks Python which runtimes can measure the chip on this machine.
func (m Model) loadProviders(target string) tea.Cmd {
	path := m.f.Path
	return func() tea.Msg {
		p, err := m.client.Providers(target, path)
		if err != nil {
			return providersMsg{target, nil}
		}
		return providersMsg{target, p}
	}
}

// loadOthers reads the model's limit on every registered chip, for step 3's read-only "On other chips".
func (m Model) loadOthers() tea.Cmd {
	path := m.f.Path
	return func() tea.Msg {
		o, err := m.client.Ceilings(path)
		if err != nil {
			return othersMsg{path, nil}
		}
		return othersMsg{path, o}
	}
}

func (m Model) loadTargets() tea.Cmd {
	return func() tea.Msg { t, _, err := m.client.Chips(); return targetsMsg{t, err} }
}

func (m Model) detect() tea.Cmd {
	return func() tea.Msg {
		d, err := m.client.Detect()
		if err == nil && d.Profile != nil && d.Profile.Status == "new" {
			name := "this GPU"
			if d.Name != nil {
				name = *d.Name
			}
			return detectMsg{count: d.GpuCount, newChip: name}
		}
		if err != nil || d.TargetID == nil {
			if err == nil {
				return detectMsg{count: d.GpuCount}
			}
			return detectMsg{}
		}
		driver := ""
		if d.DriverVersion != nil {
			driver = *d.DriverVersion
		}
		multi := ""
		if d.MultiGpu != nil {
			multi = *d.MultiGpu
		}
		return detectMsg{id: *d.TargetID, driver: driver, multi: multi, count: d.GpuCount}
	}
}

// autoscan matches this machine's GPU to a chip profile, or measures and saves a new one (Python decides which).
func (m Model) autoscan(remeasure bool) tea.Cmd {
	return func() tea.Msg { s, _, err := m.client.Autoscan(remeasure); return scanMsg{s, err} }
}

// findFiles discovers models in the shared store and beside the current selection.
func (m Model) findFiles() tea.Cmd {
	roots := []string{}
	if m.f.Path != "" {
		roots = append(roots, filepath.Dir(m.f.Path))
	}
	if home, err := os.UserHomeDir(); err == nil {
		roots = append(roots, filepath.Join(home, "models"))
	}
	return func() tea.Msg { return filesMsg(discoverModels(roots)) }
}

func (m Model) loadRuns() tea.Cmd {
	return func() tea.Msg { r, _, err := m.client.List(); return runsMsg{r, err} }
}

func (m Model) loadRun(id string) tea.Cmd {
	return func() tea.Msg { r, _, err := m.client.Show(id); return runMsg{r, err} }
}

func (m Model) loadJob(id string) tea.Cmd {
	return func() tea.Msg {
		job, err := m.store.Status(id)
		if err != nil {
			return jobMsg{nil, nil}
		}
		lines, _ := m.store.Tail(id, 200)
		return jobMsg{&job, lines}
	}
}

// loadReady asks Python whether this machine can compare kernels for the run.
func (m Model) loadReady(id string) tea.Cmd {
	return func() tea.Msg {
		r, err := m.client.CompareReady(id)
		if err != nil {
			return readyMsg{id, nil}
		}
		return readyMsg{id, r}
	}
}

// loadCompare reads the compare job's state and log tail. No job file means no comparison ever ran from here.
func (m Model) loadCompare(id string) tea.Cmd {
	return func() tea.Msg {
		job, err := m.store.Status(compareID(id))
		if err != nil {
			return compareJobMsg{nil, nil}
		}
		lines, _ := m.store.Tail(compareID(id), 200)
		return compareJobMsg{&job, lines}
	}
}

func (m Model) inspect() tea.Cmd {
	path := m.f.Path
	return func() tea.Msg { p, _, err := m.client.Inspect(path); return profileMsg{p, err} }
}

func (m Model) loadCeiling() tea.Cmd {
	path, target, context, here := m.f.Path, m.targetID(), m.context, m.f.thisChip()
	return func() tea.Msg { c, _, err := m.client.Ceiling(path, target, context, here); return ceilingMsg{c, err} }
}

// loadMtp reads Measure MTP's job for the model and chip; no job file means it never ran from here.
func (m Model) loadMtp() tea.Cmd {
	if m.f.recommendKey() == "" {
		return nil
	}
	id, store := m.f.mtpID(), m.store
	return func() tea.Msg {
		if job, err := store.Status(id); err == nil {
			return mtpJobMsg{&job, false}
		}
		return mtpJobMsg{nil, false}
	}
}

// deleteRun removes a finished run folder; a run whose job is alive is refused here, since only Go knows the job.
// loadSaved lists the saved runs.
func (m Model) loadSaved() tea.Cmd {
	return func() tea.Msg { s, _, err := m.client.SavedRuns(); return savedListMsg{s, err} }
}

func (m Model) readModel() (Model, tea.Cmd) {
	m.f.Reading, m.f.Profile, m.f.ModelErr, m.f.Ceiling, m.f.CeilErr = true, nil, "", nil, ""
	m.input.SetValue(m.f.Path)
	return m, tea.Batch(m.inspect(), m.follow())
}
