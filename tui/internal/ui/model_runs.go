package ui

// The commands that act on runs: start, follow, stop, save, open, compare, delete and emit.

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"

	tea "charm.land/bubbletea/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// startStep5 starts one of step 5's jobs (compare, or time each role) in the step's one job slot.
func (m Model) startStep5(timeOnly bool) tea.Cmd {
	run := m.f.Run
	provider := m.f.runProvider()
	return func() tea.Msg {
		if run == nil {
			return noteMsg("No run to compare kernels for.")
		}
		dir, err := m.client.RunDir(run.ID)
		if err != nil {
			return noteMsg(err.Error())
		}
		argv := m.client.CompareArgv(dir)
		if timeOnly {
			argv = m.client.RoleTimeArgv(dir, provider)
		}
		if _, err := m.store.Start(compareID(run.ID), m.client.Repo, argv); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return compareStartedMsg(run.ID)
	}
}

// follow reloads the run the checklist is about when it changed under the model, the chip or the runs list.
func (m *Model) follow() tea.Cmd {
	if m.runSet {
		return nil
	}
	id := m.newestRun()
	if id == m.runID() {
		return nil
	}
	m.f.Run, m.f.Job, m.f.Tail = nil, nil, nil
	if id == "" {
		return nil
	}
	return m.loadRun(id)
}

func (m Model) startRun(provider string) tea.Cmd {
	path, target, runs, layout := m.f.Path, m.targetID(), m.f.Runs, m.f.layout()
	if m.f.GpuCount < 2 {
		layout = "" // one GPU: the engine runs as it always has
	}
	store, batch, roleTime, workload := m.store, "", m.f.roleTimeArg(), m.f.workload()
	settings, repeats := m.f.settingsArgv(), m.f.repeats()
	if provider != m.f.Engine { // the settings are the chosen engine's
		settings = nil
	}
	trace, server := "", ""
	if workload == "replay" {
		if s := m.f.replayShape(); s != nil {
			trace = s.Name
		}
		server = m.f.ReplayArgv
	}
	if b := m.f.batch(); b > 1 && workload == "decode" { // batch is decode's: a prefill reads one prompt
		batch = strconv.Itoa(b) // batch 1 is always timed beside it
	}
	return func() tea.Msg {
		if live := gpuJobs(store); len(live) > 0 { // two measurements on one GPU would both be wrong
			return noteMsg("Run " + live[0].ID + " is still going. Stop it or wait, then press Run.")
		}
		if path == "" || target == "" {
			return noteMsg("Pick a model in step 1 and a chip in step 2 first.")
		}
		id := seam.NextName(runs, RunStem(path, target))
		dir, err := m.client.RunDir(id)
		if err != nil {
			return noteMsg(err.Error())
		}
		if abs, err := filepath.Abs(path); err == nil { // the pipeline runs in the checkout, not here
			path = abs
		}
		argv := m.client.PipelineArgv(seam.Pipeline{Model: path, RunDir: dir, Target: target, Workload: workload, Measure: "auto", Provider: provider,
			Layout: layout, Analyze: true, Batch: batch, RoleTime: roleTime, ReplayTrace: trace, ServerArgv: server,
			Settings: settings, Repeats: repeats})
		if _, err := m.store.Start(id, m.client.Repo, argv); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return startedMsg(id)
	}
}

// startMtp starts Measure MTP as a job: one engine start per draft count, on a free GPU.
func (m Model) startMtp() tea.Cmd {
	o, path, target, store := m.f.Recommend.MtpMeasure, m.f.Path, m.targetID(), m.store
	if o == nil || o.Blocked != nil || m.f.measuringMtp() {
		return nil
	}
	id := m.f.mtpID()
	return func() tea.Msg {
		if live := gpuJobs(store); len(live) > 0 { // two measurements on one GPU would both be wrong
			return noteMsg(live[0].ID + " is still going. Stop it or wait, then measure MTP.")
		}
		dir, err := m.client.RunDir(id)
		if err != nil {
			return noteMsg(err.Error())
		}
		if abs, err := filepath.Abs(path); err == nil {
			path = abs
		}
		if _, err := store.Start(id, m.client.Repo, m.f.mtpArgv(m.client, path, target, dir)); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return mtpJobMsg{&jobs.Job{ID: id, Alive: true}, true}
	}
}

func (m Model) stopRun() tea.Cmd {
	id, alive := m.runID(), m.f.alive()
	if m.f.comparing() {
		return func() tea.Msg {
			if _, err := m.store.Stop(compareID(id)); err != nil {
				return noteMsg("Stop failed: " + err.Error())
			}
			return noteMsg("Sent SIGTERM to the kernel comparison. Roles that finished keep their result.")
		}
	}
	return func() tea.Msg {
		if !alive {
			return noteMsg("No run is going from here.")
		}
		if _, err := m.store.Stop(id); err != nil {
			return noteMsg("Stop failed: " + err.Error())
		}
		return stoppedMsg{id, "Sent SIGTERM to " + id + ". The run folder keeps the stages that finished."}
	}
}

// saveRun exports the run on screen into the saved-runs folder. Python copies; this only asks.
func (m Model) saveRun() tea.Cmd {
	id := m.runID()
	return func() tea.Msg {
		s, _, err := m.client.Save(id)
		if err != nil {
			return savedMsg{id, "", err}
		}
		return savedMsg{id, s.Dir, nil}
	}
}

// openSaved reads a saved run to show its results, read-only.
func (m Model) openSaved(id string) tea.Cmd {
	return func() tea.Msg { r, err := m.client.ShowSaved(id); return savedRunMsg{r, err} }
}

// compareSaved sets two saved runs head to head; Python decides how far they compare and words the result.
func (m Model) compareSaved(before, after string) tea.Cmd {
	return func() tea.Msg {
		c, err := m.client.CompareSaved(before, after)
		if err != nil {
			return comparedMsg{Before: before, After: after, Err: err.Error()}
		}
		return comparedMsg{Before: before, After: after, Head: c.Head, Lines: c.Lines, Plain: c.Plain, PartsNote: c.PartsNote}
	}
}

// deleteSaved removes a saved run, after the second press.
func (m Model) deleteSaved(id string) tea.Cmd {
	return func() tea.Msg {
		if _, err := m.client.DeleteSaved(id); err != nil {
			return noteMsg("Delete failed: " + err.Error())
		}
		return deletedMsg("saved:" + id)
	}
}

// cleanWork deletes the unsaved runs from earlier sessions; never one of this session, never a live one.
func (m Model) cleanWork() tea.Cmd {
	keep := append([]string{}, m.session...)
	if m.f.Run != nil {
		keep = append(keep, m.f.Run.ID)
	}
	before := m.started
	return func() tea.Msg { _, err := m.client.CleanWork(before, keep); return cleanedMsg{err} }
}

func (m Model) deleteRun(id string) tea.Cmd {
	return func() tea.Msg {
		if job, err := m.store.Status(id); err == nil && job.Alive {
			return noteMsg("Stop run " + id + " before deleting it.")
		}
		if _, err := m.client.Delete(id); err != nil {
			return noteMsg("Delete failed: " + err.Error())
		}
		return deletedMsg(id)
	}
}

// runFolder is where the run on screen lives: the saved-runs folder for a saved run, else the work area.
func (m Model) runFolder(id string) (string, error) {
	if m.f.ReadOnly {
		return filepath.Join(m.client.Saved, id), nil
	}
	return m.client.RunDir(id)
}

// openReport hands report.html to the desktop.
func (m Model) openReport() tea.Cmd {
	run := m.f.Run
	dir, err := m.runFolder(run.ID)
	if err != nil {
		return func() tea.Msg { return noteMsg(err.Error()) }
	}
	return openPath(filepath.Join(dir, *run.Report), "the report")
}

// openGameplan hands gameplan.md to the desktop, as openReport does for report.html.
func (m Model) openGameplan() tea.Cmd {
	if m.f.Plan == nil {
		return func() tea.Msg { return noteMsg("No gameplan yet. Emit on Setup writes one.") }
	}
	return openPath(filepath.Join(m.f.Plan.Dir, "gameplan.md"), "the gameplan")
}

// openPath hands a file to the desktop opener. This is the one place the TUI runs something other than Python.
func openPath(path, what string) tea.Cmd {
	return func() tea.Msg {
		opener := "xdg-open"
		if runtime.GOOS == "darwin" {
			opener = "open"
		}
		if err := exec.Command(opener, path).Start(); err != nil {
			return noteMsg("Could not open " + what + ": " + err.Error())
		}
		return noteMsg("Opened " + path)
	}
}

// emitPlan asks Python to write the gameplan for the run on screen (screen emit): Python writes, Go shows.
func (m Model) emitPlan() tea.Cmd {
	run, saved := m.f.Run, m.f.ReadOnly
	return func() tea.Msg {
		if run == nil {
			return emitMsg{nil, fmt.Errorf("no run on screen")}
		}
		var e *seam.Emitted
		var err error
		if saved {
			e, _, err = m.client.EmitSaved(run.ID)
		} else {
			e, _, err = m.client.Emit(run.ID)
		}
		return emitMsg{e, err}
	}
}

// followMeasured reloads the speed limit once when the run on screen measured this chip's memory speed and the
// limit Setup shows does not use it yet: one number for the chip line, the limit and the run.
func (m *Model) followMeasured() tea.Cmd {
	r := m.f.Run
	if r == nil || m.f.ReadOnly || m.f.Profile == nil || !m.f.thisChip() {
		return nil
	}
	rc := r.Results.Ceiling
	if rc.BandwidthSource == nil || rc.PeakBandwidthGBs == nil || r.TargetID != m.targetID() {
		return nil
	}
	if c := m.f.Ceiling; c != nil && c.PeakBandwidthGBs == *rc.PeakBandwidthGBs {
		return nil
	}
	key := r.ID + " " + strconv.FormatFloat(*rc.PeakBandwidthGBs, 'f', 3, 64)
	if m.ceilAsked == key || m.f.CeilBusy {
		return nil
	}
	m.ceilAsked, m.f.CeilBusy = key, true
	return m.loadCeiling()
}
