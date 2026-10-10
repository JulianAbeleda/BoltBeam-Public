package ui

import (
	"encoding/json"
	"flag"
	"github.com/charmbracelet/bubbles/viewport"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"github.com/muesli/termenv"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

var update = flag.Bool("update", false, "rewrite the golden screens under testdata/screens")

func load(t *testing.T, name string, v any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "expected", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}

func golden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "screens", name)
	if *update {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run with -update to write it)", err)
	}
	if string(want) != got {
		t.Errorf("%s differs from the golden screen:\n%s", name, got)
	}
}

type sample struct {
	targets           seam.Targets
	profile           seam.Profile
	ceiling           seam.Ceiling
	runs              seam.Runs
	planned, measured seam.Run
	job               *jobs.Job
	tail              []string
}

func loadSample(t *testing.T) sample {
	var s sample
	load(t, "targets", &s.targets)
	load(t, "inspect", &s.profile)
	load(t, "ceiling", &s.ceiling)
	load(t, "runs", &s.runs)
	load(t, "run-001", &s.planned)
	load(t, "run-002", &s.measured)
	s.job = &jobs.Job{ID: "qwen3-8b-apple_m3_10c-001", PID: 4242, Alive: true, LogPath: "/home/u/.local/state/boltbeam-tui/qwen3-8b-apple_m3_10c-001.log",
		StartedAt: "2026-10-08T10:00:00Z", Argv: []string{"python3", "-m", "boltbeam.workflow.screen", "pipeline"}}
	s.tail = []string{"=== 2026-10-08T10:00:00Z start python3 -m boltbeam.workflow.screen pipeline", "pipeline steps: 4",
		"pipeline counted: 0,0,0,1,1,1,1", "stage load: start", "stage load: done", "stage autoscan: start", "stage autoscan: done",
		"stage analyze: start", "stage analyze: done", "stage measure_timing: start"}
	return s
}

func press(m tea.Model, k string) tea.Model {
	msg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)}
	switch k {
	case "enter":
		msg = tea.KeyMsg{Type: tea.KeyEnter}
	case "esc":
		msg = tea.KeyMsg{Type: tea.KeyEsc}
	}
	next, _ := m.Update(msg)
	return next
}

// program feeds the model the seam's answers in the order they arrive on a real start, then sizes it.
func program(s sample, run *seam.Run, job *jobs.Job, tail []string, width, height int) tea.Model {
	var m tea.Model = New(seam.Client{}, jobs.Store{}, "/models/Qwen3-8B.gguf", "", 512)
	for _, msg := range []tea.Msg{targetsMsg{&s.targets, nil}, detectMsg{id: "apple_m3_10c", count: 1}, profileMsg{&s.profile, nil},
		ceilingMsg{&s.ceiling, nil}, runsMsg{&s.runs, nil}, enginesMsg{&seam.Engines{}, nil}, providersMsg{"apple_m3_10c", engines()},
		tea.WindowSizeMsg{Width: width, Height: height}} {
		m, _ = m.Update(msg)
	}
	if run != nil {
		m, _ = m.Update(runMsg{run, nil})
	}
	if job != nil {
		m, _ = m.Update(jobMsg{job, tail})
	}
	return m
}

// engines is this Mac's answer to `screen providers`: both engines, llama.cpp without per-role capture.
func engines() *seam.Providers {
	method := "tinygrad-profile-events"
	return &seam.Providers{Providers: []seam.ProviderRow{
		{Provider: "llama.cpp", Available: true, Capture: seam.Capture{Reason: sp("metal-system-trace is not installed")}},
		{Provider: "tinygrad", Available: true, Capture: seam.Capture{Method: &method}}}}
}

// page puts the model on one page, as enter on a Setup row does.
func page(m tea.Model, p int) tea.Model {
	got := m.(Model)
	got.cursor, got.row = p, 0
	return got
}

func facts(m tea.Model) Facts { return m.(Model).f }

// savedSample is the answer of `screen saved` with two runs, newest first.
func savedSample() *seam.Saved {
	var sv seam.Saved
	_ = json.Unmarshal([]byte(`{"root":"/home/u/runs/saved","runs":[
	{"id":"qwen3-8b-apple_m3_10c-002","model_id":"Qwen3-8B","target_id":"apple_m3_10c","provider":"tinygrad","saved_at":"2026-10-09T12:10:00","tok_s":12.6,"pct_of_limit":60.4},
	{"id":"qwen3-8b-apple_m3_10c-001","model_id":"Qwen3-8B","target_id":"apple_m3_10c","provider":"llama.cpp","saved_at":"2026-10-09T11:20:00","tok_s":17.0,"pct_of_limit":81.7}]}`), &sv)
	return &sv
}

// The plain goldens are what NO_COLOR shows: lipgloss strips every colour under the Ascii profile.
func TestScreensPlain(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	empty := New(seam.Client{}, jobs.Store{}, "", "", 512)
	emptyM, _ := empty.Update(targetsMsg{&s.targets, nil})
	emptyM, _ = emptyM.Update(runsMsg{&seam.Runs{Root: "/home/u/runs"}, nil})
	ready := program(s, nil, nil, nil, 80, 24)
	ready, _ = ready.Update(savedListMsg{savedSample(), nil})
	greyed, _ := ready.Update(providersMsg{"apple_m3_10c", &seam.Providers{Providers: []seam.ProviderRow{
		{Provider: "llama.cpp", Reason: sp("llama-bench was not found"), Capture: seam.Capture{Reason: sp("x")}},
		{Provider: "tinygrad", Reason: sp("The tinygrad fork is not at /x."), Capture: seam.Capture{Reason: sp("x")}}}}})
	running := page(program(s, &s.planned, s.job, s.tail, 80, 24), pageRun)
	measured := page(program(s, &s.measured, nil, nil, 80, 24), pageRun)
	saved, _ := page(program(s, nil, nil, nil, 80, 24), pageSaved).Update(savedListMsg{savedSample(), nil})
	editing := press(press(page(emptyM, pageModel), "enter"), "/models/Q")
	grouped, _ := program(s, nil, nil, nil, 80, 24).Update(targetsMsg{chipsJSON(t, madeHere), nil})
	grouped, _ = grouped.Update(detectMsg{id: "apple_m3_10c_16g", count: 1})
	chipShut := press(press(grouped, "j"), "enter")
	engineGreyed := page(greyed, pageEngine)
	old := page(program(s, nil, nil, nil, 80, 24), pageSaved)
	old, _ = old.Update(savedListMsg{savedSample(), nil})
	oldM := old.(Model)
	oldM.f.OldWork = 1
	for name, got := range map[string]string{
		"setup-empty.txt":        emptyM.View(),
		"setup-greyed.txt":       greyed.View(),
		"setup-ready.txt":        ready.View(),
		"run-progress.txt":       running.View(),
		"run-results.txt":        measured.View(),
		"saved-runs.txt":         saved.View(),
		"model-editing.txt":      editing.View(),
		"picker-model.txt":       page(ready, pageModel).View(),
		"picker-engine.txt":      page(ready, pageEngine).View(),
		"picker-chip.txt":        chipShut.View(),
		"picker-engine-none.txt": engineGreyed.View(),
		"picker-batch.txt":       page(ready, pageBatch).View(),
		"saved-runs-clean.txt":   oldM.View(),
		"detail-run.txt":         detail(facts(measured), pageRun),
		"detail-setup.txt":       detail(facts(ready), pageSetup),
	} {
		golden(t, name, got)
	}
}

// One styled capture per screen, with true colour on a dark background: what a terminal shows.
func TestScreensStyled(t *testing.T) {
	lipgloss.SetColorProfile(termenv.TrueColor)
	lipgloss.SetHasDarkBackground(true)
	defer lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	golden(t, "styled/setup-ready.ansi", program(s, nil, nil, nil, 80, 24).View())
	golden(t, "styled/run-progress.ansi", page(program(s, &s.planned, s.job, s.tail, 80, 24), pageRun).View())
	golden(t, "styled/detail-run.ansi", detail(facts(program(s, &s.measured, nil, nil, 80, 24)), pageRun))
}

// Every frame fits 80x24: 24 lines, none wider than 80 cells.
func TestFramesFit80x24(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	saved, _ := page(program(s, nil, nil, nil, 80, 24), pageSaved).Update(savedListMsg{savedSample(), nil})
	for _, m := range []tea.Model{program(s, &s.planned, nil, nil, 80, 24), page(program(s, &s.planned, s.job, s.tail, 80, 24), pageRun),
		page(program(s, nil, nil, nil, 80, 24), pageModel), page(program(s, nil, nil, nil, 80, 24), pageChip),
		page(program(s, nil, nil, nil, 80, 24), pageEngine),
		page(program(s, &s.measured, nil, nil, 80, 24), pageRun), saved} {
		lines := strings.Split(m.View(), "\n")
		if len(lines) != 24 {
			t.Fatalf("%d lines", len(lines))
		}
		for _, l := range lines {
			if w := lipgloss.Width(l); w > 80 {
				t.Fatalf("line is %d wide: %q", w, l)
			}
		}
	}
}

// Setup is one line per choice, Run and Emit; Run is greyed, naming what is missing, until all three are set, and
// Emit is greyed with "Run first" until the run on screen has a per-role table.
func TestSetupHasThreeSectionsAndRun(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	acts := setupActions(facts(program(s, nil, nil, nil, 80, 24)))
	want := []string{"page|Model       Qwen3-8B.gguf · 36 layers · F32/Q4_K/Q6_K\t›", "page|Chip        apple_m3_10c · this Mac · 97.2 GB/s\t›",
		"page|Engine      llama.cpp\t›", "page|Batch       1\t›", "page|Measurement auto\t›", "head|", "analyze|[ Run ]", "|[ Emit ] Run first",
		"page|Saved runs (0)\t›"}
	if got := rows(acts); strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("got\n%s", strings.Join(got, "\n"))
	}
	if acts[0].arg != "4" || acts[1].arg != "5" || acts[2].arg != "6" || acts[3].arg != "7" || acts[4].arg != "8" {
		t.Fatalf("each line opens its picker: %+v", acts[:3])
	}
	none := New(seam.Client{}, jobs.Store{}, "", "", 512)
	got := rows(setupActions(none.f))
	if got[0] != "page|Model       not chosen\t›" || got[6] != "|[ Run ] needs a model, a chip, an engine" {
		t.Fatalf("empty:\n%s", strings.Join(got, "\n"))
	}
	what := facts(program(s, nil, nil, nil, 80, 24))
	what.Target, what.Picked = 0, true
	if v := chipValue(what); !strings.HasPrefix(v, "what if: amd_gfx1100") {
		t.Fatalf("a chip that is not this machine: %q", v)
	}
}

// Enter opens a picker; a pick sets the value and comes back to Setup on that line; esc changes nothing.
func TestPickersOpenPickAndGoBack(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := press(press(press(program(s, nil, nil, nil, 80, 24), "j"), "j"), "enter")
	if m.(Model).cursor != pageEngine {
		t.Fatalf("enter on Engine opens its picker: %d", m.(Model).cursor)
	}
	m = press(press(m, "j"), "esc")
	if got := m.(Model); got.cursor != pageSetup || got.row != 2 || got.f.Engine != "llama.cpp" {
		t.Fatalf("esc: page %d row %d engine %q", got.cursor, got.row, got.f.Engine)
	}
	m = press(press(press(m, "enter"), "j"), "enter")
	if got := m.(Model); got.cursor != pageSetup || got.row != 2 || got.f.Engine != "tinygrad" {
		t.Fatalf("pick: page %d row %d engine %q", got.cursor, got.row, got.f.Engine)
	}
}

func TestKeys(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := program(s, &s.measured, nil, nil, 80, 24)
	got := m.(Model)
	if got.targetID() != "apple_m3_10c" || got.cursor != pageSetup || got.f.Engine != "llama.cpp" {
		t.Fatalf("Setup first, with the detected chip and the first engine here: %q %d %q", got.targetID(), got.cursor, got.f.Engine)
	}
	if acts := got.actions(); acts[got.row].do == "head" {
		t.Fatal("the cursor must not rest on a heading")
	}
	m = page(m, pageEngine)
	for i := 0; i < 40 && m.(Model).actions()[m.(Model).row].arg != "tinygrad"; i++ {
		m = press(m, "j")
	}
	m = press(m, "enter")
	if got := m.(Model); got.f.Engine != "tinygrad" || got.cursor != pageSetup {
		t.Fatalf("engine %q cursor %d", got.f.Engine, got.cursor)
	}
	m = page(m, pageRun)
	m = press(m, "esc")
	if m.(Model).cursor != pageSetup {
		t.Fatal("esc goes back to Setup")
	}
	if _, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("q")}); cmd == nil {
		t.Fatal("q quits")
	}
	if RunStem("/models/Qwen3-8B.gguf", "apple_m3_10c") != "qwen3-8b-apple_m3_10c" {
		t.Fatalf("stem %q", RunStem("/models/Qwen3-8B.gguf", "apple_m3_10c"))
	}
}

// The Run screen offers Save until the run is saved, then says where; a saved run opened to read offers no Save.
func TestRunScreenSaves(t *testing.T) {
	s := loadSample(t)
	f := facts(program(s, &s.measured, nil, nil, 80, 24))
	if !hasLabel(runActions(f), stAccent.Render("[ Save run ]")) {
		t.Fatalf("no Save: %+v", runActions(f))
	}
	f.SavedID, f.SavedDir = f.Run.ID, "/home/u/runs/saved/"+f.Run.ID
	if acts := runActions(f); hasLabel(acts, stAccent.Render("[ Save run ]")) || !strings.Contains(plain(acts[0].label), "Saved to /home/u/runs/saved/") {
		t.Fatalf("after save: %+v", acts)
	}
	f.SavedID, f.ReadOnly = "", true
	if acts := runActions(f); hasLabel(acts, stAccent.Render("[ Save run ]")) || acts[0].label != "Back to saved runs" {
		t.Fatalf("read-only: %+v", acts)
	}
}

// detail renders the open view the way the 80x24 screen does: 21 rows under the header, note and footer.
func detail(f Facts, i int) string {
	v := viewport.New(0, 0)
	return DetailView(f, i, 0, 80, 21, &v)
}

// The measure status from the run (measure_status.json) is what step 4 says when it did not measure: the reason,
// never a silent "planned". While the job runs, the live log wins; once it ends, the record wins.
func TestMeasureStatusWording(t *testing.T) {
	reason, command := "this Mac is apple_m3_10c, not apple_m4_10c", "boltbeam metal-measure --run R"
	out := "output"
	run := &seam.Run{Summary: seam.Summary{ID: "m-001", LatestStage: &out,
		Stages:  []seam.Stage{{Key: "output", Done: true}},
		Blocked: []seam.Need{{Need: "probe_evidence"}, {Need: "timing_trace"}},
		Measure: &seam.MeasureStatus{Status: "skipped", Reason: &reason, Command: &command}}}
	f := Facts{Path: "m.gguf", Run: run}
	if mk, text := measureLine(f); mk != "wait" || !strings.Contains(text, "not measured here: "+reason) {
		t.Fatalf("skipped: %s %q", mk, text)
	}
	run.Measure.Status = "failed"
	if mk, text := measureLine(f); mk != "fail" || !strings.Contains(text, reason) {
		t.Fatalf("failed: %s %q", mk, text)
	}
	run.Measure.Status = "skipped"
	done := map[string]string{"measure": "done"}
	if rows := measureRows(run.Measure, done, false); len(rows) != 1 || rows[0][0] != "crossed" {
		t.Fatalf("a finished skip must show crossed, not the log's done: %v", rows)
	}
	running := map[string]string{"measure_probe": "done", "measure_timing": "running"}
	rows := measureRows(nil, running, true)
	if len(rows) != 2 || rows[0][0] != "pass" || rows[1][0] != "run" || stageWord(rows[1][1]) != "Time the real decode" {
		t.Fatalf("live rows: %v", rows)
	}
}

// Saved runs are listed newest first with model, chip, engine, date and the share of the limit; d deletes on
// the second press.
func TestSavedRunsListAndDeleteAsksTwice(t *testing.T) {
	f := Facts{Saved: savedSample()}
	acts := savedActions(f)
	first := plain(acts[0].label)
	if first != "Qwen3-8B · apple_m3_10c · tinygrad · 2026-10-09 12:10 · 60% of limit" || acts[0].do != "opensaved" {
		t.Fatalf("first row %q", first)
	}
	f.Confirm = "qwen3-8b-apple_m3_10c-002"
	if got := plain(savedActions(f)[0].label); !strings.HasPrefix(got, "Press d again to delete saved run qwen3-8b-apple_m3_10c-002") {
		t.Fatalf("the confirm row is %q", got)
	}
	if !strings.Contains(ansi.Strip(footer(true, true, false, false)), "d delete run") {
		t.Fatal("the footer does not name d on a saved run row")
	}
}

// d, then d again on a saved run calls the seam's delete for that folder; the first d only asks.
func TestSavedRunDeleteCallsTheSeamOnTheSecondD(t *testing.T) {
	dir := t.TempDir()
	logFile := filepath.Join(dir, "calls.txt")
	fake := filepath.Join(dir, "python")
	script := "#!/bin/sh\necho \"$@\" >> " + logFile + "\necho '{\"kind\": \"deleted\"}'\n"
	if err := os.WriteFile(fake, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	saved := filepath.Join(dir, "saved")
	s := loadSample(t)
	var m tea.Model = New(seam.Client{Python: fake, Saved: saved}, jobs.Store{Dir: dir}, "", "", 512)
	m, _ = m.Update(targetsMsg{&s.targets, nil})
	m, _ = m.Update(savedListMsg{savedSample(), nil})
	m = page(m, pageSaved)
	first, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	if cmd != nil || facts(first).Confirm != "qwen3-8b-apple_m3_10c-002" {
		t.Fatalf("the first d asks: confirm %q", facts(first).Confirm)
	}
	second, cmd := first.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	if cmd == nil || facts(second).Confirm != "" {
		t.Fatal("the second d deletes")
	}
	if msg := cmd(); msg != deletedMsg("saved:qwen3-8b-apple_m3_10c-002") {
		t.Fatalf("got %v", msg)
	}
	got, _ := os.ReadFile(logFile)
	want := "-m boltbeam.workflow.screen delete --run " + filepath.Join(saved, "qwen3-8b-apple_m3_10c-002") + " --root " + saved
	if !strings.Contains(string(got), want) {
		t.Fatalf("seam call %q, want %q", got, want)
	}
	if _, cmd := second.Update(deletedMsg("saved:qwen3-8b-apple_m3_10c-002")); cmd == nil {
		t.Fatal("the saved list is read again after a delete")
	}
}

// On start the screen is Setup, even with an unsaved last run; only a run still going opens its Run screen.
func TestStartPageIsSetupUnlessARunIsGoing(t *testing.T) {
	s := loadSample(t)
	done := program(s, &s.measured, &jobs.Job{ID: "x", Alive: false}, nil, 80, 24)
	if done.(Model).cursor != pageSetup {
		t.Fatal("a finished last run starts on Setup")
	}
	going := program(s, &s.planned, s.job, s.tail, 80, 24)
	if going.(Model).cursor != pageRun {
		t.Fatal("a run still going starts on its Run screen")
	}
	again, _ := page(going, pageSetup).Update(jobMsg{s.job, s.tail})
	if again.(Model).cursor != pageSetup {
		t.Fatal("the start page is decided once; later job updates do not move the screen")
	}
}
