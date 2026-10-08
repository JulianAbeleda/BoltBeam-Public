package ui

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
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
	s.tail = []string{"=== 2026-10-08T10:00:00Z start python3 -m boltbeam.workflow.screen pipeline",
		"stage load: start", "stage load: done", "stage autoscan: start", "stage autoscan: done", "stage analyze: start"}
	return s
}

func m3(s sample) int {
	for i, tg := range s.targets.Targets {
		if tg.ID == "apple_m3_10c" {
			return i
		}
	}
	return 0
}

// program renders the whole frame: open the run screen, move to the second row, open it, toggle the mode, resize.
func program(s sample, technical bool, width, height int) string {
	m := New(seam.Client{}, jobs.Store{}, "/models/Qwen3-8B.gguf", "apple_m3_10c", 512)
	next, _ := m.Update(targetsMsg{&s.targets, nil})
	next, _ = next.Update(runsMsg{&s.runs, nil})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("3")})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("j")})
	next, _ = next.Update(runMsg{&s.measured, nil})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("4")})
	if technical {
		next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("t")})
	}
	next, _ = next.Update(tea.WindowSizeMsg{Width: width, Height: height})
	return next.View()
}

// The plain goldens are what NO_COLOR shows: lipgloss strips every colour under the Ascii profile.
func TestScreensPlain(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	model := ModelScreen{Path: "/models/Qwen3-8B.gguf", Targets: &s.targets, Target: m3(s), Profile: &s.profile}
	empty := ModelScreen{Targets: &s.targets, Target: m3(s)}
	editing := ModelScreen{Path: "/models/Qwen3-8B.gguf", Input: "/models/Qwen3-8", Editing: true, Targets: &s.targets, Target: m3(s)}
	running := RunScreen{Runs: &s.runs, Cursor: 0, Run: &s.planned, Job: s.job, Tail: s.tail, Spin: "⠋"}
	finished := RunScreen{Runs: &s.runs, Cursor: 1, Run: &s.measured}
	for name, got := range map[string]string{
		"model-plain.txt":                ModelView(model, false, 100),
		"model-technical.txt":            ModelView(model, true, 140),
		"model-empty.txt":                ModelView(empty, false, 80),
		"model-editing.txt":              ModelView(editing, false, 80),
		"ceiling-plain.txt":              CeilingView(&s.ceiling, false, false, 100),
		"ceiling-technical.txt":          CeilingView(&s.ceiling, false, true, 120),
		"ceiling-none.txt":               CeilingView(nil, false, false, 80),
		"run-running-plain.txt":          RunView(running, false, 100),
		"run-running-technical.txt":      RunView(running, true, 140),
		"run-finished-plain.txt":         RunView(finished, false, 100),
		"run-none.txt":                   RunView(RunScreen{Runs: &seam.Runs{Root: "/home/u/runs"}}, false, 80),
		"results-measured-plain.txt":     ResultsView(&s.measured, false, 120),
		"results-measured-technical.txt": ResultsView(&s.measured, true, 140),
		"results-planned-plain.txt":      ResultsView(&s.planned, false, 100),
		"results-none.txt":               ResultsView(nil, false, 80),
		"program-results-technical.txt":  program(s, true, 120, 60),
		"program-results-80x24.txt":      program(s, false, 80, 24),
	} {
		golden(t, name, got)
	}
}

// One styled capture per shape, with true colour on a dark background: what a terminal shows.
func TestScreensStyled(t *testing.T) {
	lipgloss.SetColorProfile(termenv.TrueColor)
	lipgloss.SetHasDarkBackground(true)
	defer lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	golden(t, "styled/ceiling-plain.ansi", CeilingView(&s.ceiling, false, false, 100))
	golden(t, "styled/results-measured-plain.ansi", ResultsView(&s.measured, false, 120))
	golden(t, "styled/program-results-80x24.ansi", program(s, false, 80, 24))
	golden(t, "styled/run-running-plain.ansi", RunView(RunScreen{Runs: &s.runs, Run: &s.planned, Job: s.job, Tail: s.tail, Spin: "⠋"}, false, 100))
}

func TestModelNavigation(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := New(seam.Client{}, jobs.Store{}, "", "", 512)
	if m.mood() != faceSleep {
		t.Fatalf("nothing read and no runs sleeps, got %q", m.mood())
	}
	next, _ := m.Update(targetsMsg{&s.targets, nil})
	if next.(Model).targetID() != "amd_gfx1100" {
		t.Fatalf("with no chip asked for, the first chip with a speed limit is picked, got %q", next.(Model).targetID())
	}
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("]")})
	next, _ = next.Update(runsMsg{&s.runs, nil})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("3")})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("j")})
	next, _ = next.Update(runMsg{&s.measured, nil})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("t")})
	got := next.(Model)
	if got.screen != screenRun || !got.technical || got.jobID != "qwen3-8b-apple_m3_10c-002" || got.mood() != faceHappy || got.targetID() != "nvidia_sm89" {
		t.Fatalf("model state wrong: screen %d technical %t job %q mood %q chip %q", got.screen, got.technical, got.jobID, got.mood(), got.targetID())
	}
	next, _ = next.Update(runMsg{&s.planned, nil})
	if next.(Model).mood() != faceWaiting {
		t.Fatal("a run blocked on evidence waits")
	}
	next, _ = next.Update(jobMsg{s.job, s.tail})
	if next.(Model).mood() != faceBusy {
		t.Fatal("a live pipeline is busy")
	}
	next, _ = next.Update(jobMsg{&jobs.Job{Alive: false}, []string{"stage analyze: failed: boom"}})
	if next.(Model).mood() != faceWorried {
		t.Fatal("a failed stage worries")
	}
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("e")})
	if next.(Model).screen != screenRun || next.(Model).editing {
		t.Fatal("e edits only on the model screen")
	}
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("1")})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("e")})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("/m")})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if next.(Model).editing || next.(Model).modelPath != "" {
		t.Fatal("esc cancels the edit")
	}
	if RunStem("/models/Qwen3-8B.gguf", "apple_m3_10c") != "qwen3-8b-apple_m3_10c" {
		t.Fatalf("stem %q", RunStem("/models/Qwen3-8B.gguf", "apple_m3_10c"))
	}
}
