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
	s.tail = []string{"=== 2026-10-08T10:00:00Z start python3 -m boltbeam.workflow.screen pipeline", "pipeline steps: 7",
		"stage load: start", "stage load: done", "stage autoscan: start", "stage autoscan: done", "stage analyze: start"}
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
	for _, msg := range []tea.Msg{targetsMsg{&s.targets, nil}, detectMsg{"apple_m3_10c"}, profileMsg{&s.profile, nil},
		ceilingMsg{&s.ceiling, nil}, runsMsg{&s.runs, nil}, tea.WindowSizeMsg{Width: width, Height: height}} {
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

func facts(m tea.Model) Facts { return m.(Model).f }

// The plain goldens are what NO_COLOR shows: lipgloss strips every colour under the Ascii profile.
func TestScreensPlain(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	empty := New(seam.Client{}, jobs.Store{}, "", "", 512)
	emptyM, _ := empty.Update(targetsMsg{&s.targets, nil})
	emptyM, _ = emptyM.Update(runsMsg{&seam.Runs{Root: "/home/u/runs"}, nil})
	planned := program(s, &s.planned, nil, nil, 80, 24)
	measuring := program(s, &s.planned, s.job, s.tail, 80, 24)
	measured := program(s, &s.measured, nil, nil, 80, 24)
	editing := press(press(press(press(empty, "enter"), "enter"), "/models/Q"), "")
	opened := press(press(measured, "enter"), "j")
	for name, got := range map[string]string{
		"checklist-empty.txt":     emptyM.View(),
		"checklist-planned.txt":   planned.View(),
		"checklist-measuring.txt": measuring.View(),
		"checklist-measured.txt":  measured.View(),
		"open-result-80x24.txt":   opened.View(),
		"model-editing.txt":       editing.View(),
		"detail-model.txt":        detail(facts(measured), 0),
		"detail-chip.txt":         detail(facts(measured), 1),
		"detail-limit.txt":        detail(facts(measured), 2),
		"detail-measure.txt":      detail(facts(measuring), 3),
		"detail-result.txt":       detail(facts(measured), 4),
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
	measured := program(s, &s.measured, nil, nil, 80, 24)
	golden(t, "styled/checklist-measured.ansi", measured.View())
	golden(t, "styled/checklist-measuring.ansi", program(s, &s.planned, s.job, s.tail, 80, 24).View())
	golden(t, "styled/detail-result.ansi", detail(facts(measured), 4))
}

// Every frame fits 80x24: 24 lines, none wider than 80 cells.
func TestFramesFit80x24(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	for _, m := range []tea.Model{program(s, &s.planned, nil, nil, 80, 24), program(s, &s.planned, s.job, s.tail, 80, 24),
		press(press(program(s, &s.measured, nil, nil, 80, 24), "enter"), "j")} {
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

func TestStepStates(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	marks := func(m tea.Model) string {
		out := ""
		for _, st := range steps {
			mk, _ := st.line(facts(m))
			out += mk + " "
		}
		return out
	}
	for name, c := range map[string]struct {
		m      tea.Model
		marks  string
		cursor int
	}{
		"planned":   {program(s, &s.planned, nil, nil, 80, 24), "pass pass pass wait open ", 3},
		"measuring": {program(s, &s.planned, s.job, s.tail, 80, 24), "pass pass pass run open ", 3},
		"measured":  {program(s, &s.measured, nil, nil, 80, 24), "pass pass pass pass pass ", 4},
		"failed":    {program(s, &s.planned, &jobs.Job{}, []string{"stage analyze: failed: boom"}, 80, 24), "pass pass pass fail open ", 3},
	} {
		if got := marks(c.m); got != c.marks || c.m.(Model).cursor != c.cursor {
			t.Errorf("%s: marks %q cursor %d", name, got, c.m.(Model).cursor)
		}
	}
	none := New(seam.Client{}, jobs.Store{}, "", "", 512)
	if marks(none) != "open open open open open " || none.mood() != faceSleep {
		t.Fatalf("nothing read: %q %q", marks(none), none.mood())
	}
}

func TestKeys(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := program(s, &s.measured, nil, nil, 80, 24)
	if got := m.(Model); got.targetID() != "apple_m3_10c" || got.mood() != faceHappy {
		t.Fatalf("the detected chip is picked when none is asked for: %q, mood %q", got.targetID(), got.mood())
	}
	m = press(press(press(m, "k"), "k"), "k") // to step 2
	if m.(Model).cursor != 1 {
		t.Fatalf("cursor %d", m.(Model).cursor)
	}
	m = press(m, "enter")
	if !m.(Model).open {
		t.Fatal("enter opens the step")
	}
	if m.(Model).row != m.(Model).f.Target {
		t.Fatal("the chip list opens on the chip in use")
	}
	m = press(press(m, "k"), "enter") // the chip above it
	if got := m.(Model); got.targetID() != "apple_metal" || got.f.Ceiling != nil || !got.chipSet {
		t.Fatalf("enter on a chip row picks it and drops the old speed limit: %q", got.targetID())
	}
	m = press(m, "esc")
	if m.(Model).open || m.(Model).cursor != 1 {
		t.Fatal("esc goes back to the list, on the same step")
	}
	m = press(press(press(press(press(m, "k"), "enter"), "enter"), "/m"), "esc")
	if got := m.(Model); got.f.Editing || got.f.Path != "/models/Qwen3-8B.gguf" {
		t.Fatalf("esc cancels the edit: %q", got.f.Path)
	}
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	if note, ok := cmd().(noteMsg); !ok || !strings.Contains(string(note), "No run is going") {
		t.Fatalf("x with no live run says so, got %v", note)
	}
	if _, cmd = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("q")}); cmd == nil {
		t.Fatal("q quits")
	}
	if RunStem("/models/Qwen3-8B.gguf", "apple_m3_10c") != "qwen3-8b-apple_m3_10c" {
		t.Fatalf("stem %q", RunStem("/models/Qwen3-8B.gguf", "apple_m3_10c"))
	}
}

// detail renders the open view the way the 80x24 screen does: 21 rows under the header, note and footer.
func detail(f Facts, i int) string {
	v := viewport.New(0, 0)
	return DetailView(f, i, 0, 80, 21, &v)
}
