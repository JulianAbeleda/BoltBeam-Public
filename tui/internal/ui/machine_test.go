package ui

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// recordedDoctor is `boltbeam doctor --json` as this Mac printed it (an Apple M5: llama.cpp, the Ollama and Strata
// forks not here, the screen binary stale, vLLM and TensorRT-LLM n/a), home paths replaced. Edited after the recording:
// the PrismML fork's entry was cut from the llama.cpp row's detail when BoltBeam dropped PrismML support (2026-11-11).
func recordedDoctor(t *testing.T) *seam.Doctor {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "recorded", "doctor-mac.json"))
	if err != nil {
		t.Fatal(err)
	}
	var d seam.Doctor
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatal(err)
	}
	return &d
}

// Check this machine is Setup's first line: checking while doctor runs (after the engine scan), then how many things
// to set up and which. A gap never greys Run.
func TestMachineCheckIsSetupsFirstLine(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	s := loadSample(t)
	m := program(s, nil, nil, nil, 80, 24)
	if !facts(m).DoctorBusy {
		t.Fatal("the engine scan's answer starts the machine check")
	}
	if got := rows(setupActions(facts(m)))[0]; got != "page|Machine            checking this Mac…\t›" {
		t.Fatalf("while it runs: %q", got)
	}
	m, _ = m.Update(doctorMsg{recordedDoctor(t), nil})
	f := facts(m)
	acts := rows(setupActions(f))
	// the stale screen binary is no blocker: ready, with it counted as optional
	if acts[0] != "page|Machine            ready to run on this Mac · 1 optional\t›" || f.DoctorBusy {
		t.Fatalf("answered: %q", acts[0])
	}
	if !strings.HasPrefix(runRow(acts), "analyze|[ Run ]") {
		t.Fatalf("a gap never blocks Run:\n%s", strings.Join(acts, "\n"))
	}
	golden(t, "setup-machine.txt", frame(m))
	f.Doctor = &seam.Doctor{Rows: []seam.DoctorRow{{Item: "python", Found: ptr(true)}, {Item: "vllm"}}}
	if got := plain(doctorValue(f)); got != "ready to run on this Mac" {
		t.Fatalf("no gaps (n/a is none): %q", got)
	}
	f.Doctor.Rows = append(f.Doctor.Rows, seam.DoctorRow{Item: "chip profile", Found: ptr(false), Blocks: true})
	if got := plain(doctorValue(f)); got != "1 thing to set up before a run: chip profile" {
		t.Fatalf("only a blocker is a thing to set up: %q", got)
	}
	failed, _ := m.Update(doctorMsg{nil, errors.New("python -m boltbeam.cli doctor --json: exit status 1")})
	if got := plain(doctorValue(facts(failed))); got != "not checked: python -m boltbeam.cli doctor --json: exit status 1" {
		t.Fatalf("a failed check says so: %q", got)
	}
}

func ptr[T any](v T) *T { return &v }

// runRow is Setup's [ Run ] row among rows(), whatever its helper says.
func runRow(acts []string) string {
	for _, a := range acts {
		if strings.HasPrefix(a, "analyze|[ Run ]") || strings.HasPrefix(a, "|[ Run ]") {
			return a
		}
	}
	return ""
}

// The page lists every check with its status and what was found, and each gap's one fix under it; Check again asks
// once at a time, and esc goes back to the Machine line.
func TestMachinePageListsEveryCheck(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	s := loadSample(t)
	m, _ := program(s, nil, nil, nil, 100, 60).Update(doctorMsg{recordedDoctor(t), nil})
	m = press(m, "enter")
	if m.(Model).cursor != pageDoctor {
		t.Fatalf("enter on Machine opens its page: %d", m.(Model).cursor)
	}
	got := plain(frame(m))
	// the verdict first, then the groups in order; a fork not here is one dimmed line, never a gap
	order := []string{"Ready to run on this Mac · 1 optional.", "This GPU", "✓ GPU · Apple M5", "Engines you can run here",
		"✓ llama.cpp · upstream:", "forks not found: Ollama, Strata", "Measuring tools", "✓ capture tool", "Not on this Mac",
		"◇ vLLM · BoltBeam runs vLLM on CUDA or AMD GPUs", "BoltBeam itself", "✓ python · 3.13.13",
		"✗ boltbeam-tui · not found", "→ boltbeam tui   (rebuilds it)"}
	at := 0
	for _, want := range order {
		i := strings.Index(got[at:], want)
		if i < 0 {
			t.Fatalf("missing %q after byte %d:\n%s", want, at, got)
		}
		at += i
	}
	if strings.Contains(got, "boltbeam doctor") || strings.Contains(got, "ollama · not found") {
		t.Fatalf("plain intro, and no missing fork as a gap:\n%s", got)
	}
	golden(t, "machine-check.txt", frame(m))
	again, cmd := m.(Model).do(action{do: "doctor"})
	if cmd == nil || !again.f.DoctorBusy || rows(doctorActions(again.f))[0] != "|checking…" {
		t.Fatalf("Check again asks doctor once: %v", rows(doctorActions(again.f)))
	}
	if _, cmd := again.do(action{do: "doctor"}); cmd != nil {
		t.Fatal("a second press while it runs asks nothing")
	}
	back := press(tea.Model(again), "esc").(Model)
	if back.cursor != pageSetup || back.row != 0 {
		t.Fatalf("esc: page %d row %d", back.cursor, back.row)
	}
}
