package ui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// Workload is the fourth Setup line: decode or prefill, picked; session is shown muted until it is built. The pick
// reaches the pipeline as --workload, and a prefill takes no --batch.
func TestWorkloadPicker(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := press(press(press(press(program(s, nil, nil, nil, 80, 24), "j"), "j"), "j"), "enter")
	if m.(Model).cursor != pageWorkload {
		t.Fatalf("enter on Workload opens its picker: %d", m.(Model).cursor)
	}
	if got := m.(Model).f.workload(); got != "decode" {
		t.Fatalf("decode until another is picked: %q", got)
	}
	golden(t, "picker-workload.txt", m.View())
	m = press(press(m, "j"), "enter") // prefill
	if got := m.(Model).f.workload(); got != "prefill" || m.(Model).cursor != pageSetup {
		t.Fatalf("workload %q, page %d", got, m.(Model).cursor)
	}
	if !strings.Contains(m.View(), "prefill") {
		t.Fatal("Setup shows the picked workload")
	}
	golden(t, "setup-prefill.txt", m.View())
	argv := strings.Join(seam.Client{Python: "py"}.PipelineArgv(seam.Pipeline{Model: "m", RunDir: "r", Target: "t", Workload: m.(Model).f.workload()}), " ")
	if !strings.Contains(argv, "--workload prefill") {
		t.Fatalf("argv %v", argv)
	}
}
