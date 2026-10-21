package ui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// The results page prints the session table's own cells and its lines, and nothing when the run has none.
func TestSessionBodyPrintsTheCells(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	if sessionBody(nil) != "" || sessionBody(&seam.Session{}) != "" {
		t.Fatal("no session table, no section")
	}
	s := &seam.Session{Title: "The session: one turn at each depth", Columns: []string{"depth", "turn ms"},
		Rows:       []seam.SessionRow{{Cells: []string{"1,024", "125.1"}}, {Cells: []string{"197,084", "607.3"}}},
		MaxContext: "f16 197,660 (memory binds)", Compaction: "520.5 ms"}
	got := sessionBody(s)
	for _, want := range []string{"The session", "DEPTH", "197,084", "607.3", "f16 197,660", "520.5 ms"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in\n%s", want, got)
		}
	}
}

func TestSessionHeadlineNamesPromptThroughput(t *testing.T) {
	focus, ceiling := 2048, 16000.0
	var res seam.Results
	// Decode has its own column in the session table; the turn rate is prompt throughput.
	load(t, "results-isolated", &res)
	res.Loss.Session = &seam.Session{Focus: &focus}
	res.Ceiling.TokS = &ceiling
	if len(res.Loss.Runtimes) == 0 {
		t.Fatal("fixture needs a measured runtime")
	}
	if got := sessionHeadline(res); !strings.Contains(got, "prompt tokens per second") || !strings.Contains(got, "depth 2048") {
		t.Fatal(got)
	}
}
