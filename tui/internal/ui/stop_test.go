package ui

import (
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
)

// A 70B on two RTX 3090s (2026-11-09): a finished run was saved, a second Run was started and stopped, and Setup lost
// the finished run: Last run was gone and Emit said "Run first". Stopping the new run puts the finished one back.
func TestStoppingANewRunGivesBackTheFinishedOne(t *testing.T) {
	s := loadSample(t)
	done := s.measured
	m := program(s, &done, &jobs.Job{ID: done.ID, Alive: false}, nil, 100, 40)
	if !facts(m).outputDone() {
		t.Fatal("the sample measured run must be finished for this test")
	}
	m, _ = m.Update(startedMsg("qwen3-8b-apple_m3_10c-003"))
	if got := facts(m).Run.ID; got != "qwen3-8b-apple_m3_10c-003" {
		t.Fatalf("a started run is on screen: %s", got)
	}
	m, _ = m.Update(stoppedMsg{"qwen3-8b-apple_m3_10c-003", "Sent SIGTERM to qwen3-8b-apple_m3_10c-003."})
	f := facts(m)
	if f.Run == nil || f.Run.ID != done.ID {
		t.Fatalf("the finished run is back on screen after the stop: %+v", f.Run)
	}
	if note := m.(Model).note; !strings.Contains(note, "Back to "+shortRun(done.ID)) {
		t.Fatalf("the note says which run is back: %q", note)
	}
	if strings.Contains(plain(emitAction(f).label), "Run first") {
		t.Fatalf("Emit must read the finished run again, not ask for a run: %q", emitAction(f).label)
	}
	// a stop with no finished run before it leaves the stopped run on screen
	m, _ = m.Update(startedMsg("qwen3-8b-apple_m3_10c-004"))
	m, _ = m.Update(startedMsg("qwen3-8b-apple_m3_10c-005"))
	m, _ = m.Update(stoppedMsg{"qwen3-8b-apple_m3_10c-005", "stopped"})
	if got := facts(m).Run.ID; got != "qwen3-8b-apple_m3_10c-005" {
		t.Fatalf("no finished run to go back to: the stopped run stays: %s", got)
	}
}
