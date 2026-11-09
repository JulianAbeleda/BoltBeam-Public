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

// The AWQ 27B on the 5090 (2026-11-09): a run finished on a chip with no runs before it, and Recommended setup kept
// "Measure MTP needs this machine's facts: run once on this chip first". A run that ends asks for the recommendation again.
func TestARunThatEndsAsksForTheRecommendationAgain(t *testing.T) {
	s := loadSample(t)
	done := s.measured
	m := program(s, &done, &jobs.Job{ID: done.ID, Alive: true}, nil, 100, 40)
	mm := m.(Model)
	mm.f.RecKey = "the key before the run"
	m, _ = mm.Update(jobMsg{job: &jobs.Job{ID: done.ID, Alive: false}})
	if key := facts(m).RecKey; key == "the key before the run" {
		t.Fatal("the recommendation from before the run is still taken as current")
	}
}

// The 5090 (2026-11-09): Measure MTP ended with no run on screen for the model and chip, and the last tick asked to
// read run "" ("run id must be a folder name under the runs folder"). With no run, only the runs list is read.
func TestTheLastTickWithNoRunReadsNoRun(t *testing.T) {
	s := loadSample(t)
	m := program(s, nil, nil, nil, 100, 40)
	mm := m.(Model)
	mm.f.Run, mm.f.Job = nil, nil
	_, cmd := mm.Update(tickMsg{})
	if cmd == nil {
		t.Fatal("the runs list is still read")
	}
	if _, ok := cmd().(runsMsg); !ok { // only the list: a batch would also read run ""
		t.Fatal("a tick with no run must read only the runs list, not run \"\"")
	}
}
