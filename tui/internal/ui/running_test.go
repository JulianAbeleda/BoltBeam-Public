package ui

import (
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func init() { // every golden reads the same time: 1m 20s after the sample job started
	clock = func() time.Time { return time.Date(2026, 10, 8, 10, 1, 20, 0, time.UTC) }
}

// While a job runs the top box is the bar only; l adds the log; a failed run shows the step, the reason and the log.
func TestRunningScreens(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	running := page(program(s, &s.planned, s.job, s.tail, 80, 24), pageRun)
	golden(t, "run-running.txt", running.View())
	withLog := press(running, "l")
	golden(t, "run-running-log.txt", withLog.View())
	if !strings.Contains(withLog.View(), "stage autoscan: done") || strings.Contains(running.View(), "stage autoscan: done") {
		t.Fatal("l must show the log, and only then")
	}
	dead := *s.job
	dead.Alive = false
	failed := page(program(s, &s.planned, &dead, append(append([]string{}, s.tail...), "stage analyze: failed: no roles in the model"), 80, 24), pageRun)
	golden(t, "run-failed.txt", failed.View())
	for _, want := range []string{"Plan what to try failed", "no roles in the model", "Back to setup"} {
		if !strings.Contains(failed.View(), want) {
			t.Fatalf("the failed screen lacks %q", want)
		}
	}
}

// The bar weighs steps by the last run's seconds, adds time and p of q inside a step, and holds under 99%.
func TestProgressFraction(t *testing.T) {
	p := seam.ReadProgress([]string{"old line", "stage x: failed: old", "pipeline steps: 3", "pipeline expect: 10,80,10",
		"stage load: start", "stage load: done", "stage measure_timing: start", "stage measure_timing: progress 1/4"})
	if p.Failed != "" || p.Done != 1 || p.Current != "measure_timing" || p.SubOf != 4 {
		t.Fatalf("%+v", p)
	}
	if got := p.Fraction(0); got < 0.299 || got > 0.301 { // (10 + 80/4) / 100
		t.Fatalf("fraction %v", got)
	}
	if got := p.Fraction(70); got < 0.799 || got > 0.801 { // time wins: (10 + 70) / 100
		t.Fatalf("fraction %v", got)
	}
	if got := p.Fraction(1000); got != 0.9 { // capped at the step's weight
		t.Fatalf("fraction %v", got)
	}
	f := Facts{Job: &jobs.Job{ID: "a", Alive: true}, Tail: []string{"pipeline steps: 2", "stage load: start"}}
	f.advance(clock())
	f.advance(clock().Add(20 * time.Second))
	high := f.Frac
	f.Tail = []string{"pipeline steps: 9"} // a re-read that knows less never moves the bar back
	f.advance(clock().Add(21 * time.Second))
	if f.Frac < high || high <= 0 || high > 0.99 {
		t.Fatalf("bar %v then %v", high, f.Frac)
	}
}

// A measured run lists only the steps it ran: no handoff row, no next sentence.
func TestMeasuredStagesHideHandoff(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	r := s.measured
	r.Stages = append([]seam.Stage{}, r.Stages...)
	for i := range r.Stages {
		if r.Stages[i].Key == "runner_plan" {
			r.Stages[i].State, r.Stages[i].StateNote = "not_needed", sp("not needed: measured on this machine")
		}
	}
	r.Measure = &seam.MeasureStatus{Status: "measured", Probe: sp("measured")}
	got := measureBody(Facts{Run: &r}, 76)
	golden(t, "stages-measured.txt", got)
	if strings.Contains(got, "Prepare the handoff") || strings.Contains(got, "next") || !strings.Contains(got, "8. ") {
		t.Fatalf("stage list:\n%s", got)
	}
}

// Batch is the fourth Setup line. Its picker takes 1, 8, 32 or a typed number; an engine that decodes one stream
// shows the others as not supported; the choice reaches the pipeline as --batch.
func TestBatchPicker(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := press(press(press(press(program(s, nil, nil, nil, 80, 24), "j"), "j"), "j"), "enter")
	if m.(Model).cursor != pageBatch {
		t.Fatalf("enter on Batch opens its picker: %d", m.(Model).cursor)
	}
	m = press(press(press(m, "j"), "j"), "enter") // 32
	if got := m.(Model).f.batch(); got != 32 || m.(Model).cursor != pageSetup || m.(Model).row != 3 {
		t.Fatalf("batch %d, page %d, row %d", got, m.(Model).cursor, m.(Model).row)
	}
	typed := press(press(press(page(m, pageBatch), "1"), "6"), "enter")
	if got := typed.(Model).f.batch(); got != 16 {
		t.Fatalf("typed batch %d", got)
	}
	tg := typed.(Model)
	tg.f.Engine = "tinygrad"
	no, yes := false, true
	tg.f.Providers = &seam.Providers{Providers: []seam.ProviderRow{{Provider: "llama.cpp", Available: true, BatchOverOne: &yes},
		{Provider: "tinygrad", Available: true, BatchOverOne: &no}}}
	if tg.f.batch() != 1 {
		t.Fatal("tinygrad runs batch 1")
	}
	golden(t, "picker-batch-tinygrad.txt", page(tg, pageBatch).View())
	argv := seam.Client{Python: "py"}.PipelineArgv(seam.Pipeline{Model: "m", RunDir: "r", Target: "t", Batch: "16"})
	if !strings.Contains(strings.Join(argv, " "), "--batch 16") {
		t.Fatalf("argv %v", argv)
	}
}

// A batch run says so, and shows per stream and total beside the batch limit.
func TestBatchResults(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	r := s.measured
	pct := 61.0
	row := seam.BatchRow{Context: 512, Batch: 8, StepMs: 80, TokSStream: 12.5, TokSTotal: 100, PctOfLimit: &pct}
	row.Limit.TokSTotal, row.Limit.Bound = 164, "memory"
	pct1 := 80.0
	one := seam.BatchRow{Context: 512, Batch: 1, StepMs: 60, TokSStream: 16.6, TokSTotal: 16.6, PctOfLimit: &pct1}
	one.Limit.TokSTotal, one.Limit.Bound = 20.8, "memory"
	r.Results.Batches = []seam.BatchRow{one, row}
	r.Measure = &seam.MeasureStatus{Status: "measured", Provider: sp("llama.cpp"), Batches: []int{1, 8}}
	got := plain(resultBody(Facts{Run: &r}, 100))
	golden(t, "results-batch.txt", got)
	for _, want := range []string{"Measured with llama.cpp · roles not timed · batch 8", "PER STREAM", "100.0", "164.0", "61%"} {
		if !strings.Contains(got, want) {
			t.Fatalf("lacks %q:\n%s", want, got)
		}
	}
}
