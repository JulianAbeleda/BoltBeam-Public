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
	failed := page(program(s, &s.planned, &dead, append(append([]string{}, s.tail...), "stage measure_timing: failed: llama-bench exited 1"), 80, 24), pageRun)
	golden(t, "run-failed.txt", failed.View())
	for _, want := range []string{"failed", "llama-bench exited 1", "Back to setup", "Show details: l"} {
		if !strings.Contains(failed.View(), want) {
			t.Fatalf("the failed screen lacks %q", want)
		}
	}
}

// An engine's own error is one sentence in the box: what it said and where inside the engine it was raised. The
// traceback behind it (the pipeline's "detail:" lines) and the log wait behind l, "Show details".
func TestFailedScreenKeepsTheTracebackBehindDetails(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	dead := *s.job
	dead.Alive = false
	tail := append(append([]string{}, s.tail...),
		"stage measure_timing: failed: tinygrad decode failed: ffn_down_resadd epilogue requires rows=4096, got rows=5120 (ValueError inside the tinygrad fork at tinygrad/llm/decode_kernels.py:223)",
		"stage measure_timing: detail: Traceback (most recent call last):",
		"stage measure_timing: detail:   File \"tinygrad/llm/decode_kernels.py\", line 223, in validate",
		"stage measure_timing: detail:     raise ValueError(f\"{self.kind} epilogue requires rows=4096, got rows={rows}\")",
		"stage measure_timing: detail: ValueError: ffn_down_resadd epilogue requires rows=4096, got rows=5120")
	failed := page(program(s, &s.planned, &dead, tail, 80, 24), pageRun)
	golden(t, "run-failed-engine.txt", failed.View())
	v := failed.View()
	if !strings.Contains(v, "rows=5120") || !strings.Contains(v, "inside the tinygrad fork at") || strings.Contains(v, "Traceback") || strings.Contains(v, "stage load: done") {
		t.Fatal("the box shows the sentence only; the traceback and the log wait behind l")
	}
	if !strings.Contains(v, "l details") {
		t.Fatal("the footer offers l for the details")
	}
	shown := press(failed, "l")
	golden(t, "run-failed-engine-details.txt", shown.View())
	body := failedBody(shown.(Model).f, 76)
	if !strings.Contains(shown.View(), "Traceback (most recent call last)") || !strings.Contains(body, "Last lines of the log") {
		t.Fatal("l shows the traceback and the log")
	}
}

// The quick setup stages are not steps: while they run the step line says Starting, the count is hidden
// (never "0 of N") and the bar holds at 2% at most. Step 1 is the first counted stage.
func TestSetupStagesAreNotCounted(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	setup := []string{s.tail[0], "pipeline steps: 6", "pipeline counted: 0,0,0,0,0,1,1,1,1,1,1",
		"stage load: start", "stage load: done", "stage autoscan: start", "stage autoscan: done", "stage analyze: start",
		"stage analyze: done", "stage machine: start", "stage machine: done", "stage measure_probe: start"}
	p := seam.ReadProgress(setup)
	if !p.Setup || p.Done != 0 || p.Total != 6 {
		t.Fatalf("%+v", p)
	}
	if got := p.Fraction(1); got <= 0 || got > seam.SetupShare {
		t.Fatalf("setup fraction %v", got)
	}
	if got := p.Fraction(600); got != seam.SetupShare {
		t.Fatalf("setup fraction %v", got)
	}
	starting := page(program(s, &s.planned, s.job, setup, 80, 24), pageRun)
	golden(t, "run-starting.txt", starting.View())
	if v := starting.View(); !strings.Contains(v, "Starting…") || strings.Contains(v, " of 6") {
		t.Fatalf("setup must read Starting with no count:\n%s", v)
	}
	if line, _ := measureLine(Facts{Job: s.job, Tail: setup}); line != "run" {
		t.Fatal(line)
	}
	step1 := append(append([]string{}, setup...), "stage measure_probe: done", "stage measure_timing: start", "stage measure_timing: progress 1/4")
	p = seam.ReadProgress(step1)
	if p.Setup || p.Done != 0 || p.Current != "measure_timing" {
		t.Fatalf("%+v", p)
	}
	v := page(program(s, &s.planned, s.job, step1, 80, 24), pageRun).View()
	if !strings.Contains(v, "1 of 6") || !strings.Contains(v, "Time the real decode") {
		t.Fatalf("step 1 must read 1 of 6:\n%s", v)
	}
	done := append(append([]string{}, step1...), "stage measure_timing: done", "stage ingest_probe: start", "stage ingest_probe: done",
		"stage ingest_timing: start")
	if p = seam.ReadProgress(done); p.Done != 2 || p.Setup {
		t.Fatalf("%+v", p)
	}
	if d, n := seam.PipelineProgress(done); d != 2 || n != 6 {
		t.Fatalf("%d of %d", d, n)
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

// twelveSteps is a tinygrad run 1 s into step 6: five steps done, measure_timing started.
func twelveSteps(expect string) []string {
	lines := []string{"pipeline steps: 12"}
	if expect != "" {
		lines = append(lines, "pipeline expect: "+expect)
	}
	for _, k := range []string{"load", "autoscan", "analyze", "machine", "measure_probe"} {
		lines = append(lines, "stage "+k+": start", "stage "+k+": done")
	}
	return append(lines, "stage measure_timing: start")
}

// No time for this engine: equal weights. 6 of 12 with 1 s in step 6 is about 42%, never 100.
func TestProgressEqualWeights(t *testing.T) {
	p := seam.ReadProgress(twelveSteps(""))
	if p.Expect != nil || p.Done != 5 || p.Total != 12 {
		t.Fatalf("%+v", p)
	}
	if got := p.Fraction(1); got < 0.40 || got > 0.47 {
		t.Fatalf("fraction %v", got)
	}
	if got := p.Fraction(10000); got > 0.5 {
		t.Fatalf("one step never fills past its weight: %v", got)
	}
}

// The log keeps every run of a run id. Before the new run prints its steps, the old run's
// "pipeline done" must not count: that showed 100% at 6 of 12 and the bar never moves back.
func TestProgressIgnoresTheLastJob(t *testing.T) {
	old := append([]string{"=== old start", "pipeline steps: 9"}, "stage load: start", "stage load: done", "pipeline done: r")
	p := seam.ReadProgress(append(append([]string{}, old...), "=== new start"))
	if p.Finished || p.Total != 0 {
		t.Fatalf("the old run leaked in: %+v", p)
	}
	f := Facts{Job: &jobs.Job{ID: "a", Alive: true, StartedAt: clock().Format(time.RFC3339)}, Tail: append(append([]string{}, old...), "=== new start")}
	f.advance(clock())
	f.Tail = append(f.Tail, twelveSteps("")...)
	f.advance(clock().Add(32 * time.Second))
	if f.Frac >= 0.99 || f.Frac < 0.4 {
		t.Fatalf("bar %v", f.Frac)
	}
}

// A step that runs past the last run's time holds at its weight, the total stays at 99% at most,
// and the step line says so.
func TestProgressLongerThanLastTime(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	tail := twelveSteps("0.1,0.1,0.1,0.1,1.3,22.6,0.1,0.1,0.1,0.1,0.1,0.1")
	p := seam.ReadProgress(tail)
	if got := p.Fraction(500); got > 0.99 || false {
		t.Fatalf("fraction %v", got)
	}
	start := clock().Add(-40 * time.Second)
	f := Facts{Job: &jobs.Job{ID: "a", Alive: true, StartedAt: start.Format(time.RFC3339)}, Tail: tail}
	f.advance(start)
	f.advance(clock())
	if f.Frac > 0.99 {
		t.Fatalf("bar %v", f.Frac)
	}
	got := runningBody(f, 80)
	if !strings.Contains(got, "(longer than last time)") || strings.Contains(got, "100%") {
		t.Fatalf("running body:\n%s", got)
	}
}

// A first run has no history: the pipeline sends its default table and says so, the bar weighs steps by it (the
// search stage is most of a tinygrad run), the percent carries an estimate mark and no step is "longer than last time".
func TestFirstRunWeightsAreAnEstimate(t *testing.T) {
	lines := []string{"pipeline steps: 3", "pipeline expect: 130.0,40.0,350.0", "pipeline expect source: default",
		"stage measure_timing: start", "stage measure_timing: done", "stage role_time: start", "stage role_time: done",
		"stage search: start", "stage search: progress 1/7"}
	p := seam.ReadProgress(lines)
	if !p.Estimate || p.Done != 2 || p.Current != "search" {
		t.Fatalf("%+v", p)
	}
	if got := p.Fraction(0); got < 0.42 || got > 0.43 { // (130 + 40 + 350/7) / 520
		t.Fatalf("fraction %v", got)
	}
	if p.Over(1000) {
		t.Fatal("a default weight is not a last time to be over")
	}
	f := Facts{Job: &jobs.Job{ID: "a", Alive: true, StartedAt: "2026-10-10T01:00:00Z"}, Tail: lines}
	f.advance(clock())
	if body := runningBody(f, 80); !strings.Contains(body, "~") || !strings.Contains(body, "3 of 3") {
		t.Fatalf("the percent is not marked an estimate:\n%s", body)
	}
	hist := seam.ReadProgress(append([]string{"pipeline steps: 1", "pipeline expect: 10", "pipeline expect source: history"}, "stage search: start"))
	if hist.Estimate || !hist.Over(11) {
		t.Fatalf("history weights are real last times: %+v", hist)
	}
}
