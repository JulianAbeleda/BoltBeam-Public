package ui

import (
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"github.com/muesli/termenv"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// isolatedRun is run-002 carrying the -005 run's isolated per-role table: a finished run Emit can plan from.
func isolatedRun(t *testing.T) *seam.Run {
	run := measuredRun(t)
	load(t, "results-isolated", &run.Results)
	return run
}

// Setup shows [ Emit ] under [ Run ]: muted with "Run first" until the run on screen has a per-role table, then live.
func TestSetupEmitRowFollowsTheRun(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	ready := program(s, nil, nil, nil, 80, 24)
	acts := setupActions(facts(ready))
	if !hasLabel(acts, "[ Emit ] Run first") {
		t.Fatalf("no run: Emit must say Run first: %+v", labels(acts))
	}
	planned := program(s, &s.measured, nil, nil, 80, 24) // run-002 finished, but no per-role table
	if !hasLabel(setupActions(facts(planned)), "[ Emit gap report ]") {
		t.Fatal("a measured work breakdown must offer an optimization plan")
	}
	with := program(s, isolatedRun(t), nil, nil, 80, 24)
	acts = setupActions(facts(with))
	run, emit := -1, -1
	for i, a := range acts {
		switch a.label {
		case "[ Run ]":
			run = i
		case "[ Emit ]":
			emit = i
		}
	}
	if run < 0 || emit != run+1 || acts[emit].do != "emit" {
		t.Fatalf("Emit must be the live row right under Run: %+v", labels(acts))
	}
	running := program(s, isolatedRun(t), s.job, s.tail, 80, 24)
	if hasLabel(setupActions(facts(running)), "[ Emit ]") {
		t.Fatal("while the run goes, Emit is not offered")
	}
	golden(t, "setup-emit-active.txt", with.View())
}

func labels(acts []action) []string {
	out := []string{}
	for _, a := range acts {
		out = append(out, ansi.Strip(a.label))
	}
	return out
}

// Pressing Emit asks Python; its answer opens the Gameplan page with the markdown drawn (no # or ** left) and an action
// that opens gameplan.md. The page is golden at 80 and 110 columns.
func TestEmitPageShowsTheGameplan(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	var e seam.Emitted
	load(t, "emit", &e)
	if e.Kind != "emitted" || len(e.Plan.Roles) != 7 || e.Plan.Roles[0].Role != "attn_kv" || len(e.Plan.Roles[0].Layers) != 4 ||
		e.Plan.Roles[0].Layers[0].Layer != "bubblebeam" || e.Plan.Roles[0].Layers[3].Layer != "promotion" || e.Plan.Footer.Sentence == "" {
		t.Fatalf("emit.json decoded wrong: %+v", e.Plan.Header)
	}
	for _, cols := range []int{80, 110} {
		m := program(s, isolatedRun(t), nil, nil, cols, 30)
		got := m.(Model)
		got.row = 0
		for i, a := range setupActions(got.f) {
			if a.do == "emit" {
				got.row = i
			}
		}
		next, _ := got.do(action{"", "emit", ""})
		if !next.f.Emitting || !strings.HasPrefix(next.note, "Emit: writing the gameplan for") {
			t.Fatalf("pressing Emit must start one: emitting %v note %q", next.f.Emitting, next.note)
		}
		var mm tea.Model = next
		mm, _ = mm.Update(emitMsg{&e, nil})
		page := mm.(Model)
		if page.cursor != pageEmit || page.f.Emitting || page.f.Plan == nil {
			t.Fatalf("the answer must open the Gameplan page: cursor %d", page.cursor)
		}
		view := mm.View()
		for _, want := range []string{"Gameplan: Qwen3-8B on apple_m3_10c", "7 roles, worst first", "Open the gameplan (gameplan.md)", "[ Back to setup ]"} {
			if !strings.Contains(view, want) {
				t.Errorf("%d cols: missing %q in:\n%s", cols, want, view)
			}
		}
		for _, raw := range []string{"# Gameplan", "## 1.", "**BubbleBeam**", "```"} {
			if strings.Contains(view, raw) {
				t.Errorf("%d cols: markdown marker %q drawn as text", cols, raw)
			}
		}
		for _, l := range strings.Split(view, "\n") {
			if w := ansi.StringWidth(l); w > cols {
				t.Errorf("%d cols: line too wide (%d): %q", cols, w, l)
			}
		}
		golden(t, fmt.Sprintf("emit-%d.txt", cols), view)
	}
	// a failed Emit stays on Setup and says why
	m := program(s, isolatedRun(t), nil, nil, 80, 24).(Model)
	mm, _ := m.Update(emitMsg{nil, fmt.Errorf("no per-role table in this run: Run first")})
	if got := mm.(Model); got.cursor != pageSetup || got.note != "Emit failed for "+m.runID()+": no per-role table in this run: Run first" {
		t.Fatalf("a failed Emit: cursor %d note %q", got.cursor, got.note)
	}
}

// Returning to Setup must retain the displayed run's location for Emit and Open report.
func TestSavedRunKeepsOriginOnReturnToSetup(t *testing.T) {
	s := loadSample(t)
	m := program(s, isolatedRun(t), nil, nil, 80, 24).(Model)
	m.f.ReadOnly = true
	m.cursor = pageRun
	m.backToSetup()
	if !m.f.ReadOnly {
		t.Fatal("saved origin lost on Escape")
	}
	m.cursor = pageRun
	next, _ := m.do(action{"", "page", fmt.Sprint(pageSetup)})
	if !next.f.ReadOnly {
		t.Fatal("saved origin lost on Back to setup")
	}
}

func TestSavedEmitNamesItsSource(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := program(s, isolatedRun(t), nil, nil, 80, 24).(Model)
	m.f.ReadOnly = true
	got := emitAction(facts(m))
	if !strings.Contains(got.label, "saved run "+m.f.Run.ID) {
		t.Fatalf("Emit must identify its saved source: %s", got.label)
	}
}

func TestCapturedOperationEmitOffersOptimizationPlan(t *testing.T) {
	s := loadSample(t)
	run := isolatedRun(t)
	run.Results.Workload = "decode"
	run.Results.Loss.Roles = nil
	run.Results.Loss.SemanticCapture = true
	f := facts(program(s, run, nil, nil, 100, 24))
	a := emitAction(f)
	if a.do != "emit" || !strings.Contains(plain(a.label), "Emit gap report") {
		t.Fatalf("completed captured work must offer the shared optimization plan: %+v", a)
	}
}

func TestMultilineFailureNoticeKeepsScreenWithinTerminal(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	m := program(loadSample(t), nil, nil, nil, 170, 55).(Model)
	m.note = "FAILED: report does not reconcile with evidence\nattention has no finite ceiling; report limit refused"
	rendered := m.View()
	if lipgloss.Height(rendered) > m.height {
		t.Fatalf("failure notice overflows terminal: %d > %d", lipgloss.Height(rendered), m.height)
	}
	for _, text := range []string{"BoltBeam", "attention has no finite ceiling", "q quit"} {
		if !strings.Contains(rendered, text) {
			t.Fatalf("missing %q in failure view", text)
		}
	}
}

func TestForcedReplayCanEmitItsCheckedReport(t *testing.T) {
	s := loadSample(t)
	run := isolatedRun(t)
	run.Results.Workload = "replay"
	run.Results.Loss.Roles = nil
	run.Results.Loss.Replay = &seam.Replay{Forced: true}
	f := facts(program(s, run, nil, nil, 100, 24))
	a := emitAction(f)
	if a.do != "emit" || !strings.Contains(plain(a.label), "Emit replay report") {
		t.Fatalf("forced replay report must be available: %+v", a)
	}
}

func TestMeasuredReplayWithoutFloorStillShowsItsTable(t *testing.T) {
	s := loadSample(t)
	run := isolatedRun(t)
	run.Results.Loss = seam.Loss{Status: "measured", Replay: &seam.Replay{Forced: true, Title: "Captured replay", HeadlineLines: []string{"Timing qualification: FAILED"}, Columns: []string{"turn", "captured ms"}, Rows: []seam.SessionRow{{Cells: []string{"1", "5.000"}}}, Total: []string{"all", "5.000"}}}
	got := plain(resultBody(facts(program(s, run, nil, nil, 100, 24)), 100))
	for _, want := range []string{"Captured replay", "Timing qualification: FAILED", "5.000"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in %q", want, got)
		}
	}
	if strings.Contains(got, noKernelChoice) {
		t.Fatal("replay was replaced by kernel-search advice")
	}
}

func TestSharedOptimizationEmitAcrossWorkloads(t *testing.T) {
	for _, workload := range []string{"decode", "prefill", "session", "replay"} {
		run := isolatedRun(t)
		run.Results.Workload = workload
		run.Results.Loss.Roles = nil
		f := facts(program(loadSample(t), run, nil, nil, 100, 24))
		if !f.canEmit() || emitAction(f).do != "emit" {
			t.Fatalf("%s measured breakdown cannot emit", workload)
		}
		f.Plan = &seam.Emitted{Plan: seam.Gameplan{Kind: "gap_report"}, Markdown: "# Performance gaps\n\nMeasured gaps."}
		_, line := emitLine(f)
		if !strings.Contains(line, "gap report written") || !strings.Contains(emitBody(f, 100), "Measured gaps") {
			t.Fatal("generic plan not displayed")
		}
	}
}

func TestOptimizationPlanUsesSharedBorderedTables(t *testing.T) {
	old := tableWidth
	defer func() { tableWidth = old }()
	for _, width := range []int{76, 106} {
		tableWidth = width
		emitted := &seam.Emitted{Plan: seam.Gameplan{Kind: "gap_report"}, Markdown: "unused wide text table"}
		emitted.Plan.Presentation.Summary = "# Performance gaps"
		emitted.Plan.Presentation.Details = "## Gap summary\n\nUntested."
		emitted.Plan.Tables = []seam.Where{{Title: "Where the token goes", Columns: []string{"row", "ms now", "ms at limit", "ms at chip best", "ms above limit", "share", "+ Tok/S if roofline", "cumulative tok/s", "% of peak"},
			Rows:          []seam.WhereRow{{Name: "projection", Section: "Model operations", Cells: []string{"5.00", "4.00", "4.00", "1.00", "100%", "+50.0", "250.0", "80%"}}},
			BaselineCells: []string{"measured throughput", "", "", "", "", "", "200.0", "200.0", ""},
			SumCells:      []string{"= total at roofline", "5.00", "4.00", "4.00", "1.00", "100%", "250.0", "250.0", ""}}}
		got := emitBody(Facts{Plan: emitted}, width)
		for _, want := range []string{"Performance gaps", "┌", "CUMULATIVE", "Model operations", "+50.0", "250.0", "Gap summary"} {
			if !strings.Contains(got, want) {
				t.Fatalf("missing %q: %s", want, got)
			}
		}
		if strings.Contains(got, "unused wide text") {
			t.Fatal("used wide Markdown table instead of shared renderer")
		}
		for _, line := range strings.Split(got, "\n") {
			if strings.HasPrefix(line, "│") && maxWidth(line) > width {
				t.Fatal("Emit table exceeds terminal width")
			}
		}
	}
}
