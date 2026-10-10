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
	if !hasLabel(setupActions(facts(planned)), "[ Emit ] Run first") {
		t.Fatal("a run without a per-role table: Emit must say Run first")
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

// Pressing Emit asks Python; its answer opens the Gameplan page with the markdown as plain text and an action
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
		for _, want := range []string{"# Gameplan: Qwen3-8B on apple_m3_10c", "7 roles, worst first", "Open the gameplan (gameplan.md)", "[ Back to setup ]"} {
			if !strings.Contains(view, want) {
				t.Errorf("%d cols: missing %q in:\n%s", cols, want, view)
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
	m := program(s, isolatedRun(t), nil, nil, 80, 24)
	mm, _ := m.Update(emitMsg{nil, fmt.Errorf("no per-role table in this run: Run first")})
	if got := mm.(Model); got.cursor != pageSetup || got.note != "Emit failed: no per-role table in this run: Run first" {
		t.Fatalf("a failed Emit: cursor %d note %q", got.cursor, got.note)
	}
}
