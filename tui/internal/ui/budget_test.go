package ui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"
	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// budgetRun is a finished run on the Results screen whose plain block carries a time budget: the recorded `screen plain`
// of a prefill run with a real budget record beside its capture (name: ids, or unfit, whose trace is longer than its wall).
func budgetRun(t *testing.T, name string, width, height int) (tea.Model, *seam.Plain) {
	t.Helper()
	run, _ := recordedPlain(t, "decode")
	var p seam.Plain
	recorded(t, "plain-prefill-budget-"+name, &p)
	p.ID = run.ID // the screen shows a plain block for the run on screen
	return finished(t, run, &p, width, height), &p
}

func fits(t *testing.T, text string, width int) {
	t.Helper()
	for _, l := range strings.Split(text, "\n") {
		if ansi.StringWidth(l) > width {
			t.Fatalf("a line of %d cells: %q", ansi.StringWidth(l), l)
		}
	}
}

// Where the time went is a section of layer 2 (b opens it): each part with its rows under it, the figures in one column,
// the parts adding up to the wall the record states, then its sentences.
func TestWhereTheTimeWentSection(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m, p := budgetRun(t, "ids", 80, 24)
	tb := p.Layer2.TimeBudget
	if closed := body(m); !strings.Contains(closed, "▸ "+tb.Title) || strings.Contains(closed, tb.Rows[0].Label) {
		t.Fatalf("closed until b:\n%s", closed)
	}
	m = press(m, "b")
	if !facts(m).Open.Budget {
		t.Fatal("b opens it")
	}
	got := body(m)
	fits(t, got, 76)
	for _, r := range tb.Rows {
		line := ""
		for _, l := range strings.Split(got, "\n") {
			if strings.HasSuffix(l, " "+r.Ms) && strings.Contains(flat(l), flat(strings.Fields(r.Label)[0])) {
				line = l
			}
		}
		if line == "" {
			t.Errorf("no row %q with %s", r.Label, r.Ms)
		}
	}
	for _, l := range tb.Lines {
		if !strings.Contains(flat(got), flat(l)) {
			t.Errorf("missing %q", l)
		}
	}
	section := got[strings.Index(got, "▾ "+tb.Title):]
	section = section[:strings.Index(section, "▸ ")]
	golden(t, "results-budget.txt", section)
	t.Log("\n" + section)
	golden(t, "results-budget-80x24.txt", frame(m))
	if keys := facts(m); sectionTitle(keys.Plain, keyBudget) != tb.Title {
		t.Fatal("the ? page names b by the section's title")
	}
}

// A record whose trace runs longer than its wall says so, in the warning colour, even with the section closed.
func TestWhereTheTimeWentNeverHidesAMisfit(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m, p := budgetRun(t, "unfit", 80, 24)
	warn := *p.Layer2.TimeBudget.Warn
	if closed := body(m); !strings.Contains(flat(closed), flat(warn)) {
		t.Fatalf("closed, the caveat shows:\n%s", closed)
	}
	open := body(press(m, "b"))
	if strings.Count(flat(open), flat(warn)) != 1 {
		t.Fatalf("open, it leads the section once:\n%s", open)
	}
	t.Log("\n" + open[strings.Index(open, "▾ "):])
}

// Two prefills compared, one sent as token ids and one as a chat request: the head to head says the forms differ and
// marks no winner on time; both budgets sit side by side under it, A and B.
func TestCompareShowsFormsAndBothBudgets(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	c := recordedCompare(t, "forms")
	got := plain(wrapText(compareRunsBody(Facts{Compared: c}, 76), 76)) // the page wraps the parts' full lines
	fits(t, got, 76)
	for _, want := range []string{"The two runs sent the prompt in different forms (token ids against a chat request).",
		c.Head.TimeBudget.Title} {
		if !strings.Contains(flat(got), flat(want)) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if strings.Contains(got, "tok/s ✓") || strings.Contains(got, "ms ✓") {
		t.Error("no winner across forms")
	}
	for _, r := range c.Head.TimeBudget.Rows {
		if !strings.Contains(got, r.Cells[0]) || !strings.Contains(got, r.Cells[1]) {
			t.Errorf("row %q: %v", r.Label, r.Cells)
		}
	}
	golden(t, "compare-forms.txt", got)
	t.Log("\n" + got)
}

// The run's settings line says how a prefill sent its prompt.
func TestRunSettingsLineSaysThePromptForm(t *testing.T) {
	var res seam.Results
	recorded(t, "results-prefill-chat-engine", &res)
	got := plain(runEngineBody(res.Engine))
	if !strings.Contains(got, "Engine settings: chunk size 2048. The prompt was sent as a chat request.") {
		t.Fatalf("%q", got)
	}
	res.Engine.SettingsWords = nil
	if got := plain(runEngineBody(res.Engine)); !strings.HasPrefix(got, "The prompt was sent as a chat request.\n") {
		t.Fatalf("alone: %q", got)
	}
}
