package ui

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"
	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// recorded reads one recorded seam answer (testdata/recorded): real runs, with their ids, model paths and hosts renamed.
func recorded(t *testing.T, name string, v any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "recorded", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}

// recordedPlain is a recorded run (decode: one stream; batch: a sweep of stream counts) with its plain block.
func recordedPlain(t *testing.T, name string) (*seam.Run, *seam.Plain) {
	t.Helper()
	var run seam.Run
	var p seam.Plain
	recorded(t, "run-"+name, &run)
	recorded(t, "plain-"+name, &p)
	return &run, &p
}

// finished is the Run page of a run once the seam answered its plain block, as the screen has it.
func finished(t *testing.T, run *seam.Run, p *seam.Plain, width, height int) tea.Model {
	t.Helper()
	m := page(program(loadSample(t), run, nil, nil, width, height), pageRun)
	m, _ = m.Update(plainMsg{run.ID, p, nil})
	return m
}

// body is the Results page's body at the page's inner width, as the viewport holds it.
func body(m tea.Model) string { return drawn(wrapText(resultsBody(facts(m), 76), 76)) }

// flat is text with its wrapping undone, to find a sentence the page wrapped.
func flat(s string) string { return strings.Join(strings.Fields(ansi.Strip(s)), " ") }

// The Results screen is layer 1, then layer 2's sections, every sentence the seam's: the three answers under the
// words' labels, the top three as bars each with its gain, the trust line; the sections closed until a key opens one.
func TestResultsScreenIsLayerOneThenLayerTwo(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	run, p := recordedPlain(t, "decode")
	m := finished(t, run, p, 80, 24)
	if !facts(m).plainShown() {
		t.Fatal("a finished run with its plain block shows the layers")
	}
	closed := body(m)
	golden(t, "results-plain-decode.txt", closed)
	golden(t, "results-plain-80x24.txt", frame(m))
	l1 := p.Layer1
	for _, want := range []string{l1.Title, vocab.Label("fast") + " " + l1.Fast, vocab.Label("close") + " " + *l1.Close,
		vocab.Label("holds") + " " + *l1.Holds, vocab.Label("top"), *l1.Top[0].Gain.Text, l1.Trust.Text} {
		if !strings.Contains(flat(closed), flat(want)) {
			t.Fatalf("layer 1 lacks %q:\n%s", want, closed)
		}
	}
	for _, s := range p.Layer2.Sections {
		if !strings.Contains(closed, "▸ "+s.Title) || strings.Contains(flat(closed), flat(s.Lines[0])) {
			t.Fatalf("%s is a closed section until its key opens it:\n%s", s.Title, closed)
		}
	}
	open := press(press(press(press(m, "t"), "m"), "e"), "w")
	got := body(open)
	golden(t, "results-plain-decode-open.txt", got)
	for _, want := range append(append(slices.Clone(p.Layer2.Sections[0].Lines), p.Layer2.Sections[1].Lines...),
		p.Layer2.Glossary[0].Plain, p.Layer2.Table.Rows[0].Cells[0], p.Layer2.Table.Columns[1]) {
		if !strings.Contains(flat(got), flat(want)) {
			t.Fatalf("an open section lacks %q", want)
		}
	}
	if strings.Contains(got, "Tie-out") || !strings.Contains(got, "▸ "+titleFull) {
		t.Fatal("Full detail stays closed until f")
	}
	if !strings.Contains(body(press(open, "f")), "Tie-out, ms per token") {
		t.Fatal("f opens Full detail: the page as it always was")
	}
	all := body(press(m, "a"))
	golden(t, "results-plain-all-columns.txt", all)
	if !strings.Contains(all, "▾ "+vocab.Label("all_columns")) || !strings.Contains(all, "cumulative") {
		t.Fatal("a opens the table with every column")
	}
	if again := body(press(press(m, "m"), "m")); again != closed {
		t.Fatal("a second m closes the section")
	}
}

// Opening a section scrolls the page to its heading.
func TestOpeningASectionScrollsToIt(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	run, p := recordedPlain(t, "decode")
	m := press(finished(t, run, p, 80, 24), "w").(Model)
	if !m.f.Open.Words || m.view.YOffset() == 0 || !strings.Contains(frame(m), "▾ "+titleWords) {
		t.Fatalf("w opens the words and shows them:\n%s", frame(m))
	}
}

// A batch sweep adds its curve: total tokens a second per stream count at the best point's context, from the run's
// own batch rows; one stream count draws none.
func TestBatchSweepCurve(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	run, p := recordedPlain(t, "batch")
	got := body(finished(t, run, p, 80, 24))
	golden(t, "results-plain-batch.txt", got)
	for _, want := range []string{"context 128", "1    2    3    4    8   16", "90  166  232  311  631 1160"} {
		if !strings.Contains(got, want) {
			t.Fatalf("the curve lacks %q:\n%s", want, got)
		}
	}
	decode, dp := recordedPlain(t, "decode")
	if strings.Contains(body(finished(t, decode, dp, 80, 24)), "context 128") {
		t.Fatal("a run of one stream count has no curve")
	}
}

// A table with no gain anywhere has three columns, and layer 1 says why once, under the three bars.
func TestNoGainSaysWhyOnce(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	run, p := recordedPlain(t, "decode")
	note := "No gain per part is shown: the recording's timing could not be checked against the run without it."
	p.Layer1.GainsNote = &note
	for i := range p.Layer1.Top {
		p.Layer1.Top[i].Gain.Text = nil
	}
	p.Layer2.Table.Columns = p.Layer2.Table.Columns[:3]
	for i := range p.Layer2.Table.Rows {
		p.Layer2.Table.Rows[i].Cells = p.Layer2.Table.Rows[i].Cells[:3]
	}
	got := body(press(finished(t, run, p, 80, 24), "t"))
	if strings.Count(flat(got), note) != 1 || strings.Contains(got, "if this ran at full speed") || strings.Contains(got, "  gain") {
		t.Fatalf("three columns, the note once:\n%s", got)
	}
}

// Before the plain block is in, or when it is another run's, the page is the full detail as it always was.
func TestNoPlainBlockKeepsTheFullPage(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	run, p := recordedPlain(t, "decode")
	other := *p
	other.ID = "another-run"
	m := finished(t, run, &other, 80, 24)
	if facts(m).plainShown() || !strings.Contains(body(m), "Tie-out, ms per token") {
		t.Fatal("another run's block is not this run's")
	}
	if got := press(m, "t"); facts(got).Open.Table {
		t.Fatal("t opens nothing where no section is shown")
	}
}

// The ? page on the Results screen names each section key by its section's title.
func TestResultsHelpNamesTheSections(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	run, p := recordedPlain(t, "decode")
	m := press(finished(t, run, p, 80, 30), "?")
	got := frame(m)
	golden(t, "help-results.txt", got)
	for _, want := range []string{"t " + p.Layer2.Table.Title, "m " + p.Layer2.Sections[0].Title} {
		if !strings.Contains(got, want) {
			t.Fatalf("the ? page lacks %q:\n%s", want, got)
		}
	}
}

// Stage names come from the words (screen words); with none yet, a stage shows its key.
func TestStageNamesAreTheWords(t *testing.T) {
	if stageWord("measure") != "Can this Mac measure" || stageWord("role_time") != "Capture and attribute kernels" ||
		roleWord("ffn_down") != "feed-forward out" {
		t.Fatalf("%q %q %q", stageWord("measure"), stageWord("role_time"), roleWord("ffn_down"))
	}
	was := vocab
	defer func() { vocab = was }()
	vocab = nil
	if stageWord("measure") != "measure" || roleWord("ffn_down") != "ffn_down" {
		t.Fatal("no words: the key")
	}
}

// Time left is said only from the last run's own step times; a first run (default times) says none.
func TestTimeLeftOnlyFromLoggedTimes(t *testing.T) {
	freezeClock(t)
	setProfile(colorprofile.NoTTY)
	s := loadSample(t)
	tail := []string{"=== 2026-10-08T10:00:00Z start python3 -m boltbeam.workflow.screen pipeline", "pipeline steps: 4",
		"pipeline counted: 0,0,0,1,1,1,1", "pipeline expect: 60,120,300,30", "stage load: start", "stage load: done",
		"stage autoscan: start", "stage autoscan: done", "stage analyze: start", "stage analyze: done", "stage measure_timing: start"}
	logged := frame(page(program(s, &s.planned, s.job, tail, 80, 24), pageRun))
	golden(t, "run-time-left.txt", logged)
	if !strings.Contains(logged, "about 8 min left") || !strings.Contains(logged, "✓ Read the model · Check the machine · Plan what to try") {
		t.Fatalf("time left from the logged times, and what is done:\n%s", logged)
	}
	first := append(slices.Clone(tail[:4]), append([]string{"pipeline expect source: default"}, tail[4:]...)...)
	if got := frame(page(program(s, &s.planned, s.job, first, 80, 24), pageRun)); strings.Contains(got, "min left") {
		t.Fatalf("a first run's times are defaults: no time left\n%s", got)
	}
}

// Run again with this setup runs the command the run's own job ran, with a new run folder; a run this machine did
// not start is rebuilt from its record.
func TestRunAgainUsesTheRunsOwnSetup(t *testing.T) {
	store := jobs.Store{Dir: t.TempDir()}
	run, _ := recordedPlain(t, "batch")
	argv := []string{"py", "-m", "boltbeam.workflow.screen", "pipeline", "/models/m.gguf", "--run", "/w/old", "--target", "nvidia_sm120",
		"--provider", "vllm", "--batch", "2,16", "--analyze"}
	raw, _ := json.Marshal(jobs.Job{ID: run.ID, Argv: argv})
	if err := os.WriteFile(filepath.Join(store.Dir, run.ID+".json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	got, model, target, err := againArgv(seam.Client{Python: "py"}, store, run)
	if err != nil || model != "/models/m.gguf" || target != "nvidia_sm120" || !slices.Equal(got, argv) {
		t.Fatalf("%v %q %q %v", got, model, target, err)
	}
	dir := t.TempDir()
	r := *run
	r.ModelPath = filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(r.ModelPath, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	got, _, _, err = againArgv(seam.Client{Python: "py"}, jobs.Store{Dir: t.TempDir()}, &r)
	for _, fv := range [][2]string{{"--target", r.TargetID}, {"--provider", "vllm"}, {"--batch", "2,3,4,8,16"}, {"--model", ""}} {
		if j := slices.Index(got, fv[0]); fv[1] != "" && (err != nil || j < 0 || j+1 >= len(got) || got[j+1] != fv[1]) {
			t.Fatalf("from the record, %s: %v %v", fv[0], got, err)
		}
	}
	if !slices.Contains(got, "--analyze") || slices.Contains(got, "--role-time") || got[4] != r.ModelPath {
		t.Fatalf("from the record: %v", got)
	}
	r.ModelPath = "/models/not-here.gguf"
	if _, _, _, err := againArgv(seam.Client{}, jobs.Store{Dir: t.TempDir()}, &r); err == nil || !strings.Contains(err.Error(), "not on this machine") {
		t.Fatalf("a model not here: %v", err)
	}
}

// A compare with a plain answer (screen diff's plain key) says it first, then the seam's lines under Full detail.
func TestCompareSaysThePlainAnswerFirst(t *testing.T) {
	var pc seam.PlainCompare
	if err := json.Unmarshal([]byte(`{"layer1": {"fast": "b against a: llama.cpp 2 against 1, decode.", "lines": ["attention: unchanged; untraced time -0.010 ms."],
		"top": [{"text": "attention: -0.010 ms"}], "trust": {"checked": true, "key": "checked", "text": "Checked ✓"}}}`), &pc); err != nil {
		t.Fatal(err)
	}
	got := plain(compareRunsBody(Facts{Compared: &Compared{Lines: []string{"Engine upgrade audit: b against a"}, Plain: &pc}}, 80))
	if !strings.HasPrefix(got, "b against a: llama.cpp 2 against 1, decode.\nattention: unchanged") ||
		!strings.Contains(got, "Checked ✓\n\n"+titleFull+"\nEngine upgrade audit: b against a") {
		t.Fatalf("%q", got)
	}
}
