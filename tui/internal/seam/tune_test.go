package seam

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// `screen tune` facts only, recorded on this box for one MoE model and one build (all experts on the card, then 24 layers'
// experts in RAM): every row decodes with its value, source and reason; a knob that changes outputs carries a tag; Use
// these names what a run passes and what it cannot.
func TestTuneDecodesRecordedPages(t *testing.T) {
	for _, name := range []string{"tune-facts.json", "tune-experts-in-ram.json"} {
		var tn Tune
		recorded(t, name, &tn)
		p := tn.Page
		if tn.Kind != "tune" || p == nil || tn.Refused != nil || len(p.Rows) == 0 || p.Command == "" || p.Title == "" {
			t.Fatalf("%s: %+v", name, tn)
		}
		for _, r := range p.Rows {
			if r.Name == "" || r.Value == "" || r.SourceWords == "" || r.Reason == "" || (r.Status != "set" && r.Status != "default") {
				t.Fatalf("%s: row %+v", name, r)
			}
		}
		if len(p.Measure.Knobs) == 0 || p.Measure.None != nil || !strings.Contains(p.Measure.Progress, "{knob}") {
			t.Fatalf("%s: measure %+v", name, p.Measure)
		}
	}
	var n24 Tune
	recorded(t, "tune-experts-in-ram.json", &n24)
	tagged := slices.IndexFunc(n24.Page.Rows, func(r TuneRow) bool { return len(r.Tags) > 0 })
	if tagged < 0 || n24.Page.Rows[tagged].Use == nil || n24.Page.Rows[tagged].Use.Env == "" {
		t.Fatal("with experts in RAM one set row changes outputs, and a run passes it as environment")
	}
	u := n24.Page.Use
	// H2: the pin is copied with the prompt cache and the placement it assumes, so nothing is left as not passed
	if len(u.Settings) == 0 || len(u.Env) < 2 || u.NotPassed != nil || u.None != nil || u.Settings["cache_ram"] != "0" || u.Settings["n_cpu_moe"] == "" {
		t.Fatalf("use these: %+v", u)
	}
	for _, name := range []string{"tune-refused.json", "tune-no-build.json"} {
		var r Tune
		recorded(t, name, &r)
		if r.Page != nil || r.Refused == nil || !strings.HasPrefix(*r.Refused, "No tuning here: ") {
			t.Fatalf("%s: a refusal is a sentence and no page: %+v", name, r)
		}
	}
}

// The measure job's log is progress lines, then the page; a stopped job has no answer, and says so.
func TestReadTuneFromTheJobLog(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tui", "testdata", "recorded", "tune-facts.json"))
	if err != nil {
		t.Fatal(err)
	}
	log := append([]string{"=== 2026-11-11T10:00:00Z start python -m boltbeam.workflow.screen tune m --measure",
		"pipeline steps: 2", "stage a: start", "stage a: progress 0/4", "stage a: done", "stage b: start", "stage b: done",
		"pipeline done"}, strings.Split(string(raw), "\n")...)
	got, err := ReadTune(log)
	if err != nil || got.Page == nil || len(got.Page.Rows) == 0 {
		t.Fatalf("the answer after the progress: %+v %v", got, err)
	}
	if p := ReadProgress(log); !p.Finished || p.Total != 2 || p.Done != 2 {
		t.Fatalf("the progress lines are the pipeline's: %+v", p)
	}
	if _, err := ReadTune(log[:4]); err == nil {
		t.Fatal("a stopped job has no answer")
	}
	argv := Client{Python: "py", Root: "/r", Saved: "/s"}.TuneArgv("/m", "e", "/b", "--knob", "-ub")
	if strings.Join(argv, " ") != "py -m boltbeam.workflow.screen tune /m --engine e --knob -ub --build /b --facts-root /r --facts-root /r/.work --facts-root /s --measure" {
		t.Fatalf("argv: %v", argv)
	}
}

// The time budget in `screen plain` (a prefill run with a real budget record beside its capture): the parts add up to
// the wall as the record states it, and a record whose trace is longer than its wall says so in Warn.
func TestTimeBudgetDecodesAndAddsUp(t *testing.T) {
	var p Plain
	recorded(t, "plain-prefill-budget-ids.json", &p)
	tb := p.Layer2.TimeBudget
	if tb == nil || tb.Warn != nil || len(tb.Rows) < 6 {
		t.Fatalf("budget: %+v", tb)
	}
	sum, wall := 0.0, 0.0
	parts := []string{}
	for _, r := range tb.Rows {
		if r.Depth == 0 && r.Part != nil {
			parts = append(parts, *r.Part)
			if *r.Part == "wall" {
				wall = *r.Value
			} else {
				sum += *r.Value
			}
		}
	}
	if strings.Join(parts, ",") != "kernels,exposed,idle,host,unattributed,wall" || math.Abs(sum-wall) > 0.05 {
		t.Fatalf("parts %v add to %.3f, wall %.3f", parts, sum, wall)
	}
	var unfit Plain
	recorded(t, "plain-prefill-budget-unfit.json", &unfit)
	if w := unfit.Layer2.TimeBudget.Warn; w == nil || !strings.Contains(*w, "longer than the traced request's wall time") {
		t.Fatalf("window_fits_wall false is said: %v", w)
	}
}

// Head to head of an ids-form and a chat-form prefill: the cross-form warning, and both budgets side by side.
func TestCompareFormsAndBudgets(t *testing.T) {
	var c Compared
	recorded(t, "compare-forms.json", &c)
	h := c.Head
	if h == nil || h.Differs == nil || !strings.Contains(*h.Differs, "sent the prompt in different forms") {
		t.Fatalf("differs: %v", h)
	}
	for _, r := range h.Rows {
		if r.Better != nil && (r.Key == "speed" || r.Key == "ms") {
			t.Fatal("no winner across forms")
		}
	}
	s := h.TimeBudget
	if s == nil || len(s.Rows) == 0 || len(s.Columns) != 3 {
		t.Fatalf("side by side: %+v", s)
	}
	for _, r := range s.Rows {
		if len(r.Cells) != 2 {
			t.Fatalf("a cell per run: %+v", r)
		}
	}
	var e struct {
		Engine RunEngine `json:"engine"`
	}
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tui", "testdata", "recorded", "results-prefill-chat-engine.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &e); err != nil || e.Engine.PromptForm == nil || *e.Engine.PromptForm != "chat" ||
		!regexp.MustCompile(`chat request`).MatchString(*e.Engine.PromptFormWords) {
		t.Fatalf("prompt form: %+v %v", e.Engine, err)
	}
}

// A demo plays the measure jobs from its recording, else prints the line a successful job ends with; only outside a demo
// do they run the CLI (the GPU).
func TestMeasureJobsAreDemoSafe(t *testing.T) {
	c := Client{Python: "py"}
	if got := strings.Join(c.RoutingArgv("/m/a.gguf", "llama.cpp", "/b/llama-bench"), " "); got != "py -m boltbeam.cli routing /m/a.gguf --engine llama.cpp --build /b/llama-bench" {
		t.Fatalf("routing: %s", got)
	}
	if got := strings.Join(c.HostArgv(), " "); got != "py -m boltbeam.cli host --remeasure --json" {
		t.Fatalf("host: %s", got)
	}
	c.Demo = &Demo{Argv: map[string][]string{"routing": {"echo", "recorded"}}}
	if got := c.RoutingArgv("/m/a.gguf", "llama.cpp", ""); len(got) != 2 || got[0] != "echo" {
		t.Fatalf("a demo plays its recording: %v", got)
	}
	if got := c.HostArgv(); len(got) != 2 || got[0] != "echo" || !strings.Contains(got[1], `"remeasure_failed": false`) {
		t.Fatalf("a demo with no recording prints its success line: %v", got)
	}
	c.Demo = &Demo{}
	if got := c.RoutingArgv("/m", "e", ""); got[1] != "routing profile: demo" {
		t.Fatalf("%v", got)
	}
}
