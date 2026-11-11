package ui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/colorprofile"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// recordedCompare is a recorded `screen head-to-head` answer (testdata/recorded/compare-*.json, from saved runs) as the
// compare screen holds it.
func recordedCompare(t *testing.T, name string) *Compared {
	t.Helper()
	var c seam.Compared
	recorded(t, "compare-"+name, &c)
	if c.Head == nil || len(c.Head.Runs) != 2 {
		t.Fatalf("compare-%s: no head to head", name)
	}
	return &Compared{Before: c.Head.Runs[0].ID, After: c.Head.Runs[1].ID, Head: c.Head, Lines: c.Lines, Plain: c.Plain, PartsNote: c.PartsNote}
}

// Two runs of one model on one GPU and engine: the figures side by side with the better one marked, the line that
// says nothing differs, then their parts compared (the audit's plain answer, its lines under Full detail).
func TestCompareHeadToHeadSameModel(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	c := recordedCompare(t, "same")
	got := plain(compareRunsBody(Facts{Compared: c}, 76))
	for _, want := range []string{"A  h100-generic", "B  h100-in-model", "Same model, GPU, engine, build, weights and settings.",
		"81.5 tok/s ✓", "81.1 tok/s", "40% ✓", "+11.3 tok/s", c.Plain.Layer1.Fast, titleFull} {
		if !strings.Contains(strings.Join(strings.Fields(got), " "), strings.Join(strings.Fields(want), " ")) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if strings.Contains(got, "+11.3 tok/s ✓") {
		t.Error("the biggest gain is marked neither way")
	}
	golden(t, "compare-same-model.txt", got)
}

// Two runs of different models (and engines): the figures still side by side, what differs said in one line, and one
// line for why the parts are not compared, with nothing of the audit under it.
func TestCompareHeadToHeadOtherModel(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	c := recordedCompare(t, "other")
	got := plain(compareRunsBody(Facts{Compared: c}, 76))
	for _, want := range []string{"These differ: model (Qwen3.8-27B-UD-Q4_K_M / Qwen3-14B-AWQ)", "596.0 tok/s", "156.2 tok/s ✓",
		*c.PartsNote} {
		if !strings.Contains(strings.Join(strings.Fields(got), " "), strings.Join(strings.Fields(want), " ")) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if strings.Contains(got, titleFull) || c.Plain != nil {
		t.Error("two models: no parts compared")
	}
	golden(t, "compare-other-model.txt", got)
	if gpus := plain(compareRunsBody(Facts{Compared: recordedCompare(t, "gpus")}, 76)); !strings.Contains(gpus, "differ in GPU") {
		t.Errorf("one model on two GPUs: the parts line names the GPU:\n%s", gpus)
	}
}

// A run with no measured result is refused as one line, with no figures.
func TestCompareHeadToHeadRefused(t *testing.T) {
	refused := "Not compared: b has no measured result yet."
	h := &seam.HeadToHead{Refused: &refused}
	h.Runs = append(h.Runs, struct {
		ID    string `json:"id"`
		Title string `json:"title"`
	}{"a", "A on X with llama.cpp"}, struct {
		ID    string `json:"id"`
		Title string `json:"title"`
	}{"b", "B on X with llama.cpp"})
	got := plain(compareRunsBody(Facts{Compared: &Compared{Head: h}}, 76))
	if !strings.HasSuffix(got, refused) || strings.Contains(got, "tok/s") {
		t.Fatalf("%q", got)
	}
}
