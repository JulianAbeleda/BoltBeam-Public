package ui

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func recordedOffload(t *testing.T, name string) *seam.Offload {
	t.Helper()
	raw, err := os.ReadFile(recordedPath(name))
	if err != nil {
		t.Fatal(err)
	}
	var o seam.Offload
	if err := json.Unmarshal(raw, &o); err != nil {
		t.Fatal(err)
	}
	return &o
}

func TestOffloadPageLaysOutPythonsWords(t *testing.T) {
	o := recordedOffload(t, "offload.json")
	f := Facts{Offload: o, OffKey: "k"}
	for _, w := range []int{80, 60} {
		body := wrapText(offloadBody(f, w), w)
		got := ansi.Strip(body)
		if !strings.Contains(got, o.Page.Headline[:40]) {
			t.Fatalf("width %d lacks the headline:\n%s", w, got)
		}
		for _, s := range o.Page.Sections {
			if !strings.Contains(got, s.Title) {
				t.Fatalf("width %d lacks section %q", w, s.Title)
			}
		}
		for _, l := range strings.Split(got, "\n") {
			if ansi.StringWidth(l) > w {
				t.Fatalf("width %d: line of %d cells: %q", w, ansi.StringWidth(l), l)
			}
		}
		for _, bad := range []string{"/home/", "h2d_gbs", "{", "\"label\""} {
			if strings.Contains(got, bad) {
				t.Fatalf("width %d shows %q:\n%s", w, bad, got)
			}
		}
		// the labels of one section line up: every row's value starts in the same column
		col := -1
		for _, l := range strings.Split(got, "\n") {
			for _, lab := range []string{"On the card, replies", "In RAM, replies", "On the card, prompts", "In RAM, prompts", "Also in RAM"} {
				if strings.HasPrefix(strings.TrimLeft(l, " "), lab) {
					i := strings.Index(l, lab) + ansi.StringWidth(lab)
					c := i + len(l[i:]) - len(strings.TrimLeft(l[i:], " "))
					if col >= 0 && c != col {
						t.Fatalf("width %d: values not aligned:\n%s", w, got)
					}
					col = c
				}
			}
		}
		if col < 0 {
			t.Fatalf("width %d: no rows found", w)
		}
	}
}

func TestOffloadRefusedPageIsItsHeadline(t *testing.T) {
	o := recordedOffload(t, "offload-refused.json")
	got := ansi.Strip(wrapText(offloadBody(Facts{Offload: o, OffKey: "k"}, 60), 60))
	if !strings.Contains(strings.ReplaceAll(got, "\n", " "), "No offload plan here: llama.cpp") {
		t.Fatalf("a refused page shows its headline:\n%s", got)
	}
}

func TestOffloadSetupLineHintThenHeadline(t *testing.T) {
	f := Facts{}
	if got := plain(offloadValue(f)); got != offloadHint {
		t.Fatalf("before load the line is the hint: %q", got)
	}
	f.OffKey = "k"
	if got := plain(offloadValue(f)); !strings.Contains(got, "working out") {
		t.Fatalf("while loading it says so: %q", got)
	}
	f.Offload = recordedOffload(t, "offload.json")
	if got := offloadValue(f); got != f.Offload.Page.Headline {
		t.Fatalf("after load the line is the headline: %q", got)
	}
	f.Offload, f.OffErr = nil, "python missing"
	if got := plain(offloadValue(f)); !strings.Contains(got, "python missing") {
		t.Fatalf("an error is said in words: %q", got)
	}
}

// The Setup line opens the page and the page shows what Python wrote.
func TestOffloadLineOpensThePage(t *testing.T) {
	m := speedProgram(t, nil)
	m.f.Offload, m.f.OffKey = recordedOffload(t, "offload.json"), m.f.recommendKey()
	for _, a := range setupActions(m.f) {
		if a.do == "page" && strings.Contains(plain(a.label), "Offload") {
			m2, _ := m.do(a)
			if m2.cursor != pageOffload || !strings.Contains(plain(frame(m2)), "Where the weights go") {
				t.Fatalf("the page did not open: %d", m2.cursor)
			}
			return
		}
	}
	t.Fatal("Setup has no Offload line")
}

func TestOffloadPageRenderAt80(t *testing.T) {
	f := Facts{Offload: recordedOffload(t, "offload.json"), OffKey: "k"}
	t.Log("\n" + ansi.Strip(wrapText(offloadBody(f, 76), 76)))
}
