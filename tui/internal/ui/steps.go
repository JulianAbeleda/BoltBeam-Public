package ui

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// Facts is everything the checklist reads. Every step's state is a function of these seam facts, never of
// what the user clicked, so reopening the screen mid run lands on the same marks.
type Facts struct {
	Path     string
	Editing  bool
	Input    string
	Files    []string
	Reading  bool
	ModelErr string
	Profile  *seam.Profile
	Targets  *seam.Targets
	Target   int
	ThisMac  string
	Ceiling  *seam.Ceiling
	CeilBusy bool
	CeilErr  string
	Runs     *seam.Runs
	Run      *seam.Run
	Job      *jobs.Job
	Tail     []string
	Spin     string
	Confirm  string // the run id waiting for a second enter before it is deleted
}

// action is one row inside a step's full view; enter on it does `do` with `arg`.
type action struct{ label, do, arg string }

// step is one row of the checklist. line feeds the list and the summary box; body feeds the summary box
// (cut to fit) and the full view, so the two cannot disagree.
type step struct {
	title   string
	hint    string
	about   func(Facts) string
	line    func(Facts) (mark, text string)
	body    func(Facts, int) string
	actions func(Facts) []action
}

var steps = []step{
	{"Model", "enter: pick or type the model file", aboutModel, modelLine, modelBody, modelActions},
	{"Chip", "enter: pick the chip", aboutChip, chipLine, chipBody, chipActions},
	{"Speed limit", "enter: every role, the prompt, the assumptions", aboutChip, limitLine, limitBody, nil},
	{"Measure", "enter: start, stop, earlier runs, the full log", aboutRun, measureLine, measureBody, measureActions},
	{"Result", "enter: open the full report, measure again", aboutRun, resultLine, resultBody, resultActions},
}

// firstOpen is the step the cursor starts on: the first one not done.
func firstOpen(f Facts) int {
	for i, s := range steps {
		if m, _ := s.line(f); m != "pass" {
			return i
		}
	}
	return len(steps) - 1
}

func (f Facts) target() *seam.Target {
	if f.Targets == nil || len(f.Targets.Targets) == 0 {
		return nil
	}
	if f.Target < 0 || f.Target >= len(f.Targets.Targets) {
		return &f.Targets.Targets[0]
	}
	return &f.Targets.Targets[f.Target]
}

func (f Facts) alive() bool { return f.Job != nil && f.Job.Alive }

func (f Facts) outputDone() bool {
	if f.Run == nil {
		return false
	}
	for _, st := range f.Run.Stages {
		if st.Key == "output" {
			return st.Done
		}
	}
	return false
}

// failedStage is the stage the live log says failed, or "".
func (f Facts) failedStage() string {
	for key, state := range seam.StageEvents(f.Tail) {
		if state == "failed" {
			return key
		}
	}
	return ""
}

func aboutModel(f Facts) string {
	if f.Profile != nil {
		return f.Profile.ModelID
	}
	if f.Path == "" {
		return ""
	}
	return filepath.Base(f.Path)
}

func aboutChip(f Facts) string {
	if t := f.target(); t != nil {
		return t.ID
	}
	return ""
}

func aboutRun(f Facts) string {
	if f.Run != nil {
		return f.Run.ID
	}
	return ""
}

// --- 1 Model -----------------------------------------------------------------------------------------------

func modelLine(f Facts) (string, string) {
	base := filepath.Base(f.Path)
	switch {
	case f.Path == "":
		return "open", "pick a model file"
	case f.Reading:
		return "run", "reading " + base + "…"
	case f.ModelErr != "":
		return "fail", "could not read " + base
	case f.Profile == nil:
		return "open", base + " · not read yet"
	}
	return "pass", fmt.Sprintf("%s · %s layers · %s", base, count(f.Profile.LayerCount), strings.Join(f.Profile.Metadata.QuantTypes, ", "))
}

func modelActions(f Facts) []action {
	out := []action{{"Type a path", "edit", ""}}
	if f.Editing {
		out[0].label = f.Input + stCursor.Render("▏")
	}
	for _, path := range f.Files {
		out = append(out, action{"Read " + path, "file", path})
	}
	return out
}

// --- 2 Chip ------------------------------------------------------------------------------------------------

func gbs(v *float64) string {
	if v == nil {
		return "memory speed unknown"
	}
	return fmt.Sprintf("memory %.1f GB/s", *v)
}

func chipLine(f Facts) (string, string) {
	t := f.target()
	if t == nil {
		return "open", "reading the chip list…"
	}
	text := t.ID
	if t.ID == f.ThisMac {
		text += " · this Mac"
	}
	if !t.HasCeiling {
		return "crossed", text + " · no measured speeds, so no speed limit"
	}
	return "pass", text + " · " + gbs(t.MemoryBandwidthGBs)
}

func chipActions(f Facts) []action {
	if f.Targets == nil {
		return nil
	}
	out := []action{}
	for i, t := range f.Targets.Targets {
		here, limit := "", mark("pass")+" speed limit"
		if t.ID == f.ThisMac {
			here = "this Mac"
		}
		if !t.HasCeiling {
			limit = mark("crossed") + " no speed limit"
		}
		chosen := "  "
		if i == f.Target {
			chosen = stAccent.Render("● ")
		}
		out = append(out, action{fmt.Sprintf("%s%-14s %-9s %s", chosen, t.ID, here, limit), "chip", fmt.Sprint(i)})
	}
	return out
}

// --- 3 Speed limit -----------------------------------------------------------------------------------------

func limitLine(f Facts) (string, string) {
	switch {
	case f.Profile == nil:
		return "open", "needs step 1"
	case f.CeilBusy:
		return "run", "working it out…"
	case f.Ceiling != nil && f.Ceiling.Decode.TokS != nil:
		return "pass", fmt.Sprintf("up to %.1f tokens per second", *f.Ceiling.Decode.TokS)
	case f.CeilErr != "":
		return "crossed", "no speed limit: " + f.CeilErr
	}
	return "open", "needs step 2"
}

// --- 4 Measure ---------------------------------------------------------------------------------------------

func measureLine(f Facts) (string, string) {
	if f.alive() {
		done, total := seam.PipelineProgress(f.Tail)
		now := ""
		for key, state := range seam.StageEvents(f.Tail) {
			if state == "running" {
				now = " · " + stageWord(key) + "…"
			}
		}
		if total == 0 {
			return "run", "running" + now
		}
		return "run", fmt.Sprintf("%s  %d of %d%s", bar(float64(done)/float64(total), 20), done, total, now)
	}
	if key := f.failedStage(); key != "" {
		return "fail", "stopped: " + stageWord(key) + " failed"
	}
	if f.Run != nil && f.Run.Measure != nil && f.Run.Measure.Status == "failed" {
		return "fail", "measuring failed: " + deref(f.Run.Measure.Reason)
	}
	switch {
	case f.Run == nil && f.Path == "":
		return "open", "needs step 1"
	case f.Run == nil:
		return "open", "not started · enter to plan and measure"
	case !f.outputDone():
		last := "nothing"
		if f.Run.LatestStage != nil {
			last = word(plainStage, *f.Run.LatestStage)
		}
		return "crossed", "stopped after " + last
	case len(f.Run.Blocked) > 0 && f.Run.Measure != nil && f.Run.Measure.Status == "skipped":
		return "wait", "planned · not measured here: " + deref(f.Run.Measure.Reason) + " · enter for what to run"
	case len(f.Run.Blocked) == 1:
		n := f.Run.Blocked[0]
		return "wait", fmt.Sprintf("planned · needs %s (%s)", word(plainNeed, n.Need), n.Request)
	case len(f.Run.Blocked) > 1:
		return "wait", "planned · needs building-block tests and a timing trace"
	}
	return "pass", "run " + shortRun(f.Run.ID) + " · " + measuredMarks(f.Run.Measured)
}

// shortRun is the run's number, the part after the model and chip.
func shortRun(id string) string {
	if i := strings.LastIndex(id, "-"); i >= 0 {
		return id[i+1:]
	}
	return id
}

func measureActions(f Facts) []action {
	out := []action{}
	if f.alive() {
		out = append(out, action{"Stop the run (x)", "stop", ""})
	} else if f.Path != "" {
		out = append(out, action{"Plan and measure " + aboutModel(f) + " on " + aboutChip(f), "start", ""})
	}
	if f.Runs != nil {
		for i := len(f.Runs.Runs) - 1; i >= 0; i-- {
			r := f.Runs.Runs[i]
			label := fmt.Sprintf("Open run %s  %s  %s", r.ID, runStatus(r.Status, r.Measured.Probe && r.Measured.Timing), measuredMarks(r.Measured))
			if f.Confirm == r.ID {
				label = stBad.Render("Press d again to delete run " + r.ID + " and its report · any other key keeps it")
			}
			if f.Run != nil && r.ID == f.Run.ID {
				label += stMuted.Render("  shown")
			}
			out = append(out, action{label, "run", r.ID})
		}
	}
	return out
}

// --- 5 Result ----------------------------------------------------------------------------------------------

func resultLine(f Facts) (string, string) {
	if !f.outputDone() || f.alive() {
		return "open", "needs step 4"
	}
	res := f.Run.Results
	if res.Measured && res.Timing.TokS != nil {
		if res.Ceiling.TokS != nil && *res.Ceiling.TokS > 0 {
			return "pass", fmt.Sprintf("%.1f tokens per second · %.0f%% of the limit", *res.Timing.TokS, *res.Timing.TokS / *res.Ceiling.TokS * 100)
		}
		return "pass", fmt.Sprintf("%.1f tokens per second measured", *res.Timing.TokS)
	}
	if f.Run.Report != nil {
		return "open", "nothing measured yet · enter opens " + *f.Run.Report
	}
	return "open", "nothing measured yet"
}

func resultActions(f Facts) []action {
	out := []action{}
	if f.Run != nil && f.Run.Report != nil {
		out = append(out, action{"Open the full report (" + *f.Run.Report + ")", "report", ""})
	}
	if f.Path != "" && !f.alive() {
		out = append(out, action{"Measure again", "start", ""})
	}
	return out
}
