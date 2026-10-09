package ui

import (
	"fmt"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// Facts is everything the checklist reads. Every step's state is a function of these seam facts, never of
// what the user clicked, so reopening the screen mid run lands on the same marks.
type Facts struct {
	Path        string
	Editing     bool
	Input       string
	Files       []string
	Reading     bool
	ModelErr    string
	Profile     *seam.Profile
	Targets     *seam.Targets
	Target      int
	ThisMachine string         // the target autoscan reads this machine's GPU as
	Driver      string         // this machine's GPU driver, read live; "" where the probe reports none
	Detected    bool           // the detect call answered (ThisMachine "" then means no registered chip matched)
	NewChip     string         // this GPU's name when no chip profile fits it yet: autoscan measures one
	Scanning    bool           // an autoscan is running
	ByFlag      bool           // --target chose the chip: shown as chosen by flag, not detected
	Picked      bool           // detection failed and the user picked the closest registered chip
	Others      *seam.Ceilings // this model's limit on every registered chip, read-only (step 3)
	GpuCount    int            // every GPU the driver lists (detect)
	MultiGpu    string         // "4 x NAME (limited support...)" when more than one; Python words it
	Engine      string         // the chosen provider: "llama.cpp" or "tinygrad" (called engine on screen)
	Layout      string         // how the engine uses the GPUs: one, layer or row
	Saved       *seam.Saved    // the saved runs (screen saved)
	SavedID     string         // the run on screen was saved, as this id
	SavedDir    string         // and here
	ReadOnly    bool           // the run on screen is a saved run, opened to read
	OldWork     int            // unsaved runs from earlier sessions, which the user may delete
	Ceiling     *seam.Ceiling
	CeilBusy    bool
	CeilErr     string
	Runs        *seam.Runs
	Run         *seam.Run
	Job         *jobs.Job
	Tail        []string
	Ready       *seam.CompareReady // can this machine compare kernels for the run (step 5)
	Providers   *seam.Providers    // the runtimes that can measure the chip here (step 4)
	CJob        *jobs.Job          // the compare job, id "<run>-compare"
	CTail       []string
	Spin        string
	Confirm     string // the run id waiting for a second enter before it is deleted
}

// goos is the platform the screen runs on; a variable so a test can draw the Linux screen on a Mac.
var goos = runtime.GOOS

// here is what the screen calls the machine it runs on: "this Mac" only on macOS.
func here() string {
	if goos == "darwin" {
		return "this Mac"
	}
	return "this machine"
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

// The screens: Setup (model, chip, engine, then Run), Run (progress, then the results), Saved runs.
const (
	pageSetup = iota
	pageRun
	pageSaved
)

var steps = []step{
	{"Setup", "", func(Facts) string { return "" }, setupLine, setupBody, setupActions},
	{"Run", "", aboutRun, resultLine, resultsBody, runActions},
	{"Saved runs", "", func(Facts) string { return "" }, savedLine, savedBody, savedActions},
}

// engineRow is this machine's provider row for the chosen engine, or nil.
func (f Facts) engineRow() *seam.ProviderRow { return f.provider(f.Engine) }

// missing names what Analyze still needs; empty when it can run.
func (f Facts) missing() []string {
	out := []string{}
	if f.Profile == nil {
		out = append(out, "a model")
	}
	if f.target() == nil || !f.chipKnown() {
		out = append(out, "a chip")
	}
	if p := f.engineRow(); p == nil || !p.Available {
		out = append(out, "an engine")
	}
	return out
}

// thisChip says whether the chosen chip is this machine's: only then does Analyze measure.
func (f Facts) thisChip() bool {
	t := f.target()
	return t != nil && t.ID == f.ThisMachine
}

// chipPageBody is the chip's facts, the speed limit on it, and the limit on other chips (read-only).
func chipPageBody(f Facts, width int) string {
	return chipBody(f, width) + "\n\n" + stHeader.Render("Speed limit") + "\n" + limitBody(f, width)
}

// --- Setup ----------------------------------------------------------------------------------------------------

func setupLine(f Facts) (string, string) {
	if f.alive() {
		return "run", "running " + aboutRun(f) + "…"
	}
	if m := f.missing(); len(m) > 0 {
		return "open", "needs " + strings.Join(m, ", ")
	}
	if !f.thisChip() {
		return "pass", "ready: the speed limit only (" + aboutChip(f) + " is not " + here() + ")"
	}
	return "pass", "ready to run"
}

func setupBody(f Facts, width int) string {
	var b strings.Builder
	b.WriteString(stHeader.Render("Select a model, a chip and an engine. Then press Run.") + "\n")
	b.WriteString("Run finds the speed limit, checks the GPU is free, measures with the engine and times each role.\n")
	if f.Ceiling != nil && f.Ceiling.Decode.TokS != nil && f.chipKnown() { // no limit for a chip nobody chose
		fmt.Fprintf(&b, "Speed limit on %s: %.1f tokens per second (%.1f ms per token).\n", aboutChip(f), *f.Ceiling.Decode.TokS, f.Ceiling.Decode.FloorMs)
		if s := f.Ceiling.BandwidthSource; s != nil {
			b.WriteString(stMuted.Render(fmt.Sprintf("From memory %.1f GB/s, %s.", f.Ceiling.PeakBandwidthGBs, *s)) + "\n")
		}
	}
	if f.MultiGpu != "" {
		b.WriteString(stWarn.Render(fmt.Sprintf("%d GPUs found: %s.", f.GpuCount, f.MultiGpu)) + "\n")
	}
	if t := f.target(); t != nil {
		b.WriteString(otherChips(f, t.ID))
	}
	return b.String()
}

func setupActions(f Facts) []action {
	out := []action{head("Select model")}
	out = append(out, modelChoices(f)...)
	out = append(out, head("Select chip"))
	out = append(out, chipActions(f)...)
	out = append(out, head("Select engine"))
	out = append(out, engineActions(f)...)
	out = append(out, head(""))
	switch m := f.missing(); {
	case f.alive():
		out = append(out, action{"[ Running… ] show the progress", "page", fmt.Sprint(pageRun)})
	case len(m) > 0:
		out = append(out, action{stMuted.Render("[ Run ] needs " + strings.Join(m, ", ")), "", ""})
	default:
		out = append(out, action{stAccent.Render("[ Run ]"), "analyze", ""})
	}
	if f.Run != nil && f.outputDone() && !f.alive() && !f.ReadOnly {
		state := "not saved"
		if _, ok := f.savedAs(f.Run.ID); ok {
			state = "saved"
		}
		out = append(out, action{"Last run " + shortRun(f.Run.ID) + with(f.Run.Measure) + " (" + state + ")", "page", fmt.Sprint(pageRun)})
	}
	n := 0
	if f.Saved != nil {
		n = len(f.Saved.Runs)
	}
	out = append(out, action{fmt.Sprintf("Saved runs (%d)", n), "page", fmt.Sprint(pageSaved)})
	if f.OldWork > 0 {
		label := fmt.Sprintf("Delete the unsaved runs from earlier sessions (%d)", f.OldWork)
		if f.Confirm == "clean" {
			label = stBad.Render("Press enter again to delete them · any other key keeps them")
		}
		out = append(out, action{label, "clean", ""})
	}
	return out
}

// savedAs is where a run was saved: from this session's save, or from the saved-runs list.
func (f Facts) savedAs(id string) (string, bool) {
	if f.SavedID == id && f.SavedDir != "" {
		return f.SavedDir, true
	}
	if f.Saved != nil {
		for _, r := range f.Saved.Runs {
			if r.ID == id {
				return r.Dir, true
			}
		}
	}
	return "", false
}

// head is a section heading inside a list: drawn, never selected.
func head(title string) action { return action{title, "head", ""} }

// note is a muted line inside a list: drawn, never selected.
func note(text string) action { return action{stMuted.Render(text), "note", ""} }

// skipped says the cursor never rests on this row.
func skipped(do string) bool { return do == "head" || do == "note" }

// modelChoices are the model files found, the chosen one marked, and a row to type another path.
func modelChoices(f Facts) []action {
	out := []action{}
	for _, path := range f.Files {
		mark := "  "
		if path == f.Path {
			mark = stAccent.Render("● ")
		}
		out = append(out, action{mark + path, "file", path})
	}
	if f.Path != "" && !contains(f.Files, f.Path) {
		out = append(out, action{stAccent.Render("● ") + f.Path, "file", f.Path})
	}
	typed := "  Type a path"
	if f.Editing {
		typed = "  " + f.Input + stCursor.Render("▏")
	}
	return append(out, action{typed, "edit", ""})
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

// --- Engine ---------------------------------------------------------------------------------------------------

func engineLine(f Facts) (string, string) {
	p := f.engineRow()
	switch {
	case f.Providers == nil:
		return "run", "reading the engines on " + here() + "…"
	case p == nil:
		return "open", "pick an engine"
	case !p.Available:
		return "fail", p.Provider + " cannot run here: " + deref(p.Reason)
	}
	how := "roles not timed here"
	if p.Capture.Method != nil {
		how = "roles " + word(captureWords, *p.Capture.Method)
	}
	return "pass", p.Provider + " · " + how
}

func engineBody(f Facts, width int) string {
	if f.Providers == nil {
		return stMuted.Render("Reading the engines on " + here() + "…")
	}
	var b strings.Builder
	b.WriteString("The engine is the runtime that decodes the model. Only engines found on " + here() + " can be picked.\n\n")
	for _, p := range f.Providers.Providers {
		state := "can measure"
		if !p.Available {
			state = "not here: " + deref(p.Reason)
		}
		how := "roles: not possible here (" + deref(p.Capture.Reason) + ")"
		if p.Capture.Method != nil {
			how = "roles: " + word(captureWords, *p.Capture.Method)
		}
		fmt.Fprintf(&b, "%s · %s\n  %s\n", p.Provider, state, stMuted.Render(how))
	}
	return b.String()
}

func engineActions(f Facts) []action {
	if f.Providers == nil {
		return nil
	}
	out := []action{}
	for _, p := range f.Providers.Providers {
		chosen := "  "
		if p.Provider == f.Engine {
			chosen = stAccent.Render("● ")
		}
		if p.Available {
			out = append(out, action{chosen + p.Provider, "engine", p.Provider})
		} else {
			out = append(out, action{stMuted.Render(chosen + p.Provider + " · not here: " + deref(p.Reason)), "", ""})
		}
	}
	return out
}

// --- Results --------------------------------------------------------------------------------------------------

func resultsBody(f Facts, width int) string {
	if f.Run == nil {
		return stMuted.Render("No run yet. Run makes one.")
	}
	var b strings.Builder
	if f.alive() || f.failedStage() != "" || (f.Run.Measure != nil && f.Run.Measure.Status != "measured") {
		b.WriteString(measureBody(f, width) + "\n\n")
	}
	if f.Run.Results.Loss.Layout != nil {
		b.WriteString(layoutBody(*f.Run.Results.Loss.Layout) + "\n")
	}
	if !f.alive() {
		b.WriteString(resultBody(f, width))
	}
	return b.String()
}

// layoutBody is the limit's inputs with their sources. One GPU: one line, the read bandwidth the limit uses and
// where it came from. More GPUs: the formula and every input, labelled limited support.
func layoutBody(l seam.LayoutLimit) string {
	if l.Gpus < 2 {
		for _, in := range l.Inputs {
			if v, ok := in.Value.(float64); ok && in.Unit == "GB/s" {
				return stMuted.Render(fmt.Sprintf("Limit from memory %.1f GB/s, %s", v, shortSource(in.Source))) + "\n"
			}
		}
		return ""
	}
	var b strings.Builder
	head := fmt.Sprintf("Limit for %s on %d GPUs", l.Label, l.Gpus)
	if l.Ms != nil {
		head += fmt.Sprintf(": %.3f ms per token", *l.Ms)
	}
	b.WriteString(stHeader.Render(head) + "\n")
	if l.Support != nil {
		b.WriteString(stWarn.Render(*l.Support) + "\n")
	}
	if l.Reason != nil {
		b.WriteString("Not derived: " + *l.Reason + "\n")
	}
	if l.Formula != "" {
		b.WriteString("How: " + l.Formula + "\n")
	}
	for _, in := range l.Inputs {
		fmt.Fprintf(&b, "  %s: %v %s · %s\n", in.What, in.Value, in.Unit, stMuted.Render(in.Source))
	}
	return b.String()
}

func runActions(f Facts) []action {
	out := []action{}
	switch {
	case f.alive():
		out = append(out, action{"Stop the run", "stop", ""})
	case f.ReadOnly:
		out = append(out, action{"Back to saved runs", "page", fmt.Sprint(pageSaved)})
	case f.Run != nil && f.outputDone():
		if dir, ok := f.savedAs(f.Run.ID); ok {
			out = append(out, action{mark("pass") + " Saved to " + dir, "", ""})
		} else {
			out = append(out, action{stAccent.Render("[ Save run ]"), "save", ""})
		}
	}
	if f.Run != nil && f.Run.Report != nil && !f.alive() {
		out = append(out, action{"Open the full report (" + *f.Run.Report + ")", "report", ""})
	}
	return append(out, action{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)})
}

// --- Saved runs -------------------------------------------------------------------------------------------------

func savedLine(f Facts) (string, string) {
	if f.Saved == nil {
		return "run", "reading the saved runs…"
	}
	noun := "saved runs"
	if len(f.Saved.Runs) == 1 {
		noun = "saved run"
	}
	return "pass", fmt.Sprintf("%d %s in %s", len(f.Saved.Runs), noun, f.Saved.Root)
}

func savedBody(f Facts, width int) string {
	var b strings.Builder
	b.WriteString("A run is temporary until you save it. Saving keeps the run, its report, results.json and summary.txt.\n")
	b.WriteString("Enter opens a saved run. Press d twice to delete one.\n")
	return b.String()
}

func savedActions(f Facts) []action {
	out := []action{}
	if f.Saved != nil {
		for _, r := range f.Saved.Runs {
			pct := "not measured"
			if r.PctOfLimit != nil {
				pct = fmt.Sprintf("%.0f%% of limit", *r.PctOfLimit)
			}
			when := strings.Replace(r.SavedAt, "T", " ", 1)
			if len(when) > 16 {
				when = when[:16] // to the minute
			}
			label := fmt.Sprintf("%s · %s · %s · %s · %s", deref(r.ModelID), deref(r.TargetID), r.Provider, when, pct)
			if f.Confirm == r.ID {
				label = stBad.Render("Press d again to delete saved run " + r.ID + " · any other key keeps it")
			}
			out = append(out, action{label, "opensaved", r.ID})
		}
	}
	return append(out, action{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)})
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

// registered says whether this machine's detected chip is in the chip list.
func (f Facts) registered() bool {
	if f.ThisMachine == "" || f.Targets == nil {
		return false
	}
	for _, t := range f.Targets.Targets {
		if t.ID == f.ThisMachine {
			return true
		}
	}
	return false
}

// mayPick: only when detection failed may the user choose a chip, and only the closest registered one.
func (f Facts) mayPick() bool { return !f.ByFlag }

// chipKnown: steps 3, 4 and 5 use the detected chip, the flag's chip, or the one picked after detection failed.
func (f Facts) chipKnown() bool { return f.ByFlag || f.registered() || f.Picked }

// runProvider is the runtime the shown run was measured with; "" before a run says.
func (f Facts) runProvider() string {
	if f.Run == nil {
		return ""
	}
	if f.Run.Measure != nil && f.Run.Measure.Provider != nil {
		return *f.Run.Measure.Provider
	}
	return f.Run.Results.Loss.Provider
}

// provider is this machine's row for a runtime, or nil.
func (f Facts) provider(name string) *seam.ProviderRow {
	if f.Providers == nil {
		return nil
	}
	for i, p := range f.Providers.Providers {
		if p.Provider == name {
			return &f.Providers.Providers[i]
		}
	}
	return nil
}

// with names a run's provider for a label: " · llama.cpp", or "" when the run does not say.
func with(m *seam.MeasureStatus) string {
	if m == nil || m.Provider == nil {
		return ""
	}
	return " · " + *m.Provider
}

func (f Facts) comparing() bool { return f.CJob != nil && f.CJob.Alive }

// compareID is the job id of a run's kernel comparison.
func compareID(run string) string { return run + "-compare" }

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

// memoryFrom words a memory-speed source: a fact status from the registry ("from measured") or this machine's
// own ("measured on this GPU (native CUDA probe, 2026-10-09)").
func memoryFrom(source string) string {
	if strings.HasPrefix(source, "measured on") || strings.HasPrefix(source, "registry figure") {
		return source
	}
	return "from " + source
}

// shortSource is a limit input's source in a few words; Python's layout.short_source words the same.
func shortSource(s string) string {
	for _, p := range [][2]string{{"native CUDA read probe", "native CUDA probe"}, {"Metal read probe", "Metal read probe"},
		{"tinygrad read-sum", "tinygrad read-sum, a lower bound"}} {
		if strings.Contains(s, p[0]) {
			when := ""
			if m := dateRe.FindString(s); m != "" {
				when = ", " + m
			}
			return "measured on this GPU (" + p[1] + when + ")"
		}
	}
	if strings.HasPrefix(s, "registry figure") {
		return "registry figure, not measured on this GPU"
	}
	return s
}

var dateRe = regexp.MustCompile(`\d{4}-\d{2}-\d{2}`)

// memoryLine is the chip's memory speed as the speed limit uses it: this machine's (measured, with its source)
// when the limit carries one for this chip, else the registry's figure.
func memoryLine(f Facts, t seam.Target) string {
	if c := f.Ceiling; c != nil && c.BandwidthSource != nil && c.Target.ID == t.ID {
		return fmt.Sprintf("memory %.1f GB/s, %s", c.PeakBandwidthGBs, *c.BandwidthSource)
	}
	return gbs(t.MemoryBandwidthGBs)
}

func chipLine(f Facts) (string, string) {
	t := f.target()
	if t == nil {
		return "open", "reading the chip list…"
	}
	text := t.ID
	switch {
	case f.ByFlag:
		text += " · chosen by flag, not detected"
	case !f.Detected:
		return "run", "reading this machine's GPU…"
	case f.Picked && t.ID != f.ThisMachine:
		text += " · not " + here() + ": the speed limit only"
	case f.registered() && t.ID == f.ThisMachine:
		text += " · " + here() + " (detected)"
	case !f.Picked && f.NewChip != "":
		return "open", "new chip: " + f.NewChip + " has no profile yet · Autoscan measures it"
	case !f.Picked:
		return "open", "not detected: no registered chip matches " + here() + " · enter to pick the closest"
	default:
		text += " · picked: " + here() + " is not a registered chip"
	}
	if f.GpuCount > 1 {
		text += " · " + layoutLabel(f)
	}
	if !t.HasCeiling {
		return "crossed", text + " · no measured speeds, so no speed limit"
	}
	return "pass", text + " · " + memoryLine(f, *t)
}

func chipActions(f Facts) []action {
	if f.Targets == nil || f.ByFlag {
		return nil // set by flag: nothing to choose here
	}
	if len(f.Targets.Groups) > 0 {
		return append(chipGroups(f), layoutActions(f)...)
	}
	out := []action{}
	for i, t := range f.Targets.Targets {
		where, limit := "", ""
		if t.ID == f.ThisMachine {
			where = here() + " (detected)"
		}
		if !t.HasCeiling {
			limit = mark("crossed") + " no speed limit"
		}
		chosen := "  "
		if i == f.Target {
			chosen = stAccent.Render("● ")
		}
		out = append(out, action{fmt.Sprintf("%s%-14s %-22s %s", chosen, t.ID, where, limit), "chip", fmt.Sprint(i)})
	}
	return append(out, layoutActions(f)...)
}

// chipGroups draws Python's chip groups: this machine (with autoscan), measured chips, chips not measured yet,
// and the families folded into one line. Only this machine and measured chips can be chosen.
func chipGroups(f Facts) []action {
	index, w := map[string]int{}, 0
	for i, t := range f.Targets.Targets {
		index[t.ID], w = i, max(w, len(t.ID))
	}
	out := []action{}
	for _, g := range f.Targets.Groups {
		if g.Folded {
			ids := []string{}
			for _, c := range g.Chips {
				ids = append(ids, c.ID)
			}
			if len(ids) > 0 {
				out = append(out, note(g.Title+": "+strings.Join(ids, ", ")+" · "+g.Chips[0].Words))
			}
			continue
		}
		if g.Key == "this" {
			out = append(out, thisMachineRows(f, g, w)...)
			continue
		}
		if len(g.Chips) == 0 {
			continue
		}
		out = append(out, note(g.Title))
		for _, c := range g.Chips {
			if !g.Selectable {
				out = append(out, note(fmt.Sprintf("  %-*s  %s", w, c.ID, c.Words)))
				continue
			}
			out = append(out, chipRow(f, c.ID, index[c.ID], w, c.Words))
		}
	}
	return out
}

// chipRow is one chip that can be chosen, marked when it is the chosen one.
func chipRow(f Facts, id string, i, w int, words string) action {
	chosen := "  "
	if i == f.Target && f.chipKnown() {
		chosen = stAccent.Render("● ")
	}
	return action{fmt.Sprintf("%s%-*s  %s", chosen, w, id, stMuted.Render(words)), "chip", fmt.Sprint(i)}
}

// thisMachineRows is this machine's chip and the autoscan row: keep the fitting profile, or measure a new one.
func thisMachineRows(f Facts, g seam.ChipGroup, w int) []action {
	out := []action{note("This machine")}
	h := f.Targets.ThisMachine
	if h == nil {
		return out
	}
	for _, t := range g.Chips {
		for i, row := range f.Targets.Targets {
			if row.ID == t.ID {
				out = append(out, chipRow(f, t.ID, i, w, t.Words))
			}
		}
	}
	busy := ""
	if f.Scanning {
		busy = " " + f.Spin + " measuring…"
	}
	switch {
	case h.Status == "new":
		out = append(out, action{fmt.Sprintf("  [ Autoscan ] %s is new: measure it and save a profile%s", deref(h.Name), busy), "autoscan", ""})
		out = append(out, note("  "+h.Words))
	case h.Source != nil && *h.Source == "generated":
		out = append(out, action{"  [ Measure again ] refresh this machine's profile" + busy, "autoscan", "remeasure"})
	case h.Status == "known":
		out = append(out, action{"  [ Autoscan ] check the profile still fits" + busy, "autoscan", ""})
	default:
		out = append(out, note("  "+h.Words))
	}
	return out
}

// layoutLabel names the chosen GPU layout on a machine with more than one GPU.
func layoutLabel(f Facts) string {
	for _, l := range layoutRows(f) {
		if l.ID == f.layout() {
			return l.Label
		}
	}
	return "one GPU"
}

func (f Facts) layout() string {
	if f.Layout == "" {
		return "one"
	}
	return f.Layout
}

// layoutRows are the chosen engine's layouts on this machine (Python decides which can run).
func layoutRows(f Facts) []seam.LayoutRow {
	if p := f.engineRow(); p != nil {
		return p.Layouts
	}
	return nil
}

// layoutActions are part of the chip choice when the machine has more than one GPU.
func layoutActions(f Facts) []action {
	if f.GpuCount < 2 {
		return nil
	}
	out := []action{{stWarn.Render(fmt.Sprintf("%d GPUs: %s", f.GpuCount, f.MultiGpu)), "", ""}}
	for _, l := range layoutRows(f) {
		chosen := "  "
		if l.ID == f.layout() {
			chosen = stAccent.Render("● ")
		}
		if l.Available {
			out = append(out, action{chosen + "Use " + l.Label, "layout", l.ID})
		} else {
			out = append(out, action{stMuted.Render(chosen + "Use " + l.Label + " · " + deref(l.Reason)), "", ""})
		}
	}
	return out
}

// --- 3 Speed limit -----------------------------------------------------------------------------------------

func limitLine(f Facts) (string, string) {
	switch {
	case f.Profile == nil:
		return "open", "needs step 1"
	case !f.chipKnown():
		return "open", "needs step 2"
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
	case f.Run.Measure != nil && f.Run.Measure.Status == "measured" && deref(f.Run.Measure.Probe) == "absent":
		// timing is what the result needs; the open probe request is named, never hidden
		return "pass", "run " + shortRun(f.Run.ID) + with(f.Run.Measure) + " · timing " + mark("pass") + " · no probe for this chip"
	case len(f.Run.Blocked) > 0 && f.Run.Measure != nil && f.Run.Measure.Status == "skipped":
		return "wait", "planned · not measured here: " + deref(f.Run.Measure.Reason) + " · enter for what to run"
	case len(f.Run.Blocked) == 1:
		n := f.Run.Blocked[0]
		return "wait", fmt.Sprintf("planned · needs %s (%s)", word(plainNeed, n.Need), n.Request)
	case len(f.Run.Blocked) > 1:
		return "wait", "planned · needs building-block tests and a timing trace"
	}
	return "pass", "run " + shortRun(f.Run.ID) + with(f.Run.Measure) + " · " + measuredMarks(f.Run.Measured)
}

// shortRun is the run's number, the part after the model and chip.
func shortRun(id string) string {
	if i := strings.LastIndex(id, "-"); i >= 0 {
		return id[i+1:]
	}
	return id
}

// --- 5 Result ----------------------------------------------------------------------------------------------

func resultLine(f Facts) (string, string) {
	if f.alive() || f.failedStage() != "" {
		return measureLine(f) // the progress bar and the stage in flight, or what failed
	}
	if !f.outputDone() {
		return "open", "not run yet"
	}
	if f.comparing() {
		p := seam.ReadCompare(f.CTail)
		now := ""
		if p.Now != "" {
			now = " · " + p.Now + " " + compareDoing[p.Doing] + "…"
		}
		if p.Total == 0 {
			return "run", "working" + now
		}
		return "run", fmt.Sprintf("comparing kernels %s %d of %d%s", bar(float64(len(p.Done))/float64(p.Total), 14), len(p.Done), p.Total, now)
	}
	if p := seam.ReadCompare(f.CTail); p.Failed != "" {
		return "fail", "comparing kernels failed: " + p.Failed
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

// compareDoing words what the compare job is doing for the role in flight.
var compareDoing = map[string]string{"search": "searching kernels", "ab": "timing the whole model", "time": "timing in the model"}

func resultActions(f Facts) []action {
	out := []action{}
	switch {
	case f.comparing():
		out = append(out, action{"Stop comparing (x)", "stop", ""})
	case f.outputDone() && !f.alive():
		name := f.runProvider()
		if p := f.provider(name); p != nil && p.Available && p.Capture.Method != nil {
			out = append(out, action{"Time each role in " + name, "roletime", ""})
		}
		if f.Ready != nil && f.Ready.Ready && f.Ready.Applies {
			out = append(out, action{"Compare kernels per role", "compare", ""})
		}
	}
	if f.Run != nil && f.Run.Report != nil {
		out = append(out, action{"Open the full report (" + *f.Run.Report + ")", "report", ""})
	}
	return out
}
