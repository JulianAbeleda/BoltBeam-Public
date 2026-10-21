package ui

import (
	"fmt"
	"github.com/charmbracelet/x/ansi"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"time"

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
	EngineScan  bool           // the engine scan is running (at boot, and on [ Scan again ]); Providers waits for it
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
	Confirm     string              // the run id waiting for a second enter before it is deleted
	CompareFrom string              // the saved run picked as "before"; Saved runs then picks the "after"
	Compared    *Compared           // the last comparison of two saved runs, or why it was refused
	Batch       int                 // streams decoded at once; 0 means 1
	BatchTyped  string              // digits typed on the batch picker
	Measurement string              // how roles are timed, as picked: in_model or generic; "" is Python's default here
	Workload    string              // what the Run measures: decode or prefill; "" is decode
	Replay      *seam.ReplayChoices // a replay's shapes and the servers running here, loaded when replay is picked
	ReplayShape string              // the picked shape; "" is the default
	ReplayArgv  string              // the server command the user confirmed for a replay; "" until then
	Plan        *seam.Emitted       // the gameplan Emit wrote for the run on screen (screen emit); nil before Emit
	Emitting    bool                // an Emit is running
	ShowLog     bool                // l: the running screen shows the raw log
	Frac        float64             // the bar, 0 to 1; it never goes back while one job runs
	fracJob     string              // the job Frac belongs to
	stageAt     time.Time           // when this screen saw the current step start
	stageDone   int                 // the step count stageAt was taken at
	logRoom     int                 // screen lines the log may take under the bar; 0: up to 20
}

// clock is the time the screen reads; a variable so a test can fix it.
var clock = time.Now

// advance moves the bar: the time in the current step counts toward its expected seconds.
func (f *Facts) advance(now time.Time) {
	if f.Job == nil {
		return
	}
	p := seam.ReadProgress(f.Tail)
	if job := f.Job.ID + f.Job.StartedAt; f.fracJob != job {
		f.fracJob, f.Frac, f.stageDone = job, 0, -1
	}
	mark := p.Done
	if p.Setup {
		mark = -2 // the setup stages: their time is not step 1's time
	}
	if mark != f.stageDone {
		first := f.stageDone == -1
		f.stageAt, f.stageDone = now, mark
		if t, err := time.Parse(time.RFC3339, f.Job.StartedAt); err == nil && first && p.Done == 0 {
			f.stageAt = t // the job began here: count from its start
		}
	}
	f.Frac = max(f.Frac, p.Fraction(now.Sub(f.stageAt).Seconds()))
	if !p.Finished {
		f.Frac = min(f.Frac, 0.99) // only "pipeline done" shows 100%
	}
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

// The screens: Setup (one line each for model, chip and engine, then Run), Run (progress, then the results),
// Saved runs, and one picker each for the model, the chip and the engine. Enter on a Setup line opens its picker.
const (
	pageSetup = iota
	pageRun
	pageSaved
	pageEmit
	pageModel
	pageChip
	pageEngine
	pageWorkload
	pageBatch
	pageMeasurement
	pageCompare
)

func none(Facts) string { return "" }

var steps = []step{
	{"Setup", "", none, func(Facts) (string, string) { return "", "" }, func(Facts, int) string { return "" }, setupActions},
	{"Run", "", aboutRun, resultLine, resultsBody, runActions},
	{"Saved runs", "", none, savedLine, savedBody, savedActions},
	{"Gameplan", "", aboutRun, emitLine, emitBody, emitActions},
	{"Model", "", none, modelLine, modelBody, modelChoices},
	{"Chip", "", none, chipLine, chipPickerBody, chipActions},
	{"Engine", "", none, engineLine, engineBody, engineActions},
	{"Workload", "", none, func(Facts) (string, string) { return "", "" }, workloadBody, workloadActions},
	{"Batch", "", none, batchLine, batchBody, batchActions},
	{"Measurement", "", none, func(Facts) (string, string) { return "", "" }, measurementBody, measurementActions},
	{"Compare", "", none, compareLine, compareRunsBody, compareActions},
}

// picker reports whether a page is one of the three pickers Setup opens.
func picker(page int) bool { return page >= pageModel && page <= pageMeasurement }

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
	if f.workload() == "replay" && f.ReplayArgv == "" {
		out = append(out, "the server to replay on")
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

// chipPickerBody keeps hardware selection separate from model role analysis.
func chipPickerBody(f Facts, width int) string {
	t := f.target()
	if t == nil || !f.chipKnown() {
		return stMuted.Render("Reading the chip…")
	}
	name := t.ID
	if f.Targets != nil && f.Targets.ThisMachine != nil && f.Targets.ThisMachine.Name != nil && t.ID == f.ThisMachine {
		name = *f.Targets.ThisMachine.Name
	}
	var b strings.Builder
	fmt.Fprintf(&b, "GPU: %s\n%s\n", name, memoryLine(f, *t))
	if t.ID == f.ThisMachine && f.Driver != "" {
		fmt.Fprintf(&b, "Driver: %s\n", f.Driver)
	}
	if f.MultiGpu != "" {
		fmt.Fprintf(&b, "%d GPUs: %s\n", f.GpuCount, f.MultiGpu)
	}
	return b.String()
}

// summary is one Setup line: the name, the chosen value, and a mark that enter opens its picker.
func summary(name, value string) string { return fmt.Sprintf("%-11s %s\t›", name, value) }

// notChosen is the value of a Setup line before anything is chosen.
var notChosen = stMuted.Render("not chosen")

// modelValue is the chosen model as Setup shows it: the file, its layers and its quants.
func modelValue(f Facts) string {
	switch {
	case f.Path == "":
		return notChosen
	case f.Profile == nil:
		return filepath.Base(f.Path)
	}
	return fmt.Sprintf("%s · %s layers · %s", filepath.Base(f.Path), count(f.Profile.LayerCount), strings.Join(f.Profile.Metadata.QuantTypes, "/"))
}

// chipValue is the chosen chip as Setup shows it: its id, this machine or a what if, and its memory speed.
func chipValue(f Facts) string {
	t := f.target()
	if t == nil || !f.chipKnown() {
		return notChosen
	}
	speed := ""
	if c := f.Ceiling; c != nil && c.Target.ID == t.ID && c.BandwidthSource != nil {
		speed = fmt.Sprintf(" · %.1f GB/s", c.PeakBandwidthGBs)
	} else if t.MemoryBandwidthGBs != nil {
		speed = fmt.Sprintf(" · %.1f GB/s", *t.MemoryBandwidthGBs)
	}
	text := "what if: " + t.ID + speed
	if f.thisChip() {
		text = t.ID + " · " + here() + speed
	}
	if f.GpuCount > 1 {
		text += " · " + layoutLabel(f)
	}
	return text
}

// engineValue is the chosen engine as Setup shows it: its name only; while the boot scan looks for engines, that.
func engineValue(f Facts) string {
	if f.EngineScan || (f.Providers == nil && f.target() != nil) { // the scan, then Python's provider list
		return strings.TrimSpace(f.Spin + " looking for engines…")
	}
	if p := f.engineRow(); p != nil && p.Available {
		return p.Provider
	}
	return notChosen
}

func setupActions(f Facts) []action {
	out := []action{
		{summary("Model", modelValue(f)), "page", fmt.Sprint(pageModel)},
		{summary("Chip", chipValue(f)), "page", fmt.Sprint(pageChip)},
		{summary("Engine", engineValue(f)), "page", fmt.Sprint(pageEngine)},
		{summary("Workload", workloadValue(f)), "page", fmt.Sprint(pageWorkload)},
		{summary("Batch", fmt.Sprint(f.batch())), "page", fmt.Sprint(pageBatch)},
		{summary("Measurement", measurementValue(f)), "page", fmt.Sprint(pageMeasurement)},
	}
	out = append(append(out, measurementFallback(f)...), head(""))
	switch m := f.missing(); {
	case f.alive():
		out = append(out, action{"[ Running… ] show the progress", "page", fmt.Sprint(pageRun)})
	case len(m) > 0:
		out = append(out, action{stMuted.Render("[ Run ] needs " + strings.Join(m, ", ")), "", ""})
	case f.measurementBlocked() != "":
		out = append(out, action{stMuted.Render("[ Run ] " + f.measurementBlocked()), "", ""})
	default:
		out = append(out, action{stAccent.Render("[ Run ]"), "analyze", ""})
	}
	out = append(out, emitAction(f))
	if f.Run != nil && f.outputDone() && !f.alive() && !f.ReadOnly {
		state := "not saved"
		if _, ok := f.savedAs(f.Run.ID); ok {
			state = "saved"
		}
		out = append(out, action{stMuted.Render("Last run " + shortRun(f.Run.ID) + with(f.Run.Measure) + " (" + state + ")"), "page", fmt.Sprint(pageRun)})
	}
	n := 0
	if f.Saved != nil {
		n = len(f.Saved.Runs)
	}
	return append(out, action{fmt.Sprintf("Saved runs (%d)\t›", n), "page", fmt.Sprint(pageSaved)})
}

// emitAction is the Setup row under Run. Emit turns the run on screen into a gameplan (what kernel to emit per role,
// worst first); it needs a finished run with a per-role table, so before one the row is muted with "Run first".
func emitAction(f Facts) action {
	switch {
	case f.Emitting:
		return action{stMuted.Render("[ Emit ] writing the gameplan…"), "", ""}
	case f.failedStage() != "":
		return action{stMuted.Render("[ Emit ] run failed; see Run for details"), "", ""}
	case f.canEmit():
		label := "[ Emit ]"
		if f.Run.Results.Loss.Replay != nil && f.Run.Results.Loss.Replay.Forced {
			label = "[ Emit replay report ]"
		} else if len(f.Run.Results.Loss.Roles) == 0 && f.hasOptimizationTables() {
			label = "[ Emit gap report ]"
		}
		if f.ReadOnly {
			label += " saved run " + f.Run.ID
		}
		return action{stAccent.Render(label), "emit", ""}
	}
	if f.Run != nil && f.outputDone() {
		return action{stMuted.Render("[ Emit ] no measured work breakdown; see report for " + f.Run.ID), "", ""}
	}
	return action{stMuted.Render("[ Emit ] Run first"), "", ""}
}

// canEmit requires a completed run with kernel-specific evidence, a measured breakdown, or a forced replay report.
func (f Facts) canEmit() bool {
	return f.Run != nil && f.outputDone() && !f.alive() && f.failedStage() == "" && (len(f.Run.Results.Loss.Roles) > 0 || f.hasOptimizationTables() || (f.Run.Results.Loss.Replay != nil && f.Run.Results.Loss.Replay.Forced))
}

func (f Facts) hasOptimizationTables() bool {
	if f.Run == nil {
		return false
	}
	loss := f.Run.Results.Loss
	if loss.WhereTokenGoes != nil && len(loss.WhereTokenGoes.Rows) > 0 {
		return true
	}
	if loss.Replay != nil {
		for _, table := range loss.Replay.Wheres {
			if (table.Status == "measured" || table.Status == "benchmark_only") && len(table.Rows) > 0 {
				return true
			}
		}
	}
	return false
}

// --- Gameplan (Emit) --------------------------------------------------------------------------------------------

func emitLine(f Facts) (string, string) {
	if f.Plan == nil {
		return "open", "no gameplan yet · Emit on Setup writes one from the run on screen"
	}
	if f.Plan.Plan.Kind == "gap_report" {
		return "pass", "gap report written · gameplan.md and gameplan.json"
	}
	if f.Plan.Plan.Kind == "replay_report" {
		return "pass", "checked replay report emitted; timing qualification remains disclosed"
	}
	return "pass", fmt.Sprintf("%d roles, worst first · gameplan.md and gameplan.json written to the run", len(f.Plan.Plan.Roles))
}

func timingScopeLine(scope *string, width int) string {
	if scope == nil || *scope == "" {
		return ""
	}
	return stMuted.Render(ansi.Wordwrap(*scope, width, " ")) + "\n\n"
}

// emitBody is the gameplan as Python wrote it: the same markdown, drawn in the screen's styles (renderMarkdown).
func emitBody(f Facts, width int) string {
	if f.Plan == nil {
		return stMuted.Render("No gameplan yet. Emit on Setup writes a gap report from the measured work, or kernel implementation instructions when supported.")
	}
	if f.Plan.Plan.Kind == "gap_report" && len(f.Plan.Plan.Tables) > 0 && f.Plan.Plan.Presentation.Summary != "" {
		var b strings.Builder
		b.WriteString(renderMarkdown(f.Plan.Plan.Presentation.Summary) + "\n")
		for _, table := range f.Plan.Plan.Tables {
			b.WriteString(whereBody(&table))
		}
		b.WriteString(renderMarkdown(strings.TrimRight(f.Plan.Plan.Presentation.Details, "\n")))
		return timingScopeLine(f.Plan.Plan.TimingScope, width) + b.String()
	}
	md := strings.TrimRight(f.Plan.Markdown, "\n")
	if scope := f.Plan.Plan.TimingScope; scope != nil {
		md = strings.TrimPrefix(md, *scope+"\n\n")
	}
	return timingScopeLine(f.Plan.Plan.TimingScope, width) + renderMarkdown(md)
}

func emitActions(f Facts) []action {
	out := []action{}
	if f.Plan != nil {
		out = append(out, action{"Open the gameplan (gameplan.md)", "gameplan", ""})
	}
	return append(out, action{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)})
}

// cleanAction offers to delete the unsaved runs from earlier sessions; enter asks twice.
func cleanAction(f Facts) []action {
	if f.OldWork == 0 {
		return nil
	}
	label := fmt.Sprintf("Delete the unsaved runs from earlier sessions (%d)", f.OldWork)
	if f.Confirm == "clean" {
		label = stBad.Render("Press enter again to delete them · any other key keeps them")
	}
	return []action{{label, "clean", ""}}
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
		out = append(out, action{mark + modelLabel(path), "file", path})
	}
	if f.Path != "" && !contains(f.Files, f.Path) {
		out = append(out, action{stAccent.Render("● ") + modelLabel(f.Path), "file", f.Path})
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
	case f.EngineScan || f.Providers == nil:
		return "run", "looking for engines on " + here() + "…"
	case p == nil:
		return "open", "pick an engine"
	case !p.Available:
		return "fail", p.Provider + " · " + stateWords(*p)
	}
	how := "roles not timed here"
	if p.Capture.Method != nil {
		how = "roles " + word(captureWords, *p.Capture.Method)
	}
	return "pass", p.Provider + " · " + how
}

func engineBody(f Facts, width int) string {
	if f.EngineScan || f.Providers == nil {
		return stMuted.Render("Looking for engines on " + here() + "… (the usual folders, no env vars needed)")
	}
	var b strings.Builder
	b.WriteString("The engine is the runtime that decodes the model. Only engines found on " + here() + " can be picked. " +
		"They are looked for at start; an env var (BOLTBEAM_LLAMA_BENCH, BOLTBEAM_TINYGRAD_ROOT, …) overrides what was found.\n\n")
	for _, p := range f.Providers.Providers {
		p = f.modelProvider(p)
		if !p.Available {
			continue // the picker lists these, with why, under one row
		}
		state := "can measure"
		how := "roles: not possible here (" + deref(p.Capture.Reason) + ")"
		if p.Capture.Method != nil {
			how = "roles: " + word(captureWords, *p.Capture.Method)
		}
		fmt.Fprintf(&b, "%s · %s\n  %s\n", p.Provider, state, stMuted.Render(how))
	}
	return b.String()
}

func engineActions(f Facts) []action {
	if f.EngineScan {
		return []action{note("  " + f.Spin + " looking for engines…")}
	}
	if f.Providers == nil {
		return nil
	}
	out := []action{}
	for _, p := range f.Providers.Providers {
		p = f.modelProvider(p)
		chosen := "  "
		if p.Provider == f.Engine {
			chosen = stAccent.Render("● ")
		}
		if p.Available {
			out = append(out, action{chosen + p.Provider, "engine", p.Provider})
		} else {
			out = append(out, note(fmt.Sprintf("  %-14s %s", p.Provider, stateWords(p))))
		}
	}
	return append(out, action{"  [ Scan again ] look for engines in the usual folders", "engines-scan", ""})
}

// stateWords is an engine that cannot run here in two words; the sentence why stays in --json.
func stateWords(p seam.ProviderRow) string {
	if p.State == "not_compatible" {
		return "not compatible"
	}
	return "not here"
}

// --- Results --------------------------------------------------------------------------------------------------

func resultsBody(f Facts, width int) string {
	if f.Run == nil {
		return stMuted.Render("No run yet. Run makes one.")
	}
	if f.alive() {
		return runningBody(f, width)
	}
	if f.failedStage() != "" {
		return failedBody(f, width)
	}
	var b strings.Builder
	b.WriteString(timingScopeLine(f.Run.Results.Loss.TimingScope, width))
	if f.Run.Measure != nil && f.Run.Measure.Status != "measured" {
		b.WriteString(measureBody(f, width) + "\n\n")
	}
	if f.Run.Results.Loss.Layout != nil {
		b.WriteString(layoutBody(*f.Run.Results.Loss.Layout) + "\n")
	}
	b.WriteString(resultBody(f, width))
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
		return []action{{"Stop the run", "stop", ""}, {"Back to setup", "page", fmt.Sprint(pageSetup)},
			{stMuted.Render("The run keeps going after you leave this screen."), "note", ""}}
	case f.failedStage() != "":
		return []action{{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)}}
	case f.ReadOnly:
		out = append(out, action{"Back to saved runs", "page", fmt.Sprint(pageSaved)},
			action{"Compare with another saved run (this one is before)", "comparefrom", f.Run.ID})
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

// --- Compare: two saved runs, as the seam words them -------------------------------------------------------------

// Compared is one comparison on screen: the two ids, the seam's lines, or the refusal.
type Compared struct {
	Before, After string
	Lines         []string
	Err           string
}

func compareLine(f Facts) (string, string) {
	switch {
	case f.Compared == nil:
		return "run", "comparing…"
	case f.Compared.Err != "":
		return "fail", "not compared"
	}
	return "pass", f.Compared.Before + " (before) and " + f.Compared.After + " (after)"
}

func compareRunsBody(f Facts, width int) string {
	c := f.Compared
	switch {
	case c == nil:
		return stMuted.Render("Comparing…")
	case c.Err != "":
		return stWarn.Render(glyphWarn+" Not compared: "+c.Err+".") + "\n" +
			stMuted.Render("Compare two runs of the same model, chip, engine and workload; two replays of one trace may differ in engine.")
	}
	return strings.Join(c.Lines, "\n")
}

func compareActions(f Facts) []action {
	return []action{{"Back to saved runs", "page", fmt.Sprint(pageSaved)}, {"[ Back to setup ]", "page", fmt.Sprint(pageSetup)}}
}

// crossEngines says the two saved runs are replays from different engines: replay-cross, not diff.
func (f Facts) crossEngines(a, b string) bool {
	if f.Saved == nil {
		return false
	}
	engine := map[string]string{}
	for _, r := range f.Saved.Runs {
		if r.Workload != nil && *r.Workload == "replay" {
			engine[r.ID] = r.Provider
		}
	}
	ea, oka := engine[a]
	eb, okb := engine[b]
	return oka && okb && ea != eb
}

func savedActions(f Facts) []action {
	out := []action{}
	if f.CompareFrom != "" {
		out = append(out, action{stInfo.Render("Pick the after run to compare with " + f.CompareFrom + ". The same model, chip and " +
			"engine compare row by row; two replays of one trace may differ in engine."), "note", ""})
	}
	if f.Saved != nil {
		for _, r := range f.Saved.Runs {
			pct := "not measured"
			switch {
			case r.ValidationFailed:
				pct = "FAILED validation"
			case r.PctOfLimit != nil:
				pct = fmt.Sprintf("%.0f%% of limit", *r.PctOfLimit)
			case r.RooflinePct != nil: // a replay: the session against its roofline
				pct = fmt.Sprintf("%.0f%% of roofline", *r.RooflinePct)
			case r.Measured:
				pct = "measured"
			}
			when := strings.Replace(r.SavedAt, "T", " ", 1)
			if len(when) > 16 {
				when = when[:16] // to the minute
			}
			engine := r.Provider
			if r.Workload != nil {
				engine += " " + *r.Workload
			}
			label := fmt.Sprintf("%s · %s · %s · %s · %s", deref(r.ModelID), deref(r.TargetID), engine, when, pct)
			if f.Confirm == r.ID {
				label = stBad.Render("Press d again to delete saved run " + r.ID + " · any other key keeps it")
			}
			if f.CompareFrom != "" {
				if r.ID != f.CompareFrom {
					out = append(out, action{label, "compare", r.ID})
				}
				continue
			}
			out = append(out, action{label, "opensaved", r.ID})
		}
	}
	if f.CompareFrom != "" {
		return append(out, action{"[ Cancel compare ]", "comparecancel", ""})
	}
	out = append(out, cleanAction(f)...)
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
	for _, p := range f.Providers.Providers {
		if p.Provider == name {
			p = f.modelProvider(p)
			return &p
		}
	}
	return nil
}

// modelProvider combines installed-engine availability with its declared formats.
func (f Facts) modelProvider(p seam.ProviderRow) seam.ProviderRow {
	if f.Profile == nil || len(p.Formats) == 0 {
		return p
	}
	format := f.Profile.Metadata.FormatFamily
	if format == "" {
		return p
	}
	for _, supported := range p.Formats {
		if format == supported {
			return p
		}
	}
	reason := p.Provider + " cannot read " + format + "; choose " + strings.Join(p.Formats, " or ")
	p.Available, p.State, p.Reason = false, "not_compatible", &reason
	return p
}

// with names a run's provider for a label: " · llama.cpp", or "" when the run does not say.
func with(m *seam.MeasureStatus) string {
	if m == nil || m.Provider == nil {
		return ""
	}
	if b := slices.Max(append([]int{1}, m.Batches...)); b > 1 {
		return fmt.Sprintf(" · %s · batch %d", *m.Provider, b)
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
	if f.alive() {
		return ""
	}
	return seam.ReadProgress(f.Tail).Failed
}

func aboutChip(f Facts) string {
	if t := f.target(); t != nil {
		return t.ID
	}
	return ""
}

func aboutRun(f Facts) string {
	if f.Run != nil && f.Run.ID != "" {
		return f.Run.ID
	}
	if f.Job != nil {
		return f.Job.ID
	}
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

// chipActions is this machine's chip only, with its cached profile, and the Autoscan row. Another chip (a what
// if) is chosen with --target, for scripts; the screen offers none.
func chipActions(f Facts) []action {
	if f.Targets == nil || f.ByFlag {
		return nil // set by flag: nothing to choose here
	}
	out := []action{}
	h := f.Targets.ThisMachine
	for i, t := range f.Targets.Targets {
		if t.ID == f.ThisMachine && f.ThisMachine != "" {
			words := here() + " (detected)"
			if h != nil {
				words = h.Words
			}
			out = append(out, chipRow(f, t.ID, i, len(t.ID), words))
		}
	}
	busy := ""
	if f.Scanning {
		busy = " " + f.Spin + " measuring…"
	}
	switch {
	case h == nil && len(out) == 0:
		out = append(out, note("No profile for this GPU yet"), action{"  [ Autoscan ] measure this GPU and save a profile" + busy, "autoscan", ""})
	case h == nil:
	case h.Status == "new":
		out = append(out, note(h.Words), action{"  [ Autoscan ] measure this GPU and save a profile" + busy, "autoscan", ""})
	case h.Source != nil && *h.Source == "generated":
		out = append(out, action{"  [ Measure again ] refresh this machine's profile" + busy, "autoscan", "remeasure"})
	case h.Status == "known":
		out = append(out, action{"  [ Autoscan ] check the profile still fits" + busy, "autoscan", ""})
	default:
		out = append(out, note(h.Words))
	}
	return append(out, layoutActions(f)...)
}

// chipRow is one chip that can be chosen, marked when it is the chosen one.
func chipRow(f Facts, id string, i, w int, words string) action {
	chosen := "  "
	if i == f.Target && f.chipKnown() {
		chosen = stAccent.Render("● ")
	}
	return action{fmt.Sprintf("%s%-*s  %s", chosen, w, id, stMuted.Render(words)), "chip", fmt.Sprint(i)}
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

// --- 4 Measure ---------------------------------------------------------------------------------------------

func measureLine(f Facts) (string, string) {
	if f.alive() {
		p := seam.ReadProgress(f.Tail)
		done, total := p.Done, p.Total
		now := ""
		for key, state := range seam.StageEvents(f.Tail) {
			if state == "running" {
				now = " · " + stageWord(key) + "…"
			}
		}
		if p.Setup {
			return "run", "starting…"
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
		return "", "" // the box above holds the bar, or what failed
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
	if s := res.Loss.Session; res.Measured && s != nil && s.Focus != nil && len(res.Loss.Runtimes) > 0 && res.Ceiling.TokS != nil {
		r := res.Loss.Runtimes[0]
		return "pass", fmt.Sprintf("%.1f prompt tokens per second at depth %s · %.0f%% of the limit", r.TokS, count(s.Focus), r.TokS / *res.Ceiling.TokS * 100)
	}
	if res.Measured && res.Timing.TokS != nil {
		if res.Ceiling.TokS != nil && *res.Ceiling.TokS > 0 {
			return "pass", fmt.Sprintf("%.1f tokens per second · %.0f%% of the limit", *res.Timing.TokS, *res.Timing.TokS / *res.Ceiling.TokS * 100)
		}
		return "pass", fmt.Sprintf("%.1f tokens per second measured", *res.Timing.TokS)
	}
	if rp := res.Loss.Replay; res.Measured && rp != nil && len(rp.Rows) > 0 {
		return "pass", fmt.Sprintf("replay measured · %d turns", len(rp.Rows))
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
	}
	if f.Run != nil && f.Run.Report != nil {
		out = append(out, action{"Open the full report (" + *f.Run.Report + ")", "report", ""})
	}
	return out
}

// --- Workload -------------------------------------------------------------------------------------------------

// workloads are what a Run can measure, in Setup's order; a workload not built yet is shown muted and not picked.
var workloads = []struct{ id, about, missing string }{
	{"decode", "one token at a time, after the prompt", ""},
	{"prefill", "the prompt read at once, at lengths up to the model's cap", ""},
	{"session", "a turn appended at depths up to the context that fits", ""},
	{"replay", "a recorded session replayed on your server's flags", ""},
}

// replayShape is the picked shape, else the default the shapes list names.
func (f Facts) replayShape() *seam.ReplayShape {
	if f.Replay == nil {
		return nil
	}
	for i, s := range f.Replay.Shapes {
		if s.Name == f.ReplayShape || (f.ReplayShape == "" && s.Name == f.Replay.Default) {
			return &f.Replay.Shapes[i]
		}
	}
	return nil
}

// workloadValue is the Setup line: the workload, and for a replay its trace in a few words.
func workloadValue(f Facts) string {
	if s := f.replayShape(); f.workload() == "replay" && s != nil {
		return "replay (" + s.Label + ")"
	}
	return f.workload()
}

// workload is the chosen workload; decode until another is picked.
func (f Facts) workload() string {
	if f.Workload == "" {
		return "decode"
	}
	return f.Workload
}

func workloadBody(Facts, int) string {
	return "Workload is what the Run measures. Decode times one token at a time. Prefill times the prompt read at once, " +
		"from 512 tokens up to the model's own cap. Session times a 512-token turn appended to a cache already holding " +
		"1K, then 4 times more, up to the context that fits before compaction. Replay plays a recorded session, as " +
		"token counts, through your own llama-server command and shows each turn against its roofline."
}

func workloadActions(f Facts) []action {
	out := []action{}
	for _, w := range workloads {
		if w.missing != "" {
			out = append(out, note(stMuted.Render(fmt.Sprintf("  %-8s %s", w.id, w.missing))))
			continue
		}
		chosen := "  "
		if w.id == f.workload() {
			chosen = stAccent.Render("● ")
		}
		out = append(out, action{fmt.Sprintf("%s%-8s %s", chosen, w.id, stMuted.Render(w.about)), "workload", w.id})
	}
	if f.workload() != "replay" || f.Replay == nil {
		return out
	}
	out = append(out, head("Trace"))
	pick := f.replayShape()
	for _, s := range f.Replay.Shapes {
		chosen := "  "
		if pick != nil && s.Name == pick.Name {
			chosen = stAccent.Render("● ")
		}
		out = append(out, action{chosen + s.Label + "  " + stMuted.Render(s.Words), "replayshape", s.Name})
	}
	out = append(out, head("Server to replay on (enter confirms it)"))
	if len(f.Replay.Servers) == 0 {
		out = append(out, note("  no llama-server is running here; start yours, then pick replay again"))
	}
	for _, argv := range f.Replay.Servers {
		cmd := strings.Join(argv, " ")
		chosen := "  "
		if cmd == f.ReplayArgv {
			chosen = stAccent.Render("● ")
		}
		out = append(out, action{chosen + cmd, "replayserver", cmd})
	}
	return out
}

// --- Batch ----------------------------------------------------------------------------------------------------

// batch is the chosen batch size; an engine that decodes one stream only runs batch 1.
func (f Facts) batch() int {
	if f.Batch < 1 || !f.batchOverOne() {
		return 1
	}
	return f.Batch
}

// batchOverOne says whether the chosen engine decodes more than one stream. Python decides (batch_over_one).
func (f Facts) batchOverOne() bool {
	p := f.engineRow()
	return p == nil || p.BatchOverOne == nil || *p.BatchOverOne
}

func batchLine(Facts) (string, string) { return "", "" }

func batchBody(f Facts, width int) string {
	text := "Batch is how many streams the engine decodes at once. Batch 1 is always timed too, so the two can be compared."
	if !f.batchOverOne() {
		text += "\n" + stMuted.Render(f.Engine+" decodes one stream only.")
	}
	return text
}

func batchActions(f Facts) []action {
	out := []action{}
	for _, n := range []int{1, 8, 32} {
		chosen := "  "
		if n == f.batch() {
			chosen = stAccent.Render("● ")
		}
		if n > 1 && !f.batchOverOne() {
			out = append(out, note(stMuted.Render(fmt.Sprintf("  %-4d not supported", n))))
			continue
		}
		out = append(out, action{chosen + fmt.Sprint(n), "batch", fmt.Sprint(n)})
	}
	if f.batchOverOne() {
		typed := f.BatchTyped
		if typed == "" {
			typed = stMuted.Render("type a number")
		}
		out = append(out, action{"  Other: " + typed, "batch", f.BatchTyped})
	}
	return out
}

// --- Measurement --------------------------------------------------------------------------------------------

// measurementOptions are the chosen engine's two ways to time roles here, from Python; nil before it answers.
func (f Facts) measurementOptions() *seam.MeasurementOptions {
	if p := f.engineRow(); p != nil {
		return p.Measurement
	}
	return nil
}

// measurementOption is the option the Run will use: the pick, else Python's default for this engine and chip.
func (f Facts) measurementOption() *seam.MeasurementOption {
	o := f.measurementOptions()
	if o == nil {
		return nil
	}
	id := f.Measurement
	if id == "" && o.Default != nil {
		id = *o.Default
	}
	for i := range o.Options {
		if o.Options[i].ID == id {
			return &o.Options[i]
		}
	}
	return nil
}

// roleTimeArg is the pipeline's --role-time for the pick: in-model or generic; "" leaves Python's auto.
func (f Facts) roleTimeArg() string {
	return map[string]string{"in_model": "in-model", "generic": "generic"}[f.Measurement]
}

// measurementBlocked is why Run cannot start with the picked measurement; "" when it can.
func (f Facts) measurementBlocked() string {
	if o := f.measurementOption(); f.Measurement != "" && o != nil && !o.Available {
		return deref(o.Reason)
	}
	return ""
}

// measurementValue is the Setup line: the method the Run uses, and when Python fell back, why.
func measurementValue(f Facts) string {
	o := f.measurementOption()
	if o == nil { // an older Python, or no engine yet: the pipeline's own default
		return "auto"
	}
	if !o.Available {
		return stMuted.Render(o.Label + " (cannot run here)")
	}
	return o.Label
}

// measurementFallback is the muted row under the Setup line when Python fell back from in-model; nil otherwise.
func measurementFallback(f Facts) []action {
	opts, o := f.measurementOptions(), f.measurementOption()
	if f.Measurement != "" || opts == nil || opts.Fallback == nil || o == nil || o.ID == "in_model" {
		return nil
	}
	return []action{note("  (in-model not available here: " + strings.TrimSuffix(strings.TrimPrefix(*opts.Fallback, "in-model "), " on this Mac") + ")")}
}

func measurementBody(f Facts, width int) string {
	return "Measurement is how BoltBeam times each role. In-model watches the real token and splits it exactly. " +
		"Generic times each kernel alone with BoltBeam's kernel timer: it compares kernels across engines and chips, " +
		"and its split of the token is an estimate."
}

// measurementActions are the two rows, the one in use marked; a row that cannot run here is muted with why.
// Each row carries its one-line description under it.
func measurementActions(f Facts) []action {
	opts := f.measurementOptions()
	if opts == nil {
		return []action{note("  Pick an engine first.")}
	}
	in := f.measurementOption()
	out := []action{}
	for _, o := range opts.Options {
		chosen := "  "
		if in != nil && in.ID == o.ID {
			chosen = stAccent.Render("● ")
		}
		if o.Available {
			out = append(out, action{chosen + o.Label, "measurement", o.ID})
		} else {
			out = append(out, note("  "+o.Label+" · "+deref(o.Reason)))
		}
		for _, l := range strings.Split(ansi.Wordwrap(o.About, 66, " "), "\n") { // fits an 80-column screen
			out = append(out, note("    "+l))
		}
	}
	return out
}
