package ui

import (
	"cmp"
	"errors"
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
	ModelNeeds  *seam.ForkNeed // the fork the model needs, when only a fork reads its format (never picked for the user)
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
	Recommend   *seam.Recommend    // the recommended setup for this model, chip and engine (recommendKey)
	RecKey      string             // what Recommend was asked for
	RecErr      string
	RecChecking bool      // Check is running: one real engine start
	CJob        *jobs.Job // the compare job, id "<run>-compare"
	CTail       []string
	Spin        string
	Confirm     string              // the run id waiting for a second enter before it is deleted
	ComparePick bool                // Saved runs is picking the first of two runs to compare
	CompareFrom string              // the first run picked; Saved runs then picks the one to set beside it
	Compared    *Compared           // the last comparison of two saved runs, or why it was refused
	Batch       int                 // streams decoded at once; 0 means 1
	BatchTyped  string              // digits typed on the batch picker
	Measurement string              // how roles are timed, as picked: in_model or generic; "" is Python's default here
	Settings    map[string]string   // engine settings asked for, by name (capture_settings.ASKED); cleared with a new engine
	Form        string              // the open form as drawn (forms.go): it takes the place of the page's rows
	Repeats     int                 // times Run measures the setup; only the n Python offers counts (repeats)
	MJob        *jobs.Job           // Measure MTP's job for the model and chip (mtpID)
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
	cols        int                 // the width a row is drawn in (DetailActions); 0 outside a frame: an 80-column screen

	// Upgrade engines: each engine's build against upstream's newest release (screen engines --latest), why that look
	// failed (every standing unknown), and the engine whose newer release the Engine picker's Upgrade is installing.
	Latest    []seam.EngineStanding
	LatestErr string
	Upgrading string

	// Check this machine (boltbeam doctor): its rows, why it could not run, and whether it is running now.
	Doctor     *seam.Doctor
	DoctorErr  string
	DoctorBusy bool

	Fetch FetchForm // Fetch runs (fetch.go)
	Speed SpeedUp   // Speed up (speedup.go)
	// ForkAt is the fork ←→ rests on in each engine family on the Engine picker (families.go), by family
	ForkAt map[string]string

	// The Results screen (render_plain.go): the run's plain block (screen plain), and which of layer 2's sections are open.
	Plain *seam.Plain
	Open  Sections
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
	pageSettings
	pageRepeats
	pageCompare
	pageRecommend
	pageDoctor
	pageFetch
)

func none(Facts) string { return "" }

var steps = []step{
	{"Setup", "", none, func(Facts) (string, string) { return "", "" }, func(Facts, int) string { return "" }, setupActions},
	{"Run", "", aboutRunFrom, resultLine, resultsBody, runActions},
	{"Saved runs", "", none, savedLine, savedBody, savedActions},
	{"Gameplan", "", aboutRun, emitLine, emitBody, emitActions},
	{"Model", "", none, modelLine, modelBody, modelChoices},
	{"Chip", "", none, chipLine, chipPickerBody, chipActions},
	{"Engine", "", none, engineLine, engineBody, engineActions},
	{"Workload", "", none, func(Facts) (string, string) { return "", "" }, workloadBody, workloadActions},
	{"Batch", "", none, batchLine, batchBody, batchActions},
	{"Measurement", "", none, func(Facts) (string, string) { return "", "" }, measurementBody, measurementActions},
	{"Engine settings", "", none, func(Facts) (string, string) { return "", "" }, settingsBody, settingsActions},
	{"Repeats", "", none, func(Facts) (string, string) { return "", "" }, repeatsBody, repeatsActions},
	{"Compare", "", none, compareLine, compareRunsBody, compareActions},
	{"Recommended setup", recommendHint, none, func(Facts) (string, string) { return "", "" }, recommendBody, recommendActions},
	{"This machine", "", none, doctorLine, doctorBody, doctorActions},
	{"Fetch runs", "", none, fetchLine, fetchBody, fetchActions},
}

// picker reports whether a page is one of the three pickers Setup opens.
func picker(page int) bool { return page >= pageModel && page <= pageRepeats }

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

// runWhat is what a Run of each workload measures, in a few words.
var runWhat = map[string]string{"decode": "decode speed", "prefill": "prompt-reading speed",
	"session": "speed as the context fills", "replay": "a replayed session"}

// runHelp is what pressing Run does with the setup on screen: what it measures, of which model, where, with which
// engine. On a chip that is not this machine's, Run plans only. No time is said: nothing before a run gives one.
func runHelp(f Facts) string {
	model := strings.TrimSuffix(filepath.Base(f.Path), filepath.Ext(f.Path))
	engine := f.Engine
	if p := f.engineRow(); p != nil {
		engine = p.Provider
	}
	if !f.thisChip() {
		return fmt.Sprintf("Plans a run of %s on %s with %s; it measures only on %s's own chip", model, f.target().ID, engine, here())
	}
	what := runWhat[f.workload()]
	if f.batch() > 1 {
		what += fmt.Sprintf(" at batch %d", f.batch())
	}
	text := fmt.Sprintf("Measures %s: %s, %s, %s", what, model, here(), engine)
	if n := f.repeats(); n > 1 {
		text += fmt.Sprintf(", %d times", n)
	}
	return text
}

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
	return text
}

// engineValue is the chosen engine as Setup shows it: its name, and its layout when the machine has more than one
// GPU; while the boot scan looks for engines, that.
func engineValue(f Facts) string {
	if f.EngineScan || (f.Providers == nil && f.target() != nil) { // the scan, then Python's provider list
		return strings.TrimSpace(f.Spin + " looking for engines…")
	}
	if p := f.engineRow(); p != nil && p.Available {
		return familyValue(f, p.Provider) + layoutSuffix(f)
	}
	return notChosen
}

// setupLine is one Setup line before it is drawn: its name, its value and the picker enter opens.
type setupLine struct {
	name, value string
	page        int
}

// setupActions are Setup's lines in groups, a blank line between: the machine; what is measured (model, chip,
// engine); how the engine is set up; what the run times; Run and Emit, each saying what it does; the runs. The name
// column is as wide as the longest name.
func setupActions(f Facts) []action {
	groups := [][]setupLine{
		{{"Machine", doctorValue(f), pageDoctor}},
		{{"Model", modelValue(f), pageModel}, {"Chip", chipValue(f), pageChip}, {"Engine", engineValue(f), pageEngine}},
		// the recommendation proposes the context, speed and engine flags; Settings, under it, changes them
		{{"Recommended setup", recommendValue(f), pageRecommend}},
		{{"Workload", workloadValue(f), pageWorkload}, {"Batch", fmt.Sprint(f.batch()), pageBatch},
			{"Measurement", measurementValue(f), pageMeasurement}},
	}
	if len(f.settingsOffered()) > 0 { // what the run asks the engine for, when the engine takes anything
		groups[2] = append(groups[2], setupLine{"Settings", settingsValue(f), pageSettings})
	}
	if f.repeatRuns() > 1 {
		groups[3] = append(groups[3], setupLine{"Repeats", repeatsValue(f), pageRepeats})
	}
	w := 0
	for _, g := range groups {
		for _, l := range g {
			w = max(w, len(l.name))
		}
	}
	under := func(text string) action { return note(strings.Repeat(" ", w+2) + text) } // in the value column
	out := []action{}
	for i, g := range groups {
		if i > 0 {
			out = append(out, head(""))
		}
		for _, l := range g {
			out = append(out, action{fmt.Sprintf("%-*s  %s\t›", w, l.name, l.value), "page", fmt.Sprint(l.page)})
			switch {
			case l.page == pageModel && f.ModelErr != "" && !f.Reading:
				for _, l := range strings.Split(ansi.Wordwrap(glyphFail+" "+f.ModelErr, cmp.Or(max(f.cols-w-2, 0), 1<<10), " "), "\n") {
					out = append(out, action{strings.Repeat(" ", w+2) + stBad.Render(l), "note", ""})
				}
			case l.page == pageEngine && engineNotice(f) != "":
				out = append(out, under(engineNotice(f)))
			}
		}
		if i == 3 {
			out = append(out, measurementFallback(f)...)
		}
	}
	out = append(out, head(""))
	switch m := f.missing(); {
	case f.alive():
		out = append(out, action{button(f, w, "[ Running… ]", "Shows the run's progress", true), "page", fmt.Sprint(pageRun)})
	case needsFork(f) != "": // only a fork reads this model: say which, never switch to it
		out = append(out, action{button(f, w, "[ Run ]", needsFork(f), false), "", ""})
	case f.ModelErr != "" && f.Path != "":
		out = append(out, action{button(f, w, "[ Run ]", "Pick a model BoltBeam can read: Model says why", false), "", ""})
	case len(m) > 0:
		out = append(out, action{button(f, w, "[ Run ]", "Needs "+strings.Join(m, ", ")+": pick it above", false), "", ""})
	case f.measurementBlocked() != "":
		out = append(out, action{button(f, w, "[ Run ]", capFirst(f.measurementBlocked()), false), "", ""})
	default:
		out = append(out, action{button(f, w, "[ Run ]", runHelp(f), true), "analyze", ""})
	}
	out = append(out, emitAction(f, w), head(""))
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

// standing is Python's word on an engine's build against upstream's newest release; nil for an engine with no
// release source, or before the look answered.
func (f Facts) standing(engine string) *seam.EngineStanding {
	for i := range f.Latest {
		if f.Latest[i].Engine == engine {
			return &f.Latest[i]
		}
	}
	return nil
}

// engineNotice is Setup's line under Engine when the picked engine is not on upstream's newest release: behind (with
// where to upgrade), waiting for its comparison, or unknown. On the latest release, or before Python answered, none.
func engineNotice(f Facts) string {
	switch p, s := f.engineRow(), f.standing(f.Engine); {
	case p == nil || !p.Available:
		return ""
	case f.LatestErr != "":
		return p.Provider + ": latest release unknown (" + f.LatestErr + ")"
	case s == nil || s.State == "up_to_date":
		return ""
	case s.State == "behind":
		return p.Provider + " " + s.Words + ": Upgrade in Engine"
	default:
		return p.Provider + ": " + s.Words
	}
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
		if path == f.Path && f.ModelErr != "" && !f.Reading { // why the chosen file cannot be measured, under it
			out = append(out, action{"    " + stBad.Render(glyphFail+" "+f.ModelErr), "note", ""})
		}
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
	return "pass", p.Provider + layoutSuffix(f) + " · " + how
}

func engineBody(f Facts, width int) string {
	if f.EngineScan || f.Providers == nil {
		return stMuted.Render("Looking for engines on " + here() + "… (the usual folders, no env vars needed)")
	}
	var b strings.Builder
	families := f.Providers.Families != nil
	if families {
		b.WriteString("The engine is the runtime that runs the model. Each row is a family: its root (upstream) and its " +
			"forks. ↑↓ moves between families, ←→ between a family's forks; enter picks the one in brackets.\n\n" +
			"Only a fork found on " + here() + " that BoltBeam can run is picked. Engines are looked for at start; " +
			"an env var (BOLTBEAM_LLAMA_BENCH, …) overrides what was found.\n\n")
	} else {
		b.WriteString("The engine is the runtime that decodes the model. Only engines found on " + here() + " can be picked. " +
			"They are looked for at start; an env var (BOLTBEAM_LLAMA_BENCH, BOLTBEAM_TINYGRAD_ROOT, …) overrides what was found.\n\n")
	}
	b.WriteString(layoutLegend(f))
	for _, p := range f.Providers.Providers {
		p = f.modelProvider(p)
		if !p.Available || (families && p.Provider != f.Engine) { // by family: the picked engine's roles only
			continue // the picker lists these, with why, under one row
		}
		state := "can measure"
		how := "roles: not possible here (" + deref(p.Capture.Reason) + ")"
		if p.Capture.Method != nil {
			how = "roles: " + word(captureWords, *p.Capture.Method)
		}
		fmt.Fprintf(&b, "%s · %s\n  %s\n", p.Provider, state, stMuted.Render(how))
		b.WriteString(layoutsHere(f, p))
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
	if f.Providers.Families != nil { // one row per family, ←→ between its forks
		return familyActions(f)
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
			out = append(out, layoutActions(f, p)...)
			out = append(out, upgradeAction(f, p.Provider)...)
		} else {
			out = append(out, note(fmt.Sprintf("  %-14s %s", p.Provider, stateWords(p))))
		}
	}
	return append(out, action{"  [ Scan again ] look for engines in the usual folders", "engines-scan", ""})
}

// upgradeAction is Upgrade engines in the picker: offered only when Python knows a newer release than the build here
// (standing behind); picking it installs that release beside the current build. Nothing upgrades unless picked.
func upgradeAction(f Facts, engine string) []action {
	s := f.standing(engine)
	switch {
	case f.Upgrading == engine:
		return []action{note("    " + f.Spin + " upgrading " + engine + "…")}
	case s == nil || s.State != "behind" || s.Latest == nil:
		return nil
	}
	return []action{{"    Upgrade " + engine + " to " + *s.Latest, "upgrade", engine}}
}

// stateWords is an engine that cannot run here in two words; the sentence why stays in --json.
func stateWords(p seam.ProviderRow) string {
	if p.State == "not_compatible" {
		return "not compatible"
	}
	return "not here"
}

// --- Results --------------------------------------------------------------------------------------------------

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

// Compared is one comparison on screen: the two ids, the two runs head to head, their parts as the seam words them
// (or why the parts are not compared), or the refusal.
type Compared struct {
	Before, After string
	Head          *seam.HeadToHead
	Lines         []string
	Plain         *seam.PlainCompare // the plain answer about the parts, when the seam words one
	PartsNote     *string
	Err           string
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

func aboutChip(f Facts) string {
	if t := f.target(); t != nil {
		return t.ID
	}
	return ""
}

// modelErrWords is why a model could not be read, in one plain line: a weight format BoltBeam has no layout for is
// named from the seam's data (inspect's formats); any other refusal is its first line.
func modelErrWords(err error) string {
	var e *seam.Error
	if errors.As(err, &e) && len(e.Formats) > 0 {
		return "BoltBeam can't read this model's weight format yet (" + strings.Join(e.Formats, ", ") + ")"
	}
	first, _, _ := strings.Cut(err.Error(), "\n")
	return "BoltBeam could not read this model: " + first
}

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

// layoutLabel names the chosen GPU layout and the GPUs it uses.
func layoutLabel(f Facts) string {
	if p := f.engineRow(); p != nil {
		for _, l := range p.Layouts {
			if l.ID == f.layout() {
				return l.Label + gpusWord(l)
			}
		}
	}
	return "one GPU"
}

// layoutSuffix is the chosen layout after the engine's name on a machine with more than one GPU; nothing on one.
func layoutSuffix(f Facts) string {
	if f.GpuCount < 2 {
		return ""
	}
	return " · " + layoutLabel(f)
}

func (f Facts) layout() string { return cmp.Or(f.Layout, "one") }

// gpusWord is " (2 GPUs)" for a split; "" for one GPU (its label says so) or an older Python's row.
func gpusWord(l seam.LayoutRow) string {
	if l.Gpus < 2 {
		return ""
	}
	return fmt.Sprintf(" (%d GPUs)", l.Gpus)
}

// layoutLegend is the GPUs and what each layout does to decode speed, once, above the engines (more than one GPU).
func layoutLegend(f Facts) string {
	if f.GpuCount < 2 || f.Providers == nil {
		return ""
	}
	b := stWarn.Render(fmt.Sprintf("%d GPUs: %s", f.GpuCount, f.MultiGpu)) + "\nThe layout is how the engine uses them; pick it under the engine:\n"
	seen := map[string]bool{}
	for _, p := range f.Providers.Providers {
		for _, l := range p.Layouts {
			if !seen[l.ID] && l.Speed != "" {
				seen[l.ID], b = true, b+"  "+l.Label+": "+stMuted.Render(l.Speed)+"\n"
			}
		}
	}
	if n := f.Providers.Note; n != nil {
		b += stMuted.Render(*n) + "\n"
	}
	return b + "\n"
}

// refusals are an engine's layouts that cannot run here, grouped by why: the reasons in order, the labels per reason.
func refusals(p seam.ProviderRow) (why []string, by map[string][]string) {
	by = map[string][]string{}
	for _, l := range p.Layouts {
		if r := deref(l.Reason); !l.Available {
			if by[r] == nil {
				why = append(why, r)
			}
			by[r] = append(by[r], l.Label)
		}
	}
	return why, by
}

// layoutsHere is why an engine cannot run its other layouts here, one line per reason; "" on one GPU.
func layoutsHere(f Facts, p seam.ProviderRow) (out string) {
	why, by := refusals(p)
	for _, r := range why {
		if f.GpuCount > 1 {
			out += "  " + stMuted.Render(strings.Join(by[r], " and ")+" not here: "+r) + "\n"
		}
	}
	return out
}

// layoutActions are an engine's layouts under its row on a machine with more than one GPU: the ones it can run here
// are picked with the engine; the others say not here in one short line (the sentence why is in the body).
func layoutActions(f Facts, p seam.ProviderRow) (out []action) {
	if f.GpuCount < 2 {
		return nil
	}
	for _, l := range p.Layouts {
		chosen := "  "
		if p.Provider == f.Engine && l.ID == f.layout() {
			chosen = stAccent.Render("● ")
		}
		if l.Available {
			out = append(out, action{"    " + chosen + l.Label + stMuted.Render(gpusWord(l)), "enginelayout", p.Provider + "|" + l.ID})
		}
	}
	why, by := refusals(p)
	for _, r := range why {
		out = append(out, note("      "+strings.Join(by[r], ", ")+": not here"))
	}
	return out
}

// --- 3 Speed limit -----------------------------------------------------------------------------------------

// --- 4 Measure ---------------------------------------------------------------------------------------------

// compareDoing words what the compare job is doing for the role in flight.
var compareDoing = map[string]string{"search": "searching kernels", "ab": "timing the whole model", "time": "timing in the model"}

// --- Workload -------------------------------------------------------------------------------------------------

// workloads are what a Run can measure, in Setup's order, each with what it measures and when to pick it; a workload
// not built yet is shown muted and not picked.
var workloads = []struct{ id, about, missing string }{
	{"decode", "How fast the model writes its answer, one token at a time after the prompt. The usual first run, " +
		"and the one to pick for chat.", ""},
	{"prefill", "How fast the model reads a prompt, at lengths from 512 tokens up to the longest it takes. Pick it " +
		"when long prompts or documents matter.", ""},
	{"session", "How speed holds up as a conversation grows: a 512-token turn added to 1K tokens already read, then " +
		"4 times more, up to what fits in memory. Pick it for long chats.", ""},
	{"replay", "A conversation you recorded, played back turn by turn on your own llama-server, each turn against the " +
		"best it could do. Pick it to check your real setup.", ""},
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
	return "Pick what the Run measures. Each type says what it times below."
}

// workloadActions are the types, each its own block: the name, then what it measures and when to pick it, wrapped
// to an 80-column screen, a blank line between types.
func workloadActions(f Facts) []action {
	out := []action{}
	for i, w := range workloads {
		if i > 0 {
			out = append(out, head(""))
		}
		if w.missing != "" {
			out = append(out, note(stMuted.Render(fmt.Sprintf("  %s · %s", w.id, w.missing))))
			continue
		}
		chosen := "  "
		if w.id == f.workload() {
			chosen = stAccent.Render("● ")
		}
		out = append(out, action{chosen + w.id, "workload", w.id})
		for _, l := range strings.Split(ansi.Wordwrap(w.about, 66, " "), "\n") {
			out = append(out, note("    "+l))
		}
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
	return "Measurement is how BoltBeam times each role. Each way says below what it times and where it runs."
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
	for i, o := range opts.Options {
		if i > 0 {
			out = append(out, head("")) // a blank line between the ways
		}
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
