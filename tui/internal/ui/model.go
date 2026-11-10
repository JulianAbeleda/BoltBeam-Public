package ui

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// Five keys. Everything else is a row inside a step's full view.
var (
	keyUp    = key.NewBinding(key.WithKeys("up", "k"))
	keyDown  = key.NewBinding(key.WithKeys("down", "j"))
	keyPgUp  = key.NewBinding(key.WithKeys("pgup", "ctrl+u"))
	keyPgDn  = key.NewBinding(key.WithKeys("pgdown", "ctrl+d"))
	keyHome  = key.NewBinding(key.WithKeys("home", "g"))
	keyEnd   = key.NewBinding(key.WithKeys("end", "G"))
	keyEnter = key.NewBinding(key.WithKeys("enter"))
	keyBack  = key.NewBinding(key.WithKeys("esc"))
	keyDel   = key.NewBinding(key.WithKeys("d"))
	keyLog   = key.NewBinding(key.WithKeys("l"))
	keyQuit  = key.NewBinding(key.WithKeys("q", "ctrl+c"))
)

// Model is the checklist: seam facts, the step cursor, and whether that step's full view is open.
type Model struct {
	client    seam.Client
	store     jobs.Store
	f         Facts
	context   int
	wantChip  string
	chipSet   bool      // the user picked a chip; detection no longer moves it
	runSet    bool      // the user picked a run; the checklist no longer follows the newest one
	session   []string  // runs started in this session: never cleaned
	before    *seam.Run // the finished run on screen when a new one started: back on screen if the new one is stopped
	opened    bool      // the start page is decided (Setup, or Run when a job is still going)
	engineSet bool      // the engine was picked in this session; until then it is the last run's (rememberEngine)
	ceilAsked string    // the run and memory speed the limit was last reloaded for (followMeasured)
	started   string    // when this session began, ISO; Clean removes only unsaved runs from before it
	cursor    int
	moved     bool // the user moved; the cursor no longer follows the first open step
	open      bool
	row       int
	ticking   bool
	width     int
	height    int
	input     textinput.Model
	note      string
	view      viewport.Model
	spin      spinner.Model
	upd       selfUpdate // a newer BoltBeam: the boot prompt (selfupdate.go)
}

type targetsMsg struct {
	targets *seam.Targets
	err     error
}
type detectMsg struct {
	id, driver, multi string
	count             int
	newChip           string // the GPU's name when no chip profile fits it yet
}

// scanMsg is the answer of an autoscan: the profile kept, generated or measured again.
type scanMsg struct {
	scan *seam.ChipScan
	err  error
}
type filesMsg []string
type profileMsg struct {
	profile *seam.Profile
	err     error
}
type ceilingMsg struct {
	ceiling *seam.Ceiling
	err     error
}
type runsMsg struct {
	runs *seam.Runs
	err  error
}
type runMsg struct {
	run *seam.Run
	err error
}
type jobMsg struct {
	job  *jobs.Job
	tail []string
}
type startedMsg string
type readyMsg struct {
	id    string
	ready *seam.CompareReady
}
type compareJobMsg struct {
	job  *jobs.Job
	tail []string
}
type compareStartedMsg string
type mtpJobMsg struct {
	job     *jobs.Job
	started bool // this press started it
}
type savedListMsg struct {
	saved *seam.Saved
	err   error
}
type savedMsg struct {
	id, dir string
	err     error
}
type savedRunMsg struct {
	run *seam.Run
	err error
}
type cleanedMsg struct{ err error }
type comparedMsg Compared
type othersMsg struct {
	path   string
	others *seam.Ceilings
}
type providersMsg struct {
	target    string
	providers *seam.Providers
}

// enginesMsg is the answer of the engine scan: what was found and saved, or why the scan failed.
type enginesMsg struct {
	scan *seam.Engines
	err  error
}

// latestMsg is `engines --latest`: each engine's build against upstream's newest release, or why the look failed.
// With upgraded, it is instead what the Engine picker's Upgrade did to that engine (`engines --update`).
type latestMsg struct {
	rows     []seam.EngineStanding
	err      error
	upgraded string
}
type replayMsg struct {
	choices *seam.ReplayChoices
	err     error
}
type noteMsg string

// stoppedMsg: SIGTERM went to this run; the screen goes back to the finished run it replaced, if any.
type stoppedMsg struct{ id, note string }

// emitMsg is the answer of Emit: the gameplan Python wrote, or why it could not.
type emitMsg struct {
	plan *seam.Emitted
	err  error
}

// deletedMsg names a run folder that is gone.
type deletedMsg string
type tickMsg time.Time

// New builds the model; modelPath and target may be empty, context is the prefill length for the ceiling.
func New(client seam.Client, store jobs.Store, modelPath, target string, context int) Model {
	if modelPath == "" {
		modelPath = savedModel(store.Dir)
	}
	in := textinput.New()
	in.Prompt = ""
	in.SetValue(modelPath)
	return Model{client: client, store: store, f: Facts{Path: modelPath, Reading: modelPath != "", ByFlag: target != "", EngineScan: true}, context: context,
		wantChip: target, chipSet: target != "", open: true, input: in, started: time.Now().Format("2006-01-02T15:04:05"), width: 80, height: 24, view: viewport.New(80, 21),
		spin: spinner.New(spinner.WithSpinner(spinner.MiniDot), spinner.WithStyle(stAccent))}
}

// Start runs the program on the terminal.
func Start(client seam.Client, store jobs.Store, modelPath, target string, context int) error {
	_, err := tea.NewProgram(New(client, store, modelPath, target, context), tea.WithAltScreen(), tea.WithMouseCellMotion()).Run()
	return err
}

func (m Model) Init() tea.Cmd {
	cmds := []tea.Cmd{m.loadTargets(), m.detect(), m.scanEngines(true), m.loadRuns(), m.findFiles(), m.loadSaved(), m.checkSelfUpdate(), m.spin.Tick}
	if m.f.Path != "" {
		cmds = append(cmds, m.inspect())
	}
	return tea.Batch(cmds...)
}

func tick() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m Model) targetID() string {
	if t := m.f.target(); t != nil {
		return t.ID
	}
	return ""
}

func (m Model) runID() string {
	if m.f.Run != nil {
		return m.f.Run.ID
	}
	return ""
}

// RunStem is the run folder prefix: the model file's base name, lower case, plus the chip.
func RunStem(modelPath, target string) string {
	base := filepath.Base(modelPath)
	if st, err := os.Stat(modelPath); err != nil || !st.IsDir() { // a folder keeps its whole name (profile/ir.model_name)
		base = strings.TrimSuffix(base, filepath.Ext(base))
	}
	var b strings.Builder
	for _, r := range strings.ToLower(base) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else {
			b.WriteRune('-')
		}
	}
	return strings.Trim(b.String(), "-") + "-" + target
}

// newestRun is the run the checklist is about when the user picked none: the newest for this model and chip,
// or the newest of all when no model is set.
func (m Model) newestRun() string {
	if m.f.Runs == nil {
		return ""
	}
	stem := RunStem(m.f.Path, m.targetID()) + "-"
	for i := len(m.f.Runs.Runs) - 1; i >= 0; i-- {
		if id := m.f.Runs.Runs[i].ID; m.f.Path == "" || strings.HasPrefix(id, stem) {
			return id
		}
	}
	return ""
}

// rememberEngine picks the engine the newest run on this chip measured with, the way the model and chip come
// back on a relaunch; a pick in this session, or an engine this machine lacks, is left alone.
func (m *Model) rememberEngine() {
	t := m.f.target()
	if m.engineSet || m.f.Runs == nil || m.f.Providers == nil || t == nil {
		return
	}
	for i := len(m.f.Runs.Runs) - 1; i >= 0; i-- {
		r := m.f.Runs.Runs[i]
		if r.TargetID != t.ID || r.Measure == nil || r.Measure.Provider == nil {
			continue
		}
		if p := m.f.provider(*r.Measure.Provider); p == nil || !p.Available {
			continue
		}
		m.f.Engine = *r.Measure.Provider
		// the measurement comes back too, when the run picked one in Setup (auto stays Python's default)
		if c := r.Measure.Capture; c != nil && c.Measurement != nil && c.Measurement.Choice != "auto" && m.f.Measurement == "" {
			m.f.Measurement = c.Measurement.Choice
			if m.f.measurementBlocked() != "" {
				m.f.Measurement = ""
			}
		}
		return
	}
}

func (m *Model) pickChip(id string) {
	if m.f.Targets == nil {
		return
	}
	for i, t := range m.f.Targets.Targets {
		if t.ID == id {
			m.f.Target = i
			return
		}
	}
	for i, t := range m.f.Targets.Targets {
		if t.HasCeiling {
			m.f.Target = i
			return
		}
	}
}

// chipChanged reloads what depends on the chip: the speed limit and the run.
func (m *Model) chipChanged() tea.Cmd {
	m.f.Ceiling, m.f.CeilErr, m.f.Providers = nil, "", nil
	cmds := []tea.Cmd{m.follow()}
	if t := m.f.target(); t != nil && !m.f.EngineScan { // during the scan, enginesMsg loads the providers
		cmds = append(cmds, m.loadProviders(t.ID))
	}
	if m.f.Profile != nil {
		m.f.CeilBusy = true
		cmds = append(cmds, m.loadCeiling())
	}
	return tea.Batch(cmds...)
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if next, cmd, ok := m.selfUpdateStep(msg); ok {
		return next, cmd
	}
	next, cmd := m.update(msg)
	next.skipHeads(1) // the cursor never rests on a heading
	if c := next.loadRecommend(); c != nil {
		cmd = tea.Batch(cmd, c)
	}
	return next, cmd
}

func (m Model) update(msg tea.Msg) (Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		m.f.Spin = m.spin.View()
		return m, cmd
	case targetsMsg:
		m.f.Targets = msg.targets
		if msg.err != nil {
			m.note = "The chip list could not be read: " + msg.err.Error()
			return m, nil
		}
		m.pickChip(m.wantChip)
		return m, m.chipChanged()
	case scanMsg:
		m.f.Scanning = false
		switch {
		case msg.err != nil:
			m.note = "Autoscan failed: " + msg.err.Error()
			return m, nil
		case msg.scan.Action == "failed":
			m.note = "Autoscan could not measure this chip: " + deref(msg.scan.Reason)
			return m, nil
		}
		m.note = "Autoscan: " + deref(msg.scan.TargetID) + " " + msg.scan.Action + "."
		if !m.f.ByFlag {
			m.chipSet, m.f.Picked = false, false
		}
		return m, tea.Batch(m.loadTargets(), m.detect())
	case detectMsg:
		m.f.ThisMachine, m.f.Driver, m.f.Detected = msg.id, msg.driver, true
		m.f.GpuCount, m.f.MultiGpu, m.f.NewChip = msg.count, msg.multi, msg.newChip
		if !m.chipSet && msg.id != "" {
			m.wantChip = msg.id
			if m.f.Targets != nil {
				m.pickChip(msg.id)
				return m, m.chipChanged()
			}
		}
	case filesMsg:
		m.f.Files = msg
		m.focusModel()
	case profileMsg:
		m.f.Reading = false
		if msg.err != nil {
			m.f.Profile, m.f.ModelErr = nil, msg.err.Error()
			return m, nil
		}
		m.f.Profile, m.f.ModelErr = msg.profile, ""
		m.saveModel()
		return m, tea.Batch(m.chipChanged(), m.loadOthers())
	case othersMsg:
		if msg.path == m.f.Path {
			m.f.Others = msg.others
		}
	case ceilingMsg:
		m.f.CeilBusy = false
		m.f.Ceiling, m.f.CeilErr = msg.ceiling, ""
		if msg.err != nil {
			m.f.Ceiling, m.f.CeilErr = nil, msg.err.Error()
		}
	case runsMsg:
		m.f.Runs = msg.runs
		m.countOldWork()
		m.rememberEngine()
		if msg.err != nil {
			m.note = "The runs folder could not be read: " + msg.err.Error()
		}
		return m, m.follow()
	case runMsg:
		if msg.err != nil {
			if !m.f.alive() { // a live run has no folder until its first stage lands
				m.note = "The run could not be read: " + msg.err.Error()
			}
			return m, nil
		}
		changed := m.runID() != msg.run.ID
		m.f.Run = msg.run
		if m.f.Plan != nil && m.f.Plan.ID != msg.run.ID { // the gameplan belongs to the run it was written for
			m.f.Plan = nil
		}
		reload := m.followMeasured()
		if changed {
			m.f.Ready, m.f.CJob, m.f.CTail = nil, nil, nil
			return m, tea.Batch(m.loadJob(msg.run.ID), m.loadReady(msg.run.ID), m.loadCompare(msg.run.ID), reload)
		}
		if reload != nil {
			return m, reload
		}
	case replayMsg:
		if msg.err != nil {
			m.note = "Replay choices: " + msg.err.Error()
		}
		m.f.Replay = msg.choices
	case enginesMsg:
		m.f.EngineScan = false
		if msg.err != nil { // the Engine line shows what was found; only a failure needs words
			m.note = "The engine scan failed: " + msg.err.Error() + " (env vars still work)"
		}
		if t := m.f.target(); t != nil {
			return m, tea.Batch(m.loadProviders(t.ID), m.loadLatest())
		}
		return m, m.loadLatest()
	case latestMsg:
		switch {
		case msg.upgraded != "" && msg.err == nil && len(msg.rows) == 1:
			m.f.Upgrading, m.note = "", "Upgrade "+msg.upgraded+": "+msg.rows[0].Words
		case msg.upgraded != "":
			m.f.Upgrading, m.note = "", fmt.Sprintf("Upgrade %s failed: %v (%d rows)", msg.upgraded, msg.err, len(msg.rows))
		case msg.err != nil:
			m.f.Latest, m.f.LatestErr = nil, msg.err.Error()
		default:
			m.f.Latest, m.f.LatestErr = msg.rows, ""
		}
		if msg.upgraded != "" {
			return m, m.loadLatest()
		}
	case providersMsg:
		if t := m.f.target(); t != nil && t.ID == msg.target {
			m.f.Providers = msg.providers
			if p := m.f.engineRow(); (p == nil || !p.Available) && msg.providers != nil {
				m.f.Engine = "" // the first engine this machine has, until one is picked
				for _, r := range msg.providers.Providers {
					if m.f.modelProvider(r).Available {
						m.f.Engine = r.Provider
						m.f.Measurement = ""
						break
					}
				}
			}
			m.rememberEngine()
		}
	case recommendMsg:
		if msg.check {
			m.f.RecChecking = false
		}
		switch {
		case msg.key != m.f.RecKey:
		case msg.err != nil && msg.check:
			m.note = "Check failed: " + msg.err.Error()
		case msg.err != nil:
			m.f.RecErr = msg.err.Error()
		default:
			m.f.Recommend, m.f.RecErr = msg.rec, ""
			if msg.check {
				m.note = "Check done."
			}
		}
	case readyMsg:
		if msg.id == m.runID() {
			m.f.Ready = msg.ready
		}
	case compareJobMsg:
		m.f.CJob, m.f.CTail = msg.job, msg.tail
		if m.f.comparing() && !m.ticking {
			m.ticking = true
			return m, tick()
		}
	case mtpJobMsg:
		was := m.f.measuringMtp()
		m.f.MJob = msg.job
		switch {
		case msg.started:
			m.note = "Measuring MTP: k=0, then each draft count, one engine start each."
		case was && !m.f.measuringMtp(): // it ended: the recommendation reads the measurement it kept
			m.note = "Measure MTP ended."
			if msg.job != nil {
				m.note += " Its log: " + msg.job.LogPath
			}
			m.f.RecKey = ""
			return m, m.loadRecommend()
		}
		if m.f.measuringMtp() && !m.ticking {
			m.ticking = true
			return m, tick()
		}
	case compareStartedMsg:
		id := string(msg)
		m.ticking = true
		m.f.CJob, m.f.CTail = &jobs.Job{ID: compareID(id), Alive: true}, nil
		m.note = "Started the job for " + id + ". It takes a few minutes."
		return m, tea.Batch(m.loadCompare(id), tick())
	case jobMsg:
		was := m.f.alive()
		m.f.Job, m.f.Tail = msg.job, msg.tail
		m.f.advance(clock())
		if msg.job != nil && !msg.job.Alive && strings.HasPrefix(m.note, "Started ") {
			m.note = "" // the run ended: the Run line says how; "Started …" would be stale
		}
		var reread tea.Cmd
		if was && !m.f.alive() { // the run wrote this machine's facts: the recommendation reads them (Measure MTP needs them)
			m.f.RecKey = ""
			reread = m.loadRecommend()
		}
		if !m.opened { // on start: Setup, unless the newest run is still going, then its Run screen; decided once
			m.opened = true
			if m.f.alive() && m.cursor == pageSetup {
				m.cursor, m.row = pageRun, 0
			}
		}
		if m.f.alive() && !m.ticking {
			m.ticking = true
			return m, tick()
		}
		return m, reread
	case startedMsg:
		id := string(msg)
		m.cursor, m.row = pageRun, 0
		m.f.ReadOnly, m.session = false, append(m.session, id)
		m.runSet, m.ticking = true, true
		m.before = nil
		if m.f.Run != nil && m.f.outputDone() && !m.f.ReadOnly {
			m.before = m.f.Run
		}
		m.f.Run = &seam.Run{Summary: seam.Summary{ID: id}}
		m.f.Job, m.f.Tail = &jobs.Job{ID: id, Alive: true}, nil
		m.note = "Started " + id + "."
		return m, tea.Batch(m.loadJob(id), tick())
	case savedListMsg:
		m.f.Saved = msg.saved
		if msg.err != nil {
			m.note = "The saved runs could not be read: " + msg.err.Error()
		}
		m.countOldWork()
	case savedMsg:
		if msg.err != nil {
			m.note = "Save failed: " + msg.err.Error()
			return m, nil
		}
		m.f.SavedID, m.f.SavedDir = msg.id, msg.dir
		m.note = "Saved to " + msg.dir + "."
		return m, m.loadSaved()
	case savedRunMsg:
		if msg.err != nil {
			m.note = "The saved run could not be read: " + msg.err.Error()
			return m, nil
		}
		m.note = ""
		m.f.Run, m.f.Job, m.f.Tail, m.f.ReadOnly, m.runSet = msg.run, nil, nil, true, true
		m.cursor, m.row = pageRun, 0
		m.view.GotoTop()
	case comparedMsg:
		c := Compared(msg)
		m.f.Compared = &c
	case cleanedMsg:
		if msg.err != nil {
			m.note = "Cleaning failed: " + msg.err.Error()
		} else {
			m.note = "Deleted the unsaved runs from earlier sessions."
		}
		return m, m.loadRuns()
	case deletedMsg:
		if id, ok := strings.CutPrefix(string(msg), "saved:"); ok {
			m.note = "Deleted saved run " + id + "."
			if m.f.ReadOnly && m.runID() == id {
				m.f.Run, m.f.ReadOnly = nil, false
			}
			return m, m.loadSaved()
		}
		if m.runID() == string(msg) {
			m.f.Run, m.f.Job, m.f.Tail, m.runSet, m.row = nil, nil, nil, false, 0
		}
		m.note = "Deleted run " + string(msg) + "."
		return m, m.loadRuns()
	case stoppedMsg:
		m.note = msg.note
		if b := m.before; b != nil && m.runID() == msg.id { // Emit and Last run read the finished run again
			m.f.Run, m.f.Job, m.f.Tail, m.before = b, nil, nil, nil
			m.note += " Back to " + shortRun(b.ID) + ", the last finished run."
			return m, m.loadJob(b.ID)
		}
	case noteMsg:
		m.note = string(msg)
	case emitMsg:
		m.f.Emitting = false
		if msg.err != nil {
			m.note = "Emit failed for " + m.runID() + ": " + msg.err.Error()
			return m, nil
		}
		m.f.Plan = msg.plan
		m.cursor, m.row = pageEmit, 0
		m.view.GotoTop()
		m.note = "Wrote gameplan.md and gameplan.json for " + msg.plan.ID + "."
	case tickMsg:
		m.f.advance(clock())
		id := m.runID()
		if m.f.alive() {
			return m, tea.Batch(m.loadJob(id), m.loadRuns(), m.loadRun(id), tick())
		}
		if m.f.comparing() {
			return m, tea.Batch(m.loadCompare(id), m.loadRun(id), tick())
		}
		if m.f.measuringMtp() {
			return m, tea.Batch(m.loadMtp(), tick())
		}
		m.ticking = false
		if id == "" { // nothing on screen to read again (Measure MTP ended with no run for this model and chip)
			return m, m.loadRuns()
		}
		return m, tea.Batch(m.loadRuns(), m.loadRun(id))
	case tea.KeyMsg:
		return m.key(msg)
	case tea.MouseMsg: // the wheel scrolls a list the way the arrow keys do
		switch msg.Button {
		case tea.MouseButtonWheelDown:
			m.moveDown(1)
		case tea.MouseButtonWheelUp:
			m.moveUp(1)
		}
	}
	return m, nil
}

// PageRows is how far PgUp/PgDn (and ctrl+u/ctrl+d) move the cursor through a list.
const PageRows = 10

// moveDown moves the cursor n rows down through the step's rows; past the last row the page above them scrolls.
// The rows are shown windowed around the cursor (DetailActions), so this is the list's scrolling.
func (m *Model) moveDown(n int) {
	for ; n > 0; n-- {
		if m.row < len(m.actions())-1 {
			m.row++
			m.skipHeads(1)
		} else {
			m.view.LineDown(1)
		}
	}
}

// moveUp is moveDown's mirror: the page above scrolls back first, then the cursor climbs the rows.
func (m *Model) moveUp(n int) {
	for ; n > 0; n-- {
		if m.view.YOffset > 0 {
			m.view.LineUp(1)
		} else if m.row > 0 {
			m.row--
			m.skipHeads(-1)
		}
	}
}

func (m Model) actions() []action {
	if a := steps[m.cursor].actions; a != nil {
		return a(m.f)
	}
	return nil
}

// runRow is the run id under the row cursor when an open step shows a run row, else "".
// runRow is the saved run under the row cursor, or "": d deletes it.
func (m Model) runRow() string {
	if acts := m.actions(); m.row < len(acts) && acts[m.row].do == "opensaved" {
		return acts[m.row].arg
	}
	return ""
}

// skipHeads moves the row off section headings, in direction dir (1 down, -1 up), never past the ends.
// chosenRow is the engine picker's row for Setup's engine and layout (its layout's row first), else fallback.
func (m *Model) chosenRow(fallback int) int {
	acts, engine := m.actions(), -1
	for i, a := range acts {
		if a.do == "enginelayout" && a.arg == m.f.Engine+"|"+m.f.layout() {
			return i
		}
		if a.do == "engine" && a.arg == m.f.Engine && engine < 0 {
			engine = i
		}
	}
	if engine >= 0 {
		return engine
	}
	return fallback
}

func (m *Model) skipHeads(dir int) {
	acts := m.actions()
	for m.row >= 0 && m.row < len(acts) && skipped(acts[m.row].do) {
		m.row += dir
	}
	if m.row < 0 || m.row >= len(acts) {
		m.row -= dir
		for m.row >= 0 && m.row < len(acts) && skipped(acts[m.row].do) {
			m.row -= dir
		}
		m.row = max(0, min(m.row, len(acts)-1))
	}
}

// countOldWork counts the unsaved runs not started in this session, which Setup offers to delete.
func (m *Model) countOldWork() {
	m.f.OldWork = 0
	if m.f.Runs == nil {
		return
	}
	for _, r := range m.f.Runs.Runs {
		if !contains(m.session, r.ID) && !m.isSaved(r.ID) {
			m.f.OldWork++
		}
	}
}

func (m Model) isSaved(id string) bool {
	if m.f.Saved == nil {
		return false
	}
	for _, r := range m.f.Saved.Runs {
		if r.ID == id {
			return true
		}
	}
	return false
}

func (m Model) key(msg tea.KeyMsg) (Model, tea.Cmd) {
	onClean := key.Matches(msg, keyEnter) && m.row < len(m.actions()) && m.actions()[m.row].do == "clean"
	if !key.Matches(msg, keyDel) && !onClean {
		m.f.Confirm = "" // any other key cancels a pending delete (enter on the clean row is its second press)
	}
	if m.f.SettingEdit != "" {
		return m.typeSetting(msg)
	}
	if m.f.Editing {
		switch msg.Type {
		case tea.KeyEnter:
			m.f.Editing, m.f.Path = false, strings.TrimSpace(m.input.Value())
			m.input.Blur()
			m.backToSetup()
			return m.readModel()
		case tea.KeyEsc:
			m.f.Editing = false
			m.input.Blur()
			m.input.SetValue(m.f.Path)
			return m, nil
		}
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		m.f.Input = m.input.Value()
		return m, cmd
	}
	DetailView(m.f, m.cursor, m.row, m.width, m.detailHeight(), &m.view) // size the scroll before keys move it
	if key.Matches(msg, keyDel) {
		if id := m.runRow(); id != "" {
			return m.do(action{"", "deletesaved", id})
		}
		return m, nil
	}
	switch {
	case m.cursor == pageBatch && len(msg.Runes) == 1 && msg.Runes[0] >= '0' && msg.Runes[0] <= '9' && len(m.f.BatchTyped) < 3:
		m.f.BatchTyped += string(msg.Runes)
		m.row = len(m.actions()) - 1
	case m.cursor == pageBatch && msg.Type == tea.KeyBackspace && m.f.BatchTyped != "":
		m.f.BatchTyped = m.f.BatchTyped[:len(m.f.BatchTyped)-1]
	case key.Matches(msg, keyQuit):
		return m, tea.Quit
	case key.Matches(msg, keyLog) && m.cursor == pageRun && (m.f.alive() || m.f.failedStage() != ""):
		m.f.ShowLog = !m.f.ShowLog // the log while it runs; the details behind a failure's sentence after
	case key.Matches(msg, keyBack):
		if m.cursor != pageSetup {
			m.backToSetup()
		}
	case key.Matches(msg, keyDown): // the rows first, then the page above them scrolls
		m.moveDown(1)
	case key.Matches(msg, keyUp):
		m.moveUp(1)
	case key.Matches(msg, keyPgDn):
		m.moveDown(PageRows)
	case key.Matches(msg, keyPgUp):
		m.moveUp(PageRows)
	case key.Matches(msg, keyEnd):
		m.moveDown(len(m.actions()))
	case key.Matches(msg, keyHome):
		m.view.GotoTop()
		m.row = 0
		m.skipHeads(1)
	case key.Matches(msg, keyEnter):
		if acts := m.actions(); m.row < len(acts) {
			return m.do(acts[m.row])
		}
	}
	return m, nil
}

// typeSetting is a key while a setting's value is typed: enter keeps it (a whole number where the setting takes one;
// empty unsets it), esc leaves it as it was.
func (m Model) typeSetting(msg tea.KeyMsg) (Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyEnter:
		s := m.f.setting(m.f.SettingEdit)
		v, ok := "", s != nil
		if ok {
			v, ok = settingTyped(*s, m.input.Value())
		}
		if !ok {
			m.note = "Type a whole number above 0, or nothing to unset it."
			return m, nil
		}
		m.f.setSetting(s.Name, v)
		m.note = ""
		fallthrough
	case tea.KeyEsc:
		m.f.SettingEdit, m.f.Input = "", ""
		m.input.Blur()
		m.input.SetValue(m.f.Path)
		return m, nil
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	m.f.Input = m.input.Value()
	return m, cmd
}

// backToSetup shows Setup again; from a picker the cursor rests on that picker's line.
func (m *Model) backToSetup() {
	m.row = 0
	if picker(m.cursor) || m.cursor == pageRecommend { // the cursor rests on the Setup line that opened the page
		for i, a := range setupActions(m.f) {
			if a.do == "page" && a.arg == fmt.Sprint(m.cursor) {
				m.row = i
			}
		}
	}
	m.f.SettingEdit = ""
	m.cursor = pageSetup // keep the displayed run's saved origin until a new run replaces it
	m.skipHeads(1)
	m.view.GotoTop()
}

// do runs one action row of the open step.
func (m Model) do(a action) (Model, tea.Cmd) {
	if a.do != "delete" && a.do != "deletesaved" && a.do != "clean" { // these ask twice: the first press must stay
		m.f.Confirm = ""
	}
	switch a.do {
	case "head", "note", "":
		return m, nil
	case "save":
		return m, m.saveRun()
	case "opensaved":
		m.note = "Reading saved run " + a.arg + "…"
		return m, m.openSaved(a.arg)
	case "comparefrom":
		m.f.CompareFrom, m.cursor, m.row = a.arg, pageSaved, 0
		return m, m.loadSaved()
	case "comparecancel":
		m.f.CompareFrom, m.row = "", 0
		return m, nil
	case "compare":
		before := m.f.CompareFrom
		m.f.CompareFrom, m.f.Compared, m.cursor, m.row = "", nil, pageCompare, 0
		m.view.GotoTop()
		return m, m.compareSaved(before, a.arg)
	case "deletesaved":
		if m.f.Confirm != a.arg {
			m.f.Confirm = a.arg
			return m, nil
		}
		m.f.Confirm = ""
		return m, m.deleteSaved(a.arg)
	case "clean":
		if m.f.Confirm != "clean" {
			m.f.Confirm = "clean"
			return m, nil
		}
		m.f.Confirm = ""
		return m, m.cleanWork()
	case "delete":
		if m.f.Confirm != a.arg {
			m.f.Confirm = a.arg
			return m, nil
		}
		m.f.Confirm = ""
		return m, m.deleteRun(a.arg)
	case "edit":
		m.f.Editing, m.f.Input = true, m.f.Path
		m.input.SetValue(m.f.Path)
		return m, m.input.Focus()
	case "file":
		m.f.Path = a.arg
		m.backToSetup()
		return m.readModel()
	case "page":
		m.cursor, _ = strconv.Atoi(a.arg)
		m.row = 0
		m.focusModel()
		m.skipHeads(1)
		if m.cursor == pageEngine { // the engine picker opens on what Setup has: its layout's line, else its engine's
			m.row = m.chosenRow(m.row)
		}
		m.view.GotoTop()
		if m.cursor == pageSaved {
			return m, m.loadSaved()
		}
		return m, nil
	case "engine":
		m.f.Engine, m.f.Layout, m.engineSet = a.arg, "", true
		m.f.Measurement, m.f.Settings = "", nil
		m.backToSetup()
		return m, nil
	case "batch":
		if n, err := strconv.Atoi(a.arg); err == nil && n >= 1 && n <= 512 {
			m.f.Batch, m.f.BatchTyped = n, ""
			m.backToSetup()
			return m, nil
		}
		return m, func() tea.Msg { return noteMsg("Type a batch size from 1 to 512.") }
	case "workload":
		m.f.Workload = a.arg
		if a.arg == "replay" { // stay: the trace and the server are picked here, from what this machine has
			client := m.client
			return m, func() tea.Msg { c, err := client.ReplayChoices(); return replayMsg{c, err} }
		}
		m.backToSetup()
		return m, nil
	case "replayshape":
		m.f.ReplayShape = a.arg
		return m, nil
	case "replayserver":
		m.f.ReplayArgv = a.arg
		m.backToSetup()
		return m, nil
	case "measurement":
		m.f.Measurement = a.arg
		m.backToSetup()
		return m, nil
	case "setting": // a fixed set steps to its next value; anything else is typed
		s := m.f.setting(a.arg)
		if s == nil {
			return m, nil
		}
		if len(s.Choices) == 0 {
			m.f.SettingEdit, m.f.Input = s.Name, m.f.Settings[s.Name]
			m.input.SetValue(m.f.Input)
			return m, m.input.Focus()
		}
		m.f.setSetting(s.Name, nextChoice(s.Choices, m.f.Settings[s.Name]))
		return m, nil
	case "repeats":
		m.f.Repeats, _ = strconv.Atoi(a.arg)
		m.backToSetup()
		return m, nil
	case "mtp":
		return m, m.startMtp()
	case "enginelayout": // an engine and how it uses this machine's GPUs, picked on one row
		engine, layout, _ := strings.Cut(a.arg, "|")
		if engine != m.f.Engine {
			m.f.Measurement, m.f.Settings = "", nil // another engine's settings are not this one's
		}
		m.f.Engine, m.f.Layout, m.engineSet = engine, layout, true
		m.backToSetup()
		return m, nil
	case "reccopy":
		copyText(m.f.copyOf(a.arg))
		m.note = "Copied " + map[string]string{"command": "the engine command line", "arkey": "the Arkey vllm_args"}[a.arg] + "."
		return m, nil
	case "reccheck":
		if m.f.RecChecking || m.f.recommendKey() == "" {
			return m, nil
		}
		m.f.RecChecking, m.note = true, "Check: starting the engine once with the recommended flags…"
		return m, m.askRecommend(m.f.RecKey, true)
	case "analyze":
		return m, m.startRun(m.f.Engine)
	case "upgrade": // Upgrade engines: this engine's newer release beside its current build, as a job
		if m.f.Upgrading != "" || m.targetID() == "" {
			return m, nil
		}
		engine, argv, client := a.arg, []string{"--update", "--engine", a.arg, "--target", m.targetID()}, m.client
		m.f.Upgrading, m.note = engine, "Upgrade "+engine+": installing the newer release beside the current build (a build can take minutes)…"
		return m, func() tea.Msg { rows, err := client.EngineRows(argv...); return latestMsg{rows, err, engine} }
	case "engines-scan":
		if m.f.EngineScan {
			return m, nil
		}
		m.f.EngineScan, m.f.Providers, m.note = true, nil, ""
		return m, m.scanEngines(false)
	case "autoscan":
		if m.f.Scanning {
			return m, nil
		}
		m.f.Scanning = true
		m.note = "Autoscan: reading this machine's GPU. A new chip is measured, about a minute."
		return m, m.autoscan(a.arg == "remeasure")
	case "chip":
		if !m.f.mayPick() { // set by flag: not chosen here
			return m, nil
		}
		m.f.Target, _ = strconv.Atoi(a.arg)
		m.chipSet, m.f.Picked = true, true
		m.backToSetup()
		return m, m.chipChanged()
	case "start":
		return m, m.startRun(a.arg)
	case "stop":
		return m, m.stopRun()
	case "roletime":
		return m, m.startStep5(true)
	case "run":
		m.runSet = true
		return m, m.loadRun(a.arg)
	case "report":
		return m, m.openReport()
	case "emit":
		if m.f.Emitting || !m.f.canEmit() {
			return m, nil
		}
		m.f.Emitting = true
		m.note = "Emit: writing the gameplan for " + m.runID() + "…"
		return m, m.emitPlan()
	case "gameplan":
		return m, m.openGameplan()
	}
	return m, nil
}

// mood picks the mascot's face from the facts on screen.
func (m Model) mood() string {
	f := m.f
	switch {
	case f.Reading || f.CeilBusy || f.alive() || f.comparing():
		return faceBusy
	case f.failedStage() != "":
		return faceWorried
	case f.Run != nil && f.Run.Results.Measured:
		return faceHappy
	case f.Run != nil && len(f.Run.Blocked) > 0:
		return faceWaiting
	case f.Profile == nil && (f.Runs == nil || len(f.Runs.Runs) == 0):
		return faceSleep
	}
	return faceIdle
}

func footer(back, onRun, running, failed bool) string {
	pairs := [][2]string{{"↑↓", "move"}, {"enter", "pick"}, {"q", "quit"}}
	switch {
	case running:
		pairs = [][2]string{{"↑↓", "move"}, {"enter", "pick"}, {"l", "log"}, {"esc", "setup"}, {"q", "quit"}}
	case failed:
		pairs = [][2]string{{"↑↓", "move"}, {"enter", "pick"}, {"l", "details"}, {"esc", "setup"}, {"q", "quit"}}
	case onRun:
		pairs = [][2]string{{"↑↓", "move"}, {"pgup/pgdn", "page"}, {"enter", "open run"}, {"d", "delete run"}, {"esc", "setup"}, {"q", "quit"}}
	case back:
		pairs = [][2]string{{"↑↓", "move"}, {"pgup/pgdn", "page"}, {"enter", "pick"}, {"esc", "setup"}, {"q", "quit"}}
	}
	parts := []string{}
	for _, p := range pairs {
		parts = append(parts, stAccent.Render(p[0])+" "+stMuted.Render(p[1]))
	}
	return " " + strings.Join(parts, stMuted.Render(" · "))
}

func (m Model) detailHeight() int {
	return m.height - 2 - lipgloss.Height(truncate(m.note, m.width))
}

func (m Model) View() string {
	left := stAccent.Render(glyphBolt) + " " + gradient("BoltBeam")
	right := stAccent.Render(m.mood())
	if m.f.Reading || m.f.CeilBusy || m.f.alive() || m.f.comparing() {
		right = m.spin.View() + " " + right
	}
	header := left + strings.Repeat(" ", max(m.width-lipgloss.Width(left)-lipgloss.Width(right), 1)) + right
	room := m.detailHeight()
	view := m.view
	body := DetailView(m.f, m.cursor, m.row, m.width, room, &view)
	if m.upd.shown {
		body = selfUpdateView(m.upd, m.width, room)
	}
	return header + "\n" + body + "\n" + truncate(m.note, m.width) + "\n" + footer(m.cursor != pageSetup, m.runRow() != "", m.cursor == pageRun && m.f.alive(), m.cursor == pageRun && m.f.failedStage() != "")
}
