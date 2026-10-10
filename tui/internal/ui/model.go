package ui

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
	chipSet   bool     // the user picked a chip; detection no longer moves it
	runSet    bool     // the user picked a run; the checklist no longer follows the newest one
	session   []string // runs started in this session: never cleaned
	opened    bool     // the start page is decided (Setup, or Run when a job is still going)
	engineSet bool     // the engine was picked in this session; until then it is the last run's (rememberEngine)
	ceilAsked string   // the run and memory speed the limit was last reloaded for (followMeasured)
	started   string   // when this session began, ISO; Clean removes only unsaved runs from before it
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
type othersMsg struct {
	path   string
	others *seam.Ceilings
}
type providersMsg struct {
	target    string
	providers *seam.Providers
}
type noteMsg string

// deletedMsg names a run folder that is gone.
type deletedMsg string
type tickMsg time.Time

// New builds the model; modelPath and target may be empty, context is the prefill length for the ceiling.
func New(client seam.Client, store jobs.Store, modelPath, target string, context int) Model {
	in := textinput.New()
	in.Prompt = ""
	in.SetValue(modelPath)
	return Model{client: client, store: store, f: Facts{Path: modelPath, Reading: modelPath != "", ByFlag: target != ""}, context: context,
		wantChip: target, chipSet: target != "", open: true, input: in, started: time.Now().Format("2006-01-02T15:04:05"), width: 80, height: 24, view: viewport.New(80, 21),
		spin: spinner.New(spinner.WithSpinner(spinner.MiniDot), spinner.WithStyle(stAccent))}
}

// Start runs the program on the terminal.
func Start(client seam.Client, store jobs.Store, modelPath, target string, context int) error {
	_, err := tea.NewProgram(New(client, store, modelPath, target, context), tea.WithAltScreen()).Run()
	return err
}

func (m Model) Init() tea.Cmd {
	cmds := []tea.Cmd{m.loadTargets(), m.detect(), m.loadRuns(), m.findFiles(), m.loadSaved(), m.spin.Tick}
	if m.f.Path != "" {
		cmds = append(cmds, m.inspect())
	}
	return tea.Batch(cmds...)
}

// loadProviders asks Python which runtimes can measure the chip on this machine.
func (m Model) loadProviders(target string) tea.Cmd {
	return func() tea.Msg {
		p, err := m.client.Providers(target)
		if err != nil {
			return providersMsg{target, nil}
		}
		return providersMsg{target, p}
	}
}

// loadOthers reads the model's limit on every registered chip, for step 3's read-only "On other chips".
func (m Model) loadOthers() tea.Cmd {
	path := m.f.Path
	return func() tea.Msg {
		o, err := m.client.Ceilings(path)
		if err != nil {
			return othersMsg{path, nil}
		}
		return othersMsg{path, o}
	}
}

func (m Model) loadTargets() tea.Cmd {
	return func() tea.Msg { t, _, err := m.client.Chips(); return targetsMsg{t, err} }
}

func (m Model) detect() tea.Cmd {
	return func() tea.Msg {
		d, err := m.client.Detect()
		if err == nil && d.Profile != nil && d.Profile.Status == "new" {
			name := "this GPU"
			if d.Name != nil {
				name = *d.Name
			}
			return detectMsg{count: d.GpuCount, newChip: name}
		}
		if err != nil || d.TargetID == nil {
			if err == nil {
				return detectMsg{count: d.GpuCount}
			}
			return detectMsg{}
		}
		driver := ""
		if d.DriverVersion != nil {
			driver = *d.DriverVersion
		}
		multi := ""
		if d.MultiGpu != nil {
			multi = *d.MultiGpu
		}
		return detectMsg{id: *d.TargetID, driver: driver, multi: multi, count: d.GpuCount}
	}
}

// autoscan matches this machine's GPU to a chip profile, or measures and saves a new one (Python decides which).
func (m Model) autoscan(remeasure bool) tea.Cmd {
	return func() tea.Msg { s, _, err := m.client.Autoscan(remeasure); return scanMsg{s, err} }
}

// findFiles lists model files to pick from: the folder of the current path and ~/models. Names only; the
// model itself is read by Python.
func (m Model) findFiles() tea.Cmd {
	dirs := []string{}
	if m.f.Path != "" {
		dirs = append(dirs, filepath.Dir(m.f.Path))
	}
	if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs, filepath.Join(home, "models"))
	}
	return func() tea.Msg {
		seen, out := map[string]bool{}, []string{}
		for _, dir := range dirs {
			for _, ext := range []string{"*.gguf", "*.safetensors"} {
				found, _ := filepath.Glob(filepath.Join(dir, ext))
				for _, p := range found {
					if !seen[p] && len(out) < 10 {
						seen[p] = true
						out = append(out, p)
					}
				}
			}
		}
		return filesMsg(out)
	}
}

func (m Model) loadRuns() tea.Cmd {
	return func() tea.Msg { r, _, err := m.client.List(); return runsMsg{r, err} }
}

func (m Model) loadRun(id string) tea.Cmd {
	return func() tea.Msg { r, _, err := m.client.Show(id); return runMsg{r, err} }
}

func (m Model) loadJob(id string) tea.Cmd {
	return func() tea.Msg {
		job, err := m.store.Status(id)
		if err != nil {
			return jobMsg{nil, nil}
		}
		lines, _ := m.store.Tail(id, 200)
		return jobMsg{&job, lines}
	}
}

// loadReady asks Python whether this machine can compare kernels for the run.
func (m Model) loadReady(id string) tea.Cmd {
	return func() tea.Msg {
		r, err := m.client.CompareReady(id)
		if err != nil {
			return readyMsg{id, nil}
		}
		return readyMsg{id, r}
	}
}

// loadCompare reads the compare job's state and log tail. No job file means no comparison ever ran from here.
func (m Model) loadCompare(id string) tea.Cmd {
	return func() tea.Msg {
		job, err := m.store.Status(compareID(id))
		if err != nil {
			return compareJobMsg{nil, nil}
		}
		lines, _ := m.store.Tail(compareID(id), 200)
		return compareJobMsg{&job, lines}
	}
}

// startStep5 starts one of step 5's jobs (compare, or time each role) in the step's one job slot.
func (m Model) startStep5(timeOnly bool) tea.Cmd {
	run := m.f.Run
	provider := m.f.runProvider()
	return func() tea.Msg {
		if run == nil {
			return noteMsg("No run to compare kernels for.")
		}
		dir, err := m.client.RunDir(run.ID)
		if err != nil {
			return noteMsg(err.Error())
		}
		argv := m.client.CompareArgv(dir)
		if timeOnly {
			argv = m.client.RoleTimeArgv(dir, provider)
		}
		if _, err := m.store.Start(compareID(run.ID), m.client.Repo, argv); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return compareStartedMsg(run.ID)
	}
}

func (m Model) inspect() tea.Cmd {
	path := m.f.Path
	return func() tea.Msg { p, _, err := m.client.Inspect(path); return profileMsg{p, err} }
}

func (m Model) loadCeiling() tea.Cmd {
	path, target, context, here := m.f.Path, m.targetID(), m.context, m.f.thisChip()
	return func() tea.Msg { c, _, err := m.client.Ceiling(path, target, context, here); return ceilingMsg{c, err} }
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
	base := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))
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
		if p := m.f.provider(*r.Measure.Provider); p != nil && p.Available {
			m.f.Engine = p.Provider
		}
		return
	}
}

// follow reloads the run the checklist is about when it changed under the model, the chip or the runs list.
func (m *Model) follow() tea.Cmd {
	if m.runSet {
		return nil
	}
	id := m.newestRun()
	if id == m.runID() {
		return nil
	}
	m.f.Run, m.f.Job, m.f.Tail = nil, nil, nil
	if id == "" {
		return nil
	}
	return m.loadRun(id)
}

func (m Model) startRun(provider string) tea.Cmd {
	path, target, runs, layout := m.f.Path, m.targetID(), m.f.Runs, m.f.layout()
	if m.f.GpuCount < 2 {
		layout = "" // one GPU: the engine runs as it always has
	}
	store, batch := m.store, ""
	if b := m.f.batch(); b > 1 {
		batch = strconv.Itoa(b) // batch 1 is always timed beside it
	}
	return func() tea.Msg {
		if live := store.Alive(); len(live) > 0 { // two measurements on one GPU would both be wrong
			return noteMsg("Run " + live[0].ID + " is still going. Stop it or wait, then press Run.")
		}
		if path == "" || target == "" {
			return noteMsg("Pick a model in step 1 and a chip in step 2 first.")
		}
		id := seam.NextName(runs, RunStem(path, target))
		dir, err := m.client.RunDir(id)
		if err != nil {
			return noteMsg(err.Error())
		}
		if abs, err := filepath.Abs(path); err == nil { // the pipeline runs in the checkout, not here
			path = abs
		}
		argv := m.client.PipelineArgv(seam.Pipeline{Model: path, RunDir: dir, Target: target, Workload: "decode", Measure: "auto", Provider: provider,
			Layout: layout, Analyze: true, Batch: batch})
		if _, err := m.store.Start(id, m.client.Repo, argv); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return startedMsg(id)
	}
}

func (m Model) stopRun() tea.Cmd {
	id, alive := m.runID(), m.f.alive()
	if m.f.comparing() {
		return func() tea.Msg {
			if _, err := m.store.Stop(compareID(id)); err != nil {
				return noteMsg("Stop failed: " + err.Error())
			}
			return noteMsg("Sent SIGTERM to the kernel comparison. Roles that finished keep their result.")
		}
	}
	return func() tea.Msg {
		if !alive {
			return noteMsg("No run is going from here.")
		}
		if _, err := m.store.Stop(id); err != nil {
			return noteMsg("Stop failed: " + err.Error())
		}
		return noteMsg("Sent SIGTERM to " + id + ". The run folder keeps the stages that finished.")
	}
}

// deleteRun removes a finished run folder; a run whose job is alive is refused here, since only Go knows the job.
// loadSaved lists the saved runs.
func (m Model) loadSaved() tea.Cmd {
	return func() tea.Msg { s, _, err := m.client.SavedRuns(); return savedListMsg{s, err} }
}

// saveRun exports the run on screen into the saved-runs folder. Python copies; this only asks.
func (m Model) saveRun() tea.Cmd {
	id := m.runID()
	return func() tea.Msg {
		s, _, err := m.client.Save(id)
		if err != nil {
			return savedMsg{id, "", err}
		}
		return savedMsg{id, s.Dir, nil}
	}
}

// openSaved reads a saved run to show its results, read-only.
func (m Model) openSaved(id string) tea.Cmd {
	return func() tea.Msg { r, err := m.client.ShowSaved(id); return savedRunMsg{r, err} }
}

// deleteSaved removes a saved run, after the second press.
func (m Model) deleteSaved(id string) tea.Cmd {
	return func() tea.Msg {
		if _, err := m.client.DeleteSaved(id); err != nil {
			return noteMsg("Delete failed: " + err.Error())
		}
		return deletedMsg("saved:" + id)
	}
}

// cleanWork deletes the unsaved runs from earlier sessions; never one of this session, never a live one.
func (m Model) cleanWork() tea.Cmd {
	keep := append([]string{}, m.session...)
	if m.f.Run != nil {
		keep = append(keep, m.f.Run.ID)
	}
	before := m.started
	return func() tea.Msg { _, err := m.client.CleanWork(before, keep); return cleanedMsg{err} }
}

func (m Model) deleteRun(id string) tea.Cmd {
	return func() tea.Msg {
		if job, err := m.store.Status(id); err == nil && job.Alive {
			return noteMsg("Stop run " + id + " before deleting it.")
		}
		if _, err := m.client.Delete(id); err != nil {
			return noteMsg("Delete failed: " + err.Error())
		}
		return deletedMsg(id)
	}
}

// openReport hands report.html to the desktop. This is the one place the TUI runs something other than Python.
func (m Model) openReport() tea.Cmd {
	run := m.f.Run
	return func() tea.Msg {
		dir, err := m.client.RunDir(run.ID)
		if err != nil {
			return noteMsg(err.Error())
		}
		opener := "xdg-open"
		if runtime.GOOS == "darwin" {
			opener = "open"
		}
		if err := exec.Command(opener, filepath.Join(dir, *run.Report)).Start(); err != nil {
			return noteMsg("Could not open the report: " + err.Error())
		}
		return noteMsg("Opened " + filepath.Join(dir, *run.Report))
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

// followMeasured reloads the speed limit once when the run on screen measured this chip's memory speed and the
// limit Setup shows does not use it yet: one number for the chip line, the limit and the run.
func (m *Model) followMeasured() tea.Cmd {
	r := m.f.Run
	if r == nil || m.f.ReadOnly || m.f.Profile == nil || !m.f.thisChip() {
		return nil
	}
	rc := r.Results.Ceiling
	if rc.BandwidthSource == nil || rc.PeakBandwidthGBs == nil || r.TargetID != m.targetID() {
		return nil
	}
	if c := m.f.Ceiling; c != nil && c.PeakBandwidthGBs == *rc.PeakBandwidthGBs {
		return nil
	}
	key := r.ID + " " + strconv.FormatFloat(*rc.PeakBandwidthGBs, 'f', 3, 64)
	if m.ceilAsked == key || m.f.CeilBusy {
		return nil
	}
	m.ceilAsked, m.f.CeilBusy = key, true
	return m.loadCeiling()
}

// chipChanged reloads what depends on the chip: the speed limit and the run.
func (m *Model) chipChanged() tea.Cmd {
	m.f.Ceiling, m.f.CeilErr, m.f.Providers = nil, "", nil
	cmds := []tea.Cmd{m.follow()}
	if t := m.f.target(); t != nil {
		cmds = append(cmds, m.loadProviders(t.ID))
	}
	if m.f.Profile != nil {
		m.f.CeilBusy = true
		cmds = append(cmds, m.loadCeiling())
	}
	return tea.Batch(cmds...)
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	next, cmd := m.update(msg)
	next.skipHeads(1) // the cursor never rests on a heading
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
	case profileMsg:
		m.f.Reading = false
		if msg.err != nil {
			m.f.Profile, m.f.ModelErr = nil, msg.err.Error()
			return m, nil
		}
		m.f.Profile, m.f.ModelErr = msg.profile, ""
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
		reload := m.followMeasured()
		if changed {
			m.f.Ready, m.f.CJob, m.f.CTail = nil, nil, nil
			return m, tea.Batch(m.loadJob(msg.run.ID), m.loadReady(msg.run.ID), m.loadCompare(msg.run.ID), reload)
		}
		if reload != nil {
			return m, reload
		}
	case providersMsg:
		if t := m.f.target(); t != nil && t.ID == msg.target {
			m.f.Providers = msg.providers
			if p := m.f.engineRow(); (p == nil || !p.Available) && msg.providers != nil {
				m.f.Engine = "" // the first engine this machine has, until one is picked
				for _, r := range msg.providers.Providers {
					if r.Available {
						m.f.Engine = r.Provider
						break
					}
				}
			}
			m.rememberEngine()
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
	case compareStartedMsg:
		id := string(msg)
		m.ticking = true
		m.f.CJob, m.f.CTail = &jobs.Job{ID: compareID(id), Alive: true}, nil
		m.note = "Started the job for " + id + ". It takes a few minutes."
		return m, tea.Batch(m.loadCompare(id), tick())
	case jobMsg:
		m.f.Job, m.f.Tail = msg.job, msg.tail
		m.f.advance(clock())
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
	case startedMsg:
		id := string(msg)
		m.cursor, m.row = pageRun, 0
		m.f.ReadOnly, m.session = false, append(m.session, id)
		m.runSet, m.ticking = true, true
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
		m.f.Run, m.f.Job, m.f.Tail, m.f.ReadOnly, m.runSet = msg.run, nil, nil, true, true
		m.cursor, m.row = pageRun, 0
		m.view.GotoTop()
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
	case noteMsg:
		m.note = string(msg)
	case tickMsg:
		m.f.advance(clock())
		id := m.runID()
		if m.f.alive() {
			return m, tea.Batch(m.loadJob(id), m.loadRuns(), m.loadRun(id), tick())
		}
		if m.f.comparing() {
			return m, tea.Batch(m.loadCompare(id), m.loadRun(id), tick())
		}
		m.ticking = false
		return m, tea.Batch(m.loadRuns(), m.loadRun(id))
	case tea.KeyMsg:
		return m.key(msg)
	}
	return m, nil
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
	DetailView(m.f, m.cursor, m.row, m.width, m.height-3, &m.view) // size the scroll before keys move it
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
	case key.Matches(msg, keyLog) && m.cursor == pageRun && m.f.alive():
		m.f.ShowLog = !m.f.ShowLog
	case key.Matches(msg, keyBack):
		if m.cursor != pageSetup {
			m.backToSetup()
		}
	case key.Matches(msg, keyDown): // the rows first, then the page above them scrolls
		if m.row < len(m.actions())-1 {
			m.row++
			m.skipHeads(1)
		} else {
			m.view.LineDown(1)
		}
	case key.Matches(msg, keyUp):
		if m.view.YOffset > 0 {
			m.view.LineUp(1)
		} else if m.row > 0 {
			m.row--
			m.skipHeads(-1)
		}
	case key.Matches(msg, keyEnter):
		if acts := m.actions(); m.row < len(acts) {
			return m.do(acts[m.row])
		}
	}
	return m, nil
}

// backToSetup shows Setup again; from a picker the cursor rests on that picker's line.
func (m *Model) backToSetup() {
	m.row = 0
	if picker(m.cursor) {
		m.row = m.cursor - pageModel
	}
	m.cursor, m.f.ReadOnly = pageSetup, false
	m.skipHeads(1)
	m.view.GotoTop()
}

func (m Model) readModel() (Model, tea.Cmd) {
	m.f.Reading, m.f.Profile, m.f.ModelErr, m.f.Ceiling, m.f.CeilErr = true, nil, "", nil, ""
	m.input.SetValue(m.f.Path)
	return m, tea.Batch(m.inspect(), m.follow())
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
		return m, m.openSaved(a.arg)
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
		m.skipHeads(1)
		m.view.GotoTop()
		if m.cursor == pageSaved {
			return m, m.loadSaved()
		}
		if m.cursor == pageSetup {
			m.f.ReadOnly = false
		}
		return m, nil
	case "engine":
		m.f.Engine, m.f.Layout, m.engineSet = a.arg, "", true
		m.backToSetup()
		return m, nil
	case "batch":
		if n, err := strconv.Atoi(a.arg); err == nil && n >= 1 && n <= 512 {
			m.f.Batch, m.f.BatchTyped = n, ""
			m.backToSetup()
			return m, nil
		}
		return m, func() tea.Msg { return noteMsg("Type a batch size from 1 to 512.") }
	case "layout":
		m.f.Layout = a.arg
		return m, nil
	case "analyze":
		return m, m.startRun(m.f.Engine)
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
		if m.f.GpuCount < 2 { // more GPUs: the layout is chosen on the same picker
			m.backToSetup()
		}
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

func footer(back, onRun, running bool) string {
	pairs := [][2]string{{"↑↓", "move"}, {"enter", "pick"}, {"q", "quit"}}
	switch {
	case running:
		pairs = [][2]string{{"↑↓", "move"}, {"enter", "pick"}, {"l", "log"}, {"esc", "setup"}, {"q", "quit"}}
	case onRun:
		pairs = [][2]string{{"↑↓", "move"}, {"enter", "open run"}, {"d", "delete run"}, {"esc", "setup"}, {"q", "quit"}}
	case back:
		pairs = [][2]string{{"↑↓", "move"}, {"enter", "pick"}, {"esc", "setup"}, {"q", "quit"}}
	}
	parts := []string{}
	for _, p := range pairs {
		parts = append(parts, stAccent.Render(p[0])+" "+stMuted.Render(p[1]))
	}
	return " " + strings.Join(parts, stMuted.Render(" · "))
}

func (m Model) View() string {
	left := stAccent.Render(glyphBolt) + " " + gradient("BoltBeam")
	right := stAccent.Render(m.mood())
	if m.f.Reading || m.f.CeilBusy || m.f.alive() || m.f.comparing() {
		right = m.spin.View() + " " + right
	}
	header := left + strings.Repeat(" ", max(m.width-lipgloss.Width(left)-lipgloss.Width(right), 1)) + right
	room := m.height - 3
	view := m.view
	body := DetailView(m.f, m.cursor, m.row, m.width, room, &view)
	return header + "\n" + body + "\n" + truncate(m.note, m.width) + "\n" + footer(m.cursor != pageSetup, m.runRow() != "", m.cursor == pageRun && m.f.alive())
}
