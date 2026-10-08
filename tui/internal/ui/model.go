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
	keyStop  = key.NewBinding(key.WithKeys("x"))
	keyQuit  = key.NewBinding(key.WithKeys("q", "ctrl+c"))
)

// Model is the checklist: seam facts, the step cursor, and whether that step's full view is open.
type Model struct {
	client   seam.Client
	store    jobs.Store
	f        Facts
	context  int
	wantChip string
	chipSet  bool // the user picked a chip; detection no longer moves it
	runSet   bool // the user picked a run; the checklist no longer follows the newest one
	cursor   int
	moved    bool // the user moved; the cursor no longer follows the first open step
	open     bool
	row      int
	ticking  bool
	width    int
	height   int
	input    textinput.Model
	note     string
	view     viewport.Model
	spin     spinner.Model
}

type targetsMsg struct {
	targets *seam.Targets
	err     error
}
type detectMsg struct{ id string }
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
type noteMsg string

// deletedMsg names a run folder that is gone.
type deletedMsg string
type tickMsg time.Time

// New builds the model; modelPath and target may be empty, context is the prefill length for the ceiling.
func New(client seam.Client, store jobs.Store, modelPath, target string, context int) Model {
	in := textinput.New()
	in.Prompt = ""
	in.SetValue(modelPath)
	return Model{client: client, store: store, f: Facts{Path: modelPath, Reading: modelPath != ""}, context: context,
		wantChip: target, chipSet: target != "", input: in, width: 80, height: 24, view: viewport.New(80, 21),
		spin: spinner.New(spinner.WithSpinner(spinner.MiniDot), spinner.WithStyle(stAccent))}
}

// Start runs the program on the terminal.
func Start(client seam.Client, store jobs.Store, modelPath, target string, context int) error {
	_, err := tea.NewProgram(New(client, store, modelPath, target, context), tea.WithAltScreen()).Run()
	return err
}

func (m Model) Init() tea.Cmd {
	cmds := []tea.Cmd{m.loadTargets(), m.detect(), m.loadRuns(), m.findFiles(), m.spin.Tick}
	if m.f.Path != "" {
		cmds = append(cmds, m.inspect())
	}
	return tea.Batch(cmds...)
}

func (m Model) loadTargets() tea.Cmd {
	return func() tea.Msg { t, _, err := m.client.Targets(); return targetsMsg{t, err} }
}

func (m Model) detect() tea.Cmd {
	return func() tea.Msg {
		d, err := m.client.Detect()
		if err != nil || d.TargetID == nil {
			return detectMsg{""}
		}
		return detectMsg{*d.TargetID}
	}
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

func (m Model) inspect() tea.Cmd {
	path := m.f.Path
	return func() tea.Msg { p, _, err := m.client.Inspect(path); return profileMsg{p, err} }
}

func (m Model) loadCeiling() tea.Cmd {
	path, target, context := m.f.Path, m.targetID(), m.context
	return func() tea.Msg { c, _, err := m.client.Ceiling(path, target, context); return ceilingMsg{c, err} }
}

func tick() tea.Cmd {
	return tea.Tick(1500*time.Millisecond, func(t time.Time) tea.Msg { return tickMsg(t) })
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

func (m Model) startRun() tea.Cmd {
	path, target, runs := m.f.Path, m.targetID(), m.f.Runs
	return func() tea.Msg {
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
		argv := m.client.PipelineArgv(seam.Pipeline{Model: path, RunDir: dir, Target: target, Workload: "decode", Measure: "auto"})
		if _, err := m.store.Start(id, m.client.Repo, argv); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return startedMsg(id)
	}
}

func (m Model) stopRun() tea.Cmd {
	id, alive := m.runID(), m.f.alive()
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

// chipChanged reloads what depends on the chip: the speed limit and the run.
func (m *Model) chipChanged() tea.Cmd {
	m.f.Ceiling, m.f.CeilErr = nil, ""
	cmds := []tea.Cmd{m.follow()}
	if m.f.Profile != nil {
		m.f.CeilBusy = true
		cmds = append(cmds, m.loadCeiling())
	}
	return tea.Batch(cmds...)
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	next, cmd := m.update(msg)
	if !next.moved {
		next.cursor = firstOpen(next.f)
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
	case detectMsg:
		m.f.ThisMac = msg.id
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
		return m, m.chipChanged()
	case ceilingMsg:
		m.f.CeilBusy = false
		m.f.Ceiling, m.f.CeilErr = msg.ceiling, ""
		if msg.err != nil {
			m.f.Ceiling, m.f.CeilErr = nil, msg.err.Error()
		}
	case runsMsg:
		m.f.Runs = msg.runs
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
		if changed {
			return m, m.loadJob(msg.run.ID)
		}
	case jobMsg:
		m.f.Job, m.f.Tail = msg.job, msg.tail
		if m.f.alive() && !m.ticking {
			m.ticking = true
			return m, tick()
		}
	case startedMsg:
		id := string(msg)
		m.runSet, m.ticking = true, true
		m.f.Run = &seam.Run{Summary: seam.Summary{ID: id}}
		m.f.Job, m.f.Tail = &jobs.Job{ID: id, Alive: true}, nil
		m.note = "Started " + id + "."
		return m, tea.Batch(m.loadJob(id), tick())
	case deletedMsg:
		if m.runID() == string(msg) {
			m.f.Run, m.f.Job, m.f.Tail, m.runSet, m.row = nil, nil, nil, false, 0
		}
		m.note = "Deleted run " + string(msg) + "."
		return m, m.loadRuns()
	case noteMsg:
		m.note = string(msg)
	case tickMsg:
		id := m.runID()
		if m.f.alive() {
			return m, tea.Batch(m.loadJob(id), m.loadRuns(), m.loadRun(id), tick())
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

func (m Model) key(msg tea.KeyMsg) (Model, tea.Cmd) {
	if !key.Matches(msg, keyEnter) {
		m.f.Confirm = "" // any other key cancels a pending delete
	}
	if m.f.Editing {
		switch msg.Type {
		case tea.KeyEnter:
			m.f.Editing, m.f.Path = false, strings.TrimSpace(m.input.Value())
			m.input.Blur()
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
	if m.open {
		DetailView(m.f, m.cursor, m.row, m.width, m.height-3, &m.view) // size the scroll before keys move it
	}
	switch {
	case key.Matches(msg, keyQuit):
		return m, tea.Quit
	case key.Matches(msg, keyStop):
		return m, m.stopRun()
	case key.Matches(msg, keyBack):
		m.open = false
	case key.Matches(msg, keyDown):
		switch {
		case !m.open:
			m.moved, m.cursor = true, min(m.cursor+1, len(steps)-1)
		case m.row < len(m.actions())-1:
			m.row++
		default:
			m.view.LineDown(1)
		}
	case key.Matches(msg, keyUp):
		switch {
		case !m.open:
			m.moved, m.cursor = true, max(m.cursor-1, 0)
		case m.view.YOffset > 0:
			m.view.LineUp(1)
		case m.row > 0:
			m.row--
		}
	case key.Matches(msg, keyEnter):
		if !m.open {
			m.moved, m.open, m.row = true, true, 0
			if m.cursor == 1 { // the chip list opens on the chip in use
				m.row = m.f.Target
			}
			m.view.GotoTop()
			return m, nil
		}
		if acts := m.actions(); m.row < len(acts) {
			return m.do(acts[m.row])
		}
	}
	return m, nil
}

func (m Model) readModel() (Model, tea.Cmd) {
	m.f.Reading, m.f.Profile, m.f.ModelErr, m.f.Ceiling, m.f.CeilErr = true, nil, "", nil, ""
	m.input.SetValue(m.f.Path)
	return m, tea.Batch(m.inspect(), m.follow())
}

// do runs one action row of the open step.
func (m Model) do(a action) (Model, tea.Cmd) {
	if a.do != "delete" {
		m.f.Confirm = ""
	}
	switch a.do {
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
		return m.readModel()
	case "chip":
		m.f.Target, _ = strconv.Atoi(a.arg)
		m.chipSet = true
		return m, m.chipChanged()
	case "start":
		return m, m.startRun()
	case "stop":
		return m, m.stopRun()
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
	case f.Reading || f.CeilBusy || f.alive():
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

func footer(open bool) string {
	pairs := [][2]string{{"↑↓", "step"}, {"enter", "open"}, {"q", "quit"}}
	if open {
		pairs = [][2]string{{"↑↓", "move"}, {"enter", "pick"}, {"esc", "back"}, {"x", "stop"}, {"q", "quit"}}
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
	if m.f.Reading || m.f.CeilBusy || m.f.alive() {
		right = m.spin.View() + " " + right
	}
	header := left + strings.Repeat(" ", max(m.width-lipgloss.Width(left)-lipgloss.Width(right), 1)) + right
	room := m.height - 3
	var body string
	if m.open {
		view := m.view
		body = DetailView(m.f, m.cursor, m.row, m.width, room, &view)
	} else {
		body = ChecklistView(m.f, m.cursor, m.width, room)
		if pad := room - lipgloss.Height(body); pad > 0 {
			body += strings.Repeat("\n", pad)
		}
	}
	return header + "\n" + body + "\n" + truncate(m.note, m.width) + "\n" + footer(m.open)
}
