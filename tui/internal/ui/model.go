package ui

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

type screen int

const (
	screenModel screen = iota
	screenCeiling
	screenRun
	screenResults
)

var tabs = []string{"Model", "Ceiling", "Run", "Results"}

type keyMap struct{ Tabs, Move, Open, Back, Edit, Chip, Start, Stop, Report, Mode, Refresh, Quit key.Binding }

var keys = keyMap{
	Tabs:    key.NewBinding(key.WithKeys("1", "2", "3", "4"), key.WithHelp("1-4", "screens")),
	Move:    key.NewBinding(key.WithKeys("j", "k", "up", "down"), key.WithHelp("j/k", "move")),
	Open:    key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "read/open")),
	Back:    key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "cancel")),
	Edit:    key.NewBinding(key.WithKeys("e"), key.WithHelp("e", "edit path")),
	Chip:    key.NewBinding(key.WithKeys("[", "]"), key.WithHelp("[ ]", "chip")),
	Start:   key.NewBinding(key.WithKeys("s"), key.WithHelp("s", "start")),
	Stop:    key.NewBinding(key.WithKeys("x"), key.WithHelp("x", "stop")),
	Report:  key.NewBinding(key.WithKeys("o"), key.WithHelp("o", "open report")),
	Mode:    key.NewBinding(key.WithKeys("t"), key.WithHelp("t", "plain/technical")),
	Refresh: key.NewBinding(key.WithKeys("r"), key.WithHelp("r", "refresh")),
	Quit:    key.NewBinding(key.WithKeys("q", "ctrl+c"), key.WithHelp("q", "quit")),
}

func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Tabs, k.Move, k.Open, k.Edit, k.Chip, k.Start, k.Stop, k.Report, k.Mode, k.Refresh, k.Quit}
}
func (k keyMap) FullHelp() [][]key.Binding { return [][]key.Binding{k.ShortHelp()} }

// Model holds the seam data each screen renders. Every load is a tea.Cmd that calls the seam off the UI thread.
type Model struct {
	client     seam.Client
	store      jobs.Store
	screen     screen
	technical  bool
	cursor     int
	width      int
	height     int
	modelPath  string
	wantTarget string
	context    int
	input      textinput.Model
	editing    bool
	targets    *seam.Targets
	target     int
	profile    *seam.Profile
	ceiling    *seam.Ceiling
	reading    bool
	ceilBusy   bool
	runs       *seam.Runs
	run        *seam.Run
	jobID      string
	job        *jobs.Job
	tail       []string
	note       string
	view       viewport.Model
	spin       spinner.Model
	help       help.Model
}

type targetsMsg struct {
	targets *seam.Targets
	err     error
}
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
type noteMsg string
type tickMsg time.Time

// New builds the model; modelPath and target may be empty, context is the prefill length for the ceiling.
func New(client seam.Client, store jobs.Store, modelPath, target string, context int) Model {
	s := spinner.New(spinner.WithSpinner(spinner.MiniDot), spinner.WithStyle(stAccent))
	h := help.New()
	h.Styles.ShortKey, h.Styles.ShortDesc, h.Styles.ShortSeparator = stAccent, stMuted, stMuted
	in := textinput.New()
	in.Prompt = ""
	in.SetValue(modelPath)
	m := Model{client: client, store: store, modelPath: modelPath, context: context, input: in, width: 100, height: 30,
		view: viewport.New(100, 25), spin: s, help: h}
	m.wantTarget = target
	return m
}

// Start runs the program on the terminal.
func Start(client seam.Client, store jobs.Store, modelPath, target string, context int) error {
	_, err := tea.NewProgram(New(client, store, modelPath, target, context), tea.WithAltScreen()).Run()
	return err
}

func (m Model) Init() tea.Cmd {
	cmds := []tea.Cmd{m.loadTargets(), m.loadRuns(), m.spin.Tick}
	if m.modelPath != "" {
		cmds = append(cmds, m.inspect())
	}
	return tea.Batch(cmds...)
}

func (m Model) loadTargets() tea.Cmd {
	return func() tea.Msg { t, _, err := m.client.Targets(); return targetsMsg{t, err} }
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
	path := m.modelPath
	return func() tea.Msg { p, _, err := m.client.Inspect(path); return profileMsg{p, err} }
}

func (m Model) loadCeiling() tea.Cmd {
	path, target, context := m.modelPath, m.targetID(), m.context
	return func() tea.Msg { c, _, err := m.client.Ceiling(path, target, context); return ceilingMsg{c, err} }
}

func tick() tea.Cmd {
	return tea.Tick(1500*time.Millisecond, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m Model) targetID() string {
	if m.targets == nil || len(m.targets.Targets) == 0 {
		return ""
	}
	if m.target < 0 || m.target >= len(m.targets.Targets) {
		return m.targets.Targets[0].ID
	}
	return m.targets.Targets[m.target].ID
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

func (m Model) startRun() tea.Cmd {
	path, target, runs := m.modelPath, m.targetID(), m.runs
	return func() tea.Msg {
		if path == "" || target == "" {
			return noteMsg("Pick a model (press 1, then e) and a chip first.")
		}
		id := seam.NextName(runs, RunStem(path, target))
		dir, err := m.client.RunDir(id)
		if err != nil {
			return noteMsg(err.Error())
		}
		if abs, err := filepath.Abs(path); err == nil { // the pipeline runs in the checkout, not here
			path = abs
		}
		argv := m.client.PipelineArgv(seam.Pipeline{Model: path, RunDir: dir, Target: target, Workload: "decode"})
		if _, err := m.store.Start(id, m.client.Repo, argv); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return startedMsg(id)
	}
}

type startedMsg string

func (m Model) stopRun() tea.Cmd {
	id := m.jobID
	return func() tea.Msg {
		if id == "" {
			return noteMsg("No run was started from here.")
		}
		if _, err := m.store.Stop(id); err != nil {
			return noteMsg("Stop failed: " + err.Error())
		}
		return noteMsg("Sent SIGTERM to " + id + ". The run folder keeps the stages that finished.")
	}
}

// openReport hands report.html to the desktop. This is the one place the TUI runs something other than Python.
func (m Model) openReport() tea.Cmd {
	run := m.run
	return func() tea.Msg {
		if run == nil || run.Report == nil {
			return noteMsg("No report.html yet: the output stage has not run.")
		}
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

func (m Model) rows() int {
	if m.screen == screenRun && m.runs != nil {
		return len(m.runs.Runs)
	}
	return 0
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.view.Width, m.view.Height = msg.Width, msg.Height-4
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd
	case targetsMsg:
		m.targets = msg.targets
		if msg.err != nil {
			m.note = "The chip list could not be read: " + msg.err.Error()
		} else {
			m.pickTarget(m.wantTarget)
		}
	case profileMsg:
		m.reading = false
		m.profile = msg.profile
		if msg.err != nil {
			m.note = "The model could not be read: " + msg.err.Error()
			return m, nil
		}
		m.ceilBusy = true
		return m, m.loadCeiling()
	case ceilingMsg:
		m.ceilBusy = false
		m.ceiling = msg.ceiling
		if msg.err != nil {
			m.ceiling = nil
			m.note = "No speed limit: " + msg.err.Error()
		}
	case runsMsg:
		m.runs = msg.runs
		if msg.err != nil {
			m.note = "The runs folder could not be read: " + msg.err.Error()
		}
	case runMsg:
		if msg.err != nil {
			m.note = "The run could not be read: " + msg.err.Error()
			return m, nil
		}
		m.run = msg.run
		if m.jobID == "" || m.jobID != msg.run.ID {
			m.jobID = msg.run.ID
			return m, m.loadJob(m.jobID)
		}
	case jobMsg:
		m.job, m.tail = msg.job, msg.tail
	case startedMsg:
		m.jobID = string(msg)
		m.screen, m.note = screenRun, "Started "+m.jobID+". Each stage reports below as it finishes."
		return m, tea.Batch(m.loadRuns(), m.loadJob(m.jobID), tick())
	case noteMsg:
		m.note = string(msg)
	case tickMsg:
		if m.job != nil && m.job.Alive {
			cmds := []tea.Cmd{m.loadJob(m.jobID), m.loadRuns(), tick()}
			if m.jobID != "" {
				cmds = append(cmds, m.loadRun(m.jobID))
			}
			return m, tea.Batch(cmds...)
		}
		return m, tea.Batch(m.loadJob(m.jobID), m.loadRuns(), m.loadRun(m.jobID))
	case tea.KeyMsg:
		return m.key(msg)
	}
	return m, nil
}

func (m *Model) pickTarget(id string) {
	if m.targets == nil {
		return
	}
	for i, t := range m.targets.Targets {
		if t.ID == id {
			m.target = i
			return
		}
	}
	for i, t := range m.targets.Targets {
		if t.HasCeiling && id == "" {
			m.target = i
			return
		}
	}
}

func (m Model) key(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.editing {
		switch msg.Type {
		case tea.KeyEnter:
			m.editing, m.modelPath, m.reading, m.ceiling = false, strings.TrimSpace(m.input.Value()), true, nil
			m.input.Blur()
			return m, m.inspect()
		case tea.KeyEsc:
			m.editing = false
			m.input.Blur()
			m.input.SetValue(m.modelPath)
			return m, nil
		}
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		return m, cmd
	}
	scrolls := m.screen != screenRun
	switch {
	case key.Matches(msg, keys.Quit):
		return m, tea.Quit
	case key.Matches(msg, keys.Tabs):
		m.screen = map[string]screen{"1": screenModel, "2": screenCeiling, "3": screenRun, "4": screenResults}[msg.String()]
		m.view.GotoTop()
		if m.screen == screenRun {
			return m, tea.Batch(m.loadRuns(), m.loadJob(m.jobID))
		}
	case key.Matches(msg, keys.Move):
		down := msg.String() == "j" || msg.String() == "down"
		switch {
		case scrolls && down:
			m.view.LineDown(1)
		case scrolls:
			m.view.LineUp(1)
		case down && m.cursor+1 < m.rows():
			m.cursor++
		case !down && m.cursor > 0:
			m.cursor--
		}
	case key.Matches(msg, keys.Open):
		switch m.screen {
		case screenModel:
			if m.modelPath == "" {
				m.editing = true
				return m, m.input.Focus()
			}
			m.reading, m.ceiling = true, nil
			return m, m.inspect()
		case screenCeiling:
			if m.modelPath != "" {
				m.ceilBusy = true
				return m, m.loadCeiling()
			}
		case screenRun:
			if m.runs != nil && m.cursor < len(m.runs.Runs) {
				m.jobID = ""
				return m, m.loadRun(m.runs.Runs[m.cursor].ID)
			}
		}
	case key.Matches(msg, keys.Edit):
		if m.screen == screenModel {
			m.editing = true
			m.input.SetValue(m.modelPath)
			return m, m.input.Focus()
		}
	case key.Matches(msg, keys.Chip):
		if m.targets != nil && len(m.targets.Targets) > 0 {
			n := len(m.targets.Targets)
			if msg.String() == "]" {
				m.target = (m.target + 1) % n
			} else {
				m.target = (m.target + n - 1) % n
			}
			m.ceiling = nil
			if m.profile != nil && m.modelPath != "" {
				m.ceilBusy = true
				return m, m.loadCeiling()
			}
		}
	case key.Matches(msg, keys.Mode):
		m.technical = !m.technical
	case key.Matches(msg, keys.Refresh):
		cmds := []tea.Cmd{m.loadTargets(), m.loadRuns()}
		if m.run != nil {
			cmds = append(cmds, m.loadRun(m.run.ID), m.loadJob(m.run.ID))
		}
		return m, tea.Batch(cmds...)
	case key.Matches(msg, keys.Start):
		return m, m.startRun()
	case key.Matches(msg, keys.Stop):
		return m, m.stopRun()
	case key.Matches(msg, keys.Report):
		return m, m.openReport()
	}
	return m, nil
}

// mood picks the mascot's face from what the screen shows.
func (m Model) mood() string {
	if m.reading || m.ceilBusy || (m.job != nil && m.job.Alive) {
		return faceBusy
	}
	if m.run != nil && (m.screen == screenRun || m.screen == screenResults) {
		for _, st := range m.run.Stages {
			if seam.StageEvents(m.tail)[st.Key] == "failed" {
				return faceWorried
			}
		}
		switch {
		case m.run.Results.Measured:
			return faceHappy
		case len(m.run.Blocked) > 0:
			return faceWaiting
		}
		return faceIdle
	}
	if m.profile == nil && (m.runs == nil || len(m.runs.Runs) == 0) {
		return faceSleep
	}
	return faceIdle
}

func (m Model) body() string {
	switch m.screen {
	case screenModel:
		return ModelView(ModelScreen{Path: m.modelPath, Input: m.input.Value(), Editing: m.editing, Targets: m.targets,
			Target: m.target, Profile: m.profile, Busy: m.reading}, m.technical, m.width)
	case screenCeiling:
		return CeilingView(m.ceiling, m.ceilBusy, m.technical, m.width)
	case screenRun:
		return RunView(RunScreen{Runs: m.runs, Cursor: m.cursor, Run: m.run, Job: m.job, Tail: m.tail, Spin: m.spin.View()}, m.technical, m.width)
	}
	return ResultsView(m.run, m.technical, m.width)
}

func (m Model) View() string {
	parts := []string{stAccent.Render(glyphBolt) + " " + gradient("BoltBeam")}
	for i, t := range tabs {
		if i == int(m.screen) {
			parts = append(parts, stTabOn.Render(t))
		} else {
			parts = append(parts, stTabOff.Render(t))
		}
	}
	busy := ""
	if m.reading || m.ceilBusy || (m.job != nil && m.job.Alive) {
		busy = m.spin.View() + " "
	}
	header := strings.Join(parts, "") + "  " + busy + stAccent.Render(m.mood()) + "  " + mode(m.technical)
	view := m.view
	view.Width, view.Height = m.width, m.height-4
	view.SetContent(m.body())
	note := m.note
	if note == "" {
		note = stMuted.Render("Keys below. Press t for the technical words.")
	}
	m.help.Width = m.width
	return fmt.Sprintf("%s\n%s\n%s\n%s", header, view.View(), lipgloss.NewStyle().Width(m.width).Render(note), m.help.View(keys))
}
