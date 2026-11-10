package ui

import (
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// The boot prompt for a newer BoltBeam, as Claude Code and Codex ask: Upgrade now, Skip, Skip this version. Python
// (cli/self_update.py) owns the versions, the cache, the skip and the command; the check runs beside the boot's other
// calls, so the screen never waits for it, and a failed or switched-off check shows nothing.

const selfUpdateJob = "boltbeam-update" // the upgrade's job id in the jobs store: its log is shown while it runs

type selfUpdateMsg struct {
	u   *seam.SelfUpdate
	err error
}
type selfUpdateTickMsg struct{}

type selfUpdate struct {
	u       *seam.SelfUpdate
	shown   bool // the prompt, or the upgrade's log, covers the screen
	row     int
	running bool
	lines   []string // this upgrade's log
	ended   string   // its last line, once it stopped
}

func (m Model) checkSelfUpdate() tea.Cmd {
	return func() tea.Msg { u, err := m.client.SelfUpdate(""); return selfUpdateMsg{u, err} }
}

func selfUpdateTick() tea.Cmd {
	return tea.Tick(500*time.Millisecond, func(time.Time) tea.Msg { return selfUpdateTickMsg{} })
}

// selfUpdateRows are the prompt's choices; a checkout gets no Upgrade now, as BoltBeam runs no git for you.
func selfUpdateRows(s selfUpdate) []string {
	switch {
	case s.running:
		return nil
	case s.ended != "":
		return []string{"Continue"}
	case s.u.Command == nil:
		return []string{"Skip", "Skip this version"}
	}
	return []string{"Upgrade now", "Skip", "Skip this version"}
}

// selfUpdateStep takes the prompt's own messages and, while it is shown, the keys; false passes msg on.
func (m Model) selfUpdateStep(msg tea.Msg) (Model, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case selfUpdateMsg:
		if msg.err == nil && msg.u != nil && msg.u.Prompt && msg.u.Latest != nil && msg.u.Installed != nil {
			m.upd = selfUpdate{u: msg.u, shown: true}
		}
		return m, nil, true
	case selfUpdateTickMsg:
		m.followSelfUpdate()
		if m.upd.running {
			return m, selfUpdateTick(), true
		}
		return m, nil, true
	case tea.KeyMsg:
		if !m.upd.shown {
			return m, nil, false
		}
		rows := selfUpdateRows(m.upd)
		switch {
		case key.Matches(msg, keyQuit):
			return m, tea.Quit, true // a running upgrade is its own process group: it finishes without the screen
		case key.Matches(msg, keyDown) && m.upd.row < len(rows)-1:
			m.upd.row++
		case key.Matches(msg, keyUp) && m.upd.row > 0:
			m.upd.row--
		case key.Matches(msg, keyBack) && !m.upd.running:
			m.upd.shown = false
		case key.Matches(msg, keyEnter) && m.upd.row < len(rows):
			return m.pickSelfUpdate(rows[m.upd.row])
		}
		return m, nil, true
	}
	return m, nil, false
}

func (m Model) pickSelfUpdate(choice string) (Model, tea.Cmd, bool) {
	switch choice {
	case "Upgrade now":
		if _, err := m.store.Start(selfUpdateJob, m.client.Repo, m.client.SelfUpdateArgv()); err != nil {
			m.upd.ended, m.upd.row = seam.SelfUpdateFailed+" could not start the upgrade: "+err.Error(), 0
			return m, nil, true
		}
		m.upd.running = true
		return m, selfUpdateTick(), true
	case "Skip this version":
		m.upd.shown = false
		latest := *m.upd.u.Latest
		return m, func() tea.Msg { u, err := m.client.SelfUpdate(latest); return selfUpdateMsg{u, err} }, true
	}
	m.upd.shown = false // Skip: this session only; Continue: the upgrade is over
	return m, nil, true
}

// followSelfUpdate reads the upgrade's log since its start line; once the job is gone, its last line says how it ended.
func (m *Model) followSelfUpdate() {
	job, err := m.store.Status(selfUpdateJob)
	lines, _ := m.store.Tail(selfUpdateJob, 200)
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.HasPrefix(lines[i], "=== ") {
			lines = lines[i+1:]
			break
		}
	}
	m.upd.lines = lines
	if err == nil && job.Alive {
		return
	}
	m.upd.running, m.upd.row = false, 0
	m.upd.ended = seam.SelfUpdateFailed + " the upgrade stopped without saying how it ended; its log: " + job.LogPath
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.HasPrefix(lines[i], seam.SelfUpdateDone) || strings.HasPrefix(lines[i], seam.SelfUpdateFailed) {
			m.upd.ended = lines[i]
			break
		}
	}
}

// selfUpdateView is the whole body while the prompt is shown: one box, padded to the screen.
func selfUpdateView(s selfUpdate, width, height int) string {
	var b strings.Builder
	b.WriteString(stStrong.Render("BoltBeam "+*s.u.Latest+" is available") + " (you have " + *s.u.Installed + ").\n\n")
	lead := "Upgrade now runs: "
	if s.u.Command == nil { // a checkout: the instruction, nothing to run
		lead = "This copy is "
	}
	b.WriteString(stMuted.Render(lead + s.u.Words))
	if s.running || s.ended != "" {
		room := max(height-10, 1)
		lines := s.lines
		if n := len(lines); n > 0 && s.ended != "" && lines[n-1] == s.ended {
			lines = lines[:n-1] // the ending is shown once, colored, below
		}
		if len(lines) > room {
			lines = lines[len(lines)-room:]
		}
		b.WriteString("\n\n" + strings.Join(lines, "\n") + "\n")
	}
	switch {
	case s.running:
		b.WriteString("\n" + stInfo.Render("Upgrading…"))
	case strings.HasPrefix(s.ended, seam.SelfUpdateDone):
		b.WriteString("\n" + stOK.Render(s.ended))
	case s.ended != "":
		b.WriteString("\n" + stBad.Render(s.ended))
	}
	for i, r := range selfUpdateRows(s) {
		if i == 0 {
			b.WriteString("\n")
		}
		if i == s.row {
			b.WriteString("\n" + stCursor.Render("▸ ") + r)
		} else {
			b.WriteString("\n  " + r)
		}
	}
	out := box("Update available", b.String(), width, false)
	return out + strings.Repeat("\n", max(height-lipgloss.Height(out), 0))
}
