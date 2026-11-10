package ui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func newer(command [][]string, words string) *seam.SelfUpdate {
	have, latest := "0.3.13", "0.3.14"
	return &seam.SelfUpdate{Status: "newer", Prompt: true, Installed: &have, Latest: &latest, Install: "homebrew",
		Command: command, Words: words}
}

func booted(t *testing.T, store jobs.Store) tea.Model {
	s := loadSample(t)
	var m tea.Model = New(seam.Client{}, store, "/models/Qwen3-8B.gguf", "", 512)
	for _, msg := range []tea.Msg{targetsMsg{&s.targets, nil}, detectMsg{id: "apple_m3_10c", count: 1}, profileMsg{&s.profile, nil},
		ceilingMsg{&s.ceiling, nil}, runsMsg{&s.runs, nil}, tea.WindowSizeMsg{Width: 80, Height: 24}} {
		m, _ = m.Update(msg)
	}
	return m
}

func hit(m tea.Model, ks ...tea.KeyType) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	for _, k := range ks {
		m, cmd = m.Update(tea.KeyMsg{Type: k})
	}
	return m, cmd
}

// A newer release prompts over Setup: Upgrade now, Skip, Skip this version. Skip is this session only.
func TestSelfUpdatePromptsForANewerRelease(t *testing.T) {
	m := booted(t, jobs.Store{})
	m, _ = m.Update(selfUpdateMsg{newer([][]string{{"brew", "update"}, {"brew", "upgrade", "boltbeam"}}, "brew update && brew upgrade boltbeam"), nil})
	view := m.View()
	for _, want := range []string{"BoltBeam 0.3.14 is available (you have 0.3.13).", "Upgrade now runs: brew update && brew upgrade boltbeam",
		"▸ Upgrade now", "Skip this version"} {
		if !strings.Contains(plain(view), want) {
			t.Fatalf("want %q in the prompt:\n%s", want, view)
		}
	}
	golden(t, "update-prompt.txt", view)
	m, cmd := hit(m, tea.KeyDown, tea.KeyEnter) // Skip
	if cmd != nil || m.(Model).upd.shown || strings.Contains(m.View(), "is available") {
		t.Fatalf("Skip dismisses the prompt and records nothing:\n%s", m.View())
	}
}

// Skip this version asks Python to record it (a seam call); the prompt goes at once.
func TestSelfUpdateSkipThisVersionRecordsIt(t *testing.T) {
	m := booted(t, jobs.Store{})
	m, _ = m.Update(selfUpdateMsg{newer([][]string{{"brew", "upgrade", "boltbeam"}}, "brew upgrade boltbeam"), nil})
	m, cmd := hit(m, tea.KeyDown, tea.KeyDown, tea.KeyEnter)
	if cmd == nil || m.(Model).upd.shown {
		t.Fatal("Skip this version hides the prompt and records the version through the seam")
	}
}

// No prompt when the check failed, is switched off (BOLTBEAM_NO_UPDATE_CHECK) or found nothing newer.
func TestSelfUpdateQuietWithoutANewerRelease(t *testing.T) {
	have := "0.3.13"
	for _, msg := range []selfUpdateMsg{{nil, os.ErrDeadlineExceeded}, {&seam.SelfUpdate{Status: "off", Installed: &have}, nil},
		{&seam.SelfUpdate{Status: "unknown", Installed: &have}, nil}, {&seam.SelfUpdate{Status: "current", Installed: &have, Latest: &have}, nil},
		{&seam.SelfUpdate{Status: "skipped", Installed: &have, Latest: &have}, nil}} {
		m, _ := booted(t, jobs.Store{}).Update(msg)
		if m.(Model).upd.shown || strings.Contains(m.View(), "is available") {
			t.Fatalf("%+v must not prompt:\n%s", msg, m.View())
		}
	}
}

// A checkout is pulled by its owner: the prompt says how and offers no Upgrade now.
func TestSelfUpdateCheckoutShowsTheInstruction(t *testing.T) {
	m := booted(t, jobs.Store{})
	u := newer(nil, "a git checkout: pull it yourself (git -C /src/BoltBeam pull); BoltBeam runs no git for you")
	u.Install = "checkout"
	m, _ = m.Update(selfUpdateMsg{u, nil})
	view := plain(m.View())
	if strings.Contains(view, "Upgrade now") || !strings.Contains(view, "This copy is a git checkout: pull it yourself") {
		t.Fatalf("a checkout shows the instruction, not Upgrade now:\n%s", view)
	}
}

// The upgrade's log is shown from its start line; its last line says how it ended.
func TestSelfUpdateFollowsTheUpgradeLog(t *testing.T) {
	store := jobs.Store{Dir: t.TempDir()}
	log := filepath.Join(store.Dir, selfUpdateJob+".log")
	os.WriteFile(store.Dir+"/"+selfUpdateJob+".json", []byte(`{"id":"boltbeam-update","pid":0,"log_path":"`+log+`"}`), 0o644)
	os.WriteFile(log, []byte("=== old start\nupdate failed: earlier\n=== 2026-11-08 start boltbeam update --yes\n$ brew upgrade boltbeam\n"+
		"==> Upgrading boltbeam\nupdated: BoltBeam 0.3.14 is installed; quit and run boltbeam again to use it\n"), 0o644)
	m := booted(t, store)
	m, _ = m.Update(selfUpdateMsg{newer([][]string{{"brew", "upgrade", "boltbeam"}}, "brew upgrade boltbeam"), nil})
	mm := m.(Model)
	mm.upd.running = true
	m, cmd := mm.Update(selfUpdateTickMsg{})
	got := m.(Model).upd
	if cmd != nil || got.running || !strings.HasPrefix(got.ended, "updated: BoltBeam 0.3.14") || len(got.lines) != 3 {
		t.Fatalf("the ended upgrade: %+v", got)
	}
	view := plain(m.View())
	if !strings.Contains(view, "==> Upgrading boltbeam") || !strings.Contains(view, "▸ Continue") {
		t.Fatalf("the log and Continue:\n%s", view)
	}
	if n := strings.Count(view, "updated: BoltBeam 0.3.14"); n != 1 { // the Mac's first upgrade showed it twice
		t.Fatalf("the ending is shown once, not %d times:\n%s", n, view)
	}
	if m, _ = hit(m, tea.KeyEnter); m.(Model).upd.shown {
		t.Fatal("Continue closes the prompt")
	}
}
