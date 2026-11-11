package ui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"
	"github.com/charmbracelet/x/ansi"
)

// ? opens a page of every key the screen takes, drawn from the same bindings as the footer; ? or esc closes it and
// the screen is as it was; q still quits.
func TestHelpPageListsTheScreensKeys(t *testing.T) {
	freezeClock(t)
	setProfile(colorprofile.NoTTY)
	s := loadSample(t)
	setup := program(s, nil, nil, nil, 80, 24).(Model)
	shown := press(setup, "?").(Model)
	if !shown.showHelp {
		t.Fatal("? opens the keys page")
	}
	golden(t, "help-setup.txt", frame(shown))
	got := frame(shown)
	for _, want := range []string{"pgup/ctrl+u page up", "home/g", "end/G", "enter", "? all keys", "q quit"} {
		if !strings.Contains(got, want) {
			t.Errorf("the keys page lacks %q:\n%s", want, got)
		}
	}
	for _, absent := range []string{"delete", "compare", "esc setup", "0-9", "l log"} { // none of these does anything on Setup
		if strings.Contains(got, absent) {
			t.Errorf("the keys page names %q, which does nothing on Setup", absent)
		}
	}
	if back := press(shown, "esc").(Model); back.showHelp || frame(back) != frame(setup) {
		t.Fatal("esc closes the keys page and the screen is as it was")
	}
	if back := press(shown, "?").(Model); back.showHelp {
		t.Fatal("? closes the keys page")
	}
	if _, cmd := shown.Update(char('q')); cmd == nil {
		t.Fatal("q quits from the keys page")
	}
	running := press(page(program(s, &s.planned, s.job, s.tail, 80, 24), pageRun), "?")
	golden(t, "help-run.txt", frame(running))
	batch := press(page(program(s, nil, nil, nil, 80, 24), pageBatch), "?")
	if !strings.Contains(frame(batch), "0-9") || !strings.Contains(frame(batch), "backspace") {
		t.Fatalf("the batch page types digits:\n%s", frame(batch))
	}
}

// The footer names only keys that work where it is drawn, and fits 80 columns on every screen.
func TestFooterIsTheScreensKeys(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	s := loadSample(t)
	footerOf := func(m tea.Model) string {
		lines := strings.Split(ansi.Strip(frame(m)), "\n")
		return lines[len(lines)-1]
	}
	for name, c := range map[string]struct {
		m        tea.Model
		has, not []string
	}{
		"setup":   {program(s, nil, nil, nil, 80, 24), []string{"↑↓ move", "enter pick", "? all keys", "q quit"}, []string{"esc", "l log", "d delete"}},
		"running": {page(program(s, &s.planned, s.job, s.tail, 80, 24), pageRun), []string{"l log", "esc setup"}, []string{"d delete"}},
		"picker":  {page(program(s, nil, nil, nil, 80, 24), pageChip), []string{"esc setup"}, []string{"l log"}},
	} {
		got := footerOf(c.m)
		if ansi.StringWidth(got) > 80 {
			t.Errorf("%s: the footer is wider than 80 columns: %q", name, got)
		}
		for _, want := range c.has {
			if !strings.Contains(got, want) {
				t.Errorf("%s: footer %q lacks %q", name, got, want)
			}
		}
		for _, absent := range c.not {
			if strings.Contains(got, absent) {
				t.Errorf("%s: footer %q names %q", name, got, absent)
			}
		}
	}
}
