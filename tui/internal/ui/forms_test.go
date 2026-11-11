package ui

import (
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"
)

// settle runs cmd and the commands its messages bring, as the program would (a batch or a sequence is each of its
// commands), and returns the model after them. Timers are not waited for: the cursor's blink and tea.Tick (the
// screen's one-second tick) are dropped.
func settle(m Model, cmd tea.Cmd) Model {
	if cmd == nil {
		return m
	}
	if name := runtime.FuncForPC(reflect.ValueOf(cmd).Pointer()).Name(); strings.Contains(name, "/cursor.") ||
		strings.Contains(name, "bubbletea/v2.Tick") {
		return m
	}
	done := make(chan tea.Msg, 1)
	go func() { done <- cmd() }()
	var msg tea.Msg
	select {
	case msg = <-done:
	case <-time.After(5 * time.Second):
		return m
	}
	if msg == nil {
		return m
	}
	if v := reflect.ValueOf(msg); v.Kind() == reflect.Slice && v.Type().Elem() == reflect.TypeFor[tea.Cmd]() {
		for i := range v.Len() {
			c, _ := v.Index(i).Interface().(tea.Cmd)
			m = settle(m, c)
		}
		return m
	}
	next, more := m.Update(msg)
	return settle(next.(Model), more)
}

// typeKeys presses each key (a name like enter, backspace or right, else text one rune at a time) and settles.
func typeKeys(m Model, ks ...string) Model {
	codes := map[string]rune{"enter": tea.KeyEnter, "backspace": tea.KeyBackspace, "right": tea.KeyRight, "left": tea.KeyLeft,
		"down": tea.KeyDown, "up": tea.KeyUp, "esc": tea.KeyEscape, "space": tea.KeySpace}
	for _, k := range ks {
		msgs := []tea.KeyPressMsg{}
		if c, ok := codes[k]; ok {
			msgs = append(msgs, tea.KeyPressMsg{Code: c})
		} else {
			for _, r := range k {
				msgs = append(msgs, char(r))
			}
		}
		for _, msg := range msgs {
			next, cmd := m.Update(msg)
			m = settle(next.(Model), cmd)
		}
	}
	return m
}

// The remote is user@host or a host; the port 1 to 65535 or nothing. Each refusal says what to type.
func TestFormChecks(t *testing.T) {
	for v, ok := range map[string]bool{"root@203.0.113.7": true, "pod-a": true, "user@my.host-1": true, "": false, "-oProxy=x": false,
		"a b": false, "user@": false, "@host": false} {
		if got := checkRemote(v) == nil; got != ok {
			t.Errorf("remote %q: ok %v", v, got)
		}
	}
	for v, ok := range map[string]bool{"": true, "22": true, "65535": true, "0": false, "65536": false, "x": false, "-1": false} {
		if got := checkPort(v) == nil; got != ok {
			t.Errorf("port %q: ok %v", v, got)
		}
	}
}

// esc leaves a form and keeps nothing typed in it; ctrl+c quits; q is typed, not a quit.
func TestFormKeysAreTheForms(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := page(program(loadSample(t), nil, nil, nil, 80, 24), pageFetch).(Model)
	m = settle(m, m.openRemote())
	m = typeKeys(m, "q@host")
	if m.form == nil || !strings.Contains(frame(m), "q@host") {
		t.Fatalf("q is typed into the remote:\n%s", frame(m))
	}
	if !strings.Contains(frame(m), "esc cancel") || !strings.Contains(frame(m), "ctrl+c quit") {
		t.Fatalf("the footer names the form's keys:\n%s", frame(m))
	}
	if _, cmd := m.Update(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl}); cmd == nil {
		t.Fatal("ctrl+c quits from a form")
	}
	m = typeKeys(m, "esc")
	if m.form != nil || m.cursor != pageSetup || m.f.Fetch.Dest != "" {
		t.Fatalf("esc: form %v, page %d, dest %q", m.form != nil, m.cursor, m.f.Fetch.Dest)
	}
}
