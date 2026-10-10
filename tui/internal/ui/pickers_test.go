package ui

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// Every picker draws its rows through DetailActions, which windows a long list around the cursor and counts what
// is above and below. No picker caps its list: a row that sorts last is still reachable. Pinned for the two lists
// that grow on a real machine (model files, saved runs); Chip, Engine, Batch and Measurement use the same routine.
func TestPickersWindowLongListsWithoutDroppingRows(t *testing.T) {
	files := make([]string, 30)
	for i := range files {
		files[i] = fmt.Sprintf("/m/model-%02d.gguf", i)
	}
	saved := seam.Saved{}
	rows := make([]string, 30)
	for i := range rows {
		rows[i] = fmt.Sprintf(`{"id":"run-%02d","dir":"/r/run-%02d","saved_at":"2026-10-10T10:%02d:00","provider":"llama.cpp"}`, i, i, i)
	}
	if err := json.Unmarshal([]byte(`{"root":"/r","runs":[`+strings.Join(rows, ",")+`]}`), &saved); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		page int
		f    Facts
		last string
	}{
		{"model files", pageModel, Facts{Files: files, Path: files[0]}, "model-29.gguf"},
		{"saved runs", pageSaved, Facts{Saved: &saved}, "10:29"},
	} {
		acts := steps[c.page].actions(c.f)
		if len(acts) < 30 {
			t.Fatalf("%s: %d rows built, want all 30", c.name, len(acts))
		}
		top := DetailActions(c.f, c.page, 0, 100, 8)
		if !strings.Contains(top, "more") || strings.Contains(top, c.last) {
			t.Errorf("%s at the top: the window must say how many rows are below and not show the last row:\n%s", c.name, top)
		}
		bottom := DetailActions(c.f, c.page, len(acts)-1, 100, 8)
		if !strings.Contains(bottom, c.last) || !strings.Contains(bottom, "↑") {
			t.Errorf("%s at the bottom: the last row must be on screen with the count above it:\n%s", c.name, bottom)
		}
	}
}

// A list scrolls: PgDn/PgUp move the cursor a page, End/Home jump to the ends, and the mouse wheel moves one row,
// all through the same move routine the arrow keys use.
func TestListsScrollByPageEndsAndWheel(t *testing.T) {
	files := make([]string, 30)
	for i := range files {
		files[i] = fmt.Sprintf("/m/model-%02d.gguf", i)
	}
	var m tea.Model = Model{f: Facts{Files: files, Path: files[0]}, cursor: pageModel}
	row := func() int { return m.(Model).row }
	m, _ = m.Update(tea.KeyMsg{Type: tea.KeyPgDown})
	if row() != PageRows {
		t.Fatalf("PgDn: row %d, want %d", row(), PageRows)
	}
	m, _ = m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelDown, Action: tea.MouseActionPress})
	if row() != PageRows+1 {
		t.Fatalf("wheel down: row %d, want %d", row(), PageRows+1)
	}
	m, _ = m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelUp, Action: tea.MouseActionPress})
	m, _ = m.Update(tea.KeyMsg{Type: tea.KeyPgUp})
	if row() != 0 {
		t.Fatalf("wheel up then PgUp: row %d, want 0", row())
	}
	m, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnd})
	last := len(m.(Model).actions()) - 1
	if row() != last {
		t.Fatalf("End: row %d, want the last row %d", row(), last)
	}
	m, _ = m.Update(tea.KeyMsg{Type: tea.KeyHome})
	if row() != 0 {
		t.Fatalf("Home: row %d, want 0", row())
	}
}

// A picker shows ListRows rows at once, whatever the terminal height; the rest is reached by scrolling.
func TestPickersShowTenRowsOnATallScreen(t *testing.T) {
	files := make([]string, 30)
	for i := range files {
		files[i] = fmt.Sprintf("/m/model-%02d.gguf", i)
	}
	f := Facts{Files: files, Path: files[0]}
	view := DetailView(f, pageModel, 0, 100, 60, &viewport.Model{})
	if n := strings.Count(view, "/m/model-"); n > ListRows {
		t.Fatalf("%d model rows on a 60-line screen, want at most %d:\n%s", n, ListRows, view)
	}
	if !strings.Contains(view, "more") {
		t.Fatalf("the rows below the window must be counted:\n%s", view)
	}
}
