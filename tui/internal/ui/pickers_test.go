package ui

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

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
