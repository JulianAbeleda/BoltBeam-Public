package ui

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func TestRooflineTotalSurvivesLongNamesAtTerminalWidths(t *testing.T) {
	old := tableWidth
	defer func() { tableWidth = old }()
	for _, width := range []int{76, 106, 156} {
		tableWidth = width
		w := &seam.Where{
			Title: "Where the token goes", Words: "Signed differences add up to the modeled roofline.",
			Columns:       []string{"row", "ms now", "ms at limit", "ms at chip best", "ms above limit", "share", "+ Tok/S if roofline", "cumulative tok/s", "% of peak"},
			Rows:          []seam.WhereRow{{Section: "Model operations", Name: strings.Repeat("long_kernel_name_", 10), Cells: []string{"14.36", "10.43", "10.43", "3.93", "100%", "+26.7", "95.9", "72%"}}},
			BaselineCells: []string{"measured throughput", "", "", "", "", "", "69.2", "69.2", ""},
			SumCells:      []string{"= total at roofline", "14.36", "10.43", "10.43", "3.93", "100%", "95.9", "95.9", ""},
		}
		raw := whereBody(w)
		got := wrapText(raw, width)
		gridWidth := 0
		for _, line := range strings.Split(raw, "\n") {
			if strings.HasPrefix(line, "┌") {
				gridWidth = maxWidth(line)
			}
			if strings.HasPrefix(line, "│") && maxWidth(line) != gridWidth {
				t.Fatalf("uneven grid: %s", line)
			}
			if strings.HasPrefix(line, "│") && maxWidth(line) > width {
				t.Fatalf("table exceeds width %d: %s", width, line)
			}
			if strings.Contains(line, "+26.7") {
				for _, cell := range strings.Split(line, "│")[2:] {
					if strings.TrimSpace(cell) == "" {
						continue
					}
					left := len(cell) - len(strings.TrimLeft(cell, " "))
					right := len(cell) - len(strings.TrimRight(cell, " "))
					if left-right > 1 || right-left > 1 {
						t.Fatalf("numeric cell is not centered: %q", cell)
					}
				}
			}
		}
		total := ""
		for _, line := range strings.Split(got, "\n") {
			if strings.HasPrefix(line, "│ = total") {
				total = line
			}
		}
		if !strings.Contains(total, "95.9") {
			t.Fatalf("width %d clipped throughput: %s", width, got)
		}
		if !strings.Contains(got, "Model operations") {
			t.Fatalf("missing section: %s", got)
		}
		if !strings.Contains(got, "+26.7") || !strings.Contains(got, "69.2") {
			t.Fatalf("missing gain or measured baseline: %s", got)
		}
		if !strings.Contains(got, "95.9 tok/s") {
			t.Fatalf("missing readable footer: %s", got)
		}
		if !strings.Contains(got, "roofline.\n\n┌") {
			t.Fatalf("missing paragraph spacing: %s", got)
		}
	}
}

func TestWhereOrganizationIsSharedAcrossWorkloads(t *testing.T) {
	old := tableWidth
	defer func() { tableWidth = old }()
	for _, title := range []string{"Where the prefill time goes", "Where the turn goes at depth 512", "Captured roles: partial limits"} {
		for _, width := range []int{76, 106} {
			tableWidth = width
			w := &seam.Where{Title: title, Columns: []string{"row", "captured ms", "limit ms", "lost ms", "% of floor"},
				Rows: []seam.WhereRow{
					{Name: "projection", Section: "Model operations", Cells: []string{"3.0", "1.0", "2.0", "33.3"}},
					{Name: "helper", Section: "Supporting kernels — no model", Cells: []string{"1.0", "unavailable", "unavailable", "unavailable"}},
					{Name: "correction", Section: "Timing adjustments", Cells: []string{"-0.1", "0.0", "-0.1", ""}},
				}, SumCells: []string{"all", "3.9", "1.0 modeled", "unavailable", "unavailable"}}
			got := whereBody(w)
			for _, want := range []string{"┌", "│", "└", "Model operations", "Supporting kernels", "Timing adjustments", "unavailable"} {
				if !strings.Contains(got, want) {
					t.Fatalf("%s missing %q at %d: %s", title, want, width, got)
				}
			}
			for _, line := range strings.Split(got, "\n") {
				if strings.HasPrefix(line, "│") && maxWidth(line) > width {
					t.Fatalf("%s exceeds width %d: %s", title, width, line)
				}
			}
		}
	}
}

func TestUnavailableProjectionHasVisiblePlaceholder(t *testing.T) {
	old := tableWidth
	defer func() { tableWidth = old }()
	tableWidth = 106
	w := &seam.Where{Title: "Prefill", Columns: []string{"row", "ms now", "prefill ms if fixed"}, Rows: []seam.WhereRow{{Name: "projection", Cells: []string{"1.0", ""}}}, SumCells: []string{"all", "1.0", ""}}
	got := whereBody(w)
	for _, line := range strings.Split(got, "\n") {
		if strings.Contains(line, "projection") && !strings.Contains(line, "—") {
			t.Fatalf("unavailable value is blank: %s", line)
		}
	}
}

func TestAnOverCeilingCaptureSaysWhichRowsStandAboveTheTable(t *testing.T) {
	note := "Kernel rows qualified by replication: 3 captures agree, worst row 0.6% apart (attn_kv Q4_K) over 44 rows. Gaps shown as a bound: at most 0.31 ms (untraced token − kernels); the capture showed 1.78 ms, added by recording."
	var w seam.Where // as the results JSON carries it (tie_out.where_token_goes)
	raw := `{"title": "Where the token goes", "words": "Context 529 · 12.34 ms/token.", "qualification": "` + note + `", "columns": ["row", "ms now"],
		"rows": [{"name": "kernel gaps, at most", "cells": ["0.31"]}], "sum_cells": ["= total at roofline", "12.34"]}`
	if err := json.Unmarshal([]byte(raw), &w); err != nil {
		t.Fatal(err)
	}
	got := whereBody(&w)
	at, header := strings.Index(got, "Kernel rows qualified by replication"), strings.Index(got, "┌")
	if at < 0 || header < 0 || at > header || !strings.Contains(got, "at most 0.31 ms") {
		t.Fatalf("the qualification is not above the table:\n%s", got)
	}
}
