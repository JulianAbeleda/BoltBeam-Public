package ui

// Where the time went: a prefill's time budget (seam/budget.go) on the Results screen, a section of layer 2, and on the
// compare screen both runs side by side. The rows are Python's and add up to the wall as the record states it; the
// record's own caveat (the trace's GPU work longer than the wall) leads, and is shown even with the section closed.

import (
	"strings"

	"charm.land/lipgloss/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// budgetWarn is the record's caveat in the warning colour, or "".
func budgetWarn(warn *string, width int) string {
	if warn == nil || *warn == "" {
		return ""
	}
	return paint(stWarn, hangHard(glyphWarn+" "+*warn, 2, width)) + "\n"
}

// budgetRows lays out label and figure rows: each part at 2, the rows under it at 4, the figures right-aligned in one
// column per run after it; the wall's row has a rule above it. A label that does not fit wraps under itself.
func budgetRows(rows [][]string, depth []int, parts []bool, width int) string {
	cols := 0
	for _, r := range rows {
		cols = max(cols, len(r)-1)
	}
	cw := make([]int, cols)
	for _, r := range rows {
		for i, c := range r[1:] {
			cw[i] = max(cw[i], lipgloss.Width(c))
		}
	}
	room := width
	for _, w := range cw {
		room -= w + 2
	}
	var b strings.Builder
	for i, r := range rows {
		indent := 2 + 2*depth[i]
		lw := max(room-indent, 12)
		if i == len(rows)-1 {
			b.WriteString("  " + stMuted.Render(strings.Repeat("─", max(width-2, 10))) + "\n")
		}
		label := strings.Split(wrapLabel(r[0], lw), "\n")
		for j, l := range label {
			line := strings.Repeat(" ", indent) + l
			if j == len(label)-1 {
				line += strings.Repeat(" ", max(indent+lw-lipgloss.Width(line), 0))
				for k, c := range r[1:] {
					line += "  " + strings.Repeat(" ", cw[k]-lipgloss.Width(c)) + c
				}
			}
			if parts[i] {
				line = stHeader.Render(strings.TrimRight(line, " "))
			} else {
				line = strings.TrimRight(line, " ")
			}
			b.WriteString(line + "\n")
		}
	}
	return b.String()
}

// wrapLabel wraps a label to width without breaking a word that fits.
func wrapLabel(s string, width int) string { return strings.TrimRight(hangHard(s, 0, width), " ") }

// budgetBody is one run's budget: the caveat, what it is of, the rows with their unit, then its sentences.
func budgetBody(tb *seam.TimeBudget, width int) string {
	var b strings.Builder
	b.WriteString(budgetWarn(tb.Warn, width))
	if tb.Refused != nil {
		return b.String() + paint(stMuted, hangHard(*tb.Refused, 2, width)) + "\n"
	}
	if tb.Of != nil {
		b.WriteString(paint(stMuted, hangHard(*tb.Of+" ("+tb.Unit+")", 2, width)) + "\n")
	}
	rows, depth, parts := [][]string{}, []int{}, []bool{}
	for _, r := range tb.Rows {
		rows, depth, parts = append(rows, []string{r.Label, r.Ms}), append(depth, r.Depth), append(parts, r.Depth == 0)
	}
	b.WriteString(budgetRows(rows, depth, parts, width))
	for _, l := range tb.Lines {
		b.WriteString(paint(stMuted, hangHard(l, 2, width)) + "\n")
	}
	return b.String()
}

// budgetSides is two runs' budgets side by side, A and B as the head to head names them, then each run's caveat.
func budgetSides(s *seam.TimeBudgetSides, width int) string {
	var b strings.Builder
	b.WriteString(stHeader.Render(s.Title) + " " + stMuted.Render("("+s.Unit+")") + "\n")
	rows, depth, parts := [][]string{append([]string{""}, s.Columns[1:]...)}, []int{0}, []bool{true}
	for _, r := range s.Rows {
		rows, depth, parts = append(rows, append([]string{r.Label}, r.Cells...)), append(depth, r.Depth), append(parts, r.Depth == 0)
	}
	b.WriteString(budgetRows(rows, depth, parts, width))
	for _, n := range s.Notes {
		b.WriteString(paint(stWarn, hangHard(glyphWarn+" "+n, 2, width)) + "\n")
	}
	return b.String()
}
