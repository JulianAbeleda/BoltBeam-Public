package ui

// The tables the result pages draw: per role, where the token goes, rates and batches.

import (
	"fmt"
	"github.com/charmbracelet/x/ansi"
	"slices"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func roleRows(roles []seam.CeilingRole) string {
	rows := [][]string{{"ROLE", "QUANT", "SHARE", "FLOOR ms"}}
	for _, r := range roles {
		rows = append(rows, []string{named(plainRole, r.Role), r.Quant, bar(r.Share, 10) + fmt.Sprintf(" %2.0f%%", r.Share*100),
			fmt.Sprintf("%.2f", r.FloorMs)})
	}
	return table(rows)
}

// roleTable is one provider's per-role table: the time, the share of peak, the time per call and why. The reason
// word is the most useful column, so when the page is narrow the share bar goes first, then IDEAL; only when the
// table still does not fit do the time per call and the reason move to a second table, so no row is cut.
func roleTable(roles []seam.RoleLoss, notAttributed *float64) string {
	for _, drop := range [][]string{nil, {"SHARE OF LOSS"}, {"SHARE OF LOSS", "IDEAL ms"}} {
		if t := roleTableWithout(roles, notAttributed, drop); maxWidth(t) <= tableWidth {
			return t
		}
	}
	return roleTableNarrow(roles, notAttributed)
}

// roleTableWithout is the full table without the named columns.
func roleTableWithout(roles []seam.RoleLoss, notAttributed *float64, drop []string) string {
	rows := [][]string{{"ROLE", "QUANT", "IDEAL ms", "ACTUAL ms", "LOST ms", "SHARE OF LOSS", "% PEAK", "µs/CALL", "WHY", "BEST FOUND", "VERDICT"}}
	for _, r := range roles {
		pct, us := "", ""
		if r.PctPeak != nil {
			pct = fmt.Sprintf("%.1f%%", *r.PctPeak)
		}
		if r.UsPerCall != nil {
			us = fmt.Sprintf("%.1f", *r.UsPerCall)
		}
		best, verdict := found(r)
		rows = append(rows, []string{word(plainRole, r.Role), r.Quant, fmt.Sprintf("%.2f", r.IdealMs),
			fmt.Sprintf("%.2f", r.ActualMs), fmt.Sprintf("%.2f", r.LostMs), bar(r.Share, 5) + fmt.Sprintf(" %.0f%%", r.Share*100), pct, us, r.Reason, best, verdict})
	}
	if notAttributed != nil {
		rows = append(rows, []string{"not attributed", "", "", fmt.Sprintf("%.2f", *notAttributed), "", "", "", "", "", "", ""})
	}
	if !searched(roles) {
		drop = append(drop, "BEST FOUND", "VERDICT")
	}
	keep := []int{}
	for i, h := range rows[0] {
		if !slices.Contains(drop, h) {
			keep = append(keep, i)
		}
	}
	for j, row := range rows {
		cut := []string{}
		for _, i := range keep {
			cut = append(cut, row[i])
		}
		rows[j] = cut
	}
	return table(rows)
}

func roleTableNarrow(roles []seam.RoleLoss, notAttributed *float64) string {
	main := [][]string{{"ROLE", "QUANT", "IDEAL", "ACTUAL", "LOST", "% PEAK"}}
	why := [][]string{{"ROLE", "QUANT", "µs/CALL", "WHY"}}
	search := [][]string{{"ROLE", "QUANT", "BEST FOUND", "VERDICT"}}
	for _, r := range roles {
		pct, us := "", ""
		if r.PctPeak != nil {
			pct = fmt.Sprintf("%.1f%%", *r.PctPeak)
		}
		if r.UsPerCall != nil {
			us = fmt.Sprintf("%.1f", *r.UsPerCall)
		}
		name := word(plainRole, r.Role)
		main = append(main, []string{name, r.Quant, fmt.Sprintf("%.2f", r.IdealMs), fmt.Sprintf("%.2f", r.ActualMs),
			fmt.Sprintf("%.2f", r.LostMs), pct})
		why = append(why, []string{name, r.Quant, us, r.Reason})
		best, verdict := found(r)
		search = append(search, []string{name, r.Quant, best, verdict})
	}
	if notAttributed != nil {
		main = append(main, []string{"not attributed", "", "", fmt.Sprintf("%.2f", *notAttributed), "", ""})
	}
	out := table(main) + "\n" + table(why)
	if searched(roles) {
		out += "\n" + table(search)
	}
	return out
}

// replayTable is the per-turn table. When it is wider than the page, the FLAGS column leaves the table and becomes
// one line per flag naming its turns, so no flag is cut off at the page edge.
func replayTable(rows [][]string) string {
	t := table(rows)
	last := len(rows[0]) - 1
	if maxWidth(t) <= tableWidth || last < 1 || rows[0][last] != "FLAGS" {
		return t
	}
	turns, order := map[string][]string{}, []string{}
	cut := make([][]string, len(rows))
	for i, row := range rows {
		cut[i] = row[:min(len(row), last)]
		if i == 0 || len(row) <= last {
			continue
		}
		for _, flag := range strings.Fields(row[last]) {
			if _, ok := turns[flag]; !ok {
				order = append(order, flag)
			}
			turns[flag] = append(turns[flag], row[0])
		}
	}
	var b strings.Builder
	b.WriteString(table(cut))
	for _, flag := range order {
		b.WriteString(stMuted.Render(fmt.Sprintf("%s: %d turns (%s)", flag, len(turns[flag]), strings.Join(turns[flag], ", "))) + "\n")
	}
	return b.String()
}

func whereTable(w *seam.Where, drop []string) string {
	head := []string{}
	for _, c := range w.Columns {
		head = append(head, strings.ToUpper(c))
	}
	rows := [][]string{head}
	section := ""
	for _, r := range w.Rows {
		if r.Section != "" && r.Section != section {
			rows = append(rows, []string{r.Section})
			section = r.Section
		}
		cells := append([]string{r.Name}, r.Cells...)
		for len(cells) < len(head) {
			cells = append(cells, "")
		}
		rows = append(rows, cells)
	}
	if len(w.BaselineCells) > 0 {
		rows = append(rows, w.BaselineCells)
	}
	rows = append(rows, w.SumCells)
	keep := []int{}
	for i, h := range rows[0] {
		if !slices.Contains(drop, h) {
			keep = append(keep, i)
		}
	}
	for j, row := range rows {
		cut := []string{}
		for _, i := range keep {
			if i < len(row) {
				cut = append(cut, row[i])
			}
		}
		rows[j] = cut
	}
	return rooflineGrid(rows, len(w.BaselineCells) > 0)
}

// rooflineGrid reserves numeric columns, wraps headers and centers values inside a bordered table.
func rooflineGrid(rows [][]string, baseline bool) string {
	if len(rows) == 0 {
		return ""
	}
	widths := make([]int, len(rows[0]))
	headers := make([][]string, len(widths))
	for col, heading := range rows[0] {
		if col == 0 {
			headers[col] = []string{heading}
			continue
		}
		headers[col] = strings.Split(ansi.Wordwrap(heading, 9, " "), "\n")
		for _, line := range headers[col] {
			widths[col] = max(widths[col], lipgloss.Width(line))
		}
		for _, row := range rows[1:] {
			if col < len(row) {
				widths[col] = max(widths[col], lipgloss.Width(row[col]))
			}
		}
	}
	other := 3*len(widths) + 1
	for _, width := range widths[1:] {
		other += width
	}
	widths[0] = max(tableWidth-other, 14)
	// Do not fill a wide terminal with needless name padding.
	nameWidth := 0
	for _, row := range rows {
		if len(row) > 0 {
			nameWidth = max(nameWidth, lipgloss.Width(row[0]))
		}
	}
	widths[0] = min(widths[0], max(nameWidth, 14))
	var b strings.Builder
	rule := func(left, joint, right string) {
		b.WriteString(left)
		for col, width := range widths {
			if col > 0 {
				b.WriteString(joint)
			}
			b.WriteString(strings.Repeat("─", width+2))
		}
		b.WriteString(right + "\n")
	}
	line := func(cells []string, missing bool) {
		b.WriteString("│")
		for col, width := range widths {
			cell := ""
			if col < len(cells) {
				cell = truncate(cells[col], width)
			}
			if col > 0 && cell == "" && missing {
				cell = "—"
			}
			room := width - lipgloss.Width(cell)
			left := room / 2
			if col == 0 {
				left = 0
			}
			b.WriteString(" " + strings.Repeat(" ", left) + cell + strings.Repeat(" ", room-left) + " │")
		}
		b.WriteString("\n")
	}
	rule("┌", "┬", "┐")
	height := 1
	for _, lines := range headers {
		height = max(height, len(lines))
	}
	for index := 0; index < height; index++ {
		cells := make([]string, len(widths))
		for col, lines := range headers {
			if index < len(lines) {
				cells[col] = lines[index]
			}
		}
		line(cells, false)
	}
	rule("├", "┼", "┤")
	for index, row := range rows[1:] {
		actual := index + 1
		if actual == len(rows)-1 || (baseline && actual == len(rows)-2) {
			rule("├", "┼", "┤")
		}
		if len(row) == 1 {
			if actual > 1 {
				rule("├", "┼", "┤")
			}
			room := 3*len(widths) - 1
			for _, width := range widths {
				room += width
			}
			label := truncate(row[0], room-2)
			b.WriteString("│ " + label + strings.Repeat(" ", room-1-lipgloss.Width(label)) + "│\n")
			rule("├", "┼", "┤")
		} else {
			line(row, !(baseline && actual == len(rows)-2))
		}
	}
	rule("└", "┴", "┘")
	return b.String()
}

// isoRoleTable is an isolated run's per-role table: what was measured alone (µs per call, GB/s, the share of
// peak) and the estimated split of the token. When the rows carry a dispatch floor (lessFloor) the measured µs
// column is labelled so and the time less the floor stands beside it; GB/s and % PEAK are on that time, as the
// seam's columns_words says once. At a narrow page BYTES/CALL goes first, then IDEAL.
func isoRoleTable(roles []seam.RoleLoss, lessFloor bool) string {
	head := []string{"ROLE", "QUANT", "BYTES/CALL", "µs/CALL", "GB/s", "% PEAK", "IDEAL", "EST. ms IN TOKEN", "EST. LOST", "WHY"}
	if lessFloor {
		head = []string{"ROLE", "QUANT", "BYTES/CALL", "µs/CALL MEASURED", "LESS FLOOR", "GB/s", "% PEAK", "IDEAL", "EST. ms IN TOKEN", "EST. LOST", "WHY"}
	}
	f := func(v *float64, format string) string {
		if v == nil {
			return ""
		}
		return fmt.Sprintf(format, *v)
	}
	all := [][]string{head}
	for _, r := range roles {
		row := []string{word(plainRole, r.Role), r.Quant, f(r.MbPerCall, "%.1f MB"), f(r.UsPerCall, "%.1f")}
		if lessFloor {
			row = append(row, f(r.UsPerCallLessFloor, "%.1f"))
		}
		all = append(all, append(row, f(r.Gbs, "%.1f"), f(r.PctPeak, "%.1f%%"), fmt.Sprintf("%.2f", r.IdealMs), f(r.EstMs, "%.2f"),
			f(r.EstLostMs, "%.2f"), r.Reason))
	}
	only := func(keep ...string) string {
		rows := [][]string{}
		for _, row := range all {
			cut := []string{}
			for i, h := range head {
				if slices.Contains(keep, h) {
					cut = append(cut, row[i])
				}
			}
			rows = append(rows, cut)
		}
		return table(rows)
	}
	without := func(drop ...string) []string {
		return slices.DeleteFunc(slices.Clone(head), func(h string) bool { return slices.Contains(drop, h) })
	}
	for _, drop := range [][]string{nil, {"BYTES/CALL"}, {"BYTES/CALL", "IDEAL"}} {
		if t := only(without(drop...)...); maxWidth(t) <= tableWidth {
			return t
		}
	}
	// still too wide: the estimate in one table, what was measured alone and why in the next, so no row is cut.
	// With a floor the two µs columns take GB/s's place: % PEAK in the first table already carries the rate.
	measured := []string{"ROLE", "QUANT", "µs/CALL", "GB/s", "WHY"}
	if lessFloor {
		head[3] = "µs/CALL"
		measured = []string{"ROLE", "QUANT", "µs/CALL", "LESS FLOOR", "WHY"}
	}
	return only("ROLE", "QUANT", "% PEAK", "EST. ms IN TOKEN", "EST. LOST") + "\n" + only(measured...)
}

// rateTable is a labelled row group of read rates per role: µs per call, GB/s and the share of peak. Rows that
// carry a time less the dispatch floor get that column beside the measured one; GB/s and OF PEAK are then on it.
func rateTable(rows []seam.RateRow) string {
	lessFloor := slices.ContainsFunc(rows, func(r seam.RateRow) bool { return r.UsPerCallLessFloor != nil })
	t := [][]string{{"ROLE", "QUANT", "µs/CALL", "GB/s", "OF PEAK"}}
	if lessFloor {
		t = [][]string{{"ROLE", "QUANT", "µs/CALL MEASURED", "LESS FLOOR", "GB/s", "OF PEAK"}}
	}
	for _, r := range rows {
		pct := num(r.PctPeak)
		if r.PctPeak != nil {
			pct = fmt.Sprintf("%.0f%%", *r.PctPeak)
		}
		row := []string{word(plainRole, r.Role), r.Quant, us(r.UsPerCall)}
		if lessFloor {
			row = append(row, us(r.UsPerCallLessFloor))
		}
		t = append(t, append(row, num(r.Gbs), pct))
	}
	return table(t)
}

// batchTable is step 4's points beside their own limit, when more than batch 1 was timed: tokens per second per
// stream and in total, the limit's total, and the share of it reached.
func batchTable(rows []seam.BatchRow) string {
	wide := false
	for _, r := range rows {
		wide = wide || r.Batch > 1
	}
	if !wide {
		return ""
	}
	t := [][]string{{"BATCH", "CONTEXT", "PER STREAM", "TOTAL tok/s", "LIMIT tok/s", "OF LIMIT", "BOUND BY"}}
	for _, r := range rows {
		pct := ""
		limit := "—"
		if r.PctOfLimit != nil {
			pct = fmt.Sprintf("%.0f%%", *r.PctOfLimit)
			limit = fmt.Sprintf("%.1f", r.Limit.TokSTotal)
		}
		t = append(t, []string{fmt.Sprint(r.Batch), fmt.Sprintf("%.0f", r.Context), fmt.Sprintf("%.1f", r.TokSStream),
			fmt.Sprintf("%.1f", r.TokSTotal), limit, pct, word(plainLimit, r.Limit.Bound)})
	}
	return "\n" + table(t) + "\n"
}
