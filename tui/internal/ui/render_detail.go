package ui

// The detail view a row opens, and its actions.

import (
	"fmt"
	"github.com/charmbracelet/bubbles/viewport"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

func title(i int, f Facts) string {
	t := steps[i].title
	if about := steps[i].about(f); about != "" {
		t += " · " + about
	}
	return t
}

// DetailActions is the bottom of a screen: the page's line, then its rows. Headings are drawn as headings; when
// the rows do not fit maxRows, a window around the cursor is shown with what is above and below counted.
func DetailActions(f Facts, i, row, width, maxRows int) string {
	var b strings.Builder
	_, text := steps[i].line(f)
	b.WriteString(text)
	acts := []action{}
	if steps[i].actions != nil {
		acts = steps[i].actions(f)
	}
	start, end := 0, len(acts)
	if maxRows > 2 && len(acts) > maxRows {
		start = max(0, min(row-maxRows/2, len(acts)-maxRows+2))
		end = min(len(acts), start+maxRows-2)
		if start > 0 {
			b.WriteString("\n" + stMuted.Render(fmt.Sprintf("  ↑ %d more", start)))
		}
	}
	for j := start; j < end; j++ {
		a := acts[j]
		switch {
		case a.do == "head" && a.label == "":
			b.WriteString("\n")
		case a.do == "head" && j > start:
			b.WriteString("\n\n" + stHeader.Render(a.label)) // a blank line sets each section apart
		case a.do == "head":
			b.WriteString("\n" + stHeader.Render(a.label))
		case a.do == "note":
			b.WriteString("\n  " + a.label)
		case j == row:
			b.WriteString("\n" + stCursor.Render("▸ ") + spread(a.label, width-6))
		default:
			b.WriteString("\n  " + spread(a.label, width-6))
		}
	}
	if end < len(acts) {
		b.WriteString("\n" + stMuted.Render(fmt.Sprintf("  ↓ %d more", len(acts)-end)))
	}
	title := "What next"
	switch {
	case i == pageSetup:
		title = "Setup"
	case i == pageSaved:
		title = "Saved runs"
	case picker(i):
		title = "Choose"
	}
	body := b.String()
	if text == "" { // no line: the first row starts the box
		body = strings.TrimPrefix(body, "\n")
	}
	return box(title, body, width, false)
}

// DetailView is the open view at a given height: the step's whole result scrolls on top, actions at the bottom.
func DetailView(f Facts, i, row, width, height int, scroll *viewport.Model) string {
	if i == pageSetup { // Setup is its rows only: one box, padded to the screen
		actions := DetailActions(f, i, row, width, height-2)
		return actions + strings.Repeat("\n", max(height-lipgloss.Height(actions), 0))
	}
	if i == pageSaved { // the list is the page: the rows scroll in the box on top, the hint sits below
		hint := box("What next", steps[i].body(f, width-4), width, false)
		actions := DetailActions(f, i, row, width, listWindow(height-lipgloss.Height(hint)-3))
		return actions + strings.Repeat("\n", max(height-lipgloss.Height(actions)-lipgloss.Height(hint), 0)) + "\n" + hint
	}
	rows := max(height*2/3-3, 6) // the rows take up to two thirds of the screen
	if picker(i) {
		rows = listWindow(rows) // a picker is a scrolling list of ListRows
	}
	actions := DetailActions(f, i, row, width, rows)
	scroll.Width, scroll.Height = width-4, max(height-lipgloss.Height(actions)-3, 1)
	tableWidth = width - 4
	f.logRoom = scroll.Height - 6 // the log fits under the bar, so the bar never scrolls away
	scroll.SetContent(wrapText(steps[i].body(f, width-4), width-4))
	return box(title(i, f), scroll.View(), width, false) + "\n" + actions
}
