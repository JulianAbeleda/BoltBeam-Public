package ui

// Small text helpers every view uses: numbers, units, plain words, widths and wrapping.

import (
	"fmt"
	"github.com/charmbracelet/x/ansi"
	"strings"
	"time"

	"charm.land/lipgloss/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// found is a role's best kernel found and its verdict, as two cells.
func found(r seam.RoleLoss) (string, string) {
	best, verdict := "", ""
	if r.BestFound != nil {
		best = r.BestFound.Text
	}
	if r.Verdict != nil {
		verdict = word(plainVerdict, *r.Verdict)
	}
	return best, verdict
}

// vocab is the plain vocabulary (`screen words`, data/plain_words.json), read once when the screen starts: stage, role
// and part names, layer 1's labels, the glossary. Until it answers, a stage or role shows its own key. Like
// darkBackground, there is one per program.
var vocab *seam.Words

// stageWord is the plain name of any pipeline stage key, the report's and the measuring ones.
func stageWord(key string) string { return vocab.Stage(key, here()) }

// roleWord is the plain name of a role key.
func roleWord(key string) string { return vocab.Role(key) }

// namedRole is the role's plain name with its key beside it, muted: the detail views' wording.
func namedRole(key string) string {
	if plain := vocab.Role(key); plain != key {
		return plain + "  " + stMuted.Render(key)
	}
	return key
}

// word is the plain word for a record key, or the key itself when no plain word exists.
func word(table map[string]string, key string) string {
	if plain, ok := table[key]; ok {
		return plain
	}
	return key
}

// named is the plain word with the record's own name beside it, muted: the detail views' wording.
func named(table map[string]string, key string) string {
	if plain, ok := table[key]; ok {
		return plain + "  " + stMuted.Render(key)
	}
	return key
}

func num(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.4g", *v)
}

func gb(bytes float64) string { return fmt.Sprintf("%.2f GB", bytes/1e9) }

// us prints microseconds as a whole number; kernels are never sub-microsecond on a GPU timeline.
func us(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.0f", *v)
}

func count(v *int) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%d", *v)
}

func deref(s *string) string {
	if s == nil {
		return "-"
	}
	return *s
}

// table lays out rows two spaces apart; the first row is the header.
func table(rows [][]string) string {
	widths := []int{}
	for _, row := range rows {
		for i, cell := range row {
			if i >= len(widths) {
				widths = append(widths, 0)
			}
			widths[i] = max(widths[i], lipgloss.Width(cell))
		}
	}
	var b strings.Builder
	for r, row := range rows {
		for i, cell := range row {
			if r == 0 {
				cell = stHeader.Render(cell)
			}
			b.WriteString(cell)
			if i < len(row)-1 {
				b.WriteString(strings.Repeat(" ", widths[i]-lipgloss.Width(cell)) + "  ")
			}
		}
		b.WriteString("\n")
	}
	return b.String()
}

func pctBar(v *float64, width int) string {
	if v == nil {
		return strings.Repeat(" ", width+5)
	}
	return bar(*v/100, width) + fmt.Sprintf(" %3.0f%%", *v)
}

// --- the five bodies -----------------------------------------------------------------------------------------

func maxWidth(s string) int {
	w := 0
	for _, l := range strings.Split(s, "\n") {
		w = max(w, lipgloss.Width(l))
	}
	return w
}

// spread puts a row's right part (after a tab) at the right edge, so the "›" marks line up.
func spread(label string, width int) string {
	left, right, ok := strings.Cut(label, "\t")
	if !ok {
		return label
	}
	left = truncate(left, width-lipgloss.Width(right)-2) // a long value is cut; the mark stays
	gap := max(width-lipgloss.Width(right)-lipgloss.Width(left), 2)
	return left + strings.Repeat(" ", gap) + right
}

// listWindow is the DetailActions budget for a scrolling list: ListRows plus the two count lines, never more than
// the room on screen.
func listWindow(room int) int { return min(max(room, 6), ListRows+2) }

// wrapText wraps sentences to the width; a table row (cells apart by two spaces) is cut instead, so columns stay.
func wrapText(text string, width int) string {
	lines := strings.Split(text, "\n")
	for i, l := range lines {
		if strings.Contains(strings.TrimLeft(ansi.Strip(l), " "), "  ") {
			lines[i] = truncate(l, width)
		} else {
			lines[i] = ansi.Wordwrap(l, width, " ")
		}
	}
	return strings.Join(lines, "\n")
}

// capFirst starts a sentence with a phrase like "this machine".
func capFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// --- the running screen ----------------------------------------------------------------------------------------

// elapsed is the time since the job started, as mm:ss (h:mm:ss past an hour).
func elapsed(j *jobs.Job) string {
	if j == nil {
		return ""
	}
	t, err := time.Parse(time.RFC3339, j.StartedAt)
	if err != nil {
		return ""
	}
	d := max(clock().Sub(t), 0)
	h, m, s := int(d.Hours()), int(d.Minutes())%60, int(d.Seconds())%60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, m, s)
	}
	return fmt.Sprintf("%02d:%02d", m, s)
}

// fit is each line cut to the width: one log line, one screen line (a style pads a block to its widest line).
func fit(lines []string, width int) []string {
	out := make([]string, len(lines))
	for i, l := range lines {
		out[i] = truncate(l, width)
	}
	return out
}
