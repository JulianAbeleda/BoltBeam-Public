package ui

// The Results screen in layers (docs/todo-release-0318.md, "Product flow"): layer 1 answers how fast, how close to
// what this GPU allows and what holds it back, then the three parts that lose the most time as bars with their gains
// (a batch sweep adds its curve), then the one trust line. Layer 2 is sections that open and close: the table, How it
// was measured, Machine and engine, the words used, and Full detail, the page as it always was. Every sentence and
// label is the seam's (`screen plain`, `screen words`); the screen only lays them out.

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"charm.land/lipgloss/v2/list"
	lgtable "charm.land/lipgloss/v2/table"
	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// Sections are layer 2's open sections; all closed until a key opens one. All shows every column of the table.
type Sections struct{ Table, All, Measured, Machine, Words, Full bool }

// The two section titles the plain block does not carry: the report's own (report/html.py "Words used here", "Full detail").
const (
	titleWords = "Words used here"
	titleFull  = "Full detail"
)

// plainShown says the run on screen is finished and its plain block is in: the screen is layer 1, then layer 2.
func (f Facts) plainShown() bool {
	return f.Plain != nil && f.Run != nil && f.Plain.ID == f.Run.ID && !f.alive() && f.failedStage() == "" && f.outputDone()
}

// plainBody is the Results screen of a finished run: layer 1, then layer 2's sections.
func plainBody(f Facts, width int) string {
	return layer1Card(f, width) + "\n\n" + layer2Body(f, width)
}

// layer1Card is layer 1 as one card: the title, the three answers, the curve of a sweep, the top three, the trust line.
// Save as image draws the same card.
func layer1Card(f Facts, width int) string {
	l := f.Plain.Layer1
	var b strings.Builder
	b.WriteString(paint(stHeader, hang(l.Title, 0, width)) + "\n\n")
	if fs := figures(l.Figures, width); fs != "" {
		b.WriteString(fs + "\n")
	}
	b.WriteString(answers(l, width))
	for _, u := range l.Upgrade {
		b.WriteString(paint(stMuted, hang(u, 0, width)) + "\n")
	}
	if c := sweepCurve(f, width); c != "" {
		b.WriteString("\n" + c)
	}
	if t := topBars(l, width); t != "" {
		b.WriteString("\n" + t)
	}
	style := stWarn
	if l.Trust.Checked {
		style = stOK
	}
	b.WriteString("\n" + paint(style, hang(l.Trust.Text, 0, width)))
	return b.String()
}

// stFigure sets a headline figure apart: bold, in the accent.
var stFigure = stAccent.Bold(true)

// figures is layer 1's headline figures, one a line: the figure bold in the accent, right-aligned in a column as wide
// as the widest, its short label muted beside it and wrapped under itself.
func figures(fs []seam.PlainFigure, width int) string {
	fw := 0
	for _, f := range fs {
		fw = max(fw, lipgloss.Width(f.Figure))
	}
	var b strings.Builder
	for _, f := range fs {
		pad := strings.Repeat(" ", 2+fw-lipgloss.Width(f.Figure))
		label := strings.TrimLeft(hang(f.Label, 2+fw+2, width), " ")
		b.WriteString(pad + stFigure.Render(f.Figure) + "  " + paint(stMuted, label) + "\n")
	}
	return b.String()
}

// answers is how fast, how close and what holds it back: each label from the words, its sentence wrapped beside it.
func answers(l seam.PlainLayer1, width int) string {
	type answer struct{ label, text string }
	rows := []answer{{vocab.Label("fast"), l.Fast}, {vocab.Label("close"), deref(l.Close)}, {vocab.Label("holds"), deref(l.Holds)}}
	rows = slices.DeleteFunc(rows, func(a answer) bool { return a.text == "" })
	lw := 0
	for _, a := range rows {
		lw = max(lw, lipgloss.Width(a.label))
	}
	if lw > width/3 { // too narrow for a column of labels: each label on its own line
		lw = 0
	}
	var b strings.Builder
	for _, a := range rows {
		if lw == 0 {
			b.WriteString(stMuted.Render(a.label) + "\n" + hang(a.text, 0, width) + "\n")
			continue
		}
		text := hang(a.text, lw+2, width)
		b.WriteString(stMuted.Render(a.label) + strings.Repeat(" ", lw+2-lipgloss.Width(a.label)) + strings.TrimLeft(text, " ") + "\n")
	}
	return b.String()
}

// hang wraps text to width; every line after the first starts indent cells in, and the first is indented too
// (answers trims it to sit after its label).
func hang(text string, indent, width int) string {
	lines := strings.Split(ansi.Wordwrap(text, max(width-indent, 10), " "), "\n")
	pad := strings.Repeat(" ", indent)
	for i, l := range lines {
		lines[i] = pad + strings.TrimRight(l, " ")
	}
	return strings.Join(lines, "\n")
}

// paint styles each line on its own, so a wrapped sentence is not padded into a block.
func paint(style lipgloss.Style, text string) string {
	lines := strings.Split(text, "\n")
	for i, l := range lines {
		lines[i] = style.Render(l)
	}
	return strings.Join(lines, "\n")
}

// topBars is the three parts that lose the most time, worst first: each a bar as long as its share of the lost time,
// with its gain under it; a table that allocates no gain says why once, under the three.
func topBars(l seam.PlainLayer1, width int) string {
	if len(l.Top) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString(stHeader.Render(vocab.Label("top")) + "\n")
	nameW := 0
	for _, p := range l.Top {
		nameW = max(nameW, lipgloss.Width(p.Part))
	}
	nameW = min(nameW, width*2/5)
	barW := min(max(width-nameW-2-2-4, 8), 30)
	for _, p := range l.Top {
		share := 0.0
		if p.Share != nil {
			share = *p.Share
		}
		name := truncate(p.Part, nameW)
		fmt.Fprintf(&b, "%s%s  %s  %4s\n", name, strings.Repeat(" ", nameW-lipgloss.Width(name)), bar(share, barW), p.ShareCell)
		if p.Gain.Text != nil {
			b.WriteString(gainLine(*p.Gain.Text, p.Gain.Figure, width) + "\n")
		}
	}
	for _, note := range []*string{l.GainsNote, l.Unassigned} {
		if note != nil {
			b.WriteString(paint(stMuted, hang(*note, 0, width)) + "\n")
		}
	}
	return b.String()
}

// gainLine is a part's gain under its bar: the figure ("+56.2 tok/s") bold, the words after it in the accent.
func gainLine(text, figure string, width int) string {
	lines := strings.Split(hang(text, 2, width), "\n")
	for i, l := range lines {
		if at := strings.Index(l, figure); figure != "" && i == 0 && at >= 0 {
			lines[i] = stAccent.Render(l[:at]) + stFigure.Render(figure) + stAccent.Render(l[at+len(figure):])
			continue
		}
		lines[i] = stAccent.Render(l)
	}
	return strings.Join(lines, "\n")
}

// sweepCurve is a batch sweep's total tokens a second per stream count, at the context of its best point, as columns
// one stream count wide, from the run's own batch rows. Nothing for a run that timed one stream count.
func sweepCurve(f Facts, width int) string {
	if f.Run == nil || f.Plain.Layer1.RunType != "batch_sweep" {
		return ""
	}
	rows := f.Run.Results.Batches
	if len(rows) == 0 {
		return ""
	}
	ctx := rows[0].Context
	if c := f.Plain.Layer1.Numbers["best_context"]; c != nil {
		ctx = *c
	}
	pts := slices.DeleteFunc(slices.Clone(rows), func(r seam.BatchRow) bool { return r.Context != ctx })
	slices.SortStableFunc(pts, func(a, b seam.BatchRow) int { return a.Batch - b.Batch })
	if len(pts) < 2 {
		return ""
	}
	top := 0.0
	counts, values := make([]string, len(pts)), make([]string, len(pts))
	colW := 1
	for i, p := range pts {
		top = max(top, p.TokSTotal)
		counts[i], values[i] = fmt.Sprint(p.Batch), fmt.Sprintf("%.0f", p.TokSTotal)
		colW = max(colW, len(counts[i]), len(values[i]))
	}
	// the row captions are the words' own: the glossary's "stream" and "context", the unit "tokens a second"
	stream, unit := vocab.Term("stream"), vocab.Units["tok_s"]
	capW := max(lipgloss.Width(stream), lipgloss.Width(unit)) + 2
	if capW+len(pts)*(colW+1) > width {
		capW = 0
	}
	caption := func(s string) string {
		if capW == 0 {
			return ""
		}
		return stMuted.Render(s) + strings.Repeat(" ", capW-lipgloss.Width(s))
	}
	const height = 4
	blocks := []rune(" ▁▂▃▄▅▆▇█")
	var b strings.Builder
	if term := vocab.Term("context"); term != "" {
		b.WriteString(stMuted.Render(fmt.Sprintf("%s %g", term, ctx)) + "\n")
	}
	for row := height - 1; row >= 0; row-- {
		line := caption("")
		for _, p := range pts {
			eighths := int(math.Round(p.TokSTotal / top * height * 8))
			fill := min(max(eighths-row*8, 0), 8)
			line += stAccent.Render(strings.Repeat(string(blocks[fill]), colW)) + " "
		}
		b.WriteString(strings.TrimRight(line, " ") + "\n")
	}
	cells := func(xs []string) string {
		out := []string{}
		for _, x := range xs {
			out = append(out, fmt.Sprintf("%*s", colW, x))
		}
		return strings.Join(out, " ")
	}
	b.WriteString(caption(stream) + cells(counts) + "\n")
	b.WriteString(caption(unit) + cells(values) + "\n")
	return b.String()
}

// toggle opens or closes the section the key names, and says which; ok is false for any other key. All opens the
// table with it.
func (s Sections) toggle(msg tea.KeyPressMsg) (Sections, key.Binding, bool) {
	switch {
	case key.Matches(msg, keyTable):
		s.Table = !s.Table
		return s, keyTable, true
	case key.Matches(msg, keyAllCols):
		s.All = !s.All
		s.Table = s.Table || s.All
		return s, keyAllCols, true
	case key.Matches(msg, keyMeasured):
		s.Measured = !s.Measured
		return s, keyMeasured, true
	case key.Matches(msg, keyMachine):
		s.Machine = !s.Machine
		return s, keyMachine, true
	case key.Matches(msg, keyGlossary):
		s.Words = !s.Words
		return s, keyGlossary, true
	case key.Matches(msg, keyFull):
		s.Full = !s.Full
		return s, keyFull, true
	}
	return s, key.Binding{}, false
}

// sectionTitle is the heading a section key opens: the plain block's own titles, the report's for the last two; ""
// when the run has no such section.
func sectionTitle(p *seam.Plain, k key.Binding) string {
	l2 := p.Layer2
	switch k.Help().Key {
	case keyTable.Help().Key, keyAllCols.Help().Key:
		if l2.Table != nil {
			return l2.Table.Title
		}
	case keyMeasured.Help().Key:
		if len(l2.Sections) > 0 {
			return l2.Sections[0].Title
		}
	case keyMachine.Help().Key:
		if len(l2.Sections) > 1 {
			return l2.Sections[1].Title
		}
	case keyGlossary.Help().Key:
		if len(l2.Glossary) > 0 {
			return titleWords
		}
	case keyFull.Help().Key:
		return titleFull
	}
	return ""
}

// sectionHead is a section's heading: ▸ closed or ▾ open, its title, and the key that opens it at the right.
func sectionHead(open bool, title string, k key.Binding, width int) string {
	m := "▸"
	if open {
		m = "▾"
	}
	return spread(stHeader.Render(m+" "+title)+"\t"+stMuted.Render(k.Help().Key), width)
}

// layer2Body is the sections, each a heading, open ones with their content under it.
func layer2Body(f Facts, width int) string {
	l2, open := f.Plain.Layer2, f.Open
	var b strings.Builder
	if t := l2.Table; t != nil {
		b.WriteString(sectionHead(open.Table, t.Title, keyTable, width) + "\n")
		if open.Table {
			b.WriteString(plainTable(t, open.All, width-2))
			if t.Note != nil {
				b.WriteString(paint(stMuted, hang(*t.Note, 2, width)) + "\n")
			}
			if all := vocab.Label("all_columns"); all != "" {
				mark := "▸"
				if open.All {
					mark = "▾"
				}
				b.WriteString("  " + spread(mark+" "+all+"\t"+stMuted.Render(keyAllCols.Help().Key), width-2) + "\n")
			}
			if w := t.AllColumns.Words; open.All && w != nil && *w != "" {
				b.WriteString(paint(stMuted, hang(strings.Join(strings.Fields(*w), " "), 2, width)) + "\n")
			}
		}
	}
	for i, s := range l2.Sections {
		k, on := keyMeasured, open.Measured
		if i == 1 {
			k, on = keyMachine, open.Machine
		} else if i > 1 {
			break
		}
		b.WriteString(sectionHead(on, s.Title, k, width) + "\n")
		if on {
			b.WriteString(bullets(s.Lines, width) + "\n")
		}
	}
	if len(l2.Glossary) > 0 {
		b.WriteString(sectionHead(open.Words, titleWords, keyGlossary, width) + "\n")
		if open.Words {
			terms := []string{}
			for _, g := range l2.Glossary {
				terms = append(terms, lipgloss.NewStyle().Bold(true).Render(g.Term)+": "+g.Plain)
			}
			b.WriteString(bullets(terms, width) + "\n")
		}
	}
	b.WriteString(sectionHead(open.Full, titleFull, keyFull, width))
	if open.Full {
		b.WriteString("\n" + fullDetail(f, width))
	}
	return strings.TrimRight(b.String(), "\n")
}

// bullets is a section's sentences as a Lip Gloss list, each wrapped under its own bullet.
func bullets(lines []string, width int) string {
	items := make([]any, len(lines))
	for i, l := range lines {
		items[i] = hang(l, 0, width-4)
	}
	out := strings.Split(list.New(items...).Enumerator(list.Bullet).EnumeratorStyle(stMuted.PaddingLeft(2).PaddingRight(1)).String(), "\n")
	for i, l := range out {
		out[i] = strings.TrimRight(l, " ") // the list pads a wrapped item into a block
	}
	return strings.Join(out, "\n")
}

// plainTable is the where table through Lip Gloss table: four columns (three when no gain is known), or every column
// with the measured baseline and the sum under the rows. A table wider than the page is fitted first (fitColumns).
func plainTable(t *seam.PlainTable, all bool, width int) string {
	cols, rows := t.Columns, [][]string{}
	for _, r := range t.Rows {
		rows = append(rows, r.Cells)
	}
	if all {
		cols, rows = t.AllColumns.Columns, slices.Clone(t.AllColumns.Rows)
		for _, extra := range [][]string{t.AllColumns.Baseline, t.AllColumns.Sum} {
			if len(extra) > 0 {
				rows = append(rows, extra)
			}
		}
	}
	gap := 2
	head, body, fits := fitColumns(cols, rows, width, gap)
	if !fits { // every column at its narrowest: one space between them
		gap = 1
		head, body, _ = fitColumns(cols, rows, width, gap)
	}
	cols, rows = head, body
	last, headLines := len(cols)-1, 1
	for _, h := range cols {
		headLines = max(headLines, strings.Count(h, "\n")+1)
	}
	// the header is the first row, not Headers: Lip Gloss cuts a header to one line, and a fitted one wraps
	tb := lgtable.New().Rows(append([][]string{cols}, rows...)...).
		Border(lipgloss.HiddenBorder()).BorderTop(false).BorderBottom(false).BorderLeft(false).BorderRight(false).
		BorderColumn(false).BorderHeader(false).
		StyleFunc(func(row, col int) lipgloss.Style {
			s := lipgloss.NewStyle()
			if col < last {
				s = s.PaddingRight(gap)
			}
			if col > 0 {
				s = s.Align(lipgloss.Right)
			}
			if row == 0 {
				s = s.Inherit(stHeader)
			}
			return s
		})
	lines := strings.Split(tb.String(), "\n")
	rule := 0
	for i, l := range lines {
		lines[i] = "  " + strings.TrimRight(l, " ")
		rule = max(rule, lipgloss.Width(lines[i])-2)
	}
	if len(lines) >= headLines {
		lines = slices.Insert(lines, headLines, "  "+stMuted.Render(strings.Repeat("─", rule)))
	}
	return strings.Join(lines, "\n") + "\n"
}

// fitColumns wraps a table's cells so it fits width: the column with the most room to give wraps one cell narrower
// at a time, never under its longest word (a number never breaks; the first column's part names go down to 12).
// fits is false when even that is too wide.
func fitColumns(cols []string, rows [][]string, width, gap int) (head []string, body [][]string, fits bool) {
	n := len(cols)
	w, floor := make([]int, n), make([]int, n)
	for c, h := range cols {
		w[c], floor[c] = lipgloss.Width(h), longestWord(h)
	}
	for _, r := range rows {
		for c := 0; c < n && c < len(r); c++ {
			w[c] = max(w[c], lipgloss.Width(r[c]))
			if c > 0 {
				floor[c] = max(floor[c], longestWord(r[c]))
			}
		}
	}
	floor[0] = min(max(floor[0], 12), w[0])
	total := func() int {
		s := gap * (n - 1)
		for _, x := range w {
			s += x
		}
		return s
	}
	for total() > width {
		widest := -1
		for c := range w {
			if w[c] > floor[c] && (widest < 0 || w[c]-floor[c] > w[widest]-floor[widest]) {
				widest = c
			}
		}
		if widest < 0 {
			break // as narrow as it goes: the page cuts the rest
		}
		w[widest]--
	}
	wrap := func(s string, c int) string { return ansi.Wrap(s, w[c], " ") } // a kernel name longer than its column breaks
	head = make([]string, n)
	for c, h := range cols {
		head[c] = wrap(h, c)
	}
	body = make([][]string, len(rows))
	for i, r := range rows {
		body[i] = make([]string, len(r))
		for c, cell := range r {
			if c < n {
				cell = wrap(cell, c)
			}
			body[i][c] = cell
		}
	}
	return head, body, total() <= width
}

// longestWord is the widest word of a cell: it is never broken.
func longestWord(s string) int {
	out := 0
	for _, f := range strings.Fields(s) {
		out = max(out, lipgloss.Width(f))
	}
	return out
}

// fullDetail is the page as it always was: every table and line of the run, in the record's own terms.
func fullDetail(f Facts, width int) string {
	var b strings.Builder
	b.WriteString(timingScopeLine(f.Run.Results.Loss.TimingScope, width))
	if f.Run.Measure != nil && f.Run.Measure.Status != "measured" {
		b.WriteString(measureBody(f, width) + "\n\n")
	}
	if f.Run.Results.Loss.Layout != nil {
		b.WriteString(layoutBody(*f.Run.Results.Loss.Layout) + "\n")
	}
	b.WriteString(resultBody(f, width))
	return b.String()
}
