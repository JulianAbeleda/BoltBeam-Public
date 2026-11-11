package ui

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// offloadHint is what the Offload line says before Python's page is there.
const offloadHint = "what goes to the card and to RAM, and each setup's limits"

// offloadMsg is the answer of `screen offload` for one model, chip and engine.
type offloadMsg struct {
	key string
	off *seam.Offload
	err error
}

// loadOffload asks Python for the offload page once per model, chip and engine (the first call measures the GPU's
// copy rate, so it takes a couple of seconds).
func (m *Model) loadOffload() tea.Cmd {
	key := m.f.recommendKey()
	if key == "" || key == m.f.OffKey {
		return nil
	}
	m.f.OffKey, m.f.Offload, m.f.OffErr = key, nil, ""
	path, target, engine := m.f.Path, m.f.target().ID, m.f.Engine
	return func() tea.Msg {
		o, err := m.client.Offload(path, target, engine)
		return offloadMsg{key, o, err}
	}
}

// offloadValue is the Setup line's value: the hint until the page is there, then its headline cut to one line.
func offloadValue(f Facts) string {
	switch {
	case f.OffErr != "":
		return stWarn.Render("could not be made: " + f.OffErr)
	case f.Offload != nil:
		return f.Offload.Page.Headline
	case f.OffKey != "":
		return strings.TrimSpace(f.Spin + " working out the plan, a few seconds…")
	}
	return offloadHint
}

// offloadBody lays out Python's page: the headline, then each section's title, its rows (labels aligned, values
// wrapped under the value column) and its lines wrapped. No sentence is Go's.
func offloadBody(f Facts, width int) string {
	o := f.Offload
	switch {
	case f.OffErr != "":
		return stWarn.Render(wrapText("The offload page could not be made: "+f.OffErr, width))
	case o == nil && f.OffKey != "":
		return stMuted.Render(strings.TrimSpace(f.Spin + " working out the plan; the first one measures the card's copy rate…"))
	case o == nil:
		return stMuted.Render(wrapText(offloadHint+"\n\nChoose a model, a chip and an engine on Setup.", width))
	}
	var b strings.Builder
	b.WriteString(ansi.Wordwrap(o.Page.Headline, width, " ") + "\n")
	for _, s := range o.Page.Sections {
		b.WriteString("\n" + stHeader.Render(truncate(s.Title, width)) + "\n")
		lw := 0
		for _, r := range s.Rows {
			lw = max(lw, ansi.StringWidth(r.Label))
		}
		lw = min(lw, max(width/3, 8))
		vw := max(width-2-lw-2, 10)
		for _, r := range s.Rows {
			label := truncate(r.Label, lw)
			pad := strings.Repeat(" ", lw-ansi.StringWidth(label))
			for i, l := range strings.Split(ansi.Wrap(r.Value, vw, " "), "\n") {
				if i == 0 {
					b.WriteString("  " + stMuted.Render(label) + pad + "  " + l + "\n")
				} else {
					b.WriteString(strings.Repeat(" ", 2+lw+2) + l + "\n")
				}
			}
		}
		for _, l := range s.Lines {
			b.WriteString("  " + strings.ReplaceAll(ansi.Wrap(l, width-2, " "), "\n", "\n  ") + "\n")
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func offloadActions(Facts) []action {
	return []action{{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)}}
}
