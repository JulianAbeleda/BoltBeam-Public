package ui

// Check this machine: `boltbeam doctor --json` through the seam, Setup's first line and its page: a verdict, then the
// checks grouped by what they are about. It never blocks Run; a check not found says what will not work here, and
// its one fix.

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// doctorMsg is the answer of the machine check: its rows, or why it could not run.
type doctorMsg struct {
	doctor *seam.Doctor
	err    error
}

// checkMachine asks Python what this machine has (boltbeam doctor). It follows the engine scan, so the two never
// write the saved scan at once, and runs again after Scan again, since its engine rows read that scan.
func (m *Model) checkMachine() tea.Cmd {
	if m.f.DoctorBusy {
		return nil
	}
	m.f.DoctorBusy = true
	client := m.client
	return func() tea.Msg { d, err := client.Doctor(); return doctorMsg{d, err} }
}

// gaps are the checks not found, by name: the blocking ones (no run gets past them; Python says which), or the rest.
func (f Facts) gaps(blocking bool) []string {
	out := []string{}
	if f.Doctor != nil {
		for _, r := range f.Doctor.Rows {
			if r.Found != nil && !*r.Found && r.Blocks == blocking {
				out = append(out, r.Item)
			}
		}
	}
	return out
}

// thingsWord is "1 thing" or "n things".
func thingsWord(n int) string {
	if n == 1 {
		return "1 thing"
	}
	return fmt.Sprintf("%d things", n)
}

// verdict is the machine check in one line: ready to run here, or what to set up first. A gap that costs one engine
// or one per-role split is no blocker: it is counted, muted, after "ready".
func verdict(f Facts) string {
	block, other := f.gaps(true), f.gaps(false)
	if len(block) > 0 {
		return stWarn.Render(thingsWord(len(block))+" to set up before a run") + ": " + strings.Join(block, ", ")
	}
	text := "ready to run on " + here()
	if len(other) > 0 {
		text += stMuted.Render(fmt.Sprintf(" · %d optional", len(other)))
	}
	return text
}

// doctorValue is Setup's Machine line: the verdict, or that the check is running or could not run.
func doctorValue(f Facts) string {
	switch {
	case f.DoctorErr != "":
		return stWarn.Render("not checked: " + f.DoctorErr)
	case f.Doctor == nil:
		return strings.TrimSpace(f.Spin + " checking " + here() + "…")
	}
	return verdict(f)
}

// doctorMark is a check's glyph: found, not found, or n/a (it never runs on this GPU type).
func doctorMark(r seam.DoctorRow) string {
	switch {
	case r.Found == nil:
		return mark("skip")
	case *r.Found:
		return mark("pass")
	}
	return mark("fail")
}

// hanging wraps text to width, the lines after the first indented by indent.
func hanging(text string, width int, indent string) string {
	lines := strings.Split(ansi.Wordwrap(text, max(width-len(indent), 20), " "), "\n")
	return strings.Join(lines, "\n"+indent)
}

// doctorLine is empty: the verdict opens the page's body.
func doctorLine(Facts) (string, string) { return "", "" }

// doctorGroups are the page's groups in order. Python gives each check its group; an n/a check goes to "Not on this
// machine" whatever its group, and an engine not found to its own group, with its fix.
var doctorGroups = []struct{ key, title string }{
	{"gpu", "This GPU"}, {"engines", "Engines you can run here"}, {"engines-missing", "Engines to set up"},
	{"tools", "Measuring tools"}, {"", "Other checks"}, {"n/a", "Not on "}, {"boltbeam", "BoltBeam itself"},
}

// doctorGroup is the page group a check is drawn in.
func doctorGroup(r seam.DoctorRow) string {
	switch {
	case r.Found == nil:
		return "n/a"
	case r.Group == "engines" && !*r.Found:
		return "engines-missing"
	case r.Group == "gpu" || r.Group == "engines" || r.Group == "tools" || r.Group == "boltbeam":
		return r.Group
	}
	return ""
}

// doctorBody is the verdict, a plain intro, then the checks in groups apart by a blank line. A check found reads its
// name and what was found; one not found adds "not found" and its fix under it. An n/a check is one dimmed line, and
// so is BoltBeam's own Python, Go and screen when found.
func doctorBody(f Facts, width int) string {
	intro := "What " + here() + " has for measuring, read without changing anything. A check that is missing says " +
		"what will not work, and how to fix it."
	switch {
	case f.DoctorErr != "":
		return stWarn.Render(glyphWarn+" Not checked: "+f.DoctorErr) + "\n\n" + intro
	case f.Doctor == nil:
		return stMuted.Render("Checking "+here()+"…") + "\n\n" + intro
	}
	paras := []string{capFirst(verdict(f)) + ".", intro}
	for _, g := range doctorGroups {
		var b strings.Builder
		for _, r := range f.Doctor.Rows {
			if doctorGroup(r) != g.key {
				continue
			}
			found := r.Found != nil && *r.Found
			if r.Found == nil || (found && g.key == "boltbeam") { // one dimmed line
				b.WriteString("\n" + truncate(doctorMark(r)+" "+stMuted.Render(r.Item+" · "+r.Detail), width))
				continue
			}
			text := r.Item + " · " + r.Detail
			if !found {
				text = r.Item + " · not found · " + r.Detail
			}
			b.WriteString("\n" + doctorMark(r) + " " + hanging(text, width-2, "  "))
			if r.Fix != nil {
				b.WriteString("\n    " + stAccent.Render("→ ") + hanging(*r.Fix, width-6, "      "))
			}
			if len(r.ForksMissing) > 0 { // a fork not here is never a thing to set up: one dimmed line
				b.WriteString("\n  " + truncate(stMuted.Render("forks not found: "+strings.Join(r.ForksMissing, ", ")), width-2))
			}
		}
		title := g.title
		if g.key == "n/a" {
			title += here()
		}
		if b.Len() > 0 {
			paras = append(paras, stHeader.Render(title)+b.String())
		}
	}
	return strings.Join(paras, "\n\n")
}

func doctorActions(f Facts) []action {
	again := action{"[ Check again ]", "doctor", ""}
	if f.DoctorBusy {
		again = action{stMuted.Render(strings.TrimSpace(f.Spin + " checking…")), "", ""} // the cursor stays on it
	}
	out := append([]action{again}, sideActions(f, "host")...)
	return append(out, action{"[ Back to setup ]", "page", fmt.Sprint(pageSetup)})
}
