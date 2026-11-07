package ui

import (
	"fmt"
	"os"
	"strings"

	"github.com/atotto/clipboard"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/muesli/termenv"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// recommendHint is what the Recommended setup line says before there is a number to show.
const recommendHint = "context, speed and engine flags for this model on your GPUs"

// recommendMsg is the answer of `screen recommend` (or its Check) for one model, chip and engine.
type recommendMsg struct {
	key   string
	rec   *seam.Recommend
	err   error
	check bool
}

// recommendKey names what the recommendation is for; empty until a model, a chip and an engine are chosen.
func (f Facts) recommendKey() string {
	if f.Path == "" || f.Profile == nil || f.target() == nil || !f.chipKnown() {
		return ""
	}
	if p := f.engineRow(); p == nil || !p.Available {
		return ""
	}
	return f.Path + "|" + f.target().ID + "|" + f.Engine
}

// loadRecommend asks Python for the recommendation when the model, chip or engine changed since the last answer.
func (m *Model) loadRecommend() tea.Cmd {
	key := m.f.recommendKey()
	if key == "" || key == m.f.RecKey {
		return nil
	}
	m.f.RecKey, m.f.Recommend, m.f.RecErr = key, nil, ""
	return m.askRecommend(key, false)
}

func (m Model) askRecommend(key string, check bool) tea.Cmd {
	path, target, engine := m.f.Path, m.f.target().ID, m.f.Engine
	return func() tea.Msg {
		r, err := m.client.Recommend(path, target, engine, check)
		return recommendMsg{key, r, err, check}
	}
}

// recommendValue is the Setup line's value: the hint until the numbers are there, then the context and the layout.
func recommendValue(f Facts) string {
	r := f.Recommend
	switch {
	case f.RecChecking:
		return strings.TrimSpace(f.Spin + " checking with one engine start…")
	case r == nil || len(r.Layouts) == 0:
		return stMuted.Render(recommendHint)
	}
	one := r.Layouts[0]
	if c, ok := one.MaxContext[r.KVDtype]; ok && c.Value != nil {
		bound := ""
		if c.Kind == "upper bound" {
			bound = "up to "
		}
		return fmt.Sprintf("%scontext %s · %s KV · %d seqs", bound, commas(*c.Value), r.KVDtype, r.Seqs)
	}
	return "does not fit one GPU"
}

func commas(v float64) string {
	s := fmt.Sprint(int64(v))
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

func figLine(label string, fig *seam.RecFig, unit string) string {
	if fig == nil {
		return ""
	}
	v := "not measured here"
	if fig.Value != nil {
		v = commas(*fig.Value) + unit
	}
	return fmt.Sprintf("  %-18s %s · %s\n    %s\n", label, v, fig.Kind, stMuted.Render(fig.Formula+": "+strings.Join(fig.Inputs, "; ")))
}

func recommendBody(f Facts, width int) string {
	r := f.Recommend
	switch {
	case f.RecErr != "":
		return stWarn.Render("The recommendation could not be made: " + f.RecErr)
	case r == nil:
		return stMuted.Render(recommendHint + "\n\nChoose a model, a chip and an engine on Setup.")
	}
	var b strings.Builder
	if r.EngineNote != "" {
		b.WriteString(stMuted.Render(r.Provider+": "+r.EngineNote) + "\n\n")
	}
	b.WriteString(stMuted.Render(recommendHint+" · derived from the checkpoint and this machine's measured facts; nothing runs until Check") + "\n\n")
	if r.ContextIsUpperBound {
		b.WriteString(stWarn.Render("Contexts are upper bounds: this engine's own overhead is not measured here yet. Check measures it.") + "\n\n")
	}
	if r.OverheadNote != nil {
		b.WriteString(stMuted.Render(*r.OverheadNote) + "\n\n")
	}
	why := "no layout is recommended: " + r.Recommended.Why
	if r.Recommended.ID != nil {
		why = "use " + *r.Recommended.ID + ": " + r.Recommended.Why
	}
	b.WriteString(stHeader.Render("Layout") + "\n  " + why + "\n")
	for _, l := range r.Layouts {
		fmt.Fprintf(&b, "\n%s", stHeader.Render(l.Label))
		if !l.Available {
			b.WriteString("\n  " + stMuted.Render(deref(l.Reason)) + "\n")
			continue
		}
		b.WriteString("\n" + figLine("Weights per GPU", &l.WeightsPerGpu, " bytes"))
		if !l.Fits.Value {
			b.WriteString("  does not fit one GPU\n")
			continue
		}
		for _, k := range r.KVFormats {
			if c, ok := l.MaxContext[k]; ok {
				b.WriteString(figLine("Max context "+k, &c, " tokens"))
			}
			if c, ok := l.MaxContextWarm[k]; ok {
				b.WriteString(figLine("  after first start", &c, " tokens"))
			}
		}
		b.WriteString(figLine("Max sequences", l.MaxSeqs, "") + figLine("All GPUs", l.TotalSeqs, "") + figLine("Batch-1 limit", l.Batch1Limit, " tok/s") + figLine("Batch-1 expected", l.Batch1Expected, " tok/s"))
		if l.Batch1Limit == nil && l.Speed != "" {
			b.WriteString("  " + stMuted.Render(l.Speed) + "\n")
		}
	}
	b.WriteString("\n" + stHeader.Render("Settings") + "\n")
	for _, s := range r.Settings {
		mark, name := "  ", deref(s.Flag)
		if s.Flag == nil {
			name = "advice"
		}
		if s.Recommended {
			mark = stAccent.Render("● ")
		}
		fmt.Fprintf(&b, "%s%s\n    %s\n", mark, name, stMuted.Render(s.Changes+" · when: "+s.When))
	}
	if c := r.Check; c != nil {
		b.WriteString("\n" + stHeader.Render("Check") + "\n")
		if c.Before.State != nil {
			fmt.Fprintf(&b, "  a %s compile cache start\n", *c.Before.State)
		}
		fmt.Fprintf(&b, "  predicted %s (%s) · measured %s · %+.1f%%\n  engine overhead %.2f GiB, cached; with it the prediction is %s (%+.1f%%)\n",
			commas(c.Before.Predicted), c.Before.PredictedKind, commas(c.Before.Measured), 100*c.Before.Error,
			c.Overhead.Bytes/(1<<30), commas(c.After.Predicted), 100*c.After.Error)
	}
	return b.String()
}

func recommendActions(f Facts) []action {
	r := f.Recommend
	if r == nil || r.Command == nil {
		return []action{note("  nothing to copy yet")}
	}
	check := action{"[ Check ] start the engine once with these flags and compare", "reccheck", ""}
	if f.RecChecking {
		check = note("  " + f.Spin + " checking: one engine start, a minute or two…")
	}
	return []action{
		{"[ Copy engine command line ]", "reccopy", "command"},
		{"[ Copy Arkey vllm_args ]", "reccopy", "arkey"},
		check,
	}
}

// copyText puts text on the clipboard: the system's where there is one, and the terminal's own (OSC 52) always,
// which also reaches a local clipboard over ssh.
func copyText(text string) {
	_ = clipboard.WriteAll(text)
	termenv.NewOutput(os.Stdout).Copy(text)
}

func (f Facts) copyOf(what string) string {
	if f.Recommend == nil {
		return ""
	}
	if what == "arkey" {
		return strings.Join(f.Recommend.Arkey, " ")
	}
	return deref(f.Recommend.Command)
}
