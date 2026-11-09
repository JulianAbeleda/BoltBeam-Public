package ui

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// --- Engine settings ------------------------------------------------------------------------------------------

// settingsOffered are the settings the chosen engine takes (Python: capture_settings.offered). One that needs a draft
// head is offered only when the recommendation for this model says the checkpoint has one: unknown is not yes.
func (f Facts) settingsOffered() []seam.EngineSetting {
	p := f.engineRow()
	if p == nil {
		return nil
	}
	draft := f.Recommend != nil && f.RecKey == f.recommendKey() && f.Recommend.Weights.Mtp > 0
	out := []seam.EngineSetting{}
	for _, s := range p.Settings {
		if !s.DraftHead || draft {
			out = append(out, s)
		}
	}
	return out
}

// settingsArgv is what Run passes the pipeline: each offered setting with a value, as its flag and the value.
func (f Facts) settingsArgv() []string {
	var argv []string
	for _, s := range f.settingsOffered() {
		if v := f.Settings[s.Name]; v != "" {
			argv = append(argv, s.Flag, v)
		}
	}
	return argv
}

// settingsValue is the Setup line: the settings asked for, else the engine's own choice.
func settingsValue(f Facts) string {
	var parts []string
	for _, s := range f.settingsOffered() {
		if v := f.Settings[s.Name]; v != "" {
			parts = append(parts, s.Label+" "+v)
		}
	}
	if len(parts) == 0 {
		return "engine defaults"
	}
	return strings.Join(parts, " · ")
}

func settingsBody(f Facts, width int) string {
	return "Settings are what the run asks the engine for. Each one changes what is measured, so the Run page says it. " +
		"A setting left unset keeps the engine's own choice."
}

// settingsActions are the engine's settings, one row each, with its help under it: enter types a value, or steps
// through a fixed set of values and back to unset.
func settingsActions(f Facts) []action {
	offered := f.settingsOffered()
	if len(offered) == 0 {
		return []action{note("  This engine takes no settings here.")}
	}
	out := []action{}
	for _, s := range offered {
		v := f.Settings[s.Name]
		switch {
		case f.SettingEdit == s.Name:
			v = f.Input + "▏"
		case v == "":
			v = stMuted.Render("not set")
		}
		out = append(out, action{fmt.Sprintf("  %-18s %s", s.Label, v), "setting", s.Name})
		for _, l := range strings.Split(ansi.Wordwrap(s.About, 66, " "), "\n") { // fits an 80-column screen
			out = append(out, note("    "+l))
		}
	}
	return out
}

// setting is the offered setting by name, or nil.
func (f Facts) setting(name string) *seam.EngineSetting {
	for _, s := range f.settingsOffered() {
		if s.Name == name {
			return &s
		}
	}
	return nil
}

// settingTyped checks a typed value: a whole number above 0 where the setting takes one; "" unsets it.
func settingTyped(s seam.EngineSetting, v string) (string, bool) {
	v = strings.TrimSpace(v)
	if n, err := strconv.Atoi(v); v != "" && s.Whole && (err != nil || n < 1) {
		return "", false
	}
	return v, true
}

// setSetting asks for v ("" unsets it) in a copy of the map, so an earlier Facts keeps its own.
func (f *Facts) setSetting(name, v string) {
	f.Settings = maps.Clone(f.Settings)
	if f.Settings == nil {
		f.Settings = map[string]string{}
	}
	f.Settings[name] = v
}

// nextChoice is the value after v in a fixed set; after the last, unset again.
func nextChoice(choices []string, v string) string {
	if i := slices.Index(choices, v); i+1 < len(choices) {
		return choices[i+1]
	}
	return ""
}

// --- Repeats --------------------------------------------------------------------------------------------------

// repeatRuns is the n Repeats offers (Python: repeat_bands.PIN_RUNS); 0 when Python offers none.
func (f Facts) repeatRuns() int {
	if f.Providers == nil {
		return 0
	}
	return f.Providers.RepeatRuns
}

// repeats is how many times Run measures the setup: the offered n when picked, else once.
func (f Facts) repeats() int {
	if f.Repeats > 1 && f.Repeats == f.repeatRuns() {
		return f.Repeats
	}
	return 1
}

func repeatsValue(f Facts) string { return runsWords(f.repeats()) }

func runsWords(n int) string {
	if n > 1 {
		return fmt.Sprintf("%d runs, then their repeat band", n)
	}
	return "1 run"
}

func repeatsBody(f Facts, width int) string {
	return fmt.Sprintf("Repeats runs the same setup more than once. Over %d runs BoltBeam measures how much each number "+
		"moves on its own, so a later change can be checked against that. Each run takes as long as one Run.", f.repeatRuns())
}

func repeatsActions(f Facts) []action {
	out := []action{}
	for _, n := range []int{1, f.repeatRuns()} {
		chosen := "  "
		if n == f.repeats() {
			chosen = stAccent.Render("● ")
		}
		out = append(out, action{chosen + runsWords(n), "repeats", fmt.Sprint(n)})
	}
	return out
}

// repeatsResult is the Run page's repeat band: n runs of this setup, its lines (Python words them) and whether it may
// justify an engine pin.
func repeatsResult(res seam.Results) string {
	r := res.Repeats
	if r == nil {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Repeats: %d runs of this setup. A row reads moved by chance %.0f%% of the time.\n", r.N, 100*r.FalseMoveRate)
	for _, l := range r.Lines {
		b.WriteString("  " + l + "\n")
	}
	if r.Pin != nil {
		b.WriteString(stMuted.Render("Not enough for an engine pin: "+*r.Pin+".") + "\n")
	}
	return b.String()
}

// --- Measure MTP ----------------------------------------------------------------------------------------------

// mtpID is Measure MTP's job for the model and chip: its log, and its output folder in the work area.
func (f Facts) mtpID() string { return RunStem(f.Path, f.target().ID) + "-mtp" }

func (f Facts) measuringMtp() bool { return f.MJob != nil && f.MJob.Alive }

// mtpAction is the Recommended setup row that measures MTP here; nil when Python offers none (not vLLM, or no draft head).
func mtpAction(f Facts) []action {
	o := f.Recommend.MtpMeasure
	switch {
	case o == nil:
		return nil
	case f.measuringMtp():
		return []action{note("  " + f.Spin + " measuring MTP: one engine start per draft count, several minutes…")}
	case o.Blocked != nil:
		return []action{note("  [ Measure MTP ] " + *o.Blocked)}
	}
	ks := make([]string, len(o.Ks))
	for i, k := range o.Ks {
		ks[i] = fmt.Sprint(k)
	}
	return []action{{"[ Measure MTP ] decode with " + strings.Join(ks, ", ") + " draft tokens against none", "mtp", ""}}
}

// mtpBody is this checkpoint's MTP measurement: per depth and k, the median tok/s over the prompts, the speedup over
// k=0, the tokens kept per step and the share of drafts kept.
func mtpBody(f Facts) string {
	m := f.Recommend.MtpMeasured
	if m == nil {
		return ""
	}
	when := m.MeasuredAt
	if len(when) > 16 {
		when = strings.Replace(when[:16], "T", " ", 1) + " UTC"
	}
	var b strings.Builder
	b.WriteString("\n" + stHeader.Render("MTP measured") + " " + stMuted.Render(when) + "\n")
	byK := func(a, c string) int { x, _ := strconv.Atoi(a); y, _ := strconv.Atoi(c); return x - y }
	for _, depth := range slices.Sorted(maps.Keys(m.Depths)) {
		for _, k := range slices.SortedFunc(maps.Keys(m.Depths[depth]), byK) {
			x := m.Depths[depth][k]
			line := fmt.Sprintf("  %s, context %.0f, k=%s: %.1f tok/s, %.2fx k=0", depth, x.Context, k, x.MedianTokS, x.Speedup)
			if x.AcceptLength != nil && x.AcceptRate != nil {
				line += fmt.Sprintf(", %.2f tokens per step, %.0f%% of drafts kept", *x.AcceptLength, 100**x.AcceptRate)
			}
			b.WriteString(line + stMuted.Render(fmt.Sprintf(" (prompts spread %.0f%%)", 100*x.Spread)) + "\n")
		}
	}
	return b.String()
}
