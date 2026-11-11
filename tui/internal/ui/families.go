package ui

// The Engine picker by family: one row per engine family (data/engine_families.json through `screen providers`),
// ↑↓ between families, ←→ between a family's root (upstream) and its forks. Enter picks the highlighted fork when it
// runs here; otherwise the line under the row says why, and its one fix.

import (
	"errors"
	"strings"

	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// family is a family by name, or nil.
func (f Facts) family(name string) *seam.Family {
	if f.Providers == nil {
		return nil
	}
	for i := range f.Providers.Families {
		if f.Providers.Families[i].Family == name {
			return &f.Providers.Families[i]
		}
	}
	return nil
}

// highlighted is the fork ←→ rests on in a family: the one moved to, else the fork the chosen engine runs (picked, or
// remembered from the last run as the engine is), else the root. A fork is never the default.
func (f Facts) highlighted(fam seam.Family) int {
	for i, k := range fam.Forks {
		if f.ForkAt[fam.Family] == k.Fork {
			return i
		}
	}
	for i, k := range fam.Forks {
		if k.Provider != nil && *k.Provider == f.Engine && k.State == "available" {
			return i
		}
	}
	return 0
}

// familyOf is the family and fork an engine (a provider) is, or nil.
func (f Facts) familyOf(engine string) (*seam.Family, *seam.Fork) {
	if f.Providers == nil {
		return nil, nil
	}
	for i, fam := range f.Providers.Families {
		for j, k := range fam.Forks {
			if k.Provider != nil && *k.Provider == engine {
				return &f.Providers.Families[i], &f.Providers.Families[i].Forks[j]
			}
		}
	}
	return nil, nil
}

// forkWords is a fork's state in a few words, and its fix when it is not here.
func forkWords(k seam.Fork) string {
	switch k.State {
	case "available":
		return k.Label + " · found"
	case "found":
		where := ""
		if k.Path != nil {
			where = " at " + *k.Path
		}
		return k.Label + " · found" + where + " · BoltBeam can't run it yet: " + deref(k.Reason)
	case "not_compatible":
		return k.Label + " · not on this GPU type: " + deref(k.Reason)
	}
	return k.Label + " · not found → " + deref(k.Fix)
}

// familyActions are the families, each a row with its forks; under it, the highlighted fork's state, and the layouts
// and upgrade of the engine it runs.
func familyActions(f Facts) []action {
	out := []action{}
	for _, fam := range f.Providers.Families {
		at := f.highlighted(fam)
		compatible := false
		for _, k := range fam.Forks {
			compatible = compatible || k.State != "not_compatible" && k.Provider != nil
		}
		if !compatible { // every fork it runs is for another GPU type: one muted line, never picked
			out = append(out, note(ansi.Truncate("  "+padRight(fam.Label, 14)+"not on this GPU type: "+deref(fam.Forks[0].Reason), 74, "…")))
			continue
		}
		chosen := "  "
		if fam2, _ := f.familyOf(f.Engine); fam2 != nil && fam2.Family == fam.Family && f.engineRow() != nil && f.engineRow().Available {
			chosen = stAccent.Render("● ")
		}
		names := make([]string, len(fam.Forks))
		for i, k := range fam.Forks {
			switch {
			case i == at:
				names[i] = stAccent.Render("[" + k.Label + "]")
			case k.State == "available" || k.State == "found":
				names[i] = k.Label
			default:
				names[i] = stMuted.Render(k.Label)
			}
		}
		strip := strings.Join(names, stMuted.Render(" · "))
		if len(fam.Forks) > 1 {
			strip = stMuted.Render("‹ ") + strip + stMuted.Render(" ›")
		}
		out = append(out, action{chosen + padRight(fam.Label, 14) + strip, "fork", fam.Family})
		for _, l := range strings.Split(ansi.Wordwrap(forkWords(fam.Forks[at]), 68, " /"), "\n") {
			out = append(out, note("      "+l))
		}
		if k := fam.Forks[at]; f.forkRuns(k) {
			if p := f.provider(*k.Provider); p != nil {
				out = append(out, layoutActions(f, *p)...)
			}
			out = append(out, upgradeAction(f, *k.Provider)...)
		}
	}
	return append(out, action{"  [ Scan again ] look for engines in the usual folders", "engines-scan", ""})
}

// padRight pads a name to n columns, with at least two spaces after it.
func padRight(s string, n int) string { return s + strings.Repeat(" ", max(n-ansi.StringWidth(s), 2)) }

// moveFork moves a family's highlight one fork left (-1) or right (+1), stopping at either end.
func (f *Facts) moveFork(family string, by int) {
	fam := f.family(family)
	if fam == nil {
		return
	}
	at := min(max(f.highlighted(*fam)+by, 0), len(fam.Forks)-1)
	if f.ForkAt == nil {
		f.ForkAt = map[string]string{}
	}
	f.ForkAt[family] = fam.Forks[at].Fork
}

// pickedFork is the engine enter on a family row picks: the highlighted fork's, when it runs here; else "" and why.
func (f Facts) pickedFork(family string) (engine, why string) {
	fam := f.family(family)
	if fam == nil {
		return "", ""
	}
	k := fam.Forks[f.highlighted(*fam)]
	if f.forkRuns(k) {
		return *k.Provider, ""
	}
	if k.State == "available" { // found and runnable, but not with this model
		return "", fam.Label + " " + k.Label + " cannot read this model's weight format."
	}
	return "", fam.Label + " " + forkWords(k) + "."
}

// forkNeeded is the fork a model refusal names (inspect's needs), or nil.
func forkNeeded(err error) *seam.ForkNeed {
	var e *seam.Error
	if errors.As(err, &e) {
		return e.Needs
	}
	return nil
}

// needsFork says, when the chosen model can only run on a fork the engine is not, which fork and how to get to it:
// pick it with → on Engine, or its one fix when it is not here. The engine is never switched for the user. "" when
// the model needs no fork, or the engine already is it.
func needsFork(f Facts) string {
	n := f.ModelNeeds
	if n == nil {
		return ""
	}
	fam := f.family(n.Family)
	if fam == nil {
		return ""
	}
	for _, k := range fam.Forks {
		if k.Fork != n.Fork {
			continue
		}
		if k.Provider != nil && *k.Provider == f.Engine {
			return ""
		}
		need := "This model needs the " + n.Label + " fork of " + n.FamilyLabel
		switch {
		case f.forkRuns(k):
			return need + ": → on Engine to pick it"
		case k.State == "not_here":
			return need + ": " + deref(k.Fix)
		}
		return need + ", which BoltBeam can't run yet"
	}
	return ""
}

// forkRuns says a fork can be picked: Python found it runnable here, and its engine reads the chosen model's format.
func (f Facts) forkRuns(k seam.Fork) bool {
	if k.State != "available" || k.Provider == nil {
		return false
	}
	p := f.provider(*k.Provider)
	return p != nil && p.Available
}

// familyValue is Setup's Engine value by family: the family, and the fork when it is not the root.
func familyValue(f Facts, engine string) string {
	fam, k := f.familyOf(engine)
	switch {
	case fam == nil:
		return engine
	case k.Fork == "upstream":
		return fam.Label
	}
	if label := k.Label; strings.HasSuffix(label, "fork") {
		return fam.Label + " · " + label
	}
	return fam.Label + " · " + k.Label + " fork"
}
