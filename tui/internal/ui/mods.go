package ui

// Apply mods: a row under an engine family on the Engine picker (when Python can apply mods to that engine here) opens
// a page listing each mod that fits the build: what it does, its switch, its output label, chosen or not and why
// (`screen mods`). [ Apply mods ] builds the chosen ones into a copy of the build as a job (`screen mods --apply`,
// minutes; the user's build is never changed); the engine list is then read again and the modded build is a fork of
// the family, picked with ←→ like any other. A run on it passes --modded, and Python sets its mods' switches.

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// ModsState is the Apply mods page: the engine it is for, the listing, why it failed, and whether a listing or a
// build is running.
type ModsState struct {
	Engine   string
	Plan     *seam.Mods
	Err      string
	Loading  bool
	Applying bool
	Built    *seam.Mods // the answer of the last --apply: its build, or why none was made
}

// modsMsg is the answer of `screen mods` (applied: of `screen mods --apply`).
type modsMsg struct {
	engine  string
	mods    *seam.Mods
	err     error
	applied bool
}

// modsAction is Apply mods under the engine's row: offered when Python can apply mods to this engine here.
func modsAction(f Facts, engine string) []action {
	p := f.provider(engine)
	if p != nil && !p.Mods && p.ModsNote != nil { // mods exist, not for this GPU backend: Python's reason in the row's place
		return []action{note("    " + modsWord("row_not_here", "{why}", *p.ModsNote))}
	}
	if p == nil || !p.Mods {
		return nil
	}
	if f.Mods.Applying && f.Mods.Engine == engine {
		return []action{note("    " + f.Spin + " " + modsWord("row_building", "{engine}", engine))}
	}
	return []action{{"    " + modsWord("row"), "mods", engine}}
}

// openMods shows the Apply mods page for the engine and asks Python which mods fit (a few seconds).
func (m *Model) openMods(engine string) tea.Cmd {
	m.cursor, m.row = pageMods, 0
	m.view.GotoTop()
	if m.f.Mods.Applying || m.f.Mods.Loading && m.f.Mods.Engine == engine {
		return nil
	}
	if m.f.Mods.Engine != engine {
		m.f.Mods = ModsState{Engine: engine}
	}
	m.f.Mods.Loading, m.f.Mods.Err = true, ""
	client, target := m.client, m.targetID()
	return func() tea.Msg {
		r, err := client.Mods(engine, target, false, false)
		return modsMsg{engine, r, err, false}
	}
}

// modsConfirm is f.Confirm while [ Apply mods ] (or [ Rebuild ]) waits for its second Enter: a build takes minutes.
const modsConfirm = "mods:apply"

// applyMods builds the chosen mods into a copy, as a job, on the second Enter (the first asks); rebuild: even though
// that build is on disk. The answer reads the engine list again.
func (m *Model) applyMods(rebuild bool) tea.Cmd {
	s := m.f.Mods
	if s.Applying || s.Plan == nil || len(s.Plan.Chosen) == 0 || m.targetID() == "" {
		return nil
	}
	if m.f.Confirm != modsConfirm {
		m.f.Confirm = modsConfirm
		return nil
	}
	m.f.Confirm = ""
	if live := gpuJobs(m.store); len(live) > 0 { // a build beside a measurement would skew its timing
		m.note = "Run " + live[0].ID + " is still going. Stop it or wait, then Apply mods."
		return nil
	}
	m.f.Mods.Applying, m.f.Mods.Built = true, nil
	m.note = fmt.Sprintf("Apply mods: building %d mods into a copy of %s (a build can take minutes; yours is never changed)…", len(s.Plan.Chosen), s.Engine)
	client, target, engine := m.client, m.targetID(), s.Engine
	return func() tea.Msg {
		r, err := client.Mods(engine, target, true, rebuild)
		return modsMsg{engine, r, err, true}
	}
}

// modsAnswer takes a listing or a build. A build made: the engine list is read again, and the family rests on the
// new fork, so Engine shows it in brackets.
func (m Model) modsAnswer(msg modsMsg) (Model, tea.Cmd) {
	if msg.engine != m.f.Mods.Engine {
		return m, nil
	}
	if !msg.applied {
		m.f.Mods.Loading, m.f.Mods.Plan, m.f.Mods.Err = false, msg.mods, ""
		if msg.err != nil {
			m.f.Mods.Plan, m.f.Mods.Err = nil, msg.err.Error()
		}
		return m, nil
	}
	m.f.Mods.Applying, m.f.Mods.Built = false, msg.mods
	switch {
	case msg.err != nil:
		m.f.Mods.Built, m.note = nil, "Apply mods failed: "+msg.err.Error()
		return m, nil
	case msg.mods.Refused != nil:
		m.note = "Apply mods: nothing was built: " + *msg.mods.Refused
		return m, nil
	case msg.mods.Build == nil:
		m.note = "Apply mods: Python answered without a build"
		return m, nil
	}
	if fam, _ := m.f.familyOf(msg.engine); fam != nil {
		if m.f.ForkAt == nil {
			m.f.ForkAt = map[string]string{}
		}
		m.f.ForkAt[fam.Family] = msg.mods.Build.Key
	}
	built := "built "
	if msg.mods.Reused {
		built = "already built, not rebuilt: "
	}
	m.note = "Apply mods: " + built + msg.mods.Build.Key + ". Enter on " + msg.engine + "'s row on Engine picks it (with mods, in brackets)."
	if t := m.targetID(); t != "" {
		return m, m.loadProviders(t)
	}
	return m, nil
}

// useModded picks the modded build the chosen mods already made (the listing's key) as the engine's fork, as Enter on
// the family row would, and goes back to Setup; a build the engine list does not offer (yet) is said, not picked.
func (m Model) useModded() (Model, tea.Cmd) {
	s := m.f.Mods
	if s.Plan == nil || s.Plan.Key == "" {
		return m, nil
	}
	fam, _ := m.f.familyOf(s.Engine)
	if fam == nil {
		return m, nil
	}
	for _, k := range fam.Forks {
		if k.Modded != nil && *k.Modded == s.Plan.Key {
			if m.f.ForkAt == nil {
				m.f.ForkAt = map[string]string{}
			}
			m.f.ForkAt[fam.Family] = k.Fork
			return m.do(action{"", "fork", fam.Family})
		}
	}
	m.note = "Apply mods: " + s.Plan.Key + " is built but the engine list does not offer it yet; open Engine again."
	return m, nil
}

// modsLine is empty: the page's body says it all.
func modsLine(Facts) (string, string) { return "", "" }

// modsBody is the page: what Apply mods does, how the mods fit the build, then each mod with its switch, output label
// and whether it is chosen, and why. Every sentence about a mod is Python's.
func modsBody(f Facts, width int) string {
	s := f.Mods
	var b strings.Builder
	b.WriteString(wrapText(modsWord("intro", "{engine}", s.Engine), width) + "\n\n")
	switch {
	case s.Err != "":
		return b.String() + stWarn.Render(wrapText(modsWord("not_listed", "{error}", s.Err), width))
	case s.Plan == nil:
		return b.String() + stMuted.Render(strings.TrimSpace(f.Spin+" "+modsWord("checking")))
	case s.Plan.Refused != nil && len(s.Plan.Rows) == 0:
		return b.String() + stWarn.Render(wrapText(*s.Plan.Refused, width))
	}
	if s.Plan.Fits != "" {
		b.WriteString(stMuted.Render(wrapText(modsWord("this_build", "{fits}", s.Plan.Fits), width)) + "\n")
		if c := s.Plan.ModsCommit; c != nil && c.Words != "" {
			b.WriteString(stMuted.Render(wrapText(c.Words, width)) + "\n")
		}
		b.WriteString("\n")
	}
	for _, r := range s.Plan.Rows {
		b.WriteString(modRow(r, width))
	}
	if built := s.Built; built != nil && built.Build != nil {
		b.WriteString("\n" + stOK.Render(glyphPass+" "+modsWord("built", "{key}", built.Build.Key)) + stMuted.Render(" · "+built.Build.Status) + "\n")
		b.WriteString(stMuted.Render(wrapText(built.Build.Folder, width)) + "\n")
	}
	return b.String()
}

// modRow is one mod: chosen or not, its name and output label, then what it does, its switch and why it is (not) chosen.
func modRow(r seam.ModRow, width int) string {
	head := stMuted.Render(glyphOpen) + " " + r.Name
	if r.Chosen {
		head = stOK.Render(glyphPass) + " " + r.Name
	}
	sw := modsWord("no_switch")
	if r.Switch != nil {
		sw = modsWord("switch", "{switch}", *r.Switch)
	}
	if r.SwitchWhy != nil {
		sw += ": " + *r.SwitchWhy
	}
	out := head + stMuted.Render(" · "+r.OutputWords) + "\n"
	for _, l := range []string{r.About, sw, r.Why} {
		if l != "" {
			out += hangHard(l, 4, width) + "\n"
		}
	}
	return out
}

// modsActions are Apply mods (or why it cannot run), and the way back to Engine.
func modsActions(f Facts) []action {
	s := f.Mods
	out := []action{}
	switch {
	case s.Applying:
		out = append(out, note("  "+f.Spin+" "+modsWord("applying")))
	case s.Plan == nil:
	case len(s.Plan.Chosen) == 0:
		out = append(out, note("  "+modsWord("apply_none")))
	case s.Plan.AlreadyBuilt && s.Plan.Key != "": // that exact set is built and on disk: offer it, rebuild only when asked
		n, label := fmt.Sprint(len(s.Plan.Chosen)), "rebuild"
		if f.Confirm == modsConfirm {
			label = "rebuild_confirm"
		}
		out = append(out, action{modsWord("use", "{key}", s.Plan.Key), "modsuse", ""}, action{modsWord(label, "{n}", n), "modsapply", "rebuild"})
	default:
		label := "apply"
		if f.Confirm == modsConfirm {
			label = "confirm"
		}
		out = append(out, action{modsWord(label, "{n}", fmt.Sprint(len(s.Plan.Chosen))), "modsapply", ""})
	}
	return append(out, action{modsWord("back"), "page", fmt.Sprint(pageEngine)})
}

// modsWord is the Apply mods page's sentence or label from Python's words (plain_words.json mods.page), its {fields}
// filled from pairs (field, value, ...); "" before the words load.
func modsWord(key string, pairs ...string) string {
	return strings.NewReplacer(pairs...).Replace(vocab.Mods.Page[key])
}

// moddedFork is the modded build Enter picked for the chosen engine, while it is still offered and runs here; else nil.
func (f Facts) moddedFork() *seam.Fork {
	fam, _ := f.familyOf(f.Engine)
	if fam == nil || f.Modded == "" {
		return nil
	}
	for i, k := range fam.Forks {
		if k.Modded != nil && *k.Modded == f.Modded && k.Provider != nil && *k.Provider == f.Engine && f.forkRuns(k) {
			return &fam.Forks[i]
		}
	}
	return nil
}

// moddedKey is the picked modded build's key (Run's --modded), or "".
func (f Facts) moddedKey() string {
	if k := f.moddedFork(); k != nil {
		return *k.Modded
	}
	return ""
}

// highlightedModded is the key of the modded build the family's highlight rests on, or "".
func (f Facts) highlightedModded(family string) string {
	if fam := f.family(family); fam != nil {
		if k := fam.Forks[f.highlighted(*fam)]; k.Modded != nil {
			return *k.Modded
		}
	}
	return ""
}

// modsRowAt is the Engine picker's row index of Apply mods for the engine, or 0.
func modsRowAt(f Facts, engine string) int {
	for i, a := range engineActions(f) {
		if a.do == "mods" && a.arg == engine {
			return i
		}
	}
	return 0
}

// familyModsAction is Apply mods under a family row: offered for the engine of the family's root fork (the build the
// mods apply to), when it runs here.
func familyModsAction(f Facts, fam seam.Family) []action {
	out, seen := []action{}, map[string]bool{}
	for _, k := range fam.Forks {
		if k.Provider != nil && !seen[*k.Provider] && f.forkRuns(k) && k.Modded == nil {
			seen[*k.Provider] = true
			out = append(out, modsAction(f, *k.Provider)...)
		}
	}
	return out
}
