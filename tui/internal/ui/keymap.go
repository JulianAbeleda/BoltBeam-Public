package ui

// The keys, each with the words the footer and the ? page show. Both are drawn from these bindings by Bubbles
// help, so the screen names what a key does where it works and never a key that does nothing there (Nielsen 6,
// recognition over recall).

import (
	"strings"

	"charm.land/bubbles/v2/help"
	"charm.land/bubbles/v2/key"
	"charm.land/lipgloss/v2"
)

var (
	keyUp      = key.NewBinding(key.WithKeys("up", "k"), key.WithHelp("↑/k", "up"))
	keyDown    = key.NewBinding(key.WithKeys("down", "j"), key.WithHelp("↓/j", "down"))
	keyPgUp    = key.NewBinding(key.WithKeys("pgup", "ctrl+u"), key.WithHelp("pgup/ctrl+u", "page up"))
	keyPgDn    = key.NewBinding(key.WithKeys("pgdown", "ctrl+d"), key.WithHelp("pgdn/ctrl+d", "page down"))
	keyHome    = key.NewBinding(key.WithKeys("home", "g"), key.WithHelp("home/g", "first row"))
	keyEnd     = key.NewBinding(key.WithKeys("end", "G"), key.WithHelp("end/G", "last row"))
	keyEnter   = key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "pick"))
	keyBack    = key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "setup"))
	keyDel     = key.NewBinding(key.WithKeys("d"), key.WithHelp("d", "delete run"))
	keyLog     = key.NewBinding(key.WithKeys("l"), key.WithHelp("l", "log"))
	keyDigit   = key.NewBinding(key.WithKeys("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"), key.WithHelp("0-9", "type a batch size"))
	keyErase   = key.NewBinding(key.WithKeys("backspace"), key.WithHelp("backspace", "erase a digit"))
	keyCompare = key.NewBinding(key.WithKeys("c"), key.WithHelp("c", "compare"))
	keyHelp    = key.NewBinding(key.WithKeys("?"), key.WithHelp("?", "all keys"))
	keyFork    = key.NewBinding(key.WithKeys("left", "right"), key.WithHelp("←→", "fork")) // the Engine picker, by family
	keyQuit    = key.NewBinding(key.WithKeys("q", "ctrl+c"), key.WithHelp("q", "quit"))
	// keyMove is ↑ and ↓ as one item of the footer; the ? page lists each.
	keyMove = key.NewBinding(key.WithKeys(append(keyUp.Keys(), keyDown.Keys()...)...), key.WithHelp("↑↓", "move"))
	// keyAbort quits where q is typed: in a text field or a form.
	keyAbort = key.NewBinding(key.WithKeys("ctrl+c"), key.WithHelp("ctrl+c", "quit"))
	// The Results screen's layer 2: each key opens or closes one section (render_plain.go). On the ? page each is
	// named by the section's own title (keys, sectionTitle).
	keyTable    = key.NewBinding(key.WithKeys("t"), key.WithHelp("t", "table"))
	keyAllCols  = key.NewBinding(key.WithKeys("a"), key.WithHelp("a", "all columns"))
	keyMeasured = key.NewBinding(key.WithKeys("m"), key.WithHelp("m", "how it was measured"))
	keyMachine  = key.NewBinding(key.WithKeys("e"), key.WithHelp("e", "machine and engine"))
	keyGlossary = key.NewBinding(key.WithKeys("w"), key.WithHelp("w", "words used here"))
	keyFull     = key.NewBinding(key.WithKeys("f"), key.WithHelp("f", "full detail"))
	// keySections is the six as one item of the footer.
	keySections = key.NewBinding(key.WithKeys("t", "a", "m", "e", "w", "f"), key.WithHelp("t/a/m/e/w/f", "sections"))
)

// sectionBindings are layer 2's keys as this run offers them: each named by its section's title, off when the run
// has no such section.
func (m Model) sectionBindings() []key.Binding {
	out := []key.Binding{}
	for _, k := range []key.Binding{keyTable, keyAllCols, keyMeasured, keyMachine, keyGlossary, keyFull} {
		title := sectionTitle(m.f.Plain, k)
		if k.Help().Key == keyAllCols.Help().Key && title != "" {
			title = vocab.Label("all_columns")
		}
		out = append(out, offer(k, k.Help().Key, title, title != ""))
	}
	return out
}

// keyMap is the keys one screen offers: the footer's line, and the ? page's columns.
type keyMap struct {
	short []key.Binding
	full  [][]key.Binding
}

func (k keyMap) ShortHelp() []key.Binding  { return k.short }
func (k keyMap) FullHelp() [][]key.Binding { return k.full }

// offer is b with other words, or switched off: a copy, so the bindings above never change.
func offer(b key.Binding, keyWords, desc string, on bool) key.Binding {
	if keyWords != "" {
		b.SetHelp(keyWords, desc)
	}
	b.SetEnabled(on)
	return b
}

// keys are what this frame offers. A key that does nothing here is off, so neither the footer nor the ? page names it.
func (m Model) keys() keyMap {
	switch {
	case m.upd.shown: // the upgrade prompt answers its own keys (selfupdate.go)
		return keyMap{short: []key.Binding{keyUp, keyDown, keyEnter, offer(keyBack, "esc", "not now", !m.upd.running), keyQuit}}
	case m.form != nil: // the form's own keys for the field in focus (Huh), then leaving it
		short := append(append([]key.Binding{offer(keyBack, "esc", "cancel", true)}, m.form.f.KeyBinds()...), keyAbort)
		return keyMap{short: short, full: [][]key.Binding{short}}
	case m.f.Editing:
		typing := []key.Binding{offer(keyEnter, "enter", "keep", true), offer(keyBack, "esc", "cancel", true), keyAbort}
		return keyMap{short: typing, full: [][]key.Binding{typing}}
	case m.showHelp:
		return keyMap{short: []key.Binding{offer(keyHelp, "?/esc", "close", true), keyQuit}}
	}
	saved := m.runRow() != ""
	enter := keyEnter
	if saved {
		enter = offer(keyEnter, "enter", "open", true)
	}
	log := offer(keyLog, "l", "log", m.cursor == pageRun && m.f.alive())
	if m.cursor == pageRun && !m.f.alive() && m.f.failedStage() != "" {
		log = offer(keyLog, "l", "details", true)
	}
	del := offer(keyDel, "d", "delete", saved) // short words: the Saved runs footer holds compare too, in 80 columns
	cmp := offer(keyCompare, "", "", m.canPickCompare())
	back := offer(keyBack, "", "", m.cursor != pageSetup)
	batch := m.cursor == pageBatch
	fork := offer(keyFork, "", "", m.cursor == pageEngine && m.f.Providers != nil && m.f.Providers.Families != nil)
	full := [][]key.Binding{{keyUp, keyDown, keyPgUp, keyPgDn, keyHome, keyEnd},
		{enter, fork, log, del, cmp, offer(keyDigit, "", "", batch), offer(keyErase, "", "", batch), back}, {keyHelp, keyQuit}}
	short := []key.Binding{keyMove, fork, enter, log, del, cmp, back, keyHelp, keyQuit}
	if m.cursor == pageRun && m.f.plainShown() { // the Results screen: layer 2's sections open and close
		short = []key.Binding{keyMove, enter, keySections, back, keyHelp, keyQuit}
		full = [][]key.Binding{full[0], full[1], m.sectionBindings(), full[2]}
	}
	return keyMap{short: short, full: full}
}

// canPickCompare says Saved runs can start a compare: two saved runs at least, and no pick already under way.
func (m Model) canPickCompare() bool {
	return m.cursor == pageSaved && m.f.Saved != nil && len(m.f.Saved.Runs) > 1 && !m.f.ComparePick && m.f.CompareFrom == ""
}

// helpView is Bubbles help in the screen's styles: keys in the accent, words muted, " · " between them.
func helpView(width int) help.Model {
	h := help.New()
	h.ShortSeparator, h.FullSeparator = " · ", "    "
	h.Styles = help.Styles{ShortKey: stAccent, ShortDesc: stMuted, ShortSeparator: stMuted, Ellipsis: stMuted,
		FullKey: stAccent, FullDesc: lipgloss.NewStyle(), FullSeparator: stMuted}
	h.SetWidth(width)
	return h
}

// footer is the short help: one line, cut with "…" where the screen is narrower.
func (m Model) footer() string { return " " + helpView(m.width-1).ShortHelpView(m.keys().ShortHelp()) }

// helpPage is the ? page: every key this screen takes, from the same bindings as the footer.
func (m Model) helpPage(height int) string {
	m.showHelp = false // the keys of the page under it
	k := m.keys()
	body := "\n" + helpView(m.width-4).FullHelpView(k.FullHelp()) + "\n\n" +
		stMuted.Render("The wheel scrolls a list as ↑ and ↓ do. ? or esc closes this page.")
	page := box("Keys on "+steps[m.cursor].title, body, m.width, false)
	return page + strings.Repeat("\n", max(height-lipgloss.Height(page), 0))
}
