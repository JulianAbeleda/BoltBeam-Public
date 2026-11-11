package ui

// Fetch runs and Engine settings are Huh forms: typed fields, each checked as it is typed, a refused value
// said under its field in words, and nothing kept until the form is done. The forms only fill Facts; what then
// runs is the same seam call as before (fetch.go: listRemote, startFetch; Run passes the settings).

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// formState is the open form and the values its fields write. A pointer: every copy of the Model shares one form.
type formState struct {
	kind       string // remote (where to fetch from), pick (which runs, raw traces), settings or speedup (the confirm step)
	f          *huh.Form
	dest, port string
	picked     []string
	traces     bool
	confirmed  bool // the confirm step's answer: Make the copy
	settings   map[string]*string
	env        map[string]*string // the environment set for the engine, by name
}

// remoteRule is user@host or a host (an ssh Host alias): letters, digits, dot, dash and underscore, no leading dash.
var remoteRule = regexp.MustCompile(`^([A-Za-z0-9._-]+@)?[A-Za-z0-9_][A-Za-z0-9._-]*$`)

func checkRemote(v string) error {
	switch v = strings.TrimSpace(v); {
	case v == "":
		return errors.New("type the remote first")
	case !remoteRule.MatchString(v):
		return errors.New("type the remote as user@host, or an ssh Host alias")
	}
	return nil
}

func checkPort(v string) error {
	if v = strings.TrimSpace(v); v == "" {
		return nil
	}
	if n, err := strconv.Atoi(v); err != nil || n < 1 || n > 65535 {
		return errors.New("a port is a number from 1 to 65535, or nothing for ssh's own")
	}
	return nil
}

// formTheme is Huh drawn in the screen's tokens: purple titles, muted help, the pink cursor and marks, coral errors.
// Every choice is marked by a glyph as well as a colour, so NO_COLOR still shows which one is picked.
func formTheme(bool) *huh.Styles {
	t := huh.ThemeBase(darkBackground)
	f := &t.Focused
	f.Base = f.Base.BorderForeground(purple)
	f.Card = f.Base
	f.Title, f.NoteTitle, f.Description = stHeader, stHeader, stMuted
	f.ErrorIndicator = stBad.SetString(" *")
	f.ErrorMessage = lipgloss.NewStyle().Foreground(coral)
	f.SelectSelector = stCursor.SetString("▸ ")
	f.MultiSelectSelector = stCursor.SetString("▸ ")
	f.SelectedPrefix = stAccent.SetString("[x] ")
	f.UnselectedPrefix = lipgloss.NewStyle().SetString("[ ] ")
	f.SelectedOption = stAccent
	f.NextIndicator = stAccent.MarginLeft(1).SetString("→")
	f.PrevIndicator = stAccent.MarginRight(1).SetString("←")
	f.TextInput.Cursor, f.TextInput.Prompt, f.TextInput.Placeholder = stAccent, stAccent, stMuted
	t.Blurred = t.Focused
	t.Blurred.Base = f.Base.BorderStyle(lipgloss.HiddenBorder())
	t.Blurred.Card = t.Blurred.Base
	t.Blurred.SelectSelector = lipgloss.NewStyle().SetString("  ")
	t.Blurred.MultiSelectSelector = lipgloss.NewStyle().SetString("  ")
	t.Blurred.NextIndicator, t.Blurred.PrevIndicator = lipgloss.NewStyle(), lipgloss.NewStyle()
	t.Group.Title, t.Group.Description = stHeader, stMuted
	return t
}

// formKeys are Huh's keys without filtering: esc belongs to the screen (it leaves the form), so a filter could not end.
func formKeys() *huh.KeyMap {
	k := huh.NewDefaultKeyMap()
	k.Select.Filter.SetEnabled(false)
	k.MultiSelect.Filter.SetEnabled(false)
	return k
}

// openForm shows s on the open page: sized to the box it is drawn in, the first field focused.
func (m *Model) openForm(s *formState, fields ...huh.Field) tea.Cmd {
	s.f = huh.NewForm(huh.NewGroup(fields...)).WithTheme(huh.ThemeFunc(formTheme)).WithKeyMap(formKeys()).
		WithShowHelp(false).WithWidth(m.width - 4)
	m.form = s
	cmd := s.f.Init()
	next, _ := s.f.Update(tea.WindowSizeMsg{Width: m.width - 4, Height: max(m.detailHeight()-8, 6)})
	s.f = next.(*huh.Form)
	return cmd
}

// formStep is a message for the open form; when the form is done its values are kept and acted on.
func (m Model) formStep(msg tea.Msg) (Model, tea.Cmd) {
	next, cmd := m.form.f.Update(msg)
	m.form.f = next.(*huh.Form)
	if m.form.f.State != huh.StateCompleted {
		return m, cmd
	}
	s := m.form
	m.form = nil
	switch s.kind {
	case "remote": // another remote: list it again
		m.f.Fetch.Dest, m.f.Fetch.Port = strings.TrimSpace(s.dest), strings.TrimSpace(s.port)
		m.f.Fetch.List, m.f.Fetch.Picked, m.f.Fetch.Err, m.f.Fetch.Done = nil, nil, "", nil
		m.f.Fetch.Listing = true
		return m, m.listRemote()
	case "pick":
		m.f.Fetch.Picked, m.f.Fetch.Traces = map[string]bool{}, s.traces
		for _, id := range s.picked {
			m.f.Fetch.Picked[id] = true
		}
		if len(m.f.picked()) == len(m.f.Fetch.List.Runs) { // every run: --all, as Fetch all was
			return m.doFetch(action{"", "fetchgo", "all"})
		}
		return m.doFetch(action{"", "fetchgo", "picked"})
	case "speedup":
		if s.confirmed {
			return m, m.startSpeed()
		}
	case "settings":
		for _, set := range m.f.settingsOffered() {
			if p := s.settings[set.Name]; p != nil {
				v, _ := settingTyped(set, *p)
				m.f.setSetting(set.Name, v)
			}
		}
		for name, p := range s.env {
			m.f.setEnv(name, strings.TrimSpace(*p))
		}
		m.backToSetup()
	}
	return m, nil
}

// formKey is a key while a form is open: esc leaves it (nothing typed is kept), ctrl+c quits, the rest is the form's.
func (m Model) formKey(msg tea.KeyPressMsg) (Model, tea.Cmd) {
	switch {
	case key.Matches(msg, keyAbort):
		return m, tea.Quit
	case key.Matches(msg, keyBack):
		kind := m.form.kind
		m.form = nil
		if kind != "speedup" { // Speed up's confirm lives on the Recommended setup page: esc is Not now
			m.backToSetup()
		}
		return m, nil
	}
	return m.formStep(msg)
}

// openRemote asks where the runs are: the remote and its port, as last typed.
func (m *Model) openRemote() tea.Cmd {
	s := &formState{kind: "remote", dest: m.f.Fetch.Dest, port: m.f.Fetch.Port}
	return m.openForm(s,
		huh.NewInput().Title("Remote").Description("user@host, or an ssh Host alias").Placeholder("user@host").
			Value(&s.dest).Validate(checkRemote),
		huh.NewInput().Title("Port").Description("the remote's SSH port; empty for ssh's own (a pod names one)").
			Placeholder("ssh's own").Value(&s.port).Validate(checkPort))
}

// openPick asks which of the listed runs come home, and whether their raw traces come too.
func (m *Model) openPick() tea.Cmd {
	l := m.f.Fetch.List
	s := &formState{kind: "pick", picked: m.f.picked(), traces: m.f.Fetch.Traces}
	runs := []huh.Option[string]{}
	for _, r := range l.Runs {
		date := r.Date
		if len(date) > 10 {
			date = date[:10]
		}
		label := fmt.Sprintf("%s · %s · %s on %s · %s · %s", r.ID, remoteState[r.State], deref(r.ModelID), deref(r.TargetID),
			date, size(r.Bytes))
		runs = append(runs, huh.NewOption(truncate(label, m.width-14), r.ID)) // one line each: the box, the bar, "▸ [x] "
	}
	return m.openForm(s,
		huh.NewMultiSelect[string]().Title("Runs to fetch").
			Options(runs...).Value(&s.picked).Height(min(len(runs), ListRows)+1).Validate(func(ids []string) error {
			if len(ids) == 0 {
				return errors.New("pick at least one run (ctrl+a picks them all)")
			}
			return nil
		}),
		huh.NewSelect[bool]().Title("Raw traces").Description("they stay behind unless picked here").
			Options(huh.NewOption("leave them behind", false), huh.NewOption("copy them too", true)).Inline(true).Value(&s.traces))
}

// openSettings is the engine's settings, one field each with its help: a fixed set is chosen, anything else typed;
// then each environment variable set for the engine (Use these), typed, empty to unset it.
func (m *Model) openSettings() tea.Cmd {
	offered := m.f.settingsOffered()
	if len(offered) == 0 && len(m.f.envNames()) == 0 {
		return nil
	}
	s := &formState{kind: "settings", settings: map[string]*string{}, env: map[string]*string{}}
	fields := []huh.Field{}
	for _, set := range offered {
		v := new(string)
		*v = m.f.Settings[set.Name]
		s.settings[set.Name] = v
		if len(set.Choices) > 0 {
			options := []huh.Option[string]{huh.NewOption("not set", "")}
			for _, c := range set.Choices {
				options = append(options, huh.NewOption(c, c))
			}
			fields = append(fields, huh.NewSelect[string]().Title(set.Label).Description(set.About).Options(options...).Inline(true).Value(v))
			continue
		}
		fields = append(fields, huh.NewInput().Title(set.Label).Description(set.About).Placeholder("not set").Value(v).
			Validate(func(typed string) error {
				if _, ok := settingTyped(set, typed); !ok {
					return errors.New("a whole number above 0, or nothing to unset it")
				}
				return nil
			}))
	}
	about := "environment variable for the engine"
	if p := m.f.tunePage(); p != nil {
		about = p.Use.EnvAbout
	}
	for _, name := range m.f.envNames() {
		v := new(string)
		*v = m.f.Env[name]
		s.env[name] = v
		fields = append(fields, huh.NewInput().Title(name).Description(about).Placeholder("not set").Value(v))
	}
	return m.openForm(s, fields...)
}
