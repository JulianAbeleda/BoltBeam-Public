package ui

import (
	"fmt"

	tea "charm.land/bubbletea/v2"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/colorprofile"
	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// tunedProgram is the Recommended setup page with Tuned for this machine read: the recorded page (`screen tune` facts
// only on this box, 24 layers' experts in RAM) for what is chosen now.
func tunedProgram(t *testing.T, name string) Model {
	t.Helper()
	m := speedProgram(t, nil)
	m.f.Tune = TuneState{Key: m.f.tuneKey(), Page: recordedAs[seam.Tune](t, name)}
	return m
}

// The section lays out Python's rows: each knob's value, its source and reason, a tag where the knob changes outputs
// (never cut), the knobs not used and the command; every line fits 80 columns.
func TestTunedForThisMachineSection(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := tunedProgram(t, "tune-experts-in-ram.json")
	p := m.f.tunePage()
	got := plain(wrapText(tuneBody(m.f, 76), 76))
	golden(t, "recommend-tuned.txt", got)
	t.Log("\n" + got)
	squash := func(s string) string { return strings.Join(strings.Fields(ansi.Strip(s)), "") } // a wrap may break at a hyphen
	for _, r := range p.Rows {
		for _, want := range append([]string{tuneSetting(r) + " · " + r.SourceWords, r.Reason}, r.Tags...) {
			if !strings.Contains(squash(got), squash(want)) {
				t.Errorf("missing %q", want)
			}
		}
	}
	wants := append([]string{p.Title, p.Head, p.Says, p.Command}, p.Notes...)
	if p.Use.NotPassed != nil { // the recorded page passes every value it copies (H2: the pin with its cache and placement)
		wants = append(wants, *p.Use.NotPassed)
	}
	for _, want := range wants {
		if !strings.Contains(squash(got), squash(want)) {
			t.Errorf("missing %q", want)
		}
	}
	if f := tunedProgram(t, "tune-facts.json").f; !strings.Contains(squash(tuneBody(f, 76)), squash(f.tunePage().NotUsedTitle)) ||
		!strings.Contains(squash(tuneBody(f, 76)), squash(f.tunePage().NotUsed[0].Reason)) {
		t.Error("the knobs not used, with why")
	}
	for _, l := range strings.Split(got, "\n") {
		if ansi.StringWidth(l) > 76 {
			t.Fatalf("a line of %d cells: %q", ansi.StringWidth(l), l)
		}
	}
	acts := rows(recommendActions(m.f))
	if !slices.ContainsFunc(acts, func(a string) bool { return strings.HasPrefix(a, "tunemeasure|[ "+p.Measure.Label+" ]") }) ||
		!slices.ContainsFunc(acts, func(a string) bool { return strings.HasPrefix(a, "tuneuse|[ "+p.Use.Label+" ]") }) {
		t.Fatalf("the section's rows:\n%s", strings.Join(acts, "\n"))
	}
	golden(t, "recommend-tuned-80x24.txt", frame(m))
	// a refusal is its sentence, with no rows
	r := tunedProgram(t, "tune-refused.json")
	if got := plain(tuneBody(r.f, 76)); !strings.Contains(got, *r.f.Tune.Page.Refused) || len(tuneActions(r.f)) != 0 {
		t.Fatalf("refused: %q", got)
	}
}

// Facts only loads once per model, chip, engine and build, the way Offload loads, through the seam.
func TestTuneLoadsOnceThroughTheSeam(t *testing.T) {
	dir := t.TempDir()
	calls := filepath.Join(dir, "calls.txt")
	fake := filepath.Join(dir, "python")
	script := "#!/bin/sh\ncase \"$*\" in *\" tune \"*) echo \"$@\" >> " + calls + "; cat " + recordedPath("tune-facts.json") + " ;; *) echo '{}' ;; esac\n"
	if err := os.WriteFile(fake, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	m := speedProgram(t, nil)
	m.client = seam.Client{Python: fake, Repo: dir}
	cmd := m.loadTune()
	if cmd == nil || m.loadTune() != nil {
		t.Fatal("one load per key")
	}
	next, ok := m, false
	for _, c := range cmd().(tea.BatchMsg) { // the page, and the job of this key
		next, _, ok = next.tuneMsg(c())
	}
	if !ok || next.f.tunePage() == nil {
		t.Fatalf("the page is read: %+v", next.f.Tune)
	}
	if got, _ := os.ReadFile(calls); !strings.HasPrefix(string(got), "-m boltbeam.workflow.screen tune "+m.f.Path+" --engine "+m.f.Engine) ||
		strings.Count(string(got), "\n") != 1 || strings.Contains(string(got), "--measure") {
		t.Fatalf("calls: %q", got)
	}
	next.f.Path = "/models/other.gguf" // another model: read again
	if next.loadTune() == nil {
		t.Fatal("a new key loads again")
	}
}

// Use these copies the values into Engine settings: a run's settings by name, the rest as the engine's environment;
// Run passes both, Setup and the Settings page show them, and the form can unset one.
func TestUseTheseCopiesIntoEngineSettings(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := tunedProgram(t, "tune-experts-in-ram.json")
	p := m.f.tunePage()
	m.row = speedRow(m, "tuneuse")
	m = typeKeys(m, "enter")
	argv := strings.Join(m.f.settingsArgv(), " ")
	for name, v := range p.Use.Settings {
		if m.f.setting(name) == nil { // this program's engine (vLLM's offered settings) does not take it: not copied
			continue
		}
		if m.f.Settings[name] != v || !strings.Contains(argv, m.f.setting(name).Flag+" "+v) {
			t.Fatalf("setting %s: %q", name, argv)
		}
	}
	for name, v := range p.Use.Env {
		if !strings.Contains(argv, "--engine-env "+name+"="+v) || !strings.Contains(settingsValue(m.f), name+"="+v) {
			t.Fatalf("env %s: %q", name, argv)
		}
	}
	if p.Use.NotPassed != nil && !strings.Contains(m.note, *p.Use.NotPassed) {
		t.Fatalf("what is not copied is said: %q", m.note)
	}
	if other := m.f; true {
		other.Path = "/models/other.gguf" // tuned for another model: its run does not pass it
		if strings.Contains(strings.Join(other.settingsArgv(), " "), "--engine-env") {
			t.Fatal("another model's run passes no tuned environment")
		}
	}
	if !slices.ContainsFunc(rows(setupActions(m.f)), func(r string) bool { return strings.HasPrefix(r, "page|Settings") }) {
		t.Fatal("Setup lists Settings")
	}
	m = settle(m.do(action{"", "page", fmt.Sprint(pageSettings)}))
	if m.form == nil || len(m.form.env) != len(p.Use.Env) {
		t.Fatal("the form has a field per environment variable")
	}
	golden(t, "picker-settings-env.txt", frame(m))
	for range m.f.settingsOffered() { // keep each setting as it is
		m = typeKeys(m, "enter")
	}
	first := m.f.envNames()[0]
	for range m.f.Env[first] { // empty the first variable's field
		m = typeKeys(m, "backspace")
	}
	for range m.f.Env {
		m = typeKeys(m, "enter")
	}
	if _, ok := m.f.Env[first]; ok || m.form != nil || len(m.f.Env) != len(p.Use.Env)-1 {
		t.Fatalf("an emptied field unsets it: %v", m.f.Env)
	}
	if m.f.Env = nil; strings.Contains(strings.Join(m.f.settingsArgv(), " "), "--engine-env") {
		t.Fatal("no environment, no flag")
	}
	// another engine drops them, as it drops the settings
	m.f.Env = map[string]string{"X": "1"}
	m, _ = m.do(action{"", "engine", "llama.cpp"})
	if m.f.Env != nil {
		t.Fatal("a new engine clears the environment")
	}
}

// Measure on this machine is a job through the seam: its bar and the point in flight while it runs, Stop, and the
// measured page in place of the facts one when it ends.
func TestMeasureOnThisMachineJob(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	dir := t.TempDir()
	fake := filepath.Join(dir, "python")
	script := "#!/bin/sh\nprintf 'pipeline steps: 2\\nstage k1: start\\nstage k1: progress 0/4\\n'\nsleep ${TUNE_SLEEP:-0.3}\n" +
		"printf 'stage k1: done\\nstage k2: start\\nstage k2: done\\npipeline done\\n'\ncat " + recordedPath("tune-experts-in-ram.json") + "\n"
	if err := os.WriteFile(fake, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	m := tunedProgram(t, "tune-facts.json")
	m.client, m.store = seam.Client{Python: fake, Repo: dir}, jobs.Store{Dir: filepath.Join(dir, "state")}
	m.row = speedRow(m, "tunemeasure")
	next, cmd := m.do(m.actions()[m.row])
	m, _, _ = next.tuneMsg(cmd())
	if !m.f.tuning() {
		t.Fatal("the job runs")
	}
	time.Sleep(100 * time.Millisecond)
	m, _, _ = m.tuneMsg(m.readTune()())
	got := plain(tuneBody(m.f, 76))
	if !strings.Contains(got, "1 of 2") || !strings.Contains(got, m.f.tunePage().Measure.Running+": k1: point 1 of at most 4") {
		t.Fatalf("the bar and the point in flight:\n%s", got)
	}
	if acts := rows(tuneActions(m.f)); !strings.HasPrefix(acts[0], "tunestop|[ "+m.f.tunePage().Measure.Stop+" ]") {
		t.Fatalf("Stop while it runs: %v", acts)
	}
	for i := 0; i < 100 && m.f.tuning(); i++ {
		time.Sleep(20 * time.Millisecond)
		m, _, _ = m.tuneMsg(m.readTune()())
	}
	if m.f.tuning() || len(m.f.tunePage().Use.Env) < 2 || m.note != m.f.tunePage().Says {
		t.Fatalf("the answer replaces the page: %+v %q", m.f.Tune.Job, m.note)
	}
	// Stop ends a job; its log has no answer, and the facts page stays
	t.Setenv("TUNE_SLEEP", "30")
	s := tunedProgram(t, "tune-facts.json")
	s.client, s.store = seam.Client{Python: fake, Repo: dir}, jobs.Store{Dir: filepath.Join(dir, "state2")}
	s.row = speedRow(s, "tunemeasure")
	next, cmd = s.do(s.actions()[s.row])
	s, _, _ = next.tuneMsg(cmd())
	s.row = speedRow(s, "tunestop")
	next, cmd = s.do(s.actions()[s.row])
	stopped := cmd()
	if msg, ok := stopped.(gpuStoppedMsg); !ok || msg.note != s.f.tunePage().Measure.Stopped {
		t.Fatalf("stop says so: %v", stopped)
	}
	nn, _ := next.Update(stopped)
	next = nn.(Model)
	for i := 0; i < 100 && len(next.store.Alive()) > 0; i++ {
		time.Sleep(20 * time.Millisecond)
	}
	before := next.f.tunePage()
	next, _, _ = next.tuneMsg(next.readTune()())
	if next.f.tuning() || next.f.tunePage() != before || next.note != before.Measure.Stopped {
		t.Fatalf("a stopped job keeps the facts page and says it was stopped, not that it failed: %q", next.note)
	}
}

// Use these replaces what the last one copied: the environment is exactly the new page's, and the settings the last
// one copied are cleared (a setting typed by hand stays).
func TestUseTheseReplacesTheLastOne(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := tunedProgram(t, "tune-experts-in-ram.json")
	p := m.f.tunePage()
	var setting string
	for name := range p.Use.Settings {
		if s := m.f.setting(name); s != nil {
			setting = name
		}
	}
	if setting == "" {
		t.Skip("the recorded page copies no setting")
	}
	var other string
	for _, s := range m.f.settingsOffered() {
		if s.Name != setting {
			other = s.Name
		}
	}
	use := p.Use.Settings[setting]
	p.Use.Env = map[string]string{"A": "1", "B": "2"}
	m.row = speedRow(m, "tuneuse")
	m = typeKeys(m, "enter")
	if m.f.Settings[setting] != use || len(m.f.Env) != 2 {
		t.Fatalf("page 1: %v %v", m.f.Settings, m.f.Env)
	}
	if other != "" {
		m.f.setSetting(other, "typed")
		m.f.Tune.Key = m.f.tuneKey() // a typed flag is asked of the page: it reloads; this one stands to be used again
	}
	p.Use.Settings, p.Use.Env = nil, map[string]string{"A": "3"}
	m = typeKeys(m, "enter")
	if !maps.Equal(m.f.Env, map[string]string{"A": "3"}) {
		t.Fatalf("env is exactly page 2's: %v", m.f.Env)
	}
	if m.f.Settings[setting] != "" {
		t.Fatalf("a setting page 2 does not set is cleared: %v", m.f.Settings)
	}
	if other != "" && m.f.Settings[other] != "typed" {
		t.Fatalf("a setting typed by hand stays: %v", m.f.Settings)
	}
}

// The tuned environment is for one build of the engine: another build on the picker does not pass it.
func TestTunedEnvIsForOneEngineBuild(t *testing.T) {
	m := tunedProgram(t, "tune-experts-in-ram.json")
	m.f.Env = map[string]string{"A": "1"}
	m.f.EnvFor = m.f.envKey()
	if !strings.Contains(strings.Join(m.f.settingsArgv(), " "), "--engine-env A=1") {
		t.Fatal("the same build passes it")
	}
	m.f.EnvFor = m.f.Path + "|" + m.f.Engine + "|/builds/X"
	if m.f.engineBuild() == "/builds/X" || strings.Contains(strings.Join(m.f.settingsArgv(), " "), "--engine-env") {
		t.Fatal("env tuned for build X is not passed on build Y")
	}
}

// A whole-number setting whose least value Python gives as 0 (llama.cpp --cache-ram 0, --n-cpu-moe 0) takes 0, so
// Use these can copy the prompt cache off; one with no least still refuses 0.
func TestASettingTakesZeroWhenItsLeastIsZero(t *testing.T) {
	zero := 0
	if v, ok := settingTyped(seam.EngineSetting{Name: "cache_ram", Whole: true, Least: &zero}, "0"); !ok || v != "0" {
		t.Fatalf("cache_ram 0 refused: %q %v", v, ok)
	}
	if _, ok := settingTyped(seam.EngineSetting{Name: "chunk_size", Whole: true}, "0"); ok {
		t.Fatal("chunk_size 0 taken")
	}
}

// The decode rows (the draft count, flash attention, the KV type, threads): each with its value, source and reason, and
// a short "to confirm" naming the values --measure would time on this build and machine.
func TestTunedDecodeRows(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := tunedProgram(t, "tune-decode.json")
	got := plain(wrapText(tuneBody(m.f, 76), 76))
	golden(t, "recommend-tuned-decode.txt", got)
	squash := func(s string) string { return strings.Join(strings.Fields(ansi.Strip(s)), "") }
	for _, r := range m.f.tunePage().Rows {
		if r.Name != "-fa" && r.Name != "-ctk" && r.Name != "-t" && r.Name != "--spec-draft-n-max" {
			continue
		}
		if !strings.Contains(squash(got), squash(r.Reason)) || (r.Name != "--spec-draft-n-max" && (r.Confirm == nil || !strings.Contains(squash(got), squash(*r.Confirm)))) {
			t.Errorf("%s: its reason and what --measure confirms", r.Name)
		}
		if r.Confirm != nil && len(*r.Confirm) > 76 {
			t.Errorf("%s: the confirm line is %d characters, more than a line", r.Name, len(*r.Confirm))
		}
	}
}
