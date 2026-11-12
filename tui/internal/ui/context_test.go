package ui

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/charmbracelet/colorprofile"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func contextProgram(t *testing.T) Model {
	t.Helper()
	m := vllmProgram(t)
	for i := range m.f.Providers.Providers {
		if p := &m.f.Providers.Providers[i]; p.Provider == "vllm" {
			p.Settings = append(p.Settings, seam.EngineSetting{Name: "n_cpu_moe", Flag: "-ncmoe", Whole: true},
				seam.EngineSetting{Name: "cpu_moe", Flag: "--cpu-moe", Choices: []string{"on"}})
		}
	}
	return m
}

// One Context on Setup reaches Recommended setup, Offload and Tuned for this machine: unset it passes nothing, typed it
// is the prompt and reply together, measured it is the prompt and the file (as an absolute path); each change asks the
// pages again, and the screen reads the file, so a refusal is the page's own error.
func TestContextReachesTheThreePages(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := contextProgram(t)
	if got := contextValue(m.f); got != "each page's default" || m.f.Context.pageArgs() != nil || m.f.Context.tuneArgs() != nil {
		t.Fatalf("unset passes nothing: %q", got)
	}
	unset := m.f.recommendKey()
	m, _ = m.do(action{"", "page", "17"})
	if m.form == nil || m.form.kind != "context" {
		t.Fatal("Context opens its form")
	}
	m = typeKeys(m, "x", "enter")
	if !strings.Contains(frame(m), "a whole number above 0, or nothing") {
		t.Fatalf("the prompt is a number:\n%s", frame(m))
	}
	m = typeKeys(m, "backspace", "8000", "enter", "enter", "enter")
	if !strings.Contains(frame(m), "give the reply tokens, or a file of measured replies") {
		t.Fatalf("a prompt needs a reply:\n%s", frame(m))
	}
	m, _ = m.do(action{"", "page", "17"}) // nothing typed was kept
	m = typeKeys(m, "8000", "enter", "1000", "enter", "enter")
	if m.form != nil || m.cursor != pageSetup || contextValue(m.f) != "8,000 prompt + 1,000 reply tokens" {
		t.Fatalf("kept: %+v form %v", m.f.Context, m.form)
	}
	if got := strings.Join(m.f.Context.pageArgs(), " "); got != "--context 9000" {
		t.Fatalf("recommend and offload: %q", got)
	}
	if got := strings.Join(m.f.Context.tuneArgs(), " "); got != "--prompt 8000 --context 9000" {
		t.Fatalf("tune: %q", got)
	}
	if m.f.recommendKey() == unset || m.f.tuneKey() == "" {
		t.Fatal("another context asks the pages again")
	}
	m, _ = m.do(action{"", "page", "17"})
	m = typeKeys(m, "enter", "backspace", "backspace", "backspace", "backspace", "enter", "replies.jsonl", "enter")
	abs, _ := filepath.Abs("replies.jsonl")
	c := m.f.Context
	if c.Replies != abs || c.Reply != 0 || strings.Join(c.pageArgs(), " ") != "--prompt 8000 --replies "+abs || strings.Join(c.tuneArgs(), " ") != strings.Join(c.pageArgs(), " ") {
		t.Fatalf("measured, as an absolute path: %+v %v", c, c.pageArgs())
	}
	if got := contextValue(m.f); got != "8,000 prompt + replies measured in replies.jsonl" {
		t.Fatalf("measured value: %q", got)
	}
}

// Tuned for this machine is asked what Setup holds: the workload (session and replay serve follow-ups, decode and
// prefill single prompts), the chip, the experts in RAM from Settings, and every other setting by its name for the
// screen to turn into the engine's flag (the pipeline's flag names are not the engine's).
func TestTuneAskIsSetupsWorkloadExpertsAndSettings(t *testing.T) {
	m := contextProgram(t)
	ask := func() string { return strings.Join(m.f.tuneAsk(), " ") }
	for wl, want := range map[string]string{"": "one-shot", "prefill": "one-shot", "session": "chat", "replay": "chat"} {
		m.f.Workload = wl
		if got := ask(); got != "--workload "+want+" --target apple_m3_10c" {
			t.Fatalf("%q: %q", wl, got)
		}
	}
	m.f.Workload = ""
	m.f.setSetting("chunk_size", "512")
	m.f.setSetting("n_cpu_moe", "12")
	if got := ask(); got != "--workload one-shot --target apple_m3_10c --setting chunk_size=512 --experts-in-ram 12" {
		t.Fatalf("n_cpu_moe: %q", got)
	}
	m.f.setSetting("n_cpu_moe", "")
	m.f.setSetting("cpu_moe", "on")
	if got := ask(); got != "--workload one-shot --target apple_m3_10c --setting chunk_size=512 --experts-in-ram all" {
		t.Fatalf("cpu_moe: %q", got)
	}
	if strings.Contains(ask(), "--chunk-size") || slices.Contains(m.f.tuneAsk(), "--flags") {
		t.Fatal("the pipeline's flag names are not the engine's")
	}
	before := m.f.tuneKey()
	m.f.setSetting("chunk_size", "1024")
	if m.f.tuneKey() == before {
		t.Fatal("another setting asks the page again")
	}
}

// Measure this runs one measured row's sweep: its own action under Measure on this machine, which stays the whole sweep.
func TestMeasureThisIsOneRowsSweep(t *testing.T) {
	m := tunedProgram(t, "tune-experts-in-ram.json")
	p := m.f.tunePage()
	if len(p.Measure.Knobs) == 0 || p.Measure.One == "" {
		t.Fatalf("the recorded page names its measured rows: %+v", p.Measure)
	}
	var all, one []string
	for _, a := range tuneActions(m.f) {
		if a.do == "tunemeasure" {
			if a.arg == "" {
				all = append(all, a.label)
			} else {
				one = append(one, a.arg)
			}
		}
	}
	if len(all) != 1 || !slices.Equal(one, p.Measure.Knobs) {
		t.Fatalf("whole sweep %v, rows %v, page %v", all, one, p.Measure.Knobs)
	}
}

// A replies file typed with ~ is the home folder's, as an absolute path.
func TestContextRepliesTildeIsHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	m := contextProgram(t).contextDone([3]string{"8000", "", "~/r.jsonl"})
	if m.f.Context.Replies != filepath.Join(home, "r.jsonl") {
		t.Fatalf("%q", m.f.Context.Replies)
	}
}
