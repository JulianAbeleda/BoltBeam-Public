package ui

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// airMeasurement is `screen providers` on this Air (no Xcode): in-model cannot run, generic is the default.
const airMeasurement = `{"default": "generic", "fallback": "in-model needs Xcode's Metal System Trace on this Mac", "options": [
 {"id": "in_model", "label": "in-model", "available": false, "reason": "in-model needs Xcode's Metal System Trace on this Mac",
  "about": "the real token, split by role when the engine labels its kernels (tinygrad, nsys); llama.cpp on a Mac: whole step only"},
 {"id": "generic", "label": "generic (kernel timer)", "available": true, "reason": null,
  "about": "each kernel alone, cold, any engine, any chip; compares kernels, the token split is an estimate"}]}`

func airProgram(t *testing.T) tea.Model {
	t.Helper()
	var opts seam.MeasurementOptions
	if err := json.Unmarshal([]byte(airMeasurement), &opts); err != nil {
		t.Fatal(err)
	}
	p := engines()
	for i := range p.Providers {
		if p.Providers[i].Provider == "llama.cpp" {
			p.Providers[i].Measurement = &opts
		}
	}
	m, _ := program(loadSample(t), nil, nil, nil, 80, 24).Update(providersMsg{"apple_m3_10c", p})
	return m
}

// On this Air the Setup line says generic and why: in-model is not available here. The picker mutes in-model
// with the Xcode reason; each row carries its one-line description.
func TestMeasurementFallbackAndPicker(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	m := airProgram(t)
	got := rows(setupActions(facts(m)))
	want := "page|Measurement generic (kernel timer)\t›"
	if !slices.Contains(got, want) || !slices.Contains(got, "note|(in-model not available here: needs Xcode's Metal System Trace)") || !slices.Contains(got, "analyze|[ Run ]") {
		t.Fatalf("setup:\n%s", strings.Join(got, "\n"))
	}
	golden(t, "setup-measurement-fallback.txt", m.View())
	picker := page(m, pageMeasurement)
	acts := rows(measurementActions(facts(picker)))
	if acts[0] != "note|in-model · in-model needs Xcode's Metal System Trace on this Mac" || acts[3] != "measurement|● generic (kernel timer)" ||
		!strings.Contains(acts[1], "the real token, split by role") || !strings.Contains(acts[5], "the token split is an estimate") {
		t.Fatalf("picker:\n%s", strings.Join(acts, "\n"))
	}
	golden(t, "picker-measurement.txt", picker.View())
}

// A pick of generic is passed to the pipeline; a pick of in-model where it cannot run greys Run with the reason.
func TestMeasurementPickReachesRunOrGreysIt(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	m := airProgram(t).(Model)
	m.f.Measurement = "generic"
	if m.f.roleTimeArg() != "generic" || !strings.Contains(measurementValue(m.f), "generic (kernel timer)") {
		t.Fatalf("generic pick: %q %q", m.f.roleTimeArg(), measurementValue(m.f))
	}
	argv := seam.Client{Python: "py"}.PipelineArgv(seam.Pipeline{Model: "m", RunDir: "r", Target: "t", RoleTime: m.f.roleTimeArg()})
	if i := slices.Index(argv, "--role-time"); i < 0 || argv[i+1] != "generic" {
		t.Fatalf("argv: %v", argv)
	}
	m.f.Measurement = "in_model"
	got := rows(setupActions(m.f))
	if !slices.Contains(got, "|[ Run ] in-model needs Xcode's Metal System Trace on this Mac") {
		t.Fatalf("in-model here must grey Run:\n%s", strings.Join(got, "\n"))
	}
	golden(t, "setup-measurement-blocked.txt", m.View())
}

// The results say how the run was measured, from its own record.
func TestMeasurementSentence(t *testing.T) {
	s := func(v string) *string { return &v }
	for _, c := range []struct {
		m    seam.Measurement
		want string
	}{
		{seam.Measurement{Choice: "auto", Label: s("generic (kernel timer)"), Fallback: s("in-model needs nsys")},
			"Measurement: generic (kernel timer) (in-model not available here: in-model needs nsys)."},
		{seam.Measurement{Choice: "generic", Label: s("generic (kernel timer)")}, "Measurement: generic (kernel timer), chosen in Setup."},
		{seam.Measurement{Choice: "auto", Label: s("in-model")}, "Measurement: in-model."},
	} {
		if got := measurementSentence(&c.m); got != c.want {
			t.Errorf("got %q want %q", got, c.want)
		}
	}
}
