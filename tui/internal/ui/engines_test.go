package ui

import (
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
	tea "github.com/charmbracelet/bubbletea"
)

// The engine scan runs at boot, not when Engine is opened. Until it answers, the Setup Engine line says it is
// looking; when it answers, the providers are loaded and the line names the engine. [ Scan again ] in the picker
// runs the scan once more; opening the picker never does.
func TestEnginesAreLookedForAtBootThenListed(t *testing.T) {
	s := loadSample(t)
	var m tea.Model = New(seam.Client{}, jobs.Store{}, "/models/Qwen3-8B.gguf", "", 512)
	for _, msg := range []tea.Msg{targetsMsg{&s.targets, nil}, detectMsg{id: "apple_m3_10c", count: 1}, profileMsg{&s.profile, nil},
		ceilingMsg{&s.ceiling, nil}, runsMsg{&s.runs, nil}, tea.WindowSizeMsg{Width: 80, Height: 24}} {
		m, _ = m.Update(msg)
	}
	looking := m.View()
	if !strings.Contains(looking, "looking for engines…") {
		t.Fatalf("the Engine line must say it is looking while the scan runs:\n%s", looking)
	}
	golden(t, "setup-engines-looking.txt", looking)
	if acts := engineActions(facts(m)); len(acts) != 1 || acts[0].do != "note" {
		t.Fatalf("the picker shows the looking state only while the scan runs: %v", acts)
	}
	// the scan answers: the providers are asked for (a tea.Cmd), and the list fills when they arrive
	path := "/opt/homebrew/bin/llama-bench"
	m, cmd := m.Update(enginesMsg{&seam.Engines{Seconds: 0.3, Engines: []seam.EngineFound{{Item: "llama-bench", Path: &path}}}, nil})
	if cmd == nil {
		t.Fatal("the scan's answer must load the providers")
	}
	if facts(m).EngineScan {
		t.Fatal("EngineScan must clear when the scan answers")
	}
	m, _ = m.Update(providersMsg{"apple_m3_10c", engines()})
	listed := m.View()
	if !strings.Contains(listed, "llama.cpp") || strings.Contains(listed, "looking for engines") {
		t.Fatalf("after the scan the Engine line names the engine:\n%s", listed)
	}
	acts := rows(engineActions(facts(m)))
	if acts[len(acts)-1] != "engines-scan|[ Scan again ] look for engines in the usual folders" {
		t.Fatalf("the picker ends with [ Scan again ]: %v", acts)
	}
	golden(t, "picker-engine-scan.txt", page(m, pageEngine).View())
	// Scan again: the looking state returns and the providers are dropped until the scan answers
	again, cmd := m.(Model).do(action{"", "engines-scan", ""})
	if cmd == nil || !again.f.EngineScan || again.f.Providers != nil {
		t.Fatalf("[ Scan again ] runs the scan and shows the looking state: scan=%v providers=%v", again.f.EngineScan, again.f.Providers)
	}
}
