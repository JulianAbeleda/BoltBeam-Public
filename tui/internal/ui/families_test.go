package ui

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// familyProviders is `screen providers` on a Mac with llama.cpp, its PrismML fork built (BoltBeam cannot run it yet),
// Ollama and Strata not here, vLLM for another GPU type, and tinygrad's own fork.
func familyProviders(t *testing.T) *seam.Providers {
	t.Helper()
	var p seam.Providers
	err := json.Unmarshal([]byte(`{"target_id": "apple_m3_10c", "default": "llama.cpp", "providers": [
	  {"provider": "llama.cpp", "available": true, "state": "available", "formats": ["gguf"], "capture": {"method": null}},
	  {"provider": "tinygrad", "available": true, "state": "available", "formats": ["gguf"], "capture": {"method": "tinygrad-profile-events"}},
	  {"provider": "vllm", "available": false, "state": "not_compatible", "reason": "BoltBeam runs vLLM on CUDA or AMD GPUs", "formats": ["safetensors"], "capture": {"method": null}},
	  {"provider": "ollama", "available": false, "state": "not_here", "reason": "ollama was not found", "formats": ["gguf"], "capture": {"method": null}}],
	 "families": [
	  {"family": "llama.cpp", "label": "llama.cpp", "forks": [
	    {"fork": "upstream", "label": "upstream", "repo": "https://github.com/ggml-org/llama.cpp", "provider": "llama.cpp", "runs": true, "state": "available"},
	    {"fork": "prismml", "label": "PrismML", "repo": "https://github.com/PrismML-Eng/llama.cpp", "provider": null, "runs": false, "state": "found",
	     "path": "/opt/llama.cpp-prism/build/bin/llama-bench", "reason": "a run with it is not wired yet"},
	    {"fork": "ollama", "label": "Ollama", "repo": "https://github.com/ollama/ollama", "provider": "ollama", "runs": true, "state": "not_here",
	     "fix": "get it from https://github.com/ollama/ollama, then export BOLTBEAM_OLLAMA=<its path>"},
	    {"fork": "strata", "label": "Strata", "repo": "https://github.com/Niko1221/Strata", "provider": null, "runs": false, "state": "not_here",
	     "fix": "get it from https://github.com/Niko1221/Strata, then export BOLTBEAM_STRATA=<its path>"}]},
	  {"family": "vllm", "label": "vLLM", "forks": [
	    {"fork": "upstream", "label": "upstream", "repo": "https://github.com/vllm-project/vllm", "provider": "vllm", "runs": true, "state": "not_compatible",
	     "reason": "BoltBeam runs vLLM on CUDA or AMD GPUs"}]},
	  {"family": "tinygrad", "label": "tinygrad", "forks": [
	    {"fork": "upstream", "label": "upstream", "repo": "https://github.com/tinygrad/tinygrad", "provider": null, "runs": false, "state": "not_here",
	     "fix": "get it from https://github.com/tinygrad/tinygrad, then export BOLTBEAM_TINYGRAD_UPSTREAM=<its path>"},
	    {"fork": "own", "label": "own fork", "repo": "https://example.invalid/tinygrad-fork", "provider": "tinygrad", "runs": true, "state": "available"}]}]}`), &p)
	if err != nil {
		t.Fatal(err)
	}
	return &p
}

func arrow(m tea.Model, code rune) tea.Model { m, _ = m.Update(tea.KeyPressMsg{Code: code}); return m }

// The Engine picker is one row per family: ↑↓ between families, ←→ between a family's root and its forks, each with
// its state under the row (found, or not found with its fix). Enter picks a fork that runs here; on one that cannot
// run, it says why and keeps the engine. Setup names the family, and the fork when it is not the root.
func TestEnginePickerIsByFamily(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m, _ := program(loadSample(t), nil, nil, nil, 80, 24).Update(providersMsg{"apple_m3_10c", familyProviders(t)})
	m = page(m, pageEngine)
	acts := rows(engineActions(facts(m)))
	if acts[0] != "fork|● llama.cpp     ‹ [upstream] · PrismML · Ollama · Strata ›" || acts[1] != "note|upstream · found" ||
		!strings.HasPrefix(acts[2], "note|vLLM          not on this GPU type") || acts[3] != "fork|tinygrad      ‹ [upstream] · own fork ›" { // the root is the default, found or not
		t.Fatalf("families:\n%s", strings.Join(acts, "\n"))
	}
	if !strings.Contains(plain(m.(Model).footer()), "←→ fork") {
		t.Fatalf("the footer names ←→ on the Engine picker: %q", plain(m.(Model).footer()))
	}
	m = arrow(m, tea.KeyRight)
	if got := rows(engineActions(facts(m)))[1]; !strings.HasPrefix(got, "note|PrismML · found at /opt/llama.cpp-prism/") {
		t.Fatalf("→ moves to PrismML: %q", got)
	}
	golden(t, "picker-engine-families.txt", frame(m))
	m = arrow(m, tea.KeyRight)
	if got := rows(engineActions(facts(m))); got[0] != "fork|● llama.cpp     ‹ upstream · PrismML · [Ollama] · Strata ›" ||
		!strings.Contains(strings.Join(got, " "), "Ollama · not found → get it from https://github.com/ollama/ollama, note|then export BOLTBEAM_OLLAMA") {
		t.Fatalf("→ again: Ollama, not found with its fix:\n%s", strings.Join(got, "\n"))
	}
	stay := press(m, "enter").(Model)
	if stay.cursor != pageEngine || stay.f.Engine != "llama.cpp" {
		t.Fatalf("enter on a fork not here keeps the engine: page %d engine %q", stay.cursor, stay.f.Engine)
	}
	m = arrow(arrow(arrow(arrow(m, tea.KeyRight), tea.KeyLeft), tea.KeyLeft), tea.KeyLeft)
	if got := rows(engineActions(facts(m)))[0]; got != "fork|● llama.cpp     ‹ [upstream] · PrismML · Ollama · Strata ›" {
		t.Fatalf("← back to the root, stopping at the end: %q", got)
	}
	// the tinygrad family: → then enter picks its own fork; Setup says the family and the fork
	picked := press(arrow(press(m, "j"), tea.KeyRight), "enter").(Model)
	if picked.cursor != pageSetup || picked.f.Engine != "tinygrad" || plain(engineValue(picked.f)) != "tinygrad · own fork" {
		t.Fatalf("pick: page %d engine %q value %q", picked.cursor, picked.f.Engine, plain(engineValue(picked.f)))
	}
	if got := plain(familyValue(picked.f, "ollama")); got != "llama.cpp · Ollama fork" {
		t.Fatalf("a fork that is not the root: %q", got)
	}
}

// A model BoltBeam cannot read keeps the user's choice; Setup says why under the Model line and on Run, and the
// picker under the file, from the seam's data (inspect's formats).
func TestUnreadableModelSaysWhy(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	err := &seam.Error{Message: "no block layout for PTQ1_0", Formats: []string{"PTQ1_0"}}
	if got := modelErrWords(err); got != "BoltBeam can't read this model's weight format yet (PTQ1_0)" {
		t.Fatalf("formats: %q", got)
	}
	if got := modelErrWords(errors.New("python -m boltbeam.cli inspect m: exit status 1\ntraceback")); got !=
		"BoltBeam could not read this model: python -m boltbeam.cli inspect m: exit status 1" {
		t.Fatalf("any other refusal: %q", got)
	}
	m := program(loadSample(t), nil, nil, nil, 80, 24).(Model)
	m.f.Path = "/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
	m.f.Files = append(m.f.Files, m.f.Path)
	next, _ := m.Update(profileMsg{nil, err})
	f := facts(next)
	got := rows(setupActions(f))
	if f.Path == "" || got[2] != "page|Model              Ternary-Bonsai-2-27B-PTQ1_0.gguf\t›" ||
		got[3] != "note|✗ BoltBeam can't read this model's weight format yet (PTQ1_0)" ||
		runRow(got) != "|[ Run ]            Pick a model BoltBeam can read: Model says why" {
		t.Fatalf("setup:\n%s", strings.Join(got, "\n"))
	}
	golden(t, "setup-model-unreadable.txt", frame(next))
	// the model needs a fork (inspect's needs): the root stays picked; Run and Recommended setup name the fork
	needs := &seam.Error{Message: err.Message, Formats: err.Formats, Needs: &seam.ForkNeed{Family: "llama.cpp",
		FamilyLabel: "llama.cpp", Fork: "prismml", Label: "PrismML"}}
	withFork, _ := next.Update(providersMsg{"apple_m3_10c", familyProviders(t)})
	withFork, _ = withFork.Update(profileMsg{nil, needs})
	f = facts(withFork)
	if got := runRow(rows(setupActions(f))); f.Engine != "llama.cpp" ||
		got != "|[ Run ]            This model needs the PrismML fork of llama.cpp, which BoltBeam can't run yet" {
		t.Fatalf("a fork BoltBeam cannot run yet: engine %q, %q", f.Engine, got)
	}
	if got := plain(recommendValue(f)); got != "use the PrismML fork of llama.cpp (Engine, →)" {
		t.Fatalf("Recommended setup proposes the fork: %q", got)
	}
	prism := "prism" // once a run with it is wired, the fork runs here: → on Engine picks it
	fam := f.family("llama.cpp")
	fam.Forks[1].State, fam.Forks[1].Provider = "available", &prism
	f.Providers.Providers = append(f.Providers.Providers, seam.ProviderRow{Provider: "prism", Available: true})
	if got := needsFork(f); got != "This model needs the PrismML fork of llama.cpp: → on Engine to pick it" {
		t.Fatalf("a fork that runs: %q", got)
	}
	if f.Engine = "prism"; needsFork(f) != "" {
		t.Fatal("picked: nothing more to say")
	}
	picker := rows(modelChoices(f))
	if i := strings.Index(strings.Join(picker, "\n"), "● Ternary-Bonsai-2-27B-PTQ1_0 · GGUF\nnote|✗ BoltBeam can't read"); i < 0 {
		t.Fatalf("picker:\n%s", strings.Join(picker, "\n"))
	}
}
