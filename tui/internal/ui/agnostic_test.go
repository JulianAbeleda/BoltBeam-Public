package ui

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// engineBlind are the screen files that render whatever rows and records Python returns for any engine: Tuned for this
// machine and Where the time went. They name no engine, no engine's program, weight format or build variable, and no
// engine flag (policy/guards.py holds the Python side to the same rule).
var engineBlind = []string{"internal/seam/tune.go", "internal/seam/budget.go", "internal/ui/tune.go", "internal/ui/budget.go"}

// engineTerms are the names (matched without case) and flags (matched as whole tokens) the files must not hold: every
// engine family, fork and provider in data/engine_families.json with their labels, the programs and formats engines go
// by, and every flag a fork row lists.
func engineTerms(t *testing.T) (names, flags []string) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "boltbeam", "data", "engine_families.json"))
	if err != nil {
		t.Fatal(err)
	}
	var table struct {
		Families []struct {
			Family, Label string
			Forks         []struct {
				Fork, Label string
				Provider    *string
				Flags       map[string][]string
			}
		}
	}
	if err := json.Unmarshal(raw, &table); err != nil {
		t.Fatal(err)
	}
	role := map[string]bool{"upstream": true, "master": true, "own": true, "own fork": true} // a fork's role, not a name
	seen := map[string]bool{}
	add := func(s string) {
		if s = strings.ToLower(s); s != "" && !role[s] && !seen[s] {
			seen[s] = true
			names = append(names, s)
		}
	}
	for _, f := range table.Families {
		add(f.Family)
		add(f.Label)
		add(regexp.MustCompile(`[^a-z0-9]`).Split(strings.ToLower(f.Family), 2)[0])
		for _, k := range f.Forks {
			add(k.Fork)
			add(k.Label)
			if k.Provider != nil {
				add(*k.Provider)
			}
			for _, fl := range k.Flags {
				for _, x := range fl {
					if !seen[x] {
						seen[x] = true
						flags = append(flags, x)
					}
				}
			}
		}
	}
	for _, s := range []string{"llama", "ggml", "vllm", "strata", "tensorrt", "trtllm", "sglang", "ollama", "tinygrad",
		"flashinfer", "cutlass", "cublas", "nvjet", "rocblas", "hipblaslt", "gguf", "awq", "gptq", "safetensors", "ggml_", "llama_"} {
		add(s)
	}
	return names, flags
}

// engineHits are the names and flags text holds.
func engineHits(text string, names, flags []string) []string {
	out := []string{}
	lower := strings.ToLower(text)
	for _, n := range names {
		if strings.Contains(lower, n) {
			out = append(out, n)
		}
	}
	words := map[string]bool{}
	for _, w := range regexp.MustCompile(`[-\w]+`).FindAllString(text, -1) {
		words[w] = true
	}
	for _, f := range flags {
		if words[f] {
			out = append(out, f)
		}
	}
	return out
}

// The new screens are engine-blind: a scan of their source, comments included, finds no engine's name or flag.
func TestNewScreensNameNoEngine(t *testing.T) {
	names, flags := engineTerms(t)
	if len(names) < 15 || len(flags) < 5 {
		t.Fatalf("the table was not read: %d names, %d flags", len(names), len(flags))
	}
	for _, rel := range engineBlind {
		raw, err := os.ReadFile(filepath.Join("..", "..", rel))
		if err != nil {
			t.Fatal(err)
		}
		if hits := engineHits(string(raw), names, flags); len(hits) > 0 {
			t.Errorf("%s names %v", rel, hits)
		}
	}
	// the scan fires on an engine's name in any case, and on a fork row's flag as a whole token only
	if hits := engineHits(`if f.Engine == "vLLM" { argv = append(argv, "-ub", "512") } // -ubatch`, names, flags); len(hits) != 2 {
		t.Fatalf("the scan: %v", hits)
	}
}
