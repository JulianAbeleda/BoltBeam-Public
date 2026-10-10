package ui

import (
	"os"
	"path/filepath"
	"testing"
)

// The Model picker lists every model file in the model's folder, in name order, with no cap: on the 5090 box a
// folder of 16 .gguf files showed 10 and hid both 27B models and the 8B the wrapper passed (2026-10-10).
func TestFindFilesListsEveryModelInTheFolder(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir) // ~/models is empty here, so only the model's own folder counts
	models := filepath.Join(dir, "models")
	if err := os.MkdirAll(models, 0o755); err != nil {
		t.Fatal(err)
	}
	names := []string{"alpha-8B-Q4_K_M.gguf", "bravo-8B-Q4_K_M.gguf", "charlie-8B-Q4_K_M.gguf", "charlie-27B-Q4_K_L.gguf",
		"delta-0.6B-Q8_0.gguf", "echo-8B-Q4_K_M.gguf", "foxtrot-0.6B-Q8_0.gguf", "golf-30B-A3B-Q4_K_M.gguf", "golf-4B-BF16.gguf",
		"golf-4B-Q4_K_M.gguf", "hotel-0.6B-Q8_0.gguf", "hotel-14B-Q4_K_M.gguf", "hotel-8B-Q4_K_M.gguf", "hotel-27B-Q4_K_M.gguf",
		"hotel-27B-plain-Q4_K_M.gguf", "india-4B-Q4_K_M.gguf", "notes.txt"}
	for _, n := range names {
		if err := os.WriteFile(filepath.Join(models, n), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	m := Model{f: Facts{Path: filepath.Join(models, "hotel-8B-Q4_K_M.gguf")}}
	got, ok := m.findFiles()().(filesMsg)
	if !ok {
		t.Fatalf("findFiles did not return a filesMsg")
	}
	if len(got) != 16 {
		t.Fatalf("listed %d files, want all 16: %v", len(got), got)
	}
	for i, want := range []string{"hotel-8B-Q4_K_M.gguf", "hotel-27B-Q4_K_M.gguf", "hotel-27B-plain-Q4_K_M.gguf", "india-4B-Q4_K_M.gguf"} {
		found := false
		for _, p := range got {
			if filepath.Base(p) == want {
				found = true
			}
		}
		if !found {
			t.Errorf("%d: %s missing from the picker: %v", i, want, got)
		}
	}
	for i := 1; i < len(got); i++ {
		if got[i-1] > got[i] {
			t.Errorf("not in name order at %d: %s after %s", i, got[i], got[i-1])
		}
	}
}
