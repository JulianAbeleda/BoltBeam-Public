package ui

import (
	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestModelSelectionRestoresAcrossLaunchAndDiscoveryOrder(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chosen.gguf")
	if err := os.WriteFile(path, []byte("model"), 0600); err != nil {
		t.Fatal(err)
	}
	store := jobs.Store{Dir: filepath.Join(dir, "state")}
	m := New(seam.Client{}, store, path, "", 512)
	sample := loadSample(t)
	m.Update(profileMsg{profile: &sample.profile})
	restarted := New(seam.Client{}, store, "", "", 512)
	if restarted.f.Path != path || restarted.input.Value() != path || !restarted.f.Reading {
		t.Fatal("selected model not restored")
	}
	restarted, _ = restarted.do(action{"", "page", strconv.Itoa(pageModel)})
	next, _ := restarted.Update(filesMsg{"/before.gguf", path, "/after.gguf"})
	restarted = next.(Model)
	if a := restarted.actions()[restarted.row]; a.arg != path {
		t.Fatalf("picker not anchored to model: %+v", a)
	}
	next, _ = restarted.Update(filesMsg{path, "/before.gguf"})
	restarted = next.(Model)
	if a := restarted.actions()[restarted.row]; a.arg != path {
		t.Fatal("picker followed index instead of path")
	}
	explicit := New(seam.Client{}, store, "/explicit.gguf", "", 512)
	if explicit.f.Path != "/explicit.gguf" {
		t.Fatal("saved model overrides explicit model")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if got := New(seam.Client{}, store, "", "", 512).f.Path; got != "" {
		t.Fatal("restored missing model")
	}
}

func TestInvalidModelDoesNotReplaceSavedSelection(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(path, []byte("model"), 0600); err != nil {
		t.Fatal(err)
	}
	store := jobs.Store{Dir: dir}
	m := New(seam.Client{}, store, path, "", 512)
	m.saveModel()
	m.f.Path = "/invalid"
	m.Update(profileMsg{err: os.ErrNotExist})
	if savedModel(dir) != path {
		t.Fatal("invalid selection replaced saved model")
	}
	os.WriteFile(filepath.Join(dir, "model-selection.json"), []byte("corrupt"), 0600)
	if savedModel(dir) != "" {
		t.Fatal("corrupt state should be ignored")
	}
}
