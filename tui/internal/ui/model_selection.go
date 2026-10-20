package ui

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type modelSelection struct {
	Path string `json:"path"`
}

func savedModel(dir string) string {
	if dir == "" {
		return ""
	}
	raw, err := os.ReadFile(filepath.Join(dir, "model-selection.json"))
	var saved modelSelection
	if err != nil || json.Unmarshal(raw, &saved) != nil {
		return ""
	}
	if _, err = os.Stat(saved.Path); err != nil {
		return ""
	}
	return saved.Path
}

func (m *Model) saveModel() {
	if m.store.Dir == "" || m.f.Path == "" {
		return
	}
	path, err := filepath.Abs(m.f.Path)
	if err != nil {
		return
	}
	raw, err := json.Marshal(modelSelection{Path: path})
	if err != nil {
		return
	}
	if err = os.MkdirAll(m.store.Dir, 0700); err != nil {
		m.note = "Could not remember model: " + err.Error()
		return
	}
	file, err := os.CreateTemp(m.store.Dir, ".model-selection-*")
	if err != nil {
		m.note = "Could not remember model: " + err.Error()
		return
	}
	defer os.Remove(file.Name())
	_, err = file.Write(raw)
	closeErr := file.Close()
	if err == nil {
		err = closeErr
	}
	if err == nil {
		err = os.Rename(file.Name(), filepath.Join(m.store.Dir, "model-selection.json"))
	}
	if err != nil {
		m.note = "Could not remember model: " + err.Error()
	}
}

// Anchor to the selected path rather than an index, so discovery order can change.
func (m *Model) focusModel() {
	if m.cursor != pageModel {
		return
	}
	for index, a := range m.actions() {
		if a.do == "file" && a.arg == m.f.Path {
			m.row = index
			return
		}
	}
}
