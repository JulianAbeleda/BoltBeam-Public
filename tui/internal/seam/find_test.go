package seam

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// An agent finds a run wherever it lives: <root>/<id>, then <root>/.work/<id>, then <saved>/<id>. The screens
// read their own work area only.
func TestFindRunOrder(t *testing.T) {
	root := t.TempDir()
	saved := filepath.Join(root, "saved")
	for _, d := range []string{filepath.Join(root, "old-001"), filepath.Join(root, ".work", "tui-002"),
		filepath.Join(saved, "kept-003"), filepath.Join(root, ".work", "old-001")} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(d, "run_manifest.json"), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	c := Client{Root: root, Saved: saved}
	for id, want := range map[string]string{"old-001": filepath.Join(root, "old-001"),
		"tui-002": filepath.Join(root, ".work", "tui-002"), "kept-003": filepath.Join(saved, "kept-003")} {
		if got, err := c.FindRun(id); err != nil || got != want {
			t.Errorf("%s: got %q %v, want %q", id, got, err, want)
		}
	}
	_, err := c.FindRun("nope-009")
	var se *Error
	if !errors.As(err, &se) || se.Code != 1 || !strings.Contains(se.Message, filepath.Join(".work", "nope-009")) ||
		!strings.Contains(se.Message, filepath.Join("saved", "nope-009")) {
		t.Fatalf("missing run: %v", err)
	}
	if _, err := c.FindRun("../old-001"); err == nil {
		t.Fatal("a path must be refused")
	}
	screens := Client{Root: root, Work: filepath.Join(root, ".work"), Saved: saved}
	if got, _ := screens.FindRun("old-001"); got != filepath.Join(root, ".work", "old-001") {
		t.Fatalf("the screens read their work area: %q", got)
	}
}
