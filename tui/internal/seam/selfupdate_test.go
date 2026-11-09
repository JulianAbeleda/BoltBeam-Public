package seam

import (
	"path/filepath"
	"testing"
)

// With BOLTBEAM_NO_UPDATE_CHECK the seam answers "off" without a network call, and Skip this version is kept.
func TestLiveSelfUpdateOff(t *testing.T) {
	repo := repoRoot(t)
	python := livePython(t, repo)
	if python == "" {
		t.Skip("no interpreter that imports boltbeam (set BOLTBEAM_PYTHON)")
	}
	t.Setenv("BOLTBEAM_NO_UPDATE_CHECK", "1")
	t.Setenv("BOLTBEAM_UPDATE_FILE", filepath.Join(t.TempDir(), "update.json"))
	c := Client{Python: python, Repo: repo}
	u, err := c.SelfUpdate("0.3.14")
	if err != nil || u.Status != "off" || u.Prompt || u.Installed == nil || u.Install == "" || u.Words == "" {
		t.Fatalf("off: %+v %v", u, err)
	}
}
