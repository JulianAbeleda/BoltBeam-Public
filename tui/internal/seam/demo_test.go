package seam

import (
	"path/filepath"
	"strings"
	"testing"
)

// The tapes' demo answers every call the screens make at start from the recorded JSON, the run by its id, a refusal
// in Python's words; an unrecorded call says so; running.json keeps demo.json's answers and changes the runs.
func TestDemoAnswersFromTheRecordedData(t *testing.T) {
	dir := filepath.Join(repoRoot(t), "tui", "testdata", "demo")
	d, err := LoadDemo(filepath.Join(dir, "demo.json"))
	if err != nil {
		t.Fatal(err)
	}
	c := Client{Root: "runs", Work: "runs/.work", Saved: "runs/saved", Demo: d}
	if tg, _, err := c.Chips(); err != nil || len(tg.Targets) == 0 {
		t.Fatalf("chips: %v", err)
	}
	if p, err := c.Providers("apple_m3_10c", d.Model); err != nil || len(p.Providers) != 5 || len(p.Families) != 4 {
		t.Fatalf("providers (the runs folders among its arguments): %+v %v", p, err)
	}
	if r, _, err := c.Show("qwen3-8b-apple_m3_10c-002"); err != nil || r.ID != "qwen3-8b-apple_m3_10c-002" {
		t.Fatalf("run 002: %v", err)
	}
	if _, err := c.RemoteRuns("nobody@127.0.0.1", 1); err == nil || !strings.Contains(err.Error(), "Connection refused") {
		t.Fatalf("fetch: %v", err)
	}
	if _, _, err := c.Emit("qwen3-8b-apple_m3_10c-002"); err == nil || !strings.HasPrefix(err.Error(), "not recorded in this demo: emit") {
		t.Fatalf("emit: %v", err)
	}
	running, err := LoadDemo(filepath.Join(dir, "running.json"))
	if err != nil {
		t.Fatal(err)
	}
	c.Demo = running
	if r, _, err := c.List(); err != nil || len(r.Runs) != 1 || len(running.Jobs[r.Runs[0].ID]) == 0 || running.Model != d.Model {
		t.Fatalf("running: %+v %v", r, err)
	}
}
