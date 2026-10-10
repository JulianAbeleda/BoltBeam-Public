package seam

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

// repoRoot is the BoltBeam checkout: this file sits at tui/internal/seam.
func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func expected(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tui", "testdata", "expected", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// The pinned JSON decodes into the typed contract with every load-bearing field present.
func TestPinnedContractDecodes(t *testing.T) {
	var targets Targets
	if err := json.Unmarshal(expected(t, "targets"), &targets); err != nil {
		t.Fatal(err)
	}
	m3 := -1
	for i, tg := range targets.Targets {
		if tg.ID == "apple_m3_10c" {
			m3 = i
		}
	}
	if m3 < 0 || !targets.Targets[m3].HasCeiling || *targets.Targets[m3].MemoryBandwidthGBs != 97.2 ||
		targets.Targets[m3].FactStatus.MemoryBandwidthGBs != "measurement" {
		t.Fatalf("targets decoded wrong: %+v", targets.Targets)
	}
	var ce Ceiling
	if err := json.Unmarshal(expected(t, "ceiling"), &ce); err != nil {
		t.Fatal(err)
	}
	if ce.ModelID != "Qwen3-8B" || ce.Decode.Context != 1 || ce.Decode.Regime != "memory" || ce.Decode.TokS == nil ||
		int(*ce.Decode.TokS*10) != 207 || ce.Prefill.Context != 512 || ce.Prefill.Regime != "compute" || len(ce.Decode.Roles) != 7 {
		t.Fatalf("ceiling decoded wrong: %+v", ce)
	}
	var runs Runs
	if err := json.Unmarshal(expected(t, "runs"), &runs); err != nil {
		t.Fatal(err)
	}
	if len(runs.Runs) != 2 || runs.Runs[0].Status != "needs_measurement" || len(runs.Runs[0].Blocked) != 2 ||
		runs.Runs[1].Measured.Timing != true || len(runs.Runs[1].Stages) != 7 || runs.Runs[1].Stages[5].Key != "ingest_timing" ||
		!runs.Runs[1].Stages[5].Done || runs.Runs[0].Stages[5].Done {
		t.Fatalf("runs decoded wrong: %+v", runs.Runs)
	}
	var run Run
	if err := json.Unmarshal(expected(t, "run-002"), &run); err != nil {
		t.Fatal(err)
	}
	if run.ID != "qwen3-8b-apple_m3_10c-002" || run.Model.RoleCount != 7 || run.NextStep == "" || !run.Results.Measured ||
		run.Results.Ceiling.Status != "modeled" || run.Results.Timing.TokS == nil || *run.Results.Timing.TokS != 17.27 ||
		len(run.Results.Kernels()) != 11 || run.Results.Kernels()[0].Name != "gemv_q4k_ffn_gate_up" ||
		run.Results.Kernels()[0].Bucket != "at_peak" || len(run.Results.Regimes) != 2 || len(run.Results.Routes) != 7 {
		t.Fatalf("run decoded wrong: %+v", run.Results)
	}
	var planned Results
	if err := json.Unmarshal(expected(t, "results-001"), &planned); err != nil {
		t.Fatal(err)
	}
	if planned.Measured || planned.Timing.Status != "absent" || len(planned.Blocked) != 2 || planned.Blocked[0].Need != "probe_evidence" {
		t.Fatalf("planned results decoded wrong: %+v", planned)
	}
}

func (r Results) Kernels() []Kernel { return r.Timing.Kernels }

func TestRunDirRefusesPaths(t *testing.T) {
	c := Client{Root: "/runs"}
	for _, bad := range []string{"", "../x", "a/b", ".hidden"} {
		if _, err := c.RunDir(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	if dir, err := c.RunDir("qwen3-8b-apple_m3_10c-001"); err != nil || dir != "/runs/qwen3-8b-apple_m3_10c-001" {
		t.Fatalf("got %q %v", dir, err)
	}
}

func TestStageEvents(t *testing.T) {
	got := StageEvents([]string{"=== start", "stage load: start", "stage load: done", "stage autoscan: start",
		"noise", "stage analyze: failed: boom"})
	want := map[string]string{"load": "done", "autoscan": "running", "analyze": "failed"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v", got)
	}
}

func TestPipelineArgv(t *testing.T) {
	c := Client{Python: "py"}
	argv := c.PipelineArgv(Pipeline{Model: "m.gguf", RunDir: "/r/x", Target: "apple_m3_10c", Workload: "decode"})
	if argv[0] != "py" || argv[3] != "pipeline" || argv[4] != "m.gguf" || len(argv) != 11 {
		t.Fatalf("argv %v", argv)
	}
}

// livePython returns an interpreter that can import the seam's module, or "" (then the live test skips).
func livePython(t *testing.T, repo string) string {
	t.Helper()
	python := os.Getenv("BOLTBEAM_PYTHON")
	if python == "" {
		python = "python3"
	}
	cmd := exec.Command(python, "-c", "import boltbeam.workflow.screen")
	cmd.Dir = repo
	if err := cmd.Run(); err != nil {
		return ""
	}
	return python
}

// The live seam on the fixture prints exactly the pinned JSON; Python's own test pins the same files.
func TestLiveSeamMatchesPinnedContract(t *testing.T) {
	repo := repoRoot(t)
	python := livePython(t, repo)
	if python == "" {
		t.Skip("no interpreter that imports boltbeam (set BOLTBEAM_PYTHON); the pinned contract still holds")
	}
	// the registry only: chip profiles saved on this machine would add rows to the pinned targets
	t.Setenv("BOLTBEAM_CHIPS_DIR", t.TempDir())
	t.Setenv("BOLTBEAM_IGNORE_REGISTRY_TARGET", "")
	c := Client{Python: python, Repo: repo, Root: filepath.Join("tui", "testdata", "fixture", "runs")}
	profile := filepath.Join(c.Root, "qwen3-8b-apple_m3_10c-001", "model_profile.json")
	for name, call := range map[string]func() ([]byte, error){
		"targets": func() ([]byte, error) { _, raw, err := c.Targets(); return raw, err },
		"ceiling": func() ([]byte, error) { return c.Call("ceiling", "--profile", profile, "--target", "apple_m3_10c") },
		"runs":    func() ([]byte, error) { _, raw, err := c.List(); return raw, err },
		"run-001": func() ([]byte, error) { _, raw, err := c.Show("qwen3-8b-apple_m3_10c-001"); return raw, err },
		"run-002": func() ([]byte, error) { _, raw, err := c.Show("qwen3-8b-apple_m3_10c-002"); return raw, err },
		"results-002": func() ([]byte, error) {
			_, raw, code, err := c.Results("qwen3-8b-apple_m3_10c-002")
			if code != 0 {
				t.Errorf("a measured run must exit 0, got %d", code)
			}
			return raw, err
		},
		"results-001": func() ([]byte, error) {
			_, raw, code, err := c.Results("qwen3-8b-apple_m3_10c-001")
			if code != ExitNoMeasurement {
				t.Errorf("a run with no measurement must exit %d, got %d", ExitNoMeasurement, code)
			}
			return raw, err
		},
	} {
		raw, err := call()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var got, want any
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(expected(t, name), &want); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: live seam differs from tui/testdata/expected/%s.json", name, name)
		}
	}
	if _, _, err := c.Show("missing-run"); err == nil {
		t.Fatal("a missing run must be an error")
	} else if e, ok := err.(*Error); !ok || e.Code != 1 {
		t.Fatalf("want a seam Error with code 1, got %#v", err)
	}
	if _, _, err := c.Ceiling(profile, "nvidia_sm89", 0, false); err == nil {
		t.Fatal("a descriptor-only chip has no ceiling and must say so")
	}
}

// A failed stage's "detail:" lines are kept apart from its one-line reason.
func TestReadProgressKeepsFailureDetails(t *testing.T) {
	p := ReadProgress([]string{"pipeline steps: 2", "stage load: start", "stage load: done", "stage measure_timing: start",
		"stage measure_timing: failed: tinygrad decode failed: x (ValueError inside the tinygrad fork at a.py:1)",
		"stage measure_timing: detail: Traceback (most recent call last):", "stage measure_timing: detail: ValueError: x"})
	if p.Failed != "measure_timing" || p.Reason != "tinygrad decode failed: x (ValueError inside the tinygrad fork at a.py:1)" {
		t.Fatalf("failed %q reason %q", p.Failed, p.Reason)
	}
	if len(p.Details) != 2 || p.Details[0] != "Traceback (most recent call last):" || p.Details[1] != "ValueError: x" {
		t.Fatalf("details %q", p.Details)
	}
}

func TestPipelineProgress(t *testing.T) {
	lines := []string{"=== start", "pipeline steps: 4", "stage load: start", "stage load: done", "stage autoscan: start",
		"stage autoscan: done", "stage analyze: start"}
	if done, total := PipelineProgress(lines); done != 2 || total != 4 {
		t.Fatalf("got %d of %d", done, total)
	}
	if done, total := PipelineProgress([]string{"stage load: done"}); done != 1 || total != 0 {
		t.Fatalf("a log without the count line has total 0, got %d of %d", done, total)
	}
}

// Run (Analyze) searches kernels by default; NoSearch adds --no-search, and only with --analyze.
func TestPipelineArgvSearchFlag(t *testing.T) {
	c := Client{Python: "py"}
	has := func(argv []string, flag string) bool {
		for _, a := range argv {
			if a == flag {
				return true
			}
		}
		return false
	}
	if a := c.PipelineArgv(Pipeline{Model: "m", RunDir: "r", Target: "t", Analyze: true}); !has(a, "--analyze") || has(a, "--no-search") {
		t.Errorf("default Run: %v", a)
	}
	if a := c.PipelineArgv(Pipeline{Model: "m", RunDir: "r", Target: "t", Analyze: true, NoSearch: true}); !has(a, "--no-search") {
		t.Errorf("quick Run: %v", a)
	}
	if a := c.PipelineArgv(Pipeline{Model: "m", RunDir: "r", Target: "t", NoSearch: true}); has(a, "--no-search") {
		t.Errorf("--no-search without --analyze: %v", a)
	}
}
