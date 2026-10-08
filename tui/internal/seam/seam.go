// Package seam is the Go side of BoltBeam's one machine-readable boundary: it runs
// `python -m boltbeam.workflow.screen <command>` (and `python -m boltbeam.cli inspect`) and decodes the
// JSON they print.
//
// Python owns compute and data; this package never reads a run artifact itself and never writes one. The
// contract is pinned on both sides by tui/testdata (the fixture run folders and the expected JSON), checked
// by seam_test.go here and tests/test_workflow_screen.py in Python.
package seam

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// Client names the interpreter, the BoltBeam checkout (the cwd the modules run in; empty means the current
// directory, for an installed package) and the runs folder.
type Client struct {
	Python string
	Repo   string
	Root   string
}

// Error is what the seam reports when Python exits non-zero. Message is Python's own `error` field when it
// printed one, else the process error with its stderr.
type Error struct {
	Message string
	Stderr  string
	Code    int
}

func (e *Error) Error() string { return e.Message }

// ExitNoMeasurement is the seam's exit code for `results` on a run with no measurement: the facts are still
// printed, so it is not an error here.
const ExitNoMeasurement = 3

func (c Client) module(module string, args ...string) *exec.Cmd {
	cmd := exec.Command(c.Python, append([]string{"-m", module}, args...)...)
	cmd.Dir = c.Repo
	return cmd
}

// call runs one module and returns the raw bytes it printed with its exit code; codes other than 0 and the
// ones in `ok` are errors.
func (c Client) call(module string, ok []int, args ...string) ([]byte, int, error) {
	cmd := c.module(module, args...)
	var out, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		code = -1
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			code = exit.ExitCode()
		}
	}
	for _, accepted := range ok {
		if code == accepted {
			err = nil
		}
	}
	if err == nil {
		return bytes.TrimSpace(out.Bytes()), code, nil
	}
	var reported struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(out.Bytes(), &reported) == nil && reported.Error != "" {
		return nil, code, &Error{Message: reported.Error, Stderr: stderr.String(), Code: code}
	}
	return nil, code, &Error{Message: fmt.Sprintf("%s -m %s %s: %v", c.Python, module, strings.Join(args, " "), err),
		Stderr: strings.TrimSpace(stderr.String()), Code: code}
}

// Call runs one screen command and returns the raw JSON bytes Python printed, so agent mode can print the
// same bytes a human screen was built from.
func (c Client) Call(args ...string) ([]byte, error) {
	raw, _, err := c.call("boltbeam.workflow.screen", nil, args...)
	return raw, err
}

func (c Client) decode(v any, ok []int, module string, args ...string) ([]byte, int, error) {
	raw, code, err := c.call(module, ok, args...)
	if err != nil {
		return nil, code, err
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return nil, code, fmt.Errorf("seam output is not the expected shape: %w", err)
	}
	return raw, code, nil
}

// RunDir resolves a run id inside the runs folder and refuses path tricks.
func (c Client) RunDir(id string) (string, error) {
	if id == "" || id != filepath.Base(id) || strings.HasPrefix(id, ".") {
		return "", fmt.Errorf("run id must be a folder name under the runs folder: %q", id)
	}
	return filepath.Join(c.Root, id), nil
}

func (c Client) Targets() (*Targets, []byte, error) {
	var t Targets
	raw, _, err := c.decode(&t, nil, "boltbeam.workflow.screen", "targets")
	return &t, raw, err
}

// Detect is the chip this machine is, as autoscan's GPU probe reads it; TargetID is nil when no probe answered.
func (c Client) Detect() (*Detected, error) {
	var d Detected
	_, _, err := c.decode(&d, nil, "boltbeam.workflow.screen", "detect")
	return &d, err
}

// Inspect is `boltbeam inspect MODEL`: the model profile, read without a GPU.
func (c Client) Inspect(model string) (*Profile, []byte, error) {
	var p Profile
	raw, _, err := c.decode(&p, nil, "boltbeam.cli", "inspect", model)
	return &p, raw, err
}

func (c Client) Ceiling(model, target string, context int) (*Ceiling, []byte, error) {
	var ce Ceiling
	args := []string{"ceiling", model, "--target", target}
	if context > 0 {
		args = append(args, "--context", strconv.Itoa(context))
	}
	raw, _, err := c.decode(&ce, nil, "boltbeam.workflow.screen", args...)
	return &ce, raw, err
}

func (c Client) List() (*Runs, []byte, error) {
	var r Runs
	raw, _, err := c.decode(&r, nil, "boltbeam.workflow.screen", "runs", "--root", c.Root)
	return &r, raw, err
}

func (c Client) Show(id string) (*Run, []byte, error) {
	dir, err := c.RunDir(id)
	if err != nil {
		return nil, nil, err
	}
	var r Run
	raw, _, err := c.decode(&r, nil, "boltbeam.workflow.screen", "run", "--run", dir)
	return &r, raw, err
}

// Delete removes one run folder through the seam; Python refuses anything that is not a run under the root.
func (c Client) Delete(id string) ([]byte, error) {
	dir, err := c.RunDir(id)
	if err != nil {
		return nil, err
	}
	raw, _, err := c.call("boltbeam.workflow.screen", nil, "delete", "--run", dir, "--root", c.Root)
	return raw, err
}

// Results returns the run's results and whether anything was measured (exit 3 from the seam otherwise).
func (c Client) Results(id string) (*Results, []byte, int, error) {
	dir, err := c.RunDir(id)
	if err != nil {
		return nil, nil, 0, err
	}
	var r Results
	raw, code, err := c.decode(&r, []int{ExitNoMeasurement}, "boltbeam.workflow.screen", "results", "--run", dir)
	return &r, raw, code, err
}

// Pipeline is what `start` runs: the seam's own pipeline command, which runs the stages in order and prints a
// line per stage. Probe and timing are optional evidence files to ingest after analyze. Measure "auto" asks the
// pipeline to measure with the chip's own BoltBeam collector when this machine can (Python decides).
type Pipeline struct {
	Model, RunDir, Target, Workload, ID, Probe, Timing, Measure string
}

func (c Client) PipelineArgv(p Pipeline) []string {
	argv := []string{c.Python, "-m", "boltbeam.workflow.screen", "pipeline", p.Model, "--run", p.RunDir, "--target", p.Target}
	for flag, value := range map[string]string{"--workload": p.Workload, "--id": p.ID, "--probe": p.Probe, "--timing": p.Timing, "--measure": p.Measure} {
		if value != "" {
			argv = append(argv, flag, value)
		}
	}
	return argv
}

// NextName is the first `<prefix>-NNN` not already a run folder.
func NextName(runs *Runs, prefix string) string {
	taken := map[string]bool{}
	if runs != nil {
		for _, r := range runs.Runs {
			taken[r.ID] = true
		}
	}
	for i := 1; ; i++ {
		name := fmt.Sprintf("%s-%03d", prefix, i)
		if !taken[name] {
			return name
		}
	}
}

// StageEvents reads the pipeline's own lines ("stage load: start", "stage load: done", "stage load: failed: …")
// out of a log tail and returns the last state seen per stage key.
func StageEvents(lines []string) map[string]string {
	state := map[string]string{}
	for _, line := range lines {
		rest, ok := strings.CutPrefix(line, "stage ")
		if !ok {
			continue
		}
		key, event, ok := strings.Cut(rest, ": ")
		if !ok {
			continue
		}
		switch {
		case event == "start":
			state[key] = "running"
		case event == "done":
			state[key] = "done"
		case strings.HasPrefix(event, "failed"):
			state[key] = "failed"
		}
	}
	return state
}

// PipelineProgress counts the stages the pipeline finished out of the total its first line announced
// ("pipeline steps: N"). Total is 0 when the log does not carry that line.
func PipelineProgress(lines []string) (done, total int) {
	for _, line := range lines {
		if n, ok := strings.CutPrefix(line, "pipeline steps: "); ok {
			total, _ = strconv.Atoi(n)
			done = 0
		} else if strings.HasPrefix(line, "stage ") && strings.HasSuffix(line, ": done") {
			done++
		}
	}
	return done, total
}
