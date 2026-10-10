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
	"os"
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
	// Work is where new runs live until saved (the TUI uses <root>/.work); empty means Root, as --json always did.
	Work string
	// Saved is the saved-runs folder: a run is exported there by Save.
	Saved string
}

func (c Client) work() string {
	if c.Work != "" {
		return c.Work
	}
	return c.Root
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
	return filepath.Join(c.work(), id), nil
}

// Places are where an existing run id is looked up, in this order (screen.py run_places says the same):
// <root>/<id> (--json start and older layouts), <root>/.work/<id> (the screens' unsaved runs), <saved>/<id>.
func (c Client) Places() []Place {
	saved := c.Saved
	if saved == "" {
		saved = filepath.Join(c.Root, "saved")
	}
	return []Place{{"runs", c.Root}, {"work", filepath.Join(c.Root, ".work")}, {"saved", saved}}
}

// Place is one folder runs live in, and what it is called in `runs`: runs, work or saved.
type Place struct{ Where, Dir string }

// FindRun is the folder of an existing run: the screens read their own work area; an agent (--json, Work
// empty) gets the first of Places holding the id, or an error naming every place tried.
func (c Client) FindRun(id string) (string, error) {
	dir, err := c.RunDir(id)
	if err != nil || c.Work != "" {
		return dir, err
	}
	tried := []string{}
	for _, p := range c.Places() {
		d := filepath.Join(p.Dir, id)
		at := d // a relative root is relative to the checkout, where Python runs
		if !filepath.IsAbs(at) && c.Repo != "" {
			at = filepath.Join(c.Repo, at)
		}
		if _, err := os.Stat(filepath.Join(at, "run_manifest.json")); err == nil {
			return d, nil
		}
		tried = append(tried, d)
	}
	return "", &Error{Message: fmt.Sprintf("no run %s: looked in %s", id, strings.Join(tried, ", ")), Code: 1}
}

func (c Client) Targets() (*Targets, []byte, error) {
	var t Targets
	raw, _, err := c.decode(&t, nil, "boltbeam.workflow.screen", "targets")
	return &t, raw, err
}

// Chips is every chip in Setup's groups, with this machine's own (workflow/chips.py).
func (c Client) Chips() (*Targets, []byte, error) {
	var t Targets
	args := []string{"chips"}
	for _, p := range c.Places() { // the runs the ceiling reads this machine's measured read from: one number
		args = append(args, "--facts-root", p.Dir)
	}
	raw, _, err := c.decode(&t, nil, "boltbeam.workflow.screen", args...)
	return &t, raw, err
}

// Autoscan uses the profile that fits this machine's GPU, or measures and saves a new one; remeasure refreshes a
// profile made here.
func (c Client) Autoscan(remeasure bool) (*ChipScan, []byte, error) {
	var s ChipScan
	args := []string{"autoscan"}
	if remeasure {
		args = append(args, "--remeasure")
	}
	raw, _, err := c.decode(&s, nil, "boltbeam.workflow.screen", args...)
	return &s, raw, err
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

// Ceiling is the model's speed limit on the chip. here says the chip is this machine's: then the newest machine
// facts in the run folders set the memory speed (measured on this GPU), so the limit is the one runs compare with.
func (c Client) Ceiling(model, target string, context int, here bool) (*Ceiling, []byte, error) {
	var ce Ceiling
	args := []string{"ceiling", model, "--target", target}
	if context > 0 {
		args = append(args, "--context", strconv.Itoa(context))
	}
	if here {
		for _, p := range c.Places() {
			args = append(args, "--facts-root", p.Dir)
		}
	}
	raw, _, err := c.decode(&ce, nil, "boltbeam.workflow.screen", args...)
	return &ce, raw, err
}

// Ceilings is the model's limit on every registered chip: read-only what-if facts for step 3.
func (c Client) Ceilings(model string) (*Ceilings, error) {
	var ce Ceilings
	_, _, err := c.decode(&ce, nil, "boltbeam.workflow.screen", "ceilings", model)
	return &ce, err
}

func (c Client) List() (*Runs, []byte, error) {
	var r Runs
	args := []string{"runs", "--root", c.work()}
	if c.Work == "" { // an agent: every run, wherever it lives, each marked runs, work or saved
		args = append(args, "--all", "--saved", c.Places()[2].Dir)
	}
	raw, _, err := c.decode(&r, nil, "boltbeam.workflow.screen", args...)
	return &r, raw, err
}

func (c Client) Show(id string) (*Run, []byte, error) {
	dir, err := c.FindRun(id)
	if err != nil {
		return nil, nil, err
	}
	var r Run
	raw, _, err := c.decode(&r, nil, "boltbeam.workflow.screen", "run", "--run", dir)
	return &r, raw, err
}

// Delete removes one run folder through the seam; Python refuses anything that is not a run under the root.
func (c Client) Delete(id string) ([]byte, error) {
	dir, err := c.FindRun(id)
	if err != nil {
		return nil, err
	}
	raw, _, err := c.call("boltbeam.workflow.screen", nil, "delete", "--run", dir, "--root", filepath.Dir(dir))
	return raw, err
}

// Save exports a run into the saved-runs folder: the run, results.json, report.html, summary.txt.
func (c Client) Save(id string) (*SavedRun, []byte, error) {
	dir, err := c.FindRun(id)
	if err != nil {
		return nil, nil, err
	}
	var s SavedRun
	raw, _, err := c.decode(&s, nil, "boltbeam.workflow.screen", "save", "--run", dir, "--to", c.Saved)
	return &s, raw, err
}

// SavedRuns lists the saved runs, newest first.
func (c Client) SavedRuns() (*Saved, []byte, error) {
	var s Saved
	raw, _, err := c.decode(&s, nil, "boltbeam.workflow.screen", "saved", "--root", c.Saved)
	return &s, raw, err
}

// DeleteSaved removes one saved run; Python refuses anything that is not a run directly in the saved folder.
func (c Client) DeleteSaved(id string) ([]byte, error) {
	if id == "" || id != filepath.Base(id) || strings.HasPrefix(id, ".") {
		return nil, fmt.Errorf("saved run id must be a folder name: %q", id)
	}
	raw, _, err := c.call("boltbeam.workflow.screen", nil, "delete", "--run", filepath.Join(c.Saved, id), "--root", c.Saved)
	return raw, err
}

// ShowSaved reads a saved run the way Show reads a run in the work area.
func (c Client) ShowSaved(id string) (*Run, error) {
	var r Run
	_, _, err := c.decode(&r, nil, "boltbeam.workflow.screen", "run", "--run", filepath.Join(c.Saved, id))
	return &r, err
}

// CleanWork removes temporary runs started before the session began, keeping the ones named.
func (c Client) CleanWork(before string, keep []string) ([]byte, error) {
	args := []string{"clean-work", "--work", c.work(), "--before", before}
	for _, k := range keep {
		args = append(args, "--keep", k)
	}
	raw, _, err := c.call("boltbeam.workflow.screen", nil, args...)
	return raw, err
}

// Results returns the run's results and whether anything was measured (exit 3 from the seam otherwise).
func (c Client) Results(id string) (*Results, []byte, int, error) {
	dir, err := c.FindRun(id)
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
	Model, RunDir, Target, Workload, ID, Probe, Timing, Measure, Provider, Layout string
	Batch                                                                         string // batch sizes beside 1, as "8" or "8,32"
	// Analyze is one press: measure, the machine's facts, then per-role time with the same engine
	Analyze bool
}

func (c Client) PipelineArgv(p Pipeline) []string {
	argv := []string{c.Python, "-m", "boltbeam.workflow.screen", "pipeline", p.Model, "--run", p.RunDir, "--target", p.Target}
	for flag, value := range map[string]string{"--workload": p.Workload, "--id": p.ID, "--probe": p.Probe, "--timing": p.Timing, "--measure": p.Measure, "--provider": p.Provider, "--layout": p.Layout, "--batch": p.Batch} {
		if value != "" {
			argv = append(argv, flag, value)
		}
	}
	if p.Analyze {
		argv = append(argv, "--analyze")
	}
	return argv
}

// CompareReady asks Python whether this machine can compare kernels per role for the run.
func (c Client) CompareReady(id string) (*CompareReady, error) {
	dir, err := c.RunDir(id)
	if err != nil {
		return nil, err
	}
	var r CompareReady
	_, _, err = c.decode(&r, nil, "boltbeam.workflow.screen", "compare-ready", "--run", dir)
	return &r, err
}

// CompareArgv is what the step 5 action starts: the seam's compare command, one line per role.
func (c Client) CompareArgv(runDir string) []string {
	return []string{c.Python, "-m", "boltbeam.workflow.screen", "compare", "--run", runDir}
}

// RoleTimeArgv times every role inside a real decode with a provider; "" is the run's own provider.
func (c Client) RoleTimeArgv(runDir, provider string) []string {
	argv := []string{c.Python, "-m", "boltbeam.workflow.screen", "role-time", "--run", runDir}
	if provider != "" {
		argv = append(argv, "--provider", provider)
	}
	return argv
}

// Providers lists the runtimes that can measure the target on this machine.
func (c Client) Providers(target string) (*Providers, error) {
	var p Providers
	_, _, err := c.decode(&p, nil, "boltbeam.workflow.screen", "providers", "--target", target)
	return &p, err
}

// CompareProgress reads the compare command's own lines: "compare roles: N", "role ROLE QUANT: search|ab",
// "role ROLE QUANT: done STATUS", "compare failed: REASON". Done maps "ROLE QUANT" to its status; Now is the role
// in flight and what it is doing.
type CompareProgress struct {
	Total  int
	Done   map[string]string
	Now    string
	Doing  string
	Failed string
}

func ReadCompare(lines []string) CompareProgress {
	p := CompareProgress{Done: map[string]string{}}
	for _, line := range lines {
		if n, ok := strings.CutPrefix(line, "compare roles: "); ok {
			p = CompareProgress{Done: map[string]string{}}
			p.Total, _ = strconv.Atoi(n)
		} else if reason, ok := strings.CutPrefix(line, "compare failed: "); ok {
			p.Failed = reason
		} else if reason, ok := strings.CutPrefix(line, "role-time failed: "); ok {
			p.Failed = reason
		} else if line == "role-time: start" {
			p.Now, p.Doing = "every role", "time"
		} else if line == "role-time: done" {
			p.Now, p.Doing = "", ""
		} else if rest, ok := strings.CutPrefix(line, "role "); ok {
			key, event, ok := strings.Cut(rest, ": ")
			if !ok {
				continue
			}
			if status, done := strings.CutPrefix(event, "done "); done {
				p.Done[key], p.Now, p.Doing = status, "", ""
				continue
			}
			p.Now, p.Doing = key, event
		}
	}
	return p
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

// Progress is the current pipeline's own account of itself, read from the lines after its last
// "pipeline steps: N" line, so an older run in the same log does not leak in.
type Progress struct {
	Done, Total int
	Expect      []float64 // seconds per step from this machine's last run ("pipeline expect: a,b,…"); nil on a first run
	Current     string    // the stage running now, or ""
	Sub, SubOf  int       // "stage X: progress p/q" inside the current stage
	Finished    bool      // "pipeline done"
	Failed      string    // the stage that failed, or ""
	Reason      string    // what it said
	Lines       []string  // this pipeline's lines
}

// ReadProgress reads the pipeline's lines: steps, expected seconds, stage start, progress, done and failed.
func ReadProgress(lines []string) Progress {
	start := 0
	for i, line := range lines {
		if strings.HasPrefix(line, "pipeline steps: ") {
			start = i
		}
	}
	p := Progress{Lines: lines[start:]}
	for _, line := range p.Lines {
		if n, ok := strings.CutPrefix(line, "pipeline steps: "); ok {
			p.Total, _ = strconv.Atoi(n)
			continue
		}
		if list, ok := strings.CutPrefix(line, "pipeline expect: "); ok {
			for _, s := range strings.Split(list, ",") {
				v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
				if err != nil {
					p.Expect = nil
					break
				}
				p.Expect = append(p.Expect, v)
			}
			continue
		}
		if strings.HasPrefix(line, "pipeline done") {
			p.Finished = true
			continue
		}
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
			p.Current, p.Sub, p.SubOf = key, 0, 0
		case event == "done":
			p.Done++
			p.Current, p.Sub, p.SubOf = "", 0, 0
		case strings.HasPrefix(event, "progress "):
			a, b, _ := strings.Cut(strings.TrimPrefix(event, "progress "), "/")
			p.Sub, _ = strconv.Atoi(a)
			p.SubOf, _ = strconv.Atoi(b)
		case strings.HasPrefix(event, "failed"):
			p.Failed, p.Current = key, ""
			p.Reason = strings.TrimPrefix(strings.TrimPrefix(event, "failed"), ": ")
		}
	}
	return p
}

// Fraction is how far the pipeline is, 0 to 1. Each step weighs its expected seconds (equal weights on a first
// run); the current step adds the time spent in it, or its own p of q when larger, never past its weight.
// It stays under 0.99 until "pipeline done".
func (p Progress) Fraction(inStage float64) float64 {
	if p.Finished {
		return 1
	}
	if p.Total <= 0 {
		return 0
	}
	w := p.Expect
	if len(w) != p.Total {
		w = make([]float64, p.Total)
		for i := range w {
			w[i] = 30 // a first run: every step the same
		}
	}
	sum, done := 0.0, 0.0
	for i, x := range w {
		sum += x
		if i < p.Done {
			done += x
		}
	}
	if p.Done < len(w) && p.Current != "" {
		e := w[p.Done]
		t := min(max(inStage, 0), e)
		if p.SubOf > 0 {
			t = max(t, e*float64(min(p.Sub, p.SubOf))/float64(p.SubOf))
		}
		done += min(t, e)
	}
	if sum <= 0 {
		return 0
	}
	return min(done/sum, 0.99)
}
