package seam

// Tuned for this machine: `screen tune MODEL --engine E [--build B] [--measure]` (workflow/tune.py, data/engine_knobs.json).
// Facts only is one call; --measure runs as a job, its progress the pipeline's own lines (ReadProgress) and its answer
// the page printed after them (ReadTune). Python owns every row and sentence; nothing here knows an engine's knobs.

import (
	"encoding/json"
	"errors"
	"strings"
)

// Tune is the answer: the page, or why there is none (Refused, in Python's words).
type Tune struct {
	Schema  string    `json:"schema"`
	Kind    string    `json:"kind"`
	Engine  string    `json:"engine"`
	Model   string    `json:"model"`
	Build   *string   `json:"build"`
	Out     *string   `json:"out"`
	Page    *TunePage `json:"page"`
	Refused *string   `json:"refused"`
	Error   string    `json:"error"`
}

// TunePage is the section as the Recommended setup page lays it out.
type TunePage struct {
	Title        string       `json:"title"`
	Head         string       `json:"head"`
	Notes        []string     `json:"notes"`
	Says         string       `json:"says"`
	Rows         []TuneRow    `json:"rows"`
	NotUsedTitle string       `json:"not_used_title"`
	NotUsed      []TuneUnused `json:"not_used"`
	Absent       []string     `json:"absent"`
	Command      string       `json:"command"`
	Measure      TuneMeasure  `json:"measure"`
	Use          TuneUse      `json:"use"`
}

// TuneRow is one knob set or left at its default: its value, where the value came from (Source is the key, SourceWords
// the word shown), why, and its tags (a knob that changes outputs says so). Use is what a run passes of it; Confirm is
// what --measure would sweep to confirm it.
type TuneRow struct {
	Name        string   `json:"name"`
	Kind        string   `json:"kind"`
	Status      string   `json:"status"`
	Value       string   `json:"value"`
	Source      string   `json:"source"`
	SourceWords string   `json:"source_words"`
	Reason      string   `json:"reason"`
	Tags        []string `json:"tags"`
	Group       *string  `json:"group"`
	Use         *TuneTo  `json:"use"`
	Confirm     *string  `json:"confirm"`
}

// TuneTo is where a run takes a row's value: a setting by name, or an environment variable.
type TuneTo struct {
	Setting string `json:"setting"`
	Env     string `json:"env"`
	Value   string `json:"value"`
}

// TuneUnused is a knob that does not apply to this model and placement, and why.
type TuneUnused struct {
	Name   string `json:"name"`
	Reason string `json:"reason"`
}

// TuneMeasure are the words of "Measure on this machine": the action, while it runs, and how it ends. None says why
// nothing is offered (no row to confirm).
type TuneMeasure struct {
	Label      string   `json:"label"`
	One        string   `json:"one"` // the action on one measured row
	Helper     string   `json:"helper"`
	Knobs      []string `json:"knobs"`
	None       *string  `json:"none"`
	Running    string   `json:"running"`
	Stop       string   `json:"stop"`
	StopHelper string   `json:"stop_helper"`
	Stopped    string   `json:"stopped"`
	Failed     string   `json:"failed"`
	Starting   string   `json:"starting"`
	Progress   string   `json:"progress"`
}

// TuneUse is "Use these": the settings and environment it copies into Engine settings, and what it cannot copy.
type TuneUse struct {
	Label     string            `json:"label"`
	Helper    string            `json:"helper"`
	Settings  map[string]string `json:"settings"`
	Env       map[string]string `json:"env"`
	None      *string           `json:"none"`
	Used      string            `json:"used"`
	NotPassed *string           `json:"not_passed"`
	EnvAbout  string            `json:"env_about"`
}

// tuneArgs: the setup's own asks (extra: context, workload, experts in RAM, flags) and the run folders, whose newest
// measured time budget the screen reads.
func (c Client) tuneArgs(model, engine, build string, extra []string) []string {
	args := append([]string{"tune", model, "--engine", engine}, extra...)
	if build != "" {
		args = append(args, "--build", build)
	}
	for _, pl := range c.Places() {
		args = append(args, "--facts-root", pl.Dir)
	}
	return args
}

// Tune reads this machine's values for the engine's knobs, from facts only: no engine starts, a few seconds of CPU.
func (c Client) Tune(model, engine, build string, extra ...string) (*Tune, error) {
	var t Tune
	_, _, err := c.decode(&t, nil, "boltbeam.workflow.screen", c.tuneArgs(model, engine, build, extra)...)
	return &t, err
}

// TuneArgv is the measure job: the sweeps on this machine's GPU, progress lines, then the page.
func (c Client) TuneArgv(model, engine, build string, extra ...string) []string {
	if argv := c.Demo.argv("tune"); argv != nil { // a demo plays the job; nothing runs
		return argv
	}
	return append(append([]string{c.Python, "-m", "boltbeam.workflow.screen"}, c.tuneArgs(model, engine, build, extra)...), "--measure")
}

// ReadTune reads the measure job's answer out of its log: the JSON object after the job's start line and its progress
// lines. A log with no answer (the job was stopped, or Python crashed) is an error naming that.
func ReadTune(lines []string) (*Tune, error) {
	start := 0
	for i, l := range lines {
		if strings.HasPrefix(l, "=== ") {
			start = i + 1
		}
	}
	for i := start; i < len(lines); i++ {
		if !strings.HasPrefix(lines[i], "{") {
			continue
		}
		var t Tune
		if json.NewDecoder(strings.NewReader(strings.Join(lines[i:], "\n"))).Decode(&t) != nil {
			continue
		}
		if t.Kind == "error" {
			return nil, &Error{Message: t.Error, Code: 1}
		}
		if t.Kind == "tune" {
			return &t, nil
		}
	}
	return nil, errors.New("the measurement ended without an answer")
}

// RoutingArgv is the Measure routing job: the engine serves the model once on the GPU and the experts its router picks
// are saved as a routing profile, where the offload page and Tuned for this machine look. A demo plays a recorded job.
func (c Client) RoutingArgv(model, engine, build string) []string {
	if c.Demo != nil {
		return c.Demo.noGPU("routing", "routing profile: demo")
	}
	argv := []string{c.Python, "-m", "boltbeam.cli", "routing", model, "--engine", engine}
	if build != "" { // routing resolves the build as tune does
		argv = append(argv, "--build", build)
	}
	return argv
}

// HostArgv is the Measure again job: this machine's RAM reads and copies to the GPU, measured again and saved.
func (c Client) HostArgv() []string {
	if c.Demo != nil {
		return c.Demo.noGPU("host", `"remeasure_failed": false`)
	}
	return []string{c.Python, "-m", "boltbeam.cli", "host", "--remeasure", "--json"}
}

// noGPU is a demo's job: the recorded command, else one that prints the line a successful job ends with. A demo never
// touches the GPU.
func (d *Demo) noGPU(job, success string) []string {
	if argv := d.argv(job); argv != nil {
		return argv
	}
	return []string{"echo", success}
}
