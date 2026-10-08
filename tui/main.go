// boltbeam-tui: BoltBeam on a screen, or as JSON.
//
//	boltbeam-tui                      the Bubble Tea screens (needs a terminal)
//	boltbeam-tui --json <command>     the same facts as one JSON object; exit 0 ok, 1 error, 2 usage, 3 no measurement
//
// Both modes read through the same seam (python -m boltbeam.workflow.screen, plus `boltbeam inspect`). Go never
// writes a run artifact; `start` runs the seam's pipeline as a detached child and `stop` sends it SIGTERM.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/ui"
)

const usage = `usage: boltbeam-tui [--json] [--root RUNS] [--repo DIR] [--python PY] [--state DIR]
                    [--model FILE] [--target ID] [--context N] [command]

commands (each prints one JSON object):
  targets                                      the chips BoltBeam knows, and which carry a speed limit
  inspect MODEL                                the model profile: roles, shapes, quants (boltbeam inspect, bytes unchanged)
  ceiling MODEL --target ID [--context N]      the roofline: best tokens/s for decode (one token) and prefill (N tokens)
  runs                                         one summary per run folder
  run <id>                                     the stages, what is blocked, the next step, the results
  results <id>                                 what won per role and the timing against the ceiling (exit 3: nothing measured)
  start MODEL --target ID [--workload W] [--id RUN] [--probe FILE] [--timing FILE]
                                               run the pipeline in the background; log under --state
  stop <id>                                    SIGTERM the pipeline started here
  tail <id> [--lines N]                        the pipeline's status and the last lines of its log

env: BOLTBEAM_RUNS (runs folder), BOLTBEAM_PYTHON (interpreter; default: the first of python3.13..3.10, python3),
     BOLTBEAM_REPO (checkout; default: walk up from here), BOLTBEAM_TUI_STATE (pids and logs),
     BOLTBEAM_MODEL and BOLTBEAM_TARGET (what the screens open with).`

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// findRepo walks up from dir to the folder holding boltbeam/__init__.py.
func findRepo(dir string) string {
	for {
		if _, err := os.Stat(filepath.Join(dir, "boltbeam", "__init__.py")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// findPython picks the newest interpreter on PATH: BoltBeam refuses anything below 3.10 and a stock macOS
// names a 3.9 `python3`.
func findPython() string {
	for _, name := range []string{"python3.13", "python3.12", "python3.11", "python3.10", "python3"} {
		if _, err := exec.LookPath(name); err == nil {
			return name
		}
	}
	return "python3"
}

func emit(out io.Writer, raw []byte) int {
	fmt.Fprintln(out, string(raw))
	return 0
}

func fail(out io.Writer, err error) int {
	var detail string
	if e, ok := err.(*seam.Error); ok && e.Stderr != "" {
		detail = e.Stderr
	}
	raw, _ := json.Marshal(map[string]string{"schema": "boltbeam.tui.v1", "kind": "error", "error": err.Error(), "stderr": detail})
	fmt.Fprintln(out, string(raw))
	return 1
}

func run(args []string, out, errOut io.Writer) int {
	fs := flag.NewFlagSet("boltbeam-tui", flag.ContinueOnError)
	fs.SetOutput(errOut)
	jsonMode := fs.Bool("json", false, "print JSON (any command does; this also refuses the screens)")
	root := fs.String("root", os.Getenv("BOLTBEAM_RUNS"), "the runs folder")
	repo := fs.String("repo", os.Getenv("BOLTBEAM_REPO"), "the BoltBeam checkout")
	python := fs.String("python", envOr("BOLTBEAM_PYTHON", findPython()), "the interpreter that imports boltbeam")
	state := fs.String("state", jobs.DefaultDir(), "where pipeline pids and logs live")
	model := fs.String("model", os.Getenv("BOLTBEAM_MODEL"), "the model file the screens open with")
	target := fs.String("target", os.Getenv("BOLTBEAM_TARGET"), "the chip the screens open with")
	context := fs.Int("context", 512, "prefill tokens for the ceiling")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *repo == "" {
		cwd, _ := os.Getwd()
		*repo = findRepo(cwd)
		if *repo == "" {
			if exe, err := os.Executable(); err == nil {
				*repo = findRepo(filepath.Dir(exe))
			}
		}
	}
	if *repo == "" {
		// no checkout around: the package must then be importable on its own (pip install -e .)
		if err := exec.Command(*python, "-c", "import boltbeam").Run(); err != nil {
			fmt.Fprintf(errOut, "no BoltBeam checkout found and %s cannot import boltbeam: pass --repo, set BOLTBEAM_REPO, or run from the checkout\n", *python)
			return 2
		}
	}
	if *root == "" {
		if *repo != "" {
			*root = filepath.Join(*repo, "runs")
		} else {
			*root = "runs"
		}
	}
	client := seam.Client{Python: *python, Repo: *repo, Root: *root}
	store := jobs.Store{Dir: *state}
	rest := fs.Args()
	if len(rest) == 0 {
		if *jsonMode {
			fmt.Fprintln(errOut, usage)
			return 2
		}
		if err := ui.Start(client, store, *model, *target, *context); err != nil {
			fmt.Fprintln(errOut, err)
			return 1
		}
		return 0
	}
	return command(client, store, rest, out, errOut)
}

func command(client seam.Client, store jobs.Store, rest []string, out, errOut io.Writer) int {
	name, args := rest[0], rest[1:]
	need := func(n int) bool {
		if len(args) < n {
			fmt.Fprintln(errOut, usage)
			return false
		}
		return true
	}
	switch name {
	case "targets":
		_, raw, err := client.Targets()
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	case "inspect":
		if !need(1) {
			return 2
		}
		_, raw, err := client.Inspect(args[0])
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	case "ceiling":
		if !need(1) {
			return 2
		}
		fs := flag.NewFlagSet("ceiling", flag.ContinueOnError)
		fs.SetOutput(errOut)
		target := fs.String("target", "", "chip id")
		context := fs.Int("context", 0, "prefill tokens")
		if fs.Parse(args[1:]) != nil || *target == "" {
			fmt.Fprintln(errOut, "ceiling needs MODEL and --target")
			return 2
		}
		_, raw, err := client.Ceiling(args[0], *target, *context)
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	case "runs":
		_, raw, err := client.List()
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	case "run":
		if !need(1) {
			return 2
		}
		_, raw, err := client.Show(args[0])
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	case "results":
		if !need(1) {
			return 2
		}
		_, raw, code, err := client.Results(args[0])
		if err != nil {
			return fail(out, err)
		}
		emit(out, raw)
		return code
	case "start":
		if !need(1) {
			return 2
		}
		fs := flag.NewFlagSet("start", flag.ContinueOnError)
		fs.SetOutput(errOut)
		p := seam.Pipeline{Model: args[0]}
		fs.StringVar(&p.Target, "target", "", "chip id")
		fs.StringVar(&p.Workload, "workload", "decode", "decode or prefill")
		fs.StringVar(&p.ID, "id", "", "profile/model id override")
		fs.StringVar(&p.Probe, "probe", "", "probe_evidence.v1 JSON to ingest")
		fs.StringVar(&p.Timing, "timing", "", "timing_trace.v1 JSON to ingest")
		runID := fs.String("run", "", "run folder name (default: <model>-<chip>-NNN)")
		if fs.Parse(args[1:]) != nil || p.Target == "" {
			fmt.Fprintln(errOut, "start needs MODEL and --target")
			return 2
		}
		// the pipeline runs in the checkout, so the caller's relative paths are made absolute here
		for _, path := range []*string{&p.Model, &p.Probe, &p.Timing} {
			if *path != "" {
				if abs, err := filepath.Abs(*path); err == nil {
					*path = abs
				}
			}
		}
		if *runID == "" {
			runs, _, err := client.List()
			if err != nil {
				return fail(out, err)
			}
			*runID = seam.NextName(runs, ui.RunStem(p.Model, p.Target))
		}
		dir, err := client.RunDir(*runID)
		if err != nil {
			return fail(out, err)
		}
		p.RunDir = dir
		job, err := store.Start(*runID, client.Repo, client.PipelineArgv(p))
		if err != nil {
			return fail(out, err)
		}
		raw, _ := json.Marshal(job)
		return emit(out, raw)
	case "stop":
		if !need(1) {
			return 2
		}
		job, err := store.Stop(args[0])
		if err != nil {
			return fail(out, err)
		}
		raw, _ := json.Marshal(job)
		return emit(out, raw)
	case "tail":
		if !need(1) {
			return 2
		}
		fs := flag.NewFlagSet("tail", flag.ContinueOnError)
		fs.SetOutput(errOut)
		lines := fs.Int("lines", 30, "lines of log")
		if fs.Parse(args[1:]) != nil {
			return 2
		}
		job, err := store.Status(args[0])
		if err != nil {
			return fail(out, fmt.Errorf("no pipeline was started for %s from this machine", args[0]))
		}
		tail, _ := store.Tail(args[0], *lines)
		raw, _ := json.Marshal(map[string]any{"schema": "boltbeam.tui.v1", "kind": "job", "job": job, "lines": tail,
			"stages": seam.StageEvents(tail)})
		return emit(out, raw)
	}
	fmt.Fprintln(errOut, usage)
	return 2
}
