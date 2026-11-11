package seam

// A demo answers the seam from recorded JSON instead of Python: the screens draw the same facts with no GPU, no
// network and no interpreter. Only the VHS tapes use it (tui/tapes, BOLTBEAM_TUI_DEMO); the answers are the ones the
// Go screen tests feed (tui/testdata/expected and recorded), named in a demo file under tui/testdata/demo.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Demo is one demo file: what the screens open with, the answers by command, and the jobs alive at start.
type Demo struct {
	Model  string `json:"model"`
	Target string `json:"target"`
	// Answers maps a command's words to its answer: a file beside the demo file, or the JSON itself. A command is
	// answered by the longest key that starts with its name and whose other words appear in its arguments in order
	// (a path argument matches by its base name). An answer of kind "error" is Python's refusal, in its words.
	Answers map[string]json.RawMessage `json:"answers"`
	// Jobs are pipeline jobs shown alive at start: the run id and its log lines.
	Jobs map[string][]string `json:"jobs"`
	// Argv is what a job the screen starts runs in place of Python's command (the name of the job: "speedup"), so a
	// demo can play a start, a progress log and an end with nothing run but the shell.
	Argv map[string][]string `json:"argv"`
	dir  string
}

// LoadDemo reads a demo file; "base" names another demo file whose answers this one extends.
func LoadDemo(path string) (*Demo, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d struct {
		Demo
		Base string `json:"base"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	d.dir = filepath.Dir(path)
	if d.Base == "" {
		return &d.Demo, nil
	}
	base, err := LoadDemo(filepath.Join(d.dir, d.Base))
	if err != nil {
		return nil, err
	}
	for k, v := range d.Answers {
		base.Answers[k] = v
	}
	if d.Model != "" {
		base.Model = d.Model
	}
	if d.Target != "" {
		base.Target = d.Target
	}
	if d.Jobs != nil {
		base.Jobs = d.Jobs
	}
	if d.Argv != nil {
		base.Argv = d.Argv
	}
	return base, nil
}

// answer is the recorded answer for args, as call returns Python's.
func (d *Demo) answer(args []string) ([]byte, int, error) {
	best, words := "", -1
	for k := range d.Answers {
		if w := strings.Fields(k); len(w) > words && inOrder(w, args) {
			best, words = k, len(w)
		}
	}
	if words < 0 {
		return nil, 1, &Error{Message: "not recorded in this demo: " + strings.Join(args, " "), Code: 1}
	}
	raw := []byte(d.Answers[best])
	var file string
	if json.Unmarshal(raw, &file) == nil {
		var err error
		if raw, err = os.ReadFile(filepath.Join(d.dir, file)); err != nil {
			return nil, 1, &Error{Message: err.Error(), Code: 1}
		}
	}
	var refused struct{ Kind, Error string }
	if json.Unmarshal(raw, &refused) == nil && refused.Kind == "error" {
		return nil, 1, &Error{Message: refused.Error, Code: 1}
	}
	return raw, 0, nil
}

// inOrder: the first word is the command; the others appear after it in order.
func inOrder(words, args []string) bool {
	if len(args) == 0 || args[0] != words[0] {
		return false
	}
	i := 1
	for _, a := range args[1:] {
		if i < len(words) && (a == words[i] || filepath.Base(a) == words[i]) {
			i++
		}
	}
	return i == len(words)
}
