package seam

import (
	"encoding/json"
	"strconv"
	"strings"
)

// Apply mods: `screen mods --engine E --target T [--apply]` (workflow/mods.py). The listing is one call of a few
// seconds; --apply builds the chosen mods into a copy of the build (minutes) and records it, after which the provider
// list offers it as a fork of the engine's family. Python owns every row and sentence.

// Mods is the answer: the mods that fit the build here, each a row, and with --apply the build; Refused says why not.
type Mods struct {
	Schema  string     `json:"schema"`
	Kind    string     `json:"kind"`
	Engine  string     `json:"engine"`
	Fits    string     `json:"fits"`
	Chosen  []string   `json:"chosen"`
	Rows    []ModRow   `json:"rows"`
	Build   *ModsBuild `json:"build"`
	Refused *string    `json:"refused"`
	// Key is the modded build's key for the chosen set; AlreadyBuilt says that build is on disk (Use it, or Rebuild);
	// Reused says --apply offered it as it is instead of building it again.
	Key          string `json:"key"`
	AlreadyBuilt bool   `json:"already_built"`
	Reused       bool   `json:"reused"`
}

// ModRow is one mod: what it does in plain words, the switch that turns it on (nil: none, SwitchWhy says why), its
// output label in words, and whether it is chosen, and why or why not.
type ModRow struct {
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	About       string  `json:"about"`
	Switch      *string `json:"switch"`
	SwitchWhy   *string `json:"switch_why"`
	Output      string  `json:"output"`
	OutputWords string  `json:"output_words"`
	Chosen      bool    `json:"chosen"`
	Why         string  `json:"why"`
}

// ModsBuild is the modded build --apply made: its key (the fork the Engine picker offers), folder and status.
type ModsBuild struct {
	Key    string `json:"key"`
	Folder string `json:"folder"`
	Status string `json:"status"`
}

func modsArgs(engine, target string, apply, rebuild bool) []string {
	args := []string{"mods", "--engine", engine}
	if target != "" {
		args = append(args, "--target", target)
	}
	if apply {
		args = append(args, "--apply")
	}
	if rebuild {
		args = append(args, "--rebuild")
	}
	return args
}

// ExitRefused is `screen mods`' exit code when it refuses: its JSON still says why (refused), in Python's words.
const ExitRefused = 1

// Mods lists the mods that fit the engine's build here (apply false), or builds them into a copy (apply true; rebuild:
// even when that build is already on disk). A refusal (exit 1 with refused set) is an answer, not an error.
func (c Client) Mods(engine, target string, apply, rebuild bool) (*Mods, error) {
	var m Mods
	_, _, err := c.ModsRaw(&m, modsArgs(engine, target, apply, rebuild)...)
	return &m, err
}

// ModsRaw is one `screen mods` call decoded into m, with the bytes Python printed and its exit code: exit 1 is
// accepted when the body carries refused (agent mode prints it as it is); any other exit 1 is an error, its words
// Python's error when it printed one.
func (c Client) ModsRaw(m *Mods, args ...string) ([]byte, int, error) {
	raw, code, stderr, err := c.callStderr("boltbeam.workflow.screen", []int{ExitRefused}, args...)
	if err != nil {
		return nil, code, err
	}
	var said struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(raw, m) != nil || code == ExitRefused && m.Refused == nil {
		msg := "screen mods exited " + strconv.Itoa(code) + " without saying why (python -m boltbeam.workflow.screen " + strings.Join(args, " ") + ")"
		if json.Unmarshal(raw, &said) == nil && said.Error != "" {
			msg = said.Error
		}
		return nil, code, &Error{Message: msg, Stderr: stderr, Code: code}
	}
	return raw, code, nil
}
