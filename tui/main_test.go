package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestUnknownRoleTimeRefusedBeforeJobStarts(t *testing.T) {
	var out, err bytes.Buffer
	code := run([]string{"--json", "start", "missing.gguf", "--target", "test", "--role-time", "typo"}, &out, &err)
	if code != 2 || !strings.Contains(err.String(), "unknown --role-time") {
		t.Fatalf("code %d: %s %s", code, out.String(), err.String())
	}
}

// fakePython is an interpreter that prints its arguments as one JSON object, so agent mode's argv can be read back.
func fakePython(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "python")
	script := "#!/bin/sh\nprintf '{\"argv\": \"%s\"}\\n' \"$*\"\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// Agent mode has mods, tune, offload, recommend and doctor: each prints what Python printed, the argv built as the
// screens build it (the model made absolute, the global --target added when not given, the run folders' facts).
func TestJSONModsTuneOffloadRecommendDoctor(t *testing.T) {
	py, root := fakePython(t), t.TempDir()
	for _, c := range []struct {
		args []string
		want []string
	}{
		{[]string{"mods", "--engine", "llama.cpp"}, []string{"-m boltbeam.workflow.screen mods --engine llama.cpp --target chip9"}},
		{[]string{"mods", "--apply", "--target", "x"}, []string{"mods --apply --target x\""}},
		{[]string{"tune", "m.gguf", "--engine", "llama.cpp"}, []string{"tune /", "m.gguf --engine llama.cpp --target chip9 --facts-root " + root}},
		{[]string{"offload", "m.gguf"}, []string{"offload /", "m.gguf --target chip9 --facts-root"}},
		{[]string{"recommend", "m.gguf", "--target", "y", "--engine", "vllm"}, []string{"recommend /", "m.gguf --target y --engine vllm --facts-root"}},
		{[]string{"doctor"}, []string{"-m boltbeam.cli doctor --json"}},
	} {
		var out, errb bytes.Buffer
		code := run(append([]string{"--json", "--python", py, "--repo", root, "--root", root, "--target", "chip9"}, c.args...), &out, &errb)
		if code != 0 {
			t.Fatalf("%v: code %d %s %s", c.args, code, out.String(), errb.String())
		}
		for _, w := range c.want {
			if !strings.Contains(out.String(), w) {
				t.Errorf("%v: want %q in %s", c.args, w, out.String())
			}
		}
	}
	var out, errb bytes.Buffer
	if code := run([]string{"--json", "--python", py, "--repo", root, "offload", "m.gguf"}, &out, &errb); code != 2 || !strings.Contains(errb.String(), "offload needs MODEL and --target") {
		t.Fatalf("offload with no chip is a usage error: %d %s", code, errb.String())
	}
}

// scriptPython is an interpreter that prints body and exits with code, whatever it is asked.
func scriptPython(t *testing.T, body string, code int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "python")
	script := "#!/bin/sh\ncat <<'X'\n" + body + "\nX\nexit " + strconv.Itoa(code) + "\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// `mods` refusing exits 1 with its reason in refused: agent mode prints Python's sentence and exits 1, not "exit
// status 1"; an exit 1 that says nothing is an error that says so.
func TestJSONModsRefusalIsPrinted(t *testing.T) {
	root := t.TempDir()
	py := scriptPython(t, `{"schema": "boltbeam.tui.v1", "kind": "mods", "engine": "vllm", "refused": "no applier is wired for vllm mods yet"}`, 1)
	var out, errb bytes.Buffer
	if code := run([]string{"--json", "--python", py, "--repo", root, "--root", root, "mods", "--engine", "vllm"}, &out, &errb); code != 1 ||
		!strings.Contains(out.String(), "no applier is wired for vllm mods yet") || strings.Contains(out.String(), "exit status") {
		t.Fatalf("code %d: %s", code, out.String())
	}
	out.Reset()
	py = scriptPython(t, `{"schema": "boltbeam.tui.v1", "kind": "error", "error": "unknown target 'x'"}`, 1)
	if code := run([]string{"--json", "--python", py, "--repo", root, "--root", root, "mods", "--target", "x"}, &out, &errb); code != 1 ||
		!strings.Contains(out.String(), `"kind":"error"`) || !strings.Contains(out.String(), "unknown target 'x'") {
		t.Fatalf("an error is still an error, in Python's words: %d %s", code, out.String())
	}
}

// Agent mode's path flags reach Python absolute against the caller's directory (the seam runs in the checkout); a
// bare --build that is no file here (a program on PATH), and a ~ path, are left as they are.
func TestJSONPathFlagsAbsolute(t *testing.T) {
	py, root := fakePython(t), t.TempDir()
	here, _ := os.Getwd()
	for _, c := range []struct {
		args []string
		want []string
	}{
		{[]string{"mods", "--src", "src/llama.cpp"}, []string{"--src " + filepath.Join(here, "src/llama.cpp")}},
		{[]string{"mods", "--src=rel"}, []string{"--src=" + filepath.Join(here, "rel")}},
		{[]string{"tune", "m.gguf", "--engine", "llama.cpp", "--build", "./b/llama-bench", "--draft", "d/h.gguf"},
			[]string{"--build " + filepath.Join(here, "b/llama-bench"), "--draft " + filepath.Join(here, "d/h.gguf")}},
		{[]string{"tune", "m.gguf", "--engine", "llama.cpp", "--build", "llama-bench", "--draft", "~/h.gguf"}, []string{"--build llama-bench ", "--draft ~/h.gguf"}},
		{[]string{"offload", "--profile", "p/x.json"}, []string{"offload --profile " + filepath.Join(here, "p/x.json")}},
		{[]string{"offload", "m.gguf", "--draft", "h.gguf", "--profile", "p/x.json"}, []string{"--draft " + filepath.Join(here, "h.gguf"), "--profile " + filepath.Join(here, "p/x.json")}},
	} {
		var out, errb bytes.Buffer
		if code := run(append([]string{"--json", "--python", py, "--repo", root, "--root", root, "--target", "chip9"}, c.args...), &out, &errb); code != 0 {
			t.Fatalf("%v: code %d %s %s", c.args, code, out.String(), errb.String())
		}
		for _, w := range c.want {
			if !strings.Contains(out.String(), w) {
				t.Errorf("%v: want %q in %s", c.args, w, out.String())
			}
		}
	}
}
