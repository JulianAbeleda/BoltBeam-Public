package ui

import (
	"slices"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// Workload is the fourth Setup line: decode or prefill, picked; session is shown muted until it is built. The pick
// reaches the pipeline as --workload, and a prefill takes no --batch.
func TestWorkloadPicker(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := press(press(press(press(program(s, nil, nil, nil, 80, 24), "j"), "j"), "j"), "enter")
	if m.(Model).cursor != pageWorkload {
		t.Fatalf("enter on Workload opens its picker: %d", m.(Model).cursor)
	}
	if got := m.(Model).f.workload(); got != "decode" {
		t.Fatalf("decode until another is picked: %q", got)
	}
	golden(t, "picker-workload.txt", m.View())
	m = press(press(m, "j"), "enter") // prefill
	if got := m.(Model).f.workload(); got != "prefill" || m.(Model).cursor != pageSetup {
		t.Fatalf("workload %q, page %d", got, m.(Model).cursor)
	}
	if !strings.Contains(m.View(), "prefill") {
		t.Fatal("Setup shows the picked workload")
	}
	golden(t, "setup-prefill.txt", m.View())
	argv := strings.Join(seam.Client{Python: "py"}.PipelineArgv(seam.Pipeline{Model: "m", RunDir: "r", Target: "t", Workload: m.(Model).f.workload()}), " ")
	if !strings.Contains(argv, "--workload prefill") {
		t.Fatalf("argv %v", argv)
	}
}

// Replay is the fifth workload: its page lists the shapes and the servers running here, the Run waits for a server
// the user confirmed, the pick reaches the pipeline as --replay-trace and --server-argv, and the results print the
// table's own lines and cells.
func TestReplayWorkload(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	f := Facts{Workload: "replay", Replay: &seam.ReplayChoices{Default: "agent-60",
		Shapes:  []seam.ReplayShape{{Name: "agent-60", Label: "agent-60, 60 turns to 131K"}, {Name: "chat-300", Label: "chat-300, 300 turns to 105K"}},
		Servers: [][]string{{"/bin/llama-server", "--model", "m.gguf", "--port", "8080"}}}}
	if got := workloadValue(f); got != "replay (agent-60, 60 turns to 131K)" {
		t.Fatalf("setup line %q", got)
	}
	if !slices.Contains(f.missing(), "the server to replay on") {
		t.Fatalf("a replay waits for a confirmed server: %v", f.missing())
	}
	acts := workloadActions(f)
	var shape, server *action
	for i := range acts {
		if acts[i].do == "replayshape" && acts[i].arg == "chat-300" {
			shape = &acts[i]
		}
		if acts[i].do == "replayserver" {
			server = &acts[i]
		}
	}
	if shape == nil || server == nil || server.arg != "/bin/llama-server --model m.gguf --port 8080" {
		t.Fatalf("actions %v", acts)
	}
	f.ReplayShape, f.ReplayArgv = "chat-300", server.arg
	if slices.Contains(f.missing(), "the server to replay on") || workloadValue(f) != "replay (chat-300, 300 turns to 105K)" {
		t.Fatalf("missing %v, line %q", f.missing(), workloadValue(f))
	}
	argv := strings.Join(seam.Client{Python: "py"}.PipelineArgv(seam.Pipeline{Model: "m", RunDir: "r", Target: "t", Workload: "replay",
		ReplayTrace: "chat-300", ServerArgv: f.ReplayArgv}), " ")
	if !strings.Contains(argv, "--replay-trace chat-300") || !strings.Contains(argv, "--server-argv /bin/llama-server --model m.gguf") {
		t.Fatalf("argv %v", argv)
	}
	body := replayBody(&seam.Replay{Title: "The replay", HeadlineLines: []string{"Session at 61.0% of its roofline."},
		Columns: []string{"turn", "prefix"}, Rows: []seam.SessionRow{{Cells: []string{"1", "3,200"}}}, Total: []string{"warm", ""},
		Wheres: []seam.ReplayWhere{{Status: "unmeasured", Where: seam.Where{Words: "Turn 2 (early): unmeasured."}}}})
	for _, want := range []string{"Session at 61.0% of its roofline.", "3,200", "TURN", "Turn 2 (early): unmeasured."} {
		if !strings.Contains(body, want) {
			t.Fatalf("body lacks %q:\n%s", want, body)
		}
	}
	if replayBody(nil) != "" {
		t.Fatal("no replay, no body")
	}
}
