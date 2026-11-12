package ui

import (
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func sideRunning(kind, id string) SideJob {
	return SideJob{GPUJob: GPUJob{Job: &jobs.Job{ID: id, Alive: true}}, Kind: kind, ID: id}
}

// Measure routing is on the Offload page where the screen offers it (experts to place, an engine that captures), and
// not elsewhere; while it runs the row is progress and Stop.
func TestMeasureRoutingOnOffloadPage(t *testing.T) {
	f := Facts{Offload: recordedOffload(t, "offload.json"), OffKey: "k", Path: "/m/a.gguf", Engine: "llama.cpp"}
	f.Offload.Routing = false
	if got := rows(offloadActions(f)); len(got) != 1 || got[0] != "page|[ Back to setup ]" {
		t.Fatalf("not offered: %v", got)
	}
	f.Offload.Routing = true
	got := rows(offloadActions(f))
	if len(got) != 2 || !strings.HasPrefix(got[0], "measurerouting|[ Measure routing ] Finds which experts") {
		t.Fatalf("offered: %v", got)
	}
	f.Side = sideRunning("routing", "j")
	got = rows(offloadActions(f))
	if !strings.Contains(got[0], "Measuring which experts the model uses") || got[1] != "sidestop|[ Stop ]" {
		t.Fatalf("running: %v", got)
	}
	f.Side.Kind = "host"
	if got = rows(offloadActions(f)); strings.Contains(got[0], "Measuring") {
		t.Fatalf("another page's job is not this page's progress: %v", got)
	}
}

// ended feeds the model the end of the job it saw start at t1, with this log.
func ended(m Model, kind string, tail ...string) Model {
	id := "j"
	m.f.Side = SideJob{GPUJob: GPUJob{Seen: "t1", Stopped: m.f.Side.Stopped}, Kind: kind, ID: id}
	m, _, _ = m.sideMsg(sideJobMsg{id, &jobs.Job{ID: id, StartedAt: "t1", LogPath: "/log"}, tail})
	return m
}

// A measure job succeeded only if its log says so: Stop, a crash, a refusal and a remeasure that exited 0 but kept the
// old figure all say so in words; only the success line says it was measured.
func TestMeasureSucceedsOnlyOnItsOwnLine(t *testing.T) {
	m0 := Model{f: Facts{Path: "/m/a.gguf", Engine: "llama.cpp", OffKey: "k", RecKey: "r", Tune: TuneState{Key: "t"}}}
	failed := "Routing was not measured. The log: /log"
	for _, c := range []struct {
		kind, want string
		tail       []string
	}{
		{"routing", "Routing measured; the plan now uses it.", []string{"=== t start", "routing profile: /p.json", "  source: x"}},
		{"routing", "the model has no expert tensors", []string{"routing: the model has no expert tensors"}},
		{"routing", failed, []string{"=== t start", "Traceback (most recent call last):"}},
		{"routing", failed, []string{"=== t start"}},                                            // killed: no line at all
		{"routing", failed, []string{"routing profile: /old.json", "=== t start", "Traceback"}}, // an earlier run's success
		{"host", "This machine was measured again.", []string{"=== t start", "{", `  "remeasure_failed": false`, "}"}},
		{"host", "This machine was not measured. The log: /log", []string{`  "remeasure_failed": true`}}, // exit 0, saved figure kept
	} {
		if m := ended(m0, c.kind, c.tail...); m.note != c.want {
			t.Errorf("%s %v: note %q, want %q", c.kind, c.tail, m.note, c.want)
		}
	}
	for kind, argv := range map[string][]string{"routing": seam.Client{Demo: &seam.Demo{}}.RoutingArgv("/m", "e", ""), "host": seam.Client{Demo: &seam.Demo{}}.HostArgv()} {
		if ok, _ := sideDone(kind, argv[1:]); !ok { // a demo's job ends as a real one does
			t.Errorf("%s: a demo's job is not read as measured: %v", kind, argv)
		}
	}
	m := m0
	m.f.Side.Stopped = true
	if m = ended(m, "host", `"remeasure_failed": false`); m.note != "Stopped." || m.f.RecKey != "" || m.f.OffKey != "" || m.f.Tune.Key != "" {
		t.Fatalf("Stop sticks, and a stopped remeasure may have saved a figure, so the readers are asked again: %q %+v", m.note, m.f)
	}
	if w := vocab.Measure; !strings.Contains(w.Busy, "{id}") || !strings.Contains(w.StartFailed, "{why}") || !strings.Contains(w.StopFailed, "{why}") {
		t.Fatalf("the job messages are Python's words: %+v", w)
	}
}

// Stop is remembered on the job, so the end that follows it is not reported as a failure or a success.
func TestStopIsRememberedOnTheJob(t *testing.T) {
	m := Model{f: Facts{Side: sideRunning("host", "measure-host"), Tune: TuneState{GPUJob: GPUJob{Job: &jobs.Job{ID: "x"}}}}}
	next, _ := m.Update(gpuStoppedMsg{"measure-host", "Stopped."})
	mm := next.(Model)
	if !mm.f.Side.Stopped || mm.note != "Stopped." || mm.f.Tune.Stopped {
		t.Fatalf("stopped: side %v tune %v note %q", mm.f.Side.Stopped, mm.f.Tune.Stopped, mm.note)
	}
}

// What a job measured is read again: routing by the offload page and Tuned for this machine, the copy rate by the
// recommendation, the offload page and Tuned for this machine, and the checks of This machine run again.
func TestMeasureEndedAsksTheReadersAgain(t *testing.T) {
	m0 := Model{f: Facts{Path: "/m/a.gguf", Engine: "llama.cpp", OffKey: "k", RecKey: "r", Tune: TuneState{Key: "t"}}}
	m := ended(m0, "routing", "routing profile: /p.json")
	if m.f.OffKey != "" || m.f.Tune.Key != "" || m.f.RecKey != "r" {
		t.Fatalf("routing: off %q tune %q rec %q", m.f.OffKey, m.f.Tune.Key, m.f.RecKey)
	}
	m = ended(m0, "host", `"remeasure_failed": false`)
	if m.f.OffKey != "" || m.f.Tune.Key != "" || m.f.RecKey != "" || !m.f.DoctorBusy {
		t.Fatalf("host: %+v", m.f)
	}
}

// Measure again sits beside Check again.
func TestMeasureAgainOnMachinePage(t *testing.T) {
	got := rows(doctorActions(Facts{}))
	if len(got) != 3 || got[0] != "doctor|[ Check again ]" || !strings.HasPrefix(got[1], "measurehost|[ Measure again ] Times RAM") {
		t.Fatalf("%v", got)
	}
}

// Another job's read is not this job's: it changes nothing and ends nothing.
func TestAnotherJobsReadIsIgnored(t *testing.T) {
	m := Model{f: Facts{OffKey: "k", Side: SideJob{GPUJob: GPUJob{Seen: "t1"}, Kind: "routing", ID: "j"}}}
	m, _, _ = m.sideMsg(sideJobMsg{"other", &jobs.Job{ID: "other", StartedAt: "t1"}, []string{"routing profile: /p"}})
	if m.note != "" || m.f.OffKey != "k" || m.f.Side.Job != nil || m.f.Side.Ended != "" {
		t.Fatalf("%+v %q", m.f.Side, m.note)
	}
}
