package ui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func sp(s string) *string   { return &s }
func fp(v float64) *float64 { return &v }
func bp(v bool) *bool       { return &v }
func ip(v int) *int         { return &v }
func plain(s string) string { return ansi.Strip(s) }
func hasLabel(acts []action, label string) bool {
	for _, a := range acts {
		if a.label == label {
			return true
		}
	}
	return false
}

// measuredRun is run-002 from the pinned contract: output done, every route still unmeasured.
func measuredRun(t *testing.T) *seam.Run {
	var run seam.Run
	load(t, "run-002", &run)
	return &run
}

// Step 5 offers the action only when Python says this machine can compare; otherwise it says what is missing
// and the one command that fixes it.
func TestCompareReadiness(t *testing.T) {
	f := Facts{Path: "m.gguf", Run: measuredRun(t)}
	f.Ready = &seam.CompareReady{Applies: true, Missing: sp("fork"), Message: sp("The tinygrad fork is not at /x."),
		Fix: sp("git clone -b exp https://github.com/JulianAbeleda/tinygrad-arkey /x")}
	if hasLabel(resultActions(f), "Compare kernels per role") || hasLabel(resultActions(f), "Time each role in tinygrad") {
		t.Fatal("a machine without the fork must not offer the actions")
	}
	body := plain(resultBody(f, 100))
	for _, want := range []string{"cannot time or compare kernels here: The tinygrad fork is not at /x.",
		"Fix: git clone -b exp https://github.com/JulianAbeleda/tinygrad-arkey /x", noKernelChoice[:24]} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in:\n%s", want, body)
		}
	}
	f.Ready = &seam.CompareReady{Applies: true, Ready: true}
	if !hasLabel(resultActions(f), "Compare kernels per role") {
		t.Fatal("a ready machine must offer the action")
	}
}

func TestReadCompare(t *testing.T) {
	p := seam.ReadCompare([]string{"=== start", "compare roles: 7", "role attn_kv Q4_K: search", "role attn_kv Q4_K: ab",
		"role attn_kv Q4_K: done refuted", "role attn_kv Q6_K: search"})
	if p.Total != 7 || p.Done["attn_kv Q4_K"] != "refuted" || p.Now != "attn_kv Q6_K" || p.Doing != "search" {
		t.Fatalf("%+v", p)
	}
	if p := seam.ReadCompare([]string{"compare failed: The tinygrad fork is not at /x."}); p.Failed == "" {
		t.Fatal("a failed line must be read")
	}
}

// While the job runs, step 5 shows n of N and a row per role; once it ends, the per-role table with times.
func TestCompareProgressAndTable(t *testing.T) {
	run := measuredRun(t)
	f := Facts{Path: "m.gguf", Run: run, Ready: &seam.CompareReady{Applies: true, Ready: true, Runtime: sp("tinygrad's Metal runtime")},
		CJob: &jobs.Job{ID: compareID(run.ID), Alive: true},
		CTail: []string{"compare roles: 7", "role attn_kv Q4_K: search", "role attn_kv Q4_K: done refuted",
			"role attn_kv Q6_K: ab"}}
	if mk, text := resultLine(f); mk != "run" || !strings.Contains(text, "1 of 7") || !strings.Contains(text, "attn_kv Q6_K timing the whole model") {
		t.Fatalf("progress line: %s %q", mk, text)
	}
	body := plain(resultBody(f, 100))
	for _, want := range []string{"ruled out", "timing the whole model…", "waiting", "Times are from tinygrad's Metal runtime, not llama.cpp."} {
		if !strings.Contains(body, want) {
			t.Errorf("progress body misses %q:\n%s", want, body)
		}
	}
	if !hasLabel(resultActions(f), "Stop comparing (x)") {
		t.Fatal("a running comparison must offer stop")
	}

	f.CJob.Alive = false
	for i := range run.Results.Routes {
		run.Results.Routes[i].Status = "blocked"
	}
	run.Results.Routes[0].Status = "refuted"
	run.Results.Routes[0].Compare = &seam.RouteCompare{Plan: sp("LOCAL 256"), SearchMedianNs: fp(711458),
		DefaultMedianNs: fp(1877625), Kernel: &seam.KernelAlone{PlanUs: 711.458, ReferenceUs: fp(1877.625), ModelUsPerCall: fp(346.2), FasterThanModel: bp(false)}, MeasuredCorrect: ip(13), Candidates: ip(13), Reason: sp("speed regression of -6.7%"),
		AB: &seam.AB{BaselineTokS: fp(15), CandidateTokS: fp(14), DeltaPct: fp(-6.6667), TokenMatch: bp(true), RouteBound: bp(true)}}
	body = plain(resultBody(f, 120))
	for _, want := range []string{"KERNEL CHOICE", "ruled out · LOCAL 256", "model 346 µs to plan 711 µs (slower)", "15.00 to 14.00 (-6.7%)",
		compareNote(f), "speed regression of -6.7%", "undecided"} {
		if !strings.Contains(body, want) {
			t.Errorf("table misses %q:\n%s", want, body)
		}
	}
	if strings.Contains(body, noKernelChoice[:24]) {
		t.Error("the empty line must go once a role was compared")
	}
}

// Step 5 is titled with the run's provider and its capture; another provider's table is labelled beside it.
func TestLossBodyNamesProviderAndCapture(t *testing.T) {
	method := "nsys"
	l := seam.Loss{Status: "modeled", LimitTokS: fp(19.2), LimitMs: fp(52.0), Provider: "llama.cpp",
		Capture: &seam.Capture{Method: &method}, RolesProvider: sp("llama.cpp"), Source: sp("measured in llama.cpp, captured with nsys"),
		Roles: []seam.RoleLoss{{Role: "ffn_down", Quant: "Q6_K", IdealMs: 8.27, ActualMs: 10.58, LostMs: 2.30, Share: 1}},
		Others: []seam.OtherLoss{{Provider: "tinygrad", Capture: seam.Capture{Method: sp("tinygrad-profile-events")},
			Roles: []seam.RoleLoss{{Role: "ffn_down", Quant: "Q6_K", IdealMs: 8.27, ActualMs: 9.0, LostMs: 0.73, Share: 1}}}}}
	body := plain(lossBody(l))
	for _, want := range []string{"Measured with llama.cpp · captured with nsys", "Where llama.cpp loses time, measured in llama.cpp, captured with nsys",
		"Beside it, tinygrad · tinygrad's own timing: 0.0 tokens per second.", "9.00"} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in:\n%s", want, body)
		}
	}
}

// Step 4 offers one start row per runtime this machine has; one it lacks is named with why, and starts nothing.
// Step 5 offers per-role time only where the run's provider has a capture here.
func TestProviderActions(t *testing.T) {
	f := Facts{Path: "m.gguf", Providers: &seam.Providers{Providers: []seam.ProviderRow{
		{Provider: "llama.cpp", Available: true, Capture: seam.Capture{Reason: sp("whole step only: metal-system-trace is not installed")}},
		{Provider: "tinygrad", Available: false, Reason: sp("The tinygrad fork is not at /x.")}}}}
	acts := measureActions(f)
	if !hasLabel(acts, "Measure m.gguf on  with llama.cpp") && !strings.Contains(acts[0].label, "with llama.cpp") {
		t.Fatalf("no llama.cpp start row: %+v", acts)
	}
	if acts[0].arg != "llama.cpp" || acts[1].do != "" || !strings.Contains(plain(acts[1].label), "tinygrad cannot measure here: The tinygrad fork is not at /x.") {
		t.Fatalf("rows: %+v", acts)
	}
	f.Run = measuredRun(t)
	if hasLabel(resultActions(f), "Time each role in llama.cpp") {
		t.Fatal("llama.cpp with no capture here must not offer per-role time")
	}
	body := plain(resultBody(f, 100))
	if !strings.Contains(body, "Not possible on") || !strings.Contains(body, "metal-system-trace is not installed") {
		t.Errorf("step 5 must say why per role is not possible here:\n%s", body)
	}
	method := "metal-system-trace"
	f.Providers.Providers[0].Capture = seam.Capture{Method: &method}
	if !hasLabel(resultActions(f), "Time each role in llama.cpp") {
		t.Fatal("llama.cpp with a capture here must offer per-role time")
	}
}

// The end result: ms per token lost against the limit, per runtime, and per role for tinygrad.
func TestLossBody(t *testing.T) {
	l := seam.Loss{Status: "modeled", LimitTokS: fp(19.2), LimitMs: fp(52.0),
		Runtimes: []seam.Runtime{{Provider: "llama.cpp", TokS: 17.9, Ms: 55.8, LostMs: 3.8, Note: "Per role for llama.cpp needs Instruments."},
			{Provider: "tinygrad", TokS: 15.2, Ms: 65.9, LostMs: 13.8, PerRole: true}},
		Roles:           []seam.RoleLoss{{Role: "ffn_down", Quant: "Q6_K", IdealMs: 8.27, ActualMs: 10.58, LostMs: 2.30, Share: 0.25, CallsPerToken: 18}},
		NotAttributedMs: fp(4.46), Source: sp("measured in tinygrad's Metal runtime")}
	body := plain(lossBody(l))
	for _, want := range []string{"The limit is 52.0 ms per token.", "llama.cpp: 17.9 tokens per second, 55.8 ms per token, 3.8 ms lost.",
		"needs Instruments", "tinygrad per role: 65.9 ms of GPU time per token, 13.8 ms lost.", "Measured with llama.cpp · roles not timed yet", "measured in tinygrad's Metal runtime",
		"feed-forward out", "8.27", "10.58", "2.30", "25%", "not attributed", "4.46"} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in:\n%s", want, body)
		}
	}
	l.Roles, l.Missing = nil, sp("apple_m3_10c names no tinygrad device")
	if !strings.Contains(plain(lossBody(l)), "Per role: apple_m3_10c names no tinygrad device") {
		t.Error("a missing per-role time must say why")
	}
}

// A broken floor is refused: the reason shows, the per-role numbers never do.
func TestLossRefusedShowsTheReasonOnly(t *testing.T) {
	l := seam.Loss{Status: "modeled", LimitTokS: fp(362), LimitMs: fp(2.76),
		Refused: sp("incomplete measurement: 1.40 ms of GPU time per token is below the limit's floor of 2.76 ms")}
	body := plain(lossBody(l))
	for _, want := range []string{"Per role: not shown.", "incomplete measurement", "below the limit's floor"} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in:\n%s", want, body)
		}
	}
	if strings.Contains(body, "IDEAL ms") || strings.Contains(body, "tinygrad:") {
		t.Errorf("a refused table must show no numbers:\n%s", body)
	}
}

// The footer names the runtime Python names: CUDA on the RTX 5090, never a fixed Metal sentence.
func TestRoleTimeNoteNamesTheRuntimeFromPython(t *testing.T) {
	run := measuredRun(t)
	f := Facts{Path: "m.gguf", Run: run, Ready: &seam.CompareReady{Ready: true, Runtime: sp("tinygrad's CUDA runtime")},
		CJob: &jobs.Job{ID: compareID(run.ID), Alive: true}, CTail: []string{"role-time: start"}}
	body := plain(resultBody(f, 100))
	if !strings.Contains(body, "Times are from tinygrad's CUDA runtime, not llama.cpp.") || strings.Contains(body, "Metal") {
		t.Errorf("the footer must name CUDA only:\n%s", body)
	}
}

// Step 5 suggests comparing kernels only where the target can compare them.
func TestNoCompareSuggestionWhereItCannotRun(t *testing.T) {
	f := Facts{Path: "m.gguf", Run: measuredRun(t), Ready: &seam.CompareReady{Ready: true, Applies: false,
		CompareMessage: sp("Comparing kernels per role runs on Metal only. This run is for CUDA.")}}
	body := plain(resultBody(f, 100))
	if strings.Contains(body, compareNext) || !strings.Contains(body, noKernelChoice) {
		t.Errorf("a CUDA run must not be told to compare kernels:\n%s", body)
	}
	f.Ready.Applies = true
	if body := plain(resultBody(f, 100)); !strings.Contains(body, compareNext) {
		t.Errorf("a Metal run keeps the next step:\n%s", body)
	}
}

// NVIDIA measures timing with llama-bench and has no probe: step 4 passes and says the probe is missing.
func TestMeasureTimingOnlyPassesAndNamesTheMissingProbe(t *testing.T) {
	run := measuredRun(t)
	run.Blocked = []seam.Need{{Need: "probe_evidence", Request: "probe_request.json"}}
	run.Measure = &seam.MeasureStatus{Status: "measured", Collector: sp("llama-bench-decode"), Probe: sp("absent"),
		ProbeReason: sp("no building-block probe for this chip")}
	f := Facts{Path: "m.gguf", Run: run}
	if mk, text := measureLine(f); mk != "pass" || !strings.Contains(plain(text), "timing ✓ · no probe for this chip") {
		t.Fatalf("measure line: %s %q", mk, plain(text))
	}
	rows := measureRows(run.Measure, nil, false)
	if len(rows) != 2 || rows[0] != [3]string{"crossed", "measure_probe", "not here: no building-block probe for this chip"} ||
		rows[1] != [3]string{"pass", "measure_timing", "done"} {
		t.Fatalf("rows: %v", rows)
	}
}

// On Linux the screen says "this machine"; "this Mac" only on macOS. The chip body labels recorded facts with
// their date and shows the live driver beside them.
func TestLinuxSaysThisMachineAndTheLiveDriver(t *testing.T) {
	old := goos
	defer func() { goos = old }()
	goos = "linux"
	f := Facts{Targets: &seam.Targets{Targets: []seam.Target{{ID: "nvidia_sm120", Backend: "CUDA", BackendStatus: "descriptor_only",
		Scope: sp("NVIDIA GeForce RTX 5090, driver 595.84"), ScopeObservedAt: sp("2026-09-23"), MemoryBandwidthGBs: fp(1693.3),
		MatrixTFLOPS: map[string]float64{"fp16": 237.1}, HasCeiling: true,
		FactStatus: seam.FactStatus{MemoryBandwidthGBs: "measurement", PeakTFLOPS: "measurement"}}}},
		ThisMachine: "nvidia_sm120", Driver: "595.99.02"}
	_, line := chipLine(f)
	acts := plain(chipActions(f)[0].label)
	body := plain(chipBody(f, 100))
	for _, got := range []string{line, acts, body, stageWord("measure")} {
		if strings.Contains(got, "Mac") {
			t.Errorf("Linux must not say Mac: %q", got)
		}
	}
	if !strings.Contains(line, "this machine") || stageWord("measure") != "Can this machine measure" {
		t.Errorf("line %q, stage %q", line, stageWord("measure"))
	}
	for _, want := range []string{"driver 595.84 · recorded 2026-09-23", "This machine now: driver 595.99.02, read live",
		"compute from measurement"} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in:\n%s", want, body)
		}
	}
	goos = "darwin"
	if _, line := chipLine(f); !strings.Contains(line, "this Mac") {
		t.Errorf("macOS keeps this Mac: %q", line)
	}
}
