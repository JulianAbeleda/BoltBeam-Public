package ui

import (
	"encoding/json"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// The RTX 5090 smoke: the native CUDA probe read 1694.9 GB/s, the registry says 1693.3. With machine facts the
// chip line, Setup's speed limit and the limit page all say the measured number and where it came from.
func TestOneGpuShowsTheMeasuredBandwidthEverywhere(t *testing.T) {
	src := "measured on this GPU (native CUDA probe, 2026-10-09)"
	target := seam.Target{ID: "nvidia_sm120", HasCeiling: true, MemoryBandwidthGBs: fp(1693.3),
		FactStatus: seam.FactStatus{MemoryBandwidthGBs: "measurement", PeakTFLOPS: "measurement"}}
	f := Facts{Targets: &seam.Targets{Targets: []seam.Target{target}}, Detected: true, ThisMachine: "nvidia_sm120",
		Ceiling: &seam.Ceiling{Target: target, PeakBandwidthGBs: 1694.9, BandwidthSource: &src,
			Decode: seam.CeilingBlock{TokS: fp(362.44), FloorMs: 2.759}}}
	_, line := chipLine(f)
	setup := plain(setupBody(f, 100))
	limit := plain(limitBody(f, 100))
	chip := plain(chipBody(f, 100))
	for name, got := range map[string]string{"chip line": line, "setup": setup, "limit": limit, "chip": chip} {
		if !strings.Contains(got, "1694.9 GB/s") || strings.Contains(got, "1693.3") {
			t.Errorf("%s must use the measured 1694.9 GB/s, never the registry 1693.3:\n%s", name, got)
		}
	}
	for _, want := range []string{"memory 1694.9 GB/s, " + src, "Speed limit on", "362.4 tokens per second", "From memory 1694.9 GB/s, " + src} {
		if !strings.Contains(line+"\n"+setup, want) {
			t.Errorf("missing %q in:\n%s\n%s", want, line, setup)
		}
	}
	if !strings.Contains(limit, "memory speed "+src) {
		t.Errorf("limit page source:\n%s", limit)
	}
	// without machine facts the registry figure stays, as before
	f.Ceiling.BandwidthSource = nil
	if _, line := chipLine(f); !strings.Contains(line, "memory 1693.3 GB/s") {
		t.Errorf("registry line: %s", line)
	}
}

// One GPU: the run's limit names its read bandwidth and source in one line (it was hidden below 2 GPUs).
func TestOneGpuLayoutNamesItsInputAndSource(t *testing.T) {
	var l seam.LayoutLimit
	raw := `{"layout":"one","label":"one GPU","gpus":1,"ms":2.759,"tok_s":362.44,"formula":"weight bytes / the GPU's read bandwidth",
	  "inputs":[{"what":"GPU 0 NVIDIA GeForce RTX 5090 read bandwidth","value":1694.9,"unit":"GB/s",
	    "source":"measured on GPU 0 with BoltBeam's native CUDA read probe, best of 10 per launch shape, 2026-10-09"},
	   {"what":"weight bytes per token","value":4.676e9,"unit":"bytes","source":"the model's roofline (decode, context 1)"}]}`
	if err := json.Unmarshal([]byte(raw), &l); err != nil {
		t.Fatal(err)
	}
	got := plain(layoutBody(l))
	if want := "Limit from memory 1694.9 GB/s, measured on this GPU (native CUDA probe, 2026-10-09)"; strings.TrimSpace(got) != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	l.Inputs[0].Source = "registry figure for nvidia_sm120, not measured on this GPU (nvcc is not installed)"
	if got := plain(layoutBody(l)); !strings.Contains(got, "registry figure, not measured on this GPU") {
		t.Fatalf("registry: %q", got)
	}
	if got := shortSource("measured on this GPU with BoltBeam's Metal read probe, 2026-10-09"); got != "measured on this GPU (Metal read probe, 2026-10-09)" {
		t.Fatalf("metal: %q", got)
	}
}

// A run beside this one that never measured says why; 0.0 is never printed for absent data.
func TestBesideRunThatNeverMeasuredSaysWhy(t *testing.T) {
	why := "not measured: the GPU is busy: llama-server (pid 4242) holds it"
	l := seam.Loss{Status: "modeled", LimitTokS: fp(362.4), LimitMs: fp(2.759), Provider: "tinygrad",
		Missing: sp("no per-role time yet"),
		Others: []seam.OtherLoss{{Provider: "llama.cpp", Run: sp("qwen3-8b-q4-k-m-nvidia_sm120-001"), Missing: &why},
			{Provider: "llama.cpp", Run: sp("qwen3-8b-q4-k-m-nvidia_sm120-003"), TokS: fp(301.25)}}}
	body := plain(lossBody(l))
	if strings.Contains(body, "0.0 tokens per second") {
		t.Fatalf("0.0 printed for absent data:\n%s", body)
	}
	for _, want := range []string{"Beside it, llama.cpp (run 001)", ": " + why + ".", "(run 003)", ": 301.2 tokens per second."} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in:\n%s", want, body)
		}
	}
}

// A relaunch comes back with the engine the newest run on this chip measured with; a pick wins.
func TestEngineIsRememberedFromTheNewestRun(t *testing.T) {
	m := New(seam.Client{}, jobs.Store{}, "/models/Qwen3-8B.gguf", "", 512)
	m.f.Targets = &seam.Targets{Targets: []seam.Target{{ID: "nvidia_sm120", HasCeiling: true}}}
	tg := "tinygrad"
	runs := &seam.Runs{Runs: []seam.Summary{{ID: "a-001", TargetID: "nvidia_sm120"},
		{ID: "a-002", TargetID: "nvidia_sm120", Measure: &seam.MeasureStatus{Provider: &tg}}}}
	rows := &seam.Providers{Providers: []seam.ProviderRow{{Provider: "llama.cpp", Available: true}, {Provider: "tinygrad", Available: true}}}
	m, _ = m.update(runsMsg{runs: runs})
	m, _ = m.update(providersMsg{target: "nvidia_sm120", providers: rows})
	if m.f.Engine != "tinygrad" {
		t.Fatalf("engine %q, want tinygrad", m.f.Engine)
	}
	m.f.Engine, m.engineSet = "llama.cpp", true
	m, _ = m.update(runsMsg{runs: runs})
	if m.f.Engine != "llama.cpp" {
		t.Fatalf("a pick must win: %q", m.f.Engine)
	}
}

// At 110 columns the role table keeps µs/CALL and WHY in one table: the share bar and IDEAL go first.
func TestRoleTableKeepsWhyAt110(t *testing.T) {
	old := tableWidth
	defer func() { tableWidth = old }()
	roles := []seam.RoleLoss{{Role: "ffn_up", Quant: "Q4_K", IdealMs: 21.19, ActualMs: 25.08, LostMs: 3.89, Share: 0.29,
		PctPeak: fp(84.5), UsPerCall: fp(348.4), Reason: "too small to fill memory"},
		{Role: "output", Quant: "Q6_K", IdealMs: 5.30, ActualMs: 7.04, LostMs: 1.74, Share: 0.13, PctPeak: fp(75.3),
			UsPerCall: fp(7042.6), Reason: "slow kernel"}}
	for _, w := range []int{106, 76} {
		tableWidth = w
		got := plain(roleTable(roles, fp(1.2)))
		head := strings.SplitN(got, "\n", 2)[0]
		if !strings.Contains(got, "WHY") || !strings.Contains(got, "µs/CALL") || !strings.Contains(got, "too small to fill memory") {
			t.Errorf("width %d lost a column:\n%s", w, got)
		}
		if maxWidth(got) > w {
			t.Errorf("width %d: table is %d wide:\n%s", w, maxWidth(got), got)
		}
		if w == 106 && (!strings.Contains(head, "WHY") || strings.Count(got, "ROLE") != 1) {
			t.Errorf("at %d one table with WHY:\n%s", w, got)
		}
	}
}
