package ui

import (
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func fv(v float64) *float64 { return &v }

// sampleRecommend is `screen recommend` on the 27B AWQ on one 5090 before any engine start (an upper bound).
func sampleRecommend() *seam.Recommend {
	cmd := "vllm serve /m --max-model-len auto --max-num-seqs 1 --kv-cache-dtype fp8"
	r := &seam.Recommend{Provider: "vllm", GpuCount: 1, ContextIsUpperBound: true, KVDtype: "fp8", KVFormats: []string{"bf16", "fp8"}, Seqs: 1, Command: &cmd,
		Arkey: []string{"--max-num-seqs", "1", "--kv-cache-dtype", "fp8"}}
	r.Layouts = []seam.RecLayout{{ID: "one", Label: "one GPU", Available: true, MaxContext: map[string]seam.RecFig{
		"fp8": {Value: fv(262144), Formula: "(usable - weights) / KV", Kind: "upper bound"}}}}
	r.Layouts[0].Fits.Value = true
	id := "one"
	r.Recommended.ID, r.Recommended.Why = &id, "one GPU"
	flag := "--kv-cache-dtype fp8"
	r.Settings = []seam.RecSetting{{Flag: &flag, Recommended: true, Changes: "KV / 2", When: "short"}}
	return r
}

func TestRecommendedSetupLineAndPage(t *testing.T) {
	f := Facts{}
	if got := plain(recommendValue(f)); got != recommendHint {
		t.Fatalf("before numbers the line is the hint: %q", got)
	}
	f.Recommend = sampleRecommend()
	if got := recommendValue(f); !strings.Contains(got, "up to context 262,144") {
		t.Fatalf("an upper bound says so: %q", got)
	}
	body := plain(recommendBody(f, 80))
	for _, want := range []string{"upper bounds", "Max context fp8", "262,144 tokens", "--kv-cache-dtype fp8", "use one"} {
		if !strings.Contains(body, want) {
			t.Fatalf("page lacks %q:\n%s", want, body)
		}
	}
	acts := recommendActions(f)
	if len(acts) != 3 || acts[0].arg != "command" || acts[1].arg != "arkey" || acts[2].do != "reccheck" {
		t.Fatalf("actions: %+v", acts)
	}
}

func TestCopiedArkeyArgsHoldNoReservedFlag(t *testing.T) {
	f := Facts{Recommend: sampleRecommend()}
	for _, bad := range []string{"--model", "--host", "--port", "--served-model-name", "--max-model-len", "-m"} {
		for _, a := range strings.Fields(f.copyOf("arkey")) {
			if a == bad || strings.HasPrefix(a, bad+"=") {
				t.Fatalf("Arkey owns %s", bad)
			}
		}
	}
	if !strings.HasPrefix(f.copyOf("command"), "vllm serve ") {
		t.Fatal("the command line is copied whole")
	}
}

func TestFirstStartContextAndWarmOneAfterIt(t *testing.T) {
	f := Facts{Recommend: sampleRecommend()}
	r, note, cold := f.Recommend, "Contexts are for a warm compile cache, the only one measured", "cold"
	r.ContextIsUpperBound, r.OverheadNote = false, &note
	r.Layouts[0].MaxContext["fp8"] = seam.RecFig{Value: fv(250880), Kind: "derived"}
	r.Layouts[0].MaxContextWarm = map[string]seam.RecFig{"fp8": {Value: fv(262144), Kind: "derived"}}
	r.Check = &seam.RecCheck{Before: seam.RecCompare{State: &cold}}
	body := plain(recommendBody(f, 80))
	for _, want := range []string{"for a warm compile cache", "250,880 tokens", "after first start", "262,144 tokens", "a cold compile cache start"} {
		if !strings.Contains(body, want) {
			t.Fatalf("page lacks %q:\n%s", want, body)
		}
	}
}
