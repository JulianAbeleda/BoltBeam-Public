package ui

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// A run brought home by `boltbeam fetch` says where from and when in the Saved runs list; a run made here reads as before.
func TestSavedRunsSayWhereFetchedRunsCameFrom(t *testing.T) {
	var sv seam.Saved
	if err := json.Unmarshal([]byte(`{"root":"/r","runs":[
	{"id":"a","model_id":"Qwen3-8B","target_id":"nvidia_sm90","provider":"llama.cpp","workload":"decode","saved_at":"2026-11-09T10:00:00","pct_of_limit":75.0,
	 "fetched":{"host":"203.0.113.7","chip":"nvidia_sm90","gpu":"NVIDIA H100 80GB HBM3","fetched_at":"2026-11-09T09:58:12Z"}},
	{"id":"b","model_id":"Qwen3-8B","target_id":"apple_m4","provider":"llama.cpp","workload":"decode","saved_at":"2026-11-08T10:00:00","pct_of_limit":66.0,"fetched":null}]}`), &sv); err != nil {
		t.Fatal(err)
	}
	acts := savedActions(Facts{Saved: &sv})
	if got := plain(acts[0].label); !strings.HasSuffix(got, "75% of limit · from 203.0.113.7, fetched 2026-11-09") {
		t.Fatalf("fetched run: %q", got)
	}
	if got := plain(acts[1].label); strings.Contains(got, "fetched") || !strings.HasSuffix(got, "66% of limit") {
		t.Fatalf("run made here: %q", got)
	}
}

// Without a host the chip names where it came from; the Run page's header carries the same words.
func TestFetchedWordsAndRunHeader(t *testing.T) {
	chip := "nvidia_sm120"
	if got := fetchedWords(&seam.Fetched{Chip: &chip, FetchedAt: "2026-11-01T00:00:00Z"}); got != "from nvidia_sm120, fetched 2026-11-01" {
		t.Fatalf("chip fallback: %q", got)
	}
	var run seam.Run
	if err := json.Unmarshal([]byte(`{"kind":"run","id":"h100-generic","model_id":"Qwen3-8B","target_id":"nvidia_sm90",
	 "fetched":{"host":"linuxbox","chip":"nvidia_sm90","gpu":null,"fetched_at":"2026-11-09T09:58:12Z"}}`), &run); err != nil {
		t.Fatal(err)
	}
	if got := title(pageRun, Facts{Run: &run}); got != "Run · h100-generic · from linuxbox, fetched 2026-11-09" {
		t.Fatalf("run header: %q", got)
	}
	run.Fetched = nil
	if got := title(pageRun, Facts{Run: &run}); got != "Run · h100-generic" {
		t.Fatalf("run made here: %q", got)
	}
}
