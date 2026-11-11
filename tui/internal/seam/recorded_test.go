package seam

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// recorded reads one answer recorded from the Python side (tui/testdata/recorded).
func recorded(t *testing.T, name string, v any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tui", "testdata", "recorded", name))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}

// `boltbeam doctor --json` from a Mac decodes into Doctor: every row, its three statuses, each gap's fix.
func TestDoctorDecodesARecordedMac(t *testing.T) {
	var d Doctor
	recorded(t, "doctor-mac.json", &d)
	if len(d.Rows) != 13 || d.Gaps != 1 || d.Ready {
		t.Fatalf("decoded: %d rows, %d gaps, ready %v", len(d.Rows), d.Gaps, d.Ready)
	}
	by := map[string]DoctorRow{}
	for _, r := range d.Rows {
		by[r.Item] = r
	}
	if r := by["python"]; r.Status() != "found" || r.Fix != nil || r.Detail == "" {
		t.Fatalf("python: %+v", r)
	}
	if r := by["vLLM"]; r.Status() != "n/a" || r.Found != nil || r.Group != "engines" {
		t.Fatalf("vLLM: %+v", r)
	}
	// an engine family's forks not here are never a gap
	if r := by["llama.cpp"]; r.Status() != "found" || strings.Join(r.ForksMissing, ",") != "Ollama,Strata" {
		t.Fatalf("llama.cpp: %+v", r)
	}
	if r := by["boltbeam-tui"]; r.Status() != "not found" || r.Fix == nil || r.Blocks || r.Group != "boltbeam" {
		t.Fatalf("boltbeam-tui: %+v", r)
	}
}

// `boltbeam fetch DEST --json` (a listing) and `--all` (a fetch: one run fetched, one refused) decode into RemoteRuns.
func TestFetchAnswersDecode(t *testing.T) {
	var l RemoteRuns
	recorded(t, "fetch-list.json", &l)
	if l.Dest != "root@203.0.113.7" || l.RemoteRoot != "/root/boltbeam/runs" || len(l.Runs) != 2 || len(l.Fetched) != 0 ||
		l.Runs[1].ID != "h100-generic" || l.Runs[1].Where != ".work/h100-generic" || *l.Runs[0].GPU != "NVIDIA H100 80GB HBM3" ||
		l.Runs[0].Bytes != 232 || l.Runs[0].LeftBehindBytes != 4107 || l.Runs[0].State != "new" || len(l.Lines) != 0 {
		t.Fatalf("listing: %+v", l)
	}
	var d RemoteRuns
	recorded(t, "fetch-done.json", &d)
	if len(d.Fetched) != 2 || d.Fetched[0].Status != "fetched" || d.Fetched[0].Files != 5 ||
		d.Fetched[0].Dir != "/home/u/runs/saved/h100-in-model" || d.Fetched[1].Status != "refused" ||
		!strings.Contains(d.Fetched[1].Why, "not overwritten") || d.Runs[1].State != "conflict" {
		t.Fatalf("fetch: %+v", d.Fetched)
	}
}

// The fetch job's answer is read from its log after the job's start line; a refusal is an Error in Python's words,
// and a log with no answer says so.
func TestReadFetchFromTheJobLog(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tui", "testdata", "recorded", "fetch-done.json"))
	if err != nil {
		t.Fatal(err)
	}
	old := `{"kind": "fetch", "dest": "old", "runs": [], "fetched": []}`
	log := append([]string{"=== 2026-11-10T10:00:00Z start python -m boltbeam.cli fetch old", old,
		"=== 2026-11-10T11:00:00Z start python -m boltbeam.cli fetch root@203.0.113.7 --all"}, strings.Split(string(raw), "\n")...)
	got, err := ReadFetch(log)
	if err != nil || got.Dest != "root@203.0.113.7" || len(got.Fetched) != 2 {
		t.Fatalf("the last job's answer: %+v %v", got, err)
	}
	refused := []string{"=== start", `{"schema": "boltbeam.fetched.v1", "kind": "error", "error": "ssh to nobody@127.0.0.1 failed (exit 255): ssh: connect to host 127.0.0.1 port 1: Connection refused"}`}
	if _, err := ReadFetch(refused); err == nil || !strings.HasPrefix(err.Error(), "ssh to nobody@127.0.0.1 failed") {
		t.Fatalf("a refusal: %v", err)
	}
	if _, err := ReadFetch([]string{"=== start", "Traceback (most recent call last):", "  File x"}); err == nil ||
		err.Error() != "the fetch ended without an answer" {
		t.Fatalf("no answer: %v", err)
	}
}

// FetchArgv names the remote, its port, the screen's runs and saved folders, then the runs (or --all) and traces.
func TestFetchArgv(t *testing.T) {
	c := Client{Python: "py", Root: "/r", Saved: "/s"}
	if got := strings.Join(c.FetchArgv("root@h", 40022, []string{"a", "b"}, false), " "); got != "py -m boltbeam.cli fetch root@h -p 40022 --into /r --saved /s --json a b" {
		t.Fatalf("picked: %s", got)
	}
	if got := strings.Join(c.FetchArgv("pod", 0, nil, true), " "); got != "py -m boltbeam.cli fetch pod --into /r --saved /s --json --all --with-traces" {
		t.Fatalf("all: %s", got)
	}
}

func TestOffloadRecordedDecodes(t *testing.T) {
	var o Offload
	recorded(t, "offload.json", &o)
	if o.Status != "recommended" || o.Page.Headline == "" || len(o.Page.Sections) != 6 ||
		o.Page.Sections[0].Title != "Where the weights go" || len(o.Page.Sections[0].Rows) != 5 ||
		o.Page.Sections[0].Rows[0].Label != "On the card, replies" || len(o.Page.Sections[0].Lines) != 1 {
		t.Fatalf("offload decoded wrong: %+v", o)
	}
	var r Offload
	recorded(t, "offload-refused.json", &r)
	if r.Status != "refused" || !strings.HasPrefix(r.Page.Headline, "No offload plan here") || len(r.Page.Sections) != 0 {
		t.Fatalf("refused offload decoded wrong: %+v", r)
	}
}
