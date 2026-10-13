package ui

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// A replay has no tok/s headline and no ceiling reason: the page said "No speed limit for this chip:" with nothing
// after the colon, and What next said "nothing measured yet".
func TestReplayHeadlineIsNotNoLimit(t *testing.T) {
	var res seam.Results
	raw := `{"measured":true,"workload":"replay","ceiling":{"reason":""},"loss":{"status":"modeled","limit_ms":1000,
	  "replay":{"title":"The replay","columns":["turn","flags"],"rows":[{"cells":["1","cold"]},{"cells":["2",""]}],"total":["warm",""]}}}`
	if err := json.Unmarshal([]byte(raw), &res); err != nil {
		t.Fatal(err)
	}
	f := Facts{Run: &seam.Run{Results: res}}
	f.Run.Stages = []seam.Stage{{Key: "output", Done: true}}
	got := plain(resultBody(f, 120))
	if strings.Contains(got, "No speed limit") || !strings.Contains(got, "Replay measured: 2 turns.") {
		t.Fatalf("replay headline:\n%s", got)
	}
	if !strings.Contains(got, "kernels from the capture") || strings.Contains(got, "roles not timed") {
		t.Fatalf("replay title:\n%s", got)
	}
	if mark, line := resultLine(f); mark != "pass" || line != "replay measured · 2 turns" {
		t.Fatalf("what next: %s %q", mark, line)
	}
}

// The decode tie-out said "Tracing slowed the token" also when the captured token was faster, and never showed the
// repeat check or that the untraced time was taken at another context.
func TestTokenEndToEndSaysTheSignTheContextAndTheCheck(t *testing.T) {
	tie := seam.TieOut{Context: 145, TokenMs: fp(11.382), UntracedMs: fp(11.558), UntracedContext: ip(512),
		Band: sp("not checked: no repeat band for this setup")}
	got := plain(tokenEndToEnd(tie))
	for _, want := range []string{"captured token is 11.382 ms, the untraced median 11.558 ms (-1.5%)",
		"untraced time is at context 512 and the capture at context 145", "Repeat check: not checked"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in %q", want, got)
		}
	}
	if strings.Contains(got, "slowed") {
		t.Errorf("a faster capture is not slowed: %q", got)
	}
	tie.UntracedContext, tie.Band = ip(145), sp("repeat band ±0.120000 ms")
	if got := plain(tokenEndToEnd(tie)); strings.Contains(got, "not one setup") || !strings.Contains(got, "Repeat check: repeat band ±0.120000 ms.") {
		t.Errorf("same context, checked: %q", got)
	}
}

// Prefill and replay: the end-to-end check is one status line under the where table, in its three states.
func TestWhereEndToEndStates(t *testing.T) {
	e := seam.EndToEnd{CapturedMs: fp(112.2), UntracedMs: fp(116.1), DiffPct: fp(-3.4), BandMs: fp(0.195)}
	cases := []struct {
		confirms *bool
		want     string
	}{{nil, "not checked: no repeat band"}, {bp(true), "confirmed within the repeat band of ±0.195 ms"},
		{bp(false), "not confirmed: outside the repeat band of ±0.195 ms. The rows stand."}}
	for _, c := range cases {
		e.Confirms = c.confirms
		if got := plain(whereEndToEnd(&e)); !strings.Contains(got, "captured 112.2 ms, untraced 116.1 ms (-3.4%)") || !strings.Contains(got, c.want) {
			t.Errorf("want %q in %q", c.want, got)
		}
	}
	if whereEndToEnd(nil) != "" {
		t.Error("no check, no line")
	}
}

// "Not checked" is explained once per page, with the command that makes a band.
func TestNotCheckedIsExplainedOnce(t *testing.T) {
	var res seam.Results
	raw := `{"measured":true,"timing":{"tok_s":80},"ceiling":{"tok_s":100},"loss":{"status":"modeled","limit_ms":9,"provider":"vllm",
	  "tie_out":{"context":145,"token_ms":11.4,"token_source":"the captured run's own time per token","untraced_ms":11.6,
	    "untraced_context":145,"band":"not checked: no repeat band for this setup","lines":[]}}}`
	if err := json.Unmarshal([]byte(raw), &res); err != nil {
		t.Fatal(err)
	}
	got := plain(resultBody(Facts{Run: &seam.Run{Results: res}}, 200))
	if strings.Count(got, "Not checked is expected on a first run") != 1 || !strings.Contains(got, "screen repeat-bands --runs A B") {
		t.Fatalf("explanation:\n%s", got)
	}
}

// The engine build against its pin and the recorded toolchain are on the page; an older run says it has none.
func TestRunEngineBodyNamesBuildAndToolchain(t *testing.T) {
	if got := plain(runEngineBody(&seam.RunEngine{Provider: "llama.cpp"})); !strings.Contains(got, "Engine build: not recorded") ||
		!strings.Contains(got, "Toolchain: not recorded") {
		t.Errorf("older run: %q", got)
	}
	e := seam.RunEngine{Provider: "llama.cpp", Build: &seam.EngineBuild{Build: sp("097cf33d9"), Matches: bp(false),
		Words: "llama.cpp build 097cf33d9 is not the pin b11571 (abc)"},
		Toolchain: &seam.ToolchainFacts{RecordedAt: "2026-10-12T20:24:33+00:00",
			Versions: map[string]string{"driver": "595.99.02", "cuda": "13.2", "torch": "unknown"}, Runtime: map[string]string{"cuda": "13.0"}}}
	got := runEngineBody(&e)
	if !strings.Contains(got, glyphWarn) || !strings.Contains(plain(got), "is not the pin b11571") {
		t.Errorf("a build off its pin is a warning: %q", plain(got))
	}
	if want := "Toolchain (2026-10-12 20:24 UTC): driver 595.99.02, cuda 13.2, torch's CUDA 13.0; not found: torch."; !strings.Contains(plain(got), want) {
		t.Errorf("want %q in %q", want, plain(got))
	}
	e.Toolchain.Versions, e.Toolchain.Runtime = map[string]string{"driver": "unknown", "cuda": "unknown"}, map[string]string{"cuda": "unknown"}
	if want := "Toolchain (2026-10-12 20:24 UTC): none of driver, cuda found on this machine."; !strings.Contains(plain(runEngineBody(&e)), want) {
		t.Errorf("a Mac records no CUDA tools: %q", plain(runEngineBody(&e)))
	}
}

// The per-role repeat check is counted on the page: checked rows against the setup's band, or not checked.
func TestRoleChecksCountTheBandedRows(t *testing.T) {
	checked, not := "checked", "not checked"
	roles := []seam.RoleLoss{{CheckStatus: &checked}, {CheckStatus: &checked}, {CheckStatus: &not}}
	if got := plain(roleChecks(roles)); got != "Per-role repeat check: 2 of 3 roles checked against the measured repeat band of this setup.\n" {
		t.Errorf("got %q", got)
	}
	if got := plain(roleChecks(roles[2:])); !strings.Contains(got, "not checked: no repeat band") {
		t.Errorf("got %q", got)
	}
	if roleChecks([]seam.RoleLoss{{}}) != "" {
		t.Error("an older Python says nothing")
	}
}

// Pairing flags: an unsure pairing and each unpaired role with the solver's reason.
func TestPairingBodyShowsAmbiguousAndReasons(t *testing.T) {
	var l seam.Loss
	raw := `{"ambiguous":{"reason":"more than 50000 ways to walk 12 rows over 153 slot groups"},
	  "unpaired_roles":[{"role":"ffn_down","quant":"Q4_K","count":36,"reason":"equal launches and time"}]}`
	if err := json.Unmarshal([]byte(raw), &l); err != nil {
		t.Fatal(err)
	}
	got := plain(pairingBody(l))
	for _, want := range []string{"Pairing is not certain: more than 50000 ways", "Unpaired, so in not attributed: feed-forward out Q4_K: equal launches and time."} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in %q", want, got)
		}
	}
}

// A replay table wider than the page lost its flags at the edge ("THROTT…"); the flags now leave the table.
func TestReplayFlagsLeaveATableWiderThanThePage(t *testing.T) {
	defer func(w int) { tableWidth = w }(tableWidth)
	tableWidth = 30
	rows := [][]string{{"TURN", "CEILING MS", "CHIP BEST MS", "FLAGS"}, {"1", "8994", "8986", "cold THROTTLED"},
		{"2", "7792", "7785", "THROTTLED"}, {"3", "3327", "3321", ""}}
	got := plain(replayTable(rows))
	if strings.Contains(got, "FLAGS") || !strings.Contains(got, "cold: 1 turns (1)") || !strings.Contains(got, "THROTTLED: 2 turns (1, 2)") {
		t.Fatalf("flags:\n%s", got)
	}
	tableWidth = 200
	if got := plain(replayTable(rows)); !strings.Contains(got, "FLAGS") {
		t.Fatalf("a table that fits keeps its flags:\n%s", got)
	}
}

// Saved runs name the workload, and a replay shows its roofline share instead of "not measured".
func TestSavedRowsNameWorkloadAndReplayShare(t *testing.T) {
	var s seam.Saved
	raw := `{"root":"/s","runs":[{"id":"replay-a60","model_id":"m","target_id":"t","provider":"llama.cpp","saved_at":"2026-10-12T20:24:33",
	  "workload":"replay","measured":true,"roofline_pct":65.0},
	  {"id":"pre","model_id":"m","target_id":"t","provider":"vllm","saved_at":"2026-10-12T20:24:33","workload":"prefill","measured":true,"pct_of_limit":80}]}`
	if err := json.Unmarshal([]byte(raw), &s); err != nil {
		t.Fatal(err)
	}
	acts := savedActions(Facts{Saved: &s})
	if got := plain(acts[0].label); got != "m · t · llama.cpp replay · 2026-10-12 20:24 · 65% of roofline" {
		t.Errorf("replay row: %q", got)
	}
	if got := plain(acts[1].label); got != "m · t · vllm prefill · 2026-10-12 20:24 · 80% of limit" {
		t.Errorf("prefill row: %q", got)
	}
}

// Compare: a saved run is "before"; Saved runs then offers every other run as "after", and two replays from
// different engines go to replay-cross.
func TestCompareFlowPicksAfterAndCross(t *testing.T) {
	var s seam.Saved
	raw := `{"root":"/s","runs":[{"id":"a","provider":"llama.cpp","workload":"replay"},{"id":"b","provider":"vllm","workload":"replay"},
	  {"id":"c","provider":"llama.cpp","workload":"decode"}]}`
	if err := json.Unmarshal([]byte(raw), &s); err != nil {
		t.Fatal(err)
	}
	f := Facts{Saved: &s, CompareFrom: "a"}
	dos := []string{}
	for _, a := range savedActions(f) {
		dos = append(dos, a.do+":"+a.arg)
	}
	if strings.Join(dos, " ") != "note: compare:b compare:c comparecancel:" {
		t.Errorf("compare rows: %v", dos)
	}
	if !f.crossEngines("a", "b") || f.crossEngines("a", "c") {
		t.Error("replay-cross only for two replays from different engines")
	}
	if got := plain(compareRunsBody(Facts{Compared: &Compared{Err: "the runs differ in model"}}, 80)); !strings.Contains(got, "Not compared: the runs differ in model.") {
		t.Errorf("refusal: %q", got)
	}
	if got := compareRunsBody(Facts{Compared: &Compared{Lines: []string{"Engine upgrade audit: b against a", "  row"}}}, 80); got != "Engine upgrade audit: b against a\n  row" {
		t.Errorf("lines: %q", got)
	}
}

// A session's headline is the turn at the focus depth against the limit at that depth. It was the shallowest
// depth's turn against the focus depth's limit: 4,094 against 13,831 read 30%; the like-for-like figure is 26%.
func TestSessionHeadlineIsTheFocusDepth(t *testing.T) {
	var res seam.Results
	raw := `{"measured":true,"timing":{"tok_s":4094.1,"context":1024,"also":[{"context":16384,"tok_s":3579.7}]},
	  "ceiling":{"tok_s":13830.6},"loss":{"status":"modeled","limit_ms":37.0,"runtimes":[{"provider":"llama.cpp","tok_s":3579.7,"ms":143.0}],
	  "session":{"title":"The session","focus":16384,"columns":["depth"],"rows":[{"cells":["1,024"]}]}}}`
	if err := json.Unmarshal([]byte(raw), &res); err != nil {
		t.Fatal(err)
	}
	f := Facts{Run: &seam.Run{Results: res}}
	f.Run.Stages = []seam.Stage{{Key: "output", Done: true}}
	got := plain(resultBody(f, 160))
	if !strings.Contains(got, "3579.7 tokens per second measured for a turn at depth 16384. The limit at that depth is 13830.6.") ||
		!strings.Contains(got, "26% of the limit") || strings.Contains(got, "4094.1 tokens per second measured") {
		t.Fatalf("session headline:\n%s", got)
	}
	if _, line := resultLine(f); line != "3579.7 tokens per second at depth 16384 · 26% of the limit" {
		t.Fatalf("what next: %q", line)
	}
}
