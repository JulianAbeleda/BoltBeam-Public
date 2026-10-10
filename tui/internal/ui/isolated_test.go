package ui

import (
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// An isolated run (each kernel timed alone by BoltBeam's kernel timer, qwen3-8b on the M3, run 005): its sum
// (63.7 ms) is more than the token (55.1 ms). That is expected, so nothing says "Not tied out"; the screen says
// how it was measured and labels the split by role an estimate. The estimate takes each row's time less the
// 1.9 µs dispatch floor and says so beside the measured sum. Golden at 80 and 110 columns.
func TestIsolatedLossSaysHowItWasMeasured(t *testing.T) {
	var res seam.Results
	load(t, "results-isolated", &res)
	old := tableWidth
	defer func() { tableWidth = old }()
	for _, cols := range []int{80, 110} {
		tableWidth = cols - 4
		got := ansi.Strip(wrapText(lossBody(res.Loss), cols-4))
		for _, want := range []string{
			"Each kernel timed alone by BoltBeam's kernel timer, cold (the cache swept by a read",
			"Sum alone: 54.3 ms, less the 1.9 µs dispatch floor per launch: 53.8 ms (the estimate uses this); the real token: 66.0 ms.",
			"Per role, estimated from isolated kernel times",
			"EST. ms IN TOKEN", "(estimate from isolated times)",
			"kernels and gaps, not split", "limit at context 1 (ideal)",
			"Estimated split (isolated): weight kernels 53.8 ms, other and gaps 12.2 ms.",
			"(difference: the token less the kernels timed alone, less the 1.9 µs dispatch floor per launch; attention, norms, launches and gaps were not timed)",
			"not scaled: 53.8 ms alone less the 1.9 µs dispatch floor per launch fits the 66.0 ms token",
			"Per-role source: isolated (BoltBeam kernel timer, llama.cpp kernels",
			"slow kernel", "at the limit",
		} {
			if !strings.Contains(strings.Join(strings.Fields(got), " "), want) {
				t.Errorf("%d cols: missing %q", cols, want)
			}
		}
		own, _, _ := strings.Cut(got, "Beside it") // the other run beside it is in-model and keeps its own table
		for _, bad := range []string{"Not tied out", "15.7 ms lost", "ACTUAL", "· cache", "inconclusive, cache"} {
			if strings.Contains(own, bad) {
				t.Errorf("%d cols: says %q", cols, bad)
			}
		}
		_, roles, _ := strings.Cut(own, "Per role, estimated")
		for _, l := range strings.Split(got, "\n") {
			if w := ansi.StringWidth(l); w > cols-4 || (strings.Contains(roles, l) && strings.Contains(l, "…")) {
				t.Errorf("%d cols: line cut or too wide (%d): %q", cols, w, l)
			}
		}
		golden(t, fmt.Sprintf("loss-isolated-%d.txt", cols), got)
	}
}

// An in-model capture whose kernels sum to more than its token is still refused: only the isolated kind is exempt.
func TestInModelOvercountStillRefused(t *testing.T) {
	var res seam.Results
	load(t, "results-isolated", &res)
	l := res.Loss
	tie := *l.TieOut
	why := "profiled kernels do not fit the real token: they sum to 63.699 ms, more than the 55.124 ms token"
	tie.Refused, tie.Estimate, tie.Isolated, l.Estimate = &why, nil, false, nil
	l.TieOut = &tie
	if got := ansi.Strip(lossBody(l)); !strings.Contains(got, "Not tied out. "+why) {
		t.Errorf("in-model overcount not refused:\n%s", got)
	}
}

// A role whose isolated samples spread wider than the chip's band carries "; noisy: ±N%" after its reason word
// (tie_out.NOISY): the suffix shows in the in-model table, the narrow WHY table and the isolated table as the seam
// sends it, with no second rule here.
func TestNoisyReasonShowsInEveryRoleTable(t *testing.T) {
	old := tableWidth
	defer func() { tableWidth = old }()
	noisy := "too small to fill memory; noisy: ±13.2%"
	roles := []seam.RoleLoss{
		{Role: "attn_kv", Quant: "Q6_K", IdealMs: 0.04, ActualMs: 0.14, LostMs: 0.10, Share: 0.6, CallsPerToken: 18, PctPeak: fp(26.1),
			UsPerCall: fp(7.97), MbPerCall: fp(3.4), Gbs: fp(432.0), EstMs: fp(0.07), EstLostMs: fp(0.03), Reason: noisy,
			ReasonWord: sp("too small to fill memory"), Noisy: true, SpreadPct: fp(26.3)},
		{Role: "lm_head", Quant: "Q6_K", IdealMs: 0.30, ActualMs: 0.31, LostMs: 0.01, Share: 0.4, CallsPerToken: 1, PctPeak: fp(98.6),
			UsPerCall: fp(305.4), MbPerCall: fp(510.5), Gbs: fp(1671.0), EstMs: fp(0.30), EstLostMs: fp(0.0), Reason: "at the limit",
			ReasonWord: sp("at the limit"), SpreadPct: fp(0.7)},
	}
	var b strings.Builder
	for _, w := range []int{200, 76} {
		tableWidth = w
		b.WriteString(fmt.Sprintf("-- role table at %d --\n", w))
		b.WriteString(plain(roleTable(roles, nil)))
	}
	tableWidth = 200
	b.WriteString("-- isolated table --\n")
	b.WriteString(plain(isoRoleTable(roles)))
	got := b.String()
	if strings.Count(got, noisy) != 3 || !strings.Contains(got, "at the limit") {
		t.Fatalf("the noisy suffix must show in all three tables:\n%s", got)
	}
	golden(t, "loss-noisy.txt", got)
}
