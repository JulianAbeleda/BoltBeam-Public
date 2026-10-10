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
// how it was measured and labels the split by role an estimate. Golden at 80 and 110 columns.
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
			"Sum alone: 54.3 ms; the real token: 66.0 ms.",
			"Per role, estimated from isolated kernel times",
			"EST. ms IN TOKEN", "(estimate from isolated times)",
			"kernels and gaps, not split", "limit at context 1 (ideal)",
			"Estimated split (isolated): weight kernels 54.3 ms, other and gaps 11.7 ms.",
			"(difference: the token less the kernels timed alone; attention, norms, launches and gaps were not timed)",
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
