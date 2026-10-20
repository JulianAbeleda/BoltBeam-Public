package ui

import (
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// A prefill run (llama.cpp, Qwen3-8B, 2,048 tokens on an RTX 5090): the where table is the decode table's code with
// the prefill's own columns, read from the seam's cells. Golden at 80 and 110 columns.
func TestPrefillWhereTable(t *testing.T) {
	var res seam.Results
	load(t, "results-prefill", &res)
	old := tableWidth
	defer func() { tableWidth = old }()
	for _, cols := range []int{80, 110} {
		tableWidth = cols - 4
		got := ansi.Strip(wrapText(lossBody(res.Loss), cols-4))
		for _, want := range []string{"Where the prefill time goes", "CEILING", "= prefill", "no ceiling",
			"ffn_down + ffn_gate_up Q4_K", "attention", "lm_head Q6_K"} {
			if !strings.Contains(got, want) {
				t.Errorf("%d cols: missing %q", cols, want)
			}
		}
		golden(t, fmt.Sprintf("loss-prefill-%d.txt", cols), got)
	}
}
