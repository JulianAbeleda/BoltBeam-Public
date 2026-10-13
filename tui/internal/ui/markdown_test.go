package ui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"github.com/muesli/termenv"
)

// The gameplan's markdown is drawn, not printed: headings and **bold** come out bold with their markers gone, a
// fence's lines stay as written, and the text a reader sees is the markdown less its markers.
func TestRenderMarkdown(t *testing.T) {
	defer lipgloss.SetColorProfile(lipgloss.ColorProfile())
	lipgloss.SetColorProfile(termenv.TrueColor)
	md := "# Gameplan: X\n\n## 1. attn_kv Q4_K\n- **BubbleBeam**: not recorded in `a.json`\n  evidence: b.json\n```\nrow **kept**\n```"
	got := renderMarkdown(md)
	want := "Gameplan: X\n\n1. attn_kv Q4_K\n• BubbleBeam: not recorded in a.json\n  evidence: b.json\nrow **kept**"
	if plain := ansi.Strip(got); plain != want {
		t.Fatalf("text:\n%s\nwant:\n%s", plain, want)
	}
	lines := strings.Split(got, "\n")
	for _, i := range []int{0, 2, 3} {
		if !strings.Contains(lines[i], "\x1b[1") {
			t.Errorf("line %d is not bold: %q", i, lines[i])
		}
	}
}
