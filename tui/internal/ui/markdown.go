package ui

import (
	"regexp"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// renderMarkdown draws the markdown Python writes (gameplan.md) in the screen's styles, line by line, so the page
// wraps it as it wraps any body. It knows the few forms the writer uses: # and ## headings, - bullets, **bold**,
// `code`, ``` fences (their lines are kept as written) and indented "evidence:" pointers (muted). The file on disk
// stays the markdown; only the screen draws it.
func renderMarkdown(md string) string {
	lines := strings.Split(md, "\n")
	out := make([]string, 0, len(lines))
	fenced := false
	for _, l := range lines {
		trim := strings.TrimSpace(l)
		switch {
		case strings.HasPrefix(trim, "```"):
			fenced = !fenced
			continue
		case fenced:
			out = append(out, l)
		case strings.HasPrefix(l, "# "):
			out = append(out, stTitle.Render(strings.TrimPrefix(l, "# ")))
		case strings.HasPrefix(l, "## "), strings.HasPrefix(l, "### "):
			out = append(out, stHeader.Render(strings.TrimLeft(l, "# ")))
		case strings.HasPrefix(trim, "evidence:"):
			out = append(out, stMuted.Render(l))
		case strings.HasPrefix(l, "- "):
			out = append(out, stAccent.Render("•")+" "+mdInline(strings.TrimPrefix(l, "- ")))
		default:
			out = append(out, mdInline(l))
		}
	}
	return strings.Join(out, "\n")
}

var (
	stTitle  = lipgloss.NewStyle().Foreground(pink).Bold(true)
	stStrong = lipgloss.NewStyle().Bold(true)
	mdBold   = regexp.MustCompile(`\*\*([^*]+)\*\*`)
	mdCode   = regexp.MustCompile("`([^`]+)`")
)

// mdInline styles **bold** and `code` inside one line and drops their markers.
func mdInline(s string) string {
	s = mdBold.ReplaceAllStringFunc(s, func(m string) string { return stStrong.Render(m[2 : len(m)-2]) })
	return mdCode.ReplaceAllStringFunc(s, func(m string) string { return stInfo.Render(m[1 : len(m)-1]) })
}
