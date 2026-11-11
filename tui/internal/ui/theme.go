package ui

import (
	"fmt"
	"image/color"
	"math"
	"strings"

	"charm.land/bubbles/v2/progress"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
	colorful "github.com/lucasb-eyer/go-colorful"
)

// The palette is a set of tokens, not colours sprinkled through the views. Bubble Tea degrades them to the
// terminal's profile and strips them under NO_COLOR, so the plain golden screens are the same text.
var (
	pink   = adaptive{Light: "#C2408F", Dark: "#F5A3D7"}
	purple = adaptive{Light: "#6E56CF", Dark: "#A89CFF"}
	cyan   = adaptive{Light: "#0E8FA6", Dark: "#8BE0EF"}
	mint   = adaptive{Light: "#2F9E6F", Dark: "#9FE5B3"}
	butter = adaptive{Light: "#B7791F", Dark: "#F6D37A"}
	coral  = adaptive{Light: "#D1443E", Dark: "#FF9A8B"}
	muted  = adaptive{Light: "#8A8798", Dark: "#8C889C"}
)

// darkBackground is the terminal's answer to Bubble Tea's background query (tea.BackgroundColorMsg); until it
// comes, or when none does, dark, as Lip Gloss v1 assumed.
var darkBackground = true

// adaptive is a colour with a light and a dark variant, picked by darkBackground when it is drawn.
type adaptive struct{ Light, Dark string }

func (c adaptive) RGBA() (r, g, b, a uint32) { return lipgloss.Color(pair(c)).RGBA() }

var _ color.Color = adaptive{}

var (
	stMuted  = lipgloss.NewStyle().Foreground(muted)
	stOK     = lipgloss.NewStyle().Foreground(mint)
	stBad    = lipgloss.NewStyle().Foreground(coral).Bold(true)
	stWarn   = lipgloss.NewStyle().Foreground(butter)
	stInfo   = lipgloss.NewStyle().Foreground(cyan)
	stAccent = lipgloss.NewStyle().Foreground(pink)
	stHeader = lipgloss.NewStyle().Foreground(purple).Bold(true)
	stCursor = lipgloss.NewStyle().Foreground(pink).Bold(true)
	stBox    = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(purple).Padding(0, 1)
	stBoxDim = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(muted).Padding(0, 1)
)

// Status glyphs: one per outcome, used in every table so a row can be read at a glance. ♡ marks a kept route.
const (
	glyphPass  = "✓"
	glyphFail  = "✗"
	glyphWarn  = "⚠"
	glyphWait  = "⏸"
	glyphOpen  = "○"
	glyphSkip  = "◇" // a stage this run does not need: neither open nor done
	glyphRun   = "●"
	glyphHeart = "♡"
	glyphBolt  = "⚡"
)

// The mascot: a small bolt that sits in the header and changes its face with the state of the run.
const (
	faceSleep   = "(˘ω˘) zz"
	faceIdle    = "(•ᴗ•)"
	faceBusy    = "(•̀ᴗ•́)⚡"
	faceHappy   = "(≧◡≦)⚡"
	faceWaiting = "(•-•)⏸"
	faceWorried = "(•́︿•̀)"
)

func mark(result string) string {
	switch result {
	case "pass":
		return stOK.Render(glyphPass)
	case "fail":
		return stBad.Render(glyphFail)
	case "crossed":
		return stWarn.Render(glyphWarn)
	case "wait":
		return stInfo.Render(glyphWait)
	case "run":
		return stAccent.Render(glyphRun)
	case "kept":
		return stAccent.Render(glyphHeart)
	case "skip":
		return stMuted.Render(glyphSkip)
	}
	return stMuted.Render(glyphOpen)
}

func hexBlend(from, to string, t float64) string {
	var f, g [3]int
	fmt.Sscanf(from, "#%02x%02x%02x", &f[0], &f[1], &f[2])
	fmt.Sscanf(to, "#%02x%02x%02x", &g[0], &g[1], &g[2])
	return fmt.Sprintf("#%02x%02x%02x", int(float64(f[0])+(float64(g[0])-float64(f[0]))*t),
		int(float64(f[1])+(float64(g[1])-float64(f[1]))*t), int(float64(f[2])+(float64(g[2])-float64(f[2]))*t))
}

func pair(c adaptive) string {
	if darkBackground {
		return c.Dark
	}
	return c.Light
}

// gradient renders text with a per-rune foreground ramp from pink to cyan, bold: the title treatment.
func gradient(text string) string {
	runes := []rune(text)
	var b strings.Builder
	for i, r := range runes {
		t := 0.0
		if len(runes) > 1 {
			t = float64(i) / float64(len(runes)-1)
		}
		b.WriteString(lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(hexBlend(pair(pink), pair(cyan), t))).Render(string(r)))
	}
	return b.String()
}

// bar is a small progress bar; ratio 1 is full. The fill carries the pink-to-purple ramp.
func bar(ratio float64, width int) string {
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	// v1's ramp, kept: full blocks, cell i of the bar blended in Luv at i/(width-1), a single cell at the middle
	from, _ := colorful.Hex(pair(pink))
	to, _ := colorful.Hex(pair(purple))
	single := int(math.Round(float64(width)*ratio)) == 1
	ramp := func(_, current float64) color.Color { // current is i/width
		t := 0.5
		if !single {
			t = math.Round(current*float64(width)) / float64(width-1)
		}
		return lipgloss.Color(from.BlendLuv(to, t).Hex())
	}
	p := progress.New(progress.WithColorFunc(ramp), progress.WithoutPercentage(), progress.WithWidth(width),
		progress.WithFillCharacters(progress.DefaultFullCharFullBlock, progress.DefaultEmptyCharBlock))
	return p.ViewAs(ratio)
}

// box frames a section with a rounded border and a gradient title; lines are cut to the width, never wrapped.
func box(title, body string, width int, dim bool) string {
	style := stBox
	if dim {
		style = stBoxDim
	}
	inner := width - 4
	if inner < 10 {
		inner = 10
	}
	lines := strings.Split(strings.TrimRight(body, "\n"), "\n")
	for i, line := range lines {
		lines[i] = truncate(line, inner)
	}
	content := gradient(title) + "\n" + strings.Join(lines, "\n")
	return style.Width(width).Render(content) // Lip Gloss v2 counts the border in the width
}

// truncate cuts a styled line to width cells with "…"; the screen never wraps.
func truncate(line string, width int) string { return ansi.Truncate(line, width, "…") }
