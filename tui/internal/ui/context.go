package ui

// Context: the prompt and the reply that Recommended setup, Offload and Tuned for this machine are stated at. One
// setting on Setup; unset, each page keeps its own default. The reply is typed, or called for by a file of measured
// replies, which the screen reads (tune.reply_budget) on each page that takes it: a file it refuses is that page's
// error, in Python's words. The words are Python's (data/plain_words.json context).

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
)

// Context is what was typed: the prompt, and the reply as tokens or as a replies file.
type Context struct {
	Prompt, Reply int
	Replies       string
}

func (c Context) on() bool { return c.Prompt > 0 && (c.Reply > 0 || c.Replies != "") }

// pageArgs is the context Recommended setup and Offload are stated at.
func (c Context) pageArgs() []string {
	switch {
	case !c.on():
		return nil
	case c.Replies != "":
		return []string{"--prompt", fmt.Sprint(c.Prompt), "--replies", c.Replies}
	}
	return []string{"--context", fmt.Sprint(c.Prompt + c.Reply)}
}

// tuneArgs is the same for Tuned for this machine, which also takes the prompt it sweeps at.
func (c Context) tuneArgs() []string {
	if c.on() && c.Replies == "" {
		return append([]string{"--prompt", fmt.Sprint(c.Prompt)}, c.pageArgs()...)
	}
	return c.pageArgs()
}

func contextValue(f Facts) string {
	w, c := vocab.Context, f.Context
	r := strings.NewReplacer("{prompt}", commas(float64(c.Prompt)), "{reply}", commas(float64(c.Reply)), "{file}", filepath.Base(c.Replies))
	switch {
	case !c.on():
		return w["default"]
	case c.Replies != "":
		return r.Replace(w["value_replies"])
	}
	return r.Replace(w["value"])
}

func contextBody(Facts, int) string { return vocab.Context["body"] }

// openContext is the form: the prompt, then the reply as tokens or as a file of measured replies.
func (m *Model) openContext() tea.Cmd {
	c, w := m.f.Context, vocab.Context
	s := &formState{kind: "context", ctx: [3]string{}}
	for i, n := range []int{c.Prompt, c.Reply} {
		if n > 0 {
			s.ctx[i] = fmt.Sprint(n)
		}
	}
	s.ctx[2] = c.Replies
	whole := func(v string) error {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); v != "" && (err != nil || n < 1) {
			return errors.New(w["whole"])
		}
		return nil
	}
	return m.openForm(s,
		huh.NewInput().Title(w["prompt"]).Description(w["prompt_about"]).Value(&s.ctx[0]).Validate(whole),
		huh.NewInput().Title(w["reply"]).Description(w["reply_about"]).Value(&s.ctx[1]).Validate(whole),
		huh.NewInput().Title(w["replies"]).Description(w["replies_about"]).Value(&s.ctx[2]).Validate(func(string) error {
			p, r, f := strings.TrimSpace(s.ctx[0]), strings.TrimSpace(s.ctx[1]), strings.TrimSpace(s.ctx[2])
			switch {
			case p == "" && r == "" && f == "":
				return nil
			case p == "":
				return errors.New(w["need_prompt"])
			case (r == "") == (f == ""):
				return errors.New(w["need_reply"])
			}
			return nil
		}))
}

// contextDone keeps the form: unset, or the prompt with the reply typed or a replies file (an absolute path: the
// screen runs from the repo). The pages read the file when they are asked, and say if they cannot.
func (m Model) contextDone(v [3]string) Model {
	p, _ := strconv.Atoi(strings.TrimSpace(v[0]))
	r, _ := strconv.Atoi(strings.TrimSpace(v[1]))
	file := strings.TrimSpace(v[2])
	if home, err := os.UserHomeDir(); err == nil && (file == "~" || strings.HasPrefix(file, "~/")) {
		file = filepath.Join(home, file[1:])
	}
	if abs, err := filepath.Abs(file); file != "" && err == nil {
		file = abs
	}
	m.f.Context = Context{Prompt: p, Reply: r, Replies: file}
	m.backToSetup()
	return m
}
