package ui

// Speed up, on the Recommended setup page beside Measure MTP: a copy of the model that decodes faster, the original
// kept. A confirm step (Huh) says what happens, the disk space, the time and the GPU; the job is `derive lighter-head`,
// drawn by the stage names Python gave it; its log is worded by `screen speedup-plain`. Python owns every sentence
// (data/plain_words.json speed_up) and every figure; this only places them.

import (
	"fmt"
	"math"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// SpeedUp is Speed up for the chosen model: what Python offers, the job, its worded result, and the check of a copy.
type SpeedUp struct {
	OfferFor string // the model the offer is for (Facts.Path when it was asked)
	Offer    *seam.SpeedOffer
	Job      *jobs.Job
	Tail     []string
	Result   *seam.SpeedPlain
	Asked    string // the job start the result was asked for, so a finished job is worded once
	Verify   *seam.SpeedVerify
	Checking bool
	done     int // stages finished, to time the stage in flight
	at       time.Time
}

type speedOfferMsg struct {
	path  string
	offer *seam.SpeedOffer
	err   error
}
type speedJobMsg struct {
	id   string
	job  *jobs.Job
	tail []string
}
type speedStartedMsg struct{ job *jobs.Job }
type speedResultMsg struct {
	id, started string
	res         *seam.SpeedPlain
	err         error
}
type speedVerifyMsg struct {
	path string
	v    *seam.SpeedVerify
	err  error
}

// speedID is the Speed up job of the chosen model: its log in the TUI's state folder.
func (f Facts) speedID() string { return RunStem(f.Path, "speedup") }

func (f Facts) speeding() bool { return f.Speed.Job != nil && f.Speed.Job.Alive }

func (f Facts) speedOffered() *seam.SpeedOffer {
	if f.Path == "" || f.Speed.OfferFor != f.Path {
		return nil
	}
	return f.Speed.Offer
}

// advance times the stage in flight: it counts from the job's start for the first, else from when the last one finished.
func (s *SpeedUp) advance(now time.Time) {
	if s.Job == nil {
		return
	}
	p := seam.ReadProgress(s.Tail)
	if s.done != p.Done || s.at.IsZero() {
		s.done, s.at = p.Done, now
		if t, err := time.Parse(time.RFC3339, s.Job.StartedAt); err == nil && p.Done == 0 {
			s.at = t
		}
	}
}

// loadSpeed asks about the chosen model once per model: whether Speed up applies, and the job it may have run before.
func (m *Model) loadSpeed() tea.Cmd {
	if m.f.Path == "" || m.f.Speed.OfferFor == m.f.Path {
		return nil
	}
	path, client := m.f.Path, m.client
	m.f.Speed = SpeedUp{OfferFor: path}
	return tea.Batch(func() tea.Msg { o, err := client.SpeedOffer(path); return speedOfferMsg{path, o, err} }, m.readSpeed())
}

// readSpeed reads the job and its whole log: the answer is one JSON object, longer than a tail.
func (m Model) readSpeed() tea.Cmd {
	id, store := m.f.speedID(), m.store
	return func() tea.Msg {
		job, err := store.Status(id)
		if err != nil {
			return speedJobMsg{id, nil, nil}
		}
		lines, _ := store.Tail(id, 0)
		return speedJobMsg{id, &job, lines}
	}
}

// startSpeed starts the job on a free GPU.
func (m Model) startSpeed() tea.Cmd {
	id, path, store, repo := m.f.speedID(), m.f.Path, m.store, m.client.Repo
	argv := m.client.SpeedArgv(path)
	return func() tea.Msg {
		if live := gpuJobs(store); len(live) > 0 { // two measurements on one GPU would both be wrong
			return noteMsg(live[0].ID + " is still going. Stop it or wait, then speed up.")
		}
		job, err := store.Start(id, repo, argv)
		if err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return speedStartedMsg{&job}
	}
}

// askSpeedPlain words a finished job's log.
func (m Model) askSpeedPlain(job *jobs.Job) tea.Cmd {
	id, client, log, started := m.f.speedID(), m.client, job.LogPath, job.StartedAt
	return func() tea.Msg { r, err := client.SpeedPlain(log); return speedResultMsg{id, started, r, err} }
}

func (m Model) checkCopy() tea.Cmd {
	path, client := m.f.Path, m.client
	return func() tea.Msg { v, err := client.SpeedVerify(path); return speedVerifyMsg{path, v, err} }
}

// speedMsg handles the Speed up messages; ok says it was one.
func (m Model) speedMsg(msg tea.Msg) (Model, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case speedOfferMsg:
		if msg.path == m.f.Path && msg.err == nil {
			m.f.Speed.Offer = msg.offer
		} else if msg.path == m.f.Path {
			m.note = "Speed up could not be checked for this model: " + msg.err.Error()
		}
	case speedStartedMsg:
		m.f.Speed.Job, m.f.Speed.Tail, m.f.Speed.Result, m.f.Speed.Verify, m.f.Speed.Asked = msg.job, nil, nil, nil, ""
		m.f.Speed.done, m.f.Speed.at = -1, time.Time{}
		m.f.Speed.advance(clock())
		m.note = ""
		if !m.ticking {
			m.ticking = true
			return m, tick(), true
		}
	case speedJobMsg:
		if msg.id != m.f.speedID() {
			break
		}
		was := m.f.speeding()
		m.f.Speed.Job, m.f.Speed.Tail = msg.job, msg.tail
		m.f.Speed.advance(clock())
		switch {
		case m.f.speeding() && !m.ticking:
			m.ticking = true
			return m, tick(), true
		case msg.job != nil && !msg.job.Alive && m.f.Speed.Asked != msg.job.StartedAt:
			m.f.Speed.Asked = msg.job.StartedAt
			if was {
				m.note = "Speed up ended."
			}
			return m, m.askSpeedPlain(msg.job), true
		}
	case speedResultMsg:
		if msg.id != m.f.speedID() {
			break
		}
		if msg.err != nil {
			m.note = "The answer could not be read: " + msg.err.Error()
			break
		}
		m.f.Speed.Result = msg.res
		if msg.res.Outcome == "kept" { // the copy is a model now: lists show it
			return m, m.findFiles(), true
		}
	case speedVerifyMsg:
		m.f.Speed.Checking = false
		if msg.path != m.f.Path {
			break
		}
		if msg.err != nil {
			m.note = "The check did not run: " + msg.err.Error()
			break
		}
		m.f.Speed.Verify = msg.v
	default:
		return m, nil, false
	}
	return m, nil, true
}

// speedStage is the plain name of the stage in flight.
func (f Facts) speedStage() string {
	if p := seam.ReadProgress(f.Speed.Tail); p.Current != "" && !p.Setup {
		return stageWord(p.Current)
	}
	return "Starting"
}

// speedLeft is the time left from the seconds Python expects each stage to take, or "" before there is a stage in flight.
func (f Facts) speedLeft() string {
	if left, ok := seam.ReadProgress(f.Speed.Tail).Left(clock().Sub(f.Speed.at).Seconds()); ok {
		return strings.ReplaceAll(vocab.SpeedUp.UI.Left, "{minutes}", fmt.Sprint(int(math.Ceil(left/60))))
	}
	return ""
}

// speedActions are the page's Speed up rows: the button, or its plain reason when off; Stop while it runs; the check of
// a copy; and what next after a kept copy.
func speedActions(f Facts) []action {
	o, w := f.speedOffered(), vocab.SpeedUp
	switch {
	case o == nil:
		return nil
	case f.speeding():
		return []action{note("  " + f.Spin + " " + w.UI.Running + ": " + f.speedStage() + "…"), {"[ " + w.UI.Stop + " ] " + w.UI.StopHelper, "speedstop", ""}}
	case f.Speed.Checking:
		return []action{note("  " + f.Spin + " " + w.UI.Checking + "…")}
	case o.IsCopy:
		return []action{note("  [ " + w.Action + " ] " + deref(o.Reason)), {"[ " + w.UI.Check + " ] " + w.UI.CheckHelper, "speedcheck", ""}}
	case !o.Enabled:
		return []action{note("  [ " + w.Action + " ] " + deref(o.Reason))}
	}
	out := []action{{"[ " + w.Action + " ] " + w.Helper, "speedup", ""}}
	if r := f.Speed.Result; r != nil && r.Outcome == "kept" && r.Copy != nil {
		out = append(out, action{"[ " + w.UI.NextMtp + " ]", "speedmtp", ""}, action{"[ " + w.UI.NextCompare + " ]", "comparepick", ""})
	}
	return out
}

// speedBody is the page's Speed up block: the bar while it runs, else layer 1 of the last job, else the check of a copy.
func speedBody(f Facts, width int) string {
	var b strings.Builder
	w := vocab.SpeedUp
	switch s := f.Speed; {
	case f.speeding():
		p := seam.ReadProgress(s.Tail)
		frac := p.Fraction(clock().Sub(s.at).Seconds())
		count := ""
		if p.Total > 0 && !p.Setup {
			count = fmt.Sprintf("%d of %d", min(p.Done+1, p.Total), p.Total)
		}
		barW := max(width-4-4-lipgloss.Width(count)-6, 10)
		b.WriteString(stHeader.Render(w.Action) + "\n")
		took := elapsed(s.Job)
		if left := f.speedLeft(); left != "" {
			took += " · " + left
		}
		fmt.Fprintf(&b, "  %s  %3.0f%%  %s\n  %s…   %s\n", bar(frac, barW), frac*100, count, f.speedStage(), stMuted.Render(took))
		if done := p.DoneStages(); len(done) > 0 {
			names := make([]string, len(done))
			for i, k := range done {
				names[i] = stageWord(k)
			}
			b.WriteString(paint(stMuted, hang(glyphPass+" "+strings.Join(names, " · "), 2, width)) + "\n")
		}
	case s.Result != nil:
		r := s.Result
		style, glyph := stOK, glyphPass
		switch r.Outcome {
		case "not_kept", "crashed":
			style, glyph = stBad, glyphFail
		case "refused":
			style, glyph = stWarn, glyphWarn
		}
		b.WriteString(stHeader.Render(w.Action) + "\n" + hang(style.Render(glyph+" "+r.Head), 2, width) + "\n")
		for _, l := range r.Lines {
			line := l
			if strings.HasPrefix(l, "SPEED NOT MEASURED") {
				line = stWarn.Render(l)
			}
			b.WriteString(hang(line, 4, width) + "\n")
		}
		if r.Fix != nil {
			b.WriteString(hang(*r.Fix, 4, width) + "\n")
		}
	}
	if v := f.Speed.Verify; v != nil {
		style, glyph := stOK, glyphPass
		if !v.Ok {
			style, glyph = stBad, glyphFail
		}
		b.WriteString(hang(style.Render(glyph+" "+v.Head), 2, width) + "\n")
		for _, l := range v.Lines {
			b.WriteString(hang(l, 4, width) + "\n")
		}
		if v.Fix != nil {
			b.WriteString(hang(*v.Fix, 4, width) + "\n")
		}
	}
	if b.Len() > 0 {
		b.WriteString("\n") // a blank line before the recommendation under it
	}
	return b.String()
}

// openSpeedConfirm is the confirm step: what will happen, the disk space, the time, the GPU. The answer starts as Not now.
func (m *Model) openSpeedConfirm() tea.Cmd {
	o := m.f.speedOffered()
	if o == nil || !o.Enabled || o.Confirm == nil || m.f.speeding() {
		return nil
	}
	s := &formState{kind: "speedup"}
	c := o.Confirm
	return m.openForm(s, huh.NewConfirm().Title(c.Title).Description(strings.Join(c.Lines, "\n")).
		Affirmative(c.Yes).Negative(c.No).Value(&s.confirmed))
}

// doSpeed is one row of the page.
func (m Model) doSpeed(a action) (Model, tea.Cmd) {
	switch a.do {
	case "speedup":
		return m, m.openSpeedConfirm()
	case "speedstop":
		id, store := m.f.speedID(), m.store
		return m, func() tea.Msg {
			if _, err := store.Stop(id); err != nil {
				return noteMsg("Stop failed: " + err.Error())
			}
			return noteMsg(vocab.SpeedUp.UI.Stopped)
		}
	case "speedcheck":
		m.f.Speed.Checking, m.f.Speed.Verify = true, nil
		return m, m.checkCopy()
	case "speedmtp": // the copy becomes the model; its Recommended setup offers Measure MTP
		if r := m.f.Speed.Result; r != nil && r.Copy != nil {
			m.f.Path = *r.Copy
			m.row = 0
			m.note = "Now on the faster copy. Measure MTP is on this page once it has loaded."
			return m.readModel()
		}
	}
	return m, nil
}
