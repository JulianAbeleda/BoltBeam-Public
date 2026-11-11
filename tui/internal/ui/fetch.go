package ui

// Fetch runs, on Saved runs: `boltbeam fetch --json` through the seam. The remote (user@host or an ssh Host
// alias, a port for a pod) is typed in a form, its runs listed (one ssh call), some or all picked in a second form
// (forms.go) and fetched as a job, the way Run is: it goes on while the screen does other things, and Stop ends it.
// Python copies and checks every file; Go shows what it said.

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// fetchJob is the fetch's job id: one fetch at a time. It copies files only, so it never holds the GPU (gpuJobs).
const fetchJob = "fetch"

// FetchForm is Fetch runs: the remote as typed, what it holds, the runs picked, and the fetch job.
type FetchForm struct {
	Dest, Port string           // user@host or a Host alias; the remote's SSH port, "" for ssh's own
	Traces     bool             // also copy raw traces (--with-traces); they stay behind unless picked
	Listing    bool             // the list call is out
	List       *seam.RemoteRuns // the runs the remote holds
	Picked     map[string]bool  // the runs picked to fetch
	Err        string           // why the list or the fetch failed, in words
	Job        *jobs.Job        // the fetch job; Alive while it copies
	Stopped    bool             // Stop was pressed on this fetch
	Done       *seam.RemoteRuns // what the last fetch did
}

type remoteMsg struct {
	list *seam.RemoteRuns
	err  error
}

type fetchJobMsg struct {
	job  *jobs.Job
	tail []string
}

func (f Facts) fetching() bool { return f.Fetch.Job != nil && f.Fetch.Job.Alive }

// gpuJobs are the jobs alive that hold the GPU: every job but a fetch.
func gpuJobs(s jobs.Store) []jobs.Job {
	return slices.DeleteFunc(s.Alive(), func(j jobs.Job) bool { return j.ID == fetchJob })
}

func (f Facts) fetchPort() int { p, _ := strconv.Atoi(f.Fetch.Port); return p }

// picked are the picked runs, in the remote's order.
func (f Facts) picked() []string {
	out := []string{}
	if f.Fetch.List != nil {
		for _, r := range f.Fetch.List.Runs {
			if f.Fetch.Picked[r.ID] {
				out = append(out, r.ID)
			}
		}
	}
	return out
}

// listRemote asks the remote what runs it holds.
func (m Model) listRemote() tea.Cmd {
	client, dest, port := m.client, m.f.Fetch.Dest, m.f.fetchPort()
	return func() tea.Msg { r, err := client.RemoteRuns(dest, port); return remoteMsg{r, err} }
}

// startFetch starts the fetch of ids (none: every run) as the fetch job.
func (m Model) startFetch(ids []string) tea.Cmd {
	argv, store, repo := m.client.FetchArgv(m.f.Fetch.Dest, m.f.fetchPort(), ids, m.f.Fetch.Traces), m.store, m.client.Repo
	return func() tea.Msg {
		job, err := store.Start(fetchJob, repo, argv)
		if err != nil {
			return noteMsg("The fetch did not start: " + err.Error())
		}
		return fetchJobMsg{&job, nil}
	}
}

// loadFetch reads the fetch job and its whole log: the answer is one JSON object, longer than a tail.
func (m Model) loadFetch() tea.Cmd {
	store := m.store
	return func() tea.Msg {
		job, err := store.Status(fetchJob)
		if err != nil {
			return fetchJobMsg{nil, nil}
		}
		lines, _ := store.Tail(fetchJob, 0)
		return fetchJobMsg{&job, lines}
	}
}

func (m Model) stopFetch() tea.Cmd {
	store := m.store
	return func() tea.Msg {
		if _, err := store.Stop(fetchJob); err != nil {
			return noteMsg("Stop failed: " + err.Error())
		}
		return noteMsg("Sent SIGTERM to the fetch. Runs already in saved runs stay; the run in flight is not kept.")
	}
}

// fetchAnswer is a fetch job's message: while it copies the screen ticks; when it ends, its answer (or why there is
// none) is shown and the saved runs are read again.
func (m Model) fetchAnswer(msg fetchJobMsg) (Model, tea.Cmd) {
	was := m.f.fetching()
	m.f.Fetch.Job = msg.job
	if m.f.fetching() && m.form != nil { // a fetch from an earlier session is still copying: Stop, not the form
		m.form = nil
	}
	switch {
	case m.f.fetching() && !m.ticking:
		m.ticking = true
		return m, tick()
	case !was || m.f.fetching():
		return m, nil
	}
	done, err := seam.ReadFetch(msg.tail)
	switch {
	case m.f.Fetch.Stopped:
		m.f.Fetch.Err = "Stopped. Runs fetched before the stop are in saved runs."
	case err != nil:
		m.f.Fetch.Err = err.Error() + ". Its log: " + msg.job.LogPath
	default:
		// the listing's here/not here is from before the copy: List again reads it anew
		m.f.Fetch.Done, m.f.Fetch.List, m.f.Fetch.Picked, m.note = done, nil, nil, "The fetch ended; Saved runs lists what was fetched."
	}
	if m.cursor == pageFetch && m.form == nil { // the remote again, for the next fetch
		return m, tea.Batch(m.loadSaved(), m.openRemote())
	}
	return m, m.loadSaved()
}

// doFetch is one row of the page.
func (m Model) doFetch(a action) (Model, tea.Cmd) {
	switch a.do {
	case "fetchgo":
		ids := m.f.picked()
		if m.f.fetching() || (a.arg == "picked" && len(ids) == 0) {
			return m, nil
		}
		if a.arg == "all" {
			ids = nil
		}
		m.f.Fetch.Err, m.f.Fetch.Done, m.f.Fetch.Stopped = "", nil, false
		m.note = "Fetching from " + m.f.Fetch.Dest + ": every file is checked by sha256 before it is kept…"
		return m, m.startFetch(ids)
	case "fetchstop":
		m.f.Fetch.Stopped = true
		return m, m.stopFetch()
	}
	return m, nil
}

var remoteState = map[string]string{"new": "not here", "here": "already here", "conflict": "a different run here under this id"}

// size is a byte count in the CLI's words (cli/fetch._size).
func size(n int64) string {
	v := float64(n)
	for _, unit := range []string{"B", "KB", "MB", "GB"} {
		if v < 1024 || unit == "GB" {
			if unit == "B" {
				return fmt.Sprintf("%.0f B", v)
			}
			return fmt.Sprintf("%.1f %s", v, unit)
		}
		v /= 1024
	}
	return ""
}

func fetchLine(f Facts) (string, string) {
	switch l := f.Fetch.List; {
	case f.fetching():
		return "run", "fetching from " + f.Fetch.Dest + "…"
	case l != nil:
		return "pass", fmt.Sprintf("%d runs on %s in %s", len(l.Runs), l.Dest, l.RemoteRoot)
	}
	return "", ""
}

func fetchBody(f Facts, width int) string {
	var b strings.Builder
	b.WriteString("Fetch runs from another machine (a pod, a Linux box) over SSH, as boltbeam fetch does.\n\n" +
		"Each run lands in saved runs with its chip profile, every file checked by sha256. Nothing on the remote changes.")
	switch {
	case f.fetching():
		since := ""
		if t, err := time.Parse(time.RFC3339, f.Fetch.Job.StartedAt); err == nil {
			since = fmt.Sprintf(" (%ds)", int(clock().Sub(t).Seconds()))
		}
		b.WriteString("\n\n" + strings.TrimSpace(f.Spin+" Fetching from "+f.Fetch.Dest+since+"…") + "\n" +
			stMuted.Render("Stop the fetch keeps the runs already in saved runs."))
	case f.Fetch.Err != "":
		b.WriteString("\n\n" + stWarn.Render(glyphWarn+" "+f.Fetch.Err))
	case f.Fetch.Done != nil:
		b.WriteString("\n\n" + stHeader.Render("What was fetched") + "\n" + strings.Join(f.Fetch.Done.Lines, "\n"))
	}
	return b.String()
}

// fetchActions are the page's rows while no form is open: Stop while it copies, the listing while it is out.
func fetchActions(f Facts) []action {
	back := action{"Back to saved runs", "page", fmt.Sprint(pageSaved)}
	switch {
	case f.fetching():
		return []action{{"Stop the fetch", "fetchstop", ""}, back}
	case f.Fetch.Listing:
		return []action{{stMuted.Render(strings.TrimSpace(f.Spin + " listing the runs on " + f.Fetch.Dest + "…")), "", ""}, back}
	}
	return []action{back}
}
