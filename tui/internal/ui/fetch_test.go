package ui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func recordedPath(name string) string {
	p, _ := filepath.Abs(filepath.Join("..", "..", "testdata", "recorded", name))
	return p
}

// fakeFetch is a python that answers `boltbeam.cli fetch` with a recorded answer: the listing without run ids or
// --all, else the fetch.
func fakeFetch(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	calls := filepath.Join(dir, "calls.txt")
	script := "#!/bin/sh\necho \"$@\" >> " + calls + "\ncase \"$*\" in *--all*) cat " + recordedPath("fetch-done.json") +
		";; *) cat " + recordedPath("fetch-list.json") + ";; esac\n"
	fake := filepath.Join(dir, "python")
	if err := os.WriteFile(fake, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return fake, calls
}

// Saved runs offers Fetch runs; its page is a form for the remote and a port, each checked as typed; the
// remote's runs are listed, some picked in a second form with the raw traces, and fetched; a listing that fails says
// why in words and asks for the remote again.
func TestFetchRunsListsAndPicks(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	fake, calls := fakeFetch(t)
	s := loadSample(t)
	store := jobs.Store{Dir: t.TempDir()}
	var tm tea.Model = New(seam.Client{Python: fake, Root: "/home/u/runs", Saved: "/home/u/runs/saved"}, store, "", "", 512)
	for _, msg := range []tea.Msg{targetsMsg{&s.targets, nil}, savedListMsg{savedSample(), nil}, tea.WindowSizeMsg{Width: 80, Height: 24}} {
		tm, _ = tm.Update(msg)
	}
	m := page(tm, pageSaved).(Model)
	acts := m.actions()
	if got := rows(acts)[4]; got != "page|Fetch runs from another machine\t›" {
		t.Fatalf("Saved runs: %v", rows(acts))
	}
	m = typeKeys(keys(m, "j", "j", "j"), "enter")
	if m.cursor != pageFetch || m.form == nil || m.form.kind != "remote" {
		t.Fatalf("enter opens Fetch runs on its form: page %d", m.cursor)
	}
	golden(t, "fetch-empty.txt", frame(m))
	if m = typeKeys(m, "enter"); !strings.Contains(frame(m), "type the remote first") {
		t.Fatalf("an empty remote is refused:\n%s", frame(m))
	}
	if m = typeKeys(m, "root@ 203", "enter"); !strings.Contains(frame(m), "type the remote as user@host, or an ssh Host alias") {
		t.Fatalf("a space is refused:\n%s", frame(m))
	}
	m = typeKeys(m, "backspace", "backspace", "backspace", "backspace", "203.0.113.7", "enter", "x", "enter")
	if !strings.Contains(frame(m), "a port is a number from 1 to 65535") {
		t.Fatalf("a port is a number:\n%s", frame(m))
	}
	m = typeKeys(m, "backspace", "40022", "enter")
	if m.f.Fetch.Dest != "root@203.0.113.7" || m.f.Fetch.Port != "40022" || m.f.Fetch.List == nil || len(m.f.Fetch.List.Runs) != 2 {
		t.Fatalf("typed and listed: %+v", m.f.Fetch)
	}
	if got, _ := os.ReadFile(calls); string(got) != "-m boltbeam.cli fetch root@203.0.113.7 -p 40022 --into /home/u/runs --saved /home/u/runs/saved --json\n" {
		t.Fatalf("the listing call: %q", got)
	}
	if m.form == nil || m.form.kind != "pick" {
		t.Fatal("the listing asks which runs come home")
	}
	m = typeKeys(m, "down", "x")
	golden(t, "fetch-listed.txt", frame(m))
	m = typeKeys(m, "enter", "right", "enter") // the traces come too
	for i := 0; i < 100 && len(store.Alive()) > 0; i++ {
		time.Sleep(20 * time.Millisecond)
	}
	if got, _ := os.ReadFile(calls); !strings.HasSuffix(string(got), "--saved /home/u/runs/saved --json --with-traces h100-generic\n") {
		t.Fatalf("the fetch call: %q", got)
	}
	listing := m
	listing.form, listing.f.Fetch.Job, listing.f.Fetch.Listing = nil, nil, true
	failed, _ := listing.update(remoteMsg{nil, &seam.Error{Message: "ssh to nobody@127.0.0.1 failed (exit 255): ssh: connect to host 127.0.0.1 port 1: Connection refused"}})
	if !strings.Contains(plain(fetchBody(failed.f, 76)), "⚠ Could not list the runs: ssh to nobody@127.0.0.1 failed (exit 255): ssh: connect") ||
		failed.f.Fetch.List != nil || failed.form == nil || failed.form.kind != "remote" {
		t.Fatalf("a refused ssh in words, then the remote again: %s", plain(fetchBody(failed.f, 76)))
	}
}

// Fetch runs as a job: the screen ticks while it copies, Stop is offered, and when it ends the answer's words show
// what came home, verified and where, and the saved runs are read again.
func TestFetchRunsFetchIsAJob(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	fake, calls := fakeFetch(t)
	store := jobs.Store{Dir: t.TempDir()}
	m := page(program(loadSample(t), nil, nil, nil, 80, 24), pageFetch).(Model)
	m.client, m.store = seam.Client{Python: fake, Root: "/home/u/runs", Saved: "/home/u/runs/saved"}, store
	m.f.Fetch.Dest = "root@203.0.113.7"
	m, cmd := m.do(action{do: "fetchgo", arg: "all"})
	if cmd == nil {
		t.Fatal("Fetch all starts the job")
	}
	m, tickCmd := m.update(cmd())
	if !m.f.fetching() || tickCmd == nil || !m.ticking {
		t.Fatalf("the job is alive and the screen ticks: %+v", m.f.Fetch.Job)
	}
	if got := rows(fetchActions(m.f)); got[0] != "fetchstop|Stop the fetch" {
		t.Fatalf("while it copies: %v", got)
	}
	running := m
	running.f.Fetch.Job = &jobs.Job{ID: fetchJob, Alive: true, StartedAt: clock().Add(-12 * time.Second).Format(time.RFC3339)}
	if got := plain(fetchBody(running.f, 76)); !strings.Contains(got, "Fetching from root@203.0.113.7 (12s)…") {
		t.Fatalf("progress: %s", got)
	}
	for i := 0; i < 100 && len(store.Alive()) > 0; i++ {
		time.Sleep(20 * time.Millisecond)
	}
	m, reread := m.update(m.loadFetch()())
	if m.f.fetching() || m.f.Fetch.Done == nil || reread == nil || m.f.Fetch.Err != "" {
		t.Fatalf("ended: %+v %q", m.f.Fetch.Done, m.f.Fetch.Err)
	}
	if got, _ := os.ReadFile(calls); !strings.HasSuffix(string(got), "--saved /home/u/runs/saved --json --all\n") {
		t.Fatalf("the fetch call: %q", got)
	}
	golden(t, "fetch-done.txt", frame(m))
	got := plain(fetchBody(m.f, 200))
	for _, want := range []string{"fetched h100-in-model: 5 files, 232 B, checked by sha256 (rsync) -> /home/u/runs/saved/h100-in-model",
		"refused h100-generic: /home/u/runs/saved/h100-generic holds a different run"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q:\n%s", want, got)
		}
	}
	stopped := m
	stopped.f.Fetch.Stopped, stopped.f.Fetch.Job = true, &jobs.Job{ID: fetchJob, Alive: true}
	stopped, _ = stopped.fetchAnswer(fetchJobMsg{&jobs.Job{ID: fetchJob}, nil})
	if stopped.f.Fetch.Err != "Stopped. Runs fetched before the stop are in saved runs." {
		t.Fatalf("stopped: %q", stopped.f.Fetch.Err)
	}
}

// A fetch copies files only: it never makes Run wait for the GPU.
func TestAFetchDoesNotHoldTheGPU(t *testing.T) {
	store := jobs.Store{Dir: t.TempDir()}
	if _, err := store.Start(fetchJob, "", []string{"sleep", "5"}); err != nil {
		t.Fatal(err)
	}
	defer store.Stop(fetchJob)
	if len(store.Alive()) != 1 || len(gpuJobs(store)) != 0 {
		t.Fatalf("alive %d, holding the GPU %d", len(store.Alive()), len(gpuJobs(store)))
	}
	if _, err := seam.ReadFetch(nil); err == nil {
		t.Fatal("no log, no answer")
	}
}
