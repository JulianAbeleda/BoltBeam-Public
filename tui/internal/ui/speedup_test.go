package ui

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func recordedAs[T any](t *testing.T, name string) *T {
	t.Helper()
	raw, err := os.ReadFile(recordedPath(name))
	if err != nil {
		t.Fatal(err)
	}
	v := new(T)
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
	return v
}

// speedProgram is the vLLM sample on the Recommended setup page with Python's offer for the model in hand.
func speedProgram(t *testing.T, mutate func(*seam.SpeedOffer)) Model {
	t.Helper()
	m := vllmProgram(t)
	m.f.RecKey, m.f.Recommend = m.f.recommendKey(), sampleRecommend()
	o := recordedAs[seam.SpeedOffer](t, "speedup-offer.json")
	if mutate != nil {
		mutate(o)
	}
	m.f.Speed = SpeedUp{OfferFor: m.f.Path, Offer: o}
	m.cursor = pageRecommend
	return m
}

func speedRow(m Model, do string) int {
	for i, a := range m.actions() {
		if a.do == do {
			return i
		}
	}
	return -1
}

func off(reason string) func(*seam.SpeedOffer) {
	return func(o *seam.SpeedOffer) {
		o.Enabled, o.Reason, o.Confirm, o.Expect, o.CopyBytes = false, &reason, nil, nil, nil
	}
}

// Speed up sits under Measure MTP on the Recommended setup page; it is a button only when the model's format has a
// writer, else a muted row with the plain reason; a copy says so and offers Check this copy.
func TestSpeedUpRowBesideMeasureMtp(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := speedProgram(t, nil)
	if err := json.Unmarshal([]byte(`{"mtp_measure": {"ks": [1, 2, 3]}}`), m.f.Recommend); err != nil {
		t.Fatal(err)
	}
	got := rows(m.actions())
	if n := len(got); got[n-2] != "mtp|[ Measure MTP ] decode with 1, 2, 3 draft tokens against none" || got[n-1] != "speedup|[ Speed up ] Makes a faster copy. The original stays." {
		t.Fatalf("enabled, under Measure MTP:\n%s", strings.Join(got, "\n"))
	}
	golden(t, "speedup-row.txt", frame(m))

	m = speedProgram(t, off("This model's format has no faster copy yet."))
	if got := rows(m.actions()); got[len(got)-1] != "note|[ Speed up ] This model's format has no faster copy yet." || speedRow(m, "speedup") >= 0 {
		t.Fatalf("disabled, with its reason:\n%s", strings.Join(got, "\n"))
	}
	m = speedProgram(t, func(o *seam.SpeedOffer) { off("This model is already a faster copy.")(o); o.IsCopy = true })
	got = rows(m.actions())
	if n := len(got); got[n-2] != "note|[ Speed up ] This model is already a faster copy." || !strings.HasPrefix(got[n-1], "speedcheck|[ Check this copy ]") {
		t.Fatalf("a copy:\n%s", strings.Join(got, "\n"))
	}
	if m.f.Path = ""; speedActions(m.f) != nil {
		t.Fatal("no model, no row")
	}
}

// Before it runs, a confirm step says what happens, the disk space, the time and the GPU. It opens on Not now: enter
// starts nothing. Make the copy starts the job on a free GPU, and a GPU already busy says so.
func TestSpeedUpConfirmStartsOnlyOnYes(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	dir := t.TempDir()
	calls := filepath.Join(dir, "calls.txt")
	fake := filepath.Join(dir, "python")
	if err := os.WriteFile(fake, []byte("#!/bin/sh\ncase \"$*\" in *\" offload \"*|*\" tune \"*) ;; *) echo \"$@\" >> "+calls+" ;; esac\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	store := jobs.Store{Dir: filepath.Join(dir, "state")}
	m := speedProgram(t, nil)
	m.client, m.store = seam.Client{Python: fake, Repo: dir}, store
	m.row = speedRow(m, "speedup")
	m = typeKeys(m, "enter")
	if m.form == nil || m.form.kind != "speedup" {
		t.Fatalf("enter opens the confirm step: %v", m.form)
	}
	view := frame(m)
	golden(t, "speedup-confirm.txt", view)
	for _, want := range []string{"Make a faster copy of", "Makes a faster copy. The original stays.", "Needs up to 12.1 GB of free disk space", "Takes about 18 minutes (an estimate).",
		"Uses the GPU the whole time.", "Make the copy", "Not now"} {
		if !strings.Contains(view, want) {
			t.Fatalf("the confirm lacks %q:\n%s", want, view)
		}
	}
	if m = typeKeys(m, "enter"); m.form != nil || len(store.Alive()) > 0 || m.f.Speed.Job != nil {
		t.Fatal("enter on the default, Not now, starts nothing")
	}
	if m = typeKeys(m, "enter", "esc"); m.form != nil || m.cursor != pageRecommend {
		t.Fatalf("esc is Not now and stays on the page: %d", m.cursor)
	}
	m = typeKeys(m, "enter", "left", "enter")
	for i := 0; i < 100 && len(store.Alive()) > 0; i++ {
		time.Sleep(20 * time.Millisecond)
	}
	want := "-m boltbeam.cli derive lighter-head " + m.f.Path + " --json --progress\n"
	if got, _ := os.ReadFile(calls); string(got) != want || m.form != nil || m.f.Speed.Job == nil {
		t.Fatalf("Make the copy starts the job: %q", got)
	}
	busy := speedProgram(t, nil)
	busy.client, busy.store = seam.Client{Python: fake, Repo: dir}, jobs.Store{Dir: filepath.Join(dir, "busy")}
	if _, err := busy.store.Start("a-run", dir, []string{"sleep", "30"}); err != nil {
		t.Fatal(err)
	}
	defer busy.store.Stop("a-run")
	if busy = settle(busy, busy.startSpeed()); !strings.Contains(busy.note, "a-run is still going") {
		t.Fatalf("a busy GPU is said: %q", busy.note)
	}
}

const speedStart = "2026-10-08T10:00:00Z"

func speedRunning(t *testing.T, tail ...string) Model {
	t.Helper()
	m := speedProgram(t, nil)
	job := &jobs.Job{ID: m.f.speedID(), Alive: true, StartedAt: speedStart, LogPath: "/s/speedup.log"}
	head := []string{"=== " + speedStart + " start python3 -m boltbeam.cli derive", "pipeline steps: 3", "pipeline expect: 70.0,300.0,690.0", "pipeline expect source: estimate"}
	next, _ := m.update(speedJobMsg{m.f.speedID(), job, append(head, tail...)})
	return next
}

// While it runs the stages are plain words with the time left; the row says the same and offers Stop; the screen ticks.
func TestSpeedUpRunningShowsStagesAndTimeLeft(t *testing.T) {
	freezeClock(t)
	setProfile(colorprofile.NoTTY)
	m := speedRunning(t, "stage speed_write: start", "stage speed_write: done", "stage speed_check: start")
	body := plain(speedBody(m.f, 80))
	for _, want := range []string{"Speed up", "01:20", "7%", "2 of 3", "Check the answers…", "01:20 · about 17 min left (an estimate)", "✓ Write the faster copy"} {
		if !strings.Contains(body, want) {
			t.Fatalf("running lacks %q:\n%s", want, body)
		}
	}
	got := rows(m.actions())
	if n := len(got); !strings.HasPrefix(got[n-2], "note|") || !strings.Contains(got[n-2], "Making the faster copy: Check the answers…") || !strings.HasPrefix(got[n-1], "speedstop|[ Stop ]") {
		t.Fatalf("rows while it runs:\n%s", strings.Join(got, "\n"))
	}
	golden(t, "speedup-running.txt", frame(m))
	for _, jargon := range []string{"lm_head", "int4", "speed_check"} {
		if strings.Contains(frame(m), jargon) {
			t.Fatalf("%q in layer 1", jargon)
		}
	}
	m = speedRunning(t, "stage speed_write: start", "stage speed_write: done", "stage speed_check: start", "stage speed_check: done", "stage speed_measure: start")
	if body := plain(speedBody(m.f, 80)); !strings.Contains(body, "Measure the speed…   01:20 · about 12 min left") || !strings.Contains(body, "Write the faster copy · Check the answers") {
		t.Fatalf("the third stage:\n%s", body)
	}
}

// A job that ends is worded by Python once: the kept copy leads with its figures, a copy that was not kept says so
// loudly and says it was deleted, a refusal gives its plain reason and the one fix, a missing speed is never a speed.
func TestSpeedUpResultsInPlainWords(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	cases := []struct {
		file, golden string
		want         []string
	}{
		{"kept", "speedup-kept.txt", []string{"✓ Made m · BoltBeam.", "Short prompt, 128 tokens: 100.0 to 112.0 tok/s (+12%)", "Long prompt, 32768 tokens: 80.0 to 86.0 tok/s (+8%)",
			"Scores the same on the answer check.", "It shows in model lists as m · BoltBeam."}},
		{"not-kept", "speedup-not-kept.txt", []string{"✗ NOT KEPT: the faster copy was deleted.", "It was not faster than the original with a 32768-token prompt: 80.0 to 79.0 tok/s.", "The original is untouched."}},
		{"refused", "speedup-refused.txt", []string{"⚠ No copy was made.", "An earlier copy stopped half way.", "Fix: Delete the half-written copy, then try again."}},
		{"kept-not-measured", "speedup-not-measured.txt", []string{"✓ Made m · BoltBeam.", "SPEED NOT MEASURED: the speed was not measured, so this copy is not known to be faster."}},
		{"crashed", "", []string{"✗ The run stopped without an answer.", "Fix: Read the log, delete any half-written copy, and try again."}},
	}
	for _, c := range cases {
		m := speedProgram(t, nil)
		m.f.Speed.Result = recordedAs[seam.SpeedPlain](t, "speedup-"+c.file+".json")
		body := flat(plain(speedBody(m.f, 80)))
		for _, want := range c.want {
			if !strings.Contains(body, want) {
				t.Fatalf("%s lacks %q:\n%s", c.file, want, body)
			}
		}
		if c.golden != "" {
			golden(t, c.golden, frame(m))
		}
		if c.file == "kept-not-measured" && strings.Contains(body, "tok/s") {
			t.Fatalf("a speed that was not measured is never drawn as a speed:\n%s", body)
		}
		for _, jargon := range []string{"lm_head", "int4", "output layer"} {
			if strings.Contains(body, jargon) {
				t.Fatalf("%s: %q in layer 1", c.file, jargon)
			}
		}
	}
}

// The end of a job asks Python for the words once; a kept copy refreshes the model lists and offers What next:
// Measure MTP on the copy (the copy becomes the model) and Compare with the original (Saved runs' head to head).
func TestSpeedUpEndsWithWordsAndWhatNext(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	dir := t.TempDir()
	fake := filepath.Join(dir, "python")
	script := "#!/bin/sh\ncase \"$*\" in *\" offload \"*|*\" tune \"*) ;; *) echo \"$@\" >> " + filepath.Join(dir, "calls.txt") + " ;; esac\ncat " + recordedPath("speedup-kept.json") + "\n"
	if err := os.WriteFile(fake, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	m := speedRunning(t, "stage speed_write: start")
	m.client = seam.Client{Python: fake}
	m.f.Speed.Asked = ""
	ended := &jobs.Job{ID: m.f.speedID(), Alive: false, StartedAt: speedStart, LogPath: "/s/speedup.log"}
	m = settle(m, func() tea.Msg { return speedJobMsg{m.f.speedID(), ended, m.f.Speed.Tail} })
	if m.f.Speed.Result == nil || m.f.Speed.Result.Outcome != "kept" || m.note != "Speed up ended." {
		t.Fatalf("the ended job is worded: %+v %q", m.f.Speed.Result, m.note)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "calls.txt")); string(got) != "-m boltbeam.workflow.screen speedup-plain /s/speedup.log\n" {
		t.Fatalf("asked once: %q", got)
	}
	got := rows(m.actions())
	n := len(got)
	if got[n-2] != "speedmtp|[ Measure MTP on the copy ]" || got[n-1] != "comparepick|[ Compare with the original ]" {
		t.Fatalf("What next:\n%s", strings.Join(got, "\n"))
	}
	golden(t, "speedup-next.txt", frame(m))
	cmp, _ := m.do(action{"", "comparepick", ""})
	if !cmp.f.ComparePick || cmp.cursor != pageSaved {
		t.Fatalf("Compare opens Saved runs to pick two: %d", cmp.cursor)
	}
	mtp, _ := m.do(action{"", "speedmtp", ""})
	if mtp.f.Path != "/models/m-boltbeam" || !strings.Contains(mtp.note, "faster copy") {
		t.Fatalf("Measure MTP is on the copy: %q", mtp.f.Path)
	}
}

// Check this copy shows pass, or the plain mismatch reason and the fix.
func TestSpeedUpChecksACopy(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := speedProgram(t, func(o *seam.SpeedOffer) { off("This model is already a faster copy.")(o); o.IsCopy = true })
	m.f.Speed.Checking = true
	if got := rows(m.actions()); !strings.Contains(got[len(got)-1], "Checking the copy") {
		t.Fatalf("while it checks: %v", got)
	}
	var v seam.SpeedVerify
	_ = json.Unmarshal([]byte(`{"ok": true, "head": "This copy is good: its files and its answer check still match what was recorded.", "lines": []}`), &v)
	m, _ = m.update(speedVerifyMsg{m.f.Path, &v, nil})
	if body := plain(speedBody(m.f, 80)); m.f.Speed.Checking || !strings.Contains(body, "✓ This copy is good") {
		t.Fatalf("pass:\n%s", body)
	}
	fix := "Fix: Make the copy again from the original."
	v = seam.SpeedVerify{Ok: false, Head: "This copy does not match its record.", Lines: []string{"A file of this copy changed after it was made."}, Fix: &fix}
	m, _ = m.update(speedVerifyMsg{m.f.Path, &v, nil})
	body := plain(speedBody(m.f, 80))
	for _, want := range []string{"✗ This copy does not match its record.", "A file of this copy changed after it was made.", fix} {
		if !strings.Contains(body, want) {
			t.Fatalf("mismatch lacks %q:\n%s", want, body)
		}
	}
	golden(t, "speedup-check.txt", frame(m))
}

// The check asks Python for the copy named, and the answer is Python's words.
func TestSpeedUpCheckCallsTheSeam(t *testing.T) {
	dir := t.TempDir()
	fake := filepath.Join(dir, "python")
	script := "#!/bin/sh\ncase \"$*\" in *\" offload \"*|*\" tune \"*) ;; *) echo \"$@\" > " + filepath.Join(dir, "calls.txt") + " ;; esac\necho '{\"kind\": \"speedup_verify\", \"ok\": true, \"head\": \"fine\", \"lines\": []}'\n"
	if err := os.WriteFile(fake, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	m := speedProgram(t, func(o *seam.SpeedOffer) { o.IsCopy, o.Enabled = true, false })
	m.client = seam.Client{Python: fake}
	m, _ = m.do(action{"", "speedcheck", ""})
	m = settle(m, m.checkCopy())
	if got, _ := os.ReadFile(filepath.Join(dir, "calls.txt")); string(got) != "-m boltbeam.workflow.screen speedup-verify "+m.f.Path+"\n" || m.f.Speed.Verify == nil || !m.f.Speed.Verify.Ok {
		t.Fatalf("the check: %q %+v", got, m.f.Speed.Verify)
	}
}

// The offer is asked once per model; another model asks again and drops the last model's job and result.
func TestSpeedUpOfferIsPerModel(t *testing.T) {
	m := speedProgram(t, nil)
	m.f.Speed.Result = recordedAs[seam.SpeedPlain](t, "speedup-kept.json")
	if cmd := m.loadSpeed(); cmd != nil {
		t.Fatal("the same model is not asked twice")
	}
	m.f.Path = "/models/other"
	if cmd := m.loadSpeed(); cmd == nil || m.f.Speed.Result != nil || m.f.Speed.Offer != nil || m.f.Speed.OfferFor != "/models/other" {
		t.Fatalf("another model asks again: %+v", m.f.Speed)
	}
	next, _ := m.update(speedOfferMsg{"/models/old", recordedAs[seam.SpeedOffer](t, "speedup-offer.json"), nil})
	if next.f.Speed.Offer != nil {
		t.Fatal("an answer for another model is dropped")
	}
}

func TestSpeedUpStageWordsComeFromPython(t *testing.T) {
	for key, want := range map[string]string{"speed_write": "Write the faster copy", "speed_check": "Check the answers", "speed_measure": "Measure the speed"} {
		if got := stageWord(key); got != want {
			t.Errorf("%s: %q", key, got)
		}
	}
	if !slices.Contains([]string{"Speed up"}, vocab.SpeedUp.Action) {
		t.Fatal("the action's name is Python's")
	}
}

// A copy is a model in the lists, as "<model> · BoltBeam", a folder or a file; the original keeps its own label.
func TestSpeedUpCopyShowsInModelLists(t *testing.T) {
	dir := t.TempDir()
	folder := filepath.Join(dir, "Qwen3-14B-AWQ-boltbeam")
	if err := os.MkdirAll(folder, 0o755); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "tiny-boltbeam.gguf")
	for path, record := range map[string]string{folder: filepath.Join(folder, "speed_up.json"), file: file + ".speed_up.json"} {
		if path == file {
			_ = os.WriteFile(file, []byte("x"), 0o644)
		}
		if err := os.WriteFile(record, []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if got := modelLabel(folder); got != "Qwen3-14B-AWQ · BoltBeam" {
		t.Fatalf("folder: %q", got)
	}
	if got := modelLabel(file); got != "tiny · BoltBeam" {
		t.Fatalf("file: %q", got)
	}
	plainFile := filepath.Join(dir, "tiny.gguf")
	_ = os.WriteFile(plainFile, []byte("x"), 0o644)
	if got := modelLabel(plainFile); got != "tiny · GGUF" {
		t.Fatalf("the original: %q", got)
	}
}

// The demo the speedup tape plays is whole: the page offers Speed up, the confirm starts the recorded job, its log draws the
// stages, and its end shows the kept copy's words. Nothing runs but the shell.
func TestSpeedUpDemoPlaysTheTape(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	d, err := seam.LoadDemo(filepath.Join("..", "..", "testdata", "demo", "speedup.json"))
	if err != nil {
		t.Fatal(err)
	}
	d.Argv["speedup"][2] = strings.ReplaceAll(d.Argv["speedup"][2], "sleep 3", "sleep 0.1") // the tape waits; the test need not
	store := jobs.Store{Dir: t.TempDir()}
	client := seam.Client{Python: "boltbeam-demo-runs-nothing", Root: "runs", Work: "runs/.work", Saved: "runs/saved", Demo: d}
	m := New(client, store, d.Model, d.Target, 512)
	m = settle(m, m.Init())
	next, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = settle(next.(Model), nil)
	m = typeKeys(m, "j", "j", "j", "j", "enter")
	if m.cursor != pageRecommend || speedRow(m, "speedup") < 0 {
		t.Fatalf("the page offers Speed up:\n%s", frame(m))
	}
	m.row = speedRow(m, "speedup")
	m = typeKeys(m, "enter", "left", "enter")
	if !m.f.speeding() {
		t.Fatalf("the confirm starts the recorded job:\n%s", frame(m))
	}
	for i := 0; i < 200 && len(store.Alive()) > 0; i++ {
		time.Sleep(50 * time.Millisecond)
		m = settle(m, m.readSpeed())
	}
	m = settle(m, m.readSpeed()) // the job can end between the last read and the loop's check: read the ended job once
	if m.f.speeding() || m.f.Speed.Result == nil || !strings.Contains(plain(speedBody(m.f, 80)), "✓ Made m · BoltBeam.") {
		t.Fatalf("the end shows the kept copy:\n%s", frame(m))
	}
}
