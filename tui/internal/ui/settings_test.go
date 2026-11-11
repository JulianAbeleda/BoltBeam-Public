package ui

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// vllmSettings is `screen providers` for vLLM: capture_settings.offered("vllm"), the help cut short.
const vllmSettings = `[
 {"name": "chunk_size", "flag": "--chunk-size", "label": "chunk size", "whole": true, "about": "requested engine chunk size"},
 {"name": "gemm_backends", "flag": "--gemm-backends", "label": "matrix backends", "choices": ["ATEN", "TRITON"], "about": "the compiler's matrix backends"},
 {"name": "mtp", "flag": "--mtp", "label": "MTP draft tokens", "whole": true, "about": "draft tokens per step", "draft_head": true}]`

// vllmProgram is this Mac's sample with a vLLM row that takes settings, picked, and Repeats offered at n = 4.
func vllmProgram(t *testing.T) Model {
	t.Helper()
	var settings []seam.EngineSetting
	if err := json.Unmarshal([]byte(vllmSettings), &settings); err != nil {
		t.Fatal(err)
	}
	p := engines()
	p.Providers = append(p.Providers, seam.ProviderRow{Provider: "vllm", Available: true, Settings: settings})
	p.RepeatRuns = 4
	m, _ := program(loadSample(t), nil, nil, nil, 80, 24).Update(providersMsg{"apple_m3_10c", p})
	got := m.(Model)
	got.f.Engine = "vllm"
	return got
}

func keys(m Model, ks ...string) Model {
	var t tea.Model = m
	for _, k := range ks {
		t = press(t, k)
	}
	return t.(Model)
}

// The Settings line lists what the engine takes; a draft-head setting waits for the recommendation to say the
// checkpoint has one. Enter types a whole number (a bad one is refused), steps a fixed set, and Run passes each value
// with its flag. Another engine drops them.
func TestEngineSettingsPageAndRun(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := vllmProgram(t)
	if got := rows(setupActions(m.f)); !slices.Contains(got, "page|Settings           engine defaults\t›") {
		t.Fatalf("setup:\n%s", strings.Join(got, "\n"))
	}
	offered := func(f Facts) (names []string) {
		for _, s := range f.settingsOffered() {
			names = append(names, s.Name)
		}
		return names
	}
	if slices.Contains(offered(m.f), "mtp") {
		t.Fatalf("no recommendation yet, so no draft head is known: %v", offered(m.f))
	}
	m.f.RecKey, m.f.Recommend = m.f.recommendKey(), sampleRecommend()
	m.f.Recommend.Weights.Mtp = 1e9
	if !slices.Contains(offered(m.f), "mtp") {
		t.Fatalf("a draft head offers MTP: %v", offered(m.f))
	}
	m = settle(m.do(action{"", "page", fmt.Sprint(pageSettings)}))
	if m.form == nil || m.form.kind != "settings" {
		t.Fatal("Settings opens its form")
	}
	if m = typeKeys(m, "x", "enter"); !strings.Contains(frame(m), "a whole number above 0, or nothing to unset it") {
		t.Fatalf("a chunk size is a whole number:\n%s", frame(m))
	}
	m = typeKeys(m, "backspace", "512", "enter", "right", "right", "enter")
	golden(t, "picker-settings.txt", frame(m))
	if m = typeKeys(m, "enter"); m.form != nil || m.cursor != pageSetup || m.f.Settings["chunk_size"] != "512" {
		t.Fatalf("kept: %+v", m.f.Settings)
	}
	if got := settingsValue(m.f); got != "chunk size 512 · matrix backends TRITON" {
		t.Fatalf("the Setup value: %q", got)
	}
	dir := t.TempDir()
	m.client = seam.Client{Python: "/usr/bin/true", Root: dir, Repo: dir}
	m.store = jobs.Store{Dir: filepath.Join(dir, "state")}
	msg := m.startRun("vllm")()
	job, err := m.store.Status(string(msg.(startedMsg)))
	if err != nil || !strings.HasSuffix(strings.Join(job.Argv, " "), "--chunk-size 512 --gemm-backends TRITON") {
		t.Fatalf("argv: %v %v", job.Argv, err)
	}
	if m, _ = m.do(action{do: "engine", arg: "llama.cpp"}); m.f.Settings != nil {
		t.Fatalf("another engine's settings: %v", m.f.Settings)
	}
}

// Repeats offers one run or Python's n; Run asks for that many, and the Run page shows the band Python worded.
func TestRepeatsPickerRunAndBand(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	m := vllmProgram(t)
	if got := rows(setupActions(m.f)); !slices.Contains(got, "page|Repeats            1 run\t›") {
		t.Fatalf("setup:\n%s", strings.Join(got, "\n"))
	}
	if got := rows(repeatsActions(m.f)); !slices.Equal(got, []string{"repeats|● 1 run", "repeats|4 runs, then their repeat band"}) {
		t.Fatalf("picker: %v", got)
	}
	m.cursor = pageRepeats
	golden(t, "picker-repeats.txt", frame(m))
	m, _ = m.do(action{do: "repeats", arg: "4"})
	if m.cursor != pageSetup || m.f.repeats() != 4 || setupActions(m.f)[m.row].arg != fmt.Sprint(pageRepeats) {
		t.Fatalf("picked 4: cursor %d row %d repeats %d", m.cursor, m.row, m.f.repeats())
	}
	if m.f.Repeats = 3; m.f.repeats() != 1 {
		t.Fatal("only the n Python offers counts")
	}
	var res seam.Results
	if err := json.Unmarshal([]byte(`{"repeats": {"n": 4, "false_move_rate": 0.4, "pin": null,
	  "lines": ["decode 512: median 21.300 ms, 21.100 to 21.600 ms over 4 runs (spread 2.3%)"]}}`), &res); err != nil {
		t.Fatal(err)
	}
	got := plain(repeatsResult(res))
	for _, want := range []string{"Repeats: 4 runs of this setup. A row reads moved by chance 40% of the time.", "decode 512: median 21.300 ms"} {
		if !strings.Contains(got, want) || strings.Contains(got, "pin") {
			t.Fatalf("band: %q", got)
		}
	}
}

// Measure MTP is offered when Python says so (vLLM, a draft head), muted with its reason when blocked; the kept
// measurement shows per k; the job's end reads the recommendation again.
func TestMeasureMtp(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	f := Facts{Recommend: sampleRecommend()}
	if got := rows(recommendActions(f)); len(got) != 3 {
		t.Fatalf("no offer, no row: %v", got)
	}
	if err := json.Unmarshal([]byte(`{"mtp_measure": {"ks": [1, 2, 3], "blocked": "needs this machine's facts: run once on this chip first"},
	  "mtp_measured": {"measured_at": "2026-11-08T10:00:00+00:00", "depths": {"short": {
	    "0": {"median_tok_s": 60.0, "context": 300, "spread": 0.05, "speedup": 1.0},
	    "2": {"median_tok_s": 92.4, "context": 300, "spread": 0.12, "speedup": 1.54, "accept_length": 2.47, "accept_rate": 0.735}}}}}`), f.Recommend); err != nil {
		t.Fatal(err)
	}
	if got := rows(recommendActions(f)); got[3] != "note|[ Measure MTP ] needs this machine's facts: run once on this chip first" {
		t.Fatalf("blocked: %v", got)
	}
	f.Recommend.MtpMeasure.Blocked = nil
	if got := rows(recommendActions(f)); got[3] != "mtp|[ Measure MTP ] decode with 1, 2, 3 draft tokens against none" {
		t.Fatalf("offered: %v", got)
	}
	body := plain(recommendBody(f, 120))
	for _, want := range []string{"MTP measured 2026-11-08 10:00 UTC", "short, context 300, k=0: 60.0 tok/s, 1.00x k=0 (prompts spread 5%)",
		"short, context 300, k=2: 92.4 tok/s, 1.54x k=0, 2.47 tokens per step, 74% of drafts kept"} {
		if !strings.Contains(body, want) {
			t.Fatalf("lacks %q:\n%s", want, body)
		}
	}
	m := Model{f: f}
	m.f.MJob = &jobs.Job{ID: "q-mtp", Alive: true}
	if got := rows(mtpAction(m.f)); !strings.Contains(got[0], "measuring MTP") {
		t.Fatalf("while it runs: %v", got)
	}
	m.f.RecKey = "old"
	m, _ = m.update(mtpJobMsg{&jobs.Job{ID: "q-mtp", LogPath: "/s/q-mtp.log"}, false})
	if m.f.RecKey != "" || m.note != "Measure MTP ended. Its log: /s/q-mtp.log" {
		t.Fatalf("ended: %q %q", m.f.RecKey, m.note)
	}
}

// Recommended setup sits above Settings: the recommendation proposes the context, speed and engine flags, and
// Settings, under it, changes them.
func TestRecommendedSetupAboveSettings(t *testing.T) {
	setProfile(colorprofile.NoTTY)
	got := rows(setupActions(vllmProgram(t).f))
	want := []string{"page|Machine            checking this Mac…\t›", "head|", "page|Model              Qwen3-8B.gguf · 36 layers · F32/Q4_K/Q6_K\t›", "page|Chip               apple_m3_10c · this Mac · 97.2 GB/s\t›",
		"page|Engine             vllm\t›", "head|", "page|Recommended setup  context, speed and engine flags for this model on your GPUs\t›",
		"page|Settings           engine defaults\t›", "page|Offload            working out the plan, a few seconds…\t›", "head|", "page|Workload           decode\t›"}
	if strings.Join(got[:len(want)], "\n") != strings.Join(want, "\n") {
		t.Fatalf("got\n%s", strings.Join(got, "\n"))
	}
}
