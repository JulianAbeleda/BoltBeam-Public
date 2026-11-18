package ui

import (
	"encoding/json"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// modsProviders is familyProviders with llama.cpp able to take mods here and, when key is set, its modded build as a fork.
func modsProviders(t *testing.T, key string) *seam.Providers {
	t.Helper()
	p := familyProviders(t)
	p.Providers[0].Mods = true
	if key != "" {
		path, prov, reason := "/e/llama.cpp/"+key+"/build/bin/llama-bench", "llama.cpp", "12 mods, applied, not verified"
		fork := seam.Fork{Fork: key, Label: "with mods", Provider: &prov, Runs: true, State: "available", Path: &path, Reason: &reason, Modded: &key}
		p.Families[0].Forks = append(p.Families[0].Forks, fork)
	}
	return p
}

// modsListing is `screen mods` on a build at the mods' base: one chosen, one not, one with no switch.
func modsListing(t *testing.T) *seam.Mods {
	t.Helper()
	var m seam.Mods
	if err := json.Unmarshal([]byte(`{"schema": "boltbeam.tui.v1", "kind": "mods", "engine": "llama.cpp",
	  "fits": "base: HEAD b1b770d90 is the mods' base", "chosen": ["llama.cpp/1.0.0/kpool-incremental"],
	  "rows": [
	    {"id": "llama.cpp/1.0.0/kpool-incremental", "name": "kpool-incremental", "about": "changes 1 file", "switch": "LLAMA_KPOOL_INCREMENTAL=1",
	     "switch_why": null, "output": "identical", "output_words": "same output as stock", "chosen": true, "why": "chosen: it fits this build"},
	    {"id": "llama.cpp/1.0.0/mtp-share-head", "name": "mtp-share-head", "about": "changes 3 files", "switch": "LLAMA_MTP_SHARE_HEAD=1",
	     "switch_why": "it is on when set to /path", "output": "quality-gated", "output_words": "output may differ", "chosen": false, "why": "not chosen: its output may differ"},
	    {"id": "llama.cpp/7592b61d3/moe-prefill", "name": "moe-prefill", "about": "changes 3 files", "switch": null,
	     "switch_why": "default ON, three knobs", "output": "identical", "output_words": "same output as stock", "chosen": false, "why": "not chosen: another commit"}]}`), &m); err != nil {
		t.Fatal(err)
	}
	return &m
}

// The Engine picker offers Apply mods under llama.cpp only when Python says it can apply mods to it here; it opens the
// page, which lists each mod with its switch, output label and chosen or not, and offers Apply mods as a job.
func TestApplyModsRowPageAndJob(t *testing.T) {
	f := Facts{Providers: familyProviders(t), Engine: "llama.cpp"}
	if strings.Contains(joined(engineActions(f)), "mods|") {
		t.Fatal("no Apply mods where Python offers none")
	}
	f.Providers = modsProviders(t, "")
	acts := joined(engineActions(f))
	if !strings.Contains(acts, "mods|Apply mods") || strings.Count(acts, "mods|") != 1 {
		t.Fatalf("one Apply mods row under llama.cpp:\n%s", acts)
	}

	m := program(loadSample(t), nil, nil, nil, 100, 30).(Model)
	m.f.Providers, m.f.Engine = modsProviders(t, ""), "llama.cpp"
	opened, cmd := m.do(action{"", "mods", "llama.cpp"})
	if cmd == nil || opened.cursor != pageMods || !opened.f.Mods.Loading {
		t.Fatalf("Apply mods opens its page and asks Python: cursor %d", opened.cursor)
	}
	next, _ := opened.Update(modsMsg{"llama.cpp", modsListing(t), nil, false})
	got := next.(Model)
	body := plain(modsBody(got.f, 100))
	for _, want := range []string{"✓ kpool-incremental · same output as stock", "switch LLAMA_KPOOL_INCREMENTAL=1", "chosen: it fits this build",
		"○ mtp-share-head · output may differ", "not chosen: its output may differ", "no switch: default ON, three knobs", "switch LLAMA_MTP_SHARE_HEAD=1: it is on when set to /path", "base: HEAD b1b770d90"} {
		if !strings.Contains(body, want) {
			t.Errorf("want %q in\n%s", want, body)
		}
	}
	if acts := joined(modsActions(got.f)); !strings.Contains(acts, "modsapply|[ Apply mods ] build the 1 chosen mods") {
		t.Fatalf("Apply mods is offered:\n%s", acts)
	}
	asked, cmd := got.do(action{"", "modsapply", ""})
	if cmd != nil || asked.f.Mods.Applying || !strings.Contains(joined(modsActions(asked.f)), "press Enter again to build the 1 chosen mods") {
		t.Fatalf("the first Enter asks, it builds nothing:\n%s", joined(modsActions(asked.f)))
	}
	applying, cmd := asked.do(action{"", "modsapply", ""})
	if cmd == nil || !applying.f.Mods.Applying || !strings.Contains(joined(engineActions(applying.f)), "building the modded copy") {
		t.Fatal("Apply mods starts one job and the picker says so")
	}
	if _, again := applying.do(action{"", "modsapply", ""}); again != nil {
		t.Fatal("one build at a time")
	}
	built := *modsListing(t)
	built.Build = &seam.ModsBuild{Key: "b1b770d90-mods-1a2b3c4d", Folder: "/e/llama.cpp/b1b770d90-mods-1a2b3c4d", Status: "applied, not verified"}
	next, cmd = applying.Update(modsMsg{"llama.cpp", &built, nil, true})
	done := next.(Model)
	if done.f.Mods.Applying || cmd == nil || done.f.ForkAt["llama.cpp"] != "b1b770d90-mods-1a2b3c4d" {
		t.Fatalf("the build is shown, the engines read again and the family rests on it: %v", done.f.ForkAt)
	}
	if body := plain(modsBody(done.f, 100)); !strings.Contains(body, "Built b1b770d90-mods-1a2b3c4d") {
		t.Fatalf("the page says what was built:\n%s", body)
	}
	if back, _ := done.Update(tea.KeyPressMsg{Code: tea.KeyEscape}); back.(Model).cursor != pageEngine {
		t.Fatal("esc goes back to the Engine picker")
	}
}

// The modded fork, picked on Engine, is what Run and Tune use: Run passes --modded KEY, Tune its program as --build.
func TestModdedForkIsRunAndTuned(t *testing.T) {
	f := Facts{Providers: modsProviders(t, "b1b770d90-mods-1a2b3c4d"), Engine: "llama.cpp"}
	if f.moddedKey() != "" {
		t.Fatal("upstream in brackets: no modded build")
	}
	f.ForkAt = map[string]string{"llama.cpp": "b1b770d90-mods-1a2b3c4d"}
	if engine, why := f.pickedFork("llama.cpp"); engine != "llama.cpp" || why != "" {
		t.Fatalf("the modded fork runs on llama.cpp: %q %q", engine, why)
	}
	if f.moddedKey() != "" || f.engineBuild() != "" || familyValue(f, "llama.cpp") != "llama.cpp" {
		t.Fatal("the highlight alone picks nothing: the root's build runs until Enter")
	}
	m := program(loadSample(t), nil, nil, nil, 100, 30).(Model)
	m.f.Providers, m.f.Engine, m.f.ForkAt = f.Providers, "llama.cpp", f.ForkAt
	m.f.Env = map[string]string{"X": "1"}
	m, _ = m.do(action{"", "fork", "llama.cpp"})
	f = m.f
	if f.Modded != "b1b770d90-mods-1a2b3c4d" || familyValue(f, "llama.cpp") != "llama.cpp · with mods" || f.Env != nil {
		t.Fatalf("Enter picks it, Setup says so, another build's tuned env goes: %q %q", f.Modded, familyValue(f, "llama.cpp"))
	}
	if f.moddedKey() != "b1b770d90-mods-1a2b3c4d" || f.engineBuild() != "/e/llama.cpp/b1b770d90-mods-1a2b3c4d/build/bin/llama-bench" {
		t.Fatalf("modded %q build %q", f.moddedKey(), f.engineBuild())
	}
	argv := strings.Join(seam.Client{Python: "py"}.PipelineArgv(seam.Pipeline{Model: "m", RunDir: "r", Target: "t", Modded: f.moddedKey()}), " ")
	if !strings.Contains(argv, "--modded b1b770d90-mods-1a2b3c4d") {
		t.Fatalf("Run passes the modded build: %s", argv)
	}
	if !strings.Contains(joined(engineActions(f)), "note|with mods · 12 mods, applied, not verified") {
		t.Fatalf("the modded fork says what it holds:\n%s", joined(engineActions(f)))
	}
	if !strings.Contains(joined(engineActions(f)), "mods|Apply mods") {
		t.Fatal("Apply mods stays offered beside a modded build")
	}
}

// Tune's "none of the mods" note links to Apply mods: the Engine picker, on that row.
func TestTuneLinksToApplyMods(t *testing.T) {
	m := program(loadSample(t), nil, nil, nil, 100, 30).(Model)
	m.f.Providers, m.f.Engine = modsProviders(t, ""), "llama.cpp"
	m.f.Tune = TuneState{Key: m.f.tuneKey(), Page: &seam.Tune{Kind: "tune", Page: &seam.TunePage{}, Mods: &seam.TuneMods{None: true}}}
	if m.f.tunePage() == nil {
		t.Fatal("the sample has a model, chip and engine: Tune must have a page")
	}
	acts := tuneActions(m.f)
	if len(acts) == 0 || acts[0].do != "modsrow" || !strings.Contains(plain(acts[0].label), "[ Apply mods ]") {
		t.Fatalf("the link leads: %s", joined(acts))
	}
	went, _ := m.do(acts[0])
	if went.cursor != pageEngine || engineActions(went.f)[went.row].do != "mods" {
		t.Fatalf("the link lands on Apply mods: cursor %d row %d", went.cursor, went.row)
	}
	m.f.Tune.Page.Mods.None = false
	if strings.Contains(joined(tuneActions(m.f)), "modsrow|") {
		t.Fatal("a build with mods has no link")
	}
	key := "k1"
	m.f.Tune.Page.Mods = &seam.TuneMods{None: true, Modded: &key}
	if strings.Contains(joined(tuneActions(m.f)), "modsrow|") {
		t.Fatal("the modded build itself has no link: applying again cannot change the note")
	}
}

// The page's sentences and actions are Python's words (plain_words.json mods.page), filled with the page's facts.
func TestApplyModsPageWordsArePythons(t *testing.T) {
	f := Facts{Providers: modsProviders(t, ""), Engine: "llama.cpp", Mods: ModsState{Engine: "llama.cpp", Plan: modsListing(t)}}
	body := modsBody(f, 200)
	for _, want := range []string{strings.ReplaceAll(vocab.Mods.Page["intro"], "{engine}", "llama.cpp"),
		strings.ReplaceAll(vocab.Mods.Page["this_build"], "{fits}", "base: HEAD b1b770d90 is the mods' base")} {
		if want == "" || !strings.Contains(strings.Join(strings.Fields(body), " "), strings.Join(strings.Fields(want), " ")) {
			t.Fatalf("the page lacks Python's %q:\n%s", want, body)
		}
	}
	acts := joined(modsActions(f))
	if !strings.Contains(acts, strings.ReplaceAll(vocab.Mods.Page["apply"], "{n}", "1")) || !strings.Contains(acts, vocab.Mods.Page["back"]) {
		t.Fatalf("the actions are not Python's words:\n%s", acts)
	}
}

// Mods made for another GPU backend: the Engine picker says why in the row's place (Python's words), and offers no page.
func TestApplyModsNotHereSaysWhy(t *testing.T) {
	f := Facts{Providers: familyProviders(t), Engine: "llama.cpp"}
	why := "the llama.cpp mods are CUDA patches (NVIDIA GPUs); this machine's GPU backend is Metal: nothing to apply here"
	f.Providers.Providers[0].ModsNote = &why
	acts := joined(engineActions(f))
	if strings.Contains(acts, "mods|") || !strings.Contains(acts, "note|Apply mods · not on this machine: "+why) {
		t.Fatalf("a note with the reason, no Apply mods row:\n%s", acts)
	}
}

// A chosen set already built and on disk: the page offers Use it (picks that copy, as Enter on its fork would) and
// Rebuild (asks first, then --rebuild), never a silent rebuild.
func TestApplyModsAlreadyBuiltOffersIt(t *testing.T) {
	key := "b1b770d90-mods-1a2b3c4d"
	m := program(loadSample(t), nil, nil, nil, 100, 30).(Model)
	m.f.Providers, m.f.Engine = modsProviders(t, key), "llama.cpp"
	opened, _ := m.do(action{"", "mods", "llama.cpp"})
	listing := modsListing(t)
	listing.Key, listing.AlreadyBuilt = key, true
	next, _ := opened.Update(modsMsg{"llama.cpp", listing, nil, false})
	got := next.(Model)
	acts := joined(modsActions(got.f))
	if !strings.Contains(acts, "modsuse|[ Use it ] these mods are already built ("+key+")") || !strings.Contains(acts, "modsapply|[ Rebuild ] build the 1 chosen mods again") ||
		strings.Contains(acts, "[ Apply mods ]") {
		t.Fatalf("Use it and Rebuild, not Apply mods:\n%s", acts)
	}
	used, _ := got.do(action{"", "modsuse", ""})
	if used.f.Modded != key || used.f.moddedKey() != key || used.cursor == pageMods {
		t.Fatalf("Use it picks the built copy and leaves the page: modded %q cursor %d", used.f.Modded, used.cursor)
	}
	asked, cmd := got.do(action{"", "modsapply", "rebuild"})
	if cmd != nil || !strings.Contains(joined(modsActions(asked.f)), "[ Rebuild ] press Enter again") {
		t.Fatal("Rebuild asks first")
	}
	if building, cmd := asked.do(action{"", "modsapply", "rebuild"}); cmd == nil || !building.f.Mods.Applying {
		t.Fatal("the second Enter rebuilds")
	}
}

// Run waits while Apply mods builds (a build beside a measurement skews it); [ Run ] names a picked modded build.
func TestRunWaitsForApplyModsAndNamesModded(t *testing.T) {
	m := program(loadSample(t), nil, nil, nil, 100, 30).(Model)
	m.f.Providers, m.f.Engine = modsProviders(t, "b1b770d90-mods-1a2b3c4d"), "llama.cpp"
	m.f.Mods = ModsState{Engine: "llama.cpp", Applying: true}
	if msg := m.startRun("llama.cpp")(); msg != noteMsg("Apply mods is building a copy of llama.cpp. Wait for it, then press Run.") {
		t.Fatalf("Run waits for the build: %v", msg)
	}
	m.f.ForkAt = map[string]string{"llama.cpp": "b1b770d90-mods-1a2b3c4d"}
	m, _ = m.do(action{"", "fork", "llama.cpp"})
	if help := runHelp(m.f); !strings.Contains(help, "llama.cpp with mods") {
		t.Fatalf("Run names the modded build: %q", help)
	}
}

// The confirm survives the key path: Enter on [ Apply mods ] asks, a second Enter builds; any other key in between
// cancels the question.
func TestApplyModsEnterTwiceByKeys(t *testing.T) {
	m := program(loadSample(t), nil, nil, nil, 100, 30).(Model)
	m.f.Providers, m.f.Engine = modsProviders(t, ""), "llama.cpp"
	opened, _ := m.do(action{"", "mods", "llama.cpp"})
	next, _ := opened.Update(modsMsg{"llama.cpp", modsListing(t), nil, false})
	got := next.(Model)
	got.row = 0
	enter := tea.KeyPressMsg{Code: tea.KeyEnter}
	next, _ = got.Update(enter)
	if a := next.(Model); a.f.Mods.Applying || a.f.Confirm != modsConfirm {
		t.Fatal("the first Enter asks")
	}
	moved, _ := next.(Model).Update(tea.KeyPressMsg{Code: tea.KeyDown})
	if moved.(Model).f.Confirm != "" {
		t.Fatal("another key cancels the question")
	}
	next, _ = next.(Model).Update(enter)
	if !next.(Model).f.Mods.Applying {
		t.Fatal("the second Enter builds")
	}
}
