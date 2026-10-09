package ui

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// chipsJSON is `screen chips` cut down: Python decides the groups and the words; the screen only draws them.
func chipsJSON(t *testing.T, here string) *seam.Targets {
	t.Helper()
	raw := `{"kind": "chips",
	  "targets": [{"id": "amd_gfx1100", "has_ceiling": true}, {"id": "nvidia_sm89"}, {"id": "apple_metal"},
	              {"id": "apple_m3_10c", "has_ceiling": true}, {"id": "apple_m3_10c_16g", "has_ceiling": true, "source": "generated"}],
	  "this_machine": ` + here + `,
	  "groups": [
	    {"key": "this", "title": "This machine", "selectable": true, "chips": THIS},
	    {"key": "measured", "title": "Measured chips", "selectable": true, "chips": [
	      {"id": "amd_gfx1100", "words": "what if: 960.0 GB/s, source not recorded"},
	      {"id": "apple_m3_10c", "words": "what if: 97.2 GB/s, measured 2026-10-09"}]},
	    {"key": "not_measured", "title": "Not measured yet", "selectable": false, "chips": [
	      {"id": "nvidia_sm89", "words": "run BoltBeam on one to measure it"}]},
	    {"key": "families", "title": "Families", "selectable": false, "folded": true, "chips": [
	      {"id": "apple_metal", "words": "a family, not one chip: not a run target"}]}]}`
	this := `[]`
	if strings.Contains(here, `"known"`) {
		this = `[{"id": "apple_m3_10c_16g", "words": "profile made on this Mac, 2026-10-09"}]`
	}
	var out seam.Targets
	if err := json.Unmarshal([]byte(strings.Replace(raw, "THIS", this, 1)), &out); err != nil {
		t.Fatal(err)
	}
	return &out
}

const newChip = `{"name": "Apple M3", "target_id": null, "status": "new", "source": null, "new_id": "apple_m3_10c_16g",
  "words": "No profile for this GPU yet"}`
const madeHere = `{"name": "Apple M3", "target_id": "apple_m3_10c_16g", "status": "known", "source": "generated",
  "words": "profile made on this Mac, 2026-10-09"}`

func rows(acts []action) []string {
	out := []string{}
	for _, a := range acts {
		out = append(out, a.do+"|"+strings.TrimSpace(plain(a.label)))
	}
	return out
}

// The chip picker shows this machine's chip only, with Python's words for its cached profile, and Autoscan.
// No other chip is offered: a what if is chosen with --target.
func TestChipPickerIsThisMachineOnly(t *testing.T) {
	f := Facts{Path: "m.gguf", Targets: chipsJSON(t, madeHere), Detected: true, ThisMachine: "apple_m3_10c_16g", Target: 4}
	want := []string{
		"chip|● apple_m3_10c_16g  profile made on this Mac, 2026-10-09",
		"autoscan|[ Measure again ] refresh this machine's profile",
	}
	got := rows(chipActions(f))
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("got\n%s", strings.Join(got, "\n"))
	}
	if acts := chipActions(f); acts[0].arg != "4" || acts[1].arg != "remeasure" {
		t.Fatalf("args: %+v", acts)
	}
}

// A new chip: the picker says there is no profile yet and offers Autoscan; nothing is chosen, Run waits.
func TestNewChipOffersAutoscan(t *testing.T) {
	f := Facts{Path: "m.gguf", Targets: chipsJSON(t, newChip), Detected: true, NewChip: "Apple M3"}
	got := rows(chipActions(f))
	if len(got) != 2 || got[0] != "note|No profile for this GPU yet" || got[1] != "autoscan|[ Autoscan ] measure this GPU and save a profile" {
		t.Fatalf("got\n%s", strings.Join(got, "\n"))
	}
	if mark, text := chipLine(f); mark != "open" || !strings.Contains(text, "new chip: Apple M3 has no profile yet") {
		t.Fatalf("chip line: %s %s", mark, text)
	}
	if m := strings.Join(f.missing(), ","); !strings.Contains(m, "a chip") {
		t.Fatalf("Run must wait for a chip: %s", m)
	}
}

// An autoscan answer is said once; a failure says the probe's own reason.
func TestAutoscanAnswer(t *testing.T) {
	m := Model{f: Facts{Scanning: true}}
	next, _ := m.update(scanMsg{scan: &seam.ChipScan{Action: "generated", TargetID: sp("apple_m3_10c_16g")}})
	if next.f.Scanning || next.note != "Autoscan: apple_m3_10c_16g generated." {
		t.Fatalf("note: %q", next.note)
	}
	next, _ = m.update(scanMsg{scan: &seam.ChipScan{Action: "failed", Reason: sp("nvcc is not installed")}})
	if !strings.Contains(next.note, "nvcc is not installed") {
		t.Fatalf("failure: %q", next.note)
	}
}

// Before a chip is known, Setup shows no speed limit: the list's first chip was not chosen by anyone.
func TestNoLimitForAChipNobodyChose(t *testing.T) {
	f := Facts{Path: "m.gguf", Targets: chipsJSON(t, newChip), Detected: true, NewChip: "Apple M3",
		Ceiling: &seam.Ceiling{Decode: seam.CeilingBlock{TokS: fp(205.3), FloorMs: 4.9}}}
	if body := plain(chipPickerBody(f, 100)); strings.Contains(body, "Speed limit on") {
		t.Fatalf("limit shown for an unchosen chip:\n%s", body)
	}
	f.Picked = true
	if body := plain(chipPickerBody(f, 100)); !strings.Contains(body, "Speed limit on amd_gfx1100") {
		t.Fatalf("a picked chip shows its limit:\n%s", body)
	}
}
