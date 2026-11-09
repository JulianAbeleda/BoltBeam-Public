package ui

import (
	"errors"
	"strings"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// The run page shows Python's build lines as they are: the build beside upstream's newest and the comparison it made.
func TestRunEngineBodyShowsLatestAndComparison(t *testing.T) {
	e := seam.RunEngine{Provider: "llama.cpp", BuildLines: []string{"Engine build b12211; latest was b12230, checked 2026-11-08T10:00:00Z.",
		"Upgrade comparison: compared once with the previous build b11571 (run r-was-b11571); see engine_upgrade.json."}}
	got := plain(runEngineBody(&e))
	for _, want := range e.BuildLines {
		if !strings.Contains(got, want) {
			t.Errorf("want %q in %q", want, got)
		}
	}
}

// standings is Python's `engines --latest` answer with llama.cpp in one state.
func standings(state, words string) []seam.EngineStanding {
	latest := "0.9.0"
	return []seam.EngineStanding{{Engine: "llama.cpp", Latest: &latest, State: state, Words: words}}
}

func upgradeFacts(rows []seam.EngineStanding) Facts {
	return Facts{Providers: engines(), Engine: "llama.cpp", Latest: rows}
}

func joined(acts []action) string { return strings.Join(rows(acts), "\n") }

// Upgrade engines lives in the Engine picker, offered only when Python knows a newer release than the build here.
func TestUpgradeIsOfferedOnlyWhenANewerReleaseIsKnown(t *testing.T) {
	got := rows(engineActions(upgradeFacts(standings("behind", "0.6.0 · 0.9.0 is out"))))
	if !strings.Contains(strings.Join(got, "\n"), "upgrade|Upgrade llama.cpp to 0.9.0") || strings.Contains(strings.Join(got, "\n"), "Upgrade tinygrad") {
		t.Fatalf("behind: want the action for llama.cpp only: %v", got)
	}
	for _, rs := range [][]seam.EngineStanding{nil, standings("up_to_date", "0.9.0 is the latest release"),
		standings("unknown", "latest release unknown (never checked)"), standings("waiting", "0.9.0 installed")} {
		if acts := joined(engineActions(upgradeFacts(rs))); strings.Contains(acts, "upgrade|") {
			t.Errorf("no newer release known (%v): no Upgrade, got %s", rs, acts)
		}
	}
}

// Setup says, under Engine, when the picked engine is not on the latest release; unknown is said as unknown.
func TestSetupNoticeForNewerEqualAndUnknown(t *testing.T) {
	for _, c := range []struct {
		f    Facts
		want string
	}{
		{upgradeFacts(standings("behind", "0.6.0 · 0.9.0 is out")), "llama.cpp 0.6.0 · 0.9.0 is out: Upgrade in Engine"},
		{upgradeFacts(standings("unknown", "latest release unknown (never checked)")), "llama.cpp: latest release unknown (never checked)"},
		{Facts{Providers: engines(), Engine: "llama.cpp", LatestErr: "exit 1"}, "llama.cpp: latest release unknown (exit 1)"},
		{upgradeFacts(standings("up_to_date", "0.9.0 is the latest release")), ""},
		{upgradeFacts(nil), ""},
	} {
		if got := engineNotice(c.f); got != c.want {
			t.Errorf("notice %q, want %q", got, c.want)
		}
		setup := joined(setupActions(c.f))
		if c.want != "" && !strings.Contains(setup, "note|"+c.want) {
			t.Errorf("Setup must carry %q:\n%s", c.want, setup)
		}
		if strings.Contains(strings.ToLower(setup), "stay current") || strings.Contains(setup, "upgrade|") {
			t.Errorf("Setup has no upgrade row of its own:\n%s", setup)
		}
	}
}

// Nothing upgrades unless the user picks Upgrade: Run's argv asks for no update, and the pick starts one job for one
// engine, its answer shown and the standing asked again.
func TestNoUpgradeWithoutPickingIt(t *testing.T) {
	argv := strings.Join(seam.Client{Python: "py"}.PipelineArgv(seam.Pipeline{Model: "m", RunDir: "r", Target: "t", Analyze: true}), " ")
	if strings.Contains(argv, "engines") || strings.Contains(argv, "--update") || strings.Contains(argv, "--upgrade") {
		t.Fatalf("Run must not install: %s", argv)
	}
	m := program(loadSample(t), nil, nil, nil, 100, 30).(Model)
	m.f.Latest = standings("behind", "0.6.0 · 0.9.0 is out")
	picked, cmd := m.do(action{"", "upgrade", "llama.cpp"})
	if cmd == nil || picked.f.Upgrading != "llama.cpp" || !strings.Contains(joined(engineActions(picked.f)), "upgrading llama.cpp") {
		t.Fatalf("picking Upgrade starts the job and the picker says so: %q", picked.f.Upgrading)
	}
	if _, again := picked.do(action{"", "upgrade", "llama.cpp"}); again != nil {
		t.Fatal("one upgrade at a time")
	}
	answer := []seam.EngineStanding{{Engine: "llama.cpp", Action: "not_compatible", Words: "0.9.0 not compatible here: x; current stays"}}
	next, cmd := picked.Update(latestMsg{answer, nil, "llama.cpp"})
	if got := next.(Model); got.f.Upgrading != "" || got.note != "Upgrade llama.cpp: 0.9.0 not compatible here: x; current stays" || cmd == nil {
		t.Fatalf("the answer is shown and the standing reloaded: %q", got.note)
	}
	if !strings.Contains(joined(setupActions(next.(Model).f)), "0.6.0 · 0.9.0 is out") {
		t.Fatal("an upgrade's answer is not the standing: Setup keeps it until Python is asked again")
	}
	next, _ = picked.Update(latestMsg{nil, errors.New("exit 2"), "llama.cpp"})
	if got := next.(Model).note; got != "Upgrade llama.cpp failed: exit 2 (0 rows)" {
		t.Fatalf("a failed job is said: %q", got)
	}
}
