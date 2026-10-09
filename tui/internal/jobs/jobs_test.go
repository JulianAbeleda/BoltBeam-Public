package jobs

import (
	"strings"
	"testing"
	"time"
)

func TestStartTailStop(t *testing.T) {
	s := Store{Dir: t.TempDir()}
	job, err := s.Start("demo", t.TempDir(), []string{"sh", "-c", "echo one; echo two; sleep 30"})
	if err != nil || !job.Alive || job.PID <= 0 {
		t.Fatalf("start: %+v %v", job, err)
	}
	if _, err := s.Start("demo", "", []string{"sh", "-c", "true"}); err == nil {
		t.Fatal("a second start of a live job must be refused")
	}
	deadline := time.Now().Add(3 * time.Second)
	var lines []string
	for time.Now().Before(deadline) {
		lines, _ = s.Tail("demo", 2)
		if len(lines) == 2 && lines[1] == "two" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if len(lines) != 2 || lines[0] != "one" || lines[1] != "two" {
		t.Fatalf("tail: %q", lines)
	}
	all, _ := s.Tail("demo", 0)
	if !strings.Contains(all[0], "start sh -c") {
		t.Fatalf("the log must open with the argv line: %q", all[0])
	}
	if got, err := s.Status("demo"); err != nil || !got.Alive {
		t.Fatalf("status: %+v %v", got, err)
	}
	if _, err := s.Stop("demo"); err != nil {
		t.Fatal(err)
	}
	for time.Now().Before(deadline) {
		if got, _ := s.Status("demo"); !got.Alive {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if got, _ := s.Status("demo"); got.Alive {
		t.Fatal("SIGTERM did not end the job")
	}
	if _, err := s.Stop("demo"); err == nil {
		t.Fatal("stopping a finished job must be an error")
	}
	if _, err := s.Status("never"); err == nil {
		t.Fatal("an unknown job must be an error")
	}
}

func TestAliveListsOnlyRunningJobs(t *testing.T) {
	s := Store{Dir: t.TempDir()}
	if len(s.Alive()) != 0 {
		t.Fatal("an empty store has no live job")
	}
	if _, err := s.Start("sleeper", t.TempDir(), []string{"sleep", "30"}); err != nil {
		t.Fatal(err)
	}
	if live := s.Alive(); len(live) != 1 || live[0].ID != "sleeper" {
		t.Fatalf("live: %+v", live)
	}
	if _, err := s.Stop("sleeper"); err != nil {
		t.Fatal(err)
	}
}
