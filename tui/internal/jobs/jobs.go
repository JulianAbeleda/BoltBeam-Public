// Package jobs owns process control: it starts a BoltBeam command as a detached child, keeps the pid and the
// log in the TUI's own state folder (never inside a run folder, which Python owns), and can tail or stop it.
package jobs

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

type Store struct{ Dir string }

// Job is one started command. Alive is checked against the pid at read time, not remembered.
type Job struct {
	ID        string   `json:"id"`
	PID       int      `json:"pid"`
	Alive     bool     `json:"alive"`
	LogPath   string   `json:"log_path"`
	StartedAt string   `json:"started_at"`
	Argv      []string `json:"argv"`
	Dir       string   `json:"dir"`
}

// DefaultDir is $BOLTBEAM_TUI_STATE, else $XDG_STATE_HOME/boltbeam-tui, else ~/.local/state/boltbeam-tui.
func DefaultDir() string {
	if dir := os.Getenv("BOLTBEAM_TUI_STATE"); dir != "" {
		return dir
	}
	if dir := os.Getenv("XDG_STATE_HOME"); dir != "" {
		return filepath.Join(dir, "boltbeam-tui")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "boltbeam-tui")
	}
	return filepath.Join(home, ".local", "state", "boltbeam-tui")
}

func (s Store) path(id, ext string) string { return filepath.Join(s.Dir, id+ext) }

// Start launches argv in dir with stdout and stderr appended to the job's log. It refuses while a job with
// the same id is alive. The child gets its own process group, so it outlives the TUI.
func (s Store) Start(id, dir string, argv []string) (Job, error) {
	if len(argv) == 0 {
		return Job{}, errors.New("nothing to start")
	}
	if old, err := s.Status(id); err == nil && old.Alive {
		return old, fmt.Errorf("job %s is already running (pid %d)", id, old.PID)
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return Job{}, err
	}
	log, err := os.OpenFile(s.path(id, ".log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return Job{}, err
	}
	defer log.Close()
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Stdout, cmd.Stderr = log, log
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	fmt.Fprintf(log, "=== %s start %s\n", time.Now().Format(time.RFC3339), strings.Join(argv, " "))
	if err := cmd.Start(); err != nil {
		return Job{}, err
	}
	job := Job{ID: id, PID: cmd.Process.Pid, Alive: true, LogPath: s.path(id, ".log"),
		StartedAt: time.Now().Format(time.RFC3339), Argv: argv, Dir: dir}
	raw, _ := json.Marshal(job)
	if err := os.WriteFile(s.path(id, ".json"), raw, 0o644); err != nil {
		return job, err
	}
	go cmd.Wait() // reap if the TUI outlives the child; harmless if it does not
	return job, nil
}

// Status reads the job file and asks the kernel whether the pid is still there.
func (s Store) Status(id string) (Job, error) {
	raw, err := os.ReadFile(s.path(id, ".json"))
	if err != nil {
		return Job{}, err
	}
	var job Job
	if err := json.Unmarshal(raw, &job); err != nil {
		return Job{}, err
	}
	job.Alive = alive(job.PID)
	return job, nil
}

func alive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}

// Stop sends SIGTERM. Every stage writes its artifacts and updates run_manifest.json when it finishes, so
// what is lost is at most the stage in flight; the run folder keeps what landed.
// Alive is every job in the store whose process is still running: one GPU measurement at a time.
func (s Store) Alive() []Job {
	out := []Job{}
	entries, err := os.ReadDir(s.Dir)
	if err != nil {
		return out
	}
	for _, e := range entries {
		if id, ok := strings.CutSuffix(e.Name(), ".json"); ok {
			if j, err := s.Status(id); err == nil && j.Alive {
				out = append(out, j)
			}
		}
	}
	return out
}

func (s Store) Stop(id string) (Job, error) {
	job, err := s.Status(id)
	if err != nil {
		return Job{}, err
	}
	if !job.Alive {
		return job, fmt.Errorf("job %s is not running", id)
	}
	if err := syscall.Kill(job.PID, syscall.SIGTERM); err != nil {
		return job, err
	}
	return job, nil
}

// Tail returns the last n lines of the job's log.
func (s Store) Tail(id string, n int) ([]string, error) {
	raw, err := os.ReadFile(s.path(id, ".log"))
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	if n > 0 && len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines, nil
}
