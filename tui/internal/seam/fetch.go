package seam

// Fetch runs: `boltbeam fetch DEST --json` (cli/fetch.py, workflow/fetch.py). Listing is one call; the fetch
// itself runs as a job (FetchArgv), and its answer is read back from the job's log (ReadFetch).

import (
	"encoding/json"
	"errors"
	"strconv"
	"strings"
)

// fetchArgs is `fetch DEST [-p PORT] --into ROOT --saved SAVED --json`: runs land in the screen's own saved folder.
func (c Client) fetchArgs(dest string, port int) []string {
	args := []string{"fetch", dest}
	if port > 0 {
		args = append(args, "-p", strconv.Itoa(port))
	}
	return append(args, "--into", c.Root, "--saved", c.Saved, "--json")
}

// RemoteRuns lists the runs the remote holds (one ssh call; nothing is copied). ssh's refusal comes back in words.
func (c Client) RemoteRuns(dest string, port int) (*RemoteRuns, error) {
	var r RemoteRuns
	_, _, err := c.decode(&r, nil, "boltbeam.cli", c.fetchArgs(dest, port)...)
	return &r, err
}

// FetchArgv fetches the runs named, or every run the remote holds when ids is empty; traces also copies raw traces.
func (c Client) FetchArgv(dest string, port int, ids []string, traces bool) []string {
	argv := append([]string{c.Python, "-m", "boltbeam.cli"}, c.fetchArgs(dest, port)...)
	if len(ids) == 0 {
		argv = append(argv, "--all")
	}
	if traces {
		argv = append(argv, "--with-traces")
	}
	return append(argv, ids...)
}

// ReadFetch reads the fetch job's answer out of its log: the JSON object after the job's start line ("=== "). A
// refusal (kind error) is an Error in its own words; a log with no answer (the job was stopped, or Python crashed)
// is an error naming that.
func ReadFetch(lines []string) (*RemoteRuns, error) {
	start := 0
	for i, l := range lines {
		if strings.HasPrefix(l, "=== ") {
			start = i + 1
		}
	}
	for i := start; i < len(lines); i++ {
		if !strings.HasPrefix(lines[i], "{") {
			continue
		}
		var r RemoteRuns
		if json.NewDecoder(strings.NewReader(strings.Join(lines[i:], "\n"))).Decode(&r) != nil {
			continue
		}
		if r.Kind == "error" {
			return nil, &Error{Message: r.Error, Code: 1}
		}
		return &r, nil
	}
	return nil, errors.New("the fetch ended without an answer")
}
