package seam

// SelfUpdate is `screen update`: is a newer BoltBeam released, and how this install upgrades (cli/self_update.py owns
// every fact; Go only shows them). Prompt is true only for a newer release that was not skipped.
type SelfUpdate struct {
	Status    string     `json:"status"` // newer, skipped, current, ahead, unknown or off
	Prompt    bool       `json:"prompt"`
	Installed *string    `json:"installed"`
	Latest    *string    `json:"latest"`
	Error     *string    `json:"error"`
	Install   string     `json:"install"` // homebrew, checkout or pip
	Command   [][]string `json:"command"` // nil: nothing is run for this install; Words says what to do
	Words     string     `json:"words"`
}

// The last line `boltbeam update --yes` prints says how it ended (self_update.DONE and FAILED).
const (
	SelfUpdateDone   = "updated:"
	SelfUpdateFailed = "update failed:"
)

// SelfUpdate asks for the facts; skip records "skip this version" first.
func (c Client) SelfUpdate(skip string) (*SelfUpdate, error) {
	args := []string{"update"}
	if skip != "" {
		args = append(args, "--skip", skip)
	}
	var u SelfUpdate
	_, _, err := c.decode(&u, nil, "boltbeam.workflow.screen", args...)
	return &u, err
}

// SelfUpdateArgv runs the upgrade for this install, its output for a log: `boltbeam update --yes`.
func (c Client) SelfUpdateArgv() []string {
	return []string{c.Python, "-m", "boltbeam.cli", "update", "--yes"}
}
