package seam

// Speed up: a copy of the model that decodes faster, the original kept. Python owns every word and figure: `screen
// speedup-offer` says whether and what it costs, the job is `derive lighter-head --json --progress`, and `screen
// speedup-plain` words its log; `screen speedup-verify` checks a copy against its record.

// SpeedOffer is `screen speedup-offer`: enabled, or Reason (plain) why not; the copy's size (at most) and the seconds each
// stage is expected to take (an estimate, not a measurement).
type SpeedOffer struct {
	Schema    string             `json:"schema"`
	Kind      string             `json:"kind"`
	Model     string             `json:"model"`
	Name      string             `json:"name"`
	IsCopy    bool               `json:"is_copy"`
	Enabled   bool               `json:"enabled"`
	Reason    *string            `json:"reason"`
	CopyBytes *int64             `json:"copy_bytes"`
	Expect    map[string]float64 `json:"expect"`
	Gpu       bool               `json:"gpu"`
	Engine    string             `json:"engine"`
	Confirm   *SpeedConfirm      `json:"confirm"`
}

// SpeedConfirm is the confirm step, worded: what will happen, the disk space, the time, the GPU.
type SpeedConfirm struct {
	Title string   `json:"title"`
	Lines []string `json:"lines"`
	Yes   string   `json:"yes"`
	No    string   `json:"no"`
}

// SpeedPlain is `screen speedup-plain`: a job's layer 1. Outcome is kept, not_kept, refused or crashed; Lines go under
// Head in order; Fix is the one fix, already worded, when there is one.
type SpeedPlain struct {
	Kind          string   `json:"kind"`
	Outcome       string   `json:"outcome"`
	Head          string   `json:"head"`
	Lines         []string `json:"lines"`
	Fix           *string  `json:"fix"`
	Copy          *string  `json:"copy"`
	Name          *string  `json:"name"`
	SpeedMeasured *bool    `json:"speed_measured"`
}

// SpeedVerify is `screen speedup-verify`: Ok, or the plain reason it does not match.
type SpeedVerify struct {
	Kind  string   `json:"kind"`
	Ok    bool     `json:"ok"`
	Head  string   `json:"head"`
	Lines []string `json:"lines"`
	Fix   *string  `json:"fix"`
}

// SpeedOffer asks whether Speed up applies to the model; it reads the checkpoint's files and nothing else.
func (c Client) SpeedOffer(model string) (*SpeedOffer, error) {
	var o SpeedOffer
	_, _, err := c.decode(&o, nil, "boltbeam.workflow.screen", "speedup-offer", model)
	return &o, err
}

// SpeedArgv is the Speed up job: the copy is written beside the model, checked, and its speed measured (one GPU).
func (c Client) SpeedArgv(model string) []string {
	if argv := c.Demo.argv("speedup"); argv != nil { // a demo plays the job; nothing is made
		return argv
	}
	return []string{c.Python, "-m", "boltbeam.cli", "derive", "lighter-head", model, "--json", "--progress"}
}

// SpeedPlain words a finished job's log.
func (c Client) SpeedPlain(log string) (*SpeedPlain, error) {
	var p SpeedPlain
	_, _, err := c.decode(&p, nil, "boltbeam.workflow.screen", "speedup-plain", log)
	return &p, err
}

// SpeedVerify checks a copy against its record: files, the answer check's text, the engine and the GPU.
func (c Client) SpeedVerify(copy string) (*SpeedVerify, error) {
	var v SpeedVerify
	_, _, err := c.decode(&v, nil, "boltbeam.workflow.screen", "speedup-verify", copy)
	return &v, err
}

// argv is the demo's command for a job the screen starts, or nil outside a demo or when the demo names none.
func (d *Demo) argv(job string) []string {
	if d == nil {
		return nil
	}
	return d.Argv[job]
}
