package seam

// The shapes `python -m boltbeam.workflow.screen` prints (schema boltbeam.tui.v1) and the model profile
// `boltbeam inspect` prints (boltbeam.model_profile.v1). Pointer fields are JSON null when the artifact does not
// carry them (a chip with no measured speed, a run with no timing, a kernel with no role).

type FactStatus struct {
	MemoryBandwidthGBs string `json:"memory_bandwidth_gbs"`
	PeakTFLOPS         string `json:"peak_tflops"`
}

type Target struct {
	ID                 string             `json:"id"`
	Backend            string             `json:"backend"`
	BackendStatus      string             `json:"backend_status"`
	Scope              *string            `json:"scope"`
	MemoryBandwidthGBs *float64           `json:"memory_bandwidth_gbs"`
	PeakTFLOPS         map[string]float64 `json:"peak_tflops"`
	MatrixTFLOPS       map[string]float64 `json:"matrix_tflops"`
	FactStatus         FactStatus         `json:"fact_status"`
	HasCeiling         bool               `json:"has_ceiling"`
}

type Targets struct {
	Kind    string   `json:"kind"`
	Targets []Target `json:"targets"`
}

// Detected is `screen detect`: autoscan's GPU probe cut to what a screen shows.
type Detected struct {
	Status     string  `json:"status"`
	Name       *string `json:"name"`
	TargetID   *string `json:"target_id"`
	TargetKind *string `json:"target_kind"`
	Registered bool    `json:"registered"`
}

// Role is one row of the model profile: one (role, shape, quant) with how many tensors share it.
type Role struct {
	Role       string `json:"role"`
	RoleClass  string `json:"role_class"`
	TensorName string `json:"tensor_name"`
	Rows       int    `json:"rows"`
	Cols       int    `json:"cols"`
	Count      int    `json:"count"`
	NExpert    int    `json:"n_expert"`
	Quant      string `json:"quant"`
}

type Attention struct {
	HeadCount   *int `json:"head_count"`
	HeadCountKV *int `json:"head_count_kv"`
	HeadDim     *int `json:"head_dim"`
}

type Metadata struct {
	FormatFamily string    `json:"format_family"`
	QuantTypes   []string  `json:"quant_types"`
	TensorCount  *int      `json:"tensor_count"`
	Attention    Attention `json:"attention"`
}

type Profile struct {
	Schema            string   `json:"schema"`
	ModelID           string   `json:"model_id"`
	Source            string   `json:"source"`
	Architecture      *string  `json:"architecture"`
	ArchitectureClass string   `json:"architecture_class"`
	Complete          bool     `json:"complete"`
	LayerCount        *int     `json:"layer_count"`
	HiddenSize        *int     `json:"hidden_size"`
	FFNSize           *int     `json:"ffn_size"`
	VocabSize         *int     `json:"vocab_size"`
	Roles             []Role   `json:"roles"`
	Metadata          Metadata `json:"metadata"`
}

type Shape struct {
	K int `json:"k"`
	M int `json:"m"`
	N int `json:"n"`
}

type CeilingRole struct {
	Role       string  `json:"role"`
	Quant      string  `json:"quant"`
	Shape      Shape   `json:"shape"`
	BytesMoved float64 `json:"bytes_moved"`
	FloorMs    float64 `json:"floor_ms"`
	Share      float64 `json:"share"`
	Regime     string  `json:"regime"`
}

type CeilingBlock struct {
	Context    int           `json:"context"`
	BytesMoved float64       `json:"bytes_moved"`
	FloorMs    float64       `json:"floor_ms"`
	TokS       *float64      `json:"tok_s"`
	Regime     string        `json:"regime"`
	Roles      []CeilingRole `json:"roles"`
}

type Ceiling struct {
	Kind             string       `json:"kind"`
	ModelID          string       `json:"model_id"`
	Target           Target       `json:"target"`
	PeakBandwidthGBs float64      `json:"peak_bandwidth_gbs"`
	PeakTFLOPS       float64      `json:"peak_tflops"`
	TruthStatus      string       `json:"truth_status"`
	RidgeIntensity   float64      `json:"ridge_intensity"`
	Assumptions      []string     `json:"assumptions"`
	Decode           CeilingBlock `json:"decode"`
	Prefill          CeilingBlock `json:"prefill"`
}

type Stage struct {
	Key       string   `json:"key"`
	Label     string   `json:"label"`
	Note      string   `json:"note"`
	Done      bool     `json:"done"`
	Artifacts []string `json:"artifacts"`
}

type Need struct {
	Need    string `json:"need"`
	Request string `json:"request"`
}

type Measured struct {
	Probe  bool `json:"probe"`
	Timing bool `json:"timing"`
}

// MeasureStatus is measure_status.json: whether the pipeline measured on this machine, and if not, why and what
// to run instead. Python writes it; the screen only shows it.
type MeasureStatus struct {
	Status    string  `json:"status"` // measured | skipped | failed
	Collector *string `json:"collector"`
	Reason    *string `json:"reason"`
	Command   *string `json:"command"`
}

type Summary struct {
	ID          string         `json:"id"`
	ModelID     string         `json:"model_id"`
	ModelFormat string         `json:"model_format"`
	TargetID    string         `json:"target_id"`
	Workload    string         `json:"workload"`
	LatestStage *string        `json:"latest_stage"`
	Status      string         `json:"status"` // not_analyzed | needs_measurement | policy_seeded | ...
	Stages      []Stage        `json:"stages"`
	Blocked     []Need         `json:"blocked"`
	Measured    Measured       `json:"measured"`
	Report      *string        `json:"report"`
	Measure     *MeasureStatus `json:"measure"`
}

type Runs struct {
	Kind string    `json:"kind"`
	Root string    `json:"root"`
	Runs []Summary `json:"runs"`
}

type ModelFacts struct {
	Architecture      *string  `json:"architecture"`
	ArchitectureClass *string  `json:"architecture_class"`
	LayerCount        *int     `json:"layer_count"`
	HiddenSize        *int     `json:"hidden_size"`
	FFNSize           *int     `json:"ffn_size"`
	VocabSize         *int     `json:"vocab_size"`
	RoleCount         int      `json:"role_count"`
	QuantTypes        []string `json:"quant_types"`
}

type Route struct {
	Role          string   `json:"role"`
	Quant         string   `json:"quant"`
	Shape         []int    `json:"shape"`
	SelectedRoute *string  `json:"selected_route"`
	Status        string   `json:"status"` // promoted | refuted | blocked | unmeasured | candidate
	Candidates    []string `json:"candidates"`
	EvidenceRefs  []string `json:"evidence_refs"`
}

type RoleTiming struct {
	Role           string   `json:"role"`
	Quant          string   `json:"quant"`
	Shape          []int    `json:"shape"`
	Context        *int     `json:"context"`
	WallUs         *float64 `json:"wall_us"`
	PctStep        *float64 `json:"pct_step"`
	Classification string   `json:"classification"`
}

type Kernel struct {
	Name        string   `json:"name"`
	Kind        string   `json:"kind"`
	Us          *float64 `json:"us"`
	PctStep     *float64 `json:"pct_step"`
	PhysUtilPct *float64 `json:"phys_util_pct"`
	Bucket      string   `json:"bucket"`
	LossUs      *float64 `json:"loss_us"`
}

type Timing struct {
	Status         string       `json:"status"` // classified | absent
	DominantBucket *string      `json:"dominant_bucket"`
	NextActions    []string     `json:"next_actions"`
	Context        *int         `json:"context"`
	TotalUs        *float64     `json:"total_us"`
	TokS           *float64     `json:"tok_s"`
	Roles          []RoleTiming `json:"roles"`
	Kernels        []Kernel     `json:"kernels"`
}

type Regime struct {
	Role              string `json:"role"`
	Quant             string `json:"quant"`
	Shape             []int  `json:"shape"`
	Classification    string `json:"classification"`
	VisibleBottleneck string `json:"visible_bottleneck"`
	NextAction        string `json:"next_action"`
}

// CeilingRef is the modeled ceiling for a run's model and chip, or the reason there is none.
type CeilingRef struct {
	Status           string   `json:"status"` // modeled | absent
	Reason           string   `json:"reason"`
	Context          *int     `json:"context"`
	TokS             *float64 `json:"tok_s"`
	FloorMs          *float64 `json:"floor_ms"`
	PeakBandwidthGBs *float64 `json:"peak_bandwidth_gbs"`
}

type Results struct {
	Kind     string     `json:"kind"`
	ID       string     `json:"id"`
	ModelID  string     `json:"model_id"`
	TargetID string     `json:"target_id"`
	Workload string     `json:"workload"`
	Measured bool       `json:"measured"`
	Ceiling  CeilingRef `json:"ceiling"`
	Routes   []Route    `json:"routes"`
	Timing   Timing     `json:"timing"`
	Regimes  []Regime   `json:"regimes"`
	Blocked  []Need     `json:"blocked"`
	Report   *string    `json:"report"`
}

type Run struct {
	Summary
	Kind      string     `json:"kind"`
	ModelPath string     `json:"model_path"`
	Model     ModelFacts `json:"model"`
	NextStep  string     `json:"next_step"`
	Artifacts []string   `json:"artifacts"`
	Results   Results    `json:"results"`
}
