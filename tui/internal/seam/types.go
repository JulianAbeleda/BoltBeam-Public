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
	ScopeObservedAt    *string            `json:"scope_observed_at"` // when the scope's facts were recorded
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
	// DriverVersion is read live (nvidia-smi); nil where the probe reports none (Apple).
	DriverVersion *string `json:"driver_version"`
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
	// Probe is "measured" or "absent": whether this collector takes the building-block tests at all.
	Probe       *string `json:"probe"`
	ProbeReason *string `json:"probe_reason"`
	// Provider is the runtime step 4 measured with: "llama.cpp" or "tinygrad". Older runs leave it out.
	Provider *string `json:"provider"`
}

// Capture is how a provider's per-role time was taken: "nsys", "rocprofv3", "metal-system-trace" or
// "tinygrad-profile-events". Method nil means whole step only; Reason says why.
type Capture struct {
	Method *string `json:"method"`
	Reason *string `json:"reason"`
}

// ProviderRow is one runtime from `screen providers`: can it measure here, and how would step 5 time its roles.
type ProviderRow struct {
	Provider  string  `json:"provider"`
	Available bool    `json:"available"`
	Reason    *string `json:"reason"`
	Capture   Capture `json:"capture"`
}

type Providers struct {
	TargetID  string        `json:"target_id"`
	Default   string        `json:"default"`
	Providers []ProviderRow `json:"providers"`
}

// OtherLoss is another provider's per-role table for the same run, shown beside the chosen one, labelled.
type OtherLoss struct {
	Provider string `json:"provider"`
	// Run is set when the row is another run folder (same model, same chip); empty when it is this run's.
	Run             *string    `json:"run"`
	Capture         Capture    `json:"capture"`
	TokS            float64    `json:"tok_s"`
	Ms              float64    `json:"ms"`
	Roles           []RoleLoss `json:"roles"`
	NotAttributedMs *float64   `json:"not_attributed_ms"`
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
	Role          string        `json:"role"`
	Quant         string        `json:"quant"`
	Shape         []int         `json:"shape"`
	SelectedRoute *string       `json:"selected_route"`
	Status        string        `json:"status"` // promoted | refuted | blocked | unmeasured | candidate
	Candidates    []string      `json:"candidates"`
	EvidenceRefs  []string      `json:"evidence_refs"`
	Compare       *RouteCompare `json:"compare"`
}

// RouteCompare is one role's kernel comparison (boltbeam/search/role_compare.py), nil before any ran. Every
// time in it is a tinygrad Metal runtime time.
type RouteCompare struct {
	PlanID          *string      `json:"plan_id"`
	Plan            *string      `json:"plan"`
	SearchMedianNs  *float64     `json:"search_median_ns"`
	DefaultMedianNs *float64     `json:"default_median_ns"`
	MeasuredCorrect *int         `json:"measured_correct"`
	Candidates      *int         `json:"candidates"`
	Reason          *string      `json:"reason"`
	TimingSource    *string      `json:"timing_source"`
	Kernel          *KernelAlone `json:"kernel"`
	DecidedBy       *string      `json:"decided_by"`
	AB              *AB          `json:"ab"`
}

// KernelAlone is the winning plan alone vs the search's reference kernel, at the role shape, in one harness. The
// reference is not the model's own kernel, so it never says faster or slower; nor does it stand in for the
// whole-model number.
type KernelAlone struct {
	PlanUs          float64  `json:"plan_us"`
	ReferenceUs     *float64 `json:"reference_us"`
	Reference       string   `json:"reference"`
	ModelUsPerCall  *float64 `json:"model_us_per_call"`
	FasterThanModel *bool    `json:"faster_than_model"`
}

// AB is the matched whole-model decode A/B for one role's winning plan.
type AB struct {
	BaselineTokS  *float64 `json:"baseline_tok_s"`
	CandidateTokS *float64 `json:"candidate_tok_s"`
	DeltaPct      *float64 `json:"delta_pct"`
	TokenMatch    *bool    `json:"token_match"`
	RouteBound    *bool    `json:"route_bound"`
}

// CompareReady says whether this machine can compare kernels for a run, and if not, the missing piece and the
// one command that fixes it.
type CompareReady struct {
	Applies bool    `json:"applies"`
	Ready   bool    `json:"ready"`
	Missing *string `json:"missing"`
	Message *string `json:"message"`
	Fix     *string `json:"fix"`
	Fork    *string `json:"fork"`
	// CompareMessage says why kernels cannot be compared here even when the fork is ready (not Metal).
	CompareMessage *string `json:"compare_message"`
	// Runtime names where role times come from, for example "tinygrad's CUDA runtime". Python builds it.
	Runtime *string `json:"runtime"`
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
	Loss     Loss       `json:"loss"`
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

// Loss is the end result: measured vs the speed limit in ms per token, and per role where the run's provider
// loses time. Roles come from that provider's capture, attributed to roles by bytes and count, or from
// tinygrad's own timing.
type Loss struct {
	Status          string     `json:"status"`
	Reason          *string    `json:"reason"`
	LimitTokS       *float64   `json:"limit_tok_s"`
	LimitMs         *float64   `json:"limit_ms"`
	Runtimes        []Runtime  `json:"runtimes"`
	Roles           []RoleLoss `json:"roles"`
	NotAttributedMs *float64   `json:"not_attributed_ms"`
	Source          *string    `json:"source"`
	Missing         *string    `json:"missing"`
	// Refused is why the per-role times are not shown: a role or the token below its floor.
	Refused *string `json:"refused"`
	// Provider is the runtime step 4 measured with; RolesProvider is whose table Roles is (it may be another
	// provider's earlier table); ProviderMissing says why the chosen provider has no table yet.
	Provider        string      `json:"provider"`
	Capture         *Capture    `json:"capture"`
	RolesProvider   *string     `json:"roles_provider"`
	ProviderMissing *string     `json:"provider_missing"`
	Others          []OtherLoss `json:"others"`
	// UnpairedRoles are roles the capture could not split out by calls and bytes; their time is in not attributed.
	UnpairedRoles []UnpairedRole `json:"unpaired_roles"`
}

type UnpairedRole struct {
	Role  string `json:"role"`
	Quant string `json:"quant"`
	Count int    `json:"count"`
}

type Runtime struct {
	Provider string  `json:"provider"`
	TokS     float64 `json:"tok_s"`
	Ms       float64 `json:"ms"`
	LostMs   float64 `json:"lost_ms"`
	PerRole  bool    `json:"per_role"`
	Note     string  `json:"note"`
	Capture  *string `json:"capture"`
}

type RoleLoss struct {
	Role          string  `json:"role"`
	Quant         string  `json:"quant"`
	IdealMs       float64 `json:"ideal_ms"`
	ActualMs      float64 `json:"actual_ms"`
	LostMs        float64 `json:"lost_ms"`
	Share         float64 `json:"share"`
	CallsPerToken float64 `json:"calls_per_token"`
}
