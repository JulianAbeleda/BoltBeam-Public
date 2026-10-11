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
	Source             string             `json:"source"` // registry, or generated: a chip profile measured on this machine
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
	// From `screen chips` only: this machine's chip and the groups Setup draws, in order. Python decides both.
	ThisMachine *ChipHere   `json:"this_machine"`
	Groups      []ChipGroup `json:"groups"`
}

// ChipHere is this machine's first GPU and its profile: known (a registry row or one made here), new, or no_gpu.
type ChipHere struct {
	Name     *string `json:"name"`
	TargetID *string `json:"target_id"`
	Status   string  `json:"status"`
	Source   *string `json:"source"`
	NewID    *string `json:"new_id"` // the id a new profile will get
	Words    string  `json:"words"`
}

// ChipGroup is one heading of Setup's chip list: this machine, measured chips, not measured yet, families.
type ChipGroup struct {
	Key        string `json:"key"`
	Title      string `json:"title"`
	Selectable bool   `json:"selectable"`
	Folded     bool   `json:"folded"`
	Chips      []struct {
		ID    string `json:"id"`
		Words string `json:"words"`
	} `json:"chips"`
}

// ChipScan is `screen autoscan`: the profile kept, generated or measured again, or why measuring failed.
type ChipScan struct {
	Status   string  `json:"status"`
	Action   string  `json:"action"`
	Name     *string `json:"name"`
	TargetID *string `json:"target_id"`
	Source   *string `json:"source"`
	Path     *string `json:"path"`
	Reason   *string `json:"reason"`
}

// Engines is `screen engines`: the engine scan, run now and saved (collectors/engine_scan.py). One row per item
// looked for, with its path and how it was found (env, path, scan: <dir>), or neither.
type Engines struct {
	File      string        `json:"file"`
	ScannedAt string        `json:"scanned_at"`
	Seconds   float64       `json:"seconds"`
	Engines   []EngineFound `json:"engines"`
}

type EngineFound struct {
	Item    string  `json:"item"`
	Env     string  `json:"env"`
	Engine  string  `json:"engine"`
	Path    *string `json:"path"`
	How     *string `json:"how"`
	Skipped *string `json:"skipped"`
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
	// GpuCount is every GPU the driver lists; MultiGpu names them with the limited-support label when more than one.
	GpuCount int     `json:"gpu_count"`
	MultiGpu *string `json:"multi_gpu"`
	// Profile says whether a chip profile fits this GPU: status known or new.
	Profile *struct {
		Status string  `json:"status"`
		ID     *string `json:"id"`
	} `json:"profile"`
}

// LayoutRow is one way an engine can use this machine's GPUs (workflow/layout.py).
type LayoutRow struct {
	ID        string  `json:"id"`
	Label     string  `json:"label"`
	Available bool    `json:"available"`
	State     string  `json:"state"` // available, not_here or not_compatible: Python decides, the screen words it
	Reason    *string `json:"reason"`
}

// LayoutLimit is the limit derived for the run's layout, with its formula and every input and its source.
type LayoutLimit struct {
	Layout  string   `json:"layout"`
	Label   string   `json:"label"`
	Gpus    int      `json:"gpus"`
	Ms      *float64 `json:"ms"`
	TokS    *float64 `json:"tok_s"`
	Formula string   `json:"formula"`
	Inputs  []struct {
		What   string `json:"what"`
		Value  any    `json:"value"`
		Unit   string `json:"unit"`
		Source string `json:"source"`
	} `json:"inputs"`
	Support *string `json:"support"`
	Reason  *string `json:"reason"`
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

// SavedRun is `screen save`: where the run was exported.
type SavedRun struct {
	ID      string   `json:"id"`
	Dir     string   `json:"dir"`
	Files   []string `json:"files"`
	SavedAt string   `json:"saved_at"`
}

// Saved is `screen saved`: the saved runs, newest first.
type Saved struct {
	Root string `json:"root"`
	Runs []struct {
		ID         string   `json:"id"`
		Dir        string   `json:"dir"`
		ModelID    *string  `json:"model_id"`
		TargetID   *string  `json:"target_id"`
		Provider   string   `json:"provider"`
		SavedAt    string   `json:"saved_at"`
		TokS       *float64 `json:"tok_s"`
		PctOfLimit *float64 `json:"pct_of_limit"`
	} `json:"runs"`
}

// Ceilings is `screen ceilings`: this model's decode limit on every registered chip with a ceiling.
type Ceilings struct {
	ModelID string `json:"model_id"`
	Chips   []struct {
		ID               string   `json:"id"`
		TokS             *float64 `json:"tok_s"`
		FloorMs          float64  `json:"floor_ms"`
		PeakBandwidthGBs float64  `json:"peak_bandwidth_gbs"`
	} `json:"chips"`
}

type Ceiling struct {
	Kind             string       `json:"kind"`
	ModelID          string       `json:"model_id"`
	Target           Target       `json:"target"`
	PeakBandwidthGBs float64      `json:"peak_bandwidth_gbs"`
	BandwidthSource  *string      `json:"bandwidth_source"` // this machine's memory speed and its source; nil: the registry's
	PeakTFLOPS       float64      `json:"peak_tflops"`
	TruthStatus      string       `json:"truth_status"`
	RidgeIntensity   float64      `json:"ridge_intensity"`
	Assumptions      []string     `json:"assumptions"`
	Decode           CeilingBlock `json:"decode"`
	Prefill          CeilingBlock `json:"prefill"`
}

type Stage struct {
	Key   string `json:"key"`
	Label string `json:"label"`
	Note  string `json:"note"`
	Done  bool   `json:"done"`
	// State is "done", "not_needed" (StateNote says why) or "open"; Python decides it.
	State     string   `json:"state"`
	StateNote *string  `json:"state_note"`
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
	// Batches are the batch sizes timed; 1 is always among them.
	Batches []int `json:"batches"`
	// Capture is the per-role capture the measuring machine had, with the Setup choice of measurement.
	Capture *MeasureCapture `json:"capture"`
}

// MeasureCapture is measure_status.json's capture; Measurement is nil for runs from before the choice existed.
type MeasureCapture struct {
	Measurement *Measurement `json:"measurement"`
}

// Measurement is how a run timed its roles (providers.resolve_measurement): the choice made in Setup
// ("auto", "in_model", "generic"), what it resolved to, and why in-model was not used when auto fell back.
type Measurement struct {
	Choice   string  `json:"choice"`
	Chosen   *string `json:"chosen"`
	Label    *string `json:"label"`
	Fallback *string `json:"fallback"`
	Reason   *string `json:"reason"`
}

// MeasurementOptions are the two ways to time roles for one engine here (providers.measurement_options).
type MeasurementOptions struct {
	Default  *string             `json:"default"`
	Fallback *string             `json:"fallback"`
	Options  []MeasurementOption `json:"options"`
}

type MeasurementOption struct {
	ID        string  `json:"id"` // in_model | generic
	Label     string  `json:"label"`
	About     string  `json:"about"`
	Available bool    `json:"available"`
	Reason    *string `json:"reason"`
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
	State     string  `json:"state"` // available, not_here or not_compatible: Python decides, the screen words it
	Reason    *string `json:"reason"`
	Capture   Capture `json:"capture"`
	// Layouts are the GPU layouts this engine can run here; one GPU unless the machine has more.
	Layouts []LayoutRow `json:"layouts"`
	// BatchOverOne says the engine decodes more than one stream; nil from an older Python means yes.
	BatchOverOne *bool `json:"batch_over_one"`
	// Measurement is how this engine's roles can be timed here; nil from an older Python.
	Measurement *MeasurementOptions `json:"measurement"`
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
	TokS            *float64   `json:"tok_s"`
	Ms              *float64   `json:"ms"`
	Missing         *string    `json:"missing"` // "not measured: <reason>" when the run holds no speed
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
	Where       string         `json:"where"` // runs, work or saved: set by `runs --all`
	Dir         string         `json:"dir"`
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

// RouteCompare is one role's kernel comparison (boltbeam/search/role_compare.py), nil before any ran. Its kernel
// times are BoltBeam's kernel timer on the provider's source; its whole-model times are tinygrad runtime times.
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
	Also           []AlsoAt     `json:"also"` // the other contexts step 4 measured, labelled, never the headline
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
	BandwidthSource  *string  `json:"bandwidth_source"`
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
	// Measurement is how this run timed its roles, as chosen in Setup; nil for older runs.
	Measurement *Measurement `json:"measurement"`
	Routes      []Route      `json:"routes"`
	Timing      Timing       `json:"timing"`
	Regimes     []Regime     `json:"regimes"`
	Blocked     []Need       `json:"blocked"`
	Report      *string      `json:"report"`
	Batches     []BatchRow   `json:"batches"`
	// Probe is BoltBeam's own kernel (reference) per role: GB/s and share of peak, not the engine's kernel.
	Probe *Probe `json:"probe"`
}

// Probe is the building-block probe's rows (workflow/screen.py probe_rows): "measured" with rows, or "absent".
type Probe struct {
	Status          string    `json:"status"`
	Reason          *string   `json:"reason"`
	Label           string    `json:"label"`
	DispatchFloorUs *float64  `json:"dispatch_floor_us"`
	Rows            []RateRow `json:"rows"`
}

// RateRow is one role's kernel read rate: µs per call, GB/s and the share of the chip's peak. A cross-check row
// timed with a dispatch floor also carries UsPerCallLessFloor, and its Gbs and PctPeak are on that time
// (workflow/screen.py cross_check, the same rule as the isolated per-role table).
type RateRow struct {
	Role               string   `json:"role"`
	Quant              string   `json:"quant"`
	UsPerCall          *float64 `json:"us_per_call"`
	UsPerCallLessFloor *float64 `json:"us_per_call_less_floor"`
	Gbs                *float64 `json:"gbs"`
	PctPeak            *float64 `json:"pct_peak"`
}

// Step is THE measured token (tie_out.measured_step): the row every headline reads, with the engine's own account
// of how it ran (Source) and its graph state, and the other contexts measured (Also, never the headline).
type Step struct {
	Context     *int     `json:"context"`
	TokS        float64  `json:"tok_s"`
	Ms          float64  `json:"ms"`
	Source      *string  `json:"source"`
	Rule        string   `json:"rule"`
	GraphFailed bool     `json:"graph_failed"`
	GraphError  *string  `json:"graph_error"`
	Also        []AlsoAt `json:"also"`
}

type AlsoAt struct {
	Context *int    `json:"context"`
	TokS    float64 `json:"tok_s"`
}

// CrossCheck is the engine's kernels timed alone beside an in-model capture.
type CrossCheck struct {
	Words        string    `json:"words"`
	Reason       *string   `json:"reason"`
	Rows         []RateRow `json:"rows"`
	ColumnsWords *string   `json:"columns_words"` // which column the rates are on, when the rows carry a floor
}

// Latency is the figure the reason rule's Little's law uses, and where it came from (measured floor or assumed).
type Latency struct {
	Us     float64 `json:"us"`
	Source string  `json:"source"`
}

// BatchRow is one measured point of step 4 beside its own limit (tie_out.batch_limit).
type BatchRow struct {
	Context    float64  `json:"context"`
	Batch      int      `json:"batch"`
	StepMs     float64  `json:"step_ms"`
	TokSStream float64  `json:"tok_s_stream"`
	TokSTotal  float64  `json:"tok_s_total"`
	PctOfLimit *float64 `json:"pct_of_limit"`
	Limit      struct {
		TokSStream float64 `json:"tok_s_stream"`
		TokSTotal  float64 `json:"tok_s_total"`
		Bound      string  `json:"bound"`
	} `json:"limit"`
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
	// NotTimed are limit roles the kernel timer had no adapter for, each with its reason.
	NotTimed []NotTimedRole `json:"not_timed"`
	// TieOut is the measured token line by line against the limit (workflow/tie_out.py); RoleRule is the
	// sentence, with its numbers, behind each role's Reason.
	TieOut     *TieOut      `json:"tie_out"`
	Layout     *LayoutLimit `json:"layout"`
	RoleRule   *string      `json:"role_rule"`
	Search     *Search      `json:"search"`
	Findings   []Finding    `json:"findings"`
	Step       *Step        `json:"step"`
	CrossCheck *CrossCheck  `json:"cross_check"`
	Latency    *Latency     `json:"latency"`
	// RoleSource is "isolated" (each kernel timed alone by BoltBeam's kernel timer) or "in_model" (a capture of
	// the real token); RoleSourceWords names it for the Facts. Estimate is set for an isolated table only: the
	// token split by role is the isolated times scaled to the token, labelled an estimate.
	RoleSource      *string   `json:"role_source"`
	RoleSourceWords *string   `json:"role_source_words"`
	Estimate        *Estimate `json:"estimate"`
	// WhereTokenGoes is the tie-out as one ranked table of what each part costs the token
	// (workflow/tie_out.py where_token_goes); every renderer reads it, none recomputes it.
	WhereTokenGoes *Where `json:"where_token_goes"`
	// Per is "per prefill" on a prefill run (workflow/prefill.py); nil on decode, whose times are per token.
	Per *string `json:"per"`
}

// Where is "Where the token goes": rows worst first, summing to TokenMs (now) and LimitMs (at the limit).
// BaseMs is the headline token tok/s if fixed is taken from; Estimate says the rows come from kernels timed alone.
type Where struct {
	Title    string     `json:"title"`
	Estimate bool       `json:"estimate"`
	TokenMs  float64    `json:"token_ms"`
	BaseMs   float64    `json:"base_ms"`
	LimitMs  float64    `json:"limit_ms"`
	LostMs   float64    `json:"lost_ms"`
	Columns  []string   `json:"columns"`
	NoLimit  string     `json:"no_limit"`
	Words    string     `json:"words"`
	Rows     []WhereRow `json:"rows"`
	// SumCells is the sum row as text, Cells on each row the row as text after its name, in Columns' order: the
	// renderers print them and format no number of their own (one table for decode and prefill).
	SumCells []string `json:"sum_cells"`
}

// WhereRow is one part of the token. LimitMs is nil where the limit has no bytes for it ("no limit"); Share and
// TokSIfFixed are nil when nothing is lost in all or the row's loss is the whole token; PctPeak is set for roles.
type WhereRow struct {
	Name        string   `json:"name"`
	What        string   `json:"what"` // role | kernels | rest | limit | gaps
	NowMs       float64  `json:"now_ms"`
	LimitMs     *float64 `json:"limit_ms"`
	LostMs      float64  `json:"lost_ms"`
	Share       *float64 `json:"share"`
	TokSIfFixed *float64 `json:"tok_s_if_fixed"`
	PctPeak     *float64 `json:"pct_peak"`
	Cells       []string `json:"cells"`
}

// Estimate is an isolated run's scaled split of the token (workflow/tie_out.py _isolated). When the rows carry a
// dispatch floor, the estimate is made from the times less that floor (IsolatedSumLessFloorMs) and FloorWords
// says so ("less the 3.9 µs dispatch floor per launch"); IsolatedSumMs stays the measured sum. ColumnsWords then
// says once which per-role columns are measured and which are less the floor (tie_out.ISOLATED_COLUMNS).
type Estimate struct {
	Label                  string   `json:"label"`
	Method                 string   `json:"method"`
	Scale                  float64  `json:"scale"`
	IsolatedSumMs          float64  `json:"isolated_sum_ms"`
	IsolatedSumLessFloorMs *float64 `json:"isolated_sum_less_floor_ms"`
	FloorUs                *float64 `json:"floor_us"`
	FloorWords             *string  `json:"floor_words"`
	ColumnsWords           *string  `json:"columns_words"`
	TokenMs                float64  `json:"token_ms"`
	WeightMs               float64  `json:"weight_ms"`
	OtherMs                float64  `json:"other_ms"`
	OtherHow               string   `json:"other_how"`
	Scaled                 bool     `json:"scaled"`
}

type TieOut struct {
	Provider        string    `json:"provider"`
	Context         float64   `json:"context"`
	LimitMsCtx1     float64   `json:"limit_ms_ctx1"`
	LimitMs         float64   `json:"limit_ms"`
	KvMs            float64   `json:"kv_ms"`
	KvSource        string    `json:"kv_source"`
	ShowBoth        bool      `json:"show_both"`
	UntracedMs      *float64  `json:"untraced_ms"`
	UntracedContext *int      `json:"untraced_context"`
	TokenMs         *float64  `json:"token_ms"`
	TokenSource     *string   `json:"token_source"`
	BusyMs          *float64  `json:"busy_ms"`
	Lines           []TieLine `json:"lines"`
	Missing         *string   `json:"missing"`
	Refused         *string   `json:"refused"`
	Isolated        bool      `json:"isolated"` // the kernels were timed alone: only the ideal and the token are tied out
	Estimate        *Estimate `json:"estimate"`
}

type TieLine struct {
	Label string    `json:"label"`
	Ms    float64   `json:"ms"`
	How   string    `json:"how"` // derived | measured | difference
	Parts []TiePart `json:"parts"`
	// Kernels are set on the "weight kernels, unattributed" line only: GEMVs no role took, named with their launches.
	Kernels []TieKernel `json:"kernels"`
}

type TiePart struct {
	Kind string  `json:"kind"`
	Ms   float64 `json:"ms"`
}

type TieKernel struct {
	Kernel        string   `json:"kernel"`
	CallsPerToken float64  `json:"calls_per_token"`
	UsPerCall     *float64 `json:"us_per_call"`
	Ms            float64  `json:"ms"`
}

type UnpairedRole struct {
	Role  string `json:"role"`
	Quant string `json:"quant"`
	Count int    `json:"count"`
}

// NotTimedRole is a limit role the kernel timer had no adapter for: shown with its reason, left out of the floor
// rule and the sums (collectors/tinygrad_role_time.py loss).
type NotTimedRole struct {
	Role   string  `json:"role"`
	Quant  string  `json:"quant"`
	Status string  `json:"status"`
	Reason *string `json:"reason"`
}

type Runtime struct {
	Provider string  `json:"provider"`
	TokS     float64 `json:"tok_s"`
	Ms       float64 `json:"ms"`
	LostMs   float64 `json:"lost_ms"`
	PerRole  bool    `json:"per_role"`
	Note     string  `json:"note"`
	Capture  *string `json:"capture"`
	Isolated bool    `json:"isolated"` // a sum of kernels each timed alone: not a token, so no "lost"
}

type RoleLoss struct {
	Role          string   `json:"role"`
	Quant         string   `json:"quant"`
	IdealMs       float64  `json:"ideal_ms"`
	ActualMs      float64  `json:"actual_ms"`
	LostMs        float64  `json:"lost_ms"`
	Share         float64  `json:"share"`
	CallsPerToken float64  `json:"calls_per_token"`
	PctPeak       *float64 `json:"pct_peak"`
	UsPerCall     *float64 `json:"us_per_call"` // isolated runs: the measured time, floor included
	// UsPerCallLessFloor is an isolated row's time less the dispatch floor (tie_out.role_why): the time the rule
	// judges, so PctPeak and Gbs are on it; nil for an in-model capture or a trace with no floor.
	UsPerCallLessFloor *float64 `json:"us_per_call_less_floor"`
	MbPerCall          *float64 `json:"mb_per_call"`
	Gbs                *float64 `json:"gbs"`
	EstMs              *float64 `json:"est_ms"`      // isolated runs: the role's time scaled to the token, an estimate
	EstLostMs          *float64 `json:"est_lost_ms"` // EstMs minus the ideal, an estimate
	// Reason is the rule's word for the role; when an isolated role's reading is not firm (its samples spread more
	// than ±10%, or the dispatch floor is over a third of its measured time: tie_out.noisy_words) it carries the
	// suffix "; noisy: ±N%, F µs floor under a T µs kernel" (tie_out.NOISY) and ReasonWord keeps the firm word.
	// SpreadPct is that spread (P90 - P10 over the median, %), from the kernel timer's rows; nil for an in-model capture.
	Reason     string   `json:"reason"`
	ReasonWord *string  `json:"reason_word"`
	Noisy      bool     `json:"noisy"`
	SpreadPct  *float64 `json:"spread_pct"`
	// BestFound and Verdict are Run's kernel search for the role (search/role_compare.py role_verdict): the best
	// plan alone against the model's own kernel per call, and applied, found_not_applied, none_faster or
	// not_searched with VerdictReason.
	BestFound     *BestFound `json:"best_found"`
	Verdict       *string    `json:"verdict"`
	VerdictReason *string    `json:"verdict_reason"`
	Candidates    *int       `json:"candidates"` // plans the kernel search measured for this role
	// Evidence points into the run's raw JSON (workflow/evidence.py): the kernel rows, the search, the machine.
	Evidence []Pointer `json:"evidence"`
}

// Pointer is one place in the run folder: a file, and a path inside it (empty for the whole file).
type Pointer struct {
	File string  `json:"file"`
	Path *string `json:"path"`
}

// Finding is a next-step item read off facts the run holds: its verdict, the stats, one lever, the pointers.
type Finding struct {
	Verdict  string    `json:"verdict"`
	What     string    `json:"what"`
	Ms       float64   `json:"ms"`
	Do       string    `json:"do"`
	Evidence []Pointer `json:"evidence"`
}

// BestFound is the fastest plan the kernel search measured for a role. Speedup is the model's own kernel per call
// over the plan alone, both measured; nil when the model time is missing.
type BestFound struct {
	Plan    string   `json:"plan"`
	PlanUs  *float64 `json:"plan_us"`
	ModelUs *float64 `json:"model_us"`
	Speedup *float64 `json:"speedup"`
	Text    string   `json:"text"`
}

// Search is Run's kernel search stage for this run: searched, skipped (Reason says why) or not_run.
type Search struct {
	Status  string      `json:"status"`
	Reason  *string     `json:"reason"`
	Seconds *float64    `json:"seconds"`
	Next    *SearchNext `json:"next"`
}

// SearchNext is the next-step item for roles with a faster kernel found but not applied. Ms is measured:
// the sum of (model µs - plan µs) x calls per token.
type SearchNext struct {
	What     string    `json:"what"`
	Ms       float64   `json:"ms"`
	Do       string    `json:"do"`
	Evidence []Pointer `json:"evidence"`
}

// Emitted is `screen emit`: where Emit wrote the gameplan (workflow/gameplan.py), the plan itself and its markdown,
// the on-paper version of the same JSON. The screen shows Markdown; it never reads the files.
type Emitted struct {
	Kind     string   `json:"kind"`
	ID       string   `json:"id"`
	Dir      string   `json:"dir"`
	Files    []string `json:"files"`
	Plan     Gameplan `json:"plan"`
	Markdown string   `json:"markdown"`
}

// Gameplan is gameplan.json: the header, one block per role (worst first) with the four lineage layers, the footer.
type Gameplan struct {
	Kind      string         `json:"kind"`
	ID        string         `json:"id"`
	Header    GameplanHeader `json:"header"`
	OrderedBy string         `json:"ordered_by"`
	Roles     []GameplanRole `json:"roles"`
	Footer    struct {
		FilesRead []string `json:"files_read"`
		Date      string   `json:"date"`
		Sentence  string   `json:"sentence"`
	} `json:"footer"`
}

type GameplanHeader struct {
	ModelID     string    `json:"model_id"`
	TargetID    string    `json:"target_id"`
	Engine      string    `json:"engine"`
	Batch       int       `json:"batch"`
	Measurement *string   `json:"measurement"`
	TokenMs     *float64  `json:"token_ms"`
	TokS        *float64  `json:"tok_s"`
	LimitMs     *float64  `json:"limit_ms"`
	LimitTokS   *float64  `json:"limit_tok_s"`
	PctOfLimit  *float64  `json:"pct_of_limit"`
	LostMs      *float64  `json:"lost_ms"`
	Evidence    []Pointer `json:"evidence"`
}

// GameplanRole is one role block: its facts, the plan in one phrase, and the layers in order (BubbleBeam,
// FutureSight, Measured, Promotion), each recorded or "not recorded in this run" with the file it would be in.
type GameplanRole struct {
	Rank           int             `json:"rank"`
	Role           string          `json:"role"`
	Quant          string          `json:"quant"`
	Rows           *int            `json:"rows"`
	Cols           *int            `json:"cols"`
	LostMs         *float64        `json:"lost_ms"`
	LostIsEstimate bool            `json:"lost_is_estimate"`
	Reason         *string         `json:"reason"`
	Verdict        string          `json:"verdict"`
	Plan           *string         `json:"plan"`
	Layers         []GameplanLayer `json:"layers"`
	Evidence       []Pointer       `json:"evidence"`
}

type GameplanLayer struct {
	Layer    string    `json:"layer"`
	Title    string    `json:"title"`
	Recorded bool      `json:"recorded"`
	Lines    []string  `json:"lines"`
	Evidence []Pointer `json:"evidence"`
}
