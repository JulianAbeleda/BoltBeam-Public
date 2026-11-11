// Package ui owns the screen. Every view here is a pure function of seam facts and a size, so a test can
// render a screen to a string and pin it; model.go wires the same functions to Bubble Tea.
package ui

// BoltBeam's own plain vocabulary (gui/README.md and gui/run-graph.html). Main lines use the plain word only;
// detail views show the plain word with the record's own name beside it, muted. Stage and role names are not here:
// they come from the seam's one vocabulary (`screen words`, vocab in render_text.go).
var plainStatus = map[string]string{
	"not_analyzed": "not planned yet", "needs_measurement": "needs measuring", "policy_seeded": "plan ready",
}

var plainRoute = map[string]string{
	"promoted": "kept", "refuted": "ruled out", "blocked": "undecided", "unmeasured": "default kernel, none compared yet", "candidate": "to try",
}

var routeMark = map[string]string{"promoted": "kept", "refuted": "fail", "blocked": "crossed", "unmeasured": "wait"}

var plainBucket = map[string]string{
	"at_peak": "at the speed limit", "gemv_codegen_capped": "reads memory slower than it could",
	"latency_bound": "waiting on memory", "elementwise_dilution": "small kernel, dead time",
	"activation_bound": "real activation work", "timing_inconclusive": "not clear yet",
}

var plainRegime = map[string]string{
	"streaming_bound": "reads memory at full speed", "occupancy_starved": "not enough work in flight",
	"dequant_bound": "slowed by unpacking the numbers", "metadata_bound": "slowed by the scale tables",
	"latency_bound": "waiting on memory", "compute_bound": "limited by the sums", "inconclusive": "not clear yet",
}

var plainLimit = map[string]string{"memory": "reading weights", "compute": "doing sums"}

// plainVerdict words a role's kernel search verdict; report/html.py PLAIN_VERDICT says the same.
var plainVerdict = map[string]string{"applied": "applied", "found_not_applied": "found, not applied",
	"none_faster": "none faster", "not_reproduced": "not reproduced by BoltBeam", "not_searched": "not searched"}

var plainNeed = map[string]string{"probe_evidence": "the building-block tests", "timing_trace": "a timing trace"}

// captureWords names how per-role time was taken, in the screen's words.
var captureWords = map[string]string{
	"tinygrad-profile-events": "tinygrad's own timing",
	"nsys":                    "captured with nsys",
	"rocprofv3":               "captured with rocprofv3",
	"metal-system-trace":      "captured with Metal System Trace",
	// collectors/engine_kernels.py METHOD and WORDS: the engine's own kernels, timed alone by BoltBeam's kernel timer
	"boltbeam-kernel-timer-isolated": "isolated, timed by BoltBeam's kernel timer",
}

// notChecked is a check with no repeat band for its setup; notCheckedHow, said once per page, is how to get one, so
// "not checked" never reads as a fault.
const (
	notChecked    = "not checked: no repeat band matches this setup."
	notCheckedHow = "Not checked is expected on a first run. A verdict needs the range of n identical runs (n = 2 is weak; " +
		"a row then reads moved by chance 2/(n+1) of the time): pick Repeats on Setup, or run the same setup n times, then " +
		"`python -m boltbeam.workflow.screen repeat-bands --runs A B ... --out bands.json`, and give later runs `--bands bands.json`."
)

// noKernelChoice replaces the per-role table while no role has had kernels compared; report/html.py says the same.
const noKernelChoice = "No kernels compared yet. Every role runs the default kernel."

// ListRows is how many rows a picker or the Saved list shows at once, whatever the terminal height: a long list
// scrolls inside that window (DetailActions counts what is above and below) instead of filling the screen.
const ListRows = 10
