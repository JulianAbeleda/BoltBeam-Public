// Package ui owns the screen. Every view here is a pure function of seam data and a width, so a test can
// render a screen to a string and pin it; model.go wires the same functions to Bubble Tea.
package ui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/jobs"
	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// Plain mode uses BoltBeam's own plain vocabulary (gui/README.md and gui/run-graph.html: the stage names,
// kept / ruled out / undecided, "the speed limit"); technical mode uses the artifacts' own names.
var plainStage = map[string]string{
	"load": "Read the model", "autoscan": "Check the machine", "analyze": "Plan what to try",
	"runner_plan": "Prepare the handoff", "ingest_probe": "Test the building blocks", "ingest_timing": "Time the real run",
	"output": "Package the result",
}

var plainStatus = map[string]string{
	"not_analyzed": "not planned yet", "needs_measurement": "needs measuring", "policy_seeded": "plan ready",
}

var plainRoute = map[string]string{
	"promoted": "kept", "refuted": "ruled out", "blocked": "undecided", "unmeasured": "not measured yet", "candidate": "to try",
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

var plainRole = map[string]string{
	"attn_kv": "attention keys and values", "attn_qo": "attention query and output", "ffn_gate_up": "feed-forward in",
	"ffn_down": "feed-forward out", "lm_head": "vocabulary output", "embed": "token embedding",
}

var plainLimit = map[string]string{"memory": "reading weights", "compute": "doing sums"}

var plainNeed = map[string]string{"probe_evidence": "the building-block tests", "timing_trace": "a timing trace"}

// word picks the plain word for a key, or the key itself in technical mode (and when no plain word exists).
func word(table map[string]string, key string, technical bool) string {
	if !technical {
		if plain, ok := table[key]; ok {
			return plain
		}
	}
	return key
}

func num(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.4g", *v)
}

func gb(bytes float64) string { return fmt.Sprintf("%.2f GB", bytes/1e9) }

// us prints microseconds as a whole number; kernels are never sub-microsecond on a GPU timeline.
func us(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.0f", *v)
}

func count(v *int) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%d", *v)
}

func shapeStr(shape []int) string {
	parts := make([]string, len(shape))
	for i, s := range shape {
		parts[i] = fmt.Sprintf("%d", s)
	}
	return strings.Join(parts, "x")
}

func deref(s *string) string {
	if s == nil {
		return "-"
	}
	return *s
}

// table lays out rows two spaces apart; the first row is the header, `cursor` names the highlighted row.
func table(rows [][]string, cursor int) string {
	widths := []int{}
	for _, row := range rows {
		for i, cell := range row {
			if i >= len(widths) {
				widths = append(widths, 0)
			}
			if w := lipgloss.Width(cell); w > widths[i] {
				widths[i] = w
			}
		}
	}
	var b strings.Builder
	for r, row := range rows {
		for i, cell := range row {
			if r == 0 {
				cell = stHeader.Render(cell)
			}
			b.WriteString(cell + strings.Repeat(" ", widths[i]-lipgloss.Width(cell)))
			if i < len(row)-1 {
				b.WriteString("  ")
			}
		}
		if cursor >= 0 && r == cursor+1 {
			b.WriteString(stCursor.Render("  ◂"))
		}
		b.WriteString("\n")
	}
	return b.String()
}

func mode(technical bool) string {
	if technical {
		return stMuted.Render("technical")
	}
	return stMuted.Render("plain")
}

// --- Model ---------------------------------------------------------------------------------------------------

// ModelScreen is what the model screen shows: the path being edited or set, the chip picked, the profile read.
type ModelScreen struct {
	Path    string
	Input   string
	Editing bool
	Targets *seam.Targets
	Target  int
	Profile *seam.Profile
	Busy    bool
}

func chipLine(targets *seam.Targets, idx int, technical bool) string {
	if targets == nil || len(targets.Targets) == 0 {
		return stMuted.Render("no chips listed yet")
	}
	if idx < 0 || idx >= len(targets.Targets) {
		idx = 0
	}
	t := targets.Targets[idx]
	line := stCursor.Render("◂ ") + stAccent.Render(t.ID) + stCursor.Render(" ▸")
	if t.Scope != nil && !technical {
		line += "  " + *t.Scope
	}
	if technical {
		line += "  " + stMuted.Render(fmt.Sprintf("%s · %s · memory %s GB/s (%s) · compute %s", t.Backend, t.BackendStatus,
			num(t.MemoryBandwidthGBs), t.FactStatus.MemoryBandwidthGBs, tflops(t)))
	}
	if t.HasCeiling {
		line += "  " + mark("pass") + stMuted.Render(" has a speed limit")
	} else {
		line += "  " + mark("crossed") + stMuted.Render(" no measured speeds, so no speed limit")
	}
	return line
}

func tflops(t seam.Target) string {
	parts := []string{}
	for k, v := range t.MatrixTFLOPS {
		parts = append(parts, fmt.Sprintf("%s %.4g TFLOP/s (matrix)", k, v))
	}
	if len(parts) == 0 {
		for k, v := range t.PeakTFLOPS {
			parts = append(parts, fmt.Sprintf("%s %.4g TFLOP/s", k, v))
		}
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, ", ")
}

// ModelView: pick a model file and a chip, then the role census `inspect` read.
func ModelView(s ModelScreen, technical bool, width int) string {
	var b strings.Builder
	if s.Editing {
		fmt.Fprintf(&b, "%s %s\n", stMuted.Render("model"), stAccent.Render("▸ ")+s.Input+stCursor.Render("▏"))
	} else if s.Path == "" {
		fmt.Fprintf(&b, "%s %s\n", stMuted.Render("model"), stWarn.Render("none yet. Press e and type the path to a GGUF or safetensors file."))
	} else {
		fmt.Fprintf(&b, "%s %s\n", stMuted.Render("model"), s.Path)
	}
	fmt.Fprintf(&b, "%s  %s\n", stMuted.Render("chip "), chipLine(s.Targets, s.Target, technical))
	b.WriteString(stMuted.Render("e edit the path · [ ] pick the chip · enter read the model · s start a run"))
	out := []string{box("Model", b.String(), width, false)}

	switch {
	case s.Busy:
		out = append(out, box("Reading", stMuted.Render("Reading the model file…"), width, true))
	case s.Profile == nil:
		out = append(out, box("Roles", stMuted.Render("Nothing read yet. The model is never executed; only its shape is read."), width, true))
	default:
		p := s.Profile
		var h strings.Builder
		facts := []string{fmt.Sprintf("%s layers", count(p.LayerCount)), fmt.Sprintf("hidden %s", count(p.HiddenSize)),
			fmt.Sprintf("feed-forward %s", count(p.FFNSize)), fmt.Sprintf("vocabulary %s", count(p.VocabSize))}
		if p.Metadata.TensorCount != nil {
			facts = append(facts, fmt.Sprintf("%d tensors", *p.Metadata.TensorCount))
		}
		if len(p.Metadata.QuantTypes) > 0 {
			facts = append(facts, strings.Join(p.Metadata.QuantTypes, ", "))
		}
		if a := p.Metadata.Attention; a.HeadCount != nil && a.HeadCountKV != nil {
			facts = append(facts, fmt.Sprintf("attention %d heads, %d kv heads, dim %s", *a.HeadCount, *a.HeadCountKV, count(a.HeadDim)))
		}
		h.WriteString(strings.Join(facts, " · ") + "\n")
		if !p.Complete {
			h.WriteString(stWarn.Render(glyphWarn+" The profile is incomplete: some roles could not be read.") + "\n")
		}
		if technical {
			fmt.Fprintf(&h, "%s %s · %s %s · %s %s\n", stMuted.Render("architecture"), deref(p.Architecture), stMuted.Render("format"),
				p.Metadata.FormatFamily, stMuted.Render("source"), p.Source)
		}
		header := []string{"ROLE", "SHAPE", "QUANT", "TENSORS"}
		if technical {
			header = []string{"ROLE", "CLASS", "SHAPE", "QUANT", "TENSORS", "FIRST TENSOR"}
		}
		rows := [][]string{header}
		for _, r := range p.Roles {
			shape := fmt.Sprintf("%dx%d", r.Rows, r.Cols)
			if technical {
				rows = append(rows, []string{r.Role, stMuted.Render(r.RoleClass), shape, r.Quant, fmt.Sprintf("%d", r.Count), stMuted.Render(r.TensorName)})
			} else {
				rows = append(rows, []string{word(plainRole, r.Role, false), shape, r.Quant, fmt.Sprintf("%d", r.Count)})
			}
		}
		h.WriteString(table(rows, -1))
		title := p.ModelID + " · " + word(map[string]string{"dense_decoder": "a dense decoder"}, p.ArchitectureClass, technical)
		out = append(out, box(title, h.String(), width, false))
	}
	return strings.Join(out, "\n")
}

// --- Ceiling ----------------------------------------------------------------------------------------------------

func roleRows(roles []seam.CeilingRole, technical bool) string {
	rows := [][]string{{"ROLE", "QUANT", "SHARE", "BYTES", "FLOOR ms", "LIMIT"}}
	for _, r := range roles {
		rows = append(rows, []string{word(plainRole, r.Role, technical), r.Quant, bar(r.Share, 10) + fmt.Sprintf(" %2.0f%%", r.Share*100),
			gb(r.BytesMoved), fmt.Sprintf("%.2f", r.FloorMs), stMuted.Render(word(plainLimit, r.Regime, technical))})
	}
	return table(rows, -1)
}

// CeilingView: the roofline. What the memory and the compute limit say the best tokens/s can be.
func CeilingView(c *seam.Ceiling, busy bool, technical bool, width int) string {
	if busy {
		return box("Speed limit", stMuted.Render("Working out the speed limit…"), width, true)
	}
	if c == nil {
		return box("Speed limit", "No model read yet. Press 1, pick a model and a chip, and press enter.", width, true)
	}
	d, p := c.Decode, c.Prefill
	var b strings.Builder
	if d.TokS != nil {
		fmt.Fprintf(&b, "%s %s\n", stAccent.Render(glyphBolt), stHeader.Render(fmt.Sprintf("%.1f tokens per second, at best.", *d.TokS)))
	}
	if technical {
		fmt.Fprintf(&b, "decode: %s per token at context 1, floor %.2f ms, regime %s · ridge %.1f FLOP/byte\n", gb(d.BytesMoved), d.FloorMs, d.Regime, c.RidgeIntensity)
	} else {
		fmt.Fprintf(&b, "Each token reads %s of weights.\n", gb(d.BytesMoved))
		fmt.Fprintf(&b, "The memory moves %.1f GB/s, so one token takes at least %.1f ms.\n", c.PeakBandwidthGBs, d.FloorMs)
		fmt.Fprintf(&b, "What limits it: %s.\n", word(plainLimit, d.Regime, false))
	}
	if p.TokS != nil {
		if technical {
			fmt.Fprintf(&b, "prefill: context %d, %s, floor %.0f ms, %.0f tok/s, regime %s\n", p.Context, gb(p.BytesMoved), p.FloorMs, *p.TokS, p.Regime)
		} else {
			fmt.Fprintf(&b, "Reading a prompt of %d tokens: %.0f ms at best, %.0f tokens per second, limited by %s.\n",
				p.Context, p.FloorMs, *p.TokS, word(plainLimit, p.Regime, false))
		}
	}
	t := c.Target
	fmt.Fprintf(&b, "%s", stMuted.Render(fmt.Sprintf("memory %.1f GB/s (%s) · compute %.4g TFLOP/s (%s) · %s",
		c.PeakBandwidthGBs, t.FactStatus.MemoryBandwidthGBs, c.PeakTFLOPS, t.FactStatus.PeakTFLOPS, truth(c.TruthStatus, technical))))
	out := []string{box("Speed limit: "+c.ModelID+" on "+t.ID, b.String(), width, false)}
	out = append(out, box("Per role, one token", roleRows(d.Roles, technical), width, false))
	if technical {
		out = append(out, box(fmt.Sprintf("Per role, prefill at %d", p.Context), roleRows(p.Roles, technical), width, true))
		out = append(out, box("Assumptions", strings.Join(c.Assumptions, "\n"), width, true))
	}
	return strings.Join(out, "\n")
}

func truth(status string, technical bool) string {
	if technical {
		return "truth_status " + status
	}
	if status == "modeled" {
		return "this is arithmetic, not a measurement"
	}
	return status
}

// --- Run -----------------------------------------------------------------------------------------------------

// RunScreen is the run list, the opened run, and the process this machine started for it.
type RunScreen struct {
	Runs   *seam.Runs
	Cursor int
	Run    *seam.Run
	Job    *jobs.Job
	Tail   []string
	Spin   string
}

func status(s string, technical bool) string {
	switch s {
	case "policy_seeded":
		return stOK.Render(word(plainStatus, s, technical))
	case "needs_measurement":
		return stWarn.Render(word(plainStatus, s, technical))
	}
	return stMuted.Render(word(plainStatus, s, technical))
}

func measuredMarks(m seam.Measured) string {
	parts := []string{}
	for _, item := range []struct {
		ok   bool
		name string
	}{{m.Probe, "probe"}, {m.Timing, "timing"}} {
		if item.ok {
			parts = append(parts, mark("pass")+" "+item.name)
		} else {
			parts = append(parts, mark("wait")+" "+item.name)
		}
	}
	return strings.Join(parts, "  ")
}

// stageState: the manifest says done; the live log says running or failed; an open need says waiting.
func stageState(st seam.Stage, events map[string]string, alive bool, blocked []seam.Need) string {
	switch events[st.Key] {
	case "failed":
		return "fail"
	case "running":
		if alive {
			return "run"
		}
	}
	if st.Done {
		return "pass"
	}
	for _, n := range blocked {
		if (st.Key == "ingest_probe" && n.Need == "probe_evidence") || (st.Key == "ingest_timing" && n.Need == "timing_trace") {
			return "wait"
		}
	}
	return "open"
}

func stageLines(r *seam.Run, events map[string]string, alive bool, spin string, technical bool) string {
	var b strings.Builder
	for i, st := range r.Stages {
		state := stageState(st, events, alive, r.Blocked)
		name := word(plainStage, st.Key, technical)
		if technical {
			name = st.Label
		}
		detail := ""
		switch state {
		case "pass":
			detail = fmt.Sprintf("%d artifacts", len(st.Artifacts))
			if technical {
				detail = strings.Join(st.Artifacts, " ")
			}
		case "run":
			detail = "running " + spin
		case "fail":
			detail = "failed; see the output below"
		case "wait":
			for _, n := range r.Blocked {
				if (st.Key == "ingest_probe" && n.Need == "probe_evidence") || (st.Key == "ingest_timing" && n.Need == "timing_trace") {
					detail = fmt.Sprintf("waiting for %s (%s)", word(plainNeed, n.Need, technical), n.Request)
				}
			}
		default:
			detail = "not run"
		}
		fmt.Fprintf(&b, "%s %d. %-26s %s", mark(state), i+1, name, stMuted.Render(detail))
		if technical {
			b.WriteString("  " + stMuted.Render(st.Note))
		}
		b.WriteString("\n")
	}
	return b.String()
}

// RunView: the runs folder, then the opened run's stages with live status, then the tail of its output.
func RunView(s RunScreen, technical bool, width int) string {
	var out []string
	switch {
	case s.Runs == nil:
		out = append(out, box("Runs", stMuted.Render("Reading the runs folder…"), width, true))
	case len(s.Runs.Runs) == 0:
		out = append(out, box("Runs", fmt.Sprintf("No runs yet. Press s to start one for the model on screen 1.\n%s", stMuted.Render(s.Runs.Root)), width, true))
	default:
		rows := [][]string{{"RUN", "LAST STAGE", "STATUS", "MEASURED"}}
		if technical {
			rows[0] = append(rows[0], "MODEL", "CHIP", "WORKLOAD")
		}
		for _, r := range s.Runs.Runs {
			last := "-"
			if r.LatestStage != nil {
				last = word(plainStage, *r.LatestStage, technical)
			}
			row := []string{r.ID, last, status(r.Status, technical), measuredMarks(r.Measured)}
			if technical {
				row = append(row, r.ModelID, r.TargetID, r.Workload)
			}
			rows = append(rows, row)
		}
		out = append(out, box("Runs", table(rows, s.Cursor), width, false))
	}
	if s.Run == nil {
		out = append(out, box("Stages", stMuted.Render("Open a run with enter, or press s to start one."), width, true))
		return strings.Join(out, "\n")
	}
	r := s.Run
	alive := s.Job != nil && s.Job.Alive
	events := seam.StageEvents(s.Tail)
	var b strings.Builder
	fmt.Fprintf(&b, "%s on %s (%s) · %s\n", r.ModelID, r.TargetID, r.Workload, status(r.Status, technical))
	b.WriteString(stageLines(r, events, alive, s.Spin, technical))
	fmt.Fprintf(&b, "%s %s\n", stInfo.Render("next"), r.NextStep)
	if r.Report != nil {
		fmt.Fprintf(&b, "%s\n", stMuted.Render("press 4 for the results, o to open "+*r.Report))
	}
	out = append(out, box(r.ID, b.String(), width, false))

	var o strings.Builder
	switch {
	case s.Job == nil:
		o.WriteString(stMuted.Render("No process was started for this run from this machine.") + "\n")
	default:
		running := mark("pass") + " finished"
		if alive {
			running = mark("run") + " running " + s.Spin
		}
		fmt.Fprintf(&o, "%s   %s\n", running, stMuted.Render(fmt.Sprintf("pid %d · started %s", s.Job.PID, s.Job.StartedAt)))
		if technical {
			fmt.Fprintf(&o, "%s %s\n%s %s\n", stMuted.Render("log "), s.Job.LogPath, stMuted.Render("argv"), strings.Join(s.Job.Argv, " "))
		}
		tail := s.Tail
		if len(tail) > 12 && !technical {
			tail = tail[len(tail)-12:]
		}
		for _, line := range tail {
			o.WriteString(line + "\n")
		}
	}
	out = append(out, box("Output", o.String(), width, !alive))
	return strings.Join(out, "\n")
}

// --- Results -------------------------------------------------------------------------------------------------

func pctBar(v *float64, width int) string {
	if v == nil {
		return strings.Repeat(" ", width+5)
	}
	return bar(*v/100, width) + fmt.Sprintf(" %3.0f%%", *v)
}

// ResultsView: what won per role and its evidence, and the measured timing against the ceiling as bars.
func ResultsView(r *seam.Run, technical bool, width int) string {
	if r == nil {
		return box("Results", "No run opened. Press 3, pick a run and press enter.", width, true)
	}
	res := r.Results
	var h strings.Builder
	switch {
	case res.Timing.TokS != nil && res.Ceiling.TokS != nil && *res.Ceiling.TokS > 0:
		ratio := *res.Timing.TokS / *res.Ceiling.TokS
		fmt.Fprintf(&h, "%s %s measured. The limit is %.1f.\n", stAccent.Render(glyphBolt),
			stHeader.Render(fmt.Sprintf("%.1f tokens per second", *res.Timing.TokS)), *res.Ceiling.TokS)
		fmt.Fprintf(&h, "%s %.0f%% of the limit\n", bar(ratio, 20), ratio*100)
	case res.Ceiling.TokS != nil:
		fmt.Fprintf(&h, "%s %.1f tokens per second at best; nothing measured yet.\n", stAccent.Render(glyphBolt), *res.Ceiling.TokS)
	default:
		fmt.Fprintf(&h, "%s\n", stMuted.Render("No speed limit for this chip: "+res.Ceiling.Reason))
	}
	if technical && res.Timing.Context != nil {
		fmt.Fprintf(&h, "%s context %d · step %s µs · ceiling floor %s ms · peak %s GB/s\n", stMuted.Render("timing"), *res.Timing.Context,
			num(res.Timing.TotalUs), num(res.Ceiling.FloorMs), num(res.Ceiling.PeakBandwidthGBs))
	}
	if res.Timing.DominantBucket != nil {
		fmt.Fprintf(&h, "%s %s\n", stWarn.Render(glyphWarn+" where the time goes:"), word(plainBucket, *res.Timing.DominantBucket, technical))
	}
	if len(res.Blocked) > 0 {
		names := []string{}
		for _, n := range res.Blocked {
			names = append(names, word(plainNeed, n.Need, technical)+" ("+n.Request+")")
		}
		fmt.Fprintf(&h, "%s %s\n", stInfo.Render(glyphWait+" still needed:"), strings.Join(names, "; "))
	}
	if technical {
		for _, a := range res.Timing.NextActions {
			fmt.Fprintf(&h, "%s %s\n", stMuted.Render("next"), a)
		}
	}
	if res.Report != nil {
		h.WriteString(stMuted.Render("press o to open " + *res.Report))
	}
	out := []string{box("Results: "+r.ID, h.String(), width, false)}

	rows := [][]string{{" ", "ROLE", "QUANT", "STATUS", "ROUTE"}}
	if technical {
		rows[0] = append(rows[0], "SHAPE", "CANDIDATES", "EVIDENCE")
	}
	for _, rt := range res.Routes {
		route := stMuted.Render("none yet")
		if rt.SelectedRoute != nil {
			route = *rt.SelectedRoute
		}
		row := []string{mark(routeMark[rt.Status]), word(plainRole, rt.Role, technical), rt.Quant, word(plainRoute, rt.Status, technical), route}
		if technical {
			evidence := stMuted.Render("none")
			if len(rt.EvidenceRefs) > 0 {
				evidence = strings.Join(rt.EvidenceRefs, ", ")
			}
			row = append(row, shapeStr(rt.Shape), stMuted.Render(strings.Join(rt.Candidates, ", ")), evidence)
		}
		rows = append(rows, row)
	}
	if len(res.Routes) == 0 {
		out = append(out, box("What won per role", stMuted.Render("No route policy yet: the run has not been planned."), width, true))
	} else {
		out = append(out, box("What won per role", table(rows, -1), width, false))
	}

	if res.Timing.Status != "classified" {
		out = append(out, box("Timing against the speed limit", stMuted.Render("No timing trace ingested. Every speed here is a prediction, not a measurement."), width, true))
	} else {
		header := []string{" ", "KERNEL", "µs", "OF STEP", "OF PEAK", "VERDICT"}
		if technical {
			header = append(header, "KIND", "LOSS µs")
		}
		rows := [][]string{header}
		for _, k := range res.Timing.Kernels {
			m := "pass"
			switch k.Bucket {
			case "at_peak":
			case "gemv_codegen_capped", "latency_bound":
				m = "crossed"
			default:
				m = "open"
			}
			row := []string{mark(m), k.Name, us(k.Us), pctBar(k.PctStep, 8), pctBar(k.PhysUtilPct, 8), word(plainBucket, k.Bucket, technical)}
			if technical {
				row = append(row, stMuted.Render(k.Kind), us(k.LossUs))
			}
			rows = append(rows, row)
		}
		out = append(out, box("Timing against the speed limit", table(rows, -1), width, false))
	}

	if len(res.Regimes) > 0 {
		rows := [][]string{{"ROLE", "QUANT", "WHAT THE TEST SAW", "BOTTLENECK", "NEXT"}}
		for _, g := range res.Regimes {
			rows = append(rows, []string{word(plainRole, g.Role, technical), g.Quant, word(plainRegime, g.Classification, technical),
				stMuted.Render(g.VisibleBottleneck), stMuted.Render(g.NextAction)})
		}
		out = append(out, box("Building blocks", table(rows, -1), width, false))
	} else {
		out = append(out, box("Building blocks", stMuted.Render("No building-block tests ingested."), width, true))
	}
	return strings.Join(out, "\n")
}
