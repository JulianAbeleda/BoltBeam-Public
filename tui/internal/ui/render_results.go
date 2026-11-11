package ui

// The results page: the headline, the loss, the tie-out, where the time goes, and comparisons.

import (
	"fmt"
	"slices"
	"strings"

	"charm.land/lipgloss/v2"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

// lossTitle is step 5's heading: the provider and how its roles were timed.
func lossTitle(provider string, c *seam.Capture) string {
	how := "roles not timed"
	if c != nil && c.Method != nil {
		how = word(captureWords, *c.Method)
	}
	return provider + " · " + how
}

// searched says whether any role carries a kernel search verdict (older results carry none).
func searched(roles []seam.RoleLoss) bool {
	for _, r := range roles {
		if r.Verdict != nil {
			return true
		}
	}
	return false
}

// files is the file names a list of pointers names, once each, in order.
func files(ptrs []seam.Pointer) string {
	seen, out := map[string]bool{}, []string{}
	for _, p := range ptrs {
		if !seen[p.File] {
			seen[p.File] = true
			out = append(out, p.File)
		}
	}
	return strings.Join(out, ", ")
}

// findingsBody is the verdicts read off the run's facts, one line each with its lever, and the files behind the
// per-role table. The report links each pointer; the screen names the files only.
func findingsBody(roles []seam.RoleLoss, findings []seam.Finding) string {
	var b strings.Builder
	all := []seam.Pointer{}
	for _, r := range roles {
		all = append(all, r.Evidence...)
	}
	if len(all) > 0 {
		b.WriteString(stMuted.Render("Evidence: "+files(all)) + "\n")
	}
	for _, f := range findings {
		fmt.Fprintf(&b, "%s %s · %.2f ms\n%s\n", stInfo.Render("next"), f.What, f.Ms, stMuted.Render(f.Do+" Evidence: "+files(f.Evidence)))
	}
	return b.String()
}

// searchBody is Run's kernel search under the per-role table: why it was skipped, or the next step it found.
func searchBody(s *seam.Search) string {
	if s == nil {
		return ""
	}
	switch {
	case s.Status == "skipped":
		return stMuted.Render("Kernel search: skipped. "+deref(s.Reason)) + "\n"
	case s.Next != nil:
		return fmt.Sprintf("%s %s.\n%s\n", stInfo.Render("next"), s.Next.What, stMuted.Render(s.Next.Do))
	}
	return ""
}

// tieOutBody is the measured token line by line against the limit. The last line is the difference.
func tieOutBody(t *seam.TieOut) string {
	if t == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString(stHeader.Render(fmt.Sprintf("Tie-out, ms per token at context %g", t.Context)) + "\n")
	if t.Refused != nil {
		b.WriteString(stWarn.Render(glyphWarn+" Not tied out. ") + *t.Refused + "\n")
		return b.String()
	}
	if t.TokenMs == nil {
		return b.String()
	}
	if t.Estimate != nil {
		return b.String() + isolatedTieOut(t)
	}
	rows := [][]string{}
	for i, l := range t.Lines {
		sign := "+ "
		if i == 0 {
			sign = "  "
		}
		rows = append(rows, []string{sign + l.Label, fmt.Sprintf("%.3f", l.Ms), l.How})
	}
	rows = append(rows, []string{"= measured token", fmt.Sprintf("%.3f", *t.TokenMs), deref(t.TokenSource)})
	b.WriteString(table(rows))
	for _, l := range t.Lines {
		if len(l.Parts) > 0 {
			parts := []string{}
			for _, p := range l.Parts {
				parts = append(parts, fmt.Sprintf("%s %.3f", p.Kind, p.Ms))
			}
			b.WriteString(stMuted.Render("  other kernels: "+strings.Join(parts, ", ")) + "\n")
		}
		if len(l.Kernels) > 0 {
			names := []string{}
			for _, k := range l.Kernels {
				us := ""
				if k.UsPerCall != nil {
					us = fmt.Sprintf(", %.1f µs each", *k.UsPerCall)
				}
				names = append(names, fmt.Sprintf("%s (%.0f per token%s)", k.Kernel, k.CallsPerToken, us))
			}
			b.WriteString(stMuted.Render("  weight kernels no role took: "+strings.Join(names, "; ")) + "\n")
		}
	}
	if t.BusyMs != nil && t.Isolated {
		fmt.Fprintf(&b, "The weight kernels, timed alone, sum to %.3f ms of the %.3f ms token; the rest is attention, norms, the KV read and idle time, not timed.\n", *t.BusyMs, *t.TokenMs)
	} else if t.BusyMs != nil {
		fmt.Fprintf(&b, "All kernels sum to %.3f ms against the real token of %.3f ms; the gap is their difference.\n", *t.BusyMs, *t.TokenMs)
	}
	if t.ShowBoth {
		fmt.Fprintf(&b, "The limit is %.3f ms at context 1 and %.3f ms at context %g.\n", t.LimitMsCtx1, t.LimitMs, t.Context)
	}
	if t.UntracedMs != nil && (t.EndToEnd != nil || t.TokenSource != nil && strings.HasPrefix(*t.TokenSource, "the captured run")) {
		b.WriteString(tokenEndToEnd(*t) + "\n")
	}
	if t.Missing != nil {
		b.WriteString(stMuted.Render("Missing: "+*t.Missing) + "\n")
	}
	return b.String()
}

// whereBody is "Where the token goes": what each part costs the token, worst first, read from the seam. On a
// narrow page the peak column (% OF PEAK, MFU) goes first, then SHARE, so no row is cut.
// auditBody is the prefill audit as Python wrote it: the pointers, then each entry's fields, worst first.
func auditBody(a *seam.Audit, width int) string {
	if a == nil || len(a.Entries) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n" + stHeader.Render(a.Title) + "\n" + stMuted.Render(a.Words) + "\n")
	for _, s := range a.Sources {
		b.WriteString(stMuted.Render("["+s.ID+"] "+s.File+": "+s.What) + "\n")
	}
	for i, e := range a.Entries {
		fmt.Fprintf(&b, "%d. %s\n", i+1, stHeader.Render(e.Name))
		for _, f := range e.Fields { // the label, then its words wrapped under it: a field is a sentence, not a cell
			b.WriteString("   " + stMuted.Render(f[0]) + "\n")
			b.WriteString(lipgloss.NewStyle().PaddingLeft(5).Width(width).Render(f[1]) + "\n")
		}
	}
	if a.Collapsed != nil {
		b.WriteString(stMuted.Render(*a.Collapsed) + "\n")
	}
	return b.String()
}

func whereBody(w *seam.Where, width int) string {
	if w == nil || len(w.Rows) == 0 {
		return ""
	}
	title := w.Title
	if w.Estimate {
		title += " (estimate)"
	}
	var b strings.Builder
	b.WriteString("\n" + stHeader.Render(title) + "\n")
	if w.Qualification != "" { // which rows stand and how: above the words, so it is on screen with the table's header
		b.WriteString(stWarn.Render(glyphWarn+" "+w.Qualification) + "\n")
	}
	b.WriteString(stMuted.Render(strings.Join(strings.Fields(w.Words), " ")) + "\n\n")
	last := strings.ToUpper(w.Columns[len(w.Columns)-1]) // the peak column goes first, then the share
	drops := [][]string{nil, {last}, {last, "SHARE"}, {last, "SHARE", "MS AT CHIP BEST"}, {last, "SHARE", "MS AT CHIP BEST", "MS ABOVE LIMIT", "MS LOST"}}
	for index, drop := range drops {
		if t := whereTable(w, drop, width); maxWidth(t) <= width || index == len(drops)-1 {
			b.WriteString(t)
			break
		}
	}
	for i, column := range w.Columns {
		if column == "+ Tok/S if roofline" && i < len(w.SumCells) && w.SumCells[i] != "" {
			fmt.Fprintf(&b, "Total at modeled roofline: %s tok/s (%s ms/token).\n", w.SumCells[i], w.SumCells[2])
		}
	}
	b.WriteString(whereEndToEnd(w.EndToEnd))
	return b.String()
}

// sessionBody is the session table (workflow/session.py) when the run carries one: a row per depth, then its lines.
func sessionBody(s *seam.Session) string {
	if s == nil || len(s.Rows) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n" + stHeader.Render(s.Title) + "\n")
	b.WriteString(stMuted.Render(s.Words) + "\n")
	head := []string{}
	for _, c := range s.Columns {
		head = append(head, strings.ToUpper(c))
	}
	rows := [][]string{head}
	for _, r := range s.Rows {
		rows = append(rows, r.Cells)
	}
	b.WriteString(table(rows))
	for _, line := range []string{s.Warm, s.MaxContext, s.Compaction} {
		if line != "" {
			b.WriteString(stMuted.Render(line) + "\n")
		}
	}
	return b.String()
}

// replayBody is the replay (workflow/replay.py) when the run carries one: the headline, a row per turn, then the
// captured turns' where tables (or their one line) and the deepest decode.
func replayBody(r *seam.Replay, width int) string {
	if r == nil || len(r.Rows) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n" + stHeader.Render(r.Title) + "\n")
	for _, l := range r.HeadlineLines {
		b.WriteString(l + "\n")
	}
	for _, line := range []string{r.EvidenceWords, r.NotCaptured} {
		if line != "" {
			b.WriteString(stMuted.Render(line) + "\n")
		}
	}
	b.WriteString(stMuted.Render(r.Words) + "\n")
	head := []string{}
	for _, c := range r.Columns {
		head = append(head, strings.ToUpper(c))
	}
	rows := [][]string{head}
	for _, row := range r.Rows {
		rows = append(rows, row.Cells)
	}
	b.WriteString(replayTable(append(rows, r.Total), width))
	if len(r.Unmodelled) > 0 {
		b.WriteString(stMuted.Render("Not in the ceiling: "+strings.Join(r.Unmodelled, "; ")+".") + "\n")
	}
	for _, w := range r.Wheres {
		if w.Status != "measured" && w.Status != "benchmark_only" {
			b.WriteString(stMuted.Render(strings.Join(strings.Fields(w.Words), " ")) + "\n")
			continue
		}
		b.WriteString(whereBody(&w.Where, width))
		b.WriteString(stMuted.Render(w.ServerLine) + "\n")
	}
	if d := r.DecodeWhere; d != nil {
		if d.Status != "measured" {
			b.WriteString(stMuted.Render(d.Words) + "\n")
		} else {
			b.WriteString("\n" + stHeader.Render(d.Title) + "\n" + stMuted.Render(d.Words) + "\n")
			head := []string{}
			for _, c := range d.Columns {
				head = append(head, strings.ToUpper(c))
			}
			b.WriteString(table(append([][]string{head}, d.Rows...)))
		}
	}
	return b.String()
}

// sessionHeadline is a session run's headline: the turn at the focus depth against the limit at that depth. The
// run's first timing row is the shallowest depth; set against the focus depth's limit it overstated the share
// (4,094 against 13,831 tok/s read 30%; the 16,384-depth turn is 3,580, 26%). The other depths are in the table.
func sessionHeadline(res seam.Results) string {
	s := res.Loss.Session
	if s == nil || s.Focus == nil || len(res.Loss.Runtimes) == 0 || res.Ceiling.TokS == nil || *res.Ceiling.TokS <= 0 {
		return ""
	}
	r := res.Loss.Runtimes[0]
	ratio := r.TokS / *res.Ceiling.TokS
	return fmt.Sprintf("%s %s measured for a turn at depth %s. The limit at that depth is %.1f.\n%s %.0f%% of the limit\n",
		stAccent.Render(glyphBolt), stHeader.Render(fmt.Sprintf("%.1f prompt tokens per second", r.TokS)), count(s.Focus),
		*res.Ceiling.TokS, bar(ratio, 20), ratio*100)
}

// whereEndToEnd is the capture's qualification as one status line: the words Python wrote (workflow/tie_out.end_to_end),
// which disclose the signed overhead against the capture-overhead ceiling and the repeat-noise comparison.
func whereEndToEnd(e *seam.EndToEnd) string {
	if e == nil || e.Words == "" || e.CapturedMs == nil || e.UntracedMs == nil {
		return ""
	}
	line := "End to end: " + e.Words + "."
	switch {
	case e.Confirms == nil:
		return stMuted.Render(line) + "\n"
	case *e.Confirms:
		return stOK.Render(line) + "\n"
	}
	return stWarn.Render(glyphWarn+" "+line) + "\n"
}

// tokenEndToEnd is a decode capture's token against the untraced reference, with the repeat check and, when the two
// were taken at different contexts, that fact: they are then not one setup.
func tokenEndToEnd(t seam.TieOut) string {
	pct := 100 * (*t.TokenMs - *t.UntracedMs) / *t.UntracedMs
	line := fmt.Sprintf("End to end: the captured token is %.3f ms, the untraced reference %.3f ms (%+.1f%%).", *t.TokenMs, *t.UntracedMs, pct)
	if t.UntracedContext != nil && *t.UntracedContext != t.Context {
		line += fmt.Sprintf(" The untraced time is at context %g and the capture at context %g, so they are not one setup.", *t.UntracedContext, t.Context)
	}
	if t.TimingMatch != nil && !t.TimingMatch.Matches {
		return line + "\n" + stMuted.Render("Repeat check: not checked: "+t.TimingMatch.Words+".")
	}
	if t.EndToEnd != nil {
		return strings.TrimSuffix(whereEndToEnd(t.EndToEnd), "\n")
	}
	band := deref(t.Band)
	if band == "" || strings.HasPrefix(band, "not checked") {
		return line + "\n" + stMuted.Render("Repeat check: "+notChecked)
	}
	return line + "\n" + stMuted.Render("Repeat check: "+band+".")
}

// runEngineBody names the engine build against the pin table and the toolchain the run recorded
// (docs/engine-upgrade-audit.md). A run from before either was recorded says so; nothing is filled from the pins.
// The toolchain is whatever keys the run's vendor recorded, in their recorded order.
func runEngineBody(e *seam.RunEngine) string {
	if e == nil {
		return ""
	}
	var b strings.Builder
	if len(e.SettingsWords) > 0 { // a requested setting changes what was measured, so it leads
		b.WriteString(stMuted.Render("Engine settings: "+strings.Join(e.SettingsWords, "; ")+".") + "\n")
	}
	if len(e.SpeculationWords) > 0 { // a speculative run's limit is at the acceptance it measured: said beside the request
		b.WriteString(stMuted.Render("Speculation measured: "+strings.Join(e.SpeculationWords, "; ")+".") + "\n")
	}
	switch {
	case e.Build == nil:
		b.WriteString(stMuted.Render("Engine build: not recorded.") + "\n")
	default: // the build that ran beside the pin, both as facts: a build other than the pin is not a fault
		b.WriteString(stMuted.Render("Engine build: "+e.Build.Words+".") + "\n")
	}
	for _, line := range e.BuildLines { // latest at run time, and the comparison with the previous build (its file)
		b.WriteString(stMuted.Render(line) + "\n")
	}
	if e.Toolchain == nil {
		b.WriteString(stMuted.Render("Toolchain: not recorded.") + "\n")
		return b.String()
	}
	parts, missing := []string{}, []string{}
	for _, k := range toolchainKeys(e.Toolchain) {
		if v := e.Toolchain.Versions[k]; v != "unknown" {
			parts = append(parts, k+" "+v)
		} else {
			missing = append(missing, k)
		}
	}
	for k, v := range e.Toolchain.Runtime { // torch's own runtime, one: cuda or hip
		if v != "unknown" {
			parts = append(parts, "torch's "+strings.ToUpper(k)+" "+v)
		}
	}
	when := e.Toolchain.RecordedAt
	if len(when) > 16 {
		when = strings.Replace(when[:16], "T", " ", 1)
	}
	line := "Toolchain (" + when + " UTC): " + strings.Join(parts, ", ")
	switch {
	case len(parts) == 0:
		line = "Toolchain (" + when + " UTC): versions not recorded for " + strings.Join(missing, ", ")
	case len(missing) > 0:
		line += "; versions not recorded: " + strings.Join(missing, ", ")
	}
	b.WriteString(stMuted.Render(line+".") + "\n")
	return b.String()
}

// roleChecks says how many per-role rows have a measured repeat band for this setup; without one a row is shown,
// and its verdict is "not checked".
func roleChecks(roles []seam.RoleLoss) string {
	checked, seen := 0, 0
	for _, r := range roles {
		if r.CheckStatus == nil {
			continue
		}
		seen++
		if *r.CheckStatus == "checked" {
			checked++
		}
	}
	switch {
	case seen == 0:
		return ""
	case checked == 0:
		return stMuted.Render("Per-role repeat check: not checked: no repeat band matches this setup.") + "\n"
	}
	return stMuted.Render(fmt.Sprintf("Per-role repeat check: %d of %d roles checked against the measured repeat band of this setup.", checked, seen)) + "\n"
}

// pairingBody says which roles the capture could not pair, with the solver's reason, and whether the pairing as a
// whole is unsure (collectors/pairing.py ambiguous).
func pairingBody(l seam.Loss) string {
	var b strings.Builder
	if a := l.Ambiguous; a != nil {
		what := map[string]string{"tie": "Pairing is not certain, several pairings fit equally: ",
			"search_limit": "Pairing search stopped before it finished: ", "no_fit": "Pairing found no fit: "}[a.Kind]
		if what == "" {
			what = "Pairing is not certain: "
		}
		b.WriteString(stWarn.Render(glyphWarn+" "+what+a.Reason+". Read the per-role rows as unsure.") + "\n")
	}
	for _, u := range l.UnpairedRoles {
		why := ""
		if u.Reason != nil {
			why = ": " + *u.Reason
		}
		b.WriteString(stMuted.Render("Unpaired, so in not attributed: "+roleWord(u.Role)+" "+u.Quant+why+".") + "\n")
	}
	return b.String()
}

// isolatedTieOut is the tie-out of a run whose kernels were timed alone: only what is measured, the ideal at
// context 1 and at the attended context, the token, and their difference. The split is one estimate line.
func isolatedTieOut(t *seam.TieOut) string {
	var b strings.Builder
	rows := [][]string{{"  limit at context 1 (ideal)", fmt.Sprintf("%.3f", t.LimitMsCtx1), "derived"}}
	for i, l := range t.Lines {
		sign := "+ "
		if i == 0 {
			sign = "  "
		}
		rows = append(rows, []string{sign + l.Label, fmt.Sprintf("%.3f", l.Ms), l.How})
	}
	rows = append(rows, []string{"= measured token", fmt.Sprintf("%.3f", *t.TokenMs), deref(t.TokenSource)})
	b.WriteString(table(rows))
	e := t.Estimate
	up, least := "", ""
	if e.Scaled {
		up, least = "up to ", "at least "
	}
	fmt.Fprintf(&b, "Estimated split (isolated): weight kernels %s%.1f ms, other and gaps %s%.1f ms.\n", up, e.WeightMs, least, e.OtherMs)
	if e.OtherHow != "" {
		fmt.Fprintf(&b, "  (%s)\n", e.OtherHow)
	}
	return b.String()
}

// estimateHow is how the estimate's numbers were made from the isolated times (workflow/screen.py estimate_how,
// the same words): the scale and its inputs, or "not scaled" with the sum that fits; the sum is the one less the
// dispatch floor when the rows carry one, said so.
func estimateHow(e *seam.Estimate) string {
	base, floor := e.IsolatedSumMs, ""
	if e.FloorUs != nil && e.IsolatedSumLessFloorMs != nil {
		base = *e.IsolatedSumLessFloorMs
		if e.FloorWords != nil {
			floor = " " + *e.FloorWords
		}
	}
	if e.Scaled {
		return fmt.Sprintf("scale %.3f = (%.1f - %.1f) / %.1f%s", e.Scale, e.TokenMs, e.OtherMs, base, floor)
	}
	return fmt.Sprintf("not scaled: %.1f ms alone%s fits the %.1f ms token", base, floor, e.TokenMs)
}

// lossBody is the end result for the run's provider: its speed against the limit and where it loses time.
// Another provider's numbers for the same run are shown after it, each labelled with its provider.
func lossBody(l seam.Loss, width int) string { return lossBodyAt(l, 1, width) }

// lossBodyAt is lossBody for a run that also timed batch > 1: its heading names the batch.
func lossBodyAt(l seam.Loss, batch, width int) string {
	if l.Replay != nil && (l.Replay.Forced || l.Status != "modeled" || l.LimitMs == nil) {
		return replayBody(l.Replay, width)
	}
	if l.Status != "modeled" || l.LimitMs == nil {
		return ""
	}
	var b strings.Builder
	provider := l.Provider
	if provider == "" {
		provider = "llama.cpp"
	}
	title := "Measured with " + lossTitle(provider, l.Capture)
	if l.Replay != nil {
		title = "Measured with " + provider + " · server metrics and client clocks"
	} else if l.Capture == nil && l.Per != nil { // prefill: kernels from the capture
		title = "Measured with " + provider + " · kernels from the capture"
	}
	if batch > 1 {
		title += fmt.Sprintf(" · batch %d", batch)
	}
	b.WriteString(stHeader.Render(title) + "\n")
	tie := l.TieOut
	if tie != nil && tie.Missing != nil && l.Missing != nil { // said once, by the per-role line below
		t := *tie
		t.Missing = nil
		tie = &t
	}
	b.WriteString(tieOutBody(tie))
	per, at := "per token", " at context 1"
	if l.LimitContext != nil {
		at = fmt.Sprintf(" at context %.0f", *l.LimitContext)
	}
	if l.Per != nil { // a prefill run: its times are per prefill, its limit the sum of its kernels' ceilings
		per, at = *l.Per, ""
	}
	fmt.Fprintf(&b, "Modeled roofline: %.1f ms %s%s.\n", *l.LimitMs, per, at)
	if l.LimitBasis != nil {
		b.WriteString(stMuted.Render("Limit: "+*l.LimitBasis+".") + "\n")
	}
	for _, r := range l.Runtimes {
		what := fmt.Sprintf("%.1f tok/s · %.1f ms %s", r.TokS, r.Ms, per)
		if r.Isolated { // a sum of kernels timed alone is not a token: it has no "lost"
			fmt.Fprintf(&b, "%s %s per role: %.1f ms, the kernels timed alone and summed.\n", mark("open"), r.Provider, r.Ms)
			b.WriteString("  " + stMuted.Render("Not a token: attention, norms and gaps are not in it.") + "\n")
			continue
		}
		if r.PerRole {
			what = fmt.Sprintf("%.1f ms of GPU time %s", r.Ms, per)
		}
		label := r.Provider
		if r.PerRole {
			label += " per role"
		}
		fmt.Fprintf(&b, "%s %s: %s · %s above roofline.\n", mark("open"), label, what, stHeader.Render(fmt.Sprintf("%.1f ms", r.LostMs)))
		if !r.PerRole {
			b.WriteString("  " + stMuted.Render(r.Note) + "\n")
		}
	}
	b.WriteString(sessionBody(l.Session))
	b.WriteString(replayBody(l.Replay, width))
	b.WriteString(whereBody(l.WhereTokenGoes, width))
	b.WriteString(whereBody(l.CaptureComparison, width))
	b.WriteString(auditBody(l.PrefillAudit, width))
	if l.Refused != nil {
		b.WriteString(stWarn.Render(glyphWarn+" Per role: not shown. ") + *l.Refused + "\n")
		return b.String()
	}
	if l.Missing != nil {
		b.WriteString(stMuted.Render("Per role: "+*l.Missing) + "\n")
		b.WriteString(findingsBody(nil, l.Findings)) // what the run's own facts say (a failed graph replay) stands without roles
		return b.String() + othersBody(l.Others, provider, width)
	}
	if len(l.Roles) == 0 {
		return b.String() + othersBody(l.Others, provider, width)
	}
	shown := provider
	if l.RolesProvider != nil {
		shown = *l.RolesProvider
	}
	if shown != provider && l.ProviderMissing != nil {
		b.WriteString(stMuted.Render("Per role for "+provider+": "+*l.ProviderMissing) + "\n")
	}
	if e := l.Estimate; e != nil {
		b.WriteString("\n" + stHeader.Render("Per role, estimated from isolated kernel times") + "\n" + e.Method + "\n")
		b.WriteString(stMuted.Render(fmt.Sprintf("EST. columns %s: %s.", e.Label, estimateHow(e))) + "\n")
		if e.ColumnsWords != nil {
			b.WriteString(stMuted.Render(*e.ColumnsWords+".") + "\n")
		}
		b.WriteString(isoRoleTable(l.Roles, e.ColumnsWords != nil, width))
	} else {
		b.WriteString("\nWhere " + shown + " loses time, " + deref(l.Source) + ":\n" + roleTable(l.Roles, l.NotAttributedMs, width))
	}
	if l.RoleRule != nil {
		b.WriteString(stMuted.Render("Why: "+*l.RoleRule) + "\n")
	}
	b.WriteString(searchBody(l.Search))
	b.WriteString(findingsBody(l.Roles, l.Findings))
	b.WriteString(roleChecks(l.Roles))
	b.WriteString(pairingBody(l))
	for _, n := range l.NotTimed {
		b.WriteString(stMuted.Render(fmt.Sprintf("%s %s: not timed: %s; left out of the floor rule and the sums.", roleWord(n.Role), n.Quant, deref(n.Reason))) + "\n")
	}
	for _, n := range l.CeilingRefused {
		b.WriteString(stWarn.Render(fmt.Sprintf("%s %s: %.3f ms against a %.3f ms floor: %s.", roleWord(n.Role), n.Quant, n.ActualMs, n.IdealMs, n.Reason)) + "\n")
	}
	if len(l.CeilingRefused) > 0 {
		b.WriteString(stWarn.Render(fmt.Sprintf("Totals exclude %d row(s) whose ceiling is refused. The other rows keep their ceilings.", len(l.CeilingRefused))) + "\n")
	}
	if l.RoleSourceWords != nil {
		b.WriteString(stMuted.Render("Per-role source: "+*l.RoleSourceWords) + "\n")
	}
	if cc := l.CrossCheck; cc != nil && (len(cc.Rows) > 0 || len(cc.NotMeasured) > 0) {
		b.WriteString("\n" + stHeader.Render("Cross-check: "+cc.Words) + "\n")
		if cc.ColumnsWords != nil {
			b.WriteString(stMuted.Render(*cc.ColumnsWords+".") + "\n")
		}
		b.WriteString(rateTable(cc.Rows))
		for _, r := range cc.NotMeasured {
			fmt.Fprintf(&b, "%s %s: not measured: %s\n", r.Role, r.Quant, r.Reason)
		}
	}
	b.WriteString(othersBody(l.Others, shown, width))
	return b.String()
}

// othersBody is the other provider's numbers, each labelled with its provider and, for another run, its run.
func othersBody(others []seam.OtherLoss, shown string, width int) string {
	var b strings.Builder
	for _, o := range others {
		if o.Provider == shown && o.Run == nil {
			continue
		}
		label := o.Provider
		if o.Run != nil {
			label += " (run " + shortRun(*o.Run) + ")"
		}
		speed := "not measured" // never 0.0 for a speed that was not taken
		if o.TokS != nil && *o.TokS > 0 {
			speed = fmt.Sprintf("%.1f tokens per second", *o.TokS)
		} else if o.Missing != nil {
			speed = *o.Missing
		}
		title := lossTitle(label, &o.Capture)
		if (o.TokS == nil || *o.TokS <= 0) && len(o.Roles) == 0 {
			title = label // the reason says why; "roles not timed" would only repeat it
		}
		fmt.Fprintf(&b, "\nBeside it, %s: %s.\n", title, speed)
		if len(o.Roles) > 0 {
			b.WriteString(roleTable(o.Roles, o.NotAttributedMs, width))
		}
	}
	return b.String()
}

// compareNote says where every role and compare time comes from. Python names the runtime (compare-ready
// "runtime", from tinygrad_role_time.runtime_name); the screen never names a backend itself.
func compareNote(f Facts) string {
	runtime := "tinygrad's runtime"
	if f.Ready != nil && f.Ready.Runtime != nil {
		runtime = *f.Ready.Runtime
	}
	return "Times are from " + runtime + ", not llama.cpp."
}

// compareBody is step 5's kernel choice per role: progress while the compare job runs, what is missing when
// this machine cannot compare, and the per-role table once any role has been compared.
func compareBody(f Facts) string {
	res := f.Run.Results
	if res.Loss.Replay != nil {
		return ""
	}
	var b strings.Builder
	if f.comparing() {
		p := seam.ReadCompare(f.CTail)
		rows := [][]string{{" ", "ROLE", "QUANT", "COMPARING"}}
		for _, rt := range res.Routes {
			key, state, m := rt.Role+" "+rt.Quant, "waiting", "open"
			if status, ok := p.Done[key]; ok {
				state, m = word(plainRoute, status), routeMark[status]
			} else if key == p.Now {
				state, m = compareDoing[p.Doing]+"…", "run"
			}
			rows = append(rows, []string{mark(m), roleWord(rt.Role), rt.Quant, state})
		}
		return "\n" + table(rows) + stMuted.Render(compareNote(f)) + "\n"
	}
	compared := false
	for _, rt := range res.Routes {
		compared = compared || rt.Status != "unmeasured"
	}
	if r := f.Ready; r != nil && !compared {
		switch {
		case !r.Ready:
			fmt.Fprintf(&b, "\n%s %s\n", stInfo.Render(glyphWait+" cannot time or compare kernels here:"), deref(r.Message))
			if r.Fix != nil {
				fmt.Fprintf(&b, "Fix: %s\n", stAccent.Render(*r.Fix))
			}
		case !r.Applies:
			fmt.Fprintf(&b, "\n%s\n", stMuted.Render(deref(r.CompareMessage)))
		}
	}
	if len(res.Routes) > 0 && !compared {
		b.WriteString("\n" + stMuted.Render(noKernelChoice) + "\n")
		return b.String()
	}
	if !compared {
		return b.String()
	}
	rows := [][]string{{" ", "ROLE", "QUANT", "KERNEL CHOICE", "KERNEL ALONE", "WHOLE MODEL tok/s"}}
	reasons := []string{}
	for _, rt := range res.Routes {
		choice, kernel, whole := word(plainRoute, rt.Status), "-", "-"
		if c := rt.Compare; c != nil {
			if c.Plan != nil {
				choice += " · " + *c.Plan
			}
			if k := c.Kernel; k != nil {
				kernel = fmt.Sprintf("plan %.0f µs alone", k.PlanUs)
				if k.ModelUsPerCall != nil && k.FasterThanModel != nil {
					word := "slower"
					if *k.FasterThanModel {
						word = "faster"
					}
					kernel = fmt.Sprintf("model %.0f µs to plan %.0f µs (%s)", *k.ModelUsPerCall, k.PlanUs, word)
				}
			}
			if ab := c.AB; ab != nil && ab.BaselineTokS != nil && ab.CandidateTokS != nil && ab.DeltaPct != nil {
				whole = fmt.Sprintf("%.2f to %.2f (%+.1f%%)", *ab.BaselineTokS, *ab.CandidateTokS, *ab.DeltaPct)
			}
			if c.Reason != nil {
				reasons = append(reasons, roleWord(rt.Role)+" "+rt.Quant+": "+*c.Reason)
			}
		}
		rows = append(rows, []string{mark(routeMark[rt.Status]), roleWord(rt.Role), rt.Quant, choice, kernel, whole})
	}
	b.WriteString("\n" + table(rows))
	b.WriteString(stMuted.Render("Kernel: the role's own kernel per call in the model, to the plan alone. Whole model: default to plan.") + "\n")
	b.WriteString(stMuted.Render(compareNote(f)) + "\n")
	for _, r := range reasons {
		b.WriteString(stMuted.Render(r) + "\n")
	}
	return b.String()
}

// hereLoss says, in place of "pick Time each role", why this machine cannot time the run's provider per role.
// The run's results never hold machine facts; this machine's provider row does.
func hereLoss(f Facts, l seam.Loss) seam.Loss {
	p := f.provider(l.Provider)
	if p == nil || p.Capture.Method != nil {
		return l
	}
	why := "Not possible on " + here() + ": " + deref(p.Capture.Reason) + "."
	generic := func(s *string) bool { return s != nil && strings.Contains(*s, "Run with") } // the run recorded no reason of its own
	if generic(l.Missing) {
		l.Missing = &why
	}
	if generic(l.ProviderMissing) {
		l.ProviderMissing = &why
	}
	if l.TieOut != nil && generic(l.TieOut.Missing) {
		t := *l.TieOut
		t.Missing = &why
		l.TieOut = &t
	}
	return l
}

func resultBody(f Facts, width int) string {
	if f.Run == nil {
		return stMuted.Render("No run yet. Step 4 makes one.")
	}
	res := f.Run.Results
	var b strings.Builder
	switch {
	case sessionHeadline(res) != "":
		b.WriteString(sessionHeadline(res))
	case res.Timing.TokS != nil && res.Ceiling.TokS != nil && *res.Ceiling.TokS > 0:
		ratio := *res.Timing.TokS / *res.Ceiling.TokS
		fmt.Fprintf(&b, "%s %s measured · roofline %.1f tok/s\n", stAccent.Render(glyphBolt),
			stHeader.Render(fmt.Sprintf("%.1f tok/s", *res.Timing.TokS)), *res.Ceiling.TokS)
		fmt.Fprintf(&b, "%s %.0f%% of the limit\n", bar(ratio, 20), ratio*100)
		if s := res.Loss.Step; s != nil && s.Context != nil && s.RequestedContext != nil && *s.Context != float64(*s.RequestedContext) {
			fmt.Fprintf(&b, "Requested depth %d; measured mean attended context %g.\n", *s.RequestedContext, *s.Context)
		}
		for _, a := range res.Timing.Also { // one headline: the other contexts are listed, labelled, never mixed in
			context := "-"
			if a.Context != nil {
				context = fmt.Sprintf("%g", *a.Context)
			}
			fmt.Fprintf(&b, "%s\n", stMuted.Render(fmt.Sprintf("Context %s: %.1f tok/s (additional measurement)", context, a.TokS)))
		}
		if s := res.Loss.Step; s != nil && s.GraphFailed {
			why := ""
			if s.GraphError != nil {
				why = " (" + *s.GraphError + ")"
			}
			fmt.Fprintf(&b, "%s\n", stWarn.Render(glyphWarn+" Graph replay failed on this run"+why+": the token ran without graphs."))
		}
	case res.Loss.Replay != nil && len(res.Loss.Replay.Rows) > 0: // a replay's headline is per session, in its table
		note := "The session against its roofline is below."
		if res.Loss.Replay.Forced {
			note = "Captured session and timing qualification are below."
		}
		fmt.Fprintf(&b, "%s %s "+note+"\n", stAccent.Render(glyphBolt),
			stHeader.Render(fmt.Sprintf("Replay measured: %d turns.", len(res.Loss.Replay.Rows))))
	case res.Timing.TokS != nil: // measured with no limit to hold it against (a Mamba2 hybrid with no capture): the speed and why
		fmt.Fprintf(&b, "%s %s measured · no limit\n", stAccent.Render(glyphBolt), stHeader.Render(fmt.Sprintf("%.1f tok/s", *res.Timing.TokS)))
		if res.Ceiling.Reason != "" {
			b.WriteString(stMuted.Render("No limit: "+res.Ceiling.Reason) + "\n")
		}
		b.WriteString(pairingBody(res.Loss)) // the capture's pairing, said as on a run with a limit
	case res.Ceiling.TokS != nil:
		fmt.Fprintf(&b, "%s %.1f tokens per second at best. Nothing measured yet.\n", stAccent.Render(glyphBolt), *res.Ceiling.TokS)
	case res.Ceiling.Reason != "":
		fmt.Fprintf(&b, "%s\n", stMuted.Render("No speed limit for this chip: "+res.Ceiling.Reason))
	}
	batch := max(1, res.Loss.Batch)                                                     // the batch the figures are of, not the largest one timed (those are batchTable's)
	if s := measurementSentence(res.Measurement); s != "" && res.Loss.Estimate == nil { // the method line says it otherwise
		b.WriteString(stMuted.Render(s) + "\n")
	}
	b.WriteString(runEngineBody(res.Engine) + repeatsResult(res))
	b.WriteString(lossBodyAt(hereLoss(f, res.Loss), batch, width))
	b.WriteString(batchTable(res.Batches))
	inconclusive := res.Timing.DominantBucket != nil && *res.Timing.DominantBucket == "timing_inconclusive"
	if res.Timing.DominantBucket != nil && !(inconclusive && (res.Loss.WhereTokenGoes != nil || res.Loss.Replay != nil)) { // "not clear yet" under a where table is stale
		fmt.Fprintf(&b, "Where the time goes: %s\n", named(plainBucket, *res.Timing.DominantBucket))
	}
	for _, n := range res.Blocked {
		fmt.Fprintf(&b, "%s %s (%s)\n", stInfo.Render(glyphWait+" still needed:"), word(plainNeed, n.Need), n.Request)
	}
	b.WriteString(compareBody(f))
	if res.Timing.Status == "classified" && len(res.Timing.Kernels) > 0 {
		rows := [][]string{{" ", "KERNEL", "µs", "OF STEP", "OF PEAK"}}
		for _, k := range res.Timing.Kernels {
			m := "open"
			switch k.Bucket {
			case "at_peak":
				m = "pass"
			case "gemv_codegen_capped", "latency_bound":
				m = "crossed"
			}
			rows = append(rows, []string{mark(m), k.Name, us(k.Us), pctBar(k.PctStep, 6), pctBar(k.PhysUtilPct, 6)})
		}
		b.WriteString("\n" + table(rows))
	}
	clear := false
	for _, g := range res.Regimes {
		clear = clear || g.Classification != "inconclusive"
	}
	if p := res.Probe; p != nil && p.Status == "measured" && len(p.Rows) > 0 {
		// the probe's measured rates stand on their own: BoltBeam's reference kernel per role, labelled so
		fmt.Fprintf(&b, "\n%s\n%s", stHeader.Render(p.Label), rateTable(p.Rows))
		if p.DispatchFloorUs != nil {
			fmt.Fprintf(&b, "%s\n", stMuted.Render(fmt.Sprintf("Dispatch floor measured here: %.1f µs per launch.", *p.DispatchFloorUs)))
		}
	} else if len(res.Regimes) > 0 && !clear {
		fmt.Fprintf(&b, "\n%s\n", stMuted.Render(fmt.Sprintf("The building-block tests ran on %d roles. None could be classified:\n"+
			"this GPU does not report the counters the test needs.", len(res.Regimes))))
	}
	if clear {
		rows := [][]string{{"ROLE", "QUANT", "WHAT THE TEST SAW"}}
		for _, g := range res.Regimes {
			rows = append(rows, []string{roleWord(g.Role), g.Quant, named(plainRegime, g.Classification)})
		}
		b.WriteString("\n" + table(rows))
	}
	if strings.Contains(b.String(), "not checked") {
		b.WriteString("\n" + stMuted.Render(notCheckedHow) + "\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// --- the checklist and the full view ----------------------------------------------------------------------

// toolchainKeys is the recorded versions in the run's order; facts built without one (a test) use the order the
// NVIDIA toolchain was always written in.
func toolchainKeys(t *seam.ToolchainFacts) []string {
	if len(t.Keys) > 0 {
		return t.Keys
	}
	return slices.DeleteFunc([]string{"driver", "cuda", "torch", "nsight_systems", "nsight_compute"},
		func(k string) bool { _, ok := t.Versions[k]; return !ok })
}
