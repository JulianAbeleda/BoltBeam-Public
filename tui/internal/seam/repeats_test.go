package seam

import (
	"slices"
	"strings"
	"testing"
)

// A repeated pipeline (screen pipeline --repeats) prints each run's own lines after "repeat i/N: NAME". The bar
// gives each run an equal share, a run's "pipeline done" is not the job's, and only "repeat band: done" after the
// last run finishes it; a band the runs cannot make is a failure with Python's reason.
func TestReadProgressRepeated(t *testing.T) {
	one := []string{"pipeline steps: 2", "pipeline counted: 1,1", "stage measure: start", "stage measure: done", "stage output: start", "stage output: done", "pipeline done: r"}
	lines := append([]string{"=== start", "repeat 1/4: r"}, one...)
	lines = append(append(lines, "repeat 2/4: r-repeat-2"), one[:3]...)
	p := ReadProgress(lines)
	if p.Repeat != 2 || p.Repeats != 4 || p.Finished || p.Fraction(0) < 0.25 || p.Fraction(0) >= 0.5 {
		t.Fatalf("second of four runs: %+v %.2f", p, p.Fraction(0))
	}
	for _, i := range []string{"3", "4"} {
		lines = append(append(lines, one[3:]...), "repeat "+i+"/4: r-repeat-"+i)
		lines = append(lines, one[:3]...)
	}
	lines = append(lines, one[3:]...)
	if p = ReadProgress(lines); p.Finished || p.Fraction(0) != 0.99 {
		t.Fatalf("the last run done, the band not yet: %+v", p)
	}
	if p = ReadProgress(append(lines, "repeat band: done: n = 4, a row reads moved by chance 40% of the time")); !p.Finished || p.Fraction(0) != 1 {
		t.Fatalf("band done: %+v", p)
	}
	if p = ReadProgress(append(lines, "repeat band: failed: repeat work differs")); p.Finished || p.Failed != "repeat band" || p.Reason != "repeat work differs" {
		t.Fatalf("band refused: %+v", p)
	}
	if p = ReadProgress(one); !p.Finished || p.Repeats != 0 {
		t.Fatalf("one run is finished by its own pipeline done: %+v", p)
	}
}

// Settings and repeats reach the pipeline as given; Measure MTP names the ks Python offered and every run folder.
func TestPipelineSettingsRepeatsAndMtpArgv(t *testing.T) {
	c := Client{Python: "py", Root: "/r"}
	argv := c.PipelineArgv(Pipeline{Model: "m", RunDir: "/r/x", Target: "t", Settings: []string{"--chunk-size", "512"}, Repeats: 4})
	if i := slices.Index(argv, "--repeats"); i < 0 || argv[i+1] != "4" || !strings.HasSuffix(strings.Join(argv, " "), "--chunk-size 512") {
		t.Fatalf("argv %v", argv)
	}
	if argv = c.PipelineArgv(Pipeline{Model: "m", RunDir: "/r/x", Target: "t", Repeats: 1}); slices.Contains(argv, "--repeats") {
		t.Fatalf("one run asks for no repeats: %v", argv)
	}
	got := strings.Join(c.MtpArgv("/m/q", "nvidia_sm120", "/r/.work/q-mtp", []int{1, 2, 3}), " ")
	if want := "py -m boltbeam.workflow.screen mtp /m/q --target nvidia_sm120 --out /r/.work/q-mtp --ks 1,2,3 --facts-root /r"; !strings.HasPrefix(got, want) {
		t.Fatalf("mtp argv %q", got)
	}
}
