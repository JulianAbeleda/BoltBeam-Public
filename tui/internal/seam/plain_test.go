package seam

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The plain block and the words decode into the typed contract: layer 1's answers and trust line, layer 2's sections,
// and the vocabulary every stage, role and label is read from.
func TestPlainAndWordsDecode(t *testing.T) {
	var p Plain
	if err := json.Unmarshal(expected(t, "plain-002"), &p); err != nil {
		t.Fatal(err)
	}
	if p.Schema != "boltbeam.plain_results.v1" || p.ID != "qwen3-8b-apple_m3_10c-002" || p.Layer1.Fast != "The model writes 17.3 tok/s (tokens a second)." ||
		p.Layer1.Close == nil || len(p.Layer1.Figures) != 2 || p.Layer1.Figures[0].Figure != "17.3 tok/s" || p.Layer1.Trust.Key != "whole_only" || len(p.Layer2.Sections) != 2 || len(p.Layer2.Glossary) == 0 {
		t.Fatalf("plain decoded wrong: %+v", p.Layer1)
	}
	var w Words
	if err := json.Unmarshal(expected(t, "words"), &w); err != nil {
		t.Fatal(err)
	}
	if w.Stage("measure", "this Mac") != "Can this Mac measure" || w.Role("attn_kv") != "attention keys and values" ||
		w.Label("fast") != "How fast" || w.Term("stream") != "stream" || w.Units["tok_s"] != "tok/s" {
		t.Fatalf("words decoded wrong")
	}
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tui", "testdata", "recorded", "plain-batch.json"))
	if err != nil {
		t.Fatal(err)
	}
	var b Plain
	if err := json.Unmarshal(raw, &b); err != nil || b.Layer1.RunType != "batch_sweep" || len(b.Layer1.Top) != 3 ||
		b.Layer1.Top[0].Gain.Text == nil || b.Layer1.Top[0].Gain.Figure == "" || b.Layer2.Table == nil || len(b.Layer2.Table.Columns) != 4 ||
		len(b.Layer2.Table.AllColumns.Columns) == 0 || b.Layer1.Numbers["best_context"] == nil {
		t.Fatalf("batch plain decoded wrong: %v %+v", err, b.Layer1)
	}
}

// A saved run's plain block is read from the saved-runs folder; an id with a path in it is refused.
func TestPlainReadsTheSavedFolder(t *testing.T) {
	d := &Demo{Answers: map[string]json.RawMessage{"plain qwen3-8b-apple_m3_10c-002": json.RawMessage(expected(t, "plain-002"))}}
	c := Client{Root: "runs", Work: "runs/.work", Saved: "runs/saved", Demo: d}
	if p, err := c.Plain("qwen3-8b-apple_m3_10c-002", true); err != nil || p.ID != "qwen3-8b-apple_m3_10c-002" {
		t.Fatalf("saved: %v", err)
	}
	if _, err := c.Plain("../x", true); err == nil {
		t.Fatal("a saved id is a folder name")
	}
	// the tapes' demo answers the words and the plain block of every saved run it lists
	demo, err := LoadDemo(filepath.Join(repoRoot(t), "tui", "testdata", "demo", "demo.json"))
	if err != nil {
		t.Fatal(err)
	}
	c.Demo = demo
	if w, err := c.Words(); err != nil || w.Label("top") == "" {
		t.Fatalf("words: %v", err)
	}
	saved, _, err := c.SavedRuns()
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range saved.Runs {
		if r.ID == "qwen3-8b-apple_m3_10c-001" { // planned, never measured: no plain block to show
			continue
		}
		if p, err := c.Plain(r.ID, true); err != nil || p.ID != r.ID {
			t.Fatalf("plain %s: %v", r.ID, err)
		}
		if run, err := c.ShowSaved(r.ID); err != nil || run.ID != r.ID {
			t.Fatalf("run %s: %v", r.ID, err)
		}
	}
}
