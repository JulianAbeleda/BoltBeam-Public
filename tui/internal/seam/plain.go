package seam

// The plain words (report/plain.py, data/plain_words.json): every sentence the Results screen shows is Python's. `plain`
// is one run's two layers, `words` the vocabulary itself (stage, role and part names, the layer-1 labels, the glossary).

import (
	"fmt"
	"path/filepath"
	"strings"
)

// Plain is `screen plain`: layer 1 (the answers, the top three parts, the trust line) and layer 2 (the table, How it was
// measured, Machine and engine, the glossary).
type Plain struct {
	Schema string      `json:"schema"`
	Kind   string      `json:"kind"`
	ID     string      `json:"id"`
	Layer1 PlainLayer1 `json:"layer1"`
	Layer2 PlainLayer2 `json:"layer2"`
}

// PlainLayer1 leads with the headline figures, then answers how fast, how close to what this GPU allows, what holds it
// back; then the top three, the caveats under them (GainsNote, Unassigned) and the trust line.
type PlainLayer1 struct {
	RunType    string              `json:"run_type"`
	Title      string              `json:"title"`
	Figures    []PlainFigure       `json:"figures"`
	Fast       string              `json:"fast"`
	Close      *string             `json:"close"`
	Holds      *string             `json:"holds"`
	Upgrade    []string            `json:"upgrade"`
	Numbers    map[string]*float64 `json:"numbers"`
	Top        []PlainPart         `json:"top"`
	GainsNote  *string             `json:"gains_note"`
	Unassigned *string             `json:"unassigned"`
	Trust      PlainTrust          `json:"trust"`
}

// PlainFigure is one headline figure: Figure is the value and unit as layer 1 prints them ("100.4 tok/s", "14%"),
// Label the short words under it.
type PlainFigure struct {
	Key    string `json:"key"`
	Value  string `json:"value"`
	Unit   string `json:"unit"`
	Figure string `json:"figure"`
	Label  string `json:"label"`
}

// PlainPart is one part of the step: its plain name, time, share of the lost time and gain.
type PlainPart struct {
	Part      string    `json:"part"`
	Name      any       `json:"name"`
	What      *string   `json:"what"`
	Ms        *float64  `json:"ms"`
	Time      string    `json:"time"`
	LostMs    *float64  `json:"lost_ms"`
	Share     *float64  `json:"share"`
	ShareCell string    `json:"share_cell"`
	Gain      PlainGain `json:"gain"`
}

// PlainGain is what a part gives back at its limit; Text is nil when the table allocates none (gains_note says why, once).
type PlainGain struct {
	Value  *float64 `json:"value"`
	Unit   string   `json:"unit"`
	Cell   string   `json:"cell"`
	Figure string   `json:"figure"` // the gain as a figure, "+56.2 tok/s": the part of Text the screen sets apart
	Text   *string  `json:"text"`
}

// PlainTrust is the one trust line: "Checked ✓: …", or the caveat that matters most.
type PlainTrust struct {
	Checked bool   `json:"checked"`
	Key     string `json:"key"`
	Text    string `json:"text"`
}

// PlainLayer2 is the detail one level down.
type PlainLayer2 struct {
	Table    *PlainTable    `json:"table"`
	Sections []PlainSection `json:"sections"`
	Glossary []PlainTerm    `json:"glossary"`
}

// PlainTable is the where table at four columns (three when no gain is known); AllColumns is the table as the report
// has always printed it.
type PlainTable struct {
	Title      string     `json:"title"`
	Columns    []string   `json:"columns"`
	Rows       []PlainRow `json:"rows"`
	Note       *string    `json:"note"`
	AllColumns PlainAll   `json:"all_columns"`
}

// PlainRow is a table row: its cells, and the part it is.
type PlainRow struct {
	PlainPart
	Section *string  `json:"section"`
	Cells   []string `json:"cells"`
}

// PlainAll is every column: the rows, then the measured baseline and the sum when the table has them.
type PlainAll struct {
	Columns  []string   `json:"columns"`
	Rows     [][]string `json:"rows"`
	Baseline []string   `json:"baseline"`
	Sum      []string   `json:"sum"`
	Words    *string    `json:"words"`
}

// PlainSection is "How it was measured" or "Machine and engine": a title and its sentences.
type PlainSection struct {
	Title string   `json:"title"`
	Lines []string `json:"lines"`
}

// PlainTerm is one glossary entry the run's layer 2 uses.
type PlainTerm struct {
	Term  string `json:"term"`
	Plain string `json:"plain"`
}

// PlainCompare is the plain layer 1 of two runs compared (`screen diff`'s plain key).
type PlainCompare struct {
	Layer1 struct {
		Fast  string   `json:"fast"`
		Lines []string `json:"lines"`
		Top   []struct {
			Text string `json:"text"`
		} `json:"top"`
		Trust *PlainTrust `json:"trust"`
	} `json:"layer1"`
}

// Words is `screen words`: the one vocabulary every screen reads its plain names from.
type Words struct {
	Stages      map[string]string `json:"stages"`
	Roles       map[string]string `json:"roles"`
	ReportRoles []string          `json:"report_roles"`
	Parts       map[string]string `json:"parts"`
	Labels      map[string]string `json:"labels"`
	Units       map[string]string `json:"units"`
	Glossary    []struct {
		Term  string   `json:"term"`
		Match []string `json:"match"`
		Plain string   `json:"plain"`
	} `json:"glossary"`
	SpeedUp SpeedWords `json:"speed_up"`
}

// SpeedWords are the screen's own words for Speed up (data/plain_words.json speed_up): the rest of its sentences arrive
// worded in `speedup-offer` and `speedup-plain`.
type SpeedWords struct {
	Action string `json:"action"`
	Helper string `json:"helper"`
	UI     struct {
		Running     string `json:"running"`
		Left        string `json:"left"`
		Next        string `json:"next"`
		NextMtp     string `json:"next_mtp"`
		NextCompare string `json:"next_compare"`
		Check       string `json:"check"`
		CheckHelper string `json:"check_helper"`
		Checking    string `json:"checking"`
		Stop        string `json:"stop"`
		StopHelper  string `json:"stop_helper"`
		Stopped     string `json:"stopped"`
	} `json:"ui"`
}

// Stage is the plain name of a pipeline stage key, "{here}" filled with what the screen calls this machine; the key
// itself when the words have none.
func (w *Words) Stage(key, here string) string {
	if w != nil {
		if s, ok := w.Stages[key]; ok {
			return strings.ReplaceAll(s, "{here}", here)
		}
	}
	return key
}

// Role is the plain name of a role key, or the key.
func (w *Words) Role(key string) string {
	if w != nil {
		if s, ok := w.Roles[key]; ok {
			return s
		}
	}
	return key
}

// Label is one of layer 1's labels ("fast", "close", "holds", "top", "all_columns", "share", "top_line"), or "".
func (w *Words) Label(key string) string {
	if w == nil {
		return ""
	}
	return w.Labels[key]
}

// Term is the glossary's own spelling of a term, or "" when the glossary has no such term.
func (w *Words) Term(term string) string {
	if w != nil {
		for _, g := range w.Glossary {
			if g.Term == term {
				return g.Term
			}
		}
	}
	return ""
}

// Plain is the plain-results block of a run: in the saved-runs folder when saved, else where FindRun finds it.
func (c Client) Plain(id string, saved bool) (*Plain, error) {
	dir, err := c.FindRun(id)
	if saved {
		if id == "" || id != filepath.Base(id) || strings.HasPrefix(id, ".") {
			return nil, fmt.Errorf("saved run id must be a folder name: %q", id)
		}
		dir, err = filepath.Join(c.Saved, id), nil
	}
	if err != nil {
		return nil, err
	}
	var p Plain
	_, _, err = c.decode(&p, nil, "boltbeam.workflow.screen", "plain", "--run", dir)
	return &p, err
}

// Words is the plain vocabulary, read once per screen.
func (c Client) Words() (*Words, error) {
	var w Words
	_, _, err := c.decode(&w, nil, "boltbeam.workflow.screen", "words")
	return &w, err
}
