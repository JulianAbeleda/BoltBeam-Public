package ui

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
)

// driven runs Model through a real Bubble Tea program: its input is read and decoded as from a terminal, every key
// goes through Model.Update, and the program ends once want keys have arrived.
type driven struct {
	m    Model
	keys int
	want int
}

func (d driven) Init() tea.Cmd { return nil }

func (d driven) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if _, ok := msg.(tea.KeyPressMsg); !ok {
		return d, nil
	}
	next, _ := d.m.Update(msg)
	d.m, d.keys = next.(Model), d.keys+1
	if d.keys == d.want {
		return d, tea.Quit
	}
	return d, nil
}

func (d driven) View() tea.View { return tea.NewView("") }

// typeFast hands the program all of input in one read, as a terminal does when keys come faster than the program
// reads them, and returns the model after want key presses; fewer means keys were lost.
func typeFast(t *testing.T, m Model, input string, want int) Model {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	end, err := tea.NewProgram(driven{m: m, want: want}, tea.WithInput(bytes.NewBufferString(input)),
		tea.WithOutput(io.Discard), tea.WithContext(ctx), tea.WithWindowSize(80, 24)).Run()
	got, _ := end.(driven)
	if err != nil {
		t.Fatalf("%d of %d keys arrived: %v", got.keys, want, err)
	}
	return got.m
}

// Keys typed faster than the screen reads them each count: Bubble Tea v1 read "jjjjj" as one key with five runes,
// which no binding matched, so the cursor stayed where it was; and "12" on the batch page was not two digits.
func TestFastKeysAreNotDropped(t *testing.T) {
	files := make([]string, 30)
	for i := range files {
		files[i] = fmt.Sprintf("/m/model-%02d.gguf", i)
	}
	list := Model{f: Facts{Files: files, Path: files[0]}, cursor: pageModel, width: 80, height: 24}
	down := "\x1b[B"
	got := typeFast(t, list, "jjjjj"+strings.Repeat(down, 3)+"kjj", 11)
	if got.row != 9 {
		t.Fatalf("5 j, 3 down, k, 2 j: row %d, want 9", got.row)
	}
	if got := typeFast(t, list, "jjG", 3); got.row != len(got.actions())-1 { // a shifted letter is its own binding
		t.Fatalf("G: row %d, want the last row %d", got.row, len(got.actions())-1)
	}
	batch := Model{cursor: pageBatch, width: 80, height: 24}
	if got := typeFast(t, batch, "128", 3); got.f.BatchTyped != "128" {
		t.Fatalf("typed 128 on the batch page: %q", got.f.BatchTyped)
	}
}
