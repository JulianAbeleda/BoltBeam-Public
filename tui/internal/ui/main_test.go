package ui

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/JulianAbeleda/BoltBeam/tui/internal/seam"
)

func TestMain(m *testing.M) {
	// The saved screen fixtures describe a Mac. Platform-specific tests select
	// their own goos explicitly, so the fixtures render identically on any host.
	goos = "darwin"
	// The plain vocabulary as `screen words` prints it (pinned in testdata/expected/words.json): what a screen reads
	// at start before any stage or role is named.
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "expected", "words.json"))
	if err != nil {
		panic(err)
	}
	vocab = &seam.Words{}
	if err := json.Unmarshal(raw, vocab); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}
