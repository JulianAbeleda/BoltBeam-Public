package ui

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	// The saved screen fixtures describe a Mac. Platform-specific tests select
	// their own goos explicitly, so the fixtures render identically on any host.
	goos = "darwin"
	os.Exit(m.Run())
}
