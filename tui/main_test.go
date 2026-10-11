package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestUnknownRoleTimeRefusedBeforeJobStarts(t *testing.T) {
	var out, err bytes.Buffer
	code := run([]string{"--json", "start", "missing.gguf", "--target", "test", "--role-time", "typo"}, &out, &err)
	if code != 2 || !strings.Contains(err.String(), "unknown --role-time") {
		t.Fatalf("code %d: %s %s", code, out.String(), err.String())
	}
}
