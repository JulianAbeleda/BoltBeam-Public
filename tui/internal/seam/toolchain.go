package seam

import (
	"bytes"
	"encoding/json"
)

// UnmarshalJSON keeps the recorded order of the versions as Keys: each vendor's toolchain is its own list
// (target/vendors.py), printed in the order the run recorded it.
func (t *ToolchainFacts) UnmarshalJSON(data []byte) error {
	type plain ToolchainFacts
	var raw struct {
		Versions json.RawMessage `json:"versions"`
	}
	if err := json.Unmarshal(data, (*plain)(t)); err != nil || json.Unmarshal(data, &raw) != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(raw.Versions))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil // no versions object: no order
	}
	t.Keys = nil
	for dec.More() {
		k, _ := dec.Token()
		var skip json.RawMessage
		if dec.Decode(&skip) != nil {
			return nil
		}
		t.Keys = append(t.Keys, k.(string))
	}
	return nil
}
