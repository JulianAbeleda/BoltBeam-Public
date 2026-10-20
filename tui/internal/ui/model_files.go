package ui

import (
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func modelConfig(path string) map[string]any {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var cfg map[string]any
	if json.Unmarshal(data, &cfg) != nil {
		return nil
	}
	return cfg
}

func checkpoint(path string) bool {
	cfg := modelConfig(filepath.Join(path, "config.json"))
	kind, _ := cfg["model_type"].(string)
	weights, _ := filepath.Glob(filepath.Join(path, "*.safetensors"))
	return kind != "" && len(weights) > 0
}

func discoverModels(roots []string) []string {
	seen := map[string]bool{}
	visited := map[string]bool{}
	var out []string
	add := func(path string) {
		real, err := filepath.EvalSymlinks(path)
		if err == nil && !seen[real] {
			seen[real] = true
			out = append(out, path)
		}
	}
	for _, root := range roots {
		real, err := filepath.EvalSymlinks(root)
		if err != nil || visited[real] {
			continue
		}
		visited[real] = true
		// Walk the resolved root; retain the caller's path for selection.
		_ = filepath.WalkDir(real, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			rel, _ := filepath.Rel(real, path)
			display := filepath.Join(root, rel)
			if d.IsDir() {
				if strings.HasPrefix(d.Name(), ".") && path != real {
					return filepath.SkipDir
				}
				if path != real && (d.Name() == "comfyui" || d.Name() == "ollama") {
					return filepath.SkipDir
				}
				if _, err := os.Stat(filepath.Join(path, "adapter_config.json")); err == nil {
					return filepath.SkipDir
				}
				if checkpoint(path) {
					add(display)
					return filepath.SkipDir
				}
				return nil
			}
			ext := strings.ToLower(filepath.Ext(path))
			if strings.HasPrefix(strings.ToLower(d.Name()), "mmproj-") {
				return nil
			}
			if ext == ".gguf" || (filepath.Dir(path) == real && (ext == ".safetensors" || ext == ".onnx")) {
				add(display)
			}
			return nil
		})
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := filepath.Base(out[i]), filepath.Base(out[j])
		if a == b {
			return out[i] < out[j]
		}
		return a < b
	})
	return out
}

func modelLabel(path string) string {
	name := filepath.Base(path)
	if filepath.Base(filepath.Dir(path)) == "snapshots" {
		repo := filepath.Base(filepath.Dir(filepath.Dir(path)))
		name = strings.ReplaceAll(strings.TrimPrefix(repo, "models--"), "--", "/")
	}
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".gguf" || ext == ".safetensors" || ext == ".onnx" {
		name = strings.TrimSuffix(name, filepath.Ext(path))
		if name == "model" {
			name = filepath.Base(filepath.Dir(path))
		}
		return name + " · " + strings.ToUpper(strings.TrimPrefix(ext, "."))
	}
	cfg := modelConfig(filepath.Join(path, "config.json"))
	q, _ := cfg["quantization_config"].(map[string]any)
	if len(q) == 0 {
		for _, file := range []string{"quantize_config.json", "quant_config.json"} {
			if q = modelConfig(filepath.Join(path, file)); len(q) > 0 {
				break
			}
		}
	}
	method, _ := q["quant_method"].(string)
	format := "Safetensors"
	if method != "" {
		format += " / " + strings.ToUpper(method)
		if method == "compressed-tensors" {
			format = "Safetensors (compressed)"
		}
	}
	// ModelOpt can declare mixed precision, with NVFP4 weight groups.
	data, _ := json.Marshal(q)
	hf := modelConfig(filepath.Join(path, "hf_quant_config.json"))
	hfData, _ := json.Marshal(hf["quantization"])
	if strings.Contains(strings.ToUpper(string(data)), `"NVFP4"`) || strings.Contains(strings.ToUpper(string(hfData)), `"NVFP4"`) {
		format = "Safetensors / NVFP4"
	}
	if cfg == nil {
		return name
	}
	return name + " · " + format
}
