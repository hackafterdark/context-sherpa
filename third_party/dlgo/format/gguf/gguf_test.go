package gguf

import (
	"os"
	"path/filepath"
	"testing"
)

func testdataPath(name string) string {
	return filepath.Join("..", "..", "testdata", name)
}

func TestOpenGGUF(t *testing.T) {
	path := testdataPath("test.gguf")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("testdata not found: %s (run 'go run ./testdata/generate.go' first)", path)
	}

	gf, err := Open(path)
	if err != nil {
		t.Fatalf("Open(%q): %v", path, err)
	}

	// Version
	if gf.Version != 3 {
		t.Errorf("Version = %d, want 3", gf.Version)
	}

	// Tensor count
	if gf.TensorCount != 2 {
		t.Errorf("TensorCount = %d, want 2", gf.TensorCount)
	}

	// Metadata count
	if gf.MetadataCount != 3 {
		t.Errorf("MetadataCount = %d, want 3", gf.MetadataCount)
	}

	// Check metadata values
	if arch, ok := gf.Metadata["general.architecture"]; !ok {
		t.Error("missing metadata key 'general.architecture'")
	} else if archStr, ok := arch.(string); !ok || archStr != "test" {
		t.Errorf("general.architecture = %v, want 'test'", arch)
	}

	if name, ok := gf.Metadata["general.name"]; !ok {
		t.Error("missing metadata key 'general.name'")
	} else if nameStr, ok := name.(string); !ok || nameStr != "dlgo-test-model" {
		t.Errorf("general.name = %v, want 'dlgo-test-model'", name)
	}

	if hs, ok := gf.Metadata["test.hidden_size"]; !ok {
		t.Error("missing metadata key 'test.hidden_size'")
	} else if hsVal, ok := hs.(uint32); !ok || hsVal != 64 {
		t.Errorf("test.hidden_size = %v, want 64", hs)
	}

	// Check tensors
	if len(gf.Tensors) != 2 {
		t.Fatalf("len(Tensors) = %d, want 2", len(gf.Tensors))
	}

	ta := gf.Tensors[0]
	if ta.Name != "weight_a" {
		t.Errorf("Tensors[0].Name = %q, want 'weight_a'", ta.Name)
	}
	if ta.NDims != 2 {
		t.Errorf("weight_a.NDims = %d, want 2", ta.NDims)
	}
	if ta.Dimensions[0] != 64 || ta.Dimensions[1] != 32 {
		t.Errorf("weight_a.Dimensions = %v, want [64, 32]", ta.Dimensions)
	}
	if ta.Type != GGMLTypeF32 {
		t.Errorf("weight_a.Type = %d, want %d (F32)", ta.Type, GGMLTypeF32)
	}

	tb := gf.Tensors[1]
	if tb.Name != "bias_a" {
		t.Errorf("Tensors[1].Name = %q, want 'bias_a'", tb.Name)
	}
	if tb.NDims != 1 {
		t.Errorf("bias_a.NDims = %d, want 1", tb.NDims)
	}
	if tb.Dimensions[0] != 64 {
		t.Errorf("bias_a.Dimensions = %v, want [64]", tb.Dimensions)
	}

	// DataOffset should be > 0
	if gf.DataOffset <= 0 {
		t.Errorf("DataOffset = %d, want > 0", gf.DataOffset)
	}
}

func TestOpenGGUFInvalidMagic(t *testing.T) {
	tmp := filepath.Join(t.TempDir(), "bad.gguf")
	os.WriteFile(tmp, []byte("BAAD"), 0644)
	_, err := Open(tmp)
	if err == nil {
		t.Error("expected error for invalid magic")
	}
}

func TestOpenGGUFMissingFile(t *testing.T) {
	_, err := Open("/nonexistent/file.gguf")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestGGMLTypeConstants(t *testing.T) {
	if GGMLTypeF32 != 0 {
		t.Errorf("GGMLTypeF32 = %d, want 0", GGMLTypeF32)
	}
	if GGMLTypeF16 != 1 {
		t.Errorf("GGMLTypeF16 = %d, want 1", GGMLTypeF16)
	}
	if GGMLTypeQ4_0 != 2 {
		t.Errorf("GGMLTypeQ4_0 = %d, want 2", GGMLTypeQ4_0)
	}
	if GGMLTypeQ8_0 != 8 {
		t.Errorf("GGMLTypeQ8_0 = %d, want 8", GGMLTypeQ8_0)
	}
}

func TestOpenQuantGGUF(t *testing.T) {
	path := testdataPath("test_quant.gguf")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("testdata not found")
	}

	gf, err := Open(path)
	if err != nil {
		t.Fatalf("Open(%q): %v", path, err)
	}

	if gf.TensorCount != 2 {
		t.Errorf("TensorCount = %d, want 2", gf.TensorCount)
	}

	// First tensor should be Q8_0
	if gf.Tensors[0].Type != GGMLTypeQ8_0 {
		t.Errorf("Tensors[0].Type = %d, want %d (Q8_0)", gf.Tensors[0].Type, GGMLTypeQ8_0)
	}
	if gf.Tensors[0].Name != "layer.0.weight" {
		t.Errorf("Tensors[0].Name = %q, want 'layer.0.weight'", gf.Tensors[0].Name)
	}
	if gf.Tensors[0].Dimensions[0] != 128 || gf.Tensors[0].Dimensions[1] != 32 {
		t.Errorf("Tensors[0].Dimensions = %v, want [128, 32]", gf.Tensors[0].Dimensions)
	}

	// Second tensor should be F32
	if gf.Tensors[1].Type != GGMLTypeF32 {
		t.Errorf("Tensors[1].Type = %d, want %d (F32)", gf.Tensors[1].Type, GGMLTypeF32)
	}
}

func TestOpenMultiQuantGGUF(t *testing.T) {
	path := testdataPath("test_multi.gguf")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("testdata not found")
	}

	gf, err := Open(path)
	if err != nil {
		t.Fatalf("Open(%q): %v", path, err)
	}

	if gf.TensorCount != 4 {
		t.Errorf("TensorCount = %d, want 4", gf.TensorCount)
	}

	expectedTypes := []GGMLType{GGMLTypeF32, GGMLTypeF16, GGMLTypeQ4_0, GGMLTypeQ8_0}
	expectedNames := []string{"embed.weight", "embed.bias", "attn.weight", "ffn.weight"}

	for i, tensor := range gf.Tensors {
		if tensor.Type != expectedTypes[i] {
			t.Errorf("Tensors[%d].Type = %d, want %d", i, tensor.Type, expectedTypes[i])
		}
		if tensor.Name != expectedNames[i] {
			t.Errorf("Tensors[%d].Name = %q, want %q", i, tensor.Name, expectedNames[i])
		}
	}

	// Check metadata
	if lc, ok := gf.Metadata["test.layer_count"]; !ok {
		t.Error("missing 'test.layer_count' metadata")
	} else if lcVal, ok := lc.(uint32); !ok || lcVal != 4 {
		t.Errorf("test.layer_count = %v, want 4", lc)
	}
}
