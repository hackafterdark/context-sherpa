package ggml

import (
	"testing"
)

func TestBytesForN(t *testing.T) {
	tests := []struct {
		ggmlType uint32
		n        int
		expected int
	}{
		{0, 100, 400},   // F32
		{1, 100, 200},   // F16
		{2, 32, 18},     // Q4_0
		{2, 64, 36},     // Q4_0
		{3, 32, 20},     // Q4_1
		{6, 32, 22},     // Q5_0
		{7, 32, 24},     // Q5_1
		{8, 32, 34},     // Q8_0
		{9, 32, 36},     // Q8_1
		{10, 256, 84},   // Q2_K
		{11, 256, 110},  // Q3_K
		{12, 256, 144},  // Q4_K
		{13, 256, 176},  // Q5_K
		{14, 256, 210},  // Q6_K
		{15, 256, 292},  // Q8_K
		{30, 100, 200},  // BF16
		{99, 100, 0},    // unsupported
	}
	for _, tt := range tests {
		got := bytesForN(tt.ggmlType, tt.n)
		if got != tt.expected {
			t.Errorf("bytesForN(%d, %d) = %d, want %d", tt.ggmlType, tt.n, got, tt.expected)
		}
	}
}

func TestGenerateTensorOrder(t *testing.T) {
	order := GenerateTensorOrder(2, 2)

	// Should start with encoder globals
	if len(order) == 0 {
		t.Fatal("empty tensor order")
	}
	if order[0] != "encoder.conv1.weight" {
		t.Errorf("order[0] = %q, want 'encoder.conv1.weight'", order[0])
	}

	// Should contain encoder layer 0 and 1 tensors
	found0 := false
	found1 := false
	for _, name := range order {
		if name == "encoder.layers.0.attn.q_proj.weight" {
			found0 = true
		}
		if name == "encoder.layers.1.attn.q_proj.weight" {
			found1 = true
		}
	}
	if !found0 {
		t.Error("missing encoder.layers.0.attn.q_proj.weight")
	}
	if !found1 {
		t.Error("missing encoder.layers.1.attn.q_proj.weight")
	}

	// Should contain decoder globals
	foundDecEmb := false
	for _, name := range order {
		if name == "decoder.token_embedding.weight" {
			foundDecEmb = true
		}
	}
	if !foundDecEmb {
		t.Error("missing decoder.token_embedding.weight")
	}

	// Should contain decoder layer tensors
	foundDecLayer := false
	for _, name := range order {
		if name == "decoder.layers.0.cross_attn.q_proj.weight" {
			foundDecLayer = true
		}
	}
	if !foundDecLayer {
		t.Error("missing decoder.layers.0.cross_attn.q_proj.weight")
	}

	// Encoder has 7 globals + 2*12 layers = 31
	// Decoder has 5 globals + 2*19 layers = 43
	// Total = 74
	expectedCount := 7 + 2*12 + 5 + 2*19
	if len(order) != expectedCount {
		t.Errorf("len(order) = %d, want %d", len(order), expectedCount)
	}
}

func TestGetTensorName(t *testing.T) {
	// Current implementation returns empty string
	name := GetTensorName(0, nil)
	if name != "" {
		t.Errorf("GetTensorName(0, nil) = %q, want empty", name)
	}
}
