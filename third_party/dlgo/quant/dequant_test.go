package quant

import (
	"math"
	"testing"
)

func TestFloat16(t *testing.T) {
	tests := []struct {
		input    uint16
		expected float32
	}{
		{0x3C00, 1.0},
		{0x0000, 0.0},
		{0xBC00, -1.0},
		{0x4000, 2.0},
		{0x3800, 0.5},
	}

	for _, tt := range tests {
		result := float16ToFloat32(tt.input)
		if math.Abs(float64(result-tt.expected)) > 1e-6 {
			t.Errorf("float16ToFloat32(0x%04X) = %f, expected %f", tt.input, result, tt.expected)
		}
	}
}

func TestQ4_0Dequantization(t *testing.T) {
	// Create a known Q4_0 block
	// Block structure: 2 bytes for float16 scale + 16 bytes of nibbles
	data := make([]byte, 18)

	// Set scale to 1.0 (float16: 0x3C00)
	data[0] = 0x00
	data[1] = 0x3C

	// Set all nibbles to 0 (which represents value -8, offset by 8)
	// So with scale=1.0: (-8) * 1.0 = -8.0
	for i := 2; i < 18; i++ {
		data[i] = 0x00 // Both nibbles are 0
	}

	result := DequantizeQ4_0(data, 32)
	if len(result) != 32 {
		t.Errorf("Expected 32 values, got %d", len(result))
	}

	// All values should be -8.0
	for i, v := range result {
		expected := float32(-8.0)
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("Value at index %d: got %f, expected %f", i, v, expected)
		}
	}

	// Test with different nibbles
	data2 := make([]byte, 18)
	data2[0] = 0x00
	data2[1] = 0x3C // scale = 1.0
	data2[2] = 0xF0 // low nibble = 0 (-8), high nibble = 15 (7)

	result2 := DequantizeQ4_0(data2, 32)
	// First value (from low nibble of byte 2)
	if math.Abs(float64(result2[0]-(-8.0))) > 1e-6 {
		t.Errorf("Expected result2[0] = -8.0, got %f", result2[0])
	}
	// 17th value (from high nibble of byte 2)
	if math.Abs(float64(result2[16]-7.0)) > 1e-6 {
		t.Errorf("Expected result2[16] = 7.0, got %f", result2[16])
	}
}

func TestQ8_0Dequantization(t *testing.T) {
	// Create a known Q8_0 block
	data := make([]byte, 34)

	// Set scale to 1.0 (float16: 0x3C00)
	data[0] = 0x00
	data[1] = 0x3C

	// Set first int8 to 5
	data[2] = 5

	result := DequantizeQ8_0(data, 32)
	if len(result) != 32 {
		t.Errorf("Expected 32 values, got %d", len(result))
	}

	// First value should be 5.0
	if math.Abs(float64(result[0]-5.0)) > 1e-6 {
		t.Errorf("Expected result[0] = 5.0, got %f", result[0])
	}
}

func TestBytesForN(t *testing.T) {
	tests := []struct {
		ggmlType uint32
		n        int
		expected int
	}{
		{0, 100, 400},  // F32: 100 * 4
		{1, 100, 200},  // F16: 100 * 2
		{2, 32, 18},    // Q4_0: (32/32) * 18
		{2, 64, 36},    // Q4_0: (64/32) * 18
		{8, 32, 34},    // Q8_0: (32/32) * 34
	}

	for _, tt := range tests {
		result := BytesForN(tt.ggmlType, tt.n)
		if result != tt.expected {
			t.Errorf("BytesForN(%d, %d) = %d, expected %d", tt.ggmlType, tt.n, result, tt.expected)
		}
	}
}
