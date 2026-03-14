package quant

import (
	"encoding/binary"
	"math"
	"testing"
)

// --- Dequantize dispatcher tests ---

func TestDequantizeF32(t *testing.T) {
	n := 4
	data := make([]byte, n*4)
	vals := []float32{1.0, -2.5, 3.14, 0.0}
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}

	result, err := Dequantize(data, 0, n)
	if err != nil {
		t.Fatalf("Dequantize F32: %v", err)
	}
	for i, want := range vals {
		if math.Abs(float64(result[i]-want)) > 1e-6 {
			t.Errorf("F32[%d] = %f, want %f", i, result[i], want)
		}
	}
}

func TestDequantizeF16(t *testing.T) {
	n := 3
	data := make([]byte, n*2)
	// 1.0 = 0x3C00, 0.5 = 0x3800, -1.0 = 0xBC00
	f16vals := []uint16{0x3C00, 0x3800, 0xBC00}
	expected := []float32{1.0, 0.5, -1.0}

	for i, v := range f16vals {
		binary.LittleEndian.PutUint16(data[i*2:], v)
	}

	result, err := Dequantize(data, 1, n)
	if err != nil {
		t.Fatalf("Dequantize F16: %v", err)
	}
	for i, want := range expected {
		if math.Abs(float64(result[i]-want)) > 1e-4 {
			t.Errorf("F16[%d] = %f, want %f", i, result[i], want)
		}
	}
}

func TestDequantizeUnsupported(t *testing.T) {
	_, err := Dequantize(nil, 255, 0)
	if err == nil {
		t.Error("expected error for unsupported type 255")
	}
}

// --- DequantizeInto tests ---

func TestDequantizeIntoF32(t *testing.T) {
	n := 4
	data := make([]byte, n*4)
	vals := []float32{10, 20, 30, 40}
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}

	dst := make([]float32, n)
	DequantizeInto(dst, data, 0, n)
	for i, want := range vals {
		if dst[i] != want {
			t.Errorf("DequantizeInto F32[%d] = %f, want %f", i, dst[i], want)
		}
	}
}

func TestDequantizeIntoF16(t *testing.T) {
	n := 2
	data := make([]byte, n*2)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // 1.0
	binary.LittleEndian.PutUint16(data[2:], 0x4000) // 2.0

	dst := make([]float32, n)
	DequantizeInto(dst, data, 1, n)
	expected := []float32{1.0, 2.0}
	for i, want := range expected {
		if math.Abs(float64(dst[i]-want)) > 1e-4 {
			t.Errorf("DequantizeInto F16[%d] = %f, want %f", i, dst[i], want)
		}
	}
}

func TestDequantizeIntoQ4_0(t *testing.T) {
	// Create known Q4_0 block
	data := make([]byte, 18)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale = 1.0
	for i := 2; i < 18; i++ {
		data[i] = 0x88 // nibbles: 8 (low) and 8 (high) → (8-8)*d = 0.0
	}

	dst := make([]float32, 32)
	DequantizeInto(dst, data, 2, 32)
	for i, v := range dst {
		if v != 0 {
			t.Errorf("Q4_0[%d] = %f, want 0 (nibble 8 - 8 = 0)", i, v)
		}
	}
}

func TestDequantizeIntoFallback(t *testing.T) {
	// BF16 (type 30) is not directly handled by DequantizeInto — uses fallback
	n := 2
	data := make([]byte, n*2)
	// BF16 1.0 = 0x3F80
	binary.LittleEndian.PutUint16(data[0:], 0x3F80)
	binary.LittleEndian.PutUint16(data[2:], 0x4000) // 2.0

	dst := make([]float32, n)
	DequantizeInto(dst, data, 30, n)
	if math.Abs(float64(dst[0]-1.0)) > 0.01 {
		t.Errorf("BF16 fallback[0] = %f, want ~1.0", dst[0])
	}
}

// --- FusedDotProduct tests ---

func TestFusedDotProductF32(t *testing.T) {
	n := 4
	data := make([]byte, n*4)
	vals := []float32{1, 2, 3, 4}
	x := []float32{5, 6, 7, 8}
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}

	got := FusedDotProduct(data, 0, x, n)
	want := float32(1*5 + 2*6 + 3*7 + 4*8) // 70
	if math.Abs(float64(got-want)) > 1e-4 {
		t.Errorf("FusedDotProduct F32 = %f, want %f", got, want)
	}
}

func TestFusedDotProductF16(t *testing.T) {
	n := 2
	data := make([]byte, n*2)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // 1.0
	binary.LittleEndian.PutUint16(data[2:], 0x4000) // 2.0
	x := []float32{3.0, 4.0}

	got := FusedDotProduct(data, 1, x, n)
	want := float32(1*3 + 2*4) // 11
	if math.Abs(float64(got-want)) > 0.1 {
		t.Errorf("FusedDotProduct F16 = %f, want %f", got, want)
	}
}

func TestFusedDotProductQ4_0(t *testing.T) {
	// Create a Q4_0 block where all values are 0 (nibble=8, 8-8=0)
	data := make([]byte, 18)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale = 1.0
	for i := 2; i < 18; i++ {
		data[i] = 0x88
	}
	x := make([]float32, 32)
	for i := range x {
		x[i] = 1.0
	}

	got := FusedDotProduct(data, 2, x, 32)
	if math.Abs(float64(got)) > 1e-4 {
		t.Errorf("FusedDotProduct Q4_0 all-zero = %f, want 0", got)
	}
}

func TestFusedDotProductQ8_0(t *testing.T) {
	data := make([]byte, 34)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale = 1.0
	data[2] = byte(int8(1))                          // quant = 1
	data[3] = byte(int8(2))                          // quant = 2

	x := make([]float32, 32)
	x[0] = 10.0
	x[1] = 20.0

	got := FusedDotProduct(data, 8, x, 32)
	want := float32(1*10 + 2*20) // 50
	if math.Abs(float64(got-want)) > 0.5 {
		t.Errorf("FusedDotProduct Q8_0 = %f, want %f", got, want)
	}
}

func TestFusedDotProductFallback(t *testing.T) {
	// BF16 (type 30) should work via fallback
	n := 2
	data := make([]byte, n*2)
	binary.LittleEndian.PutUint16(data[0:], 0x3F80) // BF16 1.0
	binary.LittleEndian.PutUint16(data[2:], 0x4000) // BF16 2.0
	x := []float32{3.0, 4.0}

	got := FusedDotProduct(data, 30, x, n)
	// Should be approximately 1*3 + 2*4 = 11
	if math.Abs(float64(got-11.0)) > 0.5 {
		t.Errorf("FusedDotProduct BF16 fallback = %f, want ~11", got)
	}
}

// --- Consistency test: Dequantize then dot = FusedDot ---

func TestFusedDotConsistencyQ4_0(t *testing.T) {
	data := make([]byte, 18)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale = 1.0
	for i := 2; i < 18; i++ {
		data[i] = byte(i) // varied nibbles
	}
	x := make([]float32, 32)
	for i := range x {
		x[i] = float32(i) * 0.1
	}

	// Method 1: Dequantize then dot
	floats := DequantizeQ4_0(data, 32)
	var want float32
	for i := 0; i < 32; i++ {
		want += floats[i] * x[i]
	}

	// Method 2: FusedDotProduct
	got := FusedDotProduct(data, 2, x, 32)
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Errorf("Q4_0 consistency: fused=%f, dequant+dot=%f", got, want)
	}
}

func TestFusedDotConsistencyQ8_0(t *testing.T) {
	data := make([]byte, 34)
	binary.LittleEndian.PutUint16(data[0:], 0x3800) // scale = 0.5
	for i := 0; i < 32; i++ {
		data[2+i] = byte(int8(i - 16))
	}
	x := make([]float32, 32)
	for i := range x {
		x[i] = float32(i) * 0.05
	}

	floats := DequantizeQ8_0(data, 32)
	var want float32
	for i := 0; i < 32; i++ {
		want += floats[i] * x[i]
	}

	got := FusedDotProduct(data, 8, x, 32)
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Errorf("Q8_0 consistency: fused=%f, dequant+dot=%f", got, want)
	}
}

// --- SIMD stubs test ---

func TestSIMDDotProduct(t *testing.T) {
	n := 4
	data := make([]byte, n*4)
	vals := []float32{1, 2, 3, 4}
	x := []float32{1, 1, 1, 1}
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}

	got := SIMDDotProduct(data, 0, x, n)
	if math.Abs(float64(got-10.0)) > 1e-4 {
		t.Errorf("SIMDDotProduct F32 = %f, want 10", got)
	}
}

func TestSIMDDotBatch(t *testing.T) {
	// 2 rows of 4 F32 values each
	bytesPerRow := 4 * 4
	data := make([]byte, 2*bytesPerRow)
	row0 := []float32{1, 2, 3, 4}
	row1 := []float32{5, 6, 7, 8}
	for i, v := range row0 {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	for i, v := range row1 {
		binary.LittleEndian.PutUint32(data[bytesPerRow+i*4:], math.Float32bits(v))
	}

	x := []float32{1, 1, 1, 1}
	out := make([]float32, 2)
	SIMDDotBatch(data, 0, x, 4, out, 2, bytesPerRow)

	if math.Abs(float64(out[0]-10.0)) > 1e-4 {
		t.Errorf("SIMDDotBatch row0 = %f, want 10", out[0])
	}
	if math.Abs(float64(out[1]-26.0)) > 1e-4 {
		t.Errorf("SIMDDotBatch row1 = %f, want 26", out[1])
	}
}

func TestSIMDScaleAdd(t *testing.T) {
	out := []float32{1, 2, 3}
	src := []float32{10, 20, 30}
	SIMDScaleAdd(out, 0.5, src, 3)
	expected := []float32{6, 12, 18}
	for i, want := range expected {
		if math.Abs(float64(out[i]-want)) > 1e-6 {
			t.Errorf("SIMDScaleAdd[%d] = %f, want %f", i, out[i], want)
		}
	}
}

func TestSIMDSwiGLU(t *testing.T) {
	out := make([]float32, 3)
	gate := []float32{0, 1, -1}
	up := []float32{1, 2, 3}
	SIMDSwiGLU(out, gate, up, 3)

	// SwiGLU(0, 1) = (0 / (1+exp(0))) * 1 = 0 * 1 = 0
	if math.Abs(float64(out[0])) > 1e-4 {
		t.Errorf("SwiGLU[0] = %f, want ~0", out[0])
	}
	// SwiGLU(1, 2) = (1 / (1+exp(-1))) * 2 ≈ 0.7311 * 2 ≈ 1.4621
	if math.Abs(float64(out[1]-1.4621)) > 0.01 {
		t.Errorf("SwiGLU[1] = %f, want ~1.4621", out[1])
	}
}

func TestHasSIMDDot(t *testing.T) {
	// F32, F16, Q4_0, Q8_0 should be supported
	for _, typ := range []uint32{0, 1, 2, 8} {
		if !HasSIMDDot(typ) {
			t.Errorf("HasSIMDDot(%d) = false, want true", typ)
		}
	}
	// Unsupported types
	if HasSIMDDot(99) {
		t.Error("HasSIMDDot(99) = true, want false")
	}
}

// --- BytesForN extended tests ---

func TestBytesForNAllTypes(t *testing.T) {
	tests := []struct {
		name     string
		ggmlType uint32
		n        int
		expected int
	}{
		{"F32", 0, 1, 4},
		{"F16", 1, 1, 2},
		{"Q4_0", 2, 32, 18},
		{"Q4_1", 3, 32, 20},
		{"Q5_0", 6, 32, 22},
		{"Q5_1", 7, 32, 24},
		{"Q8_0", 8, 32, 34},
		{"Q8_1", 9, 32, 36},
		{"Q2_K", 10, 256, 84},
		{"Q3_K", 11, 256, 110},
		{"Q4_K", 12, 256, 144},
		{"Q5_K", 13, 256, 176},
		{"Q6_K", 14, 256, 210},
		{"Q8_K", 15, 256, 292},
		{"IQ2_XXS", 16, 256, 66},
		{"IQ2_XS", 17, 256, 74},
		{"IQ3_XXS", 18, 256, 98},
		{"IQ1_S", 19, 256, 50},
		{"IQ4_NL", 20, 32, 18},
		{"IQ3_S", 21, 256, 110},
		{"IQ2_S", 22, 256, 82},
		{"IQ4_XS", 23, 256, 136},
		{"IQ1_M", 29, 256, 56},
		{"BF16", 30, 1, 2},
		{"TQ1_0", 34, 256, 54},
		{"TQ2_0", 35, 256, 66},
		{"Unknown", 99, 100, 0},
	}
	for _, tt := range tests {
		got := BytesForN(tt.ggmlType, tt.n)
		if got != tt.expected {
			t.Errorf("BytesForN(%s=%d, %d) = %d, want %d", tt.name, tt.ggmlType, tt.n, got, tt.expected)
		}
	}
}

// --- BF16 test ---

func TestDequantizeBF16Roundtrip(t *testing.T) {
	n := 2
	data := make([]byte, n*2)
	binary.LittleEndian.PutUint16(data[0:], 0x3F80) // BF16 1.0
	binary.LittleEndian.PutUint16(data[2:], 0xBF80) // BF16 -1.0

	result := DequantizeBF16(data, n)
	if math.Abs(float64(result[0]-1.0)) > 0.01 {
		t.Errorf("BF16[0] = %f, want ~1.0", result[0])
	}
	if math.Abs(float64(result[1]-(-1.0))) > 0.01 {
		t.Errorf("BF16[1] = %f, want ~-1.0", result[1])
	}
}

// --- Q4_1 test ---

func TestQ4_1Dequantization(t *testing.T) {
	data := make([]byte, 20)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // d = 1.0
	binary.LittleEndian.PutUint16(data[2:], 0x3800) // m = 0.5
	// All nibbles = 0 → value = 0*d + m = 0.5
	for i := 4; i < 20; i++ {
		data[i] = 0x00
	}

	result := DequantizeQ4_1(data, 32)
	if len(result) != 32 {
		t.Fatalf("len = %d, want 32", len(result))
	}
	for i, v := range result {
		if math.Abs(float64(v-0.5)) > 1e-2 {
			t.Errorf("[%d] = %f, want ~0.5", i, v)
		}
	}
}

// --- Q5_0 test ---

func TestQ5_0Dequantization(t *testing.T) {
	data := make([]byte, 22)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // d = 1.0
	// qh = 0 (all 5th bits are 0)
	// nibbles all 0x00 → q = 0, q-16 = -16 → value = -16.0
	result := DequantizeQ5_0(data, 32)
	if len(result) != 32 {
		t.Fatalf("len = %d, want 32", len(result))
	}
	for i, v := range result {
		expected := float32(-16.0)
		if math.Abs(float64(v-expected)) > 1e-2 {
			t.Errorf("[%d] = %f, want %f", i, v, expected)
		}
	}
}
