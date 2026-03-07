package core

import (
	"math"
	"testing"
)

func TestNewQuantizedTensorF32(t *testing.T) {
	// 2 rows × 4 cols of F32 data
	data := make([]byte, 2*4*4) // 2*4 elements * 4 bytes
	for i := 0; i < 8; i++ {
		val := float32(i + 1)
		bits := math.Float32bits(val)
		data[i*4+0] = byte(bits)
		data[i*4+1] = byte(bits >> 8)
		data[i*4+2] = byte(bits >> 16)
		data[i*4+3] = byte(bits >> 24)
	}

	qt, err := NewQuantizedTensor(data, 0, 2, 4) // type 0 = F32
	if err != nil {
		t.Fatalf("NewQuantizedTensor: %v", err)
	}

	if qt.Rows != 2 || qt.Cols != 4 {
		t.Errorf("dimensions: got %d×%d, want 2×4", qt.Rows, qt.Cols)
	}

	// Dequantize row 0 → [1, 2, 3, 4]
	out := make([]float32, 4)
	if err := qt.DequantizeRow(0, out); err != nil {
		t.Fatalf("DequantizeRow(0): %v", err)
	}
	for i, want := range []float32{1, 2, 3, 4} {
		if math.Abs(float64(out[i]-want)) > 1e-6 {
			t.Errorf("row0[%d] = %f, want %f", i, out[i], want)
		}
	}

	// Dequantize row 1 → [5, 6, 7, 8]
	if err := qt.DequantizeRow(1, out); err != nil {
		t.Fatalf("DequantizeRow(1): %v", err)
	}
	for i, want := range []float32{5, 6, 7, 8} {
		if math.Abs(float64(out[i]-want)) > 1e-6 {
			t.Errorf("row1[%d] = %f, want %f", i, out[i], want)
		}
	}
}

func TestNewQuantizedTensorFP32Cache(t *testing.T) {
	data := make([]byte, 1*4*4)
	qt, err := NewQuantizedTensor(data, 0, 1, 4)
	if err != nil {
		t.Fatal(err)
	}

	qt.FP32Data = []float32{10, 20, 30, 40}

	out := make([]float32, 4)
	qt.DequantizeRow(0, out)
	for i, want := range []float32{10, 20, 30, 40} {
		if out[i] != want {
			t.Errorf("cached[%d] = %f, want %f", i, out[i], want)
		}
	}
}

func TestNewQuantizedTensorInvalidDims(t *testing.T) {
	_, err := NewQuantizedTensor(nil, 0, 0, 4)
	if err == nil {
		t.Error("expected error for rows=0")
	}
	_, err = NewQuantizedTensor(nil, 0, 2, -1)
	if err == nil {
		t.Error("expected error for cols=-1")
	}
}

func TestNewQuantizedTensorDataMismatch(t *testing.T) {
	_, err := NewQuantizedTensor([]byte{1, 2, 3}, 0, 1, 4)
	if err == nil {
		t.Error("expected error for data length mismatch")
	}
}

func TestDequantizeRowBounds(t *testing.T) {
	data := make([]byte, 16)
	qt, _ := NewQuantizedTensor(data, 0, 1, 4)

	out := make([]float32, 4)
	if err := qt.DequantizeRow(-1, out); err == nil {
		t.Error("expected error for row=-1")
	}
	if err := qt.DequantizeRow(1, out); err == nil {
		t.Error("expected error for row=1 (out of bounds)")
	}
	if err := qt.DequantizeRow(0, make([]float32, 2)); err == nil {
		t.Error("expected error for small output buffer")
	}
}

func TestClose(t *testing.T) {
	data := make([]byte, 16)
	qt, _ := NewQuantizedTensor(data, 0, 1, 4)
	qt.FP32Data = []float32{1, 2, 3, 4}

	if err := qt.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if qt.Data != nil || qt.FP32Data != nil {
		t.Error("Close did not release data")
	}
}
