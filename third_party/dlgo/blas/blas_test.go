package blas

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/quant"
)

func makeF32Tensor(rows, cols int, vals []float32) *core.QuantizedTensor {
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	qt, _ := core.NewQuantizedTensor(data, 0, rows, cols)
	return qt
}

func TestQMatVecMul_Identity(t *testing.T) {
	// 3x3 identity matrix
	W := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	qt := makeF32Tensor(3, 3, W)
	x := []float32{2, 3, 5}
	out := make([]float32, 3)
	QMatVecMul(out, qt, x)

	for i, want := range []float32{2, 3, 5} {
		if math.Abs(float64(out[i]-want)) > 1e-5 {
			t.Errorf("out[%d] = %f, want %f", i, out[i], want)
		}
	}
}

func TestQMatVecMul_WithFP32Cache(t *testing.T) {
	W := []float32{1, 2, 3, 4}
	qt := makeF32Tensor(2, 2, W)
	qt.FP32Data = W

	x := []float32{1, 1}
	out := make([]float32, 2)
	QMatVecMul(out, qt, x)

	if math.Abs(float64(out[0]-3)) > 1e-5 || math.Abs(float64(out[1]-7)) > 1e-5 {
		t.Errorf("got %v, want [3, 7]", out)
	}
}

func TestQBatchGEMM(t *testing.T) {
	// 2x2 matrix, 3 positions
	W := []float32{1, 0, 0, 1}
	qt := makeF32Tensor(2, 2, W)

	xFlat := []float32{1, 2, 3, 4, 5, 6}
	outFlat := make([]float32, 6)
	QBatchGEMM(outFlat, qt, xFlat, 3)

	for i, want := range []float32{1, 2, 3, 4, 5, 6} {
		if math.Abs(float64(outFlat[i]-want)) > 1e-5 {
			t.Errorf("outFlat[%d] = %f, want %f", i, outFlat[i], want)
		}
	}
}

func TestQBatchGEMM_ZeroPos(t *testing.T) {
	W := []float32{1, 0}
	qt := makeF32Tensor(1, 2, W)
	QBatchGEMM(nil, qt, nil, 0) // should not panic
}

func TestDequantizeAll(t *testing.T) {
	W := []float32{1.5, 2.5, 3.5, 4.5}
	qt := makeF32Tensor(2, 2, W)
	got := DequantizeAll(qt)

	for i, want := range W {
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Errorf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestPreDequantize(t *testing.T) {
	W := []float32{1, 2, 3, 4}
	data := make([]byte, 16)
	for i, v := range W {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	qt := &core.QuantizedTensor{
		Data: data,
		Type: 0,
		Rows: 2,
		Cols: 2,
	}

	PreDequantize(qt)

	if qt.FP32Data == nil {
		t.Fatal("FP32Data should be populated after PreDequantize")
	}
	if qt.Data != nil {
		t.Error("raw Data should be nil after PreDequantize")
	}

	x := []float32{1, 0}
	out := make([]float32, 2)
	QMatVecMul(out, qt, x)
	if math.Abs(float64(out[0]-1)) > 1e-5 || math.Abs(float64(out[1]-3)) > 1e-5 {
		t.Errorf("after PreDequantize: got %v, want [1, 3]", out)
	}
}

func TestQMatVecMul_Q8_0(t *testing.T) {
	cols := 32
	rows := 2
	// Create Q8_0 data: block = 2 bytes (f16 scale) + 32 bytes (int8 quants) = 34 bytes
	bytesPerRow := quant.BytesForN(8, cols)
	data := make([]byte, rows*bytesPerRow)

	// Row 0: scale=1.0, all quants=1 → values are all 1.0/127 ≈ 0.00787
	// Row 1: scale=2.0, all quants=1
	for r := 0; r < rows; r++ {
		base := r * bytesPerRow
		scale := float32(1.0) * float32(r+1)
		// Write F16 scale (simplified: use the mantissa bits that matter)
		bits := math.Float32bits(scale)
		f16 := uint16((bits >> 16) & 0x8000) // sign
		exp := int((bits>>23)&0xFF) - 127 + 15
		if exp > 0 && exp < 31 {
			f16 |= uint16(exp) << 10
			f16 |= uint16((bits >> 13) & 0x3FF)
		}
		binary.LittleEndian.PutUint16(data[base:], f16)
		for i := 0; i < cols; i++ {
			data[base+2+i] = 1 // int8 value = 1
		}
	}

	qt := &core.QuantizedTensor{Data: data, Type: 8, Rows: rows, Cols: cols}
	x := make([]float32, cols)
	for i := range x {
		x[i] = 1.0
	}
	out := make([]float32, rows)
	QMatVecMul(out, qt, x)

	// Verify both rows produce non-zero results
	if out[0] == 0 {
		t.Error("Q8_0 row 0 dot product should be non-zero")
	}
	if out[1] == 0 {
		t.Error("Q8_0 row 1 dot product should be non-zero")
	}
	// Row 1 should be ~2x row 0 (scale difference)
	ratio := out[1] / out[0]
	if math.Abs(float64(ratio)-2.0) > 0.1 {
		t.Errorf("Q8_0 row ratio = %f, want ~2.0", ratio)
	}
}
