package ops

import (
	"math"
	"testing"
)

func TestApplyRoPEInterleaved(t *testing.T) {
	headDim := 4
	vec := []float32{1, 0, 0, 0}
	ApplyRoPE(vec, 0, headDim, 10000.0, false)

	// At pos=0, all angles are 0 => cos=1, sin=0 => no change
	want := []float32{1, 0, 0, 0}
	for i, v := range vec {
		if math.Abs(float64(v-want[i])) > 1e-5 {
			t.Errorf("pos0: vec[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestApplyRoPENeoX(t *testing.T) {
	headDim := 4
	vec := []float32{1, 0, 0, 0}
	ApplyRoPE(vec, 0, headDim, 10000.0, true)

	// At pos=0, all angles are 0 => cos=1, sin=0 => no change
	want := []float32{1, 0, 0, 0}
	for i, v := range vec {
		if math.Abs(float64(v-want[i])) > 1e-5 {
			t.Errorf("pos0 neox: vec[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestRoPERotation(t *testing.T) {
	headDim := 2
	// With headDim=2 and freqBase=1.0, theta=1 at pos=1 => angle=1 radian
	vec := []float32{1, 0}
	ApplyRoPE(vec, 1, headDim, 1.0, false)

	wantX := float32(math.Cos(1))
	wantY := float32(math.Sin(1))
	if math.Abs(float64(vec[0]-wantX)) > 1e-5 || math.Abs(float64(vec[1]-wantY)) > 1e-5 {
		t.Errorf("rotation: got [%f, %f], want [%f, %f]", vec[0], vec[1], wantX, wantY)
	}
}

func TestRoPEFrequencyTable(t *testing.T) {
	maxLen := 8
	headDim := 4
	cosT, sinT := RoPEFrequencyTable(maxLen, headDim, 10000.0)

	if len(cosT) != maxLen*headDim/2 {
		t.Fatalf("cos table length = %d, want %d", len(cosT), maxLen*headDim/2)
	}

	// Verify position 0 has cos=1, sin=0
	half := headDim / 2
	for i := 0; i < half; i++ {
		if math.Abs(float64(cosT[i]-1)) > 1e-6 {
			t.Errorf("cosT[0*half+%d] = %f, want 1", i, cosT[i])
		}
		if math.Abs(float64(sinT[i])) > 1e-6 {
			t.Errorf("sinT[0*half+%d] = %f, want 0", i, sinT[i])
		}
	}
}

func TestApplyRoPEFromTableMatchesDirect(t *testing.T) {
	headDim := 8
	maxLen := 16
	freqBase := float32(10000.0)
	cosT, sinT := RoPEFrequencyTable(maxLen, headDim, freqBase)

	for pos := 0; pos < maxLen; pos++ {
		// Test interleaved
		v1 := make([]float32, headDim)
		v2 := make([]float32, headDim)
		for i := range v1 {
			v1[i] = float32(i + 1)
			v2[i] = float32(i + 1)
		}
		ApplyRoPE(v1, pos, headDim, freqBase, false)
		ApplyRoPEFromTable(v2, pos, headDim, cosT, sinT, false)
		for i := range v1 {
			if math.Abs(float64(v1[i]-v2[i])) > 1e-5 {
				t.Errorf("interleaved pos=%d dim=%d: direct=%f table=%f", pos, i, v1[i], v2[i])
			}
		}

		// Test neox
		for i := range v1 {
			v1[i] = float32(i + 1)
			v2[i] = float32(i + 1)
		}
		ApplyRoPE(v1, pos, headDim, freqBase, true)
		ApplyRoPEFromTable(v2, pos, headDim, cosT, sinT, true)
		for i := range v1 {
			if math.Abs(float64(v1[i]-v2[i])) > 1e-5 {
				t.Errorf("neox pos=%d dim=%d: direct=%f table=%f", pos, i, v1[i], v2[i])
			}
		}
	}
}

func TestApplyRoPEBatch(t *testing.T) {
	headDim := 4
	numHeads := 2
	numKVHeads := 1

	qFlat := make([]float32, numHeads*headDim)
	kFlat := make([]float32, numKVHeads*headDim)
	for i := range qFlat {
		qFlat[i] = 1
	}
	for i := range kFlat {
		kFlat[i] = 1
	}

	ApplyRoPEBatch(qFlat, numHeads, kFlat, numKVHeads, 0, headDim, 10000.0, false)

	// At pos 0 nothing should change
	for i, v := range qFlat {
		if math.Abs(float64(v-1)) > 1e-5 {
			t.Errorf("qFlat[%d] changed at pos 0: %f", i, v)
		}
	}
}
