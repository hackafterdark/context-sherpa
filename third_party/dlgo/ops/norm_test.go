package ops

import (
	"math"
	"testing"
)

func TestRMSNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	out := make([]float32, 4)
	RMSNorm(out, x, w, 1e-5)

	// RMS = sqrt(mean(x^2)+eps) = sqrt((1+4+9+16)/4) = sqrt(7.5)
	rms := float32(math.Sqrt(7.5 + 1e-5))
	for i, v := range out {
		want := x[i] / rms
		if d := v - want; d > 0.001 || d < -0.001 {
			t.Errorf("RMSNorm[%d] = %f, want %f", i, v, want)
		}
	}
}

func TestRMSNormInPlace(t *testing.T) {
	x := []float32{2, 4, 6}
	w := []float32{0.5, 0.5, 0.5}
	orig := make([]float32, 3)
	copy(orig, x)
	RMSNormInPlace(x, w, 1e-5)

	rms := float32(math.Sqrt(float64((4 + 16 + 36) / 3.0)))
	for i, v := range x {
		want := w[i] * orig[i] / rms
		if d := v - want; d > 0.001 || d < -0.001 {
			t.Errorf("RMSNormInPlace[%d] = %f, want %f", i, v, want)
		}
	}
}

func TestRMSNormWithScaledWeights(t *testing.T) {
	x := []float32{1, 1, 1, 1}
	w := []float32{2, 3, 4, 5}
	out := make([]float32, 4)
	RMSNorm(out, x, w, 1e-5)

	// All x identical => normalized = 1, so out[i] = w[i]
	for i, v := range out {
		if d := v - w[i]; d > 0.01 || d < -0.01 {
			t.Errorf("out[%d] = %f, want %f", i, v, w[i])
		}
	}
}

func TestGroupNorm(t *testing.T) {
	// 2 groups of 2 elements each
	x := []float32{1, 3, 5, 7}
	w := []float32{1, 1, 1, 1}
	b := []float32{0, 0, 0, 0}
	out := make([]float32, 4)
	GroupNorm(out, x, w, b, 2, 2, 1e-5)

	// Group 0: mean=2, var=1 => [-1, 1]
	// Group 1: mean=6, var=1 => [-1, 1]
	if math.Abs(float64(out[0])-(-1)) > 0.01 || math.Abs(float64(out[1])-1) > 0.01 {
		t.Errorf("GroupNorm group 0: got [%f, %f], want [-1, 1]", out[0], out[1])
	}
	if math.Abs(float64(out[2])-(-1)) > 0.01 || math.Abs(float64(out[3])-1) > 0.01 {
		t.Errorf("GroupNorm group 1: got [%f, %f], want [-1, 1]", out[2], out[3])
	}
}

func TestBatchNormInference(t *testing.T) {
	x := []float32{2, 4}
	gamma := []float32{1, 1}
	beta := []float32{0, 0}
	runMean := []float32{0, 0}
	runVar := []float32{1, 1}
	out := make([]float32, 2)
	BatchNormInference(out, x, gamma, beta, runMean, runVar, 2, 1e-5)

	// With zero mean, unit variance: out ≈ x
	for i, v := range out {
		if d := v - x[i]; d > 0.01 || d < -0.01 {
			t.Errorf("BatchNorm[%d] = %f, want %f", i, v, x[i])
		}
	}
}
