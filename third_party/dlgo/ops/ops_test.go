package ops

import (
	"math"
	"testing"
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) < tol }

func TestLayerNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	b := []float32{0, 0, 0, 0}
	out := make([]float32, 4)
	LayerNorm(out, x, w, b, 1e-5)

	var sum float32
	for _, v := range out {
		sum += v
	}
	if !approx(float64(sum), 0, 1e-4) {
		t.Errorf("LayerNorm output mean should be ~0, got sum=%f", sum)
	}
}

func TestResidualLayerNorm(t *testing.T) {
	prev := []float32{1, 2, 3, 4}
	sub := []float32{0.1, 0.2, 0.3, 0.4}
	w := []float32{1, 1, 1, 1}
	b := []float32{0, 0, 0, 0}
	outNorm := make([]float32, 4)
	residual := make([]float32, 4)
	ResidualLayerNorm(outNorm, residual, prev, sub, w, b, 0.5, 1e-5)

	if !approx(float64(residual[0]), 1.05, 1e-4) {
		t.Errorf("residual[0] = %f, want 1.05", residual[0])
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float32
		expected float32
		tol      float64
	}{
		{0.0, 0.5, 0.02},    // FastExp approximation: ~3% error at 0
		{100.0, 1.0, 1e-3},
		{-100.0, 0.0, 1e-3},
		{5.0, 0.9933, 0.02},
		{-5.0, 0.0067, 0.02},
	}
	for _, tt := range tests {
		got := Sigmoid(tt.input)
		if !approx(float64(got), float64(tt.expected), tt.tol) {
			t.Errorf("Sigmoid(%f) = %f, want %f", tt.input, got, tt.expected)
		}
	}
}

func TestFastExp(t *testing.T) {
	// Schraudolph's algorithm has ~3% max error; verify it's in ballpark
	for _, x := range []float32{-5, -1, 0, 1, 5} {
		want := float32(math.Exp(float64(x)))
		got := FastExp(x)
		relErr := math.Abs(float64(got-want)) / math.Max(float64(want), 1e-10)
		if relErr > 0.04 {
			t.Errorf("FastExp(%f) = %f, want ~%f (relErr=%f)", x, got, want, relErr)
		}
	}
	// Edge cases
	if FastExp(-100) != 0 {
		t.Errorf("FastExp(-100) should be 0")
	}
	if FastExp(100) != math.MaxFloat32 {
		t.Errorf("FastExp(100) should be MaxFloat32")
	}
}

func TestSiLU(t *testing.T) {
	x := []float32{-2, -1, 0, 1, 2}
	SiLU(x)
	if !approx(float64(x[2]), 0, 1e-6) {
		t.Errorf("SiLU(0) = %f, want 0", x[2])
	}
	if x[3] < 0.5 || x[3] > 1.0 {
		t.Errorf("SiLU(1) = %f, want ~0.73", x[3])
	}
}

func TestReLU(t *testing.T) {
	x := []float32{-5, -1, 0, 1, 5}
	ReLU(x)
	if x[0] != 0 || x[1] != 0 || x[2] != 0 || x[3] != 1 || x[4] != 5 {
		t.Errorf("ReLU unexpected: %v", x)
	}
}

func TestGELU(t *testing.T) {
	x := []float32{0, 1, -1}
	GELU(x)
	if !approx(float64(x[0]), 0, 1e-6) {
		t.Errorf("GELU(0) = %f, want 0", x[0])
	}
	if !approx(float64(x[1]), 0.8413, 1e-3) {
		t.Errorf("GELU(1) = %f, want ~0.8413", x[1])
	}
}

func TestGLU(t *testing.T) {
	input := []float32{1, 2, 3, 0, 0, 0} // first half [1,2,3], gate [0,0,0]
	out := make([]float32, 3)
	GLU(out, input, 3)
	// sigmoid(0) ≈ 0.5 (with FastExp approx), so out ≈ [0.5, 1.0, 1.5]
	for i, want := range []float32{0.5, 1.0, 1.5} {
		if !approx(float64(out[i]), float64(want), 0.1) {
			t.Errorf("GLU out[%d] = %f, want %f", i, out[i], want)
		}
	}
}

func TestSoftmax(t *testing.T) {
	x := []float32{1, 2, 3}
	Softmax(x)
	var sum float32
	for _, v := range x {
		sum += v
	}
	if !approx(float64(sum), 1.0, 1e-5) {
		t.Errorf("Softmax sum = %f, want 1.0", sum)
	}
	if x[2] < x[1] || x[1] < x[0] {
		t.Errorf("Softmax order wrong: %v", x)
	}
}

func TestLogSoftmax(t *testing.T) {
	x := []float32{1, 2, 3}
	LogSoftmax(x)
	for _, v := range x {
		if v > 0 {
			t.Errorf("LogSoftmax should be <= 0, got %f", v)
		}
	}
	var sumExp float64
	for _, v := range x {
		sumExp += math.Exp(float64(v))
	}
	if !approx(sumExp, 1.0, 1e-5) {
		t.Errorf("exp(LogSoftmax) sum = %f, want 1.0", sumExp)
	}
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	b := []float32{9, 8, 7, 6, 5, 4, 3, 2, 1}
	got := DotProduct(a, b, 9)
	want := float32(9+16+21+24+25+24+21+16+9)
	if !approx(float64(got), float64(want), 1e-4) {
		t.Errorf("DotProduct = %f, want %f", got, want)
	}
}

func TestAdd(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{10, 20, 30, 40, 50}
	out := make([]float32, 5)
	Add(out, a, b)
	for i := range out {
		if out[i] != a[i]+b[i] {
			t.Errorf("Add[%d] = %f, want %f", i, out[i], a[i]+b[i])
		}
	}
}

func TestArgmax(t *testing.T) {
	x := []float32{1, 5, 3, 2}
	if got := Argmax(x); got != 1 {
		t.Errorf("Argmax = %d, want 1", got)
	}
	if got := Argmax(nil); got != -1 {
		t.Errorf("Argmax(nil) = %d, want -1", got)
	}
}

func TestTopKIndices(t *testing.T) {
	vals := []float32{1, 5, 3, 7, 2}
	got := TopKIndices(vals, 3)
	if len(got) != 3 || got[0] != 3 || got[1] != 1 {
		t.Errorf("TopKIndices = %v, want [3,1,2]", got)
	}
}

func TestMatVecMul(t *testing.T) {
	W := []float32{1, 0, 0, 1} // 2x2 identity
	x := []float32{3, 7}
	out := make([]float32, 2)
	MatVecMul(out, W, x, 2, 2)
	if out[0] != 3 || out[1] != 7 {
		t.Errorf("MatVecMul identity: got %v, want [3,7]", out)
	}
}

func TestFastTanh(t *testing.T) {
	// FastTanh uses FastExp internally, so inherits its ~3% error
	for _, x := range []float32{-2, -1, 0, 1, 2} {
		want := float32(math.Tanh(float64(x)))
		got := FastTanh(x)
		if !approx(float64(got), float64(want), 0.06) {
			t.Errorf("FastTanh(%f) = %f, want ~%f", x, got, want)
		}
	}
}

func TestClear(t *testing.T) {
	s := []float32{1, 2, 3}
	Clear(s)
	for i, v := range s {
		if v != 0 {
			t.Errorf("Clear: s[%d] = %f, want 0", i, v)
		}
	}
}

func TestScale(t *testing.T) {
	x := []float32{1, 2, 3}
	Scale(x, 2.0)
	if x[0] != 2 || x[1] != 4 || x[2] != 6 {
		t.Errorf("Scale: got %v, want [2,4,6]", x)
	}
}
