package ops

import (
	"math"
	"testing"
)

func approxEq(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func TestSwiGLU(t *testing.T) {
	gate := []float32{0, 1, -1, 2}
	up := []float32{1, 1, 1, 1}
	out := make([]float32, 4)
	SwiGLU(out, gate, up, 4)

	for i := range out {
		want := siluScalar(gate[i]) * up[i]
		if !approxEq(out[i], want, 1e-5) {
			t.Errorf("SwiGLU[%d] = %f, want %f", i, out[i], want)
		}
	}
}

func TestGeGLU(t *testing.T) {
	gate := []float32{0, 1, -1, 2}
	up := []float32{1, 1, 1, 1}
	out := make([]float32, 4)
	GeGLU(out, gate, up, 4)

	for i := range out {
		want := geluScalar(gate[i]) * up[i]
		if !approxEq(out[i], want, 1e-5) {
			t.Errorf("GeGLU[%d] = %f, want %f", i, out[i], want)
		}
	}
}

func TestSwiGLUZero(t *testing.T) {
	out := make([]float32, 3)
	SwiGLU(out, []float32{0, 0, 0}, []float32{5, 5, 5}, 3)
	// SiLU(0) = 0, so out should be 0
	for i, v := range out {
		if !approxEq(v, 0, 1e-5) {
			t.Errorf("SwiGLU(0)[%d] = %f, want 0", i, v)
		}
	}
}

func TestLeakyReLU(t *testing.T) {
	x := []float32{-2, -1, 0, 1, 2}
	LeakyReLU(x, 0.01)
	want := []float32{-0.02, -0.01, 0, 1, 2}
	for i := range x {
		if !approxEq(x[i], want[i], 1e-5) {
			t.Errorf("LeakyReLU[%d] = %f, want %f", i, x[i], want[i])
		}
	}
}

func TestELU(t *testing.T) {
	x := []float32{-1, 0, 1}
	ELU(x, 1.0)
	want0 := float32(math.Exp(-1) - 1)
	if !approxEq(x[0], want0, 1e-5) {
		t.Errorf("ELU(-1) = %f, want %f", x[0], want0)
	}
	if x[1] != 0 {
		t.Errorf("ELU(0) = %f, want 0", x[1])
	}
	if x[2] != 1 {
		t.Errorf("ELU(1) = %f, want 1", x[2])
	}
}

func TestMish(t *testing.T) {
	x := []float32{0, 1, -1}
	Mish(x)
	// mish(0) = 0 * tanh(ln(2)) ≈ 0
	if !approxEq(x[0], 0, 1e-5) {
		t.Errorf("Mish(0) = %f, want 0", x[0])
	}
	// mish(1) = 1 * tanh(softplus(1))
	sp := math.Log(1 + math.Exp(1))
	want := float32(1 * math.Tanh(sp))
	if !approxEq(x[1], want, 1e-4) {
		t.Errorf("Mish(1) = %f, want %f", x[1], want)
	}
}

func TestTanhExact(t *testing.T) {
	x := []float32{0, 1, -1, 0.5}
	TanhExact(x)
	for i, v := range x {
		orig := []float64{0, 1, -1, 0.5}
		want := float32(math.Tanh(orig[i]))
		if !approxEq(v, want, 1e-6) {
			t.Errorf("TanhExact[%d] = %f, want %f", i, v, want)
		}
	}
}

func TestSigmoidExact(t *testing.T) {
	x := []float32{0, 10, -10}
	SigmoidExact(x)
	if !approxEq(x[0], 0.5, 1e-6) {
		t.Errorf("Sigmoid(0) = %f, want 0.5", x[0])
	}
	if x[1] < 0.999 {
		t.Errorf("Sigmoid(10) = %f, want ~1", x[1])
	}
	if x[2] > 0.001 {
		t.Errorf("Sigmoid(-10) = %f, want ~0", x[2])
	}
}

func TestClamp(t *testing.T) {
	x := []float32{-5, 0.5, 10}
	Clamp(x, 0, 1)
	want := []float32{0, 0.5, 1}
	for i := range x {
		if x[i] != want[i] {
			t.Errorf("Clamp[%d] = %f, want %f", i, x[i], want[i])
		}
	}
}

func TestHardSigmoid(t *testing.T) {
	x := []float32{-5, 0, 5}
	HardSigmoid(x)
	// (-5+3)/6 < 0 => 0, (0+3)/6 = 0.5, (5+3)/6 > 1 => 1
	want := []float32{0, 0.5, 1}
	for i := range x {
		if !approxEq(x[i], want[i], 1e-5) {
			t.Errorf("HardSigmoid[%d] = %f, want %f", i, x[i], want[i])
		}
	}
}

func TestHardSwish(t *testing.T) {
	x := []float32{-5, 0, 5}
	xOrig := []float32{-5, 0, 5}
	HardSwish(x)
	// -5 * 0 = 0, 0 * 0.5 = 0, 5 * 1 = 5
	want := []float32{0, 0, 5}
	for i := range x {
		_ = xOrig[i]
		if !approxEq(x[i], want[i], 1e-5) {
			t.Errorf("HardSwish[%d] = %f, want %f", i, x[i], want[i])
		}
	}
}
