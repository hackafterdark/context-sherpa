package ops

import (
	"math"
	"testing"
)

func TestMul(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	out := make([]float32, 4)
	Mul(out, a, b)
	want := []float32{5, 12, 21, 32}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("Mul[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestSub(t *testing.T) {
	a := []float32{5, 6, 7, 8}
	b := []float32{1, 2, 3, 4}
	out := make([]float32, 4)
	Sub(out, a, b)
	want := []float32{4, 4, 4, 4}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("Sub[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestMax(t *testing.T) {
	a := []float32{1, 5, 3}
	b := []float32{4, 2, 3}
	out := make([]float32, 3)
	Max(out, a, b)
	want := []float32{4, 5, 3}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("Max[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestMin(t *testing.T) {
	a := []float32{1, 5, 3}
	b := []float32{4, 2, 3}
	out := make([]float32, 3)
	Min(out, a, b)
	want := []float32{1, 2, 3}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("Min[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestScalarMulAdd(t *testing.T) {
	a := []float32{1, 2, 3}
	out := make([]float32, 3)
	ScalarMul(out, a, 3)
	for i, v := range out {
		if v != a[i]*3 {
			t.Errorf("ScalarMul[%d] = %f, want %f", i, v, a[i]*3)
		}
	}

	ScalarAdd(out, a, 10)
	for i, v := range out {
		if v != a[i]+10 {
			t.Errorf("ScalarAdd[%d] = %f, want %f", i, v, a[i]+10)
		}
	}
}

func TestPow(t *testing.T) {
	a := []float32{2, 3, 4}
	out := make([]float32, 3)
	Pow(out, a, 2)
	want := []float32{4, 9, 16}
	for i := range out {
		if math.Abs(float64(out[i]-want[i])) > 1e-5 {
			t.Errorf("Pow[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestSqrt(t *testing.T) {
	a := []float32{4, 9, 16}
	out := make([]float32, 3)
	Sqrt(out, a)
	want := []float32{2, 3, 4}
	for i := range out {
		if math.Abs(float64(out[i]-want[i])) > 1e-5 {
			t.Errorf("Sqrt[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestAbs(t *testing.T) {
	a := []float32{-3, 0, 5}
	out := make([]float32, 3)
	Abs(out, a)
	want := []float32{3, 0, 5}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("Abs[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestNeg(t *testing.T) {
	a := []float32{-3, 0, 5}
	out := make([]float32, 3)
	Neg(out, a)
	want := []float32{3, 0, -5}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("Neg[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestReciprocal(t *testing.T) {
	a := []float32{2, 4, 0.5}
	out := make([]float32, 3)
	Reciprocal(out, a)
	want := []float32{0.5, 0.25, 2}
	for i := range out {
		if math.Abs(float64(out[i]-want[i])) > 1e-5 {
			t.Errorf("Reciprocal[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestReduceOps(t *testing.T) {
	a := []float32{1, 5, 3, -1}

	if s := ReduceSum(a); s != 8 {
		t.Errorf("ReduceSum = %f, want 8", s)
	}
	if m := ReduceMax(a); m != 5 {
		t.Errorf("ReduceMax = %f, want 5", m)
	}
	if m := ReduceMin(a); m != -1 {
		t.Errorf("ReduceMin = %f, want -1", m)
	}
	if m := ReduceMean(a); m != 2 {
		t.Errorf("ReduceMean = %f, want 2", m)
	}
}

func TestWhere(t *testing.T) {
	cond := []bool{true, false, true}
	trueV := []float32{1, 2, 3}
	falseV := []float32{4, 5, 6}
	out := make([]float32, 3)
	Where(out, cond, trueV, falseV)
	want := []float32{1, 5, 3}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("Where[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestConcat(t *testing.T) {
	a := []float32{1, 2}
	b := []float32{3, 4, 5}
	c := Concat(a, b)
	if len(c) != 5 {
		t.Fatalf("Concat len = %d, want 5", len(c))
	}
	want := []float32{1, 2, 3, 4, 5}
	for i := range c {
		if c[i] != want[i] {
			t.Errorf("Concat[%d] = %f, want %f", i, c[i], want[i])
		}
	}
}
