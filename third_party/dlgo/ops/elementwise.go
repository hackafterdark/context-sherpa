package ops

import "math"

// Mul performs element-wise multiplication: out[i] = a[i] * b[i].
func Mul(out, a, b []float32) {
	n := len(out)
	i := 0
	for ; i <= n-4; i += 4 {
		out[i] = a[i] * b[i]
		out[i+1] = a[i+1] * b[i+1]
		out[i+2] = a[i+2] * b[i+2]
		out[i+3] = a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		out[i] = a[i] * b[i]
	}
}

// Sub performs element-wise subtraction: out[i] = a[i] - b[i].
func Sub(out, a, b []float32) {
	n := len(out)
	i := 0
	for ; i <= n-4; i += 4 {
		out[i] = a[i] - b[i]
		out[i+1] = a[i+1] - b[i+1]
		out[i+2] = a[i+2] - b[i+2]
		out[i+3] = a[i+3] - b[i+3]
	}
	for ; i < n; i++ {
		out[i] = a[i] - b[i]
	}
}

// Max performs element-wise maximum: out[i] = max(a[i], b[i]).
func Max(out, a, b []float32) {
	for i := range out {
		if a[i] > b[i] {
			out[i] = a[i]
		} else {
			out[i] = b[i]
		}
	}
}

// Min performs element-wise minimum: out[i] = min(a[i], b[i]).
func Min(out, a, b []float32) {
	for i := range out {
		if a[i] < b[i] {
			out[i] = a[i]
		} else {
			out[i] = b[i]
		}
	}
}

// ScalarMul multiplies all elements by a scalar: out[i] = a[i] * s.
func ScalarMul(out, a []float32, s float32) {
	for i := range out {
		out[i] = a[i] * s
	}
}

// ScalarAdd adds a scalar to all elements: out[i] = a[i] + s.
func ScalarAdd(out, a []float32, s float32) {
	for i := range out {
		out[i] = a[i] + s
	}
}

// Pow computes element-wise power: out[i] = a[i] ^ p.
func Pow(out, a []float32, p float32) {
	pf := float64(p)
	for i := range out {
		out[i] = float32(math.Pow(float64(a[i]), pf))
	}
}

// Sqrt computes element-wise square root: out[i] = sqrt(a[i]).
func Sqrt(out, a []float32) {
	for i := range out {
		out[i] = float32(math.Sqrt(float64(a[i])))
	}
}

// Abs computes element-wise absolute value: out[i] = |a[i]|.
func Abs(out, a []float32) {
	for i := range out {
		if a[i] < 0 {
			out[i] = -a[i]
		} else {
			out[i] = a[i]
		}
	}
}

// Neg negates all elements: out[i] = -a[i].
func Neg(out, a []float32) {
	for i := range out {
		out[i] = -a[i]
	}
}

// Reciprocal computes element-wise reciprocal: out[i] = 1 / a[i].
func Reciprocal(out, a []float32) {
	for i := range out {
		out[i] = 1.0 / a[i]
	}
}

// ReduceSum returns the sum of all elements.
func ReduceSum(a []float32) float32 {
	var s float32
	for _, v := range a {
		s += v
	}
	return s
}

// ReduceMax returns the maximum value.
func ReduceMax(a []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// ReduceMin returns the minimum value.
func ReduceMin(a []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	m := a[0]
	for _, v := range a[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

// ReduceMean returns the arithmetic mean.
func ReduceMean(a []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return ReduceSum(a) / float32(len(a))
}

// Where performs element-wise conditional selection:
//   out[i] = trueVal[i]  if cond[i]
//   out[i] = falseVal[i] if !cond[i]
func Where(out []float32, cond []bool, trueVal, falseVal []float32) {
	for i := range out {
		if cond[i] {
			out[i] = trueVal[i]
		} else {
			out[i] = falseVal[i]
		}
	}
}

// Concat concatenates two slices.
func Concat(a, b []float32) []float32 {
	out := make([]float32, len(a)+len(b))
	copy(out, a)
	copy(out[len(a):], b)
	return out
}

// CopySlice copies src into dst (min of both lengths).
func CopySlice(dst, src []float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	copy(dst[:n], src[:n])
}
