// Package ops provides optimized mathematical operations for neural network inference.
//
// All functions operate on flat float32 slices. Functions with "InPlace" in the name
// modify their input directly. Functions requiring output buffers take them as the
// first parameter (dst convention).
package ops

import "math"

// LayerNorm applies standard Layer Normalization.
//   out[i] = w[i] * (x[i] - mean) / sqrt(var + eps) + b[i]
func LayerNorm(out, x, w, b []float32, eps float32) {
	n := len(x)
	var sum float32
	i := 0
	for ; i <= n-4; i += 4 {
		sum += x[i] + x[i+1] + x[i+2] + x[i+3]
	}
	for ; i < n; i++ {
		sum += x[i]
	}
	mean := sum / float32(n)

	var varSum float32
	i = 0
	for ; i <= n-4; i += 4 {
		d0 := x[i] - mean
		d1 := x[i+1] - mean
		d2 := x[i+2] - mean
		d3 := x[i+3] - mean
		varSum += d0*d0 + d1*d1 + d2*d2 + d3*d3
	}
	for ; i < n; i++ {
		d := x[i] - mean
		varSum += d * d
	}
	invStd := float32(1.0 / math.Sqrt(float64(varSum/float32(n)+eps)))

	i = 0
	for ; i <= n-4; i += 4 {
		out[i] = w[i]*(x[i]-mean)*invStd + b[i]
		out[i+1] = w[i+1]*(x[i+1]-mean)*invStd + b[i+1]
		out[i+2] = w[i+2]*(x[i+2]-mean)*invStd + b[i+2]
		out[i+3] = w[i+3]*(x[i+3]-mean)*invStd + b[i+3]
	}
	for ; i < n; i++ {
		out[i] = w[i]*(x[i]-mean)*invStd + b[i]
	}
}

// ResidualLayerNorm fuses residual addition and LayerNorm:
//   residual[i] = prev[i] + scale*sub[i]
//   outNorm = LayerNorm(residual)
func ResidualLayerNorm(outNorm, residual, prev, sub, w, b []float32, scale, eps float32) {
	n := len(prev)
	for i := 0; i < n; i++ {
		residual[i] = prev[i] + scale*sub[i]
	}
	LayerNorm(outNorm, residual, w, b, eps)
}

// Sigmoid computes 1 / (1 + exp(-x)).
func Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + FastExp(-x))
}

// FastExp computes a fast approximation of exp(x) in float32 using Schraudolph's
// algorithm. ~0.3% max error, much faster than math.Exp(float64(x)).
func FastExp(x float32) float32 {
	if x < -87.3 {
		return 0
	}
	if x > 88.7 {
		return math.MaxFloat32
	}
	i := int32(12102203.0*x) + 1064866805
	return math.Float32frombits(uint32(i))
}

// FastTanh computes a fast approximation of tanh(x) = (exp(2x) - 1) / (exp(2x) + 1).
func FastTanh(x float32) float32 {
	e2x := FastExp(2.0 * x)
	return (e2x - 1.0) / (e2x + 1.0)
}

// SiLU applies SiLU (Swish) activation in-place: x[i] = x[i] * sigmoid(x[i]).
func SiLU(x []float32) {
	for i := range x {
		v := x[i]
		x[i] = v * Sigmoid(v)
	}
}

// ReLU applies ReLU activation in-place: x[i] = max(0, x[i]).
func ReLU(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

// GELU applies Gaussian Error Linear Unit activation in-place.
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func GELU(x []float32) {
	const c = 0.7978845608028654 // sqrt(2/pi)
	for i := range x {
		v := float64(x[i])
		x[i] = float32(0.5 * v * (1.0 + math.Tanh(c*(v+0.044715*v*v*v))))
	}
}

// GLU applies Gated Linear Unit. Input has size 2*dim; it is split into two
// halves and computes first_half * sigmoid(second_half).
// out must have length >= dim.
func GLU(out, input []float32, dim int) {
	for i := 0; i < dim; i++ {
		out[i] = input[i] * Sigmoid(input[dim+i])
	}
}

// fastExpf is a fast float32 exp approximation using the same algorithm as
// the AVX2 version (range reduction + 4th-order polynomial), but scalar.
// ~5 decimal digits accuracy — sufficient for softmax.
func fastExpf(x float32) float32 {
	if x < -88.0 {
		return 0
	}
	if x > 88.0 {
		x = 88.0
	}
	const log2e = 1.44269504088896
	const ln2 = 0.6931471805599453

	n := float32(math.RoundToEven(float64(x * log2e)))
	r := x - n*ln2

	p := float32(1.0/24.0)*r + float32(1.0/6.0)
	p = p*r + 0.5
	p = p*r + 1.0
	p = p*r + 1.0

	// 2^n via bit manipulation
	ni := int32(n) + 127
	pow2n := math.Float32frombits(uint32(ni) << 23)
	return p * pow2n
}

// Softmax computes softmax in-place with numerical stability.
func Softmax(x []float32) {
	n := len(x)
	if n == 0 {
		return
	}
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	var sum float32
	for i := 0; i < n; i++ {
		x[i] = fastExpf(x[i] - maxVal)
		sum += x[i]
	}
	invSum := 1.0 / sum
	for i := 0; i < n; i++ {
		x[i] *= invSum
	}
}

// LogSoftmax computes numerically stable log-softmax in-place.
func LogSoftmax(x []float32) {
	n := len(x)
	if n == 0 {
		return
	}
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	var sumExp float64
	for i := range x {
		sumExp += math.Exp(float64(x[i] - maxVal))
	}
	logDen := float32(math.Log(sumExp)) + maxVal
	for i := range x {
		x[i] -= logDen
	}
}

// DotProduct computes the dot product of two float32 slices with 8x loop unrolling.
func DotProduct(a, b []float32, n int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	limit := n - 7
	for ; i < limit; i += 8 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	sum := s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// Add performs element-wise addition: out[i] = a[i] + b[i].
func Add(out, a, b []float32) {
	n := len(out)
	i := 0
	for ; i <= n-4; i += 4 {
		out[i] = a[i] + b[i]
		out[i+1] = a[i+1] + b[i+1]
		out[i+2] = a[i+2] + b[i+2]
		out[i+3] = a[i+3] + b[i+3]
	}
	for ; i < n; i++ {
		out[i] = a[i] + b[i]
	}
}

// AddBias adds a bias vector to dst in-place: dst[i] += bias[i].
func AddBias(dst, bias []float32) {
	n := len(bias)
	i := 0
	for ; i <= n-4; i += 4 {
		dst[i] += bias[i]
		dst[i+1] += bias[i+1]
		dst[i+2] += bias[i+2]
		dst[i+3] += bias[i+3]
	}
	for ; i < n; i++ {
		dst[i] += bias[i]
	}
}

// AddScaled computes dst[i] += scale * src[i] with 8x unrolling.
func AddScaled(dst []float32, scale float32, src []float32, n int) {
	i := 0
	for ; i <= n-8; i += 8 {
		dst[i] += scale * src[i]
		dst[i+1] += scale * src[i+1]
		dst[i+2] += scale * src[i+2]
		dst[i+3] += scale * src[i+3]
		dst[i+4] += scale * src[i+4]
		dst[i+5] += scale * src[i+5]
		dst[i+6] += scale * src[i+6]
		dst[i+7] += scale * src[i+7]
	}
	for ; i < n; i++ {
		dst[i] += scale * src[i]
	}
}

// Scale multiplies all elements in x by s in-place.
func Scale(x []float32, s float32) {
	for i := range x {
		x[i] *= s
	}
}

// Argmax returns the index of the maximum value. Returns -1 for empty slices.
func Argmax(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	maxIdx := 0
	maxVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxVal {
			maxVal = x[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// TopKIndices returns the indices of the k largest values. Uses simple selection
// for small k; for production use with large k, a heap-based variant is preferred.
func TopKIndices(vals []float32, k int) []int {
	if k <= 0 || len(vals) == 0 {
		return nil
	}
	if k > len(vals) {
		k = len(vals)
	}
	out := make([]int, 0, k)
	used := make([]bool, len(vals))
	for len(out) < k {
		best := -1
		bestV := float32(math.Inf(-1))
		for i, v := range vals {
			if used[i] {
				continue
			}
			if best < 0 || v > bestV {
				best = i
				bestV = v
			}
		}
		if best < 0 {
			break
		}
		used[best] = true
		out = append(out, best)
	}
	return out
}

// Clear zeros out a float32 slice.
func Clear(s []float32) {
	for i := range s {
		s[i] = 0
	}
}

// MatVecMul performs float32 matrix-vector multiply: out[r] = dot(W[r,:], x).
// W is stored row-major as [outDim * inDim].
func MatVecMul(out, W, x []float32, outDim, inDim int) {
	for r := 0; r < outDim; r++ {
		out[r] = DotProduct(W[r*inDim:(r+1)*inDim], x, inDim)
	}
}
