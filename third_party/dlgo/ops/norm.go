package ops

import "math"

// RMSNorm applies Root Mean Square Layer Normalization.
//   out[i] = w[i] * x[i] / sqrt(mean(x²) + eps)
//
// Used by LLaMA, Qwen, Gemma, Mistral, and most modern LLMs instead of LayerNorm.
// Simpler and faster than LayerNorm: no mean subtraction, no bias term.
func RMSNorm(out, x, w []float32, eps float32) {
	n := len(x)
	var ss float32
	i := 0
	for ; i <= n-4; i += 4 {
		ss += x[i]*x[i] + x[i+1]*x[i+1] + x[i+2]*x[i+2] + x[i+3]*x[i+3]
	}
	for ; i < n; i++ {
		ss += x[i] * x[i]
	}
	scale := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))

	i = 0
	for ; i <= n-4; i += 4 {
		out[i] = x[i] * scale * w[i]
		out[i+1] = x[i+1] * scale * w[i+1]
		out[i+2] = x[i+2] * scale * w[i+2]
		out[i+3] = x[i+3] * scale * w[i+3]
	}
	for ; i < n; i++ {
		out[i] = x[i] * scale * w[i]
	}
}

// RMSNormInPlace applies RMS normalization in-place (modifies x).
// Used for QK-norm in models like Qwen3.
func RMSNormInPlace(x, w []float32, eps float32) {
	n := len(x)
	var ss float32
	for i := 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	scale := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	for i := 0; i < n; i++ {
		x[i] = x[i] * scale * w[i]
	}
}

// GroupNorm applies Group Normalization (used in diffusion models, U-Nets).
//   x: [numGroups * groupSize] flat
//   w, b: [numGroups * groupSize] (affine parameters)
//   out: [numGroups * groupSize]
//
// Each group of channels is independently normalized.
func GroupNorm(out, x, w, b []float32, numGroups, groupSize int, eps float32) {
	for g := 0; g < numGroups; g++ {
		base := g * groupSize

		var sum float32
		for i := 0; i < groupSize; i++ {
			sum += x[base+i]
		}
		mean := sum / float32(groupSize)

		var varSum float32
		for i := 0; i < groupSize; i++ {
			d := x[base+i] - mean
			varSum += d * d
		}
		invStd := float32(1.0 / math.Sqrt(float64(varSum/float32(groupSize)+eps)))

		for i := 0; i < groupSize; i++ {
			idx := base + i
			out[idx] = w[idx]*(x[idx]-mean)*invStd + b[idx]
		}
	}
}

// BatchNormInference applies Batch Normalization in inference mode.
// Uses precomputed running mean and variance (no batch statistics).
//   x, out: [channels] (or [channels * spatial] if applied per-channel)
//   gamma, beta: [channels] (learned scale/shift)
//   runMean, runVar: [channels] (running statistics from training)
func BatchNormInference(out, x, gamma, beta, runMean, runVar []float32, channels int, eps float32) {
	for c := 0; c < channels; c++ {
		invStd := float32(1.0 / math.Sqrt(float64(runVar[c]+eps)))
		out[c] = gamma[c]*(x[c]-runMean[c])*invStd + beta[c]
	}
}
