package ops

import "math"

// SwiGLU computes the SwiGLU gated activation used by LLaMA, Qwen, Mistral.
//   out[i] = SiLU(gate[i]) * up[i]
// gate and up are two parallel FFN projections; SwiGLU fuses the activation.
func SwiGLU(out, gate, up []float32, n int) {
	i := 0
	for ; i <= n-4; i += 4 {
		out[i] = siluScalar(gate[i]) * up[i]
		out[i+1] = siluScalar(gate[i+1]) * up[i+1]
		out[i+2] = siluScalar(gate[i+2]) * up[i+2]
		out[i+3] = siluScalar(gate[i+3]) * up[i+3]
	}
	for ; i < n; i++ {
		out[i] = siluScalar(gate[i]) * up[i]
	}
}

// GeGLU computes the GELU-gated activation used by Gemma.
//   out[i] = GELU(gate[i]) * up[i]
func GeGLU(out, gate, up []float32, n int) {
	i := 0
	for ; i <= n-4; i += 4 {
		out[i] = geluScalar(gate[i]) * up[i]
		out[i+1] = geluScalar(gate[i+1]) * up[i+1]
		out[i+2] = geluScalar(gate[i+2]) * up[i+2]
		out[i+3] = geluScalar(gate[i+3]) * up[i+3]
	}
	for ; i < n; i++ {
		out[i] = geluScalar(gate[i]) * up[i]
	}
}

func siluScalar(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}

func geluScalar(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	v := float64(x)
	return float32(0.5 * v * (1.0 + math.Tanh(c*(v+0.044715*v*v*v))))
}

// LeakyReLU applies Leaky ReLU in-place: x[i] = max(alpha*x[i], x[i]).
// Commonly used in GANs and some discriminator networks.
func LeakyReLU(x []float32, alpha float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] *= alpha
		}
	}
}

// ELU applies Exponential Linear Unit in-place.
//   x[i] = x[i]              if x[i] >= 0
//   x[i] = alpha*(exp(x)-1)  if x[i] < 0
func ELU(x []float32, alpha float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = alpha * (float32(math.Exp(float64(x[i]))) - 1)
		}
	}
}

// Mish applies Mish activation in-place: x * tanh(softplus(x)).
// Used in YOLOv4 and some modern architectures.
func Mish(x []float32) {
	for i := range x {
		sp := float64(x[i])
		// softplus(x) = ln(1 + exp(x))
		if sp < 20 { // avoid overflow in exp
			sp = math.Log(1 + math.Exp(sp))
		}
		x[i] = x[i] * float32(math.Tanh(sp))
	}
}

// TanhExact applies exact tanh via math.Tanh (for when precision matters more than speed).
func TanhExact(x []float32) {
	for i := range x {
		x[i] = float32(math.Tanh(float64(x[i])))
	}
}

// SigmoidExact applies exact sigmoid via math.Exp (for when precision matters more than speed).
func SigmoidExact(x []float32) {
	for i := range x {
		x[i] = float32(1.0 / (1.0 + math.Exp(float64(-x[i]))))
	}
}

// Clamp clamps all values in x to the range [minVal, maxVal] in-place.
func Clamp(x []float32, minVal, maxVal float32) {
	for i := range x {
		if x[i] < minVal {
			x[i] = minVal
		} else if x[i] > maxVal {
			x[i] = maxVal
		}
	}
}

// HardSigmoid applies hard sigmoid in-place: clamp((x + 3) / 6, 0, 1).
// Fast approximation used in MobileNetV3.
func HardSigmoid(x []float32) {
	for i := range x {
		v := (x[i] + 3.0) / 6.0
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		x[i] = v
	}
}

// HardSwish applies hard swish in-place: x * hard_sigmoid(x).
// Used in MobileNetV3 and EfficientNet.
func HardSwish(x []float32) {
	for i := range x {
		v := (x[i] + 3.0) / 6.0
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		x[i] *= v
	}
}
