package silero

import "math"

// STFT applies the learned Short-Time Fourier Transform.
// Input: 512 raw audio samples (one chunk).
// Output: [129][4] magnitude spectrogram (129 frequency bins, 4 time steps).
//
// The STFT basis is a precomputed DFT matrix stored as Conv1D weights.
// 256-point DFT → 129 unique bins (N/2+1) × 2 (real+imag) = 258 output channels.
func applySTFT(chunk []float32, basis []float32) [][]float32 {
	const (
		padSize   = 64
		kernel    = 256
		stride    = 128
		outCh     = 258
		cutoff    = 129 // outCh / 2
		paddedLen = 640 // 512 + 2*64
	)

	// 1. Reflect-pad by 64 on each side → 640 samples
	padded := make([]float32, paddedLen)

	// Left reflection: padded[0..63] = chunk[64..1]
	for i := 0; i < padSize; i++ {
		padded[i] = chunk[padSize-i] // i=0→chunk[64], i=63→chunk[1]
	}
	// Center: original signal
	copy(padded[padSize:padSize+512], chunk)
	// Right reflection: padded[576..639] = chunk[510..447]
	for i := 0; i < padSize; i++ {
		padded[padSize+512+i] = chunk[510-i] // i=0→chunk[510], i=63→chunk[447]
	}

	// 2. Conv1D: 258 output channels, kernel=256, stride=128, padding=0
	// Output length = (640 - 256) / 128 + 1 = 4
	outLen := (paddedLen - kernel) / stride + 1 // = 4

	stftOut := make([][]float32, outCh)
	for oc := 0; oc < outCh; oc++ {
		stftOut[oc] = make([]float32, outLen)
		for t := 0; t < outLen; t++ {
			var sum float32
			offset := t * stride
			basisOffset := oc * kernel
			for k := 0; k < kernel; k++ {
				sum += basis[basisOffset+k] * padded[offset+k]
			}
			stftOut[oc][t] = sum
		}
	}

	// 3. Magnitude: sqrt(real² + imag²)
	// Channels 0..128 = real, channels 129..257 = imaginary
	magnitude := make([][]float32, cutoff)
	for freq := 0; freq < cutoff; freq++ {
		magnitude[freq] = make([]float32, outLen)
		for t := 0; t < outLen; t++ {
			r := stftOut[freq][t]
			im := stftOut[cutoff+freq][t]
			magnitude[freq][t] = float32(math.Sqrt(float64(r*r + im*im)))
		}
	}

	return magnitude // [129][4]
}
