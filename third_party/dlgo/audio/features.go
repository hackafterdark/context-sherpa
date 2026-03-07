package audio

import (
	"math"
)

// MelConfig holds parameters for mel spectrogram extraction.
type MelConfig struct {
	NMels       int     // Number of mel bins (e.g. 128 for Parakeet, 80 for Whisper)
	NFFTSize    int     // FFT size (e.g. 512)
	HopLength   int     // Hop length in samples (e.g. 160)
	WindowSize  int     // Analysis window size in samples (e.g. 400)
	SampleRate  int     // Audio sample rate (e.g. 16000)
	PreEmphasis float32 // Pre-emphasis coefficient (e.g. 0.97, 0 = disabled)
}

// DefaultMelConfig returns the default configuration matching NeMo's Parakeet preprocessor.
func DefaultMelConfig() MelConfig {
	return MelConfig{
		NMels:       128,
		NFFTSize:    512,
		HopLength:   160,
		WindowSize:  400,
		SampleRate:  16000,
		PreEmphasis: 0.97,
	}
}

// WhisperMelConfig returns configuration matching Whisper's preprocessor.
func WhisperMelConfig() MelConfig {
	return MelConfig{
		NMels:      80,
		NFFTSize:   400,
		HopLength:  160,
		WindowSize: 400,
		SampleRate: 16000,
	}
}

// ExtractMelFeatures computes a log-mel spectrogram from raw audio samples.
// Matches NeMo's AudioToMelSpectrogramPreprocessor:
//   - Optional pre-emphasis: y[t] = x[t] - alpha*x[t-1]
//   - STFT with center padding (zero-pad mode)
//   - Symmetric Hann window
//   - Slaney mel filterbank with area normalization
//   - log(x + 2^-24) guard
//   - Per-feature normalization with Bessel-corrected std
//
// Returns flat [nMels * nFrames] in row-major order.
func ExtractMelFeatures(samples []float32, cfg MelConfig) []float32 {
	nMels := cfg.NMels
	nFFT := cfg.NFFTSize
	hopLength := cfg.HopLength
	sampleRate := cfg.SampleRate

	// Pre-emphasis
	input := samples
	if cfg.PreEmphasis > 0 {
		preemph := make([]float32, len(samples))
		preemph[0] = samples[0]
		for i := 1; i < len(samples); i++ {
			preemph[i] = samples[i] - cfg.PreEmphasis*samples[i-1]
		}
		input = preemph
	}

	// Hann window, zero-padded to nFFT
	winSize := cfg.WindowSize
	rawWindow := HanningWindow(winSize)
	window := make([]float32, nFFT)
	padLeft := (nFFT - winSize) / 2
	for i := 0; i < winSize; i++ {
		window[padLeft+i] = rawWindow[i]
	}

	melFilters := MelFilterbankSlaney(sampleRate, nFFT, nMels)

	// Center-pad waveform with zeros
	padAmount := nFFT / 2
	padded := make([]float32, len(input)+2*padAmount)
	copy(padded[padAmount:], input)

	length := len(padded)
	nFrames := 1 + (length-nFFT)/hopLength
	nFreqs := nFFT/2 + 1

	// STFT magnitude squared
	magnitudes := make([][]float32, nFreqs)
	for i := range magnitudes {
		magnitudes[i] = make([]float32, nFrames)
	}

	frame := make([]float64, nFFT)
	for t := 0; t < nFrames; t++ {
		startIdx := t * hopLength
		for i := 0; i < nFFT; i++ {
			idx := startIdx + i
			if idx < len(padded) {
				frame[i] = float64(padded[idx]) * float64(window[i])
			} else {
				frame[i] = 0
			}
		}
		freqs := RFFTPow2(frame, nFFT)
		for i := 0; i < nFreqs; i++ {
			re := real(freqs[i])
			im := imag(freqs[i])
			magnitudes[i][t] = float32(re*re + im*im)
		}
	}

	// Apply mel filterbank
	melSpec := make([]float32, nMels*nFrames)
	for m := 0; m < nMels; m++ {
		filter := melFilters[m]
		base := m * nFrames
		for j := 0; j < nFrames; j++ {
			var sum float32
			for k := 0; k < nFreqs; k++ {
				if filter[k] != 0 {
					sum += filter[k] * magnitudes[k][j]
				}
			}
			melSpec[base+j] = sum
		}
	}

	// Log with guard
	const logGuard = float32(5.960464477539063e-08) // 2^-24
	for i := range melSpec {
		melSpec[i] = float32(math.Log(float64(melSpec[i] + logGuard)))
	}

	// Per-feature normalization (Bessel-corrected)
	const normEps = float32(1e-5)
	if nFrames > 1 {
		for m := 0; m < nMels; m++ {
			base := m * nFrames
			var sum float64
			for j := 0; j < nFrames; j++ {
				sum += float64(melSpec[base+j])
			}
			mean := float32(sum / float64(nFrames))
			var varSum float64
			for j := 0; j < nFrames; j++ {
				d := float64(melSpec[base+j] - mean)
				varSum += d * d
			}
			std := float32(math.Sqrt(varSum/float64(nFrames-1))) + normEps
			invStd := 1.0 / std
			for j := 0; j < nFrames; j++ {
				melSpec[base+j] = (melSpec[base+j] - mean) * invStd
			}
		}
	}

	return melSpec
}

// MelFrameCount returns the number of STFT frames for a given audio length.
func MelFrameCount(nSamples int, cfg MelConfig) int {
	padAmount := cfg.NFFTSize / 2
	length := nSamples + 2*padAmount
	return 1 + (length-cfg.NFFTSize)/cfg.HopLength
}

// SubsampledLength returns the encoder sequence length after 8x subsampling
// (3 stride-2 convolutions).
func SubsampledLength(melFrames int) int {
	l := melFrames
	for i := 0; i < 3; i++ {
		l = (l + 1) / 2
	}
	return l
}

// HanningWindow computes a symmetric Hann window (periodic=False).
//   w[i] = 0.5 * (1 - cos(2π·i / (N-1)))
func HanningWindow(size int) []float32 {
	w := make([]float32, size)
	for i := 0; i < size; i++ {
		w[i] = float32(0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(size-1))))
	}
	return w
}

// RFFTPow2 computes the real FFT using Cooley-Tukey for power-of-2 sizes.
// Returns the first n/2+1 complex frequency bins.
func RFFTPow2(input []float64, n int) []complex128 {
	m := nextPow2(n)
	c := make([]complex128, m)
	for i := 0; i < n && i < len(input); i++ {
		c[i] = complex(input[i], 0)
	}
	fftInPlace(c)
	return c[:n/2+1]
}

func fftInPlace(a []complex128) {
	n := len(a)
	if n <= 1 {
		return
	}
	// Bit-reversal permutation
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	// Cooley-Tukey butterfly
	for size := 2; size <= n; size <<= 1 {
		halfSize := size >> 1
		angleStep := -2.0 * math.Pi / float64(size)
		for i := 0; i < n; i += size {
			for k := 0; k < halfSize; k++ {
				angle := angleStep * float64(k)
				s, c := math.Sincos(angle)
				w := complex(c, s)
				u := a[i+k]
				t := w * a[i+k+halfSize]
				a[i+k] = u + t
				a[i+k+halfSize] = u - t
			}
		}
	}
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// MelFilterbankSlaney computes the Slaney mel filterbank with area normalization.
// Matches librosa.filters.mel(htk=False, norm='slaney').
func MelFilterbankSlaney(sampleRate, nFFT, nMels int) [][]float32 {
	nFreqs := nFFT/2 + 1

	const (
		fSp       = 200.0 / 3.0
		minLogHz  = 1000.0
		minLogMel = 1000.0 / fSp
		logStep   = 0.06875177742094912 // ln(6.4) / 27
	)

	hzToMel := func(f float64) float64 {
		if f < minLogHz {
			return f / fSp
		}
		return minLogMel + math.Log(f/minLogHz)/logStep
	}

	melToHz := func(m float64) float64 {
		if m < minLogMel {
			return m * fSp
		}
		return minLogHz * math.Exp(logStep*(m-minLogMel))
	}

	fMax := float64(sampleRate) / 2.0
	melMin := hzToMel(0.0)
	melMax := hzToMel(fMax)

	melPoints := make([]float64, nMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + (melMax-melMin)*float64(i)/float64(nMels+1)
	}

	freqPoints := make([]float64, nMels+2)
	for i := range freqPoints {
		freqPoints[i] = melToHz(melPoints[i])
	}

	fftFreqs := make([]float64, nFreqs)
	for i := range fftFreqs {
		fftFreqs[i] = float64(i) * float64(sampleRate) / float64(nFFT)
	}

	weights := make([][]float32, nMels)
	for m := 0; m < nMels; m++ {
		weights[m] = make([]float32, nFreqs)
		fLow := freqPoints[m]
		fCenter := freqPoints[m+1]
		fHigh := freqPoints[m+2]
		enorm := 2.0 / (fHigh - fLow)

		for k := 0; k < nFreqs; k++ {
			f := fftFreqs[k]
			var w float64
			if f >= fLow && f <= fCenter && fCenter > fLow {
				w = (f - fLow) / (fCenter - fLow)
			} else if f > fCenter && f <= fHigh && fHigh > fCenter {
				w = (fHigh - f) / (fHigh - fCenter)
			}
			weights[m][k] = float32(w * enorm)
		}
	}

	return weights
}
