package whisper

import (
	"math"
	"math/cmplx"
)

// ExtractMel computes mel spectrogram from raw 16kHz audio samples.
// Returns [nFrames][nMels] as a flat array [nFrames * nMels].
func ExtractMel(samples []float32, nMels int) []float32 {
	const (
		sampleRate = 16000
		nFFT       = 400
		hopLength  = 160
		chunkSec   = 30
	)

	nSamples := chunkSec * sampleRate

	if len(samples) > nSamples {
		samples = samples[:nSamples]
	} else if len(samples) < nSamples {
		padded := make([]float32, nSamples)
		copy(padded, samples)
		samples = padded
	}

	waveform := make([]float32, len(samples)+hopLength)
	copy(waveform, samples)

	window := hanningWindow(nFFT)
	melFilters := buildMelFilters(sampleRate, nFFT, nMels)

	padAmount := nFFT / 2
	padded := make([]float32, len(waveform)+2*padAmount)
	for i := 0; i < padAmount; i++ {
		idx := padAmount - i
		if idx < len(waveform) {
			padded[i] = waveform[idx]
		}
	}
	copy(padded[padAmount:], waveform)
	for i := 0; i < padAmount; i++ {
		idx := len(waveform) - 2 - i
		if idx >= 0 {
			padded[padAmount+len(waveform)+i] = waveform[idx]
		}
	}

	length := len(padded)
	nFrames := 1 + (length-nFFT)/hopLength
	nFreqs := nFFT/2 + 1

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

		freqs := rfft(frame)
		for i := 0; i < nFreqs; i++ {
			re := real(freqs[i])
			im := imag(freqs[i])
			magnitudes[i][t] = float32(re*re + im*im)
		}
	}

	if nFrames > 0 {
		nFrames--
	}

	melSpec := make([][]float32, nMels)
	for i := range melSpec {
		melSpec[i] = make([]float32, nFrames)
	}

	for m := 0; m < nMels; m++ {
		filter := melFilters[m]
		for j := 0; j < nFrames; j++ {
			var sum float32
			for k := 0; k < nFreqs; k++ {
				if filter[k] != 0 {
					sum += filter[k] * magnitudes[k][j]
				}
			}
			melSpec[m][j] = sum
		}
	}

	var globalMax float32 = -1e30
	for i := 0; i < nMels; i++ {
		for j := 0; j < nFrames; j++ {
			val := melSpec[i][j]
			if val < 1e-10 {
				val = 1e-10
			}
			logVal := float32(math.Log10(float64(val)))
			melSpec[i][j] = logVal
			if logVal > globalMax {
				globalMax = logVal
			}
		}
	}

	threshold := globalMax - 8.0
	for i := 0; i < nMels; i++ {
		for j := 0; j < nFrames; j++ {
			v := melSpec[i][j]
			if v < threshold {
				v = threshold
			}
			melSpec[i][j] = (v + 4.0) / 4.0
		}
	}

	expectedFrames := nSamples / hopLength
	result := make([]float32, expectedFrames*nMels)
	for t := 0; t < expectedFrames; t++ {
		for m := 0; m < nMels; m++ {
			if t < nFrames && t < len(melSpec[m]) {
				result[t*nMels+m] = melSpec[m][t]
			} else {
				result[t*nMels+m] = (threshold + 4.0) / 4.0
			}
		}
	}

	return result
}

func hanningWindow(size int) []float32 {
	w := make([]float32, size)
	for i := 0; i < size; i++ {
		w[i] = float32(0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(size))))
	}
	return w
}

func rfft(input []float64) []complex128 {
	n := len(input)
	if n == 0 {
		return nil
	}
	c := make([]complex128, n)
	for i, v := range input {
		c[i] = complex(v, 0)
	}
	fftInPlace(c)
	return c[:n/2+1]
}

func fftInPlace(a []complex128) {
	n := len(a)
	if n <= 1 {
		return
	}

	if n&(n-1) != 0 {
		fftBluestein(a)
		return
	}

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
	for size := 2; size <= n; size <<= 1 {
		halfSize := size >> 1
		angleStep := -2.0 * math.Pi / float64(size)
		for i := 0; i < n; i += size {
			for k := 0; k < halfSize; k++ {
				s, c := math.Sincos(angleStep * float64(k))
				w := complex(c, s)
				u := a[i+k]
				t := w * a[i+k+halfSize]
				a[i+k] = u + t
				a[i+k+halfSize] = u - t
			}
		}
	}
}

func fftBluestein(a []complex128) {
	n := len(a)
	m := nextPow2(2*n - 1)

	chirp := make([]complex128, n)
	for k := 0; k < n; k++ {
		angle := -math.Pi * float64(k*k) / float64(n)
		s, c := math.Sincos(angle)
		chirp[k] = complex(c, s)
	}

	b := make([]complex128, m)
	b[0] = chirp[0]
	for k := 1; k < n; k++ {
		b[k] = chirp[k]
		b[m-k] = chirp[k]
	}
	fftPow2(b)

	aExt := make([]complex128, m)
	for i := 0; i < n; i++ {
		aExt[i] = a[i] * cmplx.Conj(chirp[i])
	}
	fftPow2(aExt)

	for i := range aExt {
		aExt[i] *= b[i]
	}
	ifftPow2(aExt)

	for i := 0; i < n; i++ {
		a[i] = aExt[i] * cmplx.Conj(chirp[i])
	}
}

func fftPow2(a []complex128) {
	n := len(a)
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
	for size := 2; size <= n; size <<= 1 {
		halfSize := size >> 1
		angleStep := -2.0 * math.Pi / float64(size)
		for i := 0; i < n; i += size {
			for k := 0; k < halfSize; k++ {
				s, c := math.Sincos(angleStep * float64(k))
				w := complex(c, s)
				u := a[i+k]
				t := w * a[i+k+halfSize]
				a[i+k] = u + t
				a[i+k+halfSize] = u - t
			}
		}
	}
}

func ifftPow2(a []complex128) {
	n := len(a)
	for i := range a {
		a[i] = cmplx.Conj(a[i])
	}
	fftPow2(a)
	invN := complex(1.0/float64(n), 0)
	for i := range a {
		a[i] = cmplx.Conj(a[i]) * invN
	}
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

func buildMelFilters(sampleRate, nFFT, nMels int) [][]float32 {
	nFreqs := nFFT/2 + 1

	fftfreqs := make([]float64, nFreqs)
	for i := 0; i < nFreqs; i++ {
		fftfreqs[i] = float64(i) * float64(sampleRate) / float64(nFFT)
	}

	minMel := 0.0
	maxMel := 45.245640471924965

	mels := make([]float64, nMels+2)
	for i := 0; i <= nMels+1; i++ {
		mels[i] = minMel + (maxMel-minMel)*float64(i)/float64(nMels+1)
	}

	fMin := 0.0
	fSp := 200.0 / 3.0
	freqs := make([]float64, nMels+2)
	for i := range freqs {
		freqs[i] = fMin + fSp*mels[i]
	}

	minLogHz := 1000.0
	minLogMel := (minLogHz - fMin) / fSp
	logstep := math.Log(6.4) / 27.0

	for i := range mels {
		if mels[i] >= minLogMel {
			freqs[i] = minLogHz * math.Exp(logstep*(mels[i]-minLogMel))
		}
	}

	fdiff := make([]float64, nMels+1)
	for i := 0; i < nMels+1; i++ {
		fdiff[i] = freqs[i+1] - freqs[i]
	}

	weights := make([][]float32, nMels)
	for i := 0; i < nMels; i++ {
		weights[i] = make([]float32, nFreqs)
		for j := 0; j < nFreqs; j++ {
			lower := -(freqs[i] - fftfreqs[j]) / fdiff[i]
			upper := (freqs[i+2] - fftfreqs[j]) / fdiff[i+1]
			w := math.Max(0, math.Min(lower, upper))
			enorm := 2.0 / (freqs[i+2] - freqs[i])
			weights[i][j] = float32(w * enorm)
		}
	}

	return weights
}
