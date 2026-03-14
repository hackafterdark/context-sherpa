package audio

import (
	"math"
	"testing"
)

func TestHanningWindow(t *testing.T) {
	w := HanningWindow(5)
	if len(w) != 5 {
		t.Fatalf("length = %d, want 5", len(w))
	}
	// Endpoints should be 0 for symmetric window
	if w[0] != 0 {
		t.Errorf("w[0] = %f, want 0", w[0])
	}
	if w[4] != 0 {
		t.Errorf("w[4] = %f, want 0", w[4])
	}
	// Peak at center
	if w[2] < 0.99 {
		t.Errorf("w[2] = %f, want ~1.0", w[2])
	}
}

func TestRFFTPow2(t *testing.T) {
	// FFT of constant signal should have energy only in DC bin
	n := 8
	input := make([]float64, n)
	for i := range input {
		input[i] = 1.0
	}
	freqs := RFFTPow2(input, n)
	if len(freqs) != n/2+1 {
		t.Fatalf("FFT output length = %d, want %d", len(freqs), n/2+1)
	}
	// DC component should be n
	dcMag := math.Sqrt(real(freqs[0])*real(freqs[0]) + imag(freqs[0])*imag(freqs[0]))
	if math.Abs(dcMag-float64(n)) > 1e-10 {
		t.Errorf("DC magnitude = %f, want %d", dcMag, n)
	}
	// All other bins should be ~0
	for i := 1; i < len(freqs); i++ {
		mag := math.Sqrt(real(freqs[i])*real(freqs[i]) + imag(freqs[i])*imag(freqs[i]))
		if mag > 1e-10 {
			t.Errorf("bin[%d] magnitude = %f, want ~0", i, mag)
		}
	}
}

func TestMelFilterbankSlaney(t *testing.T) {
	filters := MelFilterbankSlaney(16000, 512, 128)
	if len(filters) != 128 {
		t.Fatalf("got %d filters, want 128", len(filters))
	}
	nFreqs := 512/2 + 1
	if len(filters[0]) != nFreqs {
		t.Errorf("filter[0] length = %d, want %d", len(filters[0]), nFreqs)
	}

	// Verify filters are non-negative
	for m := 0; m < 128; m++ {
		for k := 0; k < nFreqs; k++ {
			if filters[m][k] < 0 {
				t.Errorf("filter[%d][%d] = %f, should be >= 0", m, k, filters[m][k])
			}
		}
	}

	// Lower mel bins should have energy at lower frequencies
	var lowSum, highSum float32
	for k := 0; k < nFreqs/4; k++ {
		lowSum += filters[0][k]
	}
	for k := nFreqs * 3 / 4; k < nFreqs; k++ {
		highSum += filters[0][k]
	}
	if highSum > lowSum {
		t.Error("lowest mel bin should have more energy at low frequencies")
	}
}

func TestDefaultMelConfig(t *testing.T) {
	cfg := DefaultMelConfig()
	if cfg.NMels != 128 || cfg.NFFTSize != 512 || cfg.HopLength != 160 {
		t.Errorf("unexpected default config: %+v", cfg)
	}
}

func TestWhisperMelConfig(t *testing.T) {
	cfg := WhisperMelConfig()
	if cfg.NMels != 80 || cfg.NFFTSize != 400 {
		t.Errorf("unexpected whisper config: %+v", cfg)
	}
}

func TestMelFrameCount(t *testing.T) {
	cfg := DefaultMelConfig()
	// 16000 samples = 1 second at 16kHz
	nFrames := MelFrameCount(16000, cfg)
	// Should be close to 16000/160 = 100 frames (plus padding)
	if nFrames < 90 || nFrames > 120 {
		t.Errorf("MelFrameCount(16000) = %d, want ~100", nFrames)
	}
}

func TestSubsampledLength(t *testing.T) {
	// 3 stride-2 convolutions: 100 → 50 → 25 → 13
	got := SubsampledLength(100)
	if got != 13 {
		t.Errorf("SubsampledLength(100) = %d, want 13", got)
	}

	got = SubsampledLength(1)
	if got != 1 {
		t.Errorf("SubsampledLength(1) = %d, want 1", got)
	}
}

func TestExtractMelFeatures(t *testing.T) {
	cfg := DefaultMelConfig()
	// Generate 1 second of 440Hz sine wave
	nSamples := 16000
	samples := make([]float32, nSamples)
	for i := range samples {
		samples[i] = float32(math.Sin(2.0 * math.Pi * 440.0 * float64(i) / 16000.0))
	}

	mel := ExtractMelFeatures(samples, cfg)
	nFrames := MelFrameCount(nSamples, cfg)

	if len(mel) != cfg.NMels*nFrames {
		t.Fatalf("mel length = %d, want %d", len(mel), cfg.NMels*nFrames)
	}

	// Verify all values are finite
	for i, v := range mel {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("mel[%d] = %f, not finite", i, v)
			break
		}
	}

	// After per-feature normalization, each mel bin should have mean ~0
	for m := 0; m < cfg.NMels; m++ {
		var sum float64
		for j := 0; j < nFrames; j++ {
			sum += float64(mel[m*nFrames+j])
		}
		mean := sum / float64(nFrames)
		if math.Abs(mean) > 0.1 {
			t.Errorf("mel bin %d mean = %f, want ~0 after normalization", m, mean)
			break
		}
	}
}

func TestExtractMelFeatures_NoPreEmphasis(t *testing.T) {
	cfg := DefaultMelConfig()
	cfg.PreEmphasis = 0

	samples := make([]float32, 4800) // 0.3 seconds
	for i := range samples {
		samples[i] = float32(math.Sin(2.0 * math.Pi * 1000.0 * float64(i) / 16000.0))
	}

	mel := ExtractMelFeatures(samples, cfg)
	if len(mel) == 0 {
		t.Error("mel output should not be empty")
	}
}
