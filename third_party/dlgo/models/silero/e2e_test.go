package silero

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func findJFK55sWav() string {
	candidates := []string{
		filepath.Join("..", "..", "..", "evoke", "jfk_55s.wav"),
		`C:\projects\evoke\jfk_55s.wav`,
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// ---------------------------------------------------------------------------
// E2E: Exact segment match against known reference output
// ---------------------------------------------------------------------------

func TestE2ESegmentExactMatch(t *testing.T) {
	modelPath := findRealModel()
	wavPath := findJFKWav()
	if modelPath == "" || wavPath == "" {
		t.Skip("model or jfk.wav not found")
	}

	vad, err := NewSileroVAD(modelPath)
	if err != nil {
		t.Fatalf("NewSileroVAD: %v", err)
	}

	audio, _, err := LoadWAV(wavPath)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}

	probs := vad.ProcessAudio(audio)
	params := DefaultVADParams()
	segments := DetectSegments(probs, int(vad.Model.WindowSize), params)

	// Known reference: matches evoke/govad output for jfk.wav
	type refSeg struct {
		startS, endS float32
	}
	expected := []refSeg{
		{0.32, 2.27},
		{3.27, 4.41},
		{5.38, 7.68},
		{8.16, 10.62},
	}

	if len(segments) != len(expected) {
		t.Fatalf("expected %d segments, got %d", len(expected), len(segments))
	}

	for i, exp := range expected {
		seg := segments[i]
		if math.Abs(float64(seg.StartS-exp.startS)) > 0.02 {
			t.Errorf("segment %d: start=%.2f, want %.2f", i, seg.StartS, exp.startS)
		}
		if math.Abs(float64(seg.EndS-exp.endS)) > 0.02 {
			t.Errorf("segment %d: end=%.2f, want %.2f", i, seg.EndS, exp.endS)
		}
	}
	t.Log("Exact segment match against reference: PASS")
}

// ---------------------------------------------------------------------------
// E2E: Probability range and monotonicity during speech onset
// ---------------------------------------------------------------------------

func TestE2EProbabilityBehavior(t *testing.T) {
	modelPath := findRealModel()
	wavPath := findJFKWav()
	if modelPath == "" || wavPath == "" {
		t.Skip("model or jfk.wav not found")
	}

	vad, _ := NewSileroVAD(modelPath)
	audio, _, _ := LoadWAV(wavPath)
	probs := vad.ProcessAudio(audio)

	// All probs must be in [0, 1]
	for i, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("prob[%d] = %f out of [0,1]", i, p)
		}
		if math.IsNaN(float64(p)) {
			t.Fatalf("prob[%d] is NaN", i)
		}
	}

	// Count transitions (silence→speech and speech→silence)
	thresh := float32(0.5)
	transitions := 0
	prev := probs[0] > thresh
	for _, p := range probs[1:] {
		cur := p > thresh
		if cur != prev {
			transitions++
		}
		prev = cur
	}
	t.Logf("Probability transitions (crossing 0.5): %d", transitions)

	// JFK audio has clear speech segments, so we expect transitions
	if transitions < 4 {
		t.Error("expected at least 4 probability transitions for JFK audio")
	}

	// Speech regions should have high confidence
	maxSpeechProb := float32(0)
	for _, p := range probs {
		if p > maxSpeechProb {
			maxSpeechProb = p
		}
	}
	t.Logf("Max speech probability: %.4f", maxSpeechProb)
	if maxSpeechProb < 0.9 {
		t.Error("expected at least one chunk with probability > 0.9 for clear speech")
	}
}

// ---------------------------------------------------------------------------
// E2E: Long audio (55 seconds) — stress test + segment detection
// ---------------------------------------------------------------------------

func TestE2ELongAudio(t *testing.T) {
	modelPath := findRealModel()
	wavPath := findJFK55sWav()
	if modelPath == "" || wavPath == "" {
		t.Skip("model or jfk_55s.wav not found")
	}

	vad, _ := NewSileroVAD(modelPath)
	audio, sr, err := LoadWAV(wavPath)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}
	if sr != 16000 {
		t.Fatalf("sample rate = %d, want 16000", sr)
	}

	durS := float64(len(audio)) / 16000
	t.Logf("Long audio: %d samples (%.1fs)", len(audio), durS)

	probs := vad.ProcessAudio(audio)
	t.Logf("Chunks: %d", len(probs))

	params := DefaultVADParams()
	segments := DetectSegments(probs, int(vad.Model.WindowSize), params)
	t.Logf("Segments: %d", len(segments))

	// 55s of JFK speech should have multiple speech segments
	if len(segments) < 3 {
		t.Errorf("expected at least 3 speech segments in 55s, got %d", len(segments))
	}

	for i, seg := range segments {
		t.Logf("  Segment %d: %.2fs - %.2fs (%.2fs)", i, seg.StartS, seg.EndS, seg.EndS-seg.StartS)
		if seg.EndS > float32(durS)+1 {
			t.Errorf("segment %d ends at %.2f which is beyond audio duration %.1f", i, seg.EndS, durS)
		}
	}

	// No NaN in probabilities
	for i, p := range probs {
		if math.IsNaN(float64(p)) {
			t.Fatalf("prob[%d] is NaN (long audio processing failed)", i)
		}
	}
}

// ---------------------------------------------------------------------------
// E2E: Streaming vs batch consistency
// ---------------------------------------------------------------------------

func TestE2EStreamingVsBatch(t *testing.T) {
	modelPath := findRealModel()
	wavPath := findJFKWav()
	if modelPath == "" || wavPath == "" {
		t.Skip("model or jfk.wav not found")
	}

	audio, _, _ := LoadWAV(wavPath)

	// Batch mode
	vad1, _ := NewSileroVAD(modelPath)
	batchProbs := vad1.ProcessAudio(audio)

	// Streaming mode (chunk-by-chunk)
	vad2, _ := NewSileroVAD(modelPath)
	windowSize := int(vad2.Model.WindowSize)
	var streamProbs []float32

	for i := 0; i+windowSize <= len(audio); i += windowSize {
		chunk := audio[i : i+windowSize]
		p := vad2.ProcessChunkStateful(chunk)
		streamProbs = append(streamProbs, p)
	}
	// Handle remainder
	if len(audio)%windowSize != 0 {
		remainder := make([]float32, windowSize)
		start := (len(audio) / windowSize) * windowSize
		copy(remainder, audio[start:])
		p := vad2.ProcessChunkStateful(remainder)
		streamProbs = append(streamProbs, p)
	}

	// Batch and streaming should produce identical probabilities
	if len(batchProbs) != len(streamProbs) {
		t.Fatalf("batch produced %d probs, streaming produced %d", len(batchProbs), len(streamProbs))
	}

	maxDiff := float32(0)
	for i := range batchProbs {
		diff := batchProbs[i] - streamProbs[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	if maxDiff > 1e-6 {
		t.Errorf("batch vs streaming max difference: %e (expected exact match)", maxDiff)
	} else {
		t.Logf("Batch vs streaming: exact match (%d chunks)", len(batchProbs))
	}
}

// ---------------------------------------------------------------------------
// E2E: LSTM state isolation — reset between different audio
// ---------------------------------------------------------------------------

func TestE2EStateIsolation(t *testing.T) {
	modelPath := findRealModel()
	if modelPath == "" {
		t.Skip("model not found")
	}

	vad, _ := NewSileroVAD(modelPath)

	// Process silence
	silence := make([]float32, 16000)
	silProbs := vad.ProcessAudio(silence)

	// Process a sine wave (should differ from silence)
	vad.Reset()
	sine := make([]float32, 16000)
	for i := range sine {
		sine[i] = 0.5 * float32(math.Sin(2*math.Pi*440*float64(i)/16000))
	}
	sineProbs := vad.ProcessAudio(sine)

	// Results should differ
	differs := false
	for i := range silProbs {
		if i < len(sineProbs) && silProbs[i] != sineProbs[i] {
			differs = true
			break
		}
	}
	if !differs {
		t.Error("silence and 440Hz sine produce identical probabilities — state isolation failure")
	}

	// Process silence again after reset
	vad.Reset()
	silProbs2 := vad.ProcessAudio(silence)

	for i := range silProbs {
		if silProbs[i] != silProbs2[i] {
			t.Errorf("silence prob[%d] differs after reset: %f vs %f", i, silProbs[i], silProbs2[i])
			break
		}
	}
	t.Log("State isolation: PASS (reset restores identical behavior)")
}

// ---------------------------------------------------------------------------
// E2E: Model weight sanity — STFT basis is not trivial
// ---------------------------------------------------------------------------

func TestE2EModelWeightSanity(t *testing.T) {
	modelPath := findRealModel()
	if modelPath == "" {
		t.Skip("model not found")
	}

	m, _ := LoadModel(modelPath)

	// STFT basis should not be all zeros
	allZero := true
	for _, v := range m.STFTBasis {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("STFT basis is all zeros")
	}

	// Encoder weights should have variance
	for i := 0; i < 4; i++ {
		w := m.EncoderWeights[i]
		if len(w) == 0 {
			t.Errorf("encoder weight %d is empty", i)
			continue
		}
		mn, mx := w[0], w[0]
		for _, v := range w[1:] {
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
		spread := mx - mn
		if spread < 1e-6 {
			t.Errorf("encoder weight %d has near-zero spread (min=%.6f, max=%.6f)", i, mn, mx)
		}
	}

	// LSTM weights should not be symmetric
	ihSum := float64(0)
	for _, v := range m.LSTMWeightIH {
		ihSum += float64(v)
	}
	hhSum := float64(0)
	for _, v := range m.LSTMWeightHH {
		hhSum += float64(v)
	}
	if math.Abs(ihSum-hhSum) < 1e-6 {
		t.Error("LSTM WeightIH and WeightHH have identical sums (unexpected)")
	}

	t.Logf("Weight sanity: STFT basis has %d elements, LSTM IH sum=%.2f, HH sum=%.2f",
		len(m.STFTBasis), ihSum, hhSum)
}
