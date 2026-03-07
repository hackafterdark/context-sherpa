package silero

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func testModelPath() string {
	return filepath.Join("..", "..", "testdata", "silero_tiny.ggml")
}

func skipIfNoModel(t *testing.T) {
	t.Helper()
	path := testModelPath()
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("testdata not found: %s (run 'go run ./testdata/generate.go' first)", path)
	}
}

func TestLoadModel(t *testing.T) {
	skipIfNoModel(t)
	m, err := LoadModel(testModelPath())
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	if m.ModelType != "silero_vad" {
		t.Errorf("ModelType = %q, want 'silero_vad'", m.ModelType)
	}
	if m.Version != [3]int32{5, 0, 0} {
		t.Errorf("Version = %v, want [5,0,0]", m.Version)
	}
	if m.WindowSize != 512 {
		t.Errorf("WindowSize = %d, want 512", m.WindowSize)
	}
	if m.LSTMHiddenSize != 128 {
		t.Errorf("LSTMHiddenSize = %d, want 128", m.LSTMHiddenSize)
	}
	if m.NEncoderLayers != 4 {
		t.Errorf("NEncoderLayers = %d, want 4", m.NEncoderLayers)
	}

	// Validate STFT basis size
	if len(m.STFTBasis) != 258*256 {
		t.Errorf("STFTBasis length = %d, want %d", len(m.STFTBasis), 258*256)
	}

	// Validate encoder weights
	for i := 0; i < 4; i++ {
		if m.EncoderWeights[i] == nil {
			t.Errorf("EncoderWeights[%d] is nil", i)
		}
		if m.EncoderBiases[i] == nil {
			t.Errorf("EncoderBiases[%d] is nil", i)
		}
	}

	// Validate LSTM weights
	if len(m.LSTMWeightIH) != 512*128 {
		t.Errorf("LSTMWeightIH length = %d, want %d", len(m.LSTMWeightIH), 512*128)
	}
	if len(m.LSTMWeightHH) != 512*128 {
		t.Errorf("LSTMWeightHH length = %d, want %d", len(m.LSTMWeightHH), 512*128)
	}

	// Validate final conv
	if len(m.FinalConvWeight) != 128 {
		t.Errorf("FinalConvWeight length = %d, want 128", len(m.FinalConvWeight))
	}
	if len(m.FinalConvBias) != 1 {
		t.Errorf("FinalConvBias length = %d, want 1", len(m.FinalConvBias))
	}
}

func TestLoadModelInvalidPath(t *testing.T) {
	_, err := LoadModel("/nonexistent/model.ggml")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestNewSileroVAD(t *testing.T) {
	skipIfNoModel(t)
	vad, err := NewSileroVAD(testModelPath())
	if err != nil {
		t.Fatalf("NewSileroVAD: %v", err)
	}
	if vad.Model == nil {
		t.Error("vad.Model is nil")
	}
	if vad.State == nil {
		t.Error("vad.State is nil")
	}
}

func TestForwardChunkDeterministic(t *testing.T) {
	skipIfNoModel(t)
	vad, err := NewSileroVAD(testModelPath())
	if err != nil {
		t.Fatalf("NewSileroVAD: %v", err)
	}

	chunk := make([]float32, 512)
	for i := range chunk {
		chunk[i] = float32(math.Sin(float64(i) * 0.01))
	}

	prob1 := vad.ProcessChunkStateful(chunk)
	vad.Reset()
	prob2 := vad.ProcessChunkStateful(chunk)

	if prob1 != prob2 {
		t.Errorf("non-deterministic: prob1=%f, prob2=%f", prob1, prob2)
	}

	if prob1 < 0 || prob1 > 1 {
		t.Errorf("probability out of [0,1]: %f", prob1)
	}
}

func TestProcessAudio(t *testing.T) {
	skipIfNoModel(t)
	vad, err := NewSileroVAD(testModelPath())
	if err != nil {
		t.Fatalf("NewSileroVAD: %v", err)
	}

	// 3 chunks worth of audio (1536 samples = 3 × 512)
	audio := make([]float32, 1536)
	for i := range audio {
		audio[i] = float32(math.Sin(float64(i) * 0.05))
	}

	probs := vad.ProcessAudio(audio)
	if len(probs) != 3 {
		t.Fatalf("len(probs) = %d, want 3", len(probs))
	}

	for i, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("probs[%d] = %f, want in [0,1]", i, p)
		}
	}
}

func TestProcessAudioPadding(t *testing.T) {
	skipIfNoModel(t)
	vad, err := NewSileroVAD(testModelPath())
	if err != nil {
		t.Fatalf("NewSileroVAD: %v", err)
	}

	// 600 samples = 1 full chunk + 88 sample remainder (zero-padded)
	audio := make([]float32, 600)
	probs := vad.ProcessAudio(audio)
	if len(probs) != 2 {
		t.Errorf("len(probs) = %d, want 2 (1 full + 1 padded)", len(probs))
	}
}

func TestProcessChunk(t *testing.T) {
	skipIfNoModel(t)
	m, err := LoadModel(testModelPath())
	if err != nil {
		t.Fatal(err)
	}

	chunk := make([]float32, 512)
	prob := ProcessChunk(m, chunk)
	if prob < 0 || prob > 1 {
		t.Errorf("ProcessChunk: prob=%f, want in [0,1]", prob)
	}
}

func TestVADReset(t *testing.T) {
	skipIfNoModel(t)
	vad, err := NewSileroVAD(testModelPath())
	if err != nil {
		t.Fatal(err)
	}

	// Process some audio to change state
	chunk := make([]float32, 512)
	for i := range chunk {
		chunk[i] = 0.5
	}
	vad.ProcessChunkStateful(chunk)

	// State should be non-zero
	allZero := true
	for _, v := range vad.State.H {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("state.H is all zeros after processing")
	}

	// After reset, state should be all zeros
	vad.Reset()
	for i, v := range vad.State.H {
		if v != 0 {
			t.Errorf("state.H[%d] = %f after reset, want 0", i, v)
			break
		}
	}
}

func TestNilVADReset(t *testing.T) {
	var vad *SileroVAD
	vad.Reset() // should not panic
}

// --- LSTM unit tests ---

func TestLSTMState(t *testing.T) {
	state := NewLSTMState(64)
	if len(state.H) != 64 || len(state.C) != 64 {
		t.Fatalf("NewLSTMState(64): H=%d C=%d, want 64,64", len(state.H), len(state.C))
	}

	// Initially zero
	for i := range state.H {
		if state.H[i] != 0 || state.C[i] != 0 {
			t.Error("state not zero-initialized")
			break
		}
	}

	state.H[0] = 1.0
	state.C[0] = 2.0
	state.Reset()
	if state.H[0] != 0 || state.C[0] != 0 {
		t.Error("reset did not zero state")
	}
}

// --- Encoder unit tests ---

func TestConv1d(t *testing.T) {
	// Simple 1D conv: 1 input channel, 1 output channel, kernel=3, stride=1, padding=1
	// Input: [1][5] = {{1, 2, 3, 4, 5}}
	input := [][]float32{{1, 2, 3, 4, 5}}
	weight := []float32{1, 0, -1} // kernel detects edges
	bias := []float32{0}

	out := conv1d(input, weight, bias, 1, 1, 3, 1, 1)
	if len(out) != 1 {
		t.Fatalf("output channels = %d, want 1", len(out))
	}
	// With padding=1 and stride=1: output length = (5+2-3)/1+1 = 5
	if len(out[0]) != 5 {
		t.Fatalf("output length = %d, want 5", len(out[0]))
	}

	// out[0] = 0*1 + 1*0 + 2*(-1) = -2
	// out[1] = 1*1 + 2*0 + 3*(-1) = -2
	// out[2] = 2*1 + 3*0 + 4*(-1) = -2
	// out[3] = 3*1 + 4*0 + 5*(-1) = -2
	// out[4] = 4*1 + 5*0 + 0*(-1) = 4
	expected := []float32{-2, -2, -2, -2, 4}
	for i, want := range expected {
		if math.Abs(float64(out[0][i]-want)) > 1e-6 {
			t.Errorf("out[0][%d] = %f, want %f", i, out[0][i], want)
		}
	}
}

func TestRelu2d(t *testing.T) {
	data := [][]float32{{-1, 0, 1, -2, 3}}
	relu2d(data)
	expected := []float32{0, 0, 1, 0, 3}
	for i, want := range expected {
		if data[0][i] != want {
			t.Errorf("relu2d[0][%d] = %f, want %f", i, data[0][i], want)
		}
	}
}

// --- STFT unit tests ---

func TestApplySTFTShape(t *testing.T) {
	chunk := make([]float32, 512)
	basis := make([]float32, 258*256)
	mag := applySTFT(chunk, basis)

	if len(mag) != 129 {
		t.Errorf("magnitude freq bins = %d, want 129", len(mag))
	}
	for i, row := range mag {
		if len(row) != 4 {
			t.Errorf("magnitude[%d] time steps = %d, want 4", i, len(row))
		}
	}
}

func TestApplySTFTNonNegative(t *testing.T) {
	chunk := make([]float32, 512)
	for i := range chunk {
		chunk[i] = float32(math.Sin(float64(i) * 0.1))
	}

	// Use a simple basis (identity-like)
	basis := make([]float32, 258*256)
	for i := range basis {
		basis[i] = 0.01
	}

	mag := applySTFT(chunk, basis)
	for freq := range mag {
		for ts := range mag[freq] {
			if mag[freq][ts] < 0 {
				t.Errorf("negative magnitude at [%d][%d] = %f", freq, ts, mag[freq][ts])
			}
		}
	}
}

// --- Segments unit tests ---

func TestDetectSegmentsEmpty(t *testing.T) {
	params := DefaultVADParams()
	segments := DetectSegments(nil, 512, params)
	if len(segments) != 0 {
		t.Errorf("expected 0 segments for nil probs, got %d", len(segments))
	}
}

func TestDetectSegmentsSilence(t *testing.T) {
	params := DefaultVADParams()
	probs := make([]float32, 100)
	segments := DetectSegments(probs, 512, params)
	if len(segments) != 0 {
		t.Errorf("expected 0 segments for all-silence, got %d", len(segments))
	}
}

func TestDetectSegmentsSpeech(t *testing.T) {
	params := DefaultVADParams()
	params.MinSpeechDurationMs = 50
	params.MinSilenceDurationMs = 50

	// Create a speech block in the middle
	probs := make([]float32, 200)
	for i := 50; i < 150; i++ {
		probs[i] = 0.9
	}

	segments := DetectSegments(probs, 512, params)
	if len(segments) == 0 {
		t.Error("expected at least 1 speech segment")
	}
	for _, seg := range segments {
		if seg.StartS >= seg.EndS {
			t.Errorf("invalid segment: start=%f >= end=%f", seg.StartS, seg.EndS)
		}
	}
}

func TestDefaultVADParams(t *testing.T) {
	p := DefaultVADParams()
	if p.Threshold != 0.5 {
		t.Errorf("Threshold = %f, want 0.5", p.Threshold)
	}
	if p.MinSpeechDurationMs != 250 {
		t.Errorf("MinSpeechDurationMs = %d, want 250", p.MinSpeechDurationMs)
	}
}

// --- Streaming chunker unit tests ---

func TestDefaultStreamingChunkerConfig(t *testing.T) {
	cfg := DefaultStreamingChunkerConfig()
	if cfg.StableSilenceMs != 700 {
		t.Errorf("StableSilenceMs = %d, want 700", cfg.StableSilenceMs)
	}
	if cfg.MinEmitChunkMs != 400 {
		t.Errorf("MinEmitChunkMs = %d, want 400", cfg.MinEmitChunkMs)
	}
}

func TestStreamingChunker(t *testing.T) {
	skipIfNoModel(t)

	cfg := DefaultStreamingChunkerConfig()
	chunker, err := NewStreamingChunker(testModelPath(), cfg)
	if err != nil {
		t.Fatalf("NewStreamingChunker: %v", err)
	}

	// Feed silence → should get no chunks
	silence := make([]float32, 16000)
	chunks := chunker.AddSamples(silence)
	// With silence, we expect no speech chunks
	_ = chunks

	// Flush → may or may not emit depending on detection
	flushed := chunker.Flush()
	_ = flushed

	// Reset should not panic
	chunker.Reset()
}

// --- f16ToF32 unit tests ---

func TestF16ToF32(t *testing.T) {
	tests := []struct {
		input    uint16
		expected float32
	}{
		{0x3C00, 1.0},
		{0x0000, 0.0},
		{0xBC00, -1.0},
		{0x4000, 2.0},
		{0x3800, 0.5},
		{0x7C00, float32(math.Inf(1))},  // +inf
		{0xFC00, float32(math.Inf(-1))}, // -inf
	}

	for _, tt := range tests {
		got := f16ToF32(tt.input)
		if math.IsInf(float64(tt.expected), 0) {
			if !math.IsInf(float64(got), 0) {
				t.Errorf("f16ToF32(0x%04X) = %f, want inf", tt.input, got)
			}
		} else if math.Abs(float64(got-tt.expected)) > 1e-6 {
			t.Errorf("f16ToF32(0x%04X) = %f, want %f", tt.input, got, tt.expected)
		}
	}
}

// --- WAV loading (audio.go) ---

func TestLoadWAVInvalidPath(t *testing.T) {
	_, _, err := LoadWAV("/nonexistent/audio.wav")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestEndToEndWithWAV(t *testing.T) {
	skipIfNoModel(t)

	wavPath := filepath.Join("..", "..", "testdata", "test_audio.wav")
	if _, err := os.Stat(wavPath); os.IsNotExist(err) {
		t.Skip("test_audio.wav not found")
	}

	// Load audio
	samples, sr, err := LoadWAV(wavPath)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}
	if sr != 16000 {
		t.Fatalf("sample rate = %d, want 16000", sr)
	}

	// Load model and run VAD
	vad, err := NewSileroVAD(testModelPath())
	if err != nil {
		t.Fatalf("NewSileroVAD: %v", err)
	}

	probs := vad.ProcessAudio(samples)
	if len(probs) == 0 {
		t.Fatal("ProcessAudio returned no probabilities")
	}

	// All probabilities should be in [0, 1]
	for i, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("prob[%d] = %f, want in [0, 1]", i, p)
		}
	}

	t.Logf("Processed %d samples → %d chunks, first prob=%.4f", len(samples), len(probs), probs[0])
}

func TestSamplesToCs(t *testing.T) {
	// 16000 samples = 1 second = 100 centiseconds
	cs := samplesToCs(16000)
	if cs != 100 {
		t.Errorf("samplesToCs(16000) = %d, want 100", cs)
	}

	cs = samplesToCs(0)
	if cs != 0 {
		t.Errorf("samplesToCs(0) = %d, want 0", cs)
	}

	// 8000 samples = 0.5 second = 50 centiseconds
	cs = samplesToCs(8000)
	if cs != 50 {
		t.Errorf("samplesToCs(8000) = %d, want 50", cs)
	}
}
