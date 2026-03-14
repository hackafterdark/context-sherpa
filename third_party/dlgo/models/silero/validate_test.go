package silero

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func findRealModel() string {
	candidates := []string{
		filepath.Join("..", "..", "..", "evoke", "models", "ggml-silero-v6.2.0.bin"),
		`C:\projects\evoke\models\ggml-silero-v6.2.0.bin`,
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func findJFKWav() string {
	candidates := []string{
		filepath.Join("..", "..", "..", "evoke", "jfk.wav"),
		`C:\projects\evoke\jfk.wav`,
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func TestValidateRealModel(t *testing.T) {
	modelPath := findRealModel()
	if modelPath == "" {
		t.Skip("real Silero model not found (need evoke/models/ggml-silero-v6.2.0.bin)")
	}

	m, err := LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	if m.ModelType != "silero-16k" && m.ModelType != "silero_vad" {
		t.Errorf("ModelType = %q, want silero-16k or silero_vad", m.ModelType)
	}
	if m.WindowSize != 512 {
		t.Errorf("WindowSize = %d, want 512", m.WindowSize)
	}
	if m.LSTMHiddenSize != 128 {
		t.Errorf("LSTMHiddenSize = %d, want 128", m.LSTMHiddenSize)
	}
	if len(m.STFTBasis) != 258*256 {
		t.Errorf("STFTBasis len = %d, want %d", len(m.STFTBasis), 258*256)
	}
	if len(m.LSTMWeightIH) != 512*128 {
		t.Errorf("LSTMWeightIH len = %d, want %d", len(m.LSTMWeightIH), 512*128)
	}

	t.Logf("Model loaded: %s v%d.%d.%d", m.ModelType, m.Version[0], m.Version[1], m.Version[2])
}

func TestValidateJFKWav(t *testing.T) {
	modelPath := findRealModel()
	wavPath := findJFKWav()
	if modelPath == "" || wavPath == "" {
		t.Skip("real model or jfk.wav not found")
	}

	vad, err := NewSileroVAD(modelPath)
	if err != nil {
		t.Fatalf("NewSileroVAD: %v", err)
	}

	audio, sr, err := LoadWAV(wavPath)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}
	if sr != 16000 {
		t.Fatalf("sample rate = %d, want 16000", sr)
	}
	t.Logf("Loaded jfk.wav: %d samples (%.2fs)", len(audio), float64(len(audio))/16000)

	probs := vad.ProcessAudio(audio)
	t.Logf("Generated %d probability chunks", len(probs))

	// Verify all probabilities are in [0, 1]
	for i, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("prob[%d] = %f out of [0,1]", i, p)
		}
	}

	// Log first 20 probabilities for inspection
	for i := 0; i < 20 && i < len(probs); i++ {
		marker := " "
		if probs[i] > 0.5 {
			marker = "*"
		}
		t.Logf("  chunk %3d (%.3fs): %.4f %s", i, float64(i)*0.032, probs[i], marker)
	}

	// Detect speech segments with default params
	params := DefaultVADParams()
	segments := DetectSegments(probs, int(vad.Model.WindowSize), params)

	t.Logf("\nDetected %d speech segments:", len(segments))
	for i, seg := range segments {
		t.Logf("  Segment %d: %.2fs - %.2fs (samples %d-%d)",
			i, seg.StartS, seg.EndS, seg.StartSample, seg.EndSample)
	}

	// JFK audio should contain speech - verify we detected at least 1 segment
	if len(segments) == 0 {
		t.Error("expected at least 1 speech segment in jfk.wav")
	}

	// Each segment should have valid time bounds
	for i, seg := range segments {
		if seg.StartS >= seg.EndS {
			t.Errorf("segment %d: start %.2f >= end %.2f", i, seg.StartS, seg.EndS)
		}
		if seg.StartS < 0 {
			t.Errorf("segment %d: negative start %.2f", i, seg.StartS)
		}
	}
}

func TestValidateStreamingChunkerJFK(t *testing.T) {
	modelPath := findRealModel()
	wavPath := findJFKWav()
	if modelPath == "" || wavPath == "" {
		t.Skip("real model or jfk.wav not found")
	}

	audio, _, err := LoadWAV(wavPath)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}

	cfg := DefaultStreamingChunkerConfig()
	chunker, err := NewStreamingChunker(modelPath, cfg)
	if err != nil {
		t.Fatalf("NewStreamingChunker: %v", err)
	}

	// Feed audio in 1-second increments (simulating real-time streaming)
	const chunkSize = 16000
	var allChunks []AudioChunk
	for offset := 0; offset < len(audio); offset += chunkSize {
		end := offset + chunkSize
		if end > len(audio) {
			end = len(audio)
		}
		chunks := chunker.AddSamples(audio[offset:end])
		allChunks = append(allChunks, chunks...)
	}
	flushed := chunker.Flush()
	allChunks = append(allChunks, flushed...)

	t.Logf("Streaming chunker emitted %d chunks", len(allChunks))
	for i, ch := range allChunks {
		startS := float64(ch.StartSample) / 16000
		endS := float64(ch.EndSample) / 16000
		t.Logf("  Chunk %d: %.2fs - %.2fs (%d samples)",
			i, startS, endS, len(ch.Samples))
	}

	if len(allChunks) == 0 {
		t.Error("expected at least 1 speech chunk from JFK audio")
	}
}

func TestValidateDeterminism(t *testing.T) {
	modelPath := findRealModel()
	wavPath := findJFKWav()
	if modelPath == "" || wavPath == "" {
		t.Skip("real model or jfk.wav not found")
	}

	audio, _, err := LoadWAV(wavPath)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}

	// Run twice, verify identical output
	vad1, _ := NewSileroVAD(modelPath)
	vad2, _ := NewSileroVAD(modelPath)

	probs1 := vad1.ProcessAudio(audio)
	probs2 := vad2.ProcessAudio(audio)

	if len(probs1) != len(probs2) {
		t.Fatalf("different lengths: %d vs %d", len(probs1), len(probs2))
	}

	for i := range probs1 {
		if probs1[i] != probs2[i] {
			t.Errorf("prob[%d] differs: %f vs %f", i, probs1[i], probs2[i])
			break
		}
	}
	t.Log("Determinism check: PASS (two runs produce identical output)")
}

// TestValidateSilenceDetection verifies silence produces low probabilities.
func TestValidateSilenceDetection(t *testing.T) {
	modelPath := findRealModel()
	if modelPath == "" {
		t.Skip("real model not found")
	}

	vad, err := NewSileroVAD(modelPath)
	if err != nil {
		t.Fatalf("NewSileroVAD: %v", err)
	}

	// Pure silence
	silence := make([]float32, 16000) // 1 second
	probs := vad.ProcessAudio(silence)

	maxProb := float32(0)
	for _, p := range probs {
		if p > maxProb {
			maxProb = p
		}
	}
	t.Logf("Silence max prob: %.4f (should be < 0.5)", maxProb)
	if maxProb > 0.5 {
		t.Errorf("silence detected as speech (max prob = %.4f)", maxProb)
	}
}

// TestValidatePrintSummary provides a full summary output for manual inspection.
func TestValidatePrintSummary(t *testing.T) {
	modelPath := findRealModel()
	wavPath := findJFKWav()
	if modelPath == "" || wavPath == "" {
		t.Skip("real model or jfk.wav not found")
	}

	audio, _, err := LoadWAV(wavPath)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}

	vad, _ := NewSileroVAD(modelPath)
	probs := vad.ProcessAudio(audio)
	params := DefaultVADParams()
	segments := DetectSegments(probs, int(vad.Model.WindowSize), params)

	fmt.Println("\n=== dlgo Silero VAD Validation Summary ===")
	fmt.Printf("Audio: jfk.wav (%d samples, %.2fs)\n", len(audio), float64(len(audio))/16000)
	fmt.Printf("Chunks: %d (%.1fms each)\n", len(probs), 32.0)
	fmt.Printf("Speech segments: %d\n\n", len(segments))

	for i, seg := range segments {
		fmt.Printf("  Segment %d: %6.2fs - %6.2fs  (duration: %.2fs)\n",
			i, seg.StartS, seg.EndS, seg.EndS-seg.StartS)
	}

	speechChunks := 0
	for _, p := range probs {
		if p > 0.5 {
			speechChunks++
		}
	}
	fmt.Printf("\nSpeech chunks: %d/%d (%.1f%%)\n",
		speechChunks, len(probs), float64(speechChunks)/float64(len(probs))*100)
	fmt.Println("=== Validation Complete ===")
}
