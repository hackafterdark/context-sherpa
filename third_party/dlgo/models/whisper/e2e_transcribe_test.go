package whisper

import (
	"os"
	"strings"
	"testing"
	"time"
)

func findTokenizer() string {
	candidates := []string{
		`C:\projects\evoke\models\whisper-base-onnx\tokenizer.json`,
		`C:\projects\fasterwhispergo\models\whisper-base-onnx\tokenizer.json`,
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func findJFK() string {
	candidates := []string{
		`C:\projects\evoke\jfk.wav`,
		`C:\projects\fasterwhispergo\jfk.wav`,
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func TestE2ETranscribe(t *testing.T) {
	modelPath := findWhisperModel()
	if modelPath == "" {
		t.Skip("Whisper model not found")
	}
	tokPath := findTokenizer()
	if tokPath == "" {
		t.Skip("Whisper tokenizer not found")
	}
	audioPath := findJFK()
	if audioPath == "" {
		t.Skip("jfk.wav not found")
	}

	m, err := LoadWhisperModel(modelPath)
	if err != nil {
		t.Fatalf("LoadWhisperModel: %v", err)
	}
	tok, err := LoadTokenizer(tokPath)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	start := time.Now()
	text, err := m.TranscribeFile(audioPath, tok)
	if err != nil {
		t.Fatalf("TranscribeFile: %v", err)
	}
	elapsed := time.Since(start)

	t.Logf("Transcription (%v): %q", elapsed, text)

	lower := strings.ToLower(text)
	if !strings.Contains(lower, "ask not") {
		t.Errorf("expected 'ask not' in transcription, got: %q", text)
	}
	if !strings.Contains(lower, "country") {
		t.Errorf("expected 'country' in transcription, got: %q", text)
	}
}
