package whisper

import (
	"os"
	"testing"
)

func findWhisperModel() string {
	candidates := []string{
		`C:\projects\evoke\models\whisper-base-q8_0.gguf`,
		`C:\projects\evoke\dist\models\whisper-base-q8_0.gguf`,
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func TestLoadWhisperBase(t *testing.T) {
	path := findWhisperModel()
	if path == "" {
		t.Skip("Whisper model not found")
	}

	m, err := LoadWhisperModel(path)
	if err != nil {
		t.Fatalf("LoadWhisperModel: %v", err)
	}

	cfg := m.Config
	t.Logf("Whisper config:")
	t.Logf("  Encoder layers: %d", cfg.NumEncLayers)
	t.Logf("  Decoder layers: %d", cfg.NumDecLayers)
	t.Logf("  Embedding dim: %d", cfg.EmbeddingDim)
	t.Logf("  Heads: %d (dim %d)", cfg.NumHeads, cfg.HeadDim)
	t.Logf("  FFN dim: %d", cfg.FFNDim)
	t.Logf("  Vocab: %d", cfg.VocabSize)

	if cfg.NumEncLayers == 0 {
		t.Error("expected non-zero encoder layers")
	}
}

func TestWhisperConfigDefaults(t *testing.T) {
	cfg := parseWhisperConfig(map[string]interface{}{})

	if cfg.NumMels != 80 {
		t.Errorf("NumMels = %d, want 80", cfg.NumMels)
	}
	if cfg.EmbeddingDim != 512 {
		t.Errorf("EmbeddingDim = %d, want 512", cfg.EmbeddingDim)
	}
}
