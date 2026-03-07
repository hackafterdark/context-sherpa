package whisper

import (
	"testing"

	"github.com/computerex/dlgo/format/gguf"
)

// ---------------------------------------------------------------------------
// E2E: Whisper GGUF config parsing validation
// ---------------------------------------------------------------------------

func TestE2EWhisperConfigFromGGUF(t *testing.T) {
	path := findWhisperModel()
	if path == "" {
		t.Skip("Whisper model not found")
	}

	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("gguf.Open: %v", err)
	}

	cfg := parseWhisperConfig(gf.Metadata)

	t.Logf("Parsed Whisper config from GGUF:")
	t.Logf("  Encoder layers: %d", cfg.NumEncLayers)
	t.Logf("  Decoder layers: %d", cfg.NumDecLayers)
	t.Logf("  Embedding dim:  %d", cfg.EmbeddingDim)
	t.Logf("  Heads:          %d (dim %d)", cfg.NumHeads, cfg.HeadDim)
	t.Logf("  FFN dim:        %d", cfg.FFNDim)
	t.Logf("  Num mels:       %d", cfg.NumMels)
	t.Logf("  Vocab size:     %d", cfg.VocabSize)

	// Whisper Base known dimensions
	if cfg.NumEncLayers < 1 {
		t.Error("expected at least 1 encoder layer")
	}
	if cfg.NumDecLayers < 1 {
		t.Error("expected at least 1 decoder layer")
	}
	if cfg.EmbeddingDim < 256 {
		t.Errorf("embedding dim %d seems too small for Whisper", cfg.EmbeddingDim)
	}
	if cfg.HeadDim <= 0 {
		t.Error("HeadDim must be positive")
	}
	if cfg.NumHeads*cfg.HeadDim != cfg.EmbeddingDim {
		t.Errorf("NumHeads(%d) * HeadDim(%d) = %d != EmbeddingDim(%d)",
			cfg.NumHeads, cfg.HeadDim, cfg.NumHeads*cfg.HeadDim, cfg.EmbeddingDim)
	}
}

// ---------------------------------------------------------------------------
// E2E: Whisper tensor inventory — verify expected tensors exist
// ---------------------------------------------------------------------------

func TestE2EWhisperTensorInventory(t *testing.T) {
	path := findWhisperModel()
	if path == "" {
		t.Skip("Whisper model not found")
	}

	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("gguf.Open: %v", err)
	}

	tensorNames := make(map[string]bool)
	for _, ti := range gf.Tensors {
		tensorNames[ti.Name] = true
	}

	t.Logf("Total tensors in Whisper GGUF: %d", len(gf.Tensors))

	// Check for essential Whisper tensors (actual GGUF naming convention)
	essential := []string{
		"encoder.position_embedding.weight",
		"encoder.conv1.weight",
		"encoder.conv1.bias",
		"encoder.conv2.weight",
		"encoder.conv2.bias",
		"encoder.ln.weight",
		"encoder.ln.bias",
		"decoder.position_embedding.weight",
		"decoder.token_embedding.weight",
		"decoder.ln.weight",
		"decoder.ln.bias",
		"decoder.proj.weight",
	}

	for _, name := range essential {
		if !tensorNames[name] {
			t.Errorf("missing essential tensor: %s", name)
		}
	}

	// Check for encoder block 0 — self-attention + FFN
	encBlock0 := []string{
		"encoder.blocks.0.attn.q.weight",
		"encoder.blocks.0.attn.k.weight",
		"encoder.blocks.0.attn.v.weight",
		"encoder.blocks.0.attn.out.weight",
		"encoder.blocks.0.ffn.0.weight",
		"encoder.blocks.0.ffn.2.weight",
		"encoder.blocks.0.attn_ln.weight",
		"encoder.blocks.0.ffn_ln.weight",
	}
	for _, name := range encBlock0 {
		if !tensorNames[name] {
			t.Errorf("missing encoder block 0 tensor: %s", name)
		}
	}

	// Check for decoder block 0 — self-attention + cross-attention + FFN
	decBlock0 := []string{
		"decoder.blocks.0.attn.q.weight",
		"decoder.blocks.0.attn.k.weight",
		"decoder.blocks.0.attn.v.weight",
		"decoder.blocks.0.cross_attn.q.weight",
		"decoder.blocks.0.cross_attn.k.weight",
		"decoder.blocks.0.cross_attn.v.weight",
		"decoder.blocks.0.cross_attn.out.weight",
		"decoder.blocks.0.cross_attn_ln.weight",
		"decoder.blocks.0.ffn.0.weight",
		"decoder.blocks.0.ffn.2.weight",
	}
	for _, name := range decBlock0 {
		if !tensorNames[name] {
			t.Errorf("missing decoder block 0 tensor: %s", name)
		}
	}
}

// ---------------------------------------------------------------------------
// E2E: Whisper model struct allocation
// ---------------------------------------------------------------------------

func TestE2EWhisperModelStruct(t *testing.T) {
	path := findWhisperModel()
	if path == "" {
		t.Skip("Whisper model not found")
	}

	m, err := LoadWhisperModel(path)
	if err != nil {
		t.Fatalf("LoadWhisperModel: %v", err)
	}

	cfg := m.Config

	if len(m.EncLayers) != cfg.NumEncLayers {
		t.Errorf("EncLayers: got %d, want %d", len(m.EncLayers), cfg.NumEncLayers)
	}
	if len(m.DecLayers) != cfg.NumDecLayers {
		t.Errorf("DecLayers: got %d, want %d", len(m.DecLayers), cfg.NumDecLayers)
	}

	// Encoder-decoder structure should be symmetric for Whisper Base
	if cfg.NumEncLayers != cfg.NumDecLayers {
		t.Logf("Note: enc layers (%d) != dec layers (%d)", cfg.NumEncLayers, cfg.NumDecLayers)
	}

	t.Logf("Whisper model struct: %d enc layers, %d dec layers", len(m.EncLayers), len(m.DecLayers))
}
