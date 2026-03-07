package llm

import (
	"os"
	"testing"

	"github.com/computerex/dlgo/memory"
)

func findModel(names ...string) string {
	bases := []string{
		`C:\projects\evoke\models\`,
	}
	for _, base := range bases {
		for _, name := range names {
			p := base + name
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	return ""
}

func TestLoadTinyLlama(t *testing.T) {
	path := findModel("tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
	if path == "" {
		t.Skip("TinyLlama model not found")
	}

	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	cfg := m.Config
	t.Logf("Architecture: %s", cfg.Architecture)
	t.Logf("Layers: %d, Dim: %d, Heads: %d, KVHeads: %d", cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads, cfg.NumKVHeads)
	t.Logf("FFN: %d, Vocab: %d, Context: %d", cfg.FFNDim, cfg.VocabSize, cfg.ContextLength)
	t.Logf("RoPE base: %.0f, RMSNorm eps: %e", cfg.RopeFreqBase, cfg.RMSNormEps)

	if cfg.Architecture != "llama" {
		t.Errorf("expected llama architecture, got %s", cfg.Architecture)
	}
	if m.TokenEmbed == nil {
		t.Error("TokenEmbed is nil")
	}
	if m.OutputNorm == nil {
		t.Error("OutputNorm is nil")
	}
	if len(m.Layers) != cfg.NumLayers {
		t.Errorf("expected %d layers, got %d", cfg.NumLayers, len(m.Layers))
	}

	l := m.Layers[0]
	if l.AttnNorm == nil {
		t.Error("layer 0 AttnNorm is nil")
	}
	if l.Wq == nil {
		t.Error("layer 0 Wq is nil")
	}
	if l.FFNGate == nil {
		t.Error("layer 0 FFNGate is nil")
	}
}

func TestLoadQwen(t *testing.T) {
	path := findModel("qwen2.5-0.5b-instruct-q4_k_m.gguf")
	if path == "" {
		t.Skip("Qwen model not found")
	}

	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	t.Logf("Architecture: %s", m.Config.Architecture)
	t.Logf("Layers: %d, Dim: %d", m.Config.NumLayers, m.Config.EmbeddingDim)
}

func TestLoadGemma(t *testing.T) {
	path := findModel("gemma-3-270m-it-Q8_0.gguf")
	if path == "" {
		t.Skip("Gemma model not found")
	}

	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	t.Logf("Architecture: %s", m.Config.Architecture)
	t.Logf("Layers: %d, Dim: %d, FFNGelu: %v, EmbedScale: %.2f",
		m.Config.NumLayers, m.Config.EmbeddingDim, m.Config.FFNGelu, m.Config.EmbedScale)
}

func TestLoadSmolLM2(t *testing.T) {
	path := findModel("smollm2-360m-instruct-q8_0.gguf")
	if path == "" {
		t.Skip("SmolLM2 model not found")
	}

	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	t.Logf("Architecture: %s", m.Config.Architecture)
	t.Logf("Layers: %d, Dim: %d", m.Config.NumLayers, m.Config.EmbeddingDim)
}

func TestTinyLlamaForward(t *testing.T) {
	path := findModel("tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
	if path == "" {
		t.Skip("TinyLlama model not found")
	}

	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	cfg := m.Config
	maxSeq := 64
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, maxSeq, kvDim)
	rs := NewRunState(cfg, maxSeq)

	logits := Forward(m, cfg.BOS, 0, kv, rs)

	if len(logits) != cfg.VocabSize {
		t.Fatalf("logits length = %d, want %d", len(logits), cfg.VocabSize)
	}

	allZero := true
	hasNaN := false
	for _, v := range logits {
		if v != 0 {
			allZero = false
		}
		if v != v {
			hasNaN = true
		}
	}
	if allZero {
		t.Error("all logits are zero")
	}
	if hasNaN {
		t.Error("logits contain NaN")
	}

	topIdx := 0
	topVal := logits[0]
	for i, v := range logits[1:] {
		if v > topVal {
			topVal = v
			topIdx = i + 1
		}
	}
	t.Logf("Top token after BOS: %d (logit=%.4f)", topIdx, topVal)
}

func forwardTestHelper(t *testing.T, modelFile string) {
	t.Helper()
	path := findModel(modelFile)
	if path == "" {
		t.Skipf("%s model not found", modelFile)
	}

	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	cfg := m.Config
	t.Logf("Architecture: %s, Layers: %d, Dim: %d, Heads: %d/%d",
		cfg.Architecture, cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads, cfg.NumKVHeads)

	maxSeq := 32
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, maxSeq, kvDim)
	rs := NewRunState(cfg, maxSeq)

	logits := Forward(m, cfg.BOS, 0, kv, rs)

	if len(logits) != cfg.VocabSize {
		t.Fatalf("logits length = %d, want %d", len(logits), cfg.VocabSize)
	}

	hasNaN := false
	for _, v := range logits {
		if v != v {
			hasNaN = true
			break
		}
	}
	if hasNaN {
		t.Error("logits contain NaN")
	}

	topIdx := 0
	topVal := logits[0]
	for i, v := range logits[1:] {
		if v > topVal {
			topVal = v
			topIdx = i + 1
		}
	}
	t.Logf("Top token: %d (logit=%.4f)", topIdx, topVal)
}

func TestQwenForward(t *testing.T) {
	forwardTestHelper(t, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
}

func TestGemmaForward(t *testing.T) {
	forwardTestHelper(t, "gemma-3-270m-it-Q8_0.gguf")
}

func TestSmolLM2Forward(t *testing.T) {
	forwardTestHelper(t, "smollm2-360m-instruct-q8_0.gguf")
}
