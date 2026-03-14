package llm

import (
	"math"
	"math/rand"
	"testing"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/ops"
)

// ---------------------------------------------------------------------------
// Helper: load model or skip
// ---------------------------------------------------------------------------

type modelSpec struct {
	file string
	arch string
}

var allModels = []modelSpec{
	{"tinyllama-1.1b-chat-v1.0.Q4_0.gguf", "llama"},
	{"qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2"},
	{"gemma-3-270m-it-Q8_0.gguf", "gemma3"},
	{"smollm2-360m-instruct-q8_0.gguf", "llama"},
}

func loadOrSkip(t *testing.T, file string) *Model {
	t.Helper()
	path := findModel(file)
	if path == "" {
		t.Skipf("model not found: %s", file)
	}
	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", file, err)
	}
	return m
}

func makeKV(t *testing.T, m *Model, maxSeq int) (*memory.MultiLayerKVCache, *RunState) {
	t.Helper()
	cfg := m.Config
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return memory.NewMultiLayerKVCache(cfg.NumLayers, maxSeq, kvDim), NewRunState(cfg, maxSeq)
}

// ---------------------------------------------------------------------------
// E2E: Multi-token generation for every model
// ---------------------------------------------------------------------------

func TestE2EMultiTokenGeneration(t *testing.T) {
	for _, spec := range allModels {
		t.Run(spec.file, func(t *testing.T) {
			m := loadOrSkip(t, spec.file)
			cfg := m.Config
			maxSeq := 48
			kv, rs := makeKV(t, m, maxSeq)

			sampler := ops.SamplerConfig{
				Temperature: 0.01, // near-greedy for determinism
				TopK:        1,
				TopP:        1.0,
			}
			rng := rand.New(rand.NewSource(42))

			// Forward BOS
			logits := Forward(m, cfg.BOS, 0, kv, rs)
			checkLogitsValid(t, logits, cfg.VocabSize, "pos=0 BOS")

			// Generate 16 tokens
			var generated []int32
			pos := 1
			for step := 0; step < 16; step++ {
				tok := int32(ops.SampleToken(logits, sampler, generated, rng))
				if tok == cfg.EOS {
					t.Logf("EOS at step %d", step)
					break
				}
				generated = append(generated, tok)

				if int(tok) < 0 || int(tok) >= cfg.VocabSize {
					t.Fatalf("step %d: token %d out of vocab range [0, %d)", step, tok, cfg.VocabSize)
				}

				logits = Forward(m, tok, pos, kv, rs)
				checkLogitsValid(t, logits, cfg.VocabSize, "step %d pos=%d tok=%d")
				pos++
			}

			if len(generated) == 0 {
				t.Log("generated zero tokens (immediate EOS — normal for instruct models with BOS-only input)")
			}
			t.Logf("Generated %d tokens: %v", len(generated), generated)
		})
	}
}

// ---------------------------------------------------------------------------
// E2E: Determinism — same seed produces identical output
// ---------------------------------------------------------------------------

func TestE2EDeterminism(t *testing.T) {
	for _, spec := range allModels {
		t.Run(spec.file, func(t *testing.T) {
			m := loadOrSkip(t, spec.file)
			cfg := m.Config
			maxSeq := 32

			tokens1 := generateTokens(t, m, cfg, maxSeq, 42, 8)
			tokens2 := generateTokens(t, m, cfg, maxSeq, 42, 8)

			if len(tokens1) != len(tokens2) {
				t.Fatalf("different lengths: %d vs %d", len(tokens1), len(tokens2))
			}
			for i := range tokens1 {
				if tokens1[i] != tokens2[i] {
					t.Errorf("token[%d] differs: %d vs %d", i, tokens1[i], tokens2[i])
				}
			}
			t.Logf("Determinism check passed: %d identical tokens", len(tokens1))
		})
	}
}

func generateTokens(t *testing.T, m *Model, cfg ModelConfig, maxSeq int, seed int64, nSteps int) []int32 {
	t.Helper()
	kv, rs := makeKV(t, m, maxSeq)
	sampler := ops.SamplerConfig{Temperature: 0.01, TopK: 1, TopP: 1.0}
	rng := rand.New(rand.NewSource(seed))

	logits := Forward(m, cfg.BOS, 0, kv, rs)
	var out []int32
	pos := 1
	for step := 0; step < nSteps; step++ {
		tok := int32(ops.SampleToken(logits, sampler, out, rng))
		out = append(out, tok)
		if tok == cfg.EOS {
			break
		}
		logits = Forward(m, tok, pos, kv, rs)
		pos++
	}
	return out
}

// ---------------------------------------------------------------------------
// E2E: Logit distribution sanity checks
// ---------------------------------------------------------------------------

func TestE2ELogitDistribution(t *testing.T) {
	for _, spec := range allModels {
		t.Run(spec.file, func(t *testing.T) {
			m := loadOrSkip(t, spec.file)
			cfg := m.Config
			maxSeq := 16
			kv, rs := makeKV(t, m, maxSeq)

			logits := Forward(m, cfg.BOS, 0, kv, rs)

			// 1. Check logits size
			if len(logits) != cfg.VocabSize {
				t.Fatalf("logits length = %d, want %d", len(logits), cfg.VocabSize)
			}

			// 2. No NaN or Inf
			for i, v := range logits {
				if math.IsNaN(float64(v)) {
					t.Fatalf("logits[%d] is NaN", i)
				}
				if math.IsInf(float64(v), 0) {
					t.Fatalf("logits[%d] is Inf", i)
				}
			}

			// 3. Not all identical (degenerate)
			allSame := true
			for i := 1; i < len(logits); i++ {
				if logits[i] != logits[0] {
					allSame = false
					break
				}
			}
			if allSame {
				t.Error("all logits are identical (degenerate)")
			}

			// 4. Standard deviation > 0 (model is producing a real distribution)
			mean := float64(0)
			for _, v := range logits {
				mean += float64(v)
			}
			mean /= float64(len(logits))
			variance := float64(0)
			for _, v := range logits {
				d := float64(v) - mean
				variance += d * d
			}
			variance /= float64(len(logits))
			stddev := math.Sqrt(variance)
			t.Logf("Logit stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
				mean, stddev, logitMin(logits), logitMax(logits))
			if stddev < 0.01 {
				t.Error("logit standard deviation is near zero (model not differentiating tokens)")
			}

			// 5. Softmax sums to ~1
			probs := make([]float32, len(logits))
			copy(probs, logits)
			ops.Softmax(probs)
			sum := float64(0)
			for _, p := range probs {
				sum += float64(p)
			}
			if math.Abs(sum-1.0) > 1e-3 {
				t.Errorf("softmax sum = %.6f, want ~1.0", sum)
			}
		})
	}
}

func logitMin(logits []float32) float64 {
	m := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) < m {
			m = float64(v)
		}
	}
	return m
}

func logitMax(logits []float32) float64 {
	m := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > m {
			m = float64(v)
		}
	}
	return m
}

// ---------------------------------------------------------------------------
// E2E: KV cache consistency — logits change with more context
// ---------------------------------------------------------------------------

func TestE2EKVCacheConsistency(t *testing.T) {
	for _, spec := range allModels {
		t.Run(spec.file, func(t *testing.T) {
			m := loadOrSkip(t, spec.file)
			cfg := m.Config
			maxSeq := 16

			// Run 1: just BOS
			kv1, rs1 := makeKV(t, m, maxSeq)
			logits1 := Forward(m, cfg.BOS, 0, kv1, rs1)
			top1 := argmax(logits1)

			// Run 2: BOS then a fixed token (e.g., token 100)
			kv2, rs2 := makeKV(t, m, maxSeq)
			Forward(m, cfg.BOS, 0, kv2, rs2)
			logits2 := Forward(m, 100, 1, kv2, rs2)
			top2 := argmax(logits2)

			// Run 3: BOS then different token (200)
			kv3, rs3 := makeKV(t, m, maxSeq)
			Forward(m, cfg.BOS, 0, kv3, rs3)
			logits3 := Forward(m, 200, 1, kv3, rs3)
			top3 := argmax(logits3)

			// Logits SHOULD differ between contexts (different KV cache contents)
			diff12 := logitsDiffer(logits1, logits2)
			diff23 := logitsDiffer(logits2, logits3)

			if !diff12 {
				t.Error("logits identical after BOS vs BOS+tok100 — KV cache not influencing output")
			}
			if !diff23 {
				t.Error("logits identical after BOS+tok100 vs BOS+tok200 — different tokens produce same output")
			}

			t.Logf("Top tokens: BOS→%d, BOS+100→%d, BOS+200→%d", top1, top2, top3)
		})
	}
}

func argmax(logits []float32) int {
	idx := 0
	for i, v := range logits {
		if v > logits[idx] {
			idx = i
		}
	}
	return idx
}

func logitsDiffer(a, b []float32) bool {
	for i := range a {
		if a[i] != b[i] {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// E2E: Activation range checks after each layer
// ---------------------------------------------------------------------------

func TestE2EActivationRanges(t *testing.T) {
	for _, spec := range allModels {
		t.Run(spec.file, func(t *testing.T) {
			m := loadOrSkip(t, spec.file)
			cfg := m.Config
			maxSeq := 8
			kv, rs := makeKV(t, m, maxSeq)

			Forward(m, cfg.BOS, 0, kv, rs)

			// After forward pass, check run state for sensible values
			checkFinite(t, rs.X, "X (final hidden state)")
			checkFinite(t, rs.Q, "Q")
			checkFinite(t, rs.K, "K")
			checkFinite(t, rs.V, "V")
			checkFinite(t, rs.Logits, "Logits")

			// Hidden state magnitudes should be reasonable (not exploding)
			xMax := absMax(rs.X)
			t.Logf("Final hidden |x|_max = %.4f", xMax)
			if xMax > 1e6 {
				t.Errorf("hidden state exploded: max abs = %.4f", xMax)
			}
			if xMax < 1e-6 {
				t.Errorf("hidden state collapsed to zero: max abs = %.4f", xMax)
			}
		})
	}
}

func checkFinite(t *testing.T, buf []float32, name string) {
	t.Helper()
	for i, v := range buf {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("%s[%d] = %f (non-finite)", name, i, v)
			return
		}
	}
}

func absMax(buf []float32) float64 {
	m := float64(0)
	for _, v := range buf {
		a := math.Abs(float64(v))
		if a > m {
			m = a
		}
	}
	return m
}

// ---------------------------------------------------------------------------
// E2E: Pipeline generation test
// ---------------------------------------------------------------------------

func TestE2EPipelineGenerate(t *testing.T) {
	path := findModel("tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
	if path == "" {
		t.Skip("TinyLlama not found")
	}

	pipe, err := NewPipeline(path, 64)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}

	cfg := DefaultGenerateConfig()
	cfg.MaxTokens = 12
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0.01
	cfg.Sampler.TopK = 1

	var streamed []string
	cfg.Stream = func(tok string) {
		streamed = append(streamed, tok)
	}

	tokens, err := pipe.Generate([]int32{pipe.Model.Config.BOS}, cfg)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if len(tokens) == 0 {
		t.Error("Generate returned zero tokens")
	}

	t.Logf("Pipeline generated %d tokens: %v", len(tokens), tokens)
	t.Logf("Streamed tokens: %v", streamed)

	if len(streamed) != len(tokens) {
		t.Errorf("streamed %d tokens but got %d in return slice", len(streamed), len(tokens))
	}

	// All tokens should be valid
	for i, tok := range tokens {
		if int(tok) < 0 || int(tok) >= pipe.Model.Config.VocabSize {
			t.Errorf("token[%d] = %d out of vocab range", i, tok)
		}
	}
}

// ---------------------------------------------------------------------------
// E2E: Architecture-specific checks
// ---------------------------------------------------------------------------

func TestE2EQwenAttentionBiases(t *testing.T) {
	m := loadOrSkip(t, "qwen2.5-0.5b-instruct-q4_k_m.gguf")

	if m.Config.Architecture != "qwen2" {
		t.Fatalf("expected qwen2, got %s", m.Config.Architecture)
	}

	// Qwen should have attention biases on every layer
	for i, layer := range m.Layers {
		if layer.Bq == nil {
			t.Errorf("layer %d: Bq (Q bias) is nil — Qwen2 requires attention biases", i)
		}
		if layer.Bk == nil {
			t.Errorf("layer %d: Bk (K bias) is nil", i)
		}
		if layer.Bv == nil {
			t.Errorf("layer %d: Bv (V bias) is nil", i)
		}

		if layer.Bq != nil && len(layer.Bq) != m.Config.NumHeads*m.Config.HeadDim {
			t.Errorf("layer %d: Bq len = %d, want %d", i, len(layer.Bq), m.Config.NumHeads*m.Config.HeadDim)
		}
		if i > 2 {
			break // spot-check first few layers
		}
	}

	if !m.Config.RopeNeox {
		t.Error("expected RopeNeox=true for Qwen2")
	}
	t.Logf("Qwen2 RoPE freq base: %.0f", m.Config.RopeFreqBase)
}

func TestE2EGemmaSpecifics(t *testing.T) {
	m := loadOrSkip(t, "gemma-3-270m-it-Q8_0.gguf")

	if !m.Config.FFNGelu {
		t.Error("expected FFNGelu=true for Gemma3")
	}
	if m.Config.EmbedScale == 0 {
		t.Error("expected non-zero EmbedScale for Gemma3")
	}

	expectedScale := math.Sqrt(float64(m.Config.EmbeddingDim))
	if math.Abs(float64(m.Config.EmbedScale)-expectedScale) > 0.1 {
		t.Errorf("EmbedScale = %.2f, expected sqrt(%d) = %.2f",
			m.Config.EmbedScale, m.Config.EmbeddingDim, expectedScale)
	}
	t.Logf("Gemma3: dim=%d, scale=%.2f, heads=%d/%d, GeGLU=true",
		m.Config.EmbeddingDim, m.Config.EmbedScale, m.Config.NumHeads, m.Config.NumKVHeads)
}

// ---------------------------------------------------------------------------
// E2E: Weight integrity — verify loaded weights are non-trivial
// ---------------------------------------------------------------------------

func TestE2EWeightIntegrity(t *testing.T) {
	for _, spec := range allModels {
		t.Run(spec.file, func(t *testing.T) {
			m := loadOrSkip(t, spec.file)
			cfg := m.Config

			// Token embeddings: rows should contain distinct values
			emb1 := make([]float32, cfg.EmbeddingDim)
			emb2 := make([]float32, cfg.EmbeddingDim)
			_ = m.TokenEmbed.DequantizeRow(0, emb1)
			_ = m.TokenEmbed.DequantizeRow(1, emb2)

			if slicesEqual(emb1, emb2) {
				t.Error("token embeddings for IDs 0 and 1 are identical")
			}
			checkFinite(t, emb1, "emb[0]")
			checkFinite(t, emb2, "emb[1]")

			// RMS norm weights should be close to 1.0 (typical initialization)
			normAvg := float64(0)
			for _, v := range m.Layers[0].AttnNorm {
				normAvg += float64(v)
			}
			normAvg /= float64(len(m.Layers[0].AttnNorm))
			t.Logf("Layer 0 AttnNorm mean: %.4f (expected ~1.0)", normAvg)

			// Every layer should have non-nil weights
			for i, layer := range m.Layers {
				if layer.Wq == nil {
					t.Errorf("layer %d: Wq is nil", i)
				}
				if layer.Wk == nil {
					t.Errorf("layer %d: Wk is nil", i)
				}
				if layer.Wv == nil {
					t.Errorf("layer %d: Wv is nil", i)
				}
				if layer.Wo == nil {
					t.Errorf("layer %d: Wo is nil", i)
				}
				if layer.FFNGate == nil {
					t.Errorf("layer %d: FFNGate is nil", i)
				}
				if layer.FFNUp == nil {
					t.Errorf("layer %d: FFNUp is nil", i)
				}
				if layer.FFNDown == nil {
					t.Errorf("layer %d: FFNDown is nil", i)
				}
				if layer.AttnNorm == nil {
					t.Errorf("layer %d: AttnNorm is nil", i)
				}
				if layer.FFNNorm == nil {
					t.Errorf("layer %d: FFNNorm is nil", i)
				}
			}
		})
	}
}

func slicesEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// E2E: Multi-step generation stability (no NaN explosion over many steps)
// ---------------------------------------------------------------------------

func TestE2ELongGeneration(t *testing.T) {
	m := loadOrSkip(t, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
	cfg := m.Config
	maxSeq := 96
	kv, rs := makeKV(t, m, maxSeq)

	sampler := ops.SamplerConfig{Temperature: 0.7, TopK: 40, TopP: 0.9, RepetitionPenalty: 1.1}
	rng := rand.New(rand.NewSource(42))

	logits := Forward(m, cfg.BOS, 0, kv, rs)

	var generated []int32
	pos := 1
	for step := 0; step < 64; step++ {
		tok := int32(ops.SampleToken(logits, sampler, generated, rng))
		if tok == cfg.EOS {
			break
		}
		generated = append(generated, tok)

		logits = Forward(m, tok, pos, kv, rs)
		pos++

		// Check each step for NaN/Inf
		for i, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("step %d: logits[%d] = %f (non-finite after %d tokens)", step, i, v, step+1)
			}
		}
	}

	t.Logf("Generated %d tokens without NaN/Inf over %d steps", len(generated), len(generated))
	if len(generated) < 10 {
		t.Logf("Warning: generated fewer than 10 tokens (may indicate early EOS)")
	}
}

// ---------------------------------------------------------------------------
// E2E: Tokenizer correctness
// ---------------------------------------------------------------------------

func TestE2ETokenizer(t *testing.T) {
	path := findModel("tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
	if path == "" {
		t.Skip("TinyLlama not found")
	}

	pipe, err := NewPipeline(path, 32)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}

	tok := pipe.Tokenizer
	if tok.VocabSize() == 0 {
		t.Fatal("empty vocabulary")
	}
	t.Logf("Vocabulary size: %d", tok.VocabSize())

	// BOS/EOS should be valid
	if int(tok.BOS) >= tok.VocabSize() {
		t.Errorf("BOS %d >= vocab size %d", tok.BOS, tok.VocabSize())
	}
	if int(tok.EOS) >= tok.VocabSize() {
		t.Errorf("EOS %d >= vocab size %d", tok.EOS, tok.VocabSize())
	}

	// Decode single tokens
	bosStr := tok.DecodeToken(tok.BOS)
	eosStr := tok.DecodeToken(tok.EOS)
	t.Logf("BOS=%d (%q), EOS=%d (%q)", tok.BOS, bosStr, tok.EOS, eosStr)

	// Encode a simple string and verify round-trip
	text := "hello"
	encoded := tok.Encode(text)
	t.Logf("Encode(%q) = %v", text, encoded)
	if len(encoded) == 0 {
		t.Error("Encode returned empty")
	}

	// All encoded tokens should be in vocab range
	for i, id := range encoded {
		if int(id) < 0 || int(id) >= tok.VocabSize() {
			t.Errorf("encoded[%d] = %d out of range", i, id)
		}
	}
}

// ---------------------------------------------------------------------------
// E2E: Config sanity across all models
// ---------------------------------------------------------------------------

func TestE2EConfigSanity(t *testing.T) {
	for _, spec := range allModels {
		t.Run(spec.file, func(t *testing.T) {
			m := loadOrSkip(t, spec.file)
			cfg := m.Config

			// Architecture matches expected
			if cfg.Architecture != spec.arch {
				t.Errorf("architecture = %s, expected %s", cfg.Architecture, spec.arch)
			}

			// Dimensions must be positive
			if cfg.EmbeddingDim <= 0 {
				t.Error("EmbeddingDim <= 0")
			}
			if cfg.NumLayers <= 0 {
				t.Error("NumLayers <= 0")
			}
			if cfg.NumHeads <= 0 {
				t.Error("NumHeads <= 0")
			}
			if cfg.NumKVHeads <= 0 {
				t.Error("NumKVHeads <= 0")
			}
			if cfg.FFNDim <= 0 {
				t.Error("FFNDim <= 0")
			}
			if cfg.VocabSize <= 0 {
				t.Error("VocabSize <= 0")
			}

			// NumHeads must be divisible by NumKVHeads
			if cfg.NumHeads%cfg.NumKVHeads != 0 {
				t.Errorf("NumHeads(%d) %% NumKVHeads(%d) != 0", cfg.NumHeads, cfg.NumKVHeads)
			}

			// HeadDim * NumHeads should equal EmbeddingDim (for standard architectures)
			qDim := cfg.NumHeads * cfg.HeadDim
			if qDim != cfg.EmbeddingDim {
				t.Logf("Note: NumHeads*HeadDim=%d != EmbeddingDim=%d (may be valid for some architectures)", qDim, cfg.EmbeddingDim)
			}

			// Layer count should match model layers
			if len(m.Layers) != cfg.NumLayers {
				t.Errorf("model has %d layers but config says %d", len(m.Layers), cfg.NumLayers)
			}

			t.Logf("%s: %d layers, dim=%d, heads=%d/%d, ffn=%d, vocab=%d, ctx=%d",
				cfg.Architecture, cfg.NumLayers, cfg.EmbeddingDim,
				cfg.NumHeads, cfg.NumKVHeads, cfg.FFNDim, cfg.VocabSize, cfg.ContextLength)
		})
	}
}

// ---------------------------------------------------------------------------
// E2E: SmolLM2 with real prompt tokens (not just BOS)
// ---------------------------------------------------------------------------

func TestE2ESmolLM2WithPrompt(t *testing.T) {
	m := loadOrSkip(t, "smollm2-360m-instruct-q8_0.gguf")
	cfg := m.Config
	maxSeq := 64
	kv, rs := makeKV(t, m, maxSeq)

	sampler := ops.SamplerConfig{Temperature: 0.01, TopK: 1, TopP: 1.0}
	rng := rand.New(rand.NewSource(42))

	// Use a sequence of common tokens as a prompt instead of just BOS
	prompt := []int32{cfg.BOS, 100, 200, 300, 400, 500}
	for i, tok := range prompt {
		Forward(m, tok, i, kv, rs)
	}

	logits := rs.Logits
	checkLogitsValid(t, logits, cfg.VocabSize, "after prompt")

	var generated []int32
	pos := len(prompt)
	for step := 0; step < 16; step++ {
		tok := int32(ops.SampleToken(logits, sampler, generated, rng))
		if tok == cfg.EOS {
			t.Logf("EOS at step %d", step)
			break
		}
		generated = append(generated, tok)

		logits = Forward(m, tok, pos, kv, rs)
		pos++
	}

	t.Logf("SmolLM2 with prompt generated %d tokens: %v", len(generated), generated)
	if len(generated) == 0 {
		t.Error("SmolLM2 generated zero tokens even with a real prompt")
	}
}

// ---------------------------------------------------------------------------
// E2E: Pipeline test for each model (full encode→generate→decode)
// ---------------------------------------------------------------------------

func TestE2EPipelineAllModels(t *testing.T) {
	models := []struct {
		file   string
		prompt string
	}{
		{"tinyllama-1.1b-chat-v1.0.Q4_0.gguf", "Once upon a time"},
		{"qwen2.5-0.5b-instruct-q4_k_m.gguf", "The capital of France is"},
		{"gemma-3-270m-it-Q8_0.gguf", "Hello, how are"},
		{"smollm2-360m-instruct-q8_0.gguf", "The quick brown fox"},
	}

	for _, m := range models {
		t.Run(m.file, func(t *testing.T) {
			path := findModel(m.file)
			if path == "" {
				t.Skipf("%s not found", m.file)
			}

			pipe, err := NewPipeline(path, 64)
			if err != nil {
				t.Fatalf("NewPipeline: %v", err)
			}

			tokens := pipe.Tokenizer.Encode(m.prompt)
			t.Logf("Prompt %q → %d tokens: %v", m.prompt, len(tokens), tokens)

			if len(tokens) == 0 {
				t.Fatalf("tokenizer returned empty for %q", m.prompt)
			}

			cfg := DefaultGenerateConfig()
			cfg.MaxTokens = 12
			cfg.Seed = 42
			cfg.Sampler.Temperature = 0.3
			cfg.Sampler.TopK = 10

			var streamTokens []string
			cfg.Stream = func(tok string) {
				streamTokens = append(streamTokens, tok)
			}

			generated, err := pipe.Generate(tokens, cfg)
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}

			decoded := pipe.Tokenizer.Decode(generated)
			t.Logf("Generated %d tokens → %q", len(generated), decoded)
			t.Logf("Stream output: %v", streamTokens)

			// All generated tokens must be in vocab range
			for i, tok := range generated {
				if int(tok) < 0 || int(tok) >= pipe.Model.Config.VocabSize {
					t.Errorf("token[%d] = %d out of vocab range [0, %d)", i, tok, pipe.Model.Config.VocabSize)
				}
			}

			// Stream callback count must match generated count
			if len(streamTokens) != len(generated) {
				t.Errorf("streamed %d tokens but generated %d", len(streamTokens), len(generated))
			}
		})
	}
}

// ---------------------------------------------------------------------------
// E2E: Gemma embed scaling correctness
// ---------------------------------------------------------------------------

func TestE2EGemmaEmbedScaling(t *testing.T) {
	m := loadOrSkip(t, "gemma-3-270m-it-Q8_0.gguf")
	cfg := m.Config

	// Verify scale is sqrt(dim)
	expectedScale := sqrt32(float32(cfg.EmbeddingDim))
	if math.Abs(float64(cfg.EmbedScale-expectedScale)) > 0.01 {
		t.Errorf("EmbedScale=%.4f, want sqrt(%d)=%.4f", cfg.EmbedScale, cfg.EmbeddingDim, expectedScale)
	}

	// Verify scaled embeddings differ from unscaled
	maxSeq := 8
	kv1, rs1 := makeKV(t, m, maxSeq)
	Forward(m, cfg.BOS, 0, kv1, rs1)
	scaledX := make([]float32, len(rs1.X))
	copy(scaledX, rs1.X)

	// Compare against what the same token would look like unscaled
	raw := make([]float32, cfg.EmbeddingDim)
	m.TokenEmbed.DequantizeRow(int(cfg.BOS), raw)

	// After the first layer, X should differ from raw embedding
	allSame := true
	for i := range raw {
		if raw[i] != scaledX[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("forward output is identical to raw embedding (scaling + layers had no effect)")
	}
	t.Logf("Gemma embed scaling verified: scale=%.2f, dim=%d", cfg.EmbedScale, cfg.EmbeddingDim)
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

func checkLogitsValid(t *testing.T, logits []float32, vocabSize int, context string) {
	t.Helper()
	if len(logits) != vocabSize {
		t.Fatalf("%s: logits len = %d, want %d", context, len(logits), vocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) {
			t.Fatalf("%s: logits[%d] is NaN", context, i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("%s: logits[%d] is Inf", context, i)
		}
	}
}
