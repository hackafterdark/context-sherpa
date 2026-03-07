package llm

import (
	"fmt"
	"math/rand"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/ops"
)

// GenerateConfig controls text generation behavior.
type GenerateConfig struct {
	MaxTokens int
	Sampler   ops.SamplerConfig
	Seed      int64
	Stream    func(token string) // called for each generated token (nil = no streaming)
}

// DefaultGenerateConfig returns sensible defaults.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens: 256,
		Sampler:   ops.DefaultSamplerConfig(),
		Seed:      -1,
	}
}

// Pipeline bundles a loaded model, tokenizer, KV cache, and run state for inference.
type Pipeline struct {
	Model      *Model
	Tokenizer  *Tokenizer
	KVCache    *memory.MultiLayerKVCache
	RunState   *RunState
	BatchState *BatchState
	MaxSeqLen  int
}

// NewPipeline loads a GGUF model and creates a ready-to-use inference pipeline
// with automatic tokenizer extraction from GGUF metadata.
func NewPipeline(modelPath string, maxSeqLen int) (*Pipeline, error) {
	gf, err := gguf.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}

	m, err := LoadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	if maxSeqLen <= 0 || maxSeqLen > m.Config.ContextLength {
		maxSeqLen = m.Config.ContextLength
	}

	tok, err := NewTokenizerFromGGUF(gf.Metadata, m.Config)
	if err != nil {
		tok = &Tokenizer{
			BOS:    m.Config.BOS,
			EOS:    m.Config.EOS,
			AddBOS: m.Config.AddBOS,
		}
	}

	kvDim := m.Config.NumKVHeads * m.Config.HeadDim
	kv := memory.NewMultiLayerKVCache(m.Config.NumLayers, maxSeqLen, kvDim)
	rs := NewRunState(m.Config, maxSeqLen)

	return &Pipeline{
		Model:      m,
		Tokenizer:  tok,
		KVCache:    kv,
		RunState:   rs,
		BatchState: NewBatchState(m.Config, maxSeqLen),
		MaxSeqLen:  maxSeqLen,
	}, nil
}

// Generate produces text from a prompt using the loaded model.
func (p *Pipeline) Generate(prompt []int32, cfg GenerateConfig) ([]int32, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("empty prompt")
	}
	if len(prompt) >= p.MaxSeqLen {
		return nil, fmt.Errorf("prompt too long: %d tokens (max %d)", len(prompt), p.MaxSeqLen)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	p.KVCache.Reset()
	if p.RunState.SSMState != nil {
		p.RunState.SSMState.Reset()
	}

	runtime.GC()
	prev := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(prev)

	var generated []int32
	var recentTokens []int32

	// Prefill (batch)
	ForwardBatch(p.Model, prompt, 0, p.KVCache, p.RunState, p.BatchState)

	pos := len(prompt)
	nextToken := ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng)
	generated = append(generated, int32(nextToken))
	recentTokens = append(recentTokens, int32(nextToken))

	if cfg.Stream != nil {
		cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
	}

	for step := 1; step < cfg.MaxTokens; step++ {
		if pos >= p.MaxSeqLen-1 {
			break
		}

		lastTok := int32(nextToken)
		if lastTok == p.Model.Config.EOS {
			break
		}
		for _, stop := range p.Model.Config.StopTokens {
			if lastTok == stop {
				return generated, nil
			}
		}

		Forward(p.Model, lastTok, pos, p.KVCache, p.RunState)
		pos++

		nextToken = ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng)
		generated = append(generated, int32(nextToken))

		recentTokens = append(recentTokens, int32(nextToken))
		if len(recentTokens) > 64 {
			recentTokens = recentTokens[1:]
		}

		if cfg.Stream != nil {
			cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
		}
	}

	return generated, nil
}

// GenerateText is a convenience method that takes a text prompt, encodes it,
// generates tokens, and decodes the result. Returns the generated text and
// token/second throughput.
func (p *Pipeline) GenerateText(prompt string, cfg GenerateConfig) (string, float64, error) {
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return "", 0, fmt.Errorf("tokenizer produced no tokens for prompt")
	}

	start := time.Now()
	generated, err := p.Generate(tokens, cfg)
	elapsed := time.Since(start)

	if err != nil {
		return "", 0, err
	}

	text := p.Tokenizer.Decode(generated)
	tokPerSec := float64(len(generated)) / elapsed.Seconds()
	return text, tokPerSec, nil
}

// Chat formats a user message (with optional system prompt) using the model's
// chat template, then generates a response. Returns generated text and tok/s.
func (p *Pipeline) Chat(system, user string, cfg GenerateConfig) (string, float64, error) {
	prompt := FormatChat(p.Model.Config, system, user)
	return p.GenerateText(prompt, cfg)
}

// ChatMessages formats a multi-turn conversation and generates the assistant's
// next response. Returns generated text and tok/s.
func (p *Pipeline) ChatMessages(messages []Message, cfg GenerateConfig) (string, float64, error) {
	prompt := FormatMessages(p.Model.Config, messages)
	return p.GenerateText(prompt, cfg)
}

// GenerateResult holds detailed output from a generation run.
type GenerateResult struct {
	Text          string
	Tokens        []int32
	TokensPerSec  float64
	PrefillTimeMs float64
	GenerateTimeMs float64
	TotalTokens   int
	PromptTokens  int
}

// GenerateDetailed is like GenerateText but returns detailed timing information.
func (p *Pipeline) GenerateDetailed(prompt string, cfg GenerateConfig) (*GenerateResult, error) {
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer produced no tokens for prompt")
	}
	if len(tokens) >= p.MaxSeqLen {
		return nil, fmt.Errorf("prompt too long: %d tokens (max %d)", len(tokens), p.MaxSeqLen)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	p.KVCache.Reset()
	if p.RunState.SSMState != nil {
		p.RunState.SSMState.Reset()
	}

	// Minimize GC interference during inference
	runtime.GC()
	prev := debug.SetGCPercent(-1)

	// Prefill (batch)
	prefillStart := time.Now()
	ForwardBatch(p.Model, tokens, 0, p.KVCache, p.RunState, p.BatchState)
	prefillMs := float64(time.Since(prefillStart).Microseconds()) / 1000.0

	// Generate
	genStart := time.Now()
	var generated []int32
	var recentTokens []int32

	pos := len(tokens)
	nextToken := ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng)
	generated = append(generated, int32(nextToken))
	recentTokens = append(recentTokens, int32(nextToken))

	if cfg.Stream != nil {
		cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
	}

	for step := 1; step < cfg.MaxTokens; step++ {
		if pos >= p.MaxSeqLen-1 {
			break
		}
		lastTok := int32(nextToken)
		if lastTok == p.Model.Config.EOS {
			break
		}
		for _, stop := range p.Model.Config.StopTokens {
			if lastTok == stop {
				goto done
			}
		}

		Forward(p.Model, lastTok, pos, p.KVCache, p.RunState)
		pos++

		nextToken = ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng)
		generated = append(generated, int32(nextToken))
		recentTokens = append(recentTokens, int32(nextToken))
		if len(recentTokens) > 64 {
			recentTokens = recentTokens[1:]
		}

		if cfg.Stream != nil {
			cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
		}
	}

done:
	genMs := float64(time.Since(genStart).Microseconds()) / 1000.0
	debug.SetGCPercent(prev)
	text := p.Tokenizer.Decode(generated)

	var tokPerSec float64
	if genMs > 0 {
		tokPerSec = float64(len(generated)) / (genMs / 1000.0)
	}

	return &GenerateResult{
		Text:           text,
		Tokens:         generated,
		TokensPerSec:   tokPerSec,
		PrefillTimeMs:  prefillMs,
		GenerateTimeMs: genMs,
		TotalTokens:    len(generated),
		PromptTokens:   len(tokens),
	}, nil
}

// collectStopStrings returns text-level stop sequences for the model's arch.
func collectStopStrings(cfg ModelConfig) []string {
	return []string{
		"<|im_end|>",
		"<|endoftext|>",
		"<|end|>",
		"</s>",
		"<|assistant|>",
		"<end_of_turn>",
		"<|eot_id|>",
	}
}

// GenerateTextWithStopStrings is like GenerateText but also handles text-level
// stop string detection for multi-token stop sequences.
func (p *Pipeline) GenerateTextWithStopStrings(prompt string, cfg GenerateConfig) (string, float64, error) {
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return "", 0, fmt.Errorf("tokenizer produced no tokens")
	}
	if len(tokens) >= p.MaxSeqLen {
		return "", 0, fmt.Errorf("prompt too long: %d tokens (max %d)", len(tokens), p.MaxSeqLen)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	p.KVCache.Reset()
	if p.RunState.SSMState != nil {
		p.RunState.SSMState.Reset()
	}
	stopStrings := collectStopStrings(p.Model.Config)

	for i, tok := range tokens {
		Forward(p.Model, tok, i, p.KVCache, p.RunState)
	}

	start := time.Now()
	var generated []int32
	var recentTokens []int32
	var genText strings.Builder

	pos := len(tokens)
	for step := 0; step < cfg.MaxTokens; step++ {
		if pos >= p.MaxSeqLen-1 {
			break
		}

		nextToken := int32(ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng))

		if nextToken == p.Model.Config.EOS {
			break
		}
		stopped := false
		for _, stop := range p.Model.Config.StopTokens {
			if nextToken == stop {
				stopped = true
				break
			}
		}
		if stopped {
			break
		}

		generated = append(generated, nextToken)
		recentTokens = append(recentTokens, nextToken)
		if len(recentTokens) > 64 {
			recentTokens = recentTokens[1:]
		}

		tokenText := p.Tokenizer.DecodeToken(nextToken)
		genText.WriteString(tokenText)

		if cfg.Stream != nil {
			cfg.Stream(tokenText)
		}

		// Text-level stop detection
		fullText := genText.String()
		for _, ss := range stopStrings {
			if strings.HasSuffix(fullText, ss) {
				trimmed := strings.TrimSuffix(fullText, ss)
				elapsed := time.Since(start)
				tokPerSec := float64(len(generated)) / elapsed.Seconds()
				return trimmed, tokPerSec, nil
			}
		}

		Forward(p.Model, nextToken, pos, p.KVCache, p.RunState)
		pos++
	}

	elapsed := time.Since(start)
	tokPerSec := float64(len(generated)) / elapsed.Seconds()
	return genText.String(), tokPerSec, nil
}
