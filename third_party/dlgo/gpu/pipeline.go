//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

// GpuPipeline bundles a model on GPU with all state needed for inference.
type GpuPipeline struct {
	CPUModel  *llm.Model
	GpuModel  *GpuModel
	Tokenizer *llm.Tokenizer
	KVCache   *GpuKVCache
	RunState  *GpuRunState
	MaxSeqLen int
	LogitsBuf []float32
}

// UploadModel copies all model weights to GPU memory.
func UploadModel(m *llm.Model) (*GpuModel, error) {
	gm := &GpuModel{
		Layers: make([]GpuLayer, len(m.Layers)),
	}

	var err error
	gm.TokenEmbed, err = UploadTensor(m.TokenEmbed)
	if err != nil {
		return nil, fmt.Errorf("upload token_embed: %w", err)
	}

	if m.OutputNorm != nil {
		gm.OutputNorm, err = UploadF32Slice(m.OutputNorm)
		if err != nil {
			return nil, fmt.Errorf("upload output_norm: %w", err)
		}
	}
	if m.OutputNormBias != nil {
		gm.OutputNormBias, err = UploadF32Slice(m.OutputNormBias)
		if err != nil {
			return nil, fmt.Errorf("upload output_norm_bias: %w", err)
		}
	}

	gm.Output, err = UploadTensor(m.Output)
	if err != nil {
		return nil, fmt.Errorf("upload output: %w", err)
	}

	if m.OutputBias != nil {
		gm.OutputBias, err = UploadF32Slice(m.OutputBias)
		if err != nil {
			return nil, fmt.Errorf("upload output_bias: %w", err)
		}
	}

	for l := 0; l < len(m.Layers); l++ {
		cl := &m.Layers[l]
		gl := &gm.Layers[l]

		if cl.AttnNorm != nil {
			gl.AttnNorm, err = UploadF32Slice(cl.AttnNorm)
			if err != nil {
				return nil, fmt.Errorf("layer %d attn_norm: %w", l, err)
			}
		}
		if cl.AttnNormBias != nil {
			gl.AttnNormBias, err = UploadF32Slice(cl.AttnNormBias)
			if err != nil {
				return nil, fmt.Errorf("layer %d attn_norm_bias: %w", l, err)
			}
		}

		gl.Wq, err = UploadTensor(cl.Wq)
		if err != nil {
			return nil, fmt.Errorf("layer %d wq: %w", l, err)
		}
		gl.Wk, err = UploadTensor(cl.Wk)
		if err != nil {
			return nil, fmt.Errorf("layer %d wk: %w", l, err)
		}
		gl.Wv, err = UploadTensor(cl.Wv)
		if err != nil {
			return nil, fmt.Errorf("layer %d wv: %w", l, err)
		}
		gl.Wo, err = UploadTensor(cl.Wo)
		if err != nil {
			return nil, fmt.Errorf("layer %d wo: %w", l, err)
		}

		if cl.Bq != nil {
			gl.Bq, _ = UploadF32Slice(cl.Bq)
		}
		if cl.Bk != nil {
			gl.Bk, _ = UploadF32Slice(cl.Bk)
		}
		if cl.Bv != nil {
			gl.Bv, _ = UploadF32Slice(cl.Bv)
		}
		if cl.Bo != nil {
			gl.Bo, _ = UploadF32Slice(cl.Bo)
		}
		if cl.AttnQNorm != nil {
			gl.AttnQNorm, _ = UploadF32Slice(cl.AttnQNorm)
		}
		if cl.AttnKNorm != nil {
			gl.AttnKNorm, _ = UploadF32Slice(cl.AttnKNorm)
		}
		if cl.PostAttnNorm != nil {
			gl.PostAttnNorm, _ = UploadF32Slice(cl.PostAttnNorm)
		}
		if cl.FFNNorm != nil {
			gl.FFNNorm, _ = UploadF32Slice(cl.FFNNorm)
		}

		gl.FFNGate, _ = UploadTensor(cl.FFNGate)
		gl.FFNUp, err = UploadTensor(cl.FFNUp)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_up: %w", l, err)
		}
		gl.FFNDown, err = UploadTensor(cl.FFNDown)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_down: %w", l, err)
		}

		if cl.FFNUpBias != nil {
			gl.FFNUpBias, _ = UploadF32Slice(cl.FFNUpBias)
		}
		if cl.FFNDownBias != nil {
			gl.FFNDownBias, _ = UploadF32Slice(cl.FFNDownBias)
		}
		if cl.PostFFNNorm != nil {
			gl.PostFFNNorm, _ = UploadF32Slice(cl.PostFFNNorm)
		}
	}

	return gm, nil
}

// NewGpuPipeline creates a GPU-accelerated inference pipeline.
func NewGpuPipeline(cpuPipeline *llm.Pipeline) (*GpuPipeline, error) {
	if err := Init(); err != nil {
		return nil, err
	}

	m := cpuPipeline.Model
	cfg := m.Config

	fmt.Printf("[dlgo/gpu] Uploading model to %s (%.0f MB VRAM)...\n",
		DeviceName(), float64(VRAMBytes())/(1024*1024))

	gm, err := UploadModel(m)
	if err != nil {
		return nil, fmt.Errorf("gpu upload: %w", err)
	}

	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim

	rs := NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
	kv := NewGpuKVCache(cfg.NumLayers, cpuPipeline.MaxSeqLen, kvDim)

	fmt.Printf("[dlgo/gpu] Model loaded to GPU (%d layers)\n", cfg.NumLayers)

	return &GpuPipeline{
		CPUModel:  m,
		GpuModel:  gm,
		Tokenizer: cpuPipeline.Tokenizer,
		KVCache:   kv,
		RunState:  rs,
		MaxSeqLen: cpuPipeline.MaxSeqLen,
		LogitsBuf: make([]float32, cfg.VocabSize),
	}, nil
}

// GenerateResult holds detailed output from a GPU generation run.
type GenerateResult struct {
	Text           string
	Tokens         []int32
	TokensPerSec   float64
	PrefillTimeMs  float64
	GenerateTimeMs float64
	TotalTokens    int
	PromptTokens   int
}

// GenerateDetailed runs generation on GPU with detailed timing.
func (p *GpuPipeline) GenerateDetailed(prompt string, cfg llm.GenerateConfig) (*GenerateResult, error) {
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer produced no tokens")
	}
	if len(tokens) >= p.MaxSeqLen {
		return nil, fmt.Errorf("prompt too long: %d tokens (max %d)", len(tokens), p.MaxSeqLen)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	p.KVCache.Reset()

	// Prefill: process prompt tokens one at a time on GPU
	prefillStart := time.Now()
	for i, tok := range tokens {
		GpuForward(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf)
	}
	Sync()
	prefillMs := float64(time.Since(prefillStart).Microseconds()) / 1000.0

	// Generate
	genStart := time.Now()
	var generated []int32
	var recentTokens []int32

	pos := len(tokens)
	nextToken := ops.SampleToken(p.LogitsBuf, cfg.Sampler, recentTokens, rng)
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
		if lastTok == p.CPUModel.Config.EOS {
			break
		}
		for _, stop := range p.CPUModel.Config.StopTokens {
			if lastTok == stop {
				goto done
			}
		}

		GpuForward(p.CPUModel, p.GpuModel, lastTok, pos, p.KVCache, p.RunState, p.LogitsBuf)
		pos++

		nextToken = ops.SampleToken(p.LogitsBuf, cfg.Sampler, recentTokens, rng)
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
	Sync()
	genMs := float64(time.Since(genStart).Microseconds()) / 1000.0

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
