// Package whisper implements Whisper speech recognition model inference.
//
// Supports Whisper Tiny, Base, Small, and Medium models in GGUF format.
// Architecture: audio encoder (mel → transformer) + text decoder (autoregressive transformer).
package whisper

import (
	"fmt"

	"github.com/computerex/dlgo/core"
)

// WhisperConfig holds architecture parameters.
type WhisperConfig struct {
	DModel      int // Embedding dimension (384 tiny, 512 base, 768 small, 1024 medium)
	NHeads      int
	HeadDim     int // DModel / NHeads
	NEncLayers  int // 4 (tiny), 6 (base), 12 (small), 24 (medium)
	NDecLayers  int // Same as enc
	NVocab      int // 51865 (multilingual) or 51864 (english)
	NMels       int // 80 (tiny/base/small/medium) or 128 (large)
	NAudioCtx   int // 1500 (encoder position embedding length)
	NTextCtx    int // 448 (decoder position embedding length)
	FFNDim       int // 4 * DModel
	EmbeddingDim int // Alias for DModel
	NumMels      int // Alias for NMels
	VocabSize    int // Alias for NVocab
	NumEncLayers int // Alias for NEncLayers
	NumDecLayers int // Alias for NDecLayers
	NumHeads     int // Alias for NHeads
}

// EncoderLayer holds weights for one Whisper encoder transformer layer.
type EncoderLayer struct {
	AttnLnW  []float32 // [dim] LayerNorm weight
	AttnLnB  []float32 // [dim] LayerNorm bias
	Wq       *core.QuantizedTensor
	Wk       *core.QuantizedTensor
	Wv       *core.QuantizedTensor
	Wo       *core.QuantizedTensor
	Bq       []float32
	Bv       []float32
	Bo       []float32
	FfnLnW   []float32 // [dim]
	FfnLnB   []float32 // [dim]
	FfnUp    *core.QuantizedTensor // [ffnDim × dim]
	FfnDown  *core.QuantizedTensor // [dim × ffnDim]
	FfnUpBias []float32
	FfnDownB []float32
}

// DecoderLayer holds weights for one Whisper decoder transformer layer.
type DecoderLayer struct {
	SelfAttnLnW  []float32
	SelfAttnLnB  []float32
	SelfWq       *core.QuantizedTensor
	SelfWk       *core.QuantizedTensor
	SelfWv       *core.QuantizedTensor
	SelfWo       *core.QuantizedTensor
	SelfBq       []float32
	SelfBv       []float32
	SelfBo       []float32

	CrossAttnLnW []float32
	CrossAttnLnB []float32
	CrossWq      *core.QuantizedTensor
	CrossWk      *core.QuantizedTensor
	CrossWv      *core.QuantizedTensor
	CrossWo      *core.QuantizedTensor
	CrossBq      []float32
	CrossBv      []float32
	CrossBo      []float32

	FfnLnW   []float32
	FfnLnB   []float32
	FfnUp    *core.QuantizedTensor
	FfnDown  *core.QuantizedTensor
	FfnUpBias []float32
	FfnDownB []float32
}

// WhisperModel holds all weights for Whisper inference.
type WhisperModel struct {
	Config WhisperConfig

	// Encoder
	Conv1Weight  *core.QuantizedTensor // [512 × 80 × 3] or [dim × nMels × 3]
	Conv1Bias    []float32
	Conv2Weight  *core.QuantizedTensor // [512 × 512 × 3]
	Conv2Bias    []float32
	EncPosEmb    []float32 // [1500 × dim] positional embeddings
	EncLayers    []EncoderLayer
	EncLnW       []float32 // final LayerNorm
	EncLnB       []float32

	// Decoder
	TokenEmb   *core.QuantizedTensor // [vocabSize × dim]
	DecPosEmb  []float32             // [448 × dim]
	DecLayers  []DecoderLayer
	DecLnW     []float32 // final LayerNorm
	DecLnB     []float32
	ProjOut    *core.QuantizedTensor // [vocabSize × dim]
}

// LoadWhisperModel loads a Whisper model from a GGUF file.
func LoadWhisperModel(path string) (*WhisperModel, error) {
	m, err := loadWhisperFromGGUF(path)
	if err != nil {
		return nil, fmt.Errorf("load whisper: %w", err)
	}
	return m, nil
}

func parseWhisperConfig(md map[string]interface{}) WhisperConfig {
	cfg := WhisperConfig{
		NMels:       80,
		NEncLayers:  6,
		NDecLayers:  6,
		DModel:      512,
		NHeads:      8,
		FFNDim:      2048,
		NVocab:      51865,
		NTextCtx:    448,
		NAudioCtx:   1500,
	}

	// Override from GGUF metadata
	if v, ok := intMeta(md, "whisper.encoder.block_count"); ok {
		cfg.NEncLayers = v
	}
	if v, ok := intMeta(md, "whisper.decoder.block_count"); ok {
		cfg.NDecLayers = v
	}
	if v, ok := intMeta(md, "whisper.embedding_length"); ok {
		cfg.DModel = v
	}
	if v, ok := intMeta(md, "whisper.attention.head_count"); ok {
		cfg.NHeads = v
	}
	if v, ok := intMeta(md, "whisper.feed_forward_length"); ok {
		cfg.FFNDim = v
	}

	cfg.HeadDim = cfg.DModel / cfg.NHeads
	cfg.EmbeddingDim = cfg.DModel
	cfg.NumMels = cfg.NMels
	cfg.VocabSize = cfg.NVocab
	cfg.NumEncLayers = cfg.NEncLayers
	cfg.NumDecLayers = cfg.NDecLayers
	cfg.NumHeads = cfg.NHeads

	return cfg
}

func intMeta(md map[string]interface{}, key string) (int, bool) {
	v, ok := md[key]
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
	case uint32:
		return int(x), true
	case int32:
		return int(x), true
	case uint64:
		return int(x), true
	case int64:
		return int(x), true
	default:
		return 0, false
	}
}
