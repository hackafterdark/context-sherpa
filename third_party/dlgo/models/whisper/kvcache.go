package whisper

import (
	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/ops"
)

// KVCache holds the key-value cache for the decoder during autoregressive generation.
// Self-attention KV grows per token; cross-attention KV is filled once from encoder output.
type KVCache struct {
	// Self-attention: one cache per decoder layer, grows each token
	SelfKeys [][]float32 // [nDecLayers][seqLen * dim] - keys for self-attn
	SelfVals [][]float32 // [nDecLayers][seqLen * dim] - values for self-attn
	SelfLen  int        // current sequence length in self-attn cache

	// Cross-attention: filled once from encoder output, same for all decoder steps
	CrossKeys [][]float32 // [nDecLayers][encLen * dim]
	CrossVals [][]float32 // [nDecLayers][encLen * dim]
	CrossLen  int        // encoder sequence length
}

// NewKVCache creates a new KV cache for the given model config.
func NewKVCache(cfg WhisperConfig) *KVCache {
	kc := &KVCache{
		SelfKeys:  make([][]float32, cfg.NDecLayers),
		SelfVals:  make([][]float32, cfg.NDecLayers),
		CrossKeys: make([][]float32, cfg.NDecLayers),
		CrossVals: make([][]float32, cfg.NDecLayers),
	}
	dim := cfg.DModel
	// Pre-allocate self-attn cache for max context (NTextCtx)
	maxSelf := cfg.NTextCtx * dim
	for i := 0; i < cfg.NDecLayers; i++ {
		kc.SelfKeys[i] = make([]float32, 0, maxSelf)
		kc.SelfVals[i] = make([]float32, 0, maxSelf)
	}
	return kc
}

// Reset clears the cache for a new sequence.
func (kc *KVCache) Reset() {
	kc.SelfLen = 0
	for i := range kc.SelfKeys {
		kc.SelfKeys[i] = kc.SelfKeys[i][:0]
		kc.SelfVals[i] = kc.SelfVals[i][:0]
	}
	kc.CrossLen = 0
	for i := range kc.CrossKeys {
		kc.CrossKeys[i] = nil
		kc.CrossVals[i] = nil
	}
}

// FillCross fills the cross-attention KV from encoder output.
// encOut: [encLen × dim]
func (kc *KVCache) FillCross(encOut []float32, encLen, dim int, layers []DecoderLayer) {
	kc.CrossLen = encLen
	for i := range layers {
		layer := &layers[i]
		if layer.CrossWk == nil || layer.CrossWv == nil {
			continue
		}
		kc.CrossKeys[i] = make([]float32, encLen*dim)
		kc.CrossVals[i] = make([]float32, encLen*dim)
		for j := 0; j < encLen; j++ {
			x := encOut[j*dim : (j+1)*dim]
			blas.QMatVecMul(kc.CrossKeys[i][j*dim:(j+1)*dim], layer.CrossWk, x)
			blas.QMatVecMul(kc.CrossVals[i][j*dim:(j+1)*dim], layer.CrossWv, x)
			if layer.CrossBv != nil {
				ops.AddBias(kc.CrossVals[i][j*dim:(j+1)*dim], layer.CrossBv)
			}
		}
	}
}

// AppendSelf appends one token's K and V to the self-attention cache for the given layer.
func (kc *KVCache) AppendSelf(layerIdx int, k, v []float32) {
	if layerIdx >= len(kc.SelfKeys) {
		return
	}
	kc.SelfKeys[layerIdx] = append(kc.SelfKeys[layerIdx], k...)
	kc.SelfVals[layerIdx] = append(kc.SelfVals[layerIdx], v...)
	if len(k) > 0 {
		kc.SelfLen = len(kc.SelfKeys[layerIdx]) / len(k)
	}
}
