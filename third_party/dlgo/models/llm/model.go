package llm

import (
	"github.com/computerex/dlgo/core"
)

// ---------------------------------------------------------------------------
// Layer specification enums — resolved once at load time, zero-cost dispatch.
// ---------------------------------------------------------------------------

// NormKind selects the pre-layer normalization variant.
type NormKind uint8

const (
	NormRMS   NormKind = iota // RMSNorm (LLaMA, Qwen, Gemma, SmolLM2, Qwen3.5)
	NormLayer                 // LayerNorm with bias (Phi-2)
)

// CoreKind selects the per-layer compute core.
type CoreKind uint8

const (
	CoreAttention CoreKind = iota // Standard grouped-query attention
	CoreSSM                       // Gated Delta Network (Qwen3.5 linear attention)
)

// ResKind selects the residual connection + FFN-norm wiring.
type ResKind uint8

const (
	ResStandard    ResKind = iota // Separate FFNNorm; optional PostAttnNorm/PostFFNNorm (LLaMA, Gemma, Qwen)
	ResPostAttnFFN                // PostAttnNorm doubles as FFN norm (Qwen3.5)
	ResParallel                   // Parallel attn+FFN on same pre-norm; X += attn + FFN (Phi-2)
)

// FFNKind selects the feed-forward network variant.
type FFNKind uint8

const (
	FFNSwiGLU FFNKind = iota // gate·SiLU ⊙ up → down (LLaMA, Qwen)
	FFNGeGLU                 // gate·GELU ⊙ up → down (Gemma)
	FFNPlain                 // up → GELU → down (Phi-2)
)

// LayerSpec captures all architectural choices for one transformer layer.
// Resolved once at load time from tensor presence; the forward pass dispatches
// on these fields via switch statements that compile to jump tables.
type LayerSpec struct {
	Norm     NormKind
	Core     CoreKind
	Residual ResKind
	FFN      FFNKind
	GatedQ   bool // Fused Q+gate projection (Qwen3.5 attention layers)
	QKNorm   bool // Per-head QK normalization (Gemma 3, Qwen3)
}

// Layer holds the weights for one transformer block.
type Layer struct {
	Spec LayerSpec // architectural choices, resolved at load time

	// Attention
	AttnNorm     []float32            // [dim] norm weight
	AttnNormBias []float32            // [dim] optional LayerNorm bias (Phi-2)
	Wq           *core.QuantizedTensor // [qDim × dim]
	Wk           *core.QuantizedTensor // [kvDim × dim]
	Wv           *core.QuantizedTensor // [kvDim × dim]
	Wo           *core.QuantizedTensor // [dim × qDim]
	Bq           []float32            // [qDim] optional (Qwen)
	Bk           []float32            // [kvDim] optional
	Bv           []float32            // [kvDim] optional
	Bo           []float32            // [dim] optional attn_output bias (Phi-2)
	AttnQNorm    []float32            // [headDim] optional QK norm (Qwen3/Gemma3)
	AttnKNorm    []float32            // [headDim] optional QK norm (Qwen3/Gemma3)
	AttnGate     *core.QuantizedTensor // [dim × dim] optional gated attention (Qwen3.5)

	PostAttnNorm []float32            // [dim] optional post-attention norm (Gemma 3)
	FFNNorm      []float32            // [dim] norm weight (nil = parallel attn+FFN)
	FFNGate      *core.QuantizedTensor // [ffnDim × dim] w1 (nil = plain MLP)
	FFNUp        *core.QuantizedTensor // [ffnDim × dim] w3
	FFNDown      *core.QuantizedTensor // [dim × ffnDim] w2
	FFNUpBias    []float32            // [ffnDim] optional (Phi-2)
	FFNDownBias  []float32            // [dim] optional (Phi-2)
	PostFFNNorm  []float32            // [dim] optional post-FFN norm (Gemma 3)

	// Gated Delta Net weights — only for Qwen3.5 linear attention layers
	SSMInProj  *core.QuantizedTensor // [dim × qkvDim] fused QKV in-projection (stored via attn_qkv when not standard attention)
	SSMConv1dW []float32            // [channels × convKernel] depthwise conv weights (flat)
	SSMA       []float32            // [numHeads] log(-A) decay parameter
	SSMAlpha   *core.QuantizedTensor // [dim × numHeads] dt/alpha projection
	SSMBeta    *core.QuantizedTensor // [dim × numHeads] beta/learning-rate projection
	SSMDtBias  []float32            // [numHeads] dt bias
	SSMNorm    []float32            // [headVDim] per-head output norm weight
	SSMOut     *core.QuantizedTensor // [dim × valueDim] output projection
}

// Model holds all weights for a decoder-only transformer LLM.
type Model struct {
	Config       ModelConfig
	TokenEmbed   *core.QuantizedTensor // [vocabSize × dim]
	OutputNorm   []float32            // [dim]
	OutputNormBias []float32          // [dim] optional LayerNorm bias (Phi-2)
	Output       *core.QuantizedTensor // [vocabSize × dim] (may tie with TokenEmbed)
	OutputBias   []float32            // [vocabSize] optional
	Layers       []Layer
}
