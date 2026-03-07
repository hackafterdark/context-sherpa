package llm

import (
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/ops"
	"github.com/computerex/dlgo/quant"
)

// RunState holds pre-allocated buffers for inference, avoiding per-token allocations.
type RunState struct {
	X        []float32 // [dim] current activation
	XNorm    []float32 // [dim] normalized activation
	Q        []float32 // [qDim] query projection
	K        []float32 // [kvDim] key projection
	V        []float32 // [kvDim] value projection
	AttnOut  []float32 // [qDim] attention output
	AttnProj []float32 // [dim] output projection
	FFNIn    []float32 // [dim] FFN input (after residual)
	FFNNorm  []float32 // [dim] FFN normalized
	Gate     []float32 // [ffnDim] gate projection
	Up       []float32 // [ffnDim] up projection
	Hidden   []float32 // [ffnDim] gated hidden
	FFNOut   []float32 // [dim] FFN output
	Logits   []float32 // [vocabSize] output logits
	Scores     []float32   // [maxSeqLen] attention scores scratch (legacy)
	HeadScores [][]float32 // [numHeads][maxSeqLen] per-head score buffers for parallel attention

	// Qwen3.5 gated attention: Wq outputs interleaved [Q,gate] per head
	QFull []float32 // [2*qDim] fused Q+gate output (nil for non-gated models)
	QGate []float32 // [qDim] attention gate values (nil for non-gated models)

	// Precomputed RoPE tables (populated by PrecomputeRoPE)
	ropeCos     []float32
	ropeSin     []float32
	ropeHeadDim int
	ropeDim     int // partial RoPE dimension (may be < ropeHeadDim)
	ropeNeox    bool

	// SSM (Gated Delta Net) scratch buffers — nil for pure transformer models
	SSMRun   *SSMRunState
	SSMState *memory.SSMStateCache

	// Worker pool for parallel matmul
	Pool *blas.Pool
}

// NewRunState allocates all buffers for a model.
func NewRunState(cfg ModelConfig, maxSeqLen int) *RunState {
	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim

	headScores := make([][]float32, cfg.NumHeads)
	for h := 0; h < cfg.NumHeads; h++ {
		headScores[h] = make([]float32, maxSeqLen)
	}

	rs := &RunState{
		X:          make([]float32, dim),
		XNorm:      make([]float32, dim),
		Q:          make([]float32, qDim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		AttnOut:    make([]float32, qDim),
		AttnProj:   make([]float32, dim),
		FFNIn:      make([]float32, dim),
		FFNNorm:    make([]float32, dim),
		Gate:       make([]float32, ffnDim),
		Up:         make([]float32, ffnDim),
		Hidden:     make([]float32, ffnDim),
		FFNOut:     make([]float32, dim),
		Logits:     make([]float32, cfg.VocabSize),
		Scores:     make([]float32, maxSeqLen),
		HeadScores: headScores,
		Pool:       blas.DefaultPool(),
	}
	rs.PrecomputeRoPE(maxSeqLen, cfg.RopeDim, cfg.HeadDim, cfg.RopeFreqBase)
	rs.SetRopeNeox(cfg.RopeNeox)

	if cfg.FullAttentionInterval > 0 {
		rs.QFull = make([]float32, 2*qDim)
		rs.QGate = make([]float32, qDim)
	}

	if cfg.FullAttentionInterval > 0 && cfg.SSMInnerSize > 0 {
		numHeads := cfg.SSMTimeStepRank
		headVDim := cfg.SSMInnerSize / numHeads
		headKDim := cfg.SSMStateSize
		valueDim := numHeads * headVDim
		keyDim := numHeads * headKDim
		qkvDim := keyDim*2 + valueDim

		rs.SSMRun = &SSMRunState{
			QKV:   make([]float32, qkvDim),
			Z:     make([]float32, valueDim),
			Alpha: make([]float32, numHeads),
			Beta:  make([]float32, numHeads),
			Y:     make([]float32, valueDim),
		}
		rs.SSMState = memory.NewSSMStateCache(
			cfg.NumLayers, numHeads, headKDim, headVDim,
			qkvDim, cfg.SSMConvKernel,
			func(l int) bool { return isSSMLayer(l, cfg) },
		)
	}

	return rs
}

// isSSMLayer returns true if layer l uses the SSM/delta-net path instead of attention.
func isSSMLayer(l int, cfg ModelConfig) bool {
	if cfg.FullAttentionInterval <= 0 {
		return false
	}
	return ((l + 1) % cfg.FullAttentionInterval) != 0
}

// Forward performs a single-token forward pass through the model.
func Forward(m *Model, token int32, pos int, kv *memory.MultiLayerKVCache, rs *RunState) []float32 {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvMul := numHeads / numKVHeads
	pool := rs.Pool

	_ = m.TokenEmbed.DequantizeRow(int(token), rs.X)
	if cfg.EmbedScale != 0 {
		ops.Scale(rs.X, cfg.EmbedScale)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec

		// Pre-norm
		switch spec.Norm {
		case NormRMS:
			ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)
		case NormLayer:
			ops.LayerNorm(rs.XNorm, rs.X, layer.AttnNorm, layer.AttnNormBias, cfg.RMSNormEps)
		}

		// Layer core
		switch spec.Core {
		case CoreSSM:
			forwardSSMLayer(layer, rs, rs.SSMRun, rs.SSMState.Layers[l], rs.XNorm, cfg, pool)
		case CoreAttention:
			forwardAttention(layer, rs, kv, l, pos, numHeads, numKVHeads, headDim, kvMul, cfg, pool)
		}

		// Residual wiring + FFN
		switch spec.Residual {
		case ResStandard:
			if layer.PostAttnNorm != nil {
				ops.RMSNormInPlace(rs.AttnProj, layer.PostAttnNorm, cfg.RMSNormEps)
			}
			ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
			ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
			forwardFFN(layer, rs, rs.FFNNorm, pool)
			if layer.PostFFNNorm != nil {
				ops.RMSNormInPlace(rs.FFNOut, layer.PostFFNNorm, cfg.RMSNormEps)
			}
			ops.Add(rs.X, rs.FFNIn, rs.FFNOut)

		case ResPostAttnFFN:
			ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
			ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.PostAttnNorm, cfg.RMSNormEps)
			forwardFFN(layer, rs, rs.FFNNorm, pool)
			ops.Add(rs.X, rs.FFNIn, rs.FFNOut)

		case ResParallel:
			forwardFFN(layer, rs, rs.XNorm, pool)
			for i := 0; i < dim; i++ {
				rs.X[i] = rs.X[i] + rs.AttnProj[i] + rs.FFNOut[i]
			}
		}
	}

	// Final norm
	if m.OutputNormBias != nil {
		ops.LayerNorm(rs.X[:dim], rs.X[:dim], m.OutputNorm, m.OutputNormBias, cfg.RMSNormEps)
	} else {
		ops.RMSNormInPlace(rs.X[:dim], m.OutputNorm, cfg.RMSNormEps)
	}

	// Logits
	output := m.Output
	if output == nil {
		output = m.TokenEmbed
	}
	blas.QMatVecMulParallel(rs.Logits, output, rs.X, pool)

	if m.OutputBias != nil {
		ops.AddBias(rs.Logits, m.OutputBias)
	}

	return rs.Logits
}

// forwardAttention runs one full-attention layer. Writes result to rs.AttnProj.
func forwardAttention(
	layer *Layer, rs *RunState, kv *memory.MultiLayerKVCache,
	l, pos, numHeads, numKVHeads, headDim, kvMul int,
	cfg ModelConfig, pool *blas.Pool,
) {
	qDim := numHeads * headDim

	// Q/K/V projections — fused into single dispatch when possible
	if layer.Spec.GatedQ {
		blas.QMatVecMulParallel(rs.QFull, layer.Wq, rs.XNorm, pool)
		for h := 0; h < numHeads; h++ {
			copy(rs.Q[h*headDim:(h+1)*headDim], rs.QFull[h*2*headDim:h*2*headDim+headDim])
			copy(rs.QGate[h*headDim:(h+1)*headDim], rs.QFull[h*2*headDim+headDim:(h+1)*2*headDim])
		}
		blas.QMatVecMulParallel(rs.K, layer.Wk, rs.XNorm, pool)
		blas.QMatVecMulParallel(rs.V, layer.Wv, rs.XNorm, pool)
	} else {
		blas.QTripleMatVecMulParallel(rs.Q, layer.Wq, rs.K, layer.Wk, rs.V, layer.Wv, rs.XNorm, pool)
	}

	if layer.Bq != nil {
		ops.AddBias(rs.Q, layer.Bq)
	}
	if layer.Bk != nil {
		ops.AddBias(rs.K, layer.Bk)
	}
	if layer.Bv != nil {
		ops.AddBias(rs.V, layer.Bv)
	}

	if layer.Spec.QKNorm {
		for h := 0; h < numHeads; h++ {
			ops.RMSNormInPlace(rs.Q[h*headDim:(h+1)*headDim], layer.AttnQNorm, cfg.RMSNormEps)
		}
		for h := 0; h < numKVHeads; h++ {
			ops.RMSNormInPlace(rs.K[h*headDim:(h+1)*headDim], layer.AttnKNorm, cfg.RMSNormEps)
		}
	}

	if rs.ropeCos != nil {
		for h := 0; h < numHeads; h++ {
			rs.ApplyRoPEFast(rs.Q[h*headDim:(h+1)*headDim], pos)
		}
		for h := 0; h < numKVHeads; h++ {
			rs.ApplyRoPEFast(rs.K[h*headDim:(h+1)*headDim], pos)
		}
	} else {
		ops.ApplyRoPEBatch(rs.Q, numHeads, rs.K, numKVHeads, pos, headDim, cfg.RopeFreqBase, cfg.RopeNeox)
	}

	kv.Layers[l].Store(pos, rs.K, rs.V)
	seqLen := pos + 1

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	ops.Clear(rs.AttnOut)

	pool.ParallelFor(numHeads, func(h int) {
		kvH := h / kvMul
		qHead := rs.Q[h*headDim : (h+1)*headDim]
		headOut := rs.AttnOut[h*headDim : (h+1)*headDim]
		scores := rs.HeadScores[h][:seqLen]

		for t := 0; t < seqLen; t++ {
			kHead := kv.Layers[l].Keys[t][kvH*headDim : (kvH+1)*headDim]
			scores[t] = ops.DotProduct(qHead, kHead, headDim) * scale
		}
		quant.SIMDSoftmax(scores)
		for t := 0; t < seqLen; t++ {
			vHead := kv.Layers[l].Vals[t][kvH*headDim : (kvH+1)*headDim]
			ops.AddScaled(headOut, scores[t], vHead, headDim)
		}
	})

	if layer.Spec.GatedQ {
		for i := 0; i < qDim; i++ {
			rs.AttnOut[i] *= ops.Sigmoid(rs.QGate[i])
		}
	}

	blas.QMatVecMulParallel(rs.AttnProj, layer.Wo, rs.AttnOut, pool)
	if layer.Bo != nil {
		ops.AddBias(rs.AttnProj, layer.Bo)
	}
}

// forwardFFN runs the feed-forward network. Result written to rs.FFNOut.
func forwardFFN(layer *Layer, rs *RunState, input []float32, pool *blas.Pool) {
	switch layer.Spec.FFN {
	case FFNSwiGLU:
		blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, input, pool)
		quant.SIMDSwiGLU(rs.Hidden, rs.Gate, rs.Up, len(rs.Gate))
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)

	case FFNGeGLU:
		blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, input, pool)
		ops.GeGLU(rs.Hidden, rs.Gate, rs.Up, len(rs.Gate))
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)

	case FFNPlain:
		blas.QMatVecMulParallel(rs.Up, layer.FFNUp, input, pool)
		if layer.FFNUpBias != nil {
			ops.AddBias(rs.Up, layer.FFNUpBias)
		}
		ops.GELU(rs.Up)
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Up, pool)
	}

	if layer.FFNDownBias != nil {
		ops.AddBias(rs.FFNOut, layer.FFNDownBias)
	}
}
