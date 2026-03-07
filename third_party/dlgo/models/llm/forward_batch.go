package llm

import (
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/ops"
	"github.com/computerex/dlgo/quant"
)

// BatchState holds pre-allocated buffers for batch (prefill) forward passes.
type BatchState struct {
	maxPos int
	dim    int
	qDim   int
	kvDim  int
	ffnDim int

	XBatch     []float32 // [maxPos * dim]
	XNormBatch []float32 // [maxPos * dim]
	QBatch     []float32 // [maxPos * qDim]
	KBatch     []float32 // [maxPos * kvDim]
	VBatch     []float32 // [maxPos * kvDim]
	AttnBatch  []float32 // [maxPos * qDim]
	ProjBatch  []float32 // [maxPos * dim]
	FFNInBatch []float32 // [maxPos * dim]
	NormBatch  []float32 // [maxPos * dim]
	GateBatch  []float32 // [maxPos * ffnDim]
	UpBatch    []float32 // [maxPos * ffnDim]
	HidBatch   []float32 // [maxPos * ffnDim]
	FFNBatch   []float32 // [maxPos * dim]

	Q8Buf []byte // pre-allocated Q8 quantization buffer
}

// NewBatchState allocates batch buffers for up to maxPos positions.
func NewBatchState(cfg ModelConfig, maxPos int) *BatchState {
	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim
	maxDim := dim
	if ffnDim > maxDim {
		maxDim = ffnDim
	}
	q8Size := quant.Q8BufferSize(2, maxDim) // Q4_0 Q8 size as base
	q8SizeK := quant.Q8BufferSize(12, maxDim) // K-quant Q8 size
	if q8SizeK > q8Size {
		q8Size = q8SizeK
	}

	return &BatchState{
		maxPos:     maxPos,
		dim:        dim,
		qDim:       qDim,
		kvDim:      kvDim,
		ffnDim:     ffnDim,
		XBatch:     make([]float32, maxPos*dim),
		XNormBatch: make([]float32, maxPos*dim),
		QBatch:     make([]float32, maxPos*qDim),
		KBatch:     make([]float32, maxPos*kvDim),
		VBatch:     make([]float32, maxPos*kvDim),
		AttnBatch:  make([]float32, maxPos*qDim),
		ProjBatch:  make([]float32, maxPos*dim),
		FFNInBatch: make([]float32, maxPos*dim),
		NormBatch:  make([]float32, maxPos*dim),
		GateBatch:  make([]float32, maxPos*ffnDim),
		UpBatch:    make([]float32, maxPos*ffnDim),
		HidBatch:   make([]float32, maxPos*ffnDim),
		FFNBatch:   make([]float32, maxPos*dim),
		Q8Buf:      make([]byte, maxPos*q8Size),
	}
}

// ForwardBatch processes multiple tokens in a single pass (prefill).
// Returns logits for the last position only. Fills the KV cache for all positions.
func ForwardBatch(m *Model, tokens []int32, startPos int, kv *memory.MultiLayerKVCache, rs *RunState, bs *BatchState) []float32 {
	cfg := m.Config
	nPos := len(tokens)
	if nPos == 0 {
		return rs.Logits
	}
	if nPos == 1 {
		return Forward(m, tokens[0], startPos, kv, rs)
	}

	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	kvMul := numHeads / numKVHeads
	pool := rs.Pool

	// Embed all tokens
	for p := 0; p < nPos; p++ {
		_ = m.TokenEmbed.DequantizeRow(int(tokens[p]), bs.XBatch[p*dim:(p+1)*dim])
	}
	if cfg.EmbedScale != 0 {
		for p := 0; p < nPos; p++ {
			ops.Scale(bs.XBatch[p*dim:(p+1)*dim], cfg.EmbedScale)
		}
	}

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec

		if spec.Core == CoreSSM {
			for p := 0; p < nPos; p++ {
				copy(rs.X, bs.XBatch[p*dim:(p+1)*dim])
				ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)
				forwardSSMLayer(layer, rs, rs.SSMRun, rs.SSMState.Layers[l], rs.XNorm, cfg, pool)
				copy(bs.ProjBatch[p*dim:(p+1)*dim], rs.AttnProj)
			}
			batchResidualFFN(layer, bs, rs, nPos, dim, cfg, pool)
			continue
		}

		// Batch norm
		for p := 0; p < nPos; p++ {
			x := bs.XBatch[p*dim : (p+1)*dim]
			xn := bs.XNormBatch[p*dim : (p+1)*dim]
			switch spec.Norm {
			case NormRMS:
				ops.RMSNorm(xn, x, layer.AttnNorm, cfg.RMSNormEps)
			case NormLayer:
				ops.LayerNorm(xn, x, layer.AttnNorm, layer.AttnNormBias, cfg.RMSNormEps)
			}
		}

		// Batch Q/K/V projections (fused: quantize input once, single dispatch)
		qDim := numHeads * headDim
		blas.QTripleBatchGEMMParallel(
			bs.QBatch[:nPos*qDim], layer.Wq,
			bs.KBatch[:nPos*kvDim], layer.Wk,
			bs.VBatch[:nPos*kvDim], layer.Wv,
			bs.XNormBatch[:nPos*dim], nPos, pool,
		)

		// Per-position: bias, QK norm, RoPE, KV store
		for p := 0; p < nPos; p++ {
			pos := startPos + p
			qp := bs.QBatch[p*qDim : (p+1)*qDim]
			kp := bs.KBatch[p*kvDim : (p+1)*kvDim]
			vp := bs.VBatch[p*kvDim : (p+1)*kvDim]

			if layer.Bq != nil {
				ops.AddBias(qp, layer.Bq)
			}
			if layer.Bk != nil {
				ops.AddBias(kp, layer.Bk)
			}
			if layer.Bv != nil {
				ops.AddBias(vp, layer.Bv)
			}

			if spec.QKNorm {
				for h := 0; h < numHeads; h++ {
					ops.RMSNormInPlace(qp[h*headDim:(h+1)*headDim], layer.AttnQNorm, cfg.RMSNormEps)
				}
				for h := 0; h < numKVHeads; h++ {
					ops.RMSNormInPlace(kp[h*headDim:(h+1)*headDim], layer.AttnKNorm, cfg.RMSNormEps)
				}
			}

			if rs.ropeCos != nil {
				for h := 0; h < numHeads; h++ {
					rs.ApplyRoPEFast(qp[h*headDim:(h+1)*headDim], pos)
				}
				for h := 0; h < numKVHeads; h++ {
					rs.ApplyRoPEFast(kp[h*headDim:(h+1)*headDim], pos)
				}
			} else {
				ops.ApplyRoPEBatch(qp, numHeads, kp, numKVHeads, pos, headDim, cfg.RopeFreqBase, cfg.RopeNeox)
			}

			kv.Layers[l].Store(pos, kp, vp)
		}

		// Batch causal attention
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		for p := 0; p < nPos; p++ {
			ops.Clear(bs.AttnBatch[p*qDim : (p+1)*qDim])
		}

		pool.ParallelFor(numHeads, func(h int) {
			kvH := h / kvMul
			for p := 0; p < nPos; p++ {
				pos := startPos + p
				seqLen := pos + 1
				qHead := bs.QBatch[p*qDim+h*headDim : p*qDim+(h+1)*headDim]
				headOut := bs.AttnBatch[p*qDim+h*headDim : p*qDim+(h+1)*headDim]
				scores := rs.HeadScores[h][:seqLen]

				for t := 0; t < seqLen; t++ {
					kHead := kv.Layers[l].Keys[t][kvH*headDim : (kvH+1)*headDim]
					scores[t] = ops.DotProduct(qHead, kHead, headDim) * scale
				}
				ops.Softmax(scores)
				for t := 0; t < seqLen; t++ {
					vHead := kv.Layers[l].Vals[t][kvH*headDim : (kvH+1)*headDim]
					ops.AddScaled(headOut, scores[t], vHead, headDim)
				}
			}
		})

		// Batch output projection
		blas.QBatchGEMMParallel(bs.ProjBatch[:nPos*dim], layer.Wo, bs.AttnBatch[:nPos*qDim], nPos, pool)
		for p := 0; p < nPos; p++ {
			if layer.Bo != nil {
				ops.AddBias(bs.ProjBatch[p*dim:(p+1)*dim], layer.Bo)
			}
		}

		// Residual + FFN
		batchResidualFFN(layer, bs, rs, nPos, dim, cfg, pool)
	}

	// Final norm + logits (last position only)
	lastX := bs.XBatch[(nPos-1)*dim : nPos*dim]
	copy(rs.X, lastX)

	if m.OutputNormBias != nil {
		ops.LayerNorm(rs.X[:dim], rs.X[:dim], m.OutputNorm, m.OutputNormBias, cfg.RMSNormEps)
	} else {
		ops.RMSNormInPlace(rs.X[:dim], m.OutputNorm, cfg.RMSNormEps)
	}

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

// batchResidualFFN applies residual wiring + FFN for all positions in the batch.
func batchResidualFFN(layer *Layer, bs *BatchState, rs *RunState, nPos, dim int, cfg ModelConfig, pool *blas.Pool) {
	spec := &layer.Spec
	ffnDim := cfg.FFNDim

	switch spec.Residual {
	case ResStandard:
		for p := 0; p < nPos; p++ {
			proj := bs.ProjBatch[p*dim : (p+1)*dim]
			x := bs.XBatch[p*dim : (p+1)*dim]
			ffnIn := bs.FFNInBatch[p*dim : (p+1)*dim]
			if layer.PostAttnNorm != nil {
				ops.RMSNormInPlace(proj, layer.PostAttnNorm, cfg.RMSNormEps)
			}
			ops.Add(ffnIn, x, proj)
			ops.RMSNorm(bs.NormBatch[p*dim:(p+1)*dim], ffnIn, layer.FFNNorm, cfg.RMSNormEps)
		}
		batchFFN(layer, bs, rs, nPos, dim, ffnDim, bs.NormBatch, pool)
		for p := 0; p < nPos; p++ {
			if layer.PostFFNNorm != nil {
				ops.RMSNormInPlace(bs.FFNBatch[p*dim:(p+1)*dim], layer.PostFFNNorm, cfg.RMSNormEps)
			}
			ops.Add(bs.XBatch[p*dim:(p+1)*dim], bs.FFNInBatch[p*dim:(p+1)*dim], bs.FFNBatch[p*dim:(p+1)*dim])
		}

	case ResPostAttnFFN:
		for p := 0; p < nPos; p++ {
			x := bs.XBatch[p*dim : (p+1)*dim]
			proj := bs.ProjBatch[p*dim : (p+1)*dim]
			ffnIn := bs.FFNInBatch[p*dim : (p+1)*dim]
			ops.Add(ffnIn, x, proj)
			ops.RMSNorm(bs.NormBatch[p*dim:(p+1)*dim], ffnIn, layer.PostAttnNorm, cfg.RMSNormEps)
		}
		batchFFN(layer, bs, rs, nPos, dim, ffnDim, bs.NormBatch, pool)
		for p := 0; p < nPos; p++ {
			ops.Add(bs.XBatch[p*dim:(p+1)*dim], bs.FFNInBatch[p*dim:(p+1)*dim], bs.FFNBatch[p*dim:(p+1)*dim])
		}

	case ResParallel:
		batchFFN(layer, bs, rs, nPos, dim, ffnDim, bs.XNormBatch, pool)
		for p := 0; p < nPos; p++ {
			x := bs.XBatch[p*dim : (p+1)*dim]
			proj := bs.ProjBatch[p*dim : (p+1)*dim]
			ffn := bs.FFNBatch[p*dim : (p+1)*dim]
			for i := 0; i < dim; i++ {
				x[i] = x[i] + proj[i] + ffn[i]
			}
		}
	}
}

// batchFFN runs the FFN for all positions using batch GEMM.
func batchFFN(layer *Layer, bs *BatchState, rs *RunState, nPos, dim, ffnDim int, inputBatch []float32, pool *blas.Pool) {
	switch layer.Spec.FFN {
	case FFNSwiGLU:
		blas.QDualBatchGEMMParallel(
			bs.GateBatch[:nPos*ffnDim], layer.FFNGate,
			bs.UpBatch[:nPos*ffnDim], layer.FFNUp,
			inputBatch[:nPos*dim], nPos, pool,
		)
		for p := 0; p < nPos; p++ {
			quant.SIMDSwiGLU(
				bs.HidBatch[p*ffnDim:(p+1)*ffnDim],
				bs.GateBatch[p*ffnDim:(p+1)*ffnDim],
				bs.UpBatch[p*ffnDim:(p+1)*ffnDim],
				ffnDim,
			)
		}
		blas.QBatchGEMMParallel(bs.FFNBatch[:nPos*dim], layer.FFNDown, bs.HidBatch[:nPos*ffnDim], nPos, pool)

	case FFNGeGLU:
		blas.QDualBatchGEMMParallel(
			bs.GateBatch[:nPos*ffnDim], layer.FFNGate,
			bs.UpBatch[:nPos*ffnDim], layer.FFNUp,
			inputBatch[:nPos*dim], nPos, pool,
		)
		for p := 0; p < nPos; p++ {
			ops.GeGLU(
				bs.HidBatch[p*ffnDim:(p+1)*ffnDim],
				bs.GateBatch[p*ffnDim:(p+1)*ffnDim],
				bs.UpBatch[p*ffnDim:(p+1)*ffnDim],
				ffnDim,
			)
		}
		blas.QBatchGEMMParallel(bs.FFNBatch[:nPos*dim], layer.FFNDown, bs.HidBatch[:nPos*ffnDim], nPos, pool)

	case FFNPlain:
		blas.QBatchGEMMParallel(bs.UpBatch[:nPos*ffnDim], layer.FFNUp, inputBatch[:nPos*dim], nPos, pool)
		for p := 0; p < nPos; p++ {
			up := bs.UpBatch[p*ffnDim : (p+1)*ffnDim]
			if layer.FFNUpBias != nil {
				ops.AddBias(up, layer.FFNUpBias)
			}
			ops.GELU(up)
		}
		blas.QBatchGEMMParallel(bs.FFNBatch[:nPos*dim], layer.FFNDown, bs.UpBatch[:nPos*ffnDim], nPos, pool)
	}

	if layer.FFNDownBias != nil {
		for p := 0; p < nPos; p++ {
			ops.AddBias(bs.FFNBatch[p*dim:(p+1)*dim], layer.FFNDownBias)
		}
	}
}
