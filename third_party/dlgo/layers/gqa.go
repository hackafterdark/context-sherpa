package layers

import (
	"math"

	"github.com/computerex/dlgo/ops"
)

// GroupedQueryAttention computes Grouped-Query Attention (GQA) used by modern LLMs.
//
// GQA is a generalization of MHA/MQA where numKVHeads < numHeads.
// Each KV head is shared by (numHeads / numKVHeads) query heads.
//
//   q:     [numHeads * headDim]    — query projection for this position
//   kCache: [][numKVHeads * headDim] — cached keys for all positions [0..seqLen)
//   vCache: [][numKVHeads * headDim] — cached values for all positions [0..seqLen)
//   out:   [numHeads * headDim]    — output buffer
//   seqLen: number of cached positions (including current)
//   numHeads:   total number of query heads
//   numKVHeads: number of key/value heads (≤ numHeads, must divide evenly)
//   headDim:    dimension per head
func GroupedQueryAttention(out, q []float32, kCache, vCache [][]float32,
	seqLen, numHeads, numKVHeads, headDim int) {

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	kvMul := numHeads / numKVHeads
	scores := make([]float32, seqLen)

	ops.Clear(out)

	for h := 0; h < numHeads; h++ {
		kvH := h / kvMul
		qHead := q[h*headDim : (h+1)*headDim]
		headOut := out[h*headDim : (h+1)*headDim]

		for t := 0; t < seqLen; t++ {
			kHead := kCache[t][kvH*headDim : (kvH+1)*headDim]
			scores[t] = ops.DotProduct(qHead, kHead, headDim) * scale
		}

		ops.Softmax(scores[:seqLen])

		for t := 0; t < seqLen; t++ {
			vHead := vCache[t][kvH*headDim : (kvH+1)*headDim]
			ops.AddScaled(headOut, scores[t], vHead, headDim)
		}
	}
}

// CausalSelfAttention computes standard causal self-attention for batched prefill.
// This is the non-GQA variant where numKVHeads == numHeads.
//
//   qFlat: [nPos * dim]  — all positions' Q vectors
//   kFlat: [nPos * dim]  — all positions' K vectors
//   vFlat: [nPos * dim]  — all positions' V vectors
//   out:   [nPos * dim]
func CausalSelfAttention(out, qFlat, kFlat, vFlat []float32, nPos, numHeads, headDim int) {
	MultiHeadAttention(out, qFlat, kFlat, vFlat, nPos, numHeads, headDim, true)
}

// CrossAttention computes encoder-decoder cross-attention.
// Q comes from the decoder (single position), K/V come from the encoder (cached).
//
//   q:       [numHeads * headDim]   — decoder query for current position
//   kEnc:    [encLen * dim]          — encoder key projections
//   vEnc:    [encLen * dim]          — encoder value projections
//   out:     [numHeads * headDim]   — output buffer
//   encLen:  encoder sequence length
func CrossAttention(out, q, kEnc, vEnc []float32, encLen, numHeads, headDim int) {
	dim := numHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	scores := make([]float32, encLen)

	ops.Clear(out)

	for h := 0; h < numHeads; h++ {
		hOff := h * headDim
		qHead := q[hOff : hOff+headDim]
		headOut := out[hOff : hOff+headDim]

		for t := 0; t < encLen; t++ {
			kHead := kEnc[t*dim+hOff : t*dim+hOff+headDim]
			scores[t] = ops.DotProduct(qHead, kHead, headDim) * scale
		}

		ops.Softmax(scores[:encLen])

		for t := 0; t < encLen; t++ {
			vHead := vEnc[t*dim+hOff : t*dim+hOff+headDim]
			ops.AddScaled(headOut, scores[t], vHead, headDim)
		}
	}
}

// SlidingWindowAttention computes attention with a sliding window constraint.
// Each position can only attend to at most `windowSize` previous positions.
// Used by Mistral and some Gemma variants.
func SlidingWindowAttention(out, q []float32, kCache, vCache [][]float32,
	pos, seqLen, numHeads, numKVHeads, headDim, windowSize int) {

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	kvMul := numHeads / numKVHeads

	startPos := 0
	if pos-windowSize+1 > startPos {
		startPos = pos - windowSize + 1
	}
	attnLen := seqLen - startPos
	scores := make([]float32, attnLen)

	ops.Clear(out)

	for h := 0; h < numHeads; h++ {
		kvH := h / kvMul
		qHead := q[h*headDim : (h+1)*headDim]
		headOut := out[h*headDim : (h+1)*headDim]

		for ti := 0; ti < attnLen; ti++ {
			t := startPos + ti
			kHead := kCache[t][kvH*headDim : (kvH+1)*headDim]
			scores[ti] = ops.DotProduct(qHead, kHead, headDim) * scale
		}

		ops.Softmax(scores[:attnLen])

		for ti := 0; ti < attnLen; ti++ {
			t := startPos + ti
			vHead := vCache[t][kvH*headDim : (kvH+1)*headDim]
			ops.AddScaled(headOut, scores[ti], vHead, headDim)
		}
	}
}
