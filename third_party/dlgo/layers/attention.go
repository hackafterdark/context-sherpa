package layers

import (
	"math"

	"github.com/computerex/dlgo/ops"
)

// MultiHeadAttention computes multi-head self-attention on flat float32 tensors.
//
// Implements: Attention(Q,K,V) = softmax(Q·K^T / sqrt(d_k)) · V
//
//   qFlat: [nPos * dim]   query projections
//   kFlat: [nPos * dim]   key projections
//   vFlat: [nPos * dim]   value projections
//   out:   [nPos * dim]   output (pre-allocated)
//   nPos:    sequence length
//   nHeads:  number of attention heads
//   headDim: dimension per head (dim = nHeads * headDim)
//   causal:  if true, applies causal masking (each position can only attend to prior positions)
func MultiHeadAttention(out, qFlat, kFlat, vFlat []float32, nPos, nHeads, headDim int, causal bool) {
	dim := nHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	scores := make([]float32, nPos)

	for h := 0; h < nHeads; h++ {
		hOff := h * headDim

		for qi := 0; qi < nPos; qi++ {
			qSrc := qFlat[qi*dim+hOff : qi*dim+hOff+headDim]
			outSlice := out[qi*dim+hOff : qi*dim+hOff+headDim]

			seqLen := nPos
			if causal {
				seqLen = qi + 1
			}

			for ki := 0; ki < seqLen; ki++ {
				kSrc := kFlat[ki*dim+hOff : ki*dim+hOff+headDim]
				scores[ki] = ops.DotProduct(qSrc, kSrc, headDim) * scale
			}

			row := scores[:seqLen]
			ops.Softmax(row)

			ops.Clear(outSlice)
			for ki := 0; ki < seqLen; ki++ {
				vSrc := vFlat[ki*dim+hOff : ki*dim+hOff+headDim]
				ops.AddScaled(outSlice, row[ki], vSrc, headDim)
			}
		}
	}
}

// RelativeMultiHeadAttention computes multi-head self-attention with Shaw/XL-style
// relative positional encoding.
//
//   score(qi, ki) = ((q + bias_u) · K^T + (q + bias_v) · P^T) / sqrt(d_k)
//
// Uses online softmax (FlashAttention-style) to avoid materializing the full score matrix.
//
//   qFlat:   [nPos * dim]   query projections (interleaved heads)
//   kFlat:   [nPos * headDim]   key projections for one head (head-contiguous)
//   vFlat:   [nPos * headDim]   value projections for one head
//   posFlat: [posLen * headDim] positional encodings for one head
//   biasU:   [headDim]          per-head content bias
//   biasV:   [headDim]          per-head position bias
//   attnOut: [nPos * dim]       output (accumulated across heads)
//   nPos:    sequence length
//   posLen:  2*nPos-1 (relative position encoding length)
//   dim:     total model dimension (nHeads * headDim)
//   hOff:    byte offset for this head within the interleaved Q layout
//   scale:   1/sqrt(headDim)
func RelativeMultiHeadAttentionHead(
	qFlat, kHead, vHead, posHead, biasU, biasV, attnOut []float32,
	nPos, posLen, dim, hOff int, scale float32,
) {
	headDim := len(biasU)
	negInf := float32(math.Inf(-1))
	var scoreTile [256]float32
	var acc [256]float32

	tileK := 64
	if headDim > 256 {
		tileK = headDim
	}

	for qi := 0; qi < nPos; qi++ {
		qSrc := qFlat[qi*dim+hOff : qi*dim+hOff+headDim]
		outSlice := attnOut[qi*dim+hOff : qi*dim+hOff+headDim]

		for i := 0; i < headDim; i++ {
			acc[i] = 0
			outSlice[i] = 0
		}

		m := negInf
		var l float32

		for k0 := 0; k0 < nPos; k0 += tileK {
			kN := tileK
			if rem := nPos - k0; rem < kN {
				kN = rem
			}

			tileMax := negInf
			for tj := 0; tj < kN; tj++ {
				ki := k0 + tj
				kSrc := kHead[ki*headDim : (ki+1)*headDim]
				relIdx := ki - qi + nPos - 1
				pSrc := posHead[relIdx*headDim : (relIdx+1)*headDim]

				var dotC, dotP float32
				for d := 0; d < headDim; d++ {
					qv := qSrc[d]
					dotC += (qv + biasU[d]) * kSrc[d]
					dotP += (qv + biasV[d]) * pSrc[d]
				}
				s := (dotC + dotP) * scale
				scoreTile[tj] = s
				if s > tileMax {
					tileMax = s
				}
			}

			newM := m
			if tileMax > newM {
				newM = tileMax
			}
			prevScale := float32(1)
			if m != negInf {
				prevScale = float32(math.Exp(float64(m - newM)))
			}
			if prevScale != 1 {
				for d := 0; d < headDim; d++ {
					acc[d] *= prevScale
				}
				l *= prevScale
			}

			for tj := 0; tj < kN; tj++ {
				ki := k0 + tj
				w := float32(math.Exp(float64(scoreTile[tj] - newM)))
				l += w
				vSrc := vHead[ki*headDim : (ki+1)*headDim]
				for d := 0; d < headDim; d++ {
					acc[d] += w * vSrc[d]
				}
			}
			m = newM
		}

		if l == 0 {
			continue
		}
		invL := 1 / l
		for d := 0; d < headDim; d++ {
			outSlice[d] = acc[d] * invL
		}
	}
}

// RelativePositionalEncoding generates sinusoidal relative positional encodings.
// Returns flat array [posLen * dModel] where posLen = 2*maxLen-1.
//
// NeMo's convention: index 0 = most positive relative position (key far right of query),
// index maxLen-1 = same position, index 2*maxLen-2 = most negative position.
func RelativePositionalEncoding(maxLen, dModel int) []float32 {
	posLen := 2*maxLen - 1
	pe := make([]float32, posLen*dModel)

	for pos := 0; pos < posLen; pos++ {
		relPos := float64(maxLen - 1 - pos)
		for d := 0; d < dModel; d += 2 {
			div := math.Exp(float64(d) * (-math.Log(10000.0) / float64(dModel)))
			angle := relPos * div
			pe[pos*dModel+d] = float32(math.Sin(angle))
			if d+1 < dModel {
				pe[pos*dModel+d+1] = float32(math.Cos(angle))
			}
		}
	}

	return pe
}
