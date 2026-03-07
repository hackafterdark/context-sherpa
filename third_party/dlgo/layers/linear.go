package layers

import (
	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/ops"
)

// Linear applies a linear projection with quantized weights and optional bias.
//   out = W @ x + bias
//   W: [outDim × inDim] quantized
//   x: [inDim], out: [outDim]
func Linear(out []float32, W *core.QuantizedTensor, x []float32, bias []float32) {
	blas.QMatVecMul(out, W, x)
	if bias != nil {
		ops.AddBias(out, bias)
	}
}

// LinearF32 applies a float32 linear projection with optional bias.
//   W: [outDim * inDim] row-major
func LinearF32(out, W, x []float32, outDim, inDim int, bias []float32) {
	ops.MatVecMul(out, W, x, outDim, inDim)
	if bias != nil {
		ops.AddBias(out, bias)
	}
}

// LinearBatch applies linear projection to a batch of vectors.
//   outFlat: [nPos * outDim], xFlat: [nPos * inDim]
func LinearBatch(outFlat []float32, W *core.QuantizedTensor, xFlat []float32, nPos int, bias []float32) {
	blas.QBatchGEMM(outFlat, W, xFlat, nPos)
	if bias != nil {
		outDim := W.Rows
		for p := 0; p < nPos; p++ {
			ops.AddBias(outFlat[p*outDim:(p+1)*outDim], bias)
		}
	}
}

// Embedding looks up a token embedding from a quantized embedding table.
//   table: [vocabSize × embDim] quantized
//   tokenID: index to look up
//   out: [embDim] output buffer
func Embedding(out []float32, table *core.QuantizedTensor, tokenID int) {
	_ = table.DequantizeRow(tokenID, out)
}

// EmbeddingWithScale looks up an embedding and applies a scale factor.
// Used by Gemma which scales embeddings by sqrt(dim).
func EmbeddingWithScale(out []float32, table *core.QuantizedTensor, tokenID int, scale float32) {
	_ = table.DequantizeRow(tokenID, out)
	if scale != 0 && scale != 1.0 {
		ops.Scale(out, scale)
	}
}
