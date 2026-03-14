package layers

import (
	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/ops"
)

// ConformerFF computes a Macaron-style half-step feed-forward module:
//   FF(x) = x + 0.5 * Linear(SiLU(Linear(LayerNorm(x))))
//
// NeMo's linear layers have no biases (handled by blas.QBatchGEMM).
//
//   input:  [nPos * dim]    flat input tensor
//   out:    [nPos * dim]    output (pre-allocated)
//   lnW, lnB: [dim]        LayerNorm parameters
//   upW:   [ffnDim × dim]  up-projection weights
//   downW: [dim × ffnDim]  down-projection weights
//   lnBuf: [nPos * dim]    scratch for LayerNorm output
//   ffnUp: [nPos * ffnDim] scratch for up-projection output
//   ffnDn: [nPos * dim]    scratch for down-projection output
func ConformerFF(out, input []float32, lnW, lnB []float32,
	upW, downW *core.QuantizedTensor,
	nPos, dim, ffnDim int,
	lnBuf, ffnUp, ffnDn []float32) {

	for t := 0; t < nPos; t++ {
		ops.LayerNorm(lnBuf[t*dim:(t+1)*dim], input[t*dim:(t+1)*dim], lnW, lnB, 1e-5)
	}

	blas.QBatchGEMM(ffnUp, upW, lnBuf[:nPos*dim], nPos)
	ops.SiLU(ffnUp[:nPos*ffnDim])
	blas.QBatchGEMM(ffnDn, downW, ffnUp[:nPos*ffnDim], nPos)

	for i := 0; i < nPos*dim; i++ {
		out[i] = input[i] + 0.5*ffnDn[i]
	}
}

// ConformerConvModule computes the convolution module:
//   Conv(x) = x + Pointwise2(SiLU(DepthwiseConv(GLU(Pointwise1(LayerNorm(x))))))
//
// BatchNorm is assumed fused into the depthwise conv bias during export.
//
//   convLnW, convLnB: [dim]  LayerNorm parameters
//   convPw1:  [2*dim × dim]  pointwise expand (no bias)
//   convDw:   [dim * kernelSize]  depthwise conv weights (F32)
//   convDwB:  [dim]           depthwise conv bias (fused BN)
//   convPw2:  [dim × dim]    pointwise project (no bias)
func ConformerConvModule(out, input []float32,
	convLnW, convLnB []float32,
	convPw1 *core.QuantizedTensor,
	convDw, convDwB []float32,
	convPw2 *core.QuantizedTensor,
	nPos, dim, kernelSize int,
	lnBuf, pw1Buf, gluBuf, dwBuf, pw2Buf, chanBuf []float32) {

	for t := 0; t < nPos; t++ {
		ops.LayerNorm(lnBuf[t*dim:(t+1)*dim], input[t*dim:(t+1)*dim], convLnW, convLnB, 1e-5)
	}

	blas.QBatchGEMM(pw1Buf, convPw1, lnBuf[:nPos*dim], nPos)

	for t := 0; t < nPos; t++ {
		ops.GLU(gluBuf[t*dim:(t+1)*dim], pw1Buf[t*2*dim:(t+1)*2*dim], dim)
	}

	// Transpose to [channels][seqLen] for depthwise conv
	for t := 0; t < nPos; t++ {
		for d := 0; d < dim; d++ {
			chanBuf[d*nPos+t] = gluBuf[t*dim+d]
		}
	}

	DepthwiseConv1D(dwBuf, chanBuf, convDw, convDwB, dim, nPos, kernelSize)
	ops.SiLU(dwBuf[:dim*nPos])

	// Transpose back to [seqLen][channels]
	for t := 0; t < nPos; t++ {
		for d := 0; d < dim; d++ {
			gluBuf[t*dim+d] = dwBuf[d*nPos+t]
		}
	}

	blas.QBatchGEMM(pw2Buf, convPw2, gluBuf[:nPos*dim], nPos)

	for i := 0; i < nPos*dim; i++ {
		out[i] = input[i] + pw2Buf[i]
	}
}
