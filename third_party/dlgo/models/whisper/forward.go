package whisper

import (
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/ops"
)

// EncodeAudio runs the Whisper encoder on mel spectrogram features.
// mel: [nFrames × nMels] layout; returns [seqLen × dim] encoder output.
func EncodeAudio(m *WhisperModel, mel []float32) []float32 {
	cfg := m.Config
	dim := cfg.DModel
	nMels := cfg.NMels
	nHeads := cfg.NHeads
	headDim := cfg.HeadDim
	ffnDim := cfg.FFNDim

	nFrames := len(mel) / nMels
	if nFrames <= 0 {
		return nil
	}

	// Transpose mel from [nFrames, nMels] to [nMels, nFrames] for conv
	melT := make([]float32, nMels*nFrames)
	for t := 0; t < nFrames; t++ {
		for ch := 0; ch < nMels; ch++ {
			melT[ch*nFrames+t] = mel[t*nMels+ch]
		}
	}

	conv1Out := conv1DQuantized(m.Conv1Weight, m.Conv1Bias, melT, nMels, nFrames, 3, 1, 1)
	ops.GELU(conv1Out)

	conv2Out := conv1DQuantized(m.Conv2Weight, m.Conv2Bias, conv1Out, dim, nFrames, 3, 2, 1)
	ops.GELU(conv2Out)

	seqLen := (nFrames+2*1-3)/2 + 1
	if seqLen <= 0 {
		seqLen = 1
	}

	// Transpose to [seqLen, dim] and add positional embeddings
	x := make([]float32, seqLen*dim)
	for t := 0; t < seqLen; t++ {
		for c := 0; c < dim; c++ {
			x[t*dim+c] = conv2Out[c*seqLen+t]
		}
		if t < len(m.EncPosEmb)/dim {
			for c := 0; c < dim; c++ {
				x[t*dim+c] += m.EncPosEmb[t*dim+c]
			}
		}
	}

	xNorm := make([]float32, seqLen*dim)
	qAll := make([]float32, seqLen*dim)
	kAll := make([]float32, seqLen*dim)
	vAll := make([]float32, seqLen*dim)
	attnOut := make([]float32, dim)
	ffnHidden := make([]float32, ffnDim)
	ffnOut := make([]float32, dim)
	scores := make([]float32, seqLen)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for l := 0; l < cfg.NEncLayers; l++ {
		layer := &m.EncLayers[l]
		if layer.Wq == nil {
			continue
		}

		for pos := 0; pos < seqLen; pos++ {
			ops.LayerNorm(xNorm[pos*dim:(pos+1)*dim], x[pos*dim:(pos+1)*dim], layer.AttnLnW, layer.AttnLnB, 1e-5)
		}

		blas.QBatchGEMM(qAll, layer.Wq, xNorm, seqLen)
		blas.QBatchGEMM(kAll, layer.Wk, xNorm, seqLen)
		blas.QBatchGEMM(vAll, layer.Wv, xNorm, seqLen)
		if layer.Bq != nil {
			for pos := 0; pos < seqLen; pos++ {
				ops.AddBias(qAll[pos*dim:(pos+1)*dim], layer.Bq)
			}
		}
		if layer.Bv != nil {
			for pos := 0; pos < seqLen; pos++ {
				ops.AddBias(vAll[pos*dim:(pos+1)*dim], layer.Bv)
			}
		}

		// Multi-head self-attention (no causal mask for encoder)
		for i := 0; i < seqLen; i++ {
			ops.Clear(attnOut)
			for h := 0; h < nHeads; h++ {
				qOff := i*dim + h*headDim
				for j := 0; j < seqLen; j++ {
					kOff := j*dim + h*headDim
					scores[j] = ops.DotProduct(qAll[qOff:qOff+headDim], kAll[kOff:kOff+headDim], headDim) * scale
				}
				ops.Softmax(scores[:seqLen])
				headOut := attnOut[h*headDim : (h+1)*headDim]
				for j := 0; j < seqLen; j++ {
					vOff := j*dim + h*headDim
					ops.AddScaled(headOut, scores[j], vAll[vOff:vOff+headDim], headDim)
				}
			}

			blas.QMatVecMul(ffnOut, layer.Wo, attnOut)
			if layer.Bo != nil {
				ops.AddBias(ffnOut, layer.Bo)
			}
			for c := 0; c < dim; c++ {
				x[i*dim+c] += ffnOut[c]
			}
		}

		// FFN per position
		for pos := 0; pos < seqLen; pos++ {
			xSlice := x[pos*dim : (pos+1)*dim]
			ops.LayerNorm(xNorm[pos*dim:(pos+1)*dim], xSlice, layer.FfnLnW, layer.FfnLnB, 1e-5)

			blas.QMatVecMul(ffnHidden, layer.FfnUp, xNorm[pos*dim:(pos+1)*dim])
			if layer.FfnUpBias != nil {
				ops.AddBias(ffnHidden, layer.FfnUpBias)
			}
			ops.GELU(ffnHidden)
			blas.QMatVecMul(ffnOut, layer.FfnDown, ffnHidden)
			if layer.FfnDownB != nil {
				ops.AddBias(ffnOut, layer.FfnDownB)
			}
			for c := 0; c < dim; c++ {
				xSlice[c] += ffnOut[c]
			}
		}
	}

	// Final LayerNorm
	if m.EncLnW != nil {
		for pos := 0; pos < seqLen; pos++ {
			xSlice := x[pos*dim : (pos+1)*dim]
			ops.LayerNorm(xSlice, xSlice, m.EncLnW, m.EncLnB, 1e-5)
		}
	}

	return x
}

// DecoderStep runs one token through the Whisper decoder.
func DecoderStep(m *WhisperModel, encOut []float32, encLen int, token int32, pos int, kc *KVCache) []float32 {
	cfg := m.Config
	dim := cfg.DModel
	nHeads := cfg.NHeads
	headDim := cfg.HeadDim
	ffnDim := cfg.FFNDim

	x := make([]float32, dim)
	if m.TokenEmb != nil {
		_ = m.TokenEmb.DequantizeRow(int(token), x)
	}
	if m.DecPosEmb != nil && pos*dim+dim <= len(m.DecPosEmb) {
		for i := 0; i < dim; i++ {
			x[i] += m.DecPosEmb[pos*dim+i]
		}
	}

	xNorm := make([]float32, dim)
	q := make([]float32, dim)
	k := make([]float32, dim)
	v := make([]float32, dim)
	attnOut := make([]float32, dim)
	ffnHidden := make([]float32, ffnDim)
	ffnOut := make([]float32, dim)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for l := 0; l < cfg.NDecLayers; l++ {
		layer := &m.DecLayers[l]
		if layer.SelfWq == nil {
			continue
		}

		// Self-attention
		ops.LayerNorm(xNorm, x, layer.SelfAttnLnW, layer.SelfAttnLnB, 1e-5)

		blas.QMatVecMul(q, layer.SelfWq, xNorm)
		blas.QMatVecMul(k, layer.SelfWk, xNorm)
		blas.QMatVecMul(v, layer.SelfWv, xNorm)
		if layer.SelfBq != nil {
			ops.AddBias(q, layer.SelfBq)
		}
		if layer.SelfBv != nil {
			ops.AddBias(v, layer.SelfBv)
		}

		if kc != nil {
			kc.AppendSelf(l, k, v)
		}

		selfLen := pos + 1
		if kc != nil && kc.SelfLen > 0 {
			selfLen = kc.SelfLen
		}

		ops.Clear(attnOut)
		if selfLen > 0 && kc != nil && l < len(kc.SelfKeys) && len(kc.SelfKeys[l]) >= selfLen*dim {
			scores := make([]float32, selfLen)
			for h := 0; h < nHeads; h++ {
				qHead := q[h*headDim : (h+1)*headDim]
				for j := 0; j < selfLen; j++ {
					kHead := kc.SelfKeys[l][j*dim+h*headDim : j*dim+(h+1)*headDim]
					scores[j] = ops.DotProduct(qHead, kHead, headDim) * scale
				}
				// Causal mask: future positions get -inf
				for j := pos + 1; j < selfLen; j++ {
					scores[j] = float32(math.Inf(-1))
				}
				ops.Softmax(scores[:selfLen])
				headOut := attnOut[h*headDim : (h+1)*headDim]
				for j := 0; j < selfLen; j++ {
					vHead := kc.SelfVals[l][j*dim+h*headDim : j*dim+(h+1)*headDim]
					ops.AddScaled(headOut, scores[j], vHead, headDim)
				}
			}
		}

		blas.QMatVecMul(ffnOut, layer.SelfWo, attnOut)
		if layer.SelfBo != nil {
			ops.AddBias(ffnOut, layer.SelfBo)
		}
		for i := 0; i < dim; i++ {
			x[i] += ffnOut[i]
		}

		// Cross-attention
		ops.LayerNorm(xNorm, x, layer.CrossAttnLnW, layer.CrossAttnLnB, 1e-5)

		blas.QMatVecMul(q, layer.CrossWq, xNorm)
		if layer.CrossBq != nil {
			ops.AddBias(q, layer.CrossBq)
		}

		ops.Clear(attnOut)
		if kc != nil && l < len(kc.CrossKeys) && kc.CrossKeys[l] != nil {
			crossLen := kc.CrossLen
			scores := make([]float32, crossLen)
			for h := 0; h < nHeads; h++ {
				qHead := q[h*headDim : (h+1)*headDim]
				for j := 0; j < crossLen; j++ {
					kHead := kc.CrossKeys[l][j*dim+h*headDim : j*dim+(h+1)*headDim]
					scores[j] = ops.DotProduct(qHead, kHead, headDim) * scale
				}
				ops.Softmax(scores[:crossLen])
				headOut := attnOut[h*headDim : (h+1)*headDim]
				for j := 0; j < crossLen; j++ {
					vHead := kc.CrossVals[l][j*dim+h*headDim : j*dim+(h+1)*headDim]
					ops.AddScaled(headOut, scores[j], vHead, headDim)
				}
			}
		}

		blas.QMatVecMul(ffnOut, layer.CrossWo, attnOut)
		if layer.CrossBo != nil {
			ops.AddBias(ffnOut, layer.CrossBo)
		}
		for i := 0; i < dim; i++ {
			x[i] += ffnOut[i]
		}

		// FFN
		ops.LayerNorm(xNorm, x, layer.FfnLnW, layer.FfnLnB, 1e-5)
		blas.QMatVecMul(ffnHidden, layer.FfnUp, xNorm)
		if layer.FfnUpBias != nil {
			ops.AddBias(ffnHidden, layer.FfnUpBias)
		}
		ops.GELU(ffnHidden)
		blas.QMatVecMul(ffnOut, layer.FfnDown, ffnHidden)
		if layer.FfnDownB != nil {
			ops.AddBias(ffnOut, layer.FfnDownB)
		}
		for i := 0; i < dim; i++ {
			x[i] += ffnOut[i]
		}
	}

	// Final LayerNorm
	if m.DecLnW != nil {
		ops.LayerNorm(x, x, m.DecLnW, m.DecLnB, 1e-5)
	}

	// Logits
	logits := make([]float32, cfg.NVocab)
	if m.ProjOut != nil {
		blas.QMatVecMul(logits, m.ProjOut, x)
	} else if m.TokenEmb != nil {
		for v := 0; v < cfg.NVocab; v++ {
			row := make([]float32, dim)
			_ = m.TokenEmb.DequantizeRow(v, row)
			logits[v] = ops.DotProduct(x, row, dim)
		}
	}
	return logits
}

// conv1DQuantized performs 1D conv with quantized weight using im2col + GEMM.
// input: [inCh, seqLen], weight qt: [outCh, inCh*kernelSize]
func conv1DQuantized(qt *core.QuantizedTensor, bias []float32, input []float32, inCh, seqLen, kernelSize, stride, pad int) []float32 {
	if qt == nil {
		return nil
	}
	outCh := qt.Rows
	outLen := (seqLen+2*pad-kernelSize)/stride + 1
	if outLen <= 0 {
		outLen = 1
	}

	colSize := inCh * kernelSize
	col := make([]float32, outLen*colSize)
	for t := 0; t < outLen; t++ {
		for ic := 0; ic < inCh; ic++ {
			for ki := 0; ki < kernelSize; ki++ {
				inIdx := t*stride + ki - pad
				idx := t*colSize + ic*kernelSize + ki
				if inIdx >= 0 && inIdx < seqLen {
					col[idx] = input[ic*seqLen+inIdx]
				}
			}
		}
	}

	output := make([]float32, outCh*outLen)
	blas.QBatchGEMM(output, qt, col, outLen)
	if bias != nil {
		for t := 0; t < outLen; t++ {
			for oc := 0; oc < outCh; oc++ {
				output[t*outCh+oc] += bias[oc]
			}
		}
	}

	result := make([]float32, outCh*outLen)
	for t := 0; t < outLen; t++ {
		for oc := 0; oc < outCh; oc++ {
			result[oc*outLen+t] = output[t*outCh+oc]
		}
	}
	return result
}
