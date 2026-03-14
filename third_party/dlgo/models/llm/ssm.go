package llm

import (
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/ops"
)

// SSMRunState holds pre-allocated scratch buffers for SSM (Gated Delta Net) layers.
type SSMRunState struct {
	QKV   []float32 // [qkvDim] in-projection output (goes through conv)
	Z     []float32 // [valueDim] gate projection output
	Alpha []float32 // [numHeads] raw alpha (decay param)
	Beta  []float32 // [numHeads] raw beta (learning rate)
	Y     []float32 // [valueDim] attention/SSM output
}

// forwardSSMLayer runs one Gated Delta Net layer for single-token autoregressive inference.
//
// Implements the recurrent delta rule with error correction:
//
//	S[h] = exp(g[h]) * S[h]                         // decay
//	v_pred = S^T @ k                                 // predict value from key
//	delta  = v - v_pred                              // error signal
//	S[h]  += sigmoid(beta[h]) * outer(k, delta)      // error-corrected update
//	out[h] = S^T @ (q / sqrt(headKDim))              // scaled output
func forwardSSMLayer(
	layer *Layer,
	rs *RunState,
	ssm *SSMRunState,
	ssmState *memory.SSMLayerState,
	xnorm []float32,
	cfg ModelConfig,
	pool *blas.Pool,
) []float32 {
	qkvDim := ssmState.Channels
	numHeads := ssmState.NumHeads
	headKDim := ssmState.HeadKDim
	headVDim := ssmState.HeadVDim
	convK := ssmState.ConvK
	valueDim := numHeads * headVDim
	keyDim := numHeads * headKDim

	// 1. In-projection: dim -> qkvDim
	blas.QMatVecMulParallel(ssm.QKV, layer.SSMInProj, xnorm, pool)

	// 2. Gate projection: dim -> valueDim
	blas.QMatVecMulParallel(ssm.Z, layer.AttnGate, xnorm, pool)

	// 3. Alpha/Beta projections: dim -> numHeads
	blas.QMatVecMul(ssm.Alpha, layer.SSMAlpha, xnorm)
	blas.QMatVecMul(ssm.Beta, layer.SSMBeta, xnorm)

	// 4. Causal conv1d: shift buffer, store current input, depthwise conv
	buf := ssmState.ConvBuf
	copy(buf[0:(convK-1)*qkvDim], buf[qkvDim:convK*qkvDim])
	copy(buf[(convK-1)*qkvDim:convK*qkvDim], ssm.QKV[:qkvDim])

	w := layer.SSMConv1dW
	for c := 0; c < qkvDim; c++ {
		var acc float32
		wOff := c * convK
		for k := 0; k < convK; k++ {
			acc += buf[k*qkvDim+c] * w[wOff+k]
		}
		ssm.QKV[c] = acc
	}

	// 5. SiLU activation
	ops.SiLU(ssm.QKV[:qkvDim])

	// 6. Split into Q, K, V
	q := ssm.QKV[:keyDim]
	k := ssm.QKV[keyDim : 2*keyDim]
	v := ssm.QKV[2*keyDim : 2*keyDim+valueDim]

	// 7. Compute decay (g) and learning rate (beta)
	for h := 0; h < numHeads; h++ {
		a := ssm.Alpha[h]
		if layer.SSMDtBias != nil {
			a += layer.SSMDtBias[h]
		}
		softplusA := float32(math.Log(1.0 + math.Exp(float64(a))))
		ssm.Alpha[h] = layer.SSMA[h] * softplusA

		ssm.Beta[h] = ops.Sigmoid(ssm.Beta[h])
	}

	// 8. L2-normalize Q and K per head
	for h := 0; h < numHeads; h++ {
		l2Normalize(q[h*headKDim:(h+1)*headKDim], cfg.RMSNormEps)
		l2Normalize(k[h*headKDim:(h+1)*headKDim], cfg.RMSNormEps)
	}

	// 9. Scale Q by 1/sqrt(headKDim) (matches llama.cpp)
	qScale := float32(1.0 / math.Sqrt(float64(headKDim)))
	for i := 0; i < keyDim; i++ {
		q[i] *= qScale
	}

	// 10. Delta rule recurrent step + output
	state := ssmState.State
	for h := 0; h < numHeads; h++ {
		decay := float32(math.Exp(float64(ssm.Alpha[h])))
		lr := ssm.Beta[h]
		qH := q[h*headKDim : (h+1)*headKDim]
		kH := k[h*headKDim : (h+1)*headKDim]
		vH := v[h*headVDim : (h+1)*headVDim]
		sOff := h * headKDim * headVDim

		// Step A: Decay state
		for idx := sOff; idx < sOff+headKDim*headVDim; idx++ {
			state[idx] *= decay
		}

		// Step B: Predict value from key using current state
		// v_pred[j] = sum_i S[i][j] * k[i]
		// Step C: Compute delta = v - v_pred
		// Step D: Update state: S[i][j] += beta * k[i] * delta[j]
		for j := 0; j < headVDim; j++ {
			var vPred float32
			for i := 0; i < headKDim; i++ {
				vPred += state[sOff+i*headVDim+j] * kH[i]
			}
			delta := vH[j] - vPred
			for i := 0; i < headKDim; i++ {
				state[sOff+i*headVDim+j] += lr * kH[i] * delta
			}
		}

		// Step E: Output: y[j] = sum_i S[i][j] * q_scaled[i]
		for j := 0; j < headVDim; j++ {
			var dot float32
			for i := 0; i < headKDim; i++ {
				dot += state[sOff+i*headVDim+j] * qH[i]
			}
			ssm.Y[h*headVDim+j] = dot
		}
	}

	// 11. Per-head RMSNorm + SiLU gate
	for h := 0; h < numHeads; h++ {
		yH := ssm.Y[h*headVDim : (h+1)*headVDim]
		zH := ssm.Z[h*headVDim : (h+1)*headVDim]

		ops.RMSNormInPlace(yH, layer.SSMNorm, cfg.RMSNormEps)
		for j := 0; j < headVDim; j++ {
			yH[j] *= zH[j] * ops.Sigmoid(zH[j])
		}
	}

	// 12. Out projection: valueDim -> dim
	blas.QMatVecMulParallel(rs.AttnProj, layer.SSMOut, ssm.Y, pool)

	return rs.AttnProj
}

func l2Normalize(v []float32, eps float32) {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	invNorm := float32(1.0 / math.Sqrt(float64(norm)+float64(eps)))
	for i := range v {
		v[i] *= invNorm
	}
}
