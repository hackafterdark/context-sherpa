//go:build cgo && vulkan

package gpu

import (
	"math"

	"github.com/computerex/dlgo/models/llm"
)

// GpuForward performs a single-token forward pass entirely on GPU.
// All layers are recorded into a single command buffer with explicit barriers
// placed only where data dependencies require them.
func GpuForward(m *llm.Model, gm *GpuModel, token int32, pos int,
	kv *GpuKVCache, rs *GpuRunState, logitsBuf []float32) {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim

	xCPU := make([]float32, dim)
	_ = m.TokenEmbed.DequantizeRow(int(token), xCPU)
	if cfg.EmbedScale != 0 {
		for i := range xCPU {
			xCPU[i] *= cfg.EmbedScale
		}
	}
	seqLen := pos + 1
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	BeginBatch()
	UploadF32(rs.X, xCPU)

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec
		gl := &gm.Layers[l]

		// RMSNorm reads X (written by previous layer's residual Add)
		Barrier()
		if spec.Norm == llm.NormRMS {
			RMSNorm(rs.XNorm, rs.X, gl.AttnNorm, dim, cfg.RMSNormEps)
		}

		if spec.Core == llm.CoreAttention {
			// Q/K/V all read from XNorm, write to independent buffers -> parallel
			Barrier()
			MatVec(rs.Q, gl.Wq.Buf, rs.XNorm, gl.Wq.Rows, gl.Wq.Cols, gl.Wq.Type)
			MatVec(rs.K, gl.Wk.Buf, rs.XNorm, gl.Wk.Rows, gl.Wk.Cols, gl.Wk.Type)
			MatVec(rs.V, gl.Wv.Buf, rs.XNorm, gl.Wv.Rows, gl.Wv.Cols, gl.Wv.Type)

			// Biases depend on their respective matmul outputs
			Barrier()
			if gl.Bq != 0 {
				addBuf(rs.Q, gl.Bq, numHeads*headDim)
			}
			if gl.Bk != 0 {
				addBuf(rs.K, gl.Bk, kvDim)
			}
			if gl.Bv != 0 {
				addBuf(rs.V, gl.Bv, kvDim)
			}

			if spec.QKNorm {
				Barrier()
				// Q and K norms are independent
				RMSNormHeads(rs.Q, gl.AttnQNorm, numHeads, headDim, cfg.RMSNormEps)
				RMSNormHeads(rs.K, gl.AttnKNorm, numKVHeads, headDim, cfg.RMSNormEps)
			}

			// RoPE reads Q and K
			Barrier()
			RoPE(rs.Q, rs.K, numHeads, numKVHeads, headDim, pos, cfg.RopeFreqBase, cfg.RopeNeox)

			// KVStore has internal compute→transfer and transfer→compute barriers.
			KVStore(kv.KeyBufs[l], kv.ValBufs[l], rs.K, rs.V, pos, kvDim)

			Barrier()
			Attention(rs.AttnOut, rs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, seqLen, scale)

			// Wo projection reads AttnOut
			Barrier()
			MatVec(rs.AttnProj, gl.Wo.Buf, rs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, gl.Wo.Type)
		}

		switch spec.Residual {
		case llm.ResStandard:
			Barrier()
			if gl.PostAttnNorm != 0 {
				RMSNorm(rs.AttnProj, rs.AttnProj, gl.PostAttnNorm, dim, cfg.RMSNormEps)
				Barrier()
			}
			// Fused Add + RMSNorm: saves one barrier vs separate Add then RMSNorm
			AddRMSNorm(rs.FFNNorm, rs.FFNIn, rs.X, rs.AttnProj, gl.FFNNorm, dim, cfg.RMSNormEps)
			Barrier()
			gpuForwardFFN(layer, gl, rs, rs.FFNNorm, dim, cfg)
			Barrier()
			if gl.PostFFNNorm != 0 {
				RMSNorm(rs.FFNOut, rs.FFNOut, gl.PostFFNNorm, dim, cfg.RMSNormEps)
				Barrier()
			}
			Add(rs.X, rs.FFNIn, rs.FFNOut, dim)

		case llm.ResParallel:
			Barrier()
			gpuForwardFFN(layer, gl, rs, rs.XNorm, dim, cfg)
			Barrier()
			Add(rs.X, rs.X, rs.AttnProj, dim)
			Barrier()
			Add(rs.X, rs.X, rs.FFNOut, dim)
		}
	}

	Barrier()
	RMSNorm(rs.X, rs.X, gm.OutputNorm, dim, cfg.RMSNormEps)
	Barrier()
	output := gm.Output
	if output == nil {
		output = gm.TokenEmbed
	}
	MatVec(rs.Logits, output.Buf, rs.X, output.Rows, output.Cols, output.Type)

	DownloadF32(rs.Logits, logitsBuf)
}

func addBuf(dst, src Buf, n int) {
	Add(dst, dst, src, n)
}

func gpuForwardFFN(layer *llm.Layer, gl *GpuLayer, rs *GpuRunState, input Buf, dim int, cfg llm.ModelConfig) {
	switch layer.Spec.FFN {
	case llm.FFNSwiGLU:
		// Gate and Up read from same input, write to independent buffers -> parallel
		MatVec(rs.Gate, gl.FFNGate.Buf, input, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
		MatVec(rs.Up, gl.FFNUp.Buf, input, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type)
		Barrier()
		SwiGLU(rs.Hidden, rs.Gate, rs.Up, gl.FFNGate.Rows)
		Barrier()
		MatVec(rs.FFNOut, gl.FFNDown.Buf, rs.Hidden, gl.FFNDown.Rows, gl.FFNDown.Cols, gl.FFNDown.Type)

	case llm.FFNGeGLU:
		MatVec(rs.Gate, gl.FFNGate.Buf, input, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
		MatVec(rs.Up, gl.FFNUp.Buf, input, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type)
		Barrier()
		GeGLU(rs.Hidden, rs.Gate, rs.Up, gl.FFNGate.Rows)
		Barrier()
		MatVec(rs.FFNOut, gl.FFNDown.Buf, rs.Hidden, gl.FFNDown.Rows, gl.FFNDown.Cols, gl.FFNDown.Type)

	case llm.FFNPlain:
		MatVec(rs.Up, gl.FFNUp.Buf, input, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type)
		Barrier()
		GELU(rs.Up, gl.FFNUp.Rows)
		Barrier()
		MatVec(rs.FFNOut, gl.FFNDown.Buf, rs.Up, gl.FFNDown.Rows, gl.FFNDown.Cols, gl.FFNDown.Type)
	}
}
