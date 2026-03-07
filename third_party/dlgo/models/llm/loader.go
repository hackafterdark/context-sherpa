package llm

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/quant"
)

// LoadModel opens a GGUF file, parses config from metadata, and loads all tensors.
func LoadModel(path string) (*Model, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}

	cfg, err := parseConfig(gf.Metadata)
	if err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Fix embed scale for Gemma
	if cfg.EmbedScale > 0 {
		cfg.EmbedScale = float32(math.Sqrt(float64(cfg.EmbeddingDim)))
	}

	m := &Model{
		Config: cfg,
		Layers: make([]Layer, cfg.NumLayers),
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Precompute derived dimensions for fused QKV splitting
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	for _, ti := range gf.Tensors {
		data, err := readTensorData(f, gf.DataOffset, ti)
		if err != nil {
			return nil, fmt.Errorf("read tensor %s: %w", ti.Name, err)
		}

		totalElements := int64(1)
		for _, d := range ti.Dimensions {
			totalElements *= d
		}
		rows, cols := inferRowsCols(ti.Dimensions)

		if isNormOrBias(ti.Name) {
			fp32 := dequantToF32(data, uint32(ti.Type), int(totalElements))
			mapTensorF32(m, ti.Name, fp32)
		} else {
			qt := &core.QuantizedTensor{
				Data: data,
				Type: uint32(ti.Type),
				Rows: rows,
				Cols: cols,
			}
			mapTensorQT(m, ti.Name, qt, qDim, kvDim, cfg)
		}
	}

	// Weight tying: if output projection is nil, share with token embeddings
	if m.Output == nil {
		m.Output = m.TokenEmbed
	}

	// Resolve per-layer specs from loaded tensor presence
	for i := range m.Layers {
		m.Layers[i].Spec = resolveLayerSpec(&m.Layers[i], cfg, i)
	}

	return m, nil
}

// resolveLayerSpec infers all architectural choices for a layer from its loaded
// weights. Called once at load time so the forward pass can dispatch via switch.
func resolveLayerSpec(l *Layer, cfg ModelConfig, layerIdx int) LayerSpec {
	var s LayerSpec

	if l.AttnNormBias != nil {
		s.Norm = NormLayer
	} else {
		s.Norm = NormRMS
	}

	if isSSMLayer(layerIdx, cfg) && l.SSMInProj != nil {
		s.Core = CoreSSM
	} else {
		s.Core = CoreAttention
	}

	if l.FFNNorm != nil {
		s.Residual = ResStandard
	} else if l.PostAttnNorm != nil {
		s.Residual = ResPostAttnFFN
	} else {
		s.Residual = ResParallel
	}

	if l.FFNGate != nil {
		if cfg.FFNGelu {
			s.FFN = FFNGeGLU
		} else {
			s.FFN = FFNSwiGLU
		}
	} else {
		s.FFN = FFNPlain
	}

	s.GatedQ = l.Wq != nil && l.Wq.Rows > cfg.NumHeads*cfg.HeadDim
	s.QKNorm = l.AttnQNorm != nil

	return s
}

func readTensorData(f *os.File, dataOffset int64, ti gguf.TensorInfo) ([]byte, error) {
	totalElements := int64(1)
	for _, d := range ti.Dimensions {
		totalElements *= d
	}
	nbytes := int64(quant.BytesForN(uint32(ti.Type), int(totalElements)))
	data := make([]byte, nbytes)

	offset := dataOffset + int64(ti.Offset)
	if _, err := f.ReadAt(data, offset); err != nil {
		return nil, err
	}
	return data, nil
}

func inferRowsCols(dims []int64) (int, int) {
	if len(dims) == 0 {
		return 1, 1
	}
	if len(dims) == 1 {
		return int(dims[0]), 1
	}
	// GGUF stores [cols, rows] (reversed from row-major convention)
	return int(dims[len(dims)-1]), int(dims[0])
}

// isNormOrBias returns true if the tensor name indicates a norm weight, bias,
// or other small 1D parameter that should be stored as dequantized float32.
func isNormOrBias(name string) bool {
	return strings.HasSuffix(name, "_norm.weight") ||
		strings.HasSuffix(name, ".bias") ||
		strings.HasSuffix(name, "_norm.bias") ||
		strings.HasSuffix(name, "ssm_a") ||
		strings.HasSuffix(name, "ssm_conv1d.weight")
}

func dequantToF32(data []byte, ggmlType uint32, n int) []float32 {
	result, _ := quant.Dequantize(data, ggmlType, n)
	return result
}

func mapTensorF32(m *Model, name string, data []float32) {
	switch {
	case name == "output_norm.weight":
		m.OutputNorm = data
	case name == "output_norm.bias":
		m.OutputNormBias = data
	case name == "output.bias":
		m.OutputBias = data
	default:
		if layerIdx, field := parseLayerName(name); layerIdx >= 0 && layerIdx < len(m.Layers) {
			l := &m.Layers[layerIdx]
			switch field {
			case "attn_norm.weight":
				l.AttnNorm = data
			case "attn_norm.bias":
				l.AttnNormBias = data
			case "ffn_norm.weight":
				l.FFNNorm = data
			case "attn_q.bias":
				l.Bq = data
			case "attn_k.bias":
				l.Bk = data
			case "attn_v.bias":
				l.Bv = data
			case "attn_qkv.bias":
				qDim := m.Config.NumHeads * m.Config.HeadDim
				kvDim := m.Config.NumKVHeads * m.Config.HeadDim
				l.Bq = data[:qDim]
				l.Bk = data[qDim : qDim+kvDim]
				l.Bv = data[qDim+kvDim : qDim+2*kvDim]
			case "attn_output.bias":
				l.Bo = data
			case "attn_q_norm.weight":
				l.AttnQNorm = data
			case "attn_k_norm.weight":
				l.AttnKNorm = data
			case "post_attention_norm.weight":
				l.PostAttnNorm = data
			case "post_ffw_norm.weight":
				l.PostFFNNorm = data
			case "ffn_up.bias":
				l.FFNUpBias = data
			case "ffn_down.bias":
				l.FFNDownBias = data
			case "ssm_dt.bias":
				l.SSMDtBias = data
			case "ssm_norm.weight":
				l.SSMNorm = data
			case "ssm_a":
				l.SSMA = data
			case "ssm_conv1d.weight":
				l.SSMConv1dW = data
			}
		}
	}
}

func mapTensorQT(m *Model, name string, qt *core.QuantizedTensor, qDim, kvDim int, cfg ModelConfig) {
	switch {
	case name == "token_embd.weight":
		m.TokenEmbed = qt
	case name == "output.weight":
		m.Output = qt
	default:
		if layerIdx, field := parseLayerName(name); layerIdx >= 0 && layerIdx < len(m.Layers) {
			l := &m.Layers[layerIdx]
			switch field {
			case "attn_q.weight":
				l.Wq = qt
			case "attn_k.weight":
				l.Wk = qt
			case "attn_v.weight":
				l.Wv = qt
			case "attn_qkv.weight":
				splitFusedQKV(l, qt, qDim, kvDim, cfg.EmbeddingDim)
			case "attn_output.weight":
				l.Wo = qt
			case "attn_gate.weight":
				l.AttnGate = qt
			case "ffn_gate.weight":
				l.FFNGate = qt
			case "ffn_up.weight":
				splitFusedFFNUp(l, qt, cfg.FFNDim, cfg.EmbeddingDim)
			case "ffn_down.weight":
				l.FFNDown = qt
			case "ssm_alpha.weight":
				l.SSMAlpha = qt
			case "ssm_beta.weight":
				l.SSMBeta = qt
			case "ssm_out.weight":
				l.SSMOut = qt
			}
		}
	}
}

// splitFusedQKV splits a fused [Q|K|V] weight tensor into separate Wq, Wk, Wv.
// If the total rows don't match qDim+2*kvDim, the tensor is stored as SSMInProj
// (used for Qwen3.5 SSM/delta-net layers where the in-projection has different dims).
func splitFusedQKV(l *Layer, qt *core.QuantizedTensor, qDim, kvDim, cols int) {
	expected := qDim + 2*kvDim
	if qt.Rows != expected {
		l.SSMInProj = qt
		return
	}
	bytesPerRow := quant.BytesForN(qt.Type, cols)
	qBytes := qDim * bytesPerRow
	kvBytes := kvDim * bytesPerRow
	l.Wq = &core.QuantizedTensor{Data: qt.Data[:qBytes], Type: qt.Type, Rows: qDim, Cols: cols}
	l.Wk = &core.QuantizedTensor{Data: qt.Data[qBytes : qBytes+kvBytes], Type: qt.Type, Rows: kvDim, Cols: cols}
	l.Wv = &core.QuantizedTensor{Data: qt.Data[qBytes+kvBytes : qBytes+2*kvBytes], Type: qt.Type, Rows: kvDim, Cols: cols}
}

// splitFusedFFNUp splits a fused [gate|up] weight tensor if it has 2x expected rows.
func splitFusedFFNUp(l *Layer, qt *core.QuantizedTensor, ffnDim, cols int) {
	if qt.Rows == 2*ffnDim {
		bytesPerRow := quant.BytesForN(qt.Type, cols)
		halfBytes := ffnDim * bytesPerRow
		l.FFNGate = &core.QuantizedTensor{Data: qt.Data[:halfBytes], Type: qt.Type, Rows: ffnDim, Cols: cols}
		l.FFNUp = &core.QuantizedTensor{Data: qt.Data[halfBytes : 2*halfBytes], Type: qt.Type, Rows: ffnDim, Cols: cols}
	} else {
		l.FFNUp = qt
	}
}

// parseLayerName extracts layer index and field name from "blk.{i}.{field}" patterns.
func parseLayerName(name string) (int, string) {
	if !strings.HasPrefix(name, "blk.") {
		return -1, ""
	}
	rest := name[4:]
	dotIdx := strings.IndexByte(rest, '.')
	if dotIdx < 0 {
		return -1, ""
	}
	idx, err := strconv.Atoi(rest[:dotIdx])
	if err != nil {
		return -1, ""
	}
	return idx, rest[dotIdx+1:]
}
