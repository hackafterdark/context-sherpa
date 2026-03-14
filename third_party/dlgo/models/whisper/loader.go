package whisper

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/quant"
)

func loadWhisperFromGGUF(path string) (*WhisperModel, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GGUF: %w", err)
	}

	cfg := parseWhisperConfig(gf.Metadata)
	m := &WhisperModel{
		Config:    cfg,
		EncLayers: make([]EncoderLayer, cfg.NEncLayers),
		DecLayers: make([]DecoderLayer, cfg.NDecLayers),
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	tensorMap := make(map[string][]byte)
	for _, ti := range gf.Tensors {
		data, err := readTensorData(f, gf.DataOffset, ti)
		if err != nil {
			return nil, fmt.Errorf("read tensor %s: %w", ti.Name, err)
		}
		tensorMap[ti.Name] = data
	}

	// Build tensor info map for dimensions
	tiByName := make(map[string]gguf.TensorInfo)
	for _, ti := range gf.Tensors {
		tiByName[ti.Name] = ti
	}

	// Load encoder conv
	if data, ok := tensorMap["encoder.conv1.weight"]; ok {
		ti := tiByName["encoder.conv1.weight"]
		rows, cols := inferConvRowsCols(ti.Dimensions)
		qt, err := core.NewQuantizedTensor(data, uint32(ti.Type), rows, cols)
		if err != nil {
			return nil, fmt.Errorf("encoder.conv1.weight: %w", err)
		}
		m.Conv1Weight = qt
	}
	if data, ok := tensorMap["encoder.conv1.bias"]; ok {
		ti := tiByName["encoder.conv1.bias"]
		m.Conv1Bias = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap["encoder.conv2.weight"]; ok {
		ti := tiByName["encoder.conv2.weight"]
		rows, cols := inferConvRowsCols(ti.Dimensions)
		qt, err := core.NewQuantizedTensor(data, uint32(ti.Type), rows, cols)
		if err != nil {
			return nil, fmt.Errorf("encoder.conv2.weight: %w", err)
		}
		m.Conv2Weight = qt
	}
	if data, ok := tensorMap["encoder.conv2.bias"]; ok {
		ti := tiByName["encoder.conv2.bias"]
		m.Conv2Bias = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// Encoder position embedding
	if data, ok := tensorMap["encoder.position_embedding.weight"]; ok {
		ti := tiByName["encoder.position_embedding.weight"]
		m.EncPosEmb = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// Encoder blocks
	for i := 0; i < cfg.NEncLayers; i++ {
		prefix := fmt.Sprintf("encoder.blocks.%d.", i)
		loadEncoderLayer(&m.EncLayers[i], tensorMap, tiByName, prefix)
	}

	// Encoder final LayerNorm
	if data, ok := tensorMap["encoder.ln.weight"]; ok {
		ti := tiByName["encoder.ln.weight"]
		m.EncLnW = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap["encoder.ln.bias"]; ok {
		ti := tiByName["encoder.ln.bias"]
		m.EncLnB = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// Decoder token embedding
	if data, ok := tensorMap["decoder.token_embedding.weight"]; ok {
		ti := tiByName["decoder.token_embedding.weight"]
		rows, cols := inferRowsCols(ti.Dimensions)
		qt, err := core.NewQuantizedTensor(data, uint32(ti.Type), rows, cols)
		if err != nil {
			return nil, fmt.Errorf("decoder.token_embedding.weight: %w", err)
		}
		m.TokenEmb = qt
	}

	// Decoder position embedding
	if data, ok := tensorMap["decoder.position_embedding.weight"]; ok {
		ti := tiByName["decoder.position_embedding.weight"]
		m.DecPosEmb = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// Decoder blocks
	for i := 0; i < cfg.NDecLayers; i++ {
		prefix := fmt.Sprintf("decoder.blocks.%d.", i)
		loadDecoderLayer(&m.DecLayers[i], tensorMap, tiByName, prefix)
	}

	// Decoder final LayerNorm
	if data, ok := tensorMap["decoder.ln.weight"]; ok {
		ti := tiByName["decoder.ln.weight"]
		m.DecLnW = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap["decoder.ln.bias"]; ok {
		ti := tiByName["decoder.ln.bias"]
		m.DecLnB = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// Decoder output projection
	if data, ok := tensorMap["decoder.proj.weight"]; ok {
		ti := tiByName["decoder.proj.weight"]
		rows, cols := inferRowsCols(ti.Dimensions)
		qt, err := core.NewQuantizedTensor(data, uint32(ti.Type), rows, cols)
		if err != nil {
			return nil, fmt.Errorf("decoder.proj.weight: %w", err)
		}
		m.ProjOut = qt
	}

	return m, nil
}

func loadEncoderLayer(layer *EncoderLayer, tensorMap map[string][]byte, tiByName map[string]gguf.TensorInfo, prefix string) {
	// attn_ln
	if data, ok := tensorMap[prefix+"attn_ln.weight"]; ok {
		ti := tiByName[prefix+"attn_ln.weight"]
		layer.AttnLnW = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"attn_ln.bias"]; ok {
		ti := tiByName[prefix+"attn_ln.bias"]
		layer.AttnLnB = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// attn q/k/v/out
	layer.Wq = loadQT(tensorMap, tiByName, prefix+"attn.q.weight")
	layer.Wk = loadQT(tensorMap, tiByName, prefix+"attn.k.weight")
	layer.Wv = loadQT(tensorMap, tiByName, prefix+"attn.v.weight")
	layer.Wo = loadQT(tensorMap, tiByName, prefix+"attn.out.weight")

	if data, ok := tensorMap[prefix+"attn.q.bias"]; ok {
		ti := tiByName[prefix+"attn.q.bias"]
		layer.Bq = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"attn.v.bias"]; ok {
		ti := tiByName[prefix+"attn.v.bias"]
		layer.Bv = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"attn.out.bias"]; ok {
		ti := tiByName[prefix+"attn.out.bias"]
		layer.Bo = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// ffn_ln
	if data, ok := tensorMap[prefix+"ffn_ln.weight"]; ok {
		ti := tiByName[prefix+"ffn_ln.weight"]
		layer.FfnLnW = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"ffn_ln.bias"]; ok {
		ti := tiByName[prefix+"ffn_ln.bias"]
		layer.FfnLnB = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// ffn.0 (up), ffn.2 (down)
	layer.FfnUp = loadQT(tensorMap, tiByName, prefix+"ffn.0.weight")
	layer.FfnDown = loadQT(tensorMap, tiByName, prefix+"ffn.2.weight")
	if data, ok := tensorMap[prefix+"ffn.0.bias"]; ok {
		ti := tiByName[prefix+"ffn.0.bias"]
		layer.FfnUpBias = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"ffn.2.bias"]; ok {
		ti := tiByName[prefix+"ffn.2.bias"]
		layer.FfnDownB = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
}

func loadDecoderLayer(layer *DecoderLayer, tensorMap map[string][]byte, tiByName map[string]gguf.TensorInfo, prefix string) {
	// Self-attention
	if data, ok := tensorMap[prefix+"attn_ln.weight"]; ok {
		ti := tiByName[prefix+"attn_ln.weight"]
		layer.SelfAttnLnW = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"attn_ln.bias"]; ok {
		ti := tiByName[prefix+"attn_ln.bias"]
		layer.SelfAttnLnB = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	layer.SelfWq = loadQT(tensorMap, tiByName, prefix+"attn.q.weight")
	layer.SelfWk = loadQT(tensorMap, tiByName, prefix+"attn.k.weight")
	layer.SelfWv = loadQT(tensorMap, tiByName, prefix+"attn.v.weight")
	layer.SelfWo = loadQT(tensorMap, tiByName, prefix+"attn.out.weight")

	if data, ok := tensorMap[prefix+"attn.q.bias"]; ok {
		ti := tiByName[prefix+"attn.q.bias"]
		layer.SelfBq = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"attn.v.bias"]; ok {
		ti := tiByName[prefix+"attn.v.bias"]
		layer.SelfBv = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"attn.out.bias"]; ok {
		ti := tiByName[prefix+"attn.out.bias"]
		layer.SelfBo = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// Cross-attention
	if data, ok := tensorMap[prefix+"cross_attn_ln.weight"]; ok {
		ti := tiByName[prefix+"cross_attn_ln.weight"]
		layer.CrossAttnLnW = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"cross_attn_ln.bias"]; ok {
		ti := tiByName[prefix+"cross_attn_ln.bias"]
		layer.CrossAttnLnB = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	layer.CrossWq = loadQT(tensorMap, tiByName, prefix+"cross_attn.q.weight")
	layer.CrossWk = loadQT(tensorMap, tiByName, prefix+"cross_attn.k.weight")
	layer.CrossWv = loadQT(tensorMap, tiByName, prefix+"cross_attn.v.weight")
	layer.CrossWo = loadQT(tensorMap, tiByName, prefix+"cross_attn.out.weight")

	if data, ok := tensorMap[prefix+"cross_attn.q.bias"]; ok {
		ti := tiByName[prefix+"cross_attn.q.bias"]
		layer.CrossBq = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"cross_attn.v.bias"]; ok {
		ti := tiByName[prefix+"cross_attn.v.bias"]
		layer.CrossBv = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"cross_attn.out.bias"]; ok {
		ti := tiByName[prefix+"cross_attn.out.bias"]
		layer.CrossBo = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	// FFN
	if data, ok := tensorMap[prefix+"ffn_ln.weight"]; ok {
		ti := tiByName[prefix+"ffn_ln.weight"]
		layer.FfnLnW = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"ffn_ln.bias"]; ok {
		ti := tiByName[prefix+"ffn_ln.bias"]
		layer.FfnLnB = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}

	layer.FfnUp = loadQT(tensorMap, tiByName, prefix+"ffn.0.weight")
	layer.FfnDown = loadQT(tensorMap, tiByName, prefix+"ffn.2.weight")
	if data, ok := tensorMap[prefix+"ffn.0.bias"]; ok {
		ti := tiByName[prefix+"ffn.0.bias"]
		layer.FfnUpBias = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
	if data, ok := tensorMap[prefix+"ffn.2.bias"]; ok {
		ti := tiByName[prefix+"ffn.2.bias"]
		layer.FfnDownB = dequantToF32(data, ti.Type, totalElements(ti.Dimensions))
	}
}

func loadQT(tensorMap map[string][]byte, tiByName map[string]gguf.TensorInfo, name string) *core.QuantizedTensor {
	data, ok := tensorMap[name]
	if !ok {
		return nil
	}
	ti, ok := tiByName[name]
	if !ok {
		return nil
	}
	rows, cols := inferRowsCols(ti.Dimensions)
	qt, err := core.NewQuantizedTensor(data, uint32(ti.Type), rows, cols)
	if err != nil {
		return nil
	}
	return qt
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
	// Whisper GGUF stores [rows, cols] (row-major convention)
	return int(dims[0]), int(dims[len(dims)-1])
}

func inferConvRowsCols(dims []int64) (int, int) {
	if len(dims) < 3 {
		return 1, 1
	}
	// GGUF 3D: dims=[outCh, inCh, kernelSize] for conv weights
	// Flatten to 2D: rows=outCh, cols=inCh*kernelSize
	rows := int(dims[0])
	cols := int(dims[1] * dims[2])
	return rows, cols
}

func totalElements(dims []int64) int {
	n := int64(1)
	for _, d := range dims {
		n *= d
	}
	return int(n)
}

func dequantToF32(data []byte, ggmlType gguf.GGMLType, totalElements int) []float32 {
	if ggmlType == gguf.GGMLTypeF32 {
		n := totalElements
		if n > len(data)/4 {
			n = len(data) / 4
		}
		result := make([]float32, n)
		for i := 0; i < n; i++ {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return result
	}
	result, _ := quant.Dequantize(data, uint32(ggmlType), totalElements)
	return result
}
