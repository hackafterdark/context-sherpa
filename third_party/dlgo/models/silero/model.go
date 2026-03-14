package silero

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// SileroModel holds all model weights and hyperparameters loaded from a GGML file.
type SileroModel struct {
	// Header
	ModelType      string
	Version        [3]int32
	WindowSize     int32 // 512
	ContextSize    int32 // 64
	NEncoderLayers int32 // 4
	EncoderInCh    []int32
	EncoderOutCh   []int32
	KernelSizes    []int32
	LSTMInputSize  int32 // 128
	LSTMHiddenSize int32 // 128
	FinalConvIn    int32 // 128
	FinalConvOut   int32 // 1

	// Tensors — flat []float32, stored exactly as read from file (after f16→f32).
	// No transposition. Use indexing formulas in each layer.
	STFTBasis       []float32    // [258 * 256] = 66048 elements. Index: basis[oc*256 + k]
	EncoderWeights  [4][]float32 // Each: [outCh * inCh * kernel]. Index: w[oc*inCh*kernel + ic*kernel + k]
	EncoderBiases   [4][]float32 // Each: [outCh]
	LSTMWeightIH    []float32    // [512 * 128] = 65536. Index: w[g*128 + j]
	LSTMWeightHH    []float32    // [512 * 128] = 65536. Index: w[g*128 + j]
	LSTMBiasIH      []float32    // [512]
	LSTMBiasHH      []float32    // [512]
	FinalConvWeight []float32    // [128]
	FinalConvBias   []float32    // [1]
}

// LoadModel reads a Silero VAD GGML model file.
func LoadModel(path string) (*SileroModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open model: %w", err)
	}
	defer f.Close()

	m := &SileroModel{}

	// --- Header ---

	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != 0x67676d6c {
		return nil, fmt.Errorf("invalid magic 0x%08x, expected 0x67676d6c (GGML)", magic)
	}

	// Model type string
	var strLen int32
	binary.Read(f, binary.LittleEndian, &strLen)
	strBuf := make([]byte, strLen)
	io.ReadFull(f, strBuf)
	m.ModelType = string(strBuf)

	// Version
	binary.Read(f, binary.LittleEndian, &m.Version[0])
	binary.Read(f, binary.LittleEndian, &m.Version[1])
	binary.Read(f, binary.LittleEndian, &m.Version[2])

	// Architecture params
	binary.Read(f, binary.LittleEndian, &m.WindowSize)
	binary.Read(f, binary.LittleEndian, &m.ContextSize)
	binary.Read(f, binary.LittleEndian, &m.NEncoderLayers)

	nLayers := int(m.NEncoderLayers)
	m.EncoderInCh = make([]int32, nLayers)
	m.EncoderOutCh = make([]int32, nLayers)
	m.KernelSizes = make([]int32, nLayers)
	for i := 0; i < nLayers; i++ {
		binary.Read(f, binary.LittleEndian, &m.EncoderInCh[i])
		binary.Read(f, binary.LittleEndian, &m.EncoderOutCh[i])
		binary.Read(f, binary.LittleEndian, &m.KernelSizes[i])
	}

	binary.Read(f, binary.LittleEndian, &m.LSTMInputSize)
	binary.Read(f, binary.LittleEndian, &m.LSTMHiddenSize)
	binary.Read(f, binary.LittleEndian, &m.FinalConvIn)
	binary.Read(f, binary.LittleEndian, &m.FinalConvOut)

	fmt.Printf("Loaded header: %s v%d.%d.%d, window=%d, lstm_hidden=%d\n",
		m.ModelType, m.Version[0], m.Version[1], m.Version[2],
		m.WindowSize, m.LSTMHiddenSize)

	// --- Tensors ---

	tensorCount := 0
	for {
		var nDims int32
		err := binary.Read(f, binary.LittleEndian, &nDims)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read tensor header: %w", err)
		}

		var nameLen int32
		var ftype int32
		binary.Read(f, binary.LittleEndian, &nameLen)
		binary.Read(f, binary.LittleEndian, &ftype)

		dims := make([]int32, nDims)
		for i := 0; i < int(nDims); i++ {
			binary.Read(f, binary.LittleEndian, &dims[i])
		}

		nameBuf := make([]byte, nameLen)
		io.ReadFull(f, nameBuf)
		name := string(nameBuf)

		// Total elements = product of all dims
		totalElements := int64(1)
		for _, d := range dims {
			totalElements *= int64(d)
		}

		// Read data
		var data []float32
		if ftype == 0 {
			// float32
			data = make([]float32, totalElements)
			if err := binary.Read(f, binary.LittleEndian, data); err != nil {
				return nil, fmt.Errorf("read f32 data for %s: %w", name, err)
			}
		} else if ftype == 1 {
			// float16 → convert to float32
			raw := make([]uint16, totalElements)
			if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
				return nil, fmt.Errorf("read f16 data for %s: %w", name, err)
			}
			data = make([]float32, totalElements)
			for i, v := range raw {
				data[i] = f16ToF32(v)
			}
		} else {
			return nil, fmt.Errorf("unsupported ftype %d for tensor %s", ftype, name)
		}

		fmt.Printf("  tensor[%d]: %s dims=%v ftype=%d elements=%d\n", tensorCount, name, dims, ftype, totalElements)
		tensorCount++

		// Map to struct fields by exact tensor name
		switch name {
		case "_model.stft.forward_basis_buffer":
			m.STFTBasis = data
		case "_model.encoder.0.reparam_conv.weight":
			m.EncoderWeights[0] = data
		case "_model.encoder.0.reparam_conv.bias":
			m.EncoderBiases[0] = data
		case "_model.encoder.1.reparam_conv.weight":
			m.EncoderWeights[1] = data
		case "_model.encoder.1.reparam_conv.bias":
			m.EncoderBiases[1] = data
		case "_model.encoder.2.reparam_conv.weight":
			m.EncoderWeights[2] = data
		case "_model.encoder.2.reparam_conv.bias":
			m.EncoderBiases[2] = data
		case "_model.encoder.3.reparam_conv.weight":
			m.EncoderWeights[3] = data
		case "_model.encoder.3.reparam_conv.bias":
			m.EncoderBiases[3] = data
		case "_model.decoder.rnn.weight_ih":
			m.LSTMWeightIH = data
		case "_model.decoder.rnn.weight_hh":
			m.LSTMWeightHH = data
		case "_model.decoder.rnn.bias_ih":
			m.LSTMBiasIH = data
		case "_model.decoder.rnn.bias_hh":
			m.LSTMBiasHH = data
		case "_model.decoder.decoder.2.weight":
			m.FinalConvWeight = data
		case "_model.decoder.decoder.2.bias":
			m.FinalConvBias = data
		default:
			fmt.Printf("  WARNING: unknown tensor %q, ignoring\n", name)
		}
	}

	fmt.Printf("Loaded %d tensors\n", tensorCount)

	// Validate
	if m.STFTBasis == nil {
		return nil, fmt.Errorf("missing STFT basis tensor")
	}
	for i := 0; i < 4; i++ {
		if m.EncoderWeights[i] == nil || m.EncoderBiases[i] == nil {
			return nil, fmt.Errorf("missing encoder layer %d tensors", i)
		}
	}
	if m.LSTMWeightIH == nil || m.LSTMWeightHH == nil || m.LSTMBiasIH == nil || m.LSTMBiasHH == nil {
		return nil, fmt.Errorf("missing LSTM tensors")
	}
	if m.FinalConvWeight == nil || m.FinalConvBias == nil {
		return nil, fmt.Errorf("missing final conv tensors")
	}

	return m, nil
}

// f16ToF32 converts IEEE 754 half-precision float to single-precision.
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	if exp == 0 {
		if mant == 0 {
			// ±zero
			return math.Float32frombits(sign << 31)
		}
		// Subnormal: normalize
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
	} else if exp == 0x1f {
		// Inf or NaN
		return math.Float32frombits((sign << 31) | 0x7f800000 | (mant << 13))
	}

	// Normalized: rebias exponent from f16 bias (15) to f32 bias (127)
	return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}
