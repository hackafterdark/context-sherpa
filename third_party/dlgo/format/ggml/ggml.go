// Package ggml provides a parser for the legacy GGML (GPT-Generated Model Language) format.
//
// GGML is the predecessor to the GGUF format and was used by early Whisper.cpp models.
// While GGUF is the modern standard with memory-mapped access and extensive metadata,
// GGML remains important for backward compatibility with existing models.
//
// The GGML format consists of:
// 1. Magic bytes (0x67676d6c little-endian or 0x6c6d6767 big-endian)
// 2. Hyperparameters (11 int32 values: n_vocab, n_audio_ctx, n_audio_state, etc.)
// 3. Mel filter coefficients (n_mels × n_fft × 4 bytes)
// 4. Vocabulary (count + variable-length tokens)
// 5. Tensor data (repeating: n_dims, name_len, type, dims[], name[], data[])
//
// Unlike GGUF's memory-mapped approach, GGML loads all tensor data into memory.
// This is simpler but less efficient for very large models.
//
// For new models, prefer GGUF format. This package exists to support existing
// Whisper.cpp GGML models.
package ggml

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// RawTensor stores tensor payload in original GGML encoding.
type RawTensor struct {
	Data        []byte  // Raw quantized tensor data
	Type        uint32  // GGML quantization type (matches quant package types)
	NumElements int     // Total number of elements in the tensor
	Dimensions  []int32 // Tensor shape (e.g., [1, 256, 64, 64])
}

// LoadGGMLModel reads a GGML format model file from disk.
//
// This function supports both little-endian (0x67676d6c) and big-endian (0x6c6d6767)
// magic bytes. It loads the entire model into memory, including all tensor data.
//
// Parameters:
//   - path: Path to the GGML model file
//
// Returns:
//   - metadata: Raw bytes containing 11 hyperparameters (44 bytes total)
//   - tensors: Map of tensor name → RawTensor with loaded data
//   - error: Any error encountered during parsing
//
// Note: For modern models with memory-mapped access and extensive metadata,
// use GGUF format instead (see format/gguf package).
func LoadGGMLModel(path string) ([]byte, map[string]RawTensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open model: %w", err)
	}
	defer f.Close()

	// Read magic bytes
	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, nil, fmt.Errorf("read magic: %w", err)
	}

	// Check for both byte orders
	const (
		ggmlMagicLittle = 0x67676d6c // "ggml" little-endian
		ggmlMagicBig    = 0x6c6d6767 // "ggml" big-endian
	)

	if magic != ggmlMagicLittle && magic != ggmlMagicBig {
		return nil, nil, fmt.Errorf("invalid magic 0x%08x, expected 0x%08x or 0x%08x (GGML)", magic, ggmlMagicLittle, ggmlMagicBig)
	}

	fmt.Printf("✓ GGML model detected (magic: 0x%08x)\n", magic)

	// Legacy whisper.cpp GGML layout starts with hyperparameters right after magic.
	// hparams: n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
	//          n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype
	metadata := make([]int32, 11)
	for i := 0; i < len(metadata); i++ {
		if err := binary.Read(f, binary.LittleEndian, &metadata[i]); err != nil {
			return nil, nil, fmt.Errorf("read metadata[%d]: %w", i, err)
		}
	}
	fmt.Printf("  Metadata: %v\n", metadata)

	// Store metadata in the first return value
	metadataBytes := make([]byte, len(metadata)*4)
	for i, val := range metadata {
		binary.LittleEndian.PutUint32(metadataBytes[i*4:(i+1)*4], uint32(val))
	}

	// Read and skip mel filters section.
	var nMels, nFFT int32
	if err := binary.Read(f, binary.LittleEndian, &nMels); err != nil {
		return nil, nil, fmt.Errorf("read mel n_mels: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &nFFT); err != nil {
		return nil, nil, fmt.Errorf("read mel n_fft: %w", err)
	}
	if nMels <= 0 || nFFT <= 0 {
		return nil, nil, fmt.Errorf("invalid mel filter dims: n_mels=%d n_fft=%d", nMels, nFFT)
	}
	melBytes := int64(nMels) * int64(nFFT) * 4
	if _, err := f.Seek(melBytes, io.SeekCurrent); err != nil {
		return nil, nil, fmt.Errorf("skip mel filters (%d bytes): %w", melBytes, err)
	}

	// Read and skip vocabulary section.
	var vocabCount int32
	if err := binary.Read(f, binary.LittleEndian, &vocabCount); err != nil {
		return nil, nil, fmt.Errorf("read vocab count: %w", err)
	}
	if vocabCount < 0 {
		return nil, nil, fmt.Errorf("invalid vocab count: %d", vocabCount)
	}
	for i := int32(0); i < vocabCount; i++ {
		var tokenLen int32
		if err := binary.Read(f, binary.LittleEndian, &tokenLen); err != nil {
			return nil, nil, fmt.Errorf("read token len %d: %w", i, err)
		}
		if tokenLen < 0 {
			return nil, nil, fmt.Errorf("invalid token len at %d: %d", i, tokenLen)
		}
		if _, err := f.Seek(int64(tokenLen), io.SeekCurrent); err != nil {
			return nil, nil, fmt.Errorf("skip token %d bytes (%d): %w", i, tokenLen, err)
		}
	}

	// Read all tensors.
	tensors := make(map[string]RawTensor)
	for {
		// Tensor header format:
		// int32 n_dims, int32 name_len, int32 tensor_type, int32 dims[n_dims], char name[name_len]
		var nDims int32
		err := binary.Read(f, binary.LittleEndian, &nDims)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("read tensor n_dims: %w", err)
		}
		if nDims <= 0 || nDims > 4 {
			return nil, nil, fmt.Errorf("invalid tensor n_dims: %d", nDims)
		}

		var nameLen int32
		if err := binary.Read(f, binary.LittleEndian, &nameLen); err != nil {
			return nil, nil, fmt.Errorf("read tensor name length: %w", err)
		}
		if nameLen <= 0 || nameLen > 256 {
			return nil, nil, fmt.Errorf("invalid tensor name length: %d", nameLen)
		}

		var tensorType int32
		if err := binary.Read(f, binary.LittleEndian, &tensorType); err != nil {
			return nil, nil, fmt.Errorf("read tensor type: %w", err)
		}

		dims := make([]int32, nDims)
		totalElements := int64(1)
		for i := 0; i < int(nDims); i++ {
			if err := binary.Read(f, binary.LittleEndian, &dims[i]); err != nil {
				return nil, nil, fmt.Errorf("read tensor dimension %d: %w", i, err)
			}
			if dims[i] <= 0 {
				return nil, nil, fmt.Errorf("invalid tensor dimension %d: %d", i, dims[i])
			}
			totalElements *= int64(dims[i])
		}

		nameBuf := make([]byte, nameLen)
		if _, err := io.ReadFull(f, nameBuf); err != nil {
			return nil, nil, fmt.Errorf("read tensor name: %w", err)
		}
		name := string(nameBuf)

		byteSize := bytesForN(uint32(tensorType), int(totalElements))
		if byteSize <= 0 {
			return nil, nil, fmt.Errorf("unsupported tensor type %d for %s", tensorType, name)
		}

		rawData := make([]byte, byteSize)
		if _, err := io.ReadFull(f, rawData); err != nil {
			return nil, nil, fmt.Errorf("read tensor data for %s: %w", name, err)
		}

		tensors[name] = RawTensor{
			Data:        rawData,
			Type:        uint32(tensorType),
			NumElements: int(totalElements),
			Dimensions:  dims,
		}
	}

	fmt.Printf("✓ Loaded %d tensors from GGML model\n", len(tensors))

	return metadataBytes, tensors, nil
}

// bytesForN returns how many bytes are needed for n elements of the given type.
//
// This function duplicates the logic in quant.BytesForN(). The duplication exists
// to keep the ggml package self-contained and independent. For the authoritative
// version with comprehensive documentation, see quant.BytesForN().
//
// TODO: Consider refactoring to share this logic with quant.BytesForN() to avoid duplication.
func bytesForN(ggmlType uint32, n int) int {
	switch ggmlType {
	case 0: // F32
		return n * 4
	case 1: // F16
		return n * 2
	case 2: // Q4_0
		return (n / 32) * 18
	case 3: // Q4_1
		return (n / 32) * 20
	case 6: // Q5_0
		return (n / 32) * 22
	case 7: // Q5_1
		return (n / 32) * 24
	case 8: // Q8_0
		return (n / 32) * 34
	case 9: // Q8_1
		return (n / 32) * 36
	case 10: // Q2_K
		return (n / 256) * 84
	case 11: // Q3_K
		return (n / 256) * 110
	case 12: // Q4_K
		return (n / 256) * 144
	case 13: // Q5_K
		return (n / 256) * 176
	case 14: // Q6_K
		return (n / 256) * 210
	case 15: // Q8_K
		return (n / 256) * 292
	case 16: // IQ2_XXS
		return (n / 256) * 66
	case 17: // IQ2_XS
		return (n / 256) * 74
	case 18: // IQ3_XXS
		return (n / 256) * 98
	case 19: // IQ1_S
		return (n / 256) * 50
	case 20: // IQ4_NL
		return (n / 32) * 18
	case 21: // IQ3_S
		return (n / 256) * 110
	case 22: // IQ2_S
		return (n / 256) * 82
	case 23: // IQ4_XS
		return (n / 256) * 136
	case 29: // IQ1_M
		return (n / 256) * 56
	case 30: // BF16
		return n * 2
	case 34: // TQ1_0
		return (n / 256) * 54
	case 35: // TQ2_0
		return (n / 256) * 66
	default:
		return 0
	}
}