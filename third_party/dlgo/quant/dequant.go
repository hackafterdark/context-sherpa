// Package quant provides comprehensive support for GGML quantization formats.
//
// This package implements dequantization for 20+ GGML quantization formats including:
//
// Traditional formats (32 values per block):
//   - Q4_0, Q5_0, Q8_0: Uniform quantization with varying bit depth
//   - Q4_1, Q5_1, Q8_1: Uniform quantization with separate min/max scaling
//
// K-schemes (256 values per block, better quality):
//   - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K: Advanced quantization with super-blocks
//
// I-Mixtures (256 values per block, best compression):
//   - IQ1_S, IQ1_M: 1-bit mixture models
//   - IQ2_S, IQ2_XS, IQ2_XXS: 2-bit mixture models
//   - IQ3_S, IQ3_XXS: 3-bit mixture models
//   - IQ4_XS: 4-bit mixture model
//   - IQ4_NL: 4-bit non-linear quantization
//
// Tile quantization (256 values per block):
//   - TQ1_0, TQ2_0: Tile-based quantization
//
// Half precision:
//   - F16: Half-precision floating point
//   - BF16: Bfloat16 (brain floating point)
//
// The package supports on-the-fly dequantization for efficient inference,
// as well as full tensor dequantization when needed.
//
// Performance:
// - Many formats support SIMD-optimized fused dot products
// - Streaming dequantization into existing buffers to avoid allocations
// - Pre-dequantization support for models that require FP32 weights
package quant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Dequantize dispatches to the correct dequantizer based on GGMLType.
//
// Parameters:
//   - data: Raw quantized bytes
//   - ggmlType: GGML quantization type identifier (0-35)
//   - n: Number of elements to dequantize
//
// Returns a new float32 slice with the dequantized values.
//
// Supported types:
//   0: F32, 1: F16, 30: BF16
//   2: Q4_0, 3: Q4_1, 6: Q5_0, 7: Q5_1, 8: Q8_0, 9: Q8_1
//   10: Q2_K, 11: Q3_K, 12: Q4_K, 13: Q5_K, 14: Q6_K, 15: Q8_K
//   16: IQ2_XXS, 17: IQ2_XS, 18: IQ3_XXS, 19: IQ1_S, 20: IQ4_NL
//   21: IQ3_S, 22: IQ2_S, 23: IQ4_XS, 29: IQ1_M
//   34: TQ1_0, 35: TQ2_0
func Dequantize(data []byte, ggmlType uint32, n int) ([]float32, error) {
	switch ggmlType {
	case 0: // F32
		return dequantizeF32(data, n), nil
	case 1: // F16
		return dequantizeF16(data, n), nil
	case 2: // Q4_0
		return DequantizeQ4_0(data, n), nil
	case 3: // Q4_1
		return DequantizeQ4_1(data, n), nil
	case 6: // Q5_0
		return DequantizeQ5_0(data, n), nil
	case 7: // Q5_1
		return DequantizeQ5_1(data, n), nil
	case 8: // Q8_0
		return DequantizeQ8_0(data, n), nil
	case 9: // Q8_1
		return DequantizeQ8_1(data, n), nil
	case 10: // Q2_K
		return DequantizeQ2_K(data, n), nil
	case 11: // Q3_K
		return DequantizeQ3_K(data, n), nil
	case 12: // Q4_K
		return DequantizeQ4_K(data, n), nil
	case 13: // Q5_K
		return DequantizeQ5_K(data, n), nil
	case 14: // Q6_K
		return DequantizeQ6_K(data, n), nil
	case 15: // Q8_K
		return DequantizeQ8_K(data, n), nil
	case 16: // IQ2_XXS
		return DequantizeIQ2XXS(data, n), nil
	case 17: // IQ2_XS
		return DequantizeIQ2XS(data, n), nil
	case 18: // IQ3_XXS
		return DequantizeIQ3XXS(data, n), nil
	case 19: // IQ1_S
		return DequantizeIQ1S(data, n), nil
	case 20: // IQ4_NL
		return DequantizeIQ4_NL(data, n), nil
	case 21: // IQ3_S
		return DequantizeIQ3S(data, n), nil
	case 22: // IQ2_S
		return DequantizeIQ2S(data, n), nil
	case 23: // IQ4_XS
		return DequantizeIQ4_XS(data, n), nil
	case 29: // IQ1_M
		return DequantizeIQ1M(data, n), nil
	case 30: // BF16
		return DequantizeBF16(data, n), nil
	case 34: // TQ1_0
		return DequantizeTQ1_0(data, n), nil
	case 35: // TQ2_0
		return DequantizeTQ2_0(data, n), nil
	default:
		return nil, fmt.Errorf("unsupported quantization type: %d", ggmlType)
	}
}

func dequantizeF32(data []byte, n int) []float32 {
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		result[i] = math.Float32frombits(bits)
	}
	return result
}

func dequantizeF16(data []byte, n int) []float32 {
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2:])
		result[i] = float16ToFloat32(bits)
	}
	return result
}

// BytesForN returns how many bytes are needed for n elements of the given type.
//
// This is useful for calculating buffer sizes when working with quantized tensors.
// Returns 0 for unsupported types.
func BytesForN(ggmlType uint32, n int) int {
	switch ggmlType {
	case 0: // F32
		return n * 4
	case 1: // F16
		return n * 2
	case 2: // Q4_0: 32 values per block, 18 bytes
		return (n / 32) * 18
	case 3: // Q4_1: 32 values per block, 20 bytes
		return (n / 32) * 20
	case 6: // Q5_0: 32 values per block, 22 bytes
		return (n / 32) * 22
	case 7: // Q5_1: 32 values per block, 24 bytes
		return (n / 32) * 24
	case 8: // Q8_0: 32 values per block, 34 bytes
		return (n / 32) * 34
	case 9: // Q8_1: 32 values per block, 36 bytes
		return (n / 32) * 36
	case 10: // Q2_K: 256 values per block, 84 bytes
		return (n / 256) * 84
	case 11: // Q3_K: 256 values per block, 110 bytes
		return (n / 256) * 110
	case 12: // Q4_K: 256 values per block, 144 bytes
		return (n / 256) * 144
	case 13: // Q5_K: 256 values per block, 176 bytes
		return (n / 256) * 176
	case 14: // Q6_K: 256 values per block, 210 bytes
		return (n / 256) * 210
	case 15: // Q8_K: 256 values per block, 292 bytes
		return (n / 256) * 292
	case 16: // IQ2_XXS: 256 values per block, 66 bytes
		return (n / 256) * 66
	case 17: // IQ2_XS: 256 values per block, 74 bytes
		return (n / 256) * 74
	case 18: // IQ3_XXS: 256 values per block, 98 bytes
		return (n / 256) * 98
	case 19: // IQ1_S: 256 values per block, 50 bytes
		return (n / 256) * 50
	case 20: // IQ4_NL: 32 values per block, 18 bytes
		return (n / 32) * 18
	case 21: // IQ3_S: 256 values per block, 110 bytes
		return (n / 256) * 110
	case 22: // IQ2_S: 256 values per block, 82 bytes
		return (n / 256) * 82
	case 23: // IQ4_XS: 256 values per block, 136 bytes
		return (n / 256) * 136
	case 29: // IQ1_M: 256 values per block, 56 bytes
		return (n / 256) * 56
	case 30: // BF16
		return n * 2
	case 34: // TQ1_0: 256 values per block, 54 bytes
		return (n / 256) * 54
	case 35: // TQ2_0: 256 values per block, 66 bytes
		return (n / 256) * 66
	default:
		return 0
	}
}