package quant

import "math"

// DequantizeBF16 converts BF16 (bfloat16) data to float32.
//
// Bfloat16 is a 16-bit floating point format with:
// - 1 sign bit
// - 8 exponent bits (same range as float32)
// - 7 mantissa bits (vs 23 in float32)
//
// The key difference from float16 is that BF16 preserves the full exponent
// range of float32, only reducing mantissa precision. This makes it better
// for deep learning as it avoids underflow/overflow issues.
//
// Conversion is trivial: BF16 bits are the same as the upper 16 bits
// of a float32 value, so we just shift left by 16 bits.
func DequantizeBF16(data []byte, n int) []float32 {
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint16(data[i*2]) | uint16(data[i*2+1])<<8
		// BF16 → F32: just shift to upper 16 bits
		f32bits := uint32(bits) << 16
		result[i] = math.Float32frombits(f32bits)
	}
	return result
}