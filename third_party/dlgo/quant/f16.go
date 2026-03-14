package quant

import "math"

// float16ToFloat32 converts a IEEE 754 half-precision float to float32.
// This is critical — get it right or all weights will be wrong.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	var f uint32
	switch {
	case exp == 0:
		if mant == 0 {
			// Zero
			f = sign << 31
		} else {
			// Subnormal: convert to normalized float32
			exp2 := uint32(127 - 14)
			for mant&0x400 == 0 {
				mant <<= 1
				exp2--
			}
			mant &= 0x3FF
			f = (sign << 31) | (exp2 << 23) | (mant << 13)
		}
	case exp == 0x1F:
		// Inf or NaN
		f = (sign << 31) | (0xFF << 23) | (mant << 13)
	default:
		// Normalized
		f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(f)
}