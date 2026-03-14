package quant

// Q5_1 quantization
// Block structure: 24 bytes per block of 32 values
// Layout:
//   d: float16 scale (2 bytes)
//   m: float16 minimum (2 bytes)
//   qh: 4 bytes — 5th bit for each of 32 values (uint32 LE)
//   qs: 16 bytes — lower 4-bit nibble pairs
// Formula: y = quant * d + m (quant is 5-bit unsigned [0..31])

const BlockSizeQ5_1 = 32
const BlockBytesQ5_1 = 24

// DequantizeQ5_1 converts Q5_1 quantized data to float32.
func DequantizeQ5_1(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ5_1

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesQ5_1

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		mBits := uint16(data[off+2]) | uint16(data[off+3])<<8
		d := float16ToFloat32(dBits)
		m := float16ToFloat32(mBits)

		// 5th bit packed as uint32 LE
		qh := uint32(data[off+4]) | uint32(data[off+5])<<8 |
			uint32(data[off+6])<<16 | uint32(data[off+7])<<24

		base := block * BlockSizeQ5_1
		for j := 0; j < 16; j++ {
			qByte := data[off+8+j]

			// Low nibble + 5th bit → first 16 values
			x0 := int(qByte & 0x0F)
			xh0 := int((qh >> uint(j)) & 1)
			x0 |= xh0 << 4

			// High nibble + 5th bit → last 16 values
			x1 := int(qByte >> 4)
			xh1 := int((qh >> uint(j+16)) & 1)
			x1 |= xh1 << 4

			result[base+j] = float32(x0)*d + m
			result[base+j+16] = float32(x1)*d + m
		}
	}
	return result
}
