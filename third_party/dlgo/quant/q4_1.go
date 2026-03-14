package quant

// Q4_1 quantization
// Block structure: 20 bytes per block of 32 values
// Layout:
//   d: float16 scale (2 bytes)
//   m: float16 minimum (2 bytes)
//   qs: 16 bytes — 4-bit unsigned quants (nibble pairs)
// Formula: y = quant * d + m

const BlockSizeQ4_1 = 32
const BlockBytesQ4_1 = 20

// DequantizeQ4_1 converts Q4_1 quantized data to float32.
func DequantizeQ4_1(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ4_1

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesQ4_1

		// Float16 scale and minimum
		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		mBits := uint16(data[off+2]) | uint16(data[off+3])<<8
		d := float16ToFloat32(dBits)
		m := float16ToFloat32(mBits)

		base := block * BlockSizeQ4_1
		for j := 0; j < 16; j++ {
			qByte := data[off+4+j]
			x0 := int(qByte & 0x0F)
			x1 := int(qByte >> 4)
			result[base+j] = float32(x0)*d + m
			result[base+j+16] = float32(x1)*d + m
		}
	}
	return result
}
