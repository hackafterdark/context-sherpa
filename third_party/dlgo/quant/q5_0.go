package quant

// Q5_0 quantization
// Block structure: 22 bytes per block of 32 values
// Layout:
//   d: float16 scale (2 bytes)
//   qh: 4 bytes — bit-packed 5th bit for 32 values (uint32 little-endian)
//   qs: 16 bytes — lower 4 bits as nibble pairs

const BlockSizeQ5_0 = 32
const BlockBytesQ5_0 = 22

// DequantizeQ5_0 converts Q5_0 quantized data to float32.
func DequantizeQ5_0(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ5_0

	for block := 0; block < numBlocks; block++ {
		blockOff := block * BlockBytesQ5_0

		// Float16 scale
		dBits := uint16(data[blockOff]) | uint16(data[blockOff+1])<<8
		d := float16ToFloat32(dBits)

		// 5th bit packed as uint32 (little-endian)
		qh := uint32(data[blockOff+2]) | uint32(data[blockOff+3])<<8 |
			uint32(data[blockOff+4])<<16 | uint32(data[blockOff+5])<<24

		// 16 bytes of nibble pairs at offset 6
		qsOff := blockOff + 6

		for j := 0; j < 32; j++ {
			// Get 4-bit value
			var q int
			if j < 16 {
				q = int(data[qsOff+j] & 0x0F)
			} else {
				q = int(data[qsOff+j-16] >> 4)
			}

			// Add 5th bit
			if (qh>>j)&1 != 0 {
				q |= 0x10
			}

			// Dequantize: subtract 16 (zero-point) and multiply by scale
			result[block*BlockSizeQ5_0+j] = float32(q-16) * d
		}
	}
	return result
}
