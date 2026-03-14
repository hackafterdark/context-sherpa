package quant

const BlockSizeQ4_0 = 32  // number of float32 values per block
const BlockBytesQ4_0 = 18 // bytes per block (2 + 16)

// DequantizeQ4_0 converts Q4_0 quantized data to float32.
// data: raw bytes from GGUF file
// n: number of float32 values to produce (MUST be a multiple of 32)
// Returns: []float32 of length n
func DequantizeQ4_0(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ4_0

	for block := 0; block < numBlocks; block++ {
		blockOffset := block * BlockBytesQ4_0

		// Read float16 scale factor (2 bytes, little-endian)
		dBits := uint16(data[blockOffset]) | uint16(data[blockOffset+1])<<8
		d := float16ToFloat32(dBits)

		// Read 16 bytes of nibbles → 32 values
		for j := 0; j < 16; j++ {
			qByte := data[blockOffset+2+j]

			// Low nibble → first 16 values of block
			x0 := int(qByte&0x0F) - 8
			result[block*32+j] = float32(x0) * d

			// High nibble → last 16 values of block
			x1 := int(qByte>>4) - 8
			result[block*32+j+16] = float32(x1) * d
		}
	}
	return result
}
