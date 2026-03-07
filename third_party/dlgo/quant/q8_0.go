package quant

const BlockSizeQ8_0 = 32  // number of float32 values per block
const BlockBytesQ8_0 = 34 // bytes per block (2 + 32)

// DequantizeQ8_0 converts Q8_0 quantized data to float32.
// data: raw bytes from GGUF file
// n: number of float32 values to produce (MUST be a multiple of 32)
// Returns: []float32 of length n
func DequantizeQ8_0(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / 32
	for block := 0; block < numBlocks; block++ {
		off := block * 34
		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)
		for j := 0; j < 32; j++ {
			q := int8(data[off+2+j])
			result[block*32+j] = float32(q) * d
		}
	}
	return result
}
