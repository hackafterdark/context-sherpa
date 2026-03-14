package quant

// Q8_1 quantization (intermediate type, but can appear in model files)
// Block structure: 36 bytes per block of 32 values
// Layout:
//   d: float16 delta (2 bytes)
//   s: float16 sum (2 bytes) — precomputed d*sum(qs), unused for dequant
//   qs: 32 bytes — signed 8-bit quants
// Formula: y = d * qs[j]

const BlockSizeQ8_1 = 32
const BlockBytesQ8_1 = 36

// DequantizeQ8_1 converts Q8_1 quantized data to float32.
func DequantizeQ8_1(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ8_1

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesQ8_1

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)
		// skip s (2 bytes at off+2..off+3), not needed for dequantization

		base := block * BlockSizeQ8_1
		for j := 0; j < 32; j++ {
			q := int8(data[off+4+j])
			result[base+j] = float32(q) * d
		}
	}
	return result
}
