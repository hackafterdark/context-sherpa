package quant

import (
	"encoding/binary"
	"math"
)

// Q8_K quantization
// Block structure: 292 bytes per super-block of 256 values
// Layout:
//   d: float32 delta (4 bytes, offset 0)
//   qs[256]: signed 8-bit quants (256 bytes, offset 4)
//   bsums[16]: int16 sums of groups of 16 (32 bytes, offset 260) — unused for dequant
// Formula: y = d * qs[j]

const BlockSizeQ8_K = 256
const BlockBytesQ8_K = 292

// DequantizeQ8_K converts Q8_K quantized data to float32.
func DequantizeQ8_K(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ8_K

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesQ8_K

		// d is float32 (not float16!)
		d := math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4]))

		base := block * BlockSizeQ8_K
		for j := 0; j < 256; j++ {
			q := int8(data[off+4+j])
			result[base+j] = d * float32(q)
		}
	}
	return result
}
