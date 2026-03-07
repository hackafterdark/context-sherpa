package quant

// Q6_K quantization — matches llama.cpp reference dequantize_row_q6_K.
// Block structure: 210 bytes per block of 256 values
// Layout:
//   ql[128]: lower 4 bits of quantized values (offset 0)
//   qh[64]:  upper 2 bits of quantized values (offset 128)
//   scales[16]: int8 scales, one per 16 values (offset 192)
//   d: float16 super-block scale (offset 208)
//
// Values are packed in 2 groups of 128. Within each group of 128,
// 32 iterations produce 4 values each at positions l+0, l+32, l+64, l+96:
//   q1 = ql[l] low nibble  + qh[l] bits 0-1
//   q2 = ql[l+32] low nibble + qh[l] bits 2-3
//   q3 = ql[l] high nibble + qh[l] bits 4-5
//   q4 = ql[l+32] high nibble + qh[l] bits 6-7

const BlockSizeQ6_K = 256
const BlockBytesQ6_K = 210

// DequantizeQ6_K converts Q6_K quantized data to float32.
func DequantizeQ6_K(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ6_K

	for block := 0; block < numBlocks; block++ {
		blockOffset := block * BlockBytesQ6_K

		// Read float16 scale factor (last 2 bytes of block)
		dBits := uint16(data[blockOffset+208]) | uint16(data[blockOffset+209])<<8
		d := float16ToFloat32(dBits)

		outBase := block * BlockSizeQ6_K
		qlBase := blockOffset          // 128 bytes of ql
		qhBase := blockOffset + 128    // 64 bytes of qh
		scBase := blockOffset + 192    // 16 bytes of scales

		// Process 2 groups of 128 values
		for n128 := 0; n128 < 2; n128++ {
			qlOff := qlBase + n128*64
			qhOff := qhBase + n128*32

			for l := 0; l < 32; l++ {
				qlByte0 := data[qlOff+l]
				qlByte32 := data[qlOff+l+32]
				qhByte := data[qhOff+l]

				// Reconstruct 6-bit signed values (4 bits from ql + 2 bits from qh - 32)
				q1 := (int(qlByte0&0x0F) | (int((qhByte>>0)&3) << 4)) - 32
				q2 := (int(qlByte32&0x0F) | (int((qhByte>>2)&3) << 4)) - 32
				q3 := (int(qlByte0>>4) | (int((qhByte>>4)&3) << 4)) - 32
				q4 := (int(qlByte32>>4) | (int((qhByte>>6)&3) << 4)) - 32

				// Scale index: each 16-value sub-group has its own scale
				// (matches SYCL/CUDA: is = 8*ip + il/16)
				is := 8*n128 + l/16

				sc0 := float32(int8(data[scBase+is]))
				sc2 := float32(int8(data[scBase+is+2]))
				sc4 := float32(int8(data[scBase+is+4]))
				sc6 := float32(int8(data[scBase+is+6]))

				pos := outBase + n128*128 + l
				result[pos+0] = d * sc0 * float32(q1)
				result[pos+32] = d * sc2 * float32(q2)
				result[pos+64] = d * sc4 * float32(q3)
				result[pos+96] = d * sc6 * float32(q4)
			}
		}
	}
	return result
}
