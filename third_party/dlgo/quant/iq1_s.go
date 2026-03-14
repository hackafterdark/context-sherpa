package quant

// IQ1_S quantization (type 19)
// Block structure: 50 bytes per block of 256 values
// Layout:
//   d:  float16 super-block scale (2 bytes)
//   qs: uint8[32]  — low 8 bits of 11-bit grid index
//   qh: uint16[8]  — bits 12..14: 3-bit scale, bit 15: delta sign,
//                     bits [3*l..3*l+2]: high 3 bits of grid index for sub-group l
//
// Grid values are signed int8 {-1, 0, +1} with ±0.125 delta offset.

const BlockSizeIQ1S = 256
const BlockBytesIQ1S = 50
const iq1sDelta float32 = 0.125

// DequantizeIQ1S converts IQ1_S quantized data to float32.
func DequantizeIQ1S(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeIQ1S

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ1S

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsOff := off + 2       // 32 bytes
		qhOff := off + 2 + 32  // 16 bytes (8 x uint16)
		outBase := block * BlockSizeIQ1S

		for ib := 0; ib < 8; ib++ {
			qh := uint16(data[qhOff+ib*2]) | uint16(data[qhOff+ib*2+1])<<8

			dl := d * float32(2*int((qh>>12)&7)+1)
			var delta float32
			if qh&0x8000 != 0 {
				delta = -iq1sDelta
			} else {
				delta = iq1sDelta
			}

			for l := 0; l < 4; l++ {
				// 11-bit grid index: low 8 from qs, high 3 from qh
				gridIdx := uint16(data[qsOff+l]) | ((qh >> uint(3*l) & 7) << 8)
				grid := iq1s_grid[gridIdx]

				oIdx := outBase + ib*32 + l*8
				for j := 0; j < 8; j++ {
					// Grid values are signed int8
					gridByte := uint8(grid >> (8 * uint(j)))
					gridVal := float32(int8(gridByte))
					result[oIdx+j] = dl * (gridVal + delta)
				}
			}
			qsOff += 4
		}
	}
	return result
}
