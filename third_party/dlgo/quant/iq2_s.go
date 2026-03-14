package quant

// IQ2_S quantization (type 22)
// Block structure: 82 bytes per block of 256 values
// Layout:
//   d:      float16 super-block scale (2 bytes)
//   qs:     uint8[64] — first 32 bytes: low 8 bits of grid index, next 32 bytes: sign bytes
//   qh:     uint8[8]  — high 2 bits of 10-bit grid index
//   scales: uint8[8]  — two 4-bit scale nibbles per byte

const BlockSizeIQ2S = 256
const BlockBytesIQ2S = 82

// DequantizeIQ2S converts IQ2_S quantized data to float32.
func DequantizeIQ2S(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeIQ2S

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ2S

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsOff := off + 2      // 64 bytes total
		qhOff := off + 2 + 64 // 8 bytes
		scOff := off + 2 + 64 + 8 // 8 bytes
		outBase := block * BlockSizeIQ2S

		// qs first 32 bytes = grid index low bytes (4 per group of 32)
		// qs next 32 bytes = sign bytes (4 per group of 32)
		gridLowOff := qsOff
		signsOff := qsOff + 32

		for ib32 := 0; ib32 < 8; ib32++ {
			scByte := data[scOff+ib32]
			db0 := d * (0.5 + float32(scByte&0xf)) * 0.25
			db1 := d * (0.5 + float32(scByte>>4)) * 0.25
			qhByte := data[qhOff+ib32]

			for l := 0; l < 4; l++ {
				// 10-bit grid index: low 8 from qs, high 2 from qh
				lowByte := uint16(data[gridLowOff])
				highBits := (uint16(qhByte) << (8 - 2*uint(l))) & 0x300
				gridIdx := lowByte | highBits
				grid := iq2s_grid[gridIdx]

				signByte := data[signsOff]

				var db float32
				if l < 2 {
					db = db0
				} else {
					db = db1
				}

				oIdx := outBase + ib32*32 + l*8
				for j := 0; j < 8; j++ {
					gridVal := float32(uint8(grid >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j] != 0 {
						result[oIdx+j] = -db * gridVal
					} else {
						result[oIdx+j] = db * gridVal
					}
				}
				gridLowOff++
				signsOff++
			}
		}
	}
	return result
}
