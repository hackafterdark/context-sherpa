package quant

// IQ2_XS quantization (type 17)
// Block structure: 74 bytes per block of 256 values
// Layout:
//   d:      float16 super-block scale (2 bytes)
//   qs:     uint16[32] = 64 bytes — each uint16: bits[0..8]=9-bit grid index, bits[9..15]=7-bit sign index
//   scales: uint8[8] — each byte has two 4-bit scale nibbles

const BlockSizeIQ2XS = 256
const BlockBytesIQ2XS = 74

// DequantizeIQ2XS converts IQ2_XS quantized data to float32.
func DequantizeIQ2XS(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeIQ2XS

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ2XS

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsOff := off + 2     // 64 bytes of uint16 qs
		scOff := off + 2 + 64 // 8 bytes of scales
		outBase := block * BlockSizeIQ2XS

		for ib32 := 0; ib32 < 8; ib32++ {
			scByte := data[scOff+ib32]
			db0 := d * (0.5 + float32(scByte&0xf)) * 0.25
			db1 := d * (0.5 + float32(scByte>>4)) * 0.25

			for l := 0; l < 4; l++ {
				qIdx := qsOff + (ib32*4+l)*2
				qs := uint16(data[qIdx]) | uint16(data[qIdx+1])<<8

				gridIdx := qs & 511
				signIdx := qs >> 9
				grid := iq2xs_grid[gridIdx]
				signs := ksigns_iq2xs[signIdx]

				var db float32
				if l < 2 {
					db = db0
				} else {
					db = db1
				}

				oIdx := outBase + ib32*32 + l*8
				for j := 0; j < 8; j++ {
					gridVal := float32(uint8(grid >> (8 * uint(j))))
					if signs&kmask_iq2xs[j] != 0 {
						result[oIdx+j] = -db * gridVal
					} else {
						result[oIdx+j] = db * gridVal
					}
				}
			}
		}
	}
	return result
}
