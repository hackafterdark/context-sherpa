package quant

// IQ3_S quantization (type 21)
// Block structure: 110 bytes per block of 256 values
// Layout:
//   d:      float16 super-block scale (2 bytes)
//   qs:     uint8[64] — low 8 bits of 9-bit grid index
//   qh:     uint8[8]  — high 1 bit of 9-bit grid index
//   signs:  uint8[32] — sign bytes
//   scales: uint8[4]  — two 4-bit scale nibbles per byte (IQ3S_N_SCALE = QK_K/64 = 4)

const BlockSizeIQ3S = 256
const BlockBytesIQ3S = 110

// DequantizeIQ3S converts IQ3_S quantized data to float32.
func DequantizeIQ3S(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeIQ3S

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ3S

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsStart := off + 2          // 64 bytes
		qhStart := off + 2 + 64     // 8 bytes
		signStart := off + 2 + 64 + 8 // 32 bytes
		scStart := off + 2 + 64 + 8 + 32 // 4 bytes
		outBase := block * BlockSizeIQ3S

		qsOff := qsStart
		qhOff := qhStart
		signsOff := signStart

		// Process 2 groups of 32 at a time (ib32 += 2)
		for ib32 := 0; ib32 < 8; ib32 += 2 {
			scByte := data[scStart+ib32/2]
			db1 := d * float32(1+2*int(scByte&0xf))
			db2 := d * float32(1+2*int(scByte>>4))

			qh0 := data[qhOff]
			qh1 := data[qhOff+1]

			// First group of 32 (uses db1, qh0)
			for l := 0; l < 4; l++ {
				gridIdx1 := uint16(data[qsOff+2*l]) | ((uint16(qh0) << (8 - 2*uint(l))) & 256)
				gridIdx2 := uint16(data[qsOff+2*l+1]) | ((uint16(qh0) << (7 - 2*uint(l))) & 256)
				grid1 := iq3s_grid[gridIdx1]
				grid2 := iq3s_grid[gridIdx2]
				signByte := data[signsOff+l]

				oIdx := outBase + ib32*32 + l*8
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid1 >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j] != 0 {
						result[oIdx+j] = -db1 * gridVal
					} else {
						result[oIdx+j] = db1 * gridVal
					}
				}
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid2 >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j+4] != 0 {
						result[oIdx+4+j] = -db1 * gridVal
					} else {
						result[oIdx+4+j] = db1 * gridVal
					}
				}
			}
			qsOff += 8
			signsOff += 4

			// Second group of 32 (uses db2, qh1)
			for l := 0; l < 4; l++ {
				gridIdx1 := uint16(data[qsOff+2*l]) | ((uint16(qh1) << (8 - 2*uint(l))) & 256)
				gridIdx2 := uint16(data[qsOff+2*l+1]) | ((uint16(qh1) << (7 - 2*uint(l))) & 256)
				grid1 := iq3s_grid[gridIdx1]
				grid2 := iq3s_grid[gridIdx2]
				signByte := data[signsOff+l]

				oIdx := outBase + (ib32+1)*32 + l*8
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid1 >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j] != 0 {
						result[oIdx+j] = -db2 * gridVal
					} else {
						result[oIdx+j] = db2 * gridVal
					}
				}
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid2 >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j+4] != 0 {
						result[oIdx+4+j] = -db2 * gridVal
					} else {
						result[oIdx+4+j] = db2 * gridVal
					}
				}
			}
			qhOff += 2
			qsOff += 8
			signsOff += 4
		}
	}
	return result
}
