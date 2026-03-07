package quant

// IQ1_M quantization (type 29)
// Block structure: 56 bytes per block of 256 values
// Layout (NO separate d field!):
//   qs:     uint8[32]  — low 8 bits of 11-bit grid index
//   qh:     uint8[16]  — high 3 bits of grid index + delta sign bit per pair
//   scales: uint8[8]   — 3-bit per-group scales + embedded f16 super-scale in top bits
//
// The f16 super-scale is reconstructed from the top 4 bits of each uint16 in scales:
//   scale.u16 = (sc[0]>>12) | ((sc[1]>>8)&0x00f0) | ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000)
//
// Grid values are signed int8 {-1, 0, +1} with ±0.125 delta offset.

const BlockSizeIQ1M = 256
const BlockBytesIQ1M = 56
const iq1mDelta float32 = 0.125

// DequantizeIQ1M converts IQ1_M quantized data to float32.
func DequantizeIQ1M(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeIQ1M

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ1M

		qsStart := off         // 32 bytes
		qhStart := off + 32    // 16 bytes
		scStart := off + 32 + 16 // 8 bytes

		// Read scales as 4 uint16 values (8 bytes total)
		sc := [4]uint16{
			uint16(data[scStart]) | uint16(data[scStart+1])<<8,
			uint16(data[scStart+2]) | uint16(data[scStart+3])<<8,
			uint16(data[scStart+4]) | uint16(data[scStart+5])<<8,
			uint16(data[scStart+6]) | uint16(data[scStart+7])<<8,
		}

		// Reconstruct f16 super-scale from top 4 bits of each sc[]
		scaleU16 := (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
		d := float16ToFloat32(scaleU16)

		outBase := block * BlockSizeIQ1M

		qsOff := qsStart
		qhOff := qhStart

		for ib := 0; ib < 8; ib++ {
			// Per-group scales from sc
			scIdx := ib / 2
			scShift := 6 * (ib % 2)
			dl1 := d * float32(2*int((sc[scIdx]>>uint(scShift))&0x7)+1)
			dl2 := d * float32(2*int((sc[scIdx]>>uint(scShift+3))&0x7)+1)

			qh0 := data[qhOff]
			qh1 := data[qhOff+1]

			// 4 sub-groups of 8
			var idx [4]uint16
			var delta [4]float32

			idx[0] = uint16(data[qsOff]) | (uint16(qh0)<<8)&0x700
			idx[1] = uint16(data[qsOff+1]) | (uint16(qh0)<<4)&0x700
			idx[2] = uint16(data[qsOff+2]) | (uint16(qh1)<<8)&0x700
			idx[3] = uint16(data[qsOff+3]) | (uint16(qh1)<<4)&0x700

			if qh0&0x08 != 0 {
				delta[0] = -iq1mDelta
			} else {
				delta[0] = iq1mDelta
			}
			if qh0&0x80 != 0 {
				delta[1] = -iq1mDelta
			} else {
				delta[1] = iq1mDelta
			}
			if qh1&0x08 != 0 {
				delta[2] = -iq1mDelta
			} else {
				delta[2] = iq1mDelta
			}
			if qh1&0x80 != 0 {
				delta[3] = -iq1mDelta
			} else {
				delta[3] = iq1mDelta
			}

			// First 2 sub-groups use dl1
			for l := 0; l < 2; l++ {
				grid := iq1s_grid[idx[l]]
				oIdx := outBase + ib*32 + l*8
				for j := 0; j < 8; j++ {
					gridByte := uint8(grid >> (8 * uint(j)))
					gridVal := float32(int8(gridByte))
					result[oIdx+j] = dl1 * (gridVal + delta[l])
				}
			}
			// Last 2 sub-groups use dl2
			for l := 2; l < 4; l++ {
				grid := iq1s_grid[idx[l]]
				oIdx := outBase + ib*32 + l*8
				for j := 0; j < 8; j++ {
					gridByte := uint8(grid >> (8 * uint(j)))
					gridVal := float32(int8(gridByte))
					result[oIdx+j] = dl2 * (gridVal + delta[l])
				}
			}

			qsOff += 4
			qhOff += 2
		}
	}
	return result
}
