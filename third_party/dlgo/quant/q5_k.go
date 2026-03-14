package quant

// Q5_K quantization
// Block structure: 176 bytes per block of 256 values
// Layout:
//   d: float16 super-block scale (2 bytes)
//   dmin: float16 super-block minimum (2 bytes)
//   scales: 12 bytes packed 6-bit scales and mins (same as Q4_K)
//   qh: 32 bytes — 5th bit for each of the 256 values
//   qs: 128 bytes — lower 4 bits (nibble pairs)

const BlockSizeQ5_K = 256
const BlockBytesQ5_K = 176

// DequantizeQ5_K converts Q5_K quantized data to float32.
func DequantizeQ5_K(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ5_K

	for block := 0; block < numBlocks; block++ {
		blockOff := block * BlockBytesQ5_K

		// Read super-block scale and minimum
		dBits := uint16(data[blockOff]) | uint16(data[blockOff+1])<<8
		dminBits := uint16(data[blockOff+2]) | uint16(data[blockOff+3])<<8
		d := float16ToFloat32(dBits)
		dmin := float16ToFloat32(dminBits)

		// Unpack 6-bit scales and mins from 12 bytes (same layout as Q4_K)
		scalesOff := blockOff + 4
		var sc [8]float32
		var mn [8]float32

		for i := 0; i < 4; i++ {
			sc[i] = d * float32(data[scalesOff+i]&0x3F)
			mn[i] = dmin * float32(data[scalesOff+4+i]&0x3F)
		}
		for i := 0; i < 4; i++ {
			scHi := (data[scalesOff+i] >> 6) & 0x03
			mnHi := (data[scalesOff+4+i] >> 6) & 0x03
			scLo := data[scalesOff+8+i] & 0x0F
			mnLo := (data[scalesOff+8+i] >> 4) & 0x0F
			sc[4+i] = d * float32(scLo|scHi<<4)
			mn[4+i] = dmin * float32(mnLo|mnHi<<4)
		}

		// High bits: 32 bytes at offset 16
		qhOff := blockOff + 16
		// Low bits: 128 bytes at offset 48
		qsOff := blockOff + 48

		// Dequantize 256 values in groups of 64
		// Each group: first 32 use low nibbles, next 32 use high nibbles
		// High bit (5th): qh[l] bit position advances by 2 per group
		outOff := block * BlockSizeQ5_K
		is := 0
		u1 := byte(1)
		u2 := byte(2)
		for grp := 0; grp < 4; grp++ {
			d1 := sc[is]
			m1 := mn[is]
			d2 := sc[is+1]
			m2 := mn[is+1]

			qlOff := qsOff + grp*32

			// First 32: low nibbles
			for l := 0; l < 32; l++ {
				q := int(data[qlOff+l] & 0x0F)
				if data[qhOff+l]&u1 != 0 {
					q |= 16
				}
				result[outOff] = float32(q)*d1 - m1
				outOff++
			}
			// Next 32: high nibbles
			for l := 0; l < 32; l++ {
				q := int(data[qlOff+l] >> 4)
				if data[qhOff+l]&u2 != 0 {
					q |= 16
				}
				result[outOff] = float32(q)*d2 - m2
				outOff++
			}

			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
	return result
}
