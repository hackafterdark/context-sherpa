package quant

// Q4_K quantization
// Block structure: 144 bytes per block of 256 values
// Layout (super-block of 256 values):
//   d: float16 super-block scale (2 bytes)
//   dmin: float16 super-block minimum (2 bytes)
//   scales: 12 bytes packed 6-bit scales and mins
//   qs: 128 bytes of 4-bit quantized values

const BlockSizeQ4_K = 256
const BlockBytesQ4_K = 144

// DequantizeQ4_K converts Q4_K quantized data to float32.
func DequantizeQ4_K(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ4_K

	for block := 0; block < numBlocks; block++ {
		blockOff := block * BlockBytesQ4_K

		// Read super-block scale and minimum
		dBits := uint16(data[blockOff]) | uint16(data[blockOff+1])<<8
		dminBits := uint16(data[blockOff+2]) | uint16(data[blockOff+3])<<8
		d := float16ToFloat32(dBits)
		dmin := float16ToFloat32(dminBits)

		// Unpack 6-bit scales and mins from 12 bytes at offset 4
		scalesOff := blockOff + 4
		var sc [8]float32
		var mn [8]float32

		// First 4 scales/mins: lower 6 bits from bytes 0-3 / 4-7
		for i := 0; i < 4; i++ {
			sc[i] = d * float32(data[scalesOff+i]&0x3F)
			mn[i] = dmin * float32(data[scalesOff+4+i]&0x3F)
		}
		// Last 4 scales/mins: combine upper 2 bits from bytes 0-3/4-7 + lower 4 bits from bytes 8-11
		for i := 0; i < 4; i++ {
			scHi := (data[scalesOff+i] >> 6) & 0x03
			mnHi := (data[scalesOff+4+i] >> 6) & 0x03
			scLo := data[scalesOff+8+i] & 0x0F
			mnLo := (data[scalesOff+8+i] >> 4) & 0x0F
			sc[4+i] = d * float32(scLo|scHi<<4)
			mn[4+i] = dmin * float32(mnLo|mnHi<<4)
		}

		// Dequantize 256 values from 128 bytes of 4-bit quantized values
		// Layout: 4 groups of 64 values. Each group:
		//   first 32 values = low nibbles of 32 q bytes
		//   next  32 values = high nibbles of SAME 32 q bytes
		qsOff := blockOff + 16
		outOff := block * BlockSizeQ4_K
		is := 0 // scale index (0..7)
		for grp := 0; grp < 4; grp++ {
			d1 := sc[is]
			m1 := mn[is]
			d2 := sc[is+1]
			m2 := mn[is+1]

			qOff := qsOff + grp*32
			// First 32: low nibbles
			for l := 0; l < 32; l++ {
				result[outOff] = d1*float32(data[qOff+l]&0x0F) - m1
				outOff++
			}
			// Next 32: high nibbles of same bytes
			for l := 0; l < 32; l++ {
				result[outOff] = d2*float32(data[qOff+l]>>4) - m2
				outOff++
			}
			is += 2
		}
	}
	return result
}
