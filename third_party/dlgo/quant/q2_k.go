package quant

// Q2_K quantization
// Block structure: 84 bytes per super-block of 256 values
// Layout:
//   scales[16]: 4-bit scale+min pairs (16 bytes, offset 0)
//   qs[64]: 2-bit quants packed 4 per byte (64 bytes, offset 16)
//   d: float16 super-block scale (2 bytes, offset 80)
//   dmin: float16 super-block min (2 bytes, offset 82)
// Formula: y = d*(scale_nibble)*quant_2bit - min*(min_nibble)

const BlockSizeQ2_K = 256
const BlockBytesQ2_K = 84

// DequantizeQ2_K converts Q2_K quantized data to float32.
func DequantizeQ2_K(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ2_K

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesQ2_K

		// scales at offset 0 (16 bytes)
		scOff := off
		// qs at offset 16 (64 bytes)
		qOff := off + 16
		// d at offset 80, dmin at offset 82
		dBits := uint16(data[off+80]) | uint16(data[off+81])<<8
		dminBits := uint16(data[off+82]) | uint16(data[off+83])<<8
		d := float16ToFloat32(dBits)
		dmin := float16ToFloat32(dminBits)

		outOff := block * BlockSizeQ2_K
		is := 0
		for n128 := 0; n128 < 2; n128++ {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc := data[scOff+is]
				is++
				dl := d * float32(sc&0xF)
				ml := dmin * float32(sc>>4)
				for l := 0; l < 16; l++ {
					q := (data[qOff+l] >> shift) & 3
					result[outOff] = dl*float32(q) - ml
					outOff++
				}

				sc = data[scOff+is]
				is++
				dl = d * float32(sc&0xF)
				ml = dmin * float32(sc>>4)
				for l := 0; l < 16; l++ {
					q := (data[qOff+l+16] >> shift) & 3
					result[outOff] = dl*float32(q) - ml
					outOff++
				}

				shift += 2
			}
			qOff += 32
		}
	}
	return result
}
