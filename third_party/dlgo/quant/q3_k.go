package quant

// Q3_K quantization
// Block structure: 110 bytes per super-block of 256 values
// Layout:
//   hmask[32]: high bit of 3-bit quants (32 bytes, offset 0)
//   qs[64]: low 2 bits of quants (64 bytes, offset 32)
//   scales[12]: 6-bit scales packed (12 bytes, offset 96)
//   d: float16 super-block scale (2 bytes, offset 108)
// Formula: y = d * (scale-32) * quant_3bit
// quant_3bit = (2bit_val) - (hmask_bit ? 0 : 4)

const BlockSizeQ3_K = 256
const BlockBytesQ3_K = 110

// DequantizeQ3_K converts Q3_K quantized data to float32.
func DequantizeQ3_K(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ3_K

	const kmask1 = uint32(0x03030303)
	const kmask2 = uint32(0x0f0f0f0f)

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesQ3_K

		// d at offset 108
		dBits := uint16(data[off+108]) | uint16(data[off+109])<<8
		dAll := float16ToFloat32(dBits)

		hmOff := off      // hmask: 32 bytes
		qOff := off + 32  // qs: 64 bytes
		scOff := off + 96 // scales: 12 bytes

		// Unpack 16 x 6-bit scales from 12 bytes
		var aux [4]uint32
		for i := 0; i < 12; i++ {
			byteIdx := i / 4
			shift := uint((i % 4) * 8)
			aux[byteIdx] |= uint32(data[scOff+i]) << shift
		}
		tmp := aux[2]
		aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
		aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
		aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
		aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)

		// Extract scales as int8
		var scales [16]int8
		for i := 0; i < 16; i++ {
			scales[i] = int8(byte(aux[i/4] >> uint((i%4)*8)))
		}

		outOff := block * BlockSizeQ3_K
		is := 0
		m := byte(1)
		for n128 := 0; n128 < 2; n128++ {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := dAll * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					q2 := int((data[qOff+l] >> shift) & 3)
					hBit := 0
					if data[hmOff+l]&m != 0 {
						hBit = 0
					} else {
						hBit = 4
					}
					result[outOff] = dl * float32(q2-hBit)
					outOff++
				}

				dl = dAll * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					q2 := int((data[qOff+l+16] >> shift) & 3)
					hBit := 0
					if data[hmOff+l+16]&m != 0 {
						hBit = 0
					} else {
						hBit = 4
					}
					result[outOff] = dl * float32(q2-hBit)
					outOff++
				}

				shift += 2
				m <<= 1
			}
			qOff += 32
		}
	}
	return result
}
