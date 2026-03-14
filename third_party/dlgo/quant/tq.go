package quant

// TQ1_0 — Ternary quantization, 1.6875 bpw
// Block structure: 54 bytes per super-block of 256 values
// Layout:
//   qs[48]: 5 ternary digits per byte via base-3 (3^5=243 < 256)
//   qh[4]: 4 ternary digits per byte (remaining 16 values)
//   d: float16 scale (2 bytes)
// Values are {-1, 0, +1} * d

const BlockSizeTQ1_0 = 256
const BlockBytesTQ1_0 = 54

// DequantizeTQ1_0 converts TQ1_0 quantized data to float32.
func DequantizeTQ1_0(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeTQ1_0
	pow3 := [6]uint8{1, 3, 9, 27, 81, 243}

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesTQ1_0
		outOff := block * BlockSizeTQ1_0

		// d is at offset 52 (after qs[48] + qh[4])
		dBits := uint16(data[off+52]) | uint16(data[off+53])<<8
		d := float16ToFloat32(dBits)

		// Main qs: 48 bytes, 5 trits each = 240 values
		// Process in 32-byte chunks, then remaining 16-byte chunks
		qsLen := 48
		qsChunk32 := qsLen - (qsLen % 32) // = 32
		// Process 32-byte chunk
		for j := 0; j < qsChunk32; j += 32 {
			for nn := 0; nn < 5; nn++ {
				for m := 0; m < 32; m++ {
					q := data[off+j+m]
					q = uint8((uint16(q) * uint16(pow3[nn])) >> 0)
					xi := int16((uint16(q) * 3) >> 8)
					result[outOff] = float32(xi-1) * d
					outOff++
				}
			}
		}
		// Process remaining 16-byte chunk (48-32=16)
		for j := qsChunk32; j < qsLen; j += 16 {
			for nn := 0; nn < 5; nn++ {
				for m := 0; m < 16; m++ {
					q := data[off+j+m]
					q = uint8((uint16(q) * uint16(pow3[nn])) >> 0)
					xi := int16((uint16(q) * 3) >> 8)
					result[outOff] = float32(xi-1) * d
					outOff++
				}
			}
		}

		// qh: 4 bytes at offset 48, 4 trits per byte = 16 values
		qhOff := off + 48
		for nn := 0; nn < 4; nn++ {
			for j := 0; j < 4; j++ {
				q := data[qhOff+j]
				q = uint8((uint16(q) * uint16(pow3[nn])) >> 0)
				xi := int16((uint16(q) * 3) >> 8)
				result[outOff] = float32(xi-1) * d
				outOff++
			}
		}
	}
	return result
}

// TQ2_0 — Ternary quantization, 2.0625 bpw
// Block structure: 66 bytes per super-block of 256 values
// Layout:
//   qs[64]: 2 bits per value, 4 values per byte (64 bytes)
//   d: float16 scale (2 bytes)
// Values are {-1, 0, +1} * d, stored as {0, 1, 2} mapped to {-1, 0, +1}

const BlockSizeTQ2_0 = 256
const BlockBytesTQ2_0 = 66

// DequantizeTQ2_0 converts TQ2_0 quantized data to float32.
func DequantizeTQ2_0(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeTQ2_0

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesTQ2_0
		outOff := block * BlockSizeTQ2_0

		// d is at offset 64
		dBits := uint16(data[off+64]) | uint16(data[off+65])<<8
		d := float16ToFloat32(dBits)

		// Process 64 bytes in 32-byte chunks
		for j := 0; j < 64; j += 32 {
			for l := 0; l < 4; l++ {
				shift := uint(l * 2)
				for m := 0; m < 32; m++ {
					q := int8((data[off+j+m] >> shift) & 3)
					result[outOff] = float32(q-1) * d
					outOff++
				}
			}
		}
	}
	return result
}
