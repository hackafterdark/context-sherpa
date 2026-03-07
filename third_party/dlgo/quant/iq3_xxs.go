package quant

import "encoding/binary"

// IQ3_XXS quantization (type 18)
// Block structure: 98 bytes per block of 256 values
// Layout:
//   d:  float16 super-block scale (2 bytes)
//   qs: uint8[96] — first 64 bytes: grid indices, last 32 bytes: scales + signs
//
// 8 groups of 32 values. Each group:
//   Grid indices: 8 bytes from qs (indices into iq3xxs_grid[256], each gives 4 values)
//   Scales+signs: 4 bytes read as uint32:
//     bits 28..31 = 4-bit scale
//     4 groups of 7-bit sign indices at bits [7*l..7*l+6]

const BlockSizeIQ3XXS = 256
const BlockBytesIQ3XXS = 98

// DequantizeIQ3XXS converts IQ3_XXS quantized data to float32.
func DequantizeIQ3XXS(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeIQ3XXS

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ3XXS

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsOff := off + 2                 // grid indices start
		scalesSignsOff := off + 2 + 64   // scales_and_signs start (32 bytes)
		outBase := block * BlockSizeIQ3XXS

		gridOff := qsOff

		for ib32 := 0; ib32 < 8; ib32++ {
			// Read 4-byte scales+signs block
			aux32 := binary.LittleEndian.Uint32(data[scalesSignsOff+ib32*4:])
			db := d * (0.5 + float32(aux32>>28)) * 0.5

			for l := 0; l < 4; l++ {
				signIdx := (aux32 >> (7 * uint(l))) & 127
				signs := ksigns_iq2xs[signIdx]

				// Two grid lookups per sub-group of 8
				grid1 := iq3xxs_grid[data[gridOff+2*l]]
				grid2 := iq3xxs_grid[data[gridOff+2*l+1]]

				oIdx := outBase + ib32*32 + l*8
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid1 >> (8 * uint(j))))
					if signs&kmask_iq2xs[j] != 0 {
						result[oIdx+j] = -db * gridVal
					} else {
						result[oIdx+j] = db * gridVal
					}
				}
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid2 >> (8 * uint(j))))
					if signs&kmask_iq2xs[j+4] != 0 {
						result[oIdx+4+j] = -db * gridVal
					} else {
						result[oIdx+4+j] = db * gridVal
					}
				}
			}
			gridOff += 8
		}
	}
	return result
}
