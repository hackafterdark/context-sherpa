package quant

import "encoding/binary"

// IQ2_XXS quantization (type 16)
// Block structure: 66 bytes per block of 256 values
// Layout:
//   d:  float16 super-block scale (2 bytes)
//   qs: uint16[32] = 64 bytes — packed grid indices + signs + scales
//
// Processed as 8 groups of 32 values. Each group uses 8 bytes from qs:
//   aux32[0] (bytes 0..3): 4 grid indices into iq2xxs_grid[256]
//   aux32[1] (bytes 4..7): bits 28..31 = 4-bit scale,
//                          4 groups of 7-bit sign indices at bits [7*l..7*l+6]

const BlockSizeIQ2XXS = 256
const BlockBytesIQ2XXS = 66

// DequantizeIQ2XXS converts IQ2_XXS quantized data to float32.
func DequantizeIQ2XXS(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeIQ2XXS

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ2XXS

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsOff := off + 2 // start of qs (64 bytes of uint16 data)
		outBase := block * BlockSizeIQ2XXS

		for ib32 := 0; ib32 < 8; ib32++ {
			// Read 8 bytes as two uint32s
			byteOff := qsOff + ib32*8
			aux1 := binary.LittleEndian.Uint32(data[byteOff+4:])

			db := d * (0.5 + float32(aux1>>28)) * 0.25

			for l := 0; l < 4; l++ {
				gridIdx := data[byteOff+l] // aux8[l] = byte l of aux32[0]
				grid := iq2xxs_grid[gridIdx]

				signIdx := (aux1 >> (7 * uint(l))) & 127
				signs := ksigns_iq2xs[signIdx]

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
