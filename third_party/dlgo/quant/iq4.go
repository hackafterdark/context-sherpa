package quant

// IQ4_NL — Importance-quantized 4-bit, Non-Linear
// Block structure: 18 bytes per block of 32 values
// Layout:
//   d: float16 scale (2 bytes)
//   qs[16]: 4-bit indices packed in nibble pairs (16 bytes)
// Each nibble indexes into kvalues_iq4nl[16] for non-linear mapping.
// Formula: y = d * kvalues_iq4nl[nibble]

const BlockSizeIQ4_NL = 32
const BlockBytesIQ4_NL = 18

// kvalues_iq4nl is the non-linear reconstruction table.
// Each 4-bit index (0-15) maps to one of these 16 levels.
var kvalues_iq4nl = [16]int8{
	-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
}

// DequantizeIQ4_NL converts IQ4_NL quantized data to float32.
func DequantizeIQ4_NL(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeIQ4_NL

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ4_NL

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		base := block * BlockSizeIQ4_NL
		for j := 0; j < 16; j++ {
			qByte := data[off+2+j]
			result[base+j] = d * float32(kvalues_iq4nl[qByte&0xf])
			result[base+j+16] = d * float32(kvalues_iq4nl[qByte>>4])
		}
	}
	return result
}

// IQ4_XS — Super-block version of IQ4_NL (4.25 bpw)
// Block structure: 136 bytes per super-block of 256 values
// Layout:
//   d: float16 super-block scale (2 bytes)
//   scales_h: uint16 high 2 bits of each of 8 scales (2 bytes)
//   scales_l[4]: low 4 bits of each of 8 scales (4 bytes)
//   qs[128]: 4-bit indices packed in nibble pairs (128 bytes)
// 8 sub-blocks of 32 elements, each with 6-bit scale (offset by -32).
// Formula: y = d * (scale - 32) * kvalues_iq4nl[nibble]

const BlockSizeIQ4_XS = 256
const BlockBytesIQ4_XS = 136

// DequantizeIQ4_XS converts IQ4_XS quantized data to float32.
func DequantizeIQ4_XS(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeIQ4_XS

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ4_XS

		// d at offset 0
		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		// scales_h at offset 2 (uint16 LE)
		scalesH := uint16(data[off+2]) | uint16(data[off+3])<<8

		// scales_l at offset 4 (4 bytes)
		scalesLOff := off + 4

		// qs at offset 8 (128 bytes)
		qsOff := off + 8

		outOff := block * BlockSizeIQ4_XS
		for ib := 0; ib < 8; ib++ {
			// Reconstruct 6-bit scale: 4 low bits from scales_l + 2 high bits from scales_h
			lo := (data[scalesLOff+ib/2] >> uint(4*(ib%2))) & 0xf
			hi := (byte(scalesH>>uint(2*ib)) & 3) << 4
			ls := int(lo | hi)
			dl := d * float32(ls-32)

			for j := 0; j < 16; j++ {
				qByte := data[qsOff+ib*16+j]
				result[outOff+j] = dl * float32(kvalues_iq4nl[qByte&0xf])
				result[outOff+j+16] = dl * float32(kvalues_iq4nl[qByte>>4])
			}
			outOff += 32
		}
	}
	return result
}
