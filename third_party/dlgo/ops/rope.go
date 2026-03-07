package ops

import "math"

// ApplyRoPE applies Rotary Positional Embedding to a single attention head vector.
//
//   vec:      [headDim] — one Q or K head vector (modified in-place)
//   pos:      absolute position of this token
//   headDim:  dimension per attention head
//   freqBase: RoPE frequency base (10000.0 for LLaMA, 1000000.0 for some Qwen)
//   neox:     if true, use split-half pairing (GPT-NeoX/Qwen style);
//             if false, use interleaved pairing (LLaMA style)
func ApplyRoPE(vec []float32, pos int, headDim int, freqBase float32, neox bool) {
	half := headDim / 2
	for i := 0; i < half; i++ {
		theta := 1.0 / math.Pow(float64(freqBase), float64(2*i)/float64(headDim))
		angle := float64(pos) * theta
		cos := float32(math.Cos(angle))
		sin := float32(math.Sin(angle))

		if neox {
			x0 := vec[i]
			x1 := vec[i+half]
			vec[i] = x0*cos - x1*sin
			vec[i+half] = x0*sin + x1*cos
		} else {
			x0 := vec[2*i]
			x1 := vec[2*i+1]
			vec[2*i] = x0*cos - x1*sin
			vec[2*i+1] = x0*sin + x1*cos
		}
	}
}

// RoPEFrequencyTable precomputes the cos/sin tables for RoPE.
// Returns (cosTable, sinTable) each of shape [maxLen * headDim/2].
// Use with ApplyRoPEFromTable for faster inference when processing many positions.
func RoPEFrequencyTable(maxLen, headDim int, freqBase float32) (cosTable, sinTable []float32) {
	half := headDim / 2
	cosTable = make([]float32, maxLen*half)
	sinTable = make([]float32, maxLen*half)

	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < half; i++ {
			theta := 1.0 / math.Pow(float64(freqBase), float64(2*i)/float64(headDim))
			angle := float64(pos) * theta
			cosTable[pos*half+i] = float32(math.Cos(angle))
			sinTable[pos*half+i] = float32(math.Sin(angle))
		}
	}
	return
}

// ApplyRoPEFromTable applies RoPE using precomputed cos/sin tables.
// Faster than ApplyRoPE for repeated calls at different positions.
func ApplyRoPEFromTable(vec []float32, pos int, headDim int, cosTable, sinTable []float32, neox bool) {
	half := headDim / 2
	base := pos * half

	if neox {
		for i := 0; i < half; i++ {
			cos := cosTable[base+i]
			sin := sinTable[base+i]
			x0 := vec[i]
			x1 := vec[i+half]
			vec[i] = x0*cos - x1*sin
			vec[i+half] = x0*sin + x1*cos
		}
	} else {
		for i := 0; i < half; i++ {
			cos := cosTable[base+i]
			sin := sinTable[base+i]
			x0 := vec[2*i]
			x1 := vec[2*i+1]
			vec[2*i] = x0*cos - x1*sin
			vec[2*i+1] = x0*sin + x1*cos
		}
	}
}

// ApplyRoPEBatch applies RoPE to all Q and K heads at a given position.
//   qFlat: [numHeads * headDim]   — all Q heads concatenated
//   kFlat: [numKVHeads * headDim] — all K heads concatenated
func ApplyRoPEBatch(qFlat []float32, numHeads int, kFlat []float32, numKVHeads int,
	pos, headDim int, freqBase float32, neox bool) {
	for h := 0; h < numHeads; h++ {
		ApplyRoPE(qFlat[h*headDim:(h+1)*headDim], pos, headDim, freqBase, neox)
	}
	for h := 0; h < numKVHeads; h++ {
		ApplyRoPE(kFlat[h*headDim:(h+1)*headDim], pos, headDim, freqBase, neox)
	}
}
