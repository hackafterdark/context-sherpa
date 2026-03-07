package silero

// encode runs the 4-layer Conv1D encoder with ReLU activations.
// Input: [129][4] magnitude spectrogram from STFT.
// Output: [128] feature vector (first time step of final layer output).
//
// Layer 0: Conv1D(129→128, k=3, s=1, p=1) + ReLU → [128][4]
// Layer 1: Conv1D(128→64,  k=3, s=2, p=1) + ReLU → [64][2]
// Layer 2: Conv1D(64→64,   k=3, s=2, p=1) + ReLU → [64][1]
// Layer 3: Conv1D(64→128,  k=3, s=1, p=1) + ReLU → [128][1]
func encode(input [][]float32, model *SileroModel) []float32 {
	strides := [4]int{1, 2, 2, 1}
	cur := input

	for i := 0; i < 4; i++ {
		inCh := int(model.EncoderInCh[i])
		outCh := int(model.EncoderOutCh[i])
		k := int(model.KernelSizes[i])
		cur = conv1d(cur, model.EncoderWeights[i], model.EncoderBiases[i],
			outCh, inCh, k, strides[i], 1)
		relu2d(cur)
	}

	// Extract first time step → flat [128] vector
	result := make([]float32, len(cur))
	for i := range cur {
		result[i] = cur[i][0]
	}
	return result
}

// conv1d performs standard 1D convolution with zero-padding.
// Weight flat layout: weight[oc*(inCh*kernel) + ic*kernel + k]
// Returns [outCh][outLen].
func conv1d(input [][]float32, weight []float32, bias []float32,
	outCh, inCh, kernel, stride, padding int) [][]float32 {

	inLen := len(input[0])
	outLen := (inLen + 2*padding - kernel) / stride + 1

	output := make([][]float32, outCh)
	for oc := 0; oc < outCh; oc++ {
		output[oc] = make([]float32, outLen)
		for t := 0; t < outLen; t++ {
			sum := bias[oc]
			start := t*stride - padding
			for ic := 0; ic < inCh; ic++ {
				wBase := oc*inCh*kernel + ic*kernel
				for k := 0; k < kernel; k++ {
					inIdx := start + k
					if inIdx >= 0 && inIdx < inLen {
						sum += weight[wBase+k] * input[ic][inIdx]
					}
				}
			}
			output[oc][t] = sum
		}
	}
	return output
}

// relu2d applies ReLU activation in-place on a [channels][length] array.
func relu2d(data [][]float32) {
	for i := range data {
		for j := range data[i] {
			if data[i][j] < 0 {
				data[i][j] = 0
			}
		}
	}
}
