package layers

// MaxPool1D applies 1D max pooling.
//   input:  [channels * inLen]  stored as [channels][inLen]
//   output: [channels * outLen] where outLen = (inLen - kernelSize) / stride + 1
func MaxPool1D(output, input []float32, channels, inLen, kernelSize, stride int) {
	outLen := (inLen-kernelSize)/stride + 1
	for c := 0; c < channels; c++ {
		inBase := c * inLen
		outBase := c * outLen
		for i := 0; i < outLen; i++ {
			start := i * stride
			maxVal := input[inBase+start]
			for k := 1; k < kernelSize; k++ {
				v := input[inBase+start+k]
				if v > maxVal {
					maxVal = v
				}
			}
			output[outBase+i] = maxVal
		}
	}
}

// AvgPool1D applies 1D average pooling.
func AvgPool1D(output, input []float32, channels, inLen, kernelSize, stride int) {
	outLen := (inLen-kernelSize)/stride + 1
	invK := 1.0 / float32(kernelSize)
	for c := 0; c < channels; c++ {
		inBase := c * inLen
		outBase := c * outLen
		for i := 0; i < outLen; i++ {
			start := i * stride
			var sum float32
			for k := 0; k < kernelSize; k++ {
				sum += input[inBase+start+k]
			}
			output[outBase+i] = sum * invK
		}
	}
}

// MaxPool2D applies 2D max pooling.
//   input:  [channels * inH * inW]
//   output: [channels * outH * outW]
func MaxPool2D(output, input []float32, channels, inH, inW, kH, kW, strH, strW int) {
	outH := (inH-kH)/strH + 1
	outW := (inW-kW)/strW + 1
	for c := 0; c < channels; c++ {
		inBase := c * inH * inW
		outBase := c * outH * outW
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				maxVal := input[inBase+oh*strH*inW+ow*strW]
				for fh := 0; fh < kH; fh++ {
					for fw := 0; fw < kW; fw++ {
						v := input[inBase+(oh*strH+fh)*inW+ow*strW+fw]
						if v > maxVal {
							maxVal = v
						}
					}
				}
				output[outBase+oh*outW+ow] = maxVal
			}
		}
	}
}

// AvgPool2D applies 2D average pooling.
func AvgPool2D(output, input []float32, channels, inH, inW, kH, kW, strH, strW int) {
	outH := (inH-kH)/strH + 1
	outW := (inW-kW)/strW + 1
	invK := 1.0 / float32(kH*kW)
	for c := 0; c < channels; c++ {
		inBase := c * inH * inW
		outBase := c * outH * outW
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				var sum float32
				for fh := 0; fh < kH; fh++ {
					for fw := 0; fw < kW; fw++ {
						sum += input[inBase+(oh*strH+fh)*inW+ow*strW+fw]
					}
				}
				output[outBase+oh*outW+ow] = sum * invK
			}
		}
	}
}

// GlobalAvgPool1D computes global average pooling over the time dimension.
//   input: [channels * seqLen], output: [channels]
func GlobalAvgPool1D(output, input []float32, channels, seqLen int) {
	invLen := 1.0 / float32(seqLen)
	for c := 0; c < channels; c++ {
		var sum float32
		base := c * seqLen
		for i := 0; i < seqLen; i++ {
			sum += input[base+i]
		}
		output[c] = sum * invLen
	}
}
