// Package layers provides neural network layer implementations for inference.
//
// All layers operate on flat float32 slices using row-major layout.
// Quantized weights use core.QuantizedTensor and the blas package for matmul.
package layers

import "github.com/computerex/dlgo/ops"

// DepthwiseConv1D performs depthwise 1D convolution with same padding.
// Each channel is convolved independently with its own kernel.
//
// Layout (all row-major):
//   x:      [channels][seqLen]
//   kernel: [channels][kernelSize]
//   bias:   [channels] (nil = no bias)
//   out:    [channels][seqLen]
func DepthwiseConv1D(out, x, kernel, bias []float32, channels, seqLen, kernelSize int) {
	pad := kernelSize / 2
	for c := 0; c < channels; c++ {
		xBase := c * seqLen
		kBase := c * kernelSize
		oBase := c * seqLen
		for t := 0; t < seqLen; t++ {
			var sum float32
			for k := 0; k < kernelSize; k++ {
				inIdx := t + k - pad
				if inIdx >= 0 && inIdx < seqLen {
					sum += x[xBase+inIdx] * kernel[kBase+k]
				}
			}
			b := float32(0)
			if bias != nil {
				b = bias[c]
			}
			out[oBase+t] = sum + b
		}
	}
}

// Conv2DDepthwise performs 2D depthwise convolution.
// Each output channel is computed from one input channel (groups = outCh).
//
//   weight: [outCh * kH * kW]  (one kH×kW filter per output channel)
//   bias:   [outCh] (nil = no bias)
//   input:  [inCh * inH * inW]
//   output: [outCh * outH * outW]  where outH = (inH+2*padH-kH)/strH+1
func Conv2DDepthwise(output, input, weight, bias []float32,
	inCh, outCh, inH, inW, kH, kW, strH, strW, padH, padW int) {

	outH := (inH+2*padH-kH)/strH + 1
	outW := (inW+2*padW-kW)/strW + 1

	for oc := 0; oc < outCh; oc++ {
		ic := oc % inCh
		b := float32(0)
		if bias != nil && oc < len(bias) {
			b = bias[oc]
		}
		inBase := ic * inH * inW
		outBase := oc * outH * outW
		wBase := oc * kH * kW

		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				var sum float32
				for fh := 0; fh < kH; fh++ {
					ih := oh*strH - padH + fh
					if ih < 0 || ih >= inH {
						continue
					}
					for fw := 0; fw < kW; fw++ {
						iw := ow*strW - padW + fw
						if iw < 0 || iw >= inW {
							continue
						}
						sum += input[inBase+ih*inW+iw] * weight[wBase+fh*kW+fw]
					}
				}
				output[outBase+oh*outW+ow] = sum + b
			}
		}
	}
}

// Conv2DPointwise performs 2D pointwise (1×1) convolution.
//   weight: [outCh * inCh]   (one 1×1 filter per output channel)
//   bias:   [outCh] (nil = no bias)
//   input:  [inCh * h * w]
//   output: [outCh * h * w]
func Conv2DPointwise(output, input, weight, bias []float32,
	inCh, outCh, h, w int) {

	spatialSize := h * w

	// Transpose input for cache-friendly access in the inner dot product.
	inputT := make([]float32, spatialSize*inCh)
	for s := 0; s < spatialSize; s++ {
		for ic := 0; ic < inCh; ic++ {
			inputT[s*inCh+ic] = input[ic*spatialSize+s]
		}
	}

	for oc := 0; oc < outCh; oc++ {
		b := float32(0)
		if bias != nil && oc < len(bias) {
			b = bias[oc]
		}
		wRow := weight[oc*inCh : (oc+1)*inCh]
		outRow := output[oc*spatialSize : (oc+1)*spatialSize]
		for s := 0; s < spatialSize; s++ {
			outRow[s] = ops.DotProduct(wRow, inputT[s*inCh:(s+1)*inCh], inCh) + b
		}
	}
}

// Conv1D performs standard 1D convolution (not depthwise).
//   weight: [outCh * inCh * kernelSize]
//   bias:   [outCh] (nil = no bias)
//   input:  [inCh * seqLen]
//   output: [outCh * outLen]  where outLen = (seqLen+2*pad-kernelSize)/stride+1
func Conv1D(output, input, weight, bias []float32,
	outCh, inCh, kernelSize, stride, pad, seqLen int) {

	outLen := (seqLen+2*pad-kernelSize)/stride + 1

	for oc := 0; oc < outCh; oc++ {
		b := float32(0)
		if bias != nil {
			b = bias[oc]
		}
		for t := 0; t < outLen; t++ {
			var sum float32
			for ic := 0; ic < inCh; ic++ {
				for k := 0; k < kernelSize; k++ {
					inIdx := t*stride + k - pad
					if inIdx >= 0 && inIdx < seqLen {
						sum += input[ic*seqLen+inIdx] * weight[oc*inCh*kernelSize+ic*kernelSize+k]
					}
				}
			}
			output[oc*outLen+t] = sum + b
		}
	}
}
