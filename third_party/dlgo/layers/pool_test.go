package layers

import (
	"math"
	"testing"
)

func TestMaxPool1D(t *testing.T) {
	// 1 channel, length 6, kernel 3, stride 2
	input := []float32{1, 3, 2, 5, 4, 6}
	outLen := (6-3)/2 + 1 // 2
	output := make([]float32, outLen)
	MaxPool1D(output, input, 1, 6, 3, 2)

	// Window 0: [1,3,2] -> 3
	// Window 1: [2,5,4] -> 5
	want := []float32{3, 5}
	for i := range output {
		if output[i] != want[i] {
			t.Errorf("MaxPool1D[%d] = %f, want %f", i, output[i], want[i])
		}
	}
}

func TestAvgPool1D(t *testing.T) {
	input := []float32{1, 3, 2, 5, 4, 6}
	outLen := (6-3)/2 + 1
	output := make([]float32, outLen)
	AvgPool1D(output, input, 1, 6, 3, 2)

	// Window 0: (1+3+2)/3 = 2
	// Window 1: (2+5+4)/3 = 3.666...
	if math.Abs(float64(output[0]-2)) > 0.01 {
		t.Errorf("AvgPool1D[0] = %f, want 2", output[0])
	}
	if math.Abs(float64(output[1]-11.0/3.0)) > 0.01 {
		t.Errorf("AvgPool1D[1] = %f, want %f", output[1], 11.0/3.0)
	}
}

func TestMaxPool2D(t *testing.T) {
	// 1 channel, 4x4 input, 2x2 kernel, stride 2
	input := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	outH := (4-2)/2 + 1
	outW := (4-2)/2 + 1
	output := make([]float32, outH*outW)
	MaxPool2D(output, input, 1, 4, 4, 2, 2, 2, 2)

	want := []float32{6, 8, 14, 16}
	for i := range output {
		if output[i] != want[i] {
			t.Errorf("MaxPool2D[%d] = %f, want %f", i, output[i], want[i])
		}
	}
}

func TestAvgPool2D(t *testing.T) {
	input := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	outH := (4-2)/2 + 1
	outW := (4-2)/2 + 1
	output := make([]float32, outH*outW)
	AvgPool2D(output, input, 1, 4, 4, 2, 2, 2, 2)

	want := []float32{3.5, 5.5, 11.5, 13.5}
	for i := range output {
		if math.Abs(float64(output[i]-want[i])) > 0.01 {
			t.Errorf("AvgPool2D[%d] = %f, want %f", i, output[i], want[i])
		}
	}
}

func TestGlobalAvgPool1D(t *testing.T) {
	// 2 channels, seqLen 3
	input := []float32{1, 2, 3, 4, 5, 6}
	output := make([]float32, 2)
	GlobalAvgPool1D(output, input, 2, 3)

	// Channel 0: (1+2+3)/3 = 2
	// Channel 1: (4+5+6)/3 = 5
	if math.Abs(float64(output[0]-2)) > 0.01 {
		t.Errorf("GlobalAvgPool1D[0] = %f, want 2", output[0])
	}
	if math.Abs(float64(output[1]-5)) > 0.01 {
		t.Errorf("GlobalAvgPool1D[1] = %f, want 5", output[1])
	}
}
