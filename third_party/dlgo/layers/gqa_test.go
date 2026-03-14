package layers

import (
	"math"
	"testing"
)

func TestGroupedQueryAttentionSinglePos(t *testing.T) {
	numHeads := 2
	numKVHeads := 1
	headDim := 4
	seqLen := 1

	// Q: all ones
	q := make([]float32, numHeads*headDim)
	for i := range q {
		q[i] = 1
	}

	// Single K/V position: identity-like
	kCache := make([][]float32, seqLen)
	vCache := make([][]float32, seqLen)
	kCache[0] = make([]float32, numKVHeads*headDim)
	vCache[0] = make([]float32, numKVHeads*headDim)
	for i := 0; i < numKVHeads*headDim; i++ {
		kCache[0][i] = 1
		vCache[0][i] = float32(i + 1)
	}

	out := make([]float32, numHeads*headDim)
	GroupedQueryAttention(out, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim)

	// With single position, softmax = [1], so out = V
	for h := 0; h < numHeads; h++ {
		for d := 0; d < headDim; d++ {
			want := vCache[0][d] // both heads share the same KV head
			got := out[h*headDim+d]
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Errorf("GQA head %d dim %d: got %f, want %f", h, d, got, want)
			}
		}
	}
}

func TestGroupedQueryAttentionMHA(t *testing.T) {
	// When numKVHeads == numHeads, GQA reduces to standard MHA
	numHeads := 2
	numKVHeads := 2
	headDim := 2
	seqLen := 2

	q := []float32{1, 0, 0, 1} // head 0: [1,0], head 1: [0,1]

	kCache := make([][]float32, seqLen)
	vCache := make([][]float32, seqLen)

	// K and V for 2 positions
	kCache[0] = []float32{1, 0, 0, 1} // KV head 0: [1,0], head 1: [0,1]
	kCache[1] = []float32{0, 1, 1, 0}
	vCache[0] = []float32{1, 0, 0, 1}
	vCache[1] = []float32{0, 1, 1, 0}

	out := make([]float32, numHeads*headDim)
	GroupedQueryAttention(out, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim)

	// Output should be valid (not NaN)
	for i, v := range out {
		if math.IsNaN(float64(v)) {
			t.Errorf("GQA MHA out[%d] is NaN", i)
		}
	}
}

func TestCrossAttention(t *testing.T) {
	numHeads := 1
	headDim := 2
	encLen := 3

	q := []float32{1, 0}

	// Encoder K/V
	kEnc := []float32{
		1, 0, // pos 0
		0, 1, // pos 1
		1, 1, // pos 2
	}
	vEnc := []float32{
		1, 0,
		0, 1,
		0.5, 0.5,
	}

	out := make([]float32, numHeads*headDim)
	CrossAttention(out, q, kEnc, vEnc, encLen, numHeads, headDim)

	// Should be a weighted combination of V
	for i, v := range out {
		if math.IsNaN(float64(v)) {
			t.Errorf("CrossAttention out[%d] is NaN", i)
		}
	}

	// q=[1,0] should attend more to k[0]=[1,0] and k[2]=[1,1] than k[1]=[0,1]
	// So output should lean toward v[0] and v[2]
	if out[0] < 0.3 {
		t.Errorf("CrossAttention expected higher out[0], got %f", out[0])
	}
}

func TestSlidingWindowAttention(t *testing.T) {
	numHeads := 1
	numKVHeads := 1
	headDim := 2
	seqLen := 4
	windowSize := 2

	q := []float32{1, 0}

	kCache := make([][]float32, seqLen)
	vCache := make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		kCache[i] = []float32{1, 0}
		vCache[i] = []float32{float32(i), float32(i)}
	}

	out := make([]float32, numHeads*headDim)

	// At pos=3 with window=2, should only attend to positions 2 and 3
	SlidingWindowAttention(out, q, kCache, vCache, 3, seqLen, numHeads, numKVHeads, headDim, windowSize)

	// Both positions have same K, so equal attention => mean of v[2] and v[3]
	expected := float32(2.5)
	if math.Abs(float64(out[0]-expected)) > 0.01 {
		t.Errorf("SlidingWindow out[0] = %f, want %f", out[0], expected)
	}
}
