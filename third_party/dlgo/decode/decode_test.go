package decode

import (
	"math"
	"testing"
)

func TestGreedyDecode(t *testing.T) {
	// Vocabulary: 0=a, 1=b, 2=eot
	// logitsFunc always returns highest logit for token 0, then 1, then eot
	step := 0
	tokens := GreedyDecode(0, 2, 10, func(tokenID, pos int) []float32 {
		step++
		switch step {
		case 1:
			return []float32{10, 1, 0}
		case 2:
			return []float32{1, 10, 0}
		default:
			return []float32{0, 0, 10} // eot
		}
	})

	if len(tokens) != 2 {
		t.Fatalf("got %d tokens, want 2", len(tokens))
	}
	if tokens[0] != 0 || tokens[1] != 1 {
		t.Errorf("tokens = %v, want [0, 1]", tokens)
	}
}

func TestGreedyDecode_MaxSteps(t *testing.T) {
	// Never produce eot
	tokens := GreedyDecode(0, 99, 5, func(tokenID, pos int) []float32 {
		return []float32{10, 1} // always emit token 0
	})

	if len(tokens) != 5 {
		t.Errorf("got %d tokens, want 5 (maxSteps)", len(tokens))
	}
}

func TestGreedyDecode_ImmediateEOT(t *testing.T) {
	tokens := GreedyDecode(0, 1, 100, func(tokenID, pos int) []float32 {
		return []float32{0, 10} // always eot
	})

	if len(tokens) != 0 {
		t.Errorf("got %d tokens, want 0", len(tokens))
	}
}

func TestLengthNormalization(t *testing.T) {
	// alpha=0 should return 1.0
	if LengthNormalization(10, 0) != 1.0 {
		t.Error("alpha=0 should return 1.0")
	}

	// alpha=1: penalty = (5+seqLen)/6
	got := LengthNormalization(7, 1.0)
	want := float32(12.0 / 6.0)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Errorf("LengthNormalization(7, 1.0) = %f, want %f", got, want)
	}

	// Longer sequences should have higher normalization
	short := LengthNormalization(1, 1.0)
	long := LengthNormalization(100, 1.0)
	if long <= short {
		t.Error("longer sequences should have higher normalization")
	}
}

func TestTopKIndicesHeap(t *testing.T) {
	vals := []float32{1, 5, 3, 7, 2, 8, 4}
	top3 := TopKIndicesHeap(vals, 3)

	if len(top3) != 3 {
		t.Fatalf("got %d indices, want 3", len(top3))
	}

	// Should be indices of 8, 7, 5 (values at positions 5, 3, 1)
	if top3[0] != 5 {
		t.Errorf("top1 index = %d, want 5 (val=8)", top3[0])
	}
	if top3[1] != 3 {
		t.Errorf("top2 index = %d, want 3 (val=7)", top3[1])
	}
}

func TestTopKIndicesHeap_KLargerThanN(t *testing.T) {
	vals := []float32{3, 1, 2}
	got := TopKIndicesHeap(vals, 10)
	if len(got) != 3 {
		t.Errorf("got %d indices, want 3 (clamped to len)", len(got))
	}
}

func TestTopKIndicesHeap_Empty(t *testing.T) {
	got := TopKIndicesHeap(nil, 5)
	if len(got) != 0 {
		t.Error("should return empty for nil input")
	}
}
