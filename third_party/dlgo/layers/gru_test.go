package layers

import (
	"math"
	"testing"

	"github.com/computerex/dlgo/ops"
)

func TestGRUCellF32Zero(t *testing.T) {
	hiddenDim := 2
	inputDim := 2
	gatesDim := 3 * hiddenDim

	// Zero weights/biases
	wIH := make([]float32, gatesDim*inputDim)
	wHH := make([]float32, gatesDim*hiddenDim)
	bIH := make([]float32, gatesDim)
	bHH := make([]float32, gatesDim)

	state := NewGRUState(hiddenDim)
	x := []float32{1, 1}
	scratch := make([]float32, 2*gatesDim)

	GRUCellF32(x, state, wIH, wHH, bIH, bHH, hiddenDim, inputDim, scratch)

	// With zero weights: r=sigmoid(0)=0.5, z=sigmoid(0)=0.5
	// n = tanh(0 + 0.5 * 0) = tanh(0) ≈ 0 (approximate)
	// h' = (1-0.5)*n + 0.5*0 ≈ 0
	for j := 0; j < hiddenDim; j++ {
		if math.Abs(float64(state.H[j])) > 0.05 {
			t.Errorf("H[%d] = %f, want ~0", j, state.H[j])
		}
	}
}

func TestGRUStateClone(t *testing.T) {
	state := NewGRUState(3)
	state.H[0] = 1.5
	state.H[1] = -2.5
	state.H[2] = 3.0

	c := state.Clone()
	state.H[0] = 999

	if c.H[0] != 1.5 {
		t.Error("Clone was not deep")
	}
}

func TestGRUStateReset(t *testing.T) {
	state := NewGRUState(3)
	state.H[0] = 5
	state.Reset()
	for i, v := range state.H {
		if v != 0 {
			t.Errorf("Reset: H[%d] = %f, want 0", i, v)
		}
	}
}

func TestGRUCellF32UpdateGate(t *testing.T) {
	// Test that the GRU output changes over multiple steps
	hiddenDim := 2
	inputDim := 2
	gatesDim := 3 * hiddenDim

	wIH := make([]float32, gatesDim*inputDim)
	wHH := make([]float32, gatesDim*hiddenDim)
	bIH := make([]float32, gatesDim)
	bHH := make([]float32, gatesDim)

	// Set some non-trivial weights for the new gate (last hiddenDim rows)
	for i := 2 * hiddenDim * inputDim; i < gatesDim*inputDim; i++ {
		wIH[i] = 0.5
	}

	state := NewGRUState(hiddenDim)
	x := []float32{1, 1}
	scratch := make([]float32, 2*gatesDim)

	_ = ops.Clear // ensure ops is used
	GRUCellF32(x, state, wIH, wHH, bIH, bHH, hiddenDim, inputDim, scratch)

	// After one step with non-zero new-gate weights, H should be non-zero
	nonZero := false
	for _, v := range state.H {
		if math.Abs(float64(v)) > 1e-3 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Error("expected non-zero hidden state after step with non-trivial weights")
	}
}
