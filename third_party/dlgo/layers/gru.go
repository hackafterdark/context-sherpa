package layers

import (
	"github.com/computerex/dlgo/ops"
)

// GRUState holds the hidden state for a GRU layer.
type GRUState struct {
	H []float32 // hidden state [hiddenDim]
}

// NewGRUState creates a zero-initialized GRU state.
func NewGRUState(hiddenDim int) *GRUState {
	return &GRUState{H: make([]float32, hiddenDim)}
}

// Reset zeros out the GRU state.
func (s *GRUState) Reset() {
	ops.Clear(s.H)
}

// Clone returns a deep copy of the GRU state.
func (s *GRUState) Clone() *GRUState {
	c := &GRUState{H: make([]float32, len(s.H))}
	copy(c.H, s.H)
	return c
}

// GRUCellF32 computes one GRU time step with float32 weight matrices.
//
//   r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)   (reset gate)
//   z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)   (update gate)
//   n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn)) (new gate)
//   h' = (1 - z) * n + z * h
//
//   wIH: [3*hiddenDim × inputDim]  (stacked r,z,n input weights)
//   wHH: [3*hiddenDim × hiddenDim] (stacked r,z,n hidden weights)
//   bIH: [3*hiddenDim]
//   bHH: [3*hiddenDim]
//   scratch: must be >= 6*hiddenDim
func GRUCellF32(x []float32, state *GRUState,
	wIH, wHH, bIH, bHH []float32,
	hiddenDim, inputDim int,
	scratch []float32) {

	gatesDim := 3 * hiddenDim
	gatesI := scratch[:gatesDim]
	gatesH := scratch[gatesDim : 2*gatesDim]

	ops.MatVecMul(gatesI, wIH, x, gatesDim, inputDim)
	ops.AddBias(gatesI, bIH)

	ops.MatVecMul(gatesH, wHH, state.H, gatesDim, hiddenDim)
	ops.AddBias(gatesH, bHH)

	for j := 0; j < hiddenDim; j++ {
		rGate := ops.Sigmoid(gatesI[j] + gatesH[j])
		zGate := ops.Sigmoid(gatesI[hiddenDim+j] + gatesH[hiddenDim+j])
		nGate := ops.FastTanh(gatesI[2*hiddenDim+j] + rGate*gatesH[2*hiddenDim+j])

		state.H[j] = (1-zGate)*nGate + zGate*state.H[j]
	}
}
