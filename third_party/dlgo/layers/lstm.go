package layers

import (
	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/ops"
)

// LSTMState holds the hidden and cell states for an LSTM layer.
type LSTMState struct {
	H []float32 // hidden state [hiddenDim]
	C []float32 // cell state [hiddenDim]
}

// NewLSTMState creates a zero-initialized LSTM state.
func NewLSTMState(hiddenDim int) *LSTMState {
	return &LSTMState{
		H: make([]float32, hiddenDim),
		C: make([]float32, hiddenDim),
	}
}

// Reset zeros out the LSTM state.
func (s *LSTMState) Reset() {
	ops.Clear(s.H)
	ops.Clear(s.C)
}

// Clone returns a deep copy of the LSTM state.
func (s *LSTMState) Clone() *LSTMState {
	c := &LSTMState{
		H: make([]float32, len(s.H)),
		C: make([]float32, len(s.C)),
	}
	copy(c.H, s.H)
	copy(c.C, s.C)
	return c
}

// LSTMCell computes one LSTM time step with quantized weight matrices.
//
// Gate order follows ONNX convention: input, output, forget, cell (i, o, f, c).
//
//   x:     input vector [inputDim]
//   state: previous hidden/cell states (modified in-place)
//   wIH:   [4*hiddenDim × inputDim]  input-to-hidden weights
//   wHH:   [4*hiddenDim × hiddenDim] hidden-to-hidden weights
//   bIH:   [4*hiddenDim]
//   bHH:   [4*hiddenDim]
//   scratch: temporary buffer, must be >= 8*hiddenDim
func LSTMCell(x []float32, state *LSTMState,
	wIH, wHH *core.QuantizedTensor,
	bIH, bHH []float32,
	scratch []float32) {

	hiddenDim := len(state.H)
	gatesDim := 4 * hiddenDim

	gates := scratch[:gatesDim]
	temp := scratch[gatesDim : 2*gatesDim]

	blas.QMatVecMul(gates, wIH, x)
	ops.AddBias(gates, bIH)

	blas.QMatVecMul(temp, wHH, state.H)
	ops.Add(gates, gates, temp)
	ops.AddBias(gates, bHH)

	for j := 0; j < hiddenDim; j++ {
		iGate := ops.Sigmoid(gates[j])
		oGate := ops.Sigmoid(gates[hiddenDim+j])
		fGate := ops.Sigmoid(gates[2*hiddenDim+j])
		cGate := ops.FastTanh(gates[3*hiddenDim+j])

		state.C[j] = fGate*state.C[j] + iGate*cGate
		state.H[j] = oGate * ops.FastTanh(state.C[j])
	}
}

// LSTMCellF32 computes one LSTM time step with float32 weight matrices.
// Same gate ordering as LSTMCell (ONNX: i, o, f, c).
//
//   wIH: [4*hiddenDim * inputDim]  row-major
//   wHH: [4*hiddenDim * hiddenDim] row-major
func LSTMCellF32(x []float32, state *LSTMState,
	wIH, wHH, bIH, bHH []float32,
	hiddenDim, inputDim int,
	scratch []float32) {

	gatesDim := 4 * hiddenDim
	gates := scratch[:gatesDim]
	temp := scratch[gatesDim : 2*gatesDim]

	ops.MatVecMul(gates, wIH, x, gatesDim, inputDim)
	ops.AddBias(gates, bIH)

	ops.MatVecMul(temp, wHH, state.H, gatesDim, hiddenDim)
	ops.Add(gates, gates, temp)
	ops.AddBias(gates, bHH)

	for j := 0; j < hiddenDim; j++ {
		iGate := ops.Sigmoid(gates[j])
		oGate := ops.Sigmoid(gates[hiddenDim+j])
		fGate := ops.Sigmoid(gates[2*hiddenDim+j])
		cGate := ops.FastTanh(gates[3*hiddenDim+j])

		state.C[j] = fGate*state.C[j] + iGate*cGate
		state.H[j] = oGate * ops.FastTanh(state.C[j])
	}
}
