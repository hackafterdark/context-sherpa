package silero

import "math"

// LSTMState holds the persistent hidden and cell state for the LSTM.
// Must be reset at the start of each new audio file.
type LSTMState struct {
	H []float32 // hidden state [hiddenSize]
	C []float32 // cell state   [hiddenSize]
}

// NewLSTMState creates a zero-initialized LSTM state.
func NewLSTMState(hiddenSize int) *LSTMState {
	return &LSTMState{
		H: make([]float32, hiddenSize),
		C: make([]float32, hiddenSize),
	}
}

// Reset zeros out both hidden and cell state.
func (s *LSTMState) Reset() {
	for i := range s.H {
		s.H[i] = 0
		s.C[i] = 0
	}
}

// lstmForward performs one LSTM time step.
// Input: x [128], modifies state in place, returns copy of new hidden state.
//
// Weight layout (flat, no transpose):
//   weightIH[g*inputSize + j]  maps input j to gate g
//   weightHH[g*hiddenSize + j] maps hidden j to gate g
//   g ∈ [0..511], split into 4 gates of 128 each:
//     [0..127]   = input gate  (i)
//     [128..255] = forget gate (f)
//     [256..383] = cell gate   (g) — tanh
//     [384..511] = output gate (o)
func lstmForward(x []float32, state *LSTMState, model *SileroModel) []float32 {
	hiddenSize := int(model.LSTMHiddenSize)
	inputSize := int(model.LSTMInputSize)
	gateSize := hiddenSize * 4 // 512

	// Compute all gates: gates = W_ih·x + b_ih + W_hh·h + b_hh
	gates := make([]float32, gateSize)
	for g := 0; g < gateSize; g++ {
		sum := model.LSTMBiasIH[g] + model.LSTMBiasHH[g]
		wihBase := g * inputSize
		whhBase := g * hiddenSize
		for j := 0; j < inputSize; j++ {
			sum += model.LSTMWeightIH[wihBase+j] * x[j]
		}
		for j := 0; j < hiddenSize; j++ {
			sum += model.LSTMWeightHH[whhBase+j] * state.H[j]
		}
		gates[g] = sum
	}

	// Apply activations and update state
	for h := 0; h < hiddenSize; h++ {
		iGate := sigmoidF(gates[h])              // input gate
		fGate := sigmoidF(gates[hiddenSize+h])   // forget gate
		gGate := tanhF(gates[2*hiddenSize+h])    // cell gate
		oGate := sigmoidF(gates[3*hiddenSize+h]) // output gate

		state.C[h] = fGate*state.C[h] + iGate*gGate
		state.H[h] = oGate * tanhF(state.C[h])
	}

	result := make([]float32, hiddenSize)
	copy(result, state.H)
	return result
}

func sigmoidF(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func tanhF(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}
