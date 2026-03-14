package silero

import "math"

// SileroVAD is the main voice activity detector.
type SileroVAD struct {
	Model *SileroModel
	State *LSTMState
}

// NewSileroVAD loads the GGML model and creates a ready-to-use VAD.
func NewSileroVAD(modelPath string) (*SileroVAD, error) {
	model, err := LoadModel(modelPath)
	if err != nil {
		return nil, err
	}
	return &SileroVAD{
		Model: model,
		State: NewLSTMState(int(model.LSTMHiddenSize)),
	}, nil
}

// ProcessAudio processes an entire audio signal (16kHz mono float32)
// and returns per-chunk speech probabilities.
// Each probability corresponds to one WindowSize-sample chunk (32ms at 16kHz).
func (v *SileroVAD) ProcessAudio(audio []float32) []float32 {
	windowSize := int(v.Model.WindowSize)

	// Reset LSTM state for new audio
	v.State.Reset()

	// Number of chunks (last chunk zero-padded if needed)
	nChunks := len(audio) / windowSize
	if len(audio)%windowSize != 0 {
		nChunks++
	}

	probs := make([]float32, nChunks)

	chunk := make([]float32, windowSize)
	for i := 0; i < nChunks; i++ {
		start := i * windowSize
		end := start + windowSize

		// Clear chunk (zero-pad)
		for j := range chunk {
			chunk[j] = 0
		}

		if end <= len(audio) {
			copy(chunk, audio[start:end])
		} else {
			copy(chunk, audio[start:])
		}

		probs[i] = v.forwardChunk(chunk)
	}

	return probs
}

// forwardChunk runs the full forward pass for one 512-sample audio chunk.
// Returns speech probability ∈ [0, 1].
// ProcessChunkStateful runs VAD on one chunk and preserves LSTM state across calls.
func (v *SileroVAD) ProcessChunkStateful(chunk []float32) float32 {
	return v.forwardChunk(chunk)
}

// Reset clears internal recurrent state before a new independent audio stream.
func (v *SileroVAD) Reset() {
	if v == nil || v.State == nil {
		return
	}
	v.State.Reset()
}

func (v *SileroVAD) forwardChunk(chunk []float32) float32 {
	// 1. STFT: raw audio → magnitude spectrogram [129][4]
	magnitude := applySTFT(chunk, v.Model.STFTBasis)

	// 2. Encoder: magnitude → feature vector [128]
	features := encode(magnitude, v.Model)

	// 3. LSTM: feature vector → hidden state [128] (stateful)
	hidden := lstmForward(features, v.State, v.Model)

	// 4. ReLU
	for i := range hidden {
		if hidden[i] < 0 {
			hidden[i] = 0
		}
	}

	// 5. Final linear projection (Conv1D with time=1 is just a dot product)
	// FinalConvWeight: [128], FinalConvBias: [1]
	var sum float32
	for j := 0; j < int(v.Model.FinalConvIn); j++ {
		sum += v.Model.FinalConvWeight[j] * hidden[j]
	}
	sum += v.Model.FinalConvBias[0]

	// 6. Sigmoid
	prob := float32(1.0 / (1.0 + math.Exp(-float64(sum))))
	return prob
}

// ProcessChunk runs a single forward pass for one 512-sample audio chunk.
// This is a convenience function that creates a temporary VAD instance.
func ProcessChunk(model *SileroModel, chunk []float32) float32 {
	// Create temporary state
	state := NewLSTMState(int(model.LSTMHiddenSize))

	// Run forward pass
	vad := &SileroVAD{
		Model: model,
		State: state,
	}

	return vad.forwardChunk(chunk)
}
