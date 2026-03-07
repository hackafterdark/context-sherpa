package whisper

import "github.com/computerex/dlgo/ops"

const (
	defaultSOT = 50257 // <|startoftranscript|>
	defaultEOT = 50256 // <|endoftext|>
)

// Transcribe runs the full Whisper pipeline: encode audio → greedy decode.
// mel: mel spectrogram features [nFrames × nMels].
// tok: tokenizer for decoding token IDs to text (if nil, returns empty string).
func (m *WhisperModel) Transcribe(mel []float32, tok *Tokenizer) (string, error) {
	encOut := EncodeAudio(m, mel)
	if encOut == nil {
		return "", nil
	}

	cfg := m.Config
	dim := cfg.DModel
	encLen := len(encOut) / dim

	kc := NewKVCache(cfg)
	kc.FillCross(encOut, encLen, dim, m.DecLayers)

	// Build start-of-transcript prompt
	var prompt []int
	if tok != nil {
		prompt = tok.SotSequence()
		noTs := tok.NoTimestamps()
		if noTs >= 0 {
			prompt = append(prompt, noTs)
		}
	} else {
		prompt = []int{defaultSOT}
	}

	// Prefill the prompt tokens
	var logits []float32
	for i, t := range prompt {
		logits = DecoderStep(m, encOut, encLen, int32(t), i, kc)
	}
	pos := len(prompt)

	// Greedy decode until EOT or max length
	eot := defaultEOT
	if tok != nil {
		eot = tok.Eot()
	}

	maxTokens := cfg.NTextCtx - len(prompt)
	var decoded []int
	for i := 0; i < maxTokens; i++ {
		next := int32(ops.Argmax(logits))
		if int(next) == eot {
			break
		}
		decoded = append(decoded, int(next))
		logits = DecoderStep(m, encOut, encLen, next, pos, kc)
		pos++
	}

	if tok != nil {
		return tok.Decode(decoded), nil
	}
	return "", nil
}

// TranscribeFile loads a WAV, extracts mel features, and transcribes.
func (m *WhisperModel) TranscribeFile(wavPath string, tok *Tokenizer) (string, error) {
	samples, err := LoadWAV(wavPath)
	if err != nil {
		return "", err
	}
	mel := ExtractMel(samples, m.Config.NMels)
	return m.Transcribe(mel, tok)
}
