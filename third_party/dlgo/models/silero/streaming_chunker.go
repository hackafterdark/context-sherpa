package silero

// AudioChunk is a speech chunk ready to send to a streaming ASR pipeline.
// Sample indices are absolute within the current stream and use 16kHz mono PCM.
type AudioChunk struct {
	StartSample int
	EndSample   int
	Samples     []float32
}

// StreamingChunkerConfig controls conservative streaming chunk emission.
type StreamingChunkerConfig struct {
	VAD             VADParams
	StableSilenceMs int // extra silence after a segment end before emitting it
	MinEmitChunkMs  int // drop tiny chunks after segmentation (safety net)
}

// DefaultStreamingChunkerConfig favors quality over latency.
func DefaultStreamingChunkerConfig() StreamingChunkerConfig {
	p := DefaultVADParams()
	p.MinSpeechDurationMs = 300
	p.MinSilenceDurationMs = 350
	p.SpeechPadMs = 200
	return StreamingChunkerConfig{
		VAD:             p,
		StableSilenceMs: 700,
		MinEmitChunkMs:  400,
	}
}

// StreamingChunker incrementally runs Silero VAD and emits conservative chunks.
// It keeps all audio for the current stream in memory (suitable per-utterance/session).
type StreamingChunker struct {
	vad       *SileroVAD
	cfg       StreamingChunkerConfig
	window    int
	pending   []float32 // < window unprocessed tail
	audio     []float32 // full stream audio so far
	probs     []float32 // per-window speech probabilities
	emittedTo int       // absolute sample offset already emitted up to
}

// NewStreamingChunker loads a Silero model and creates a stateful chunker.
func NewStreamingChunker(modelPath string, cfg StreamingChunkerConfig) (*StreamingChunker, error) {
	vad, err := NewSileroVAD(modelPath)
	if err != nil {
		return nil, err
	}
	if cfg.StableSilenceMs <= 0 {
		cfg.StableSilenceMs = 700
	}
	return &StreamingChunker{
		vad:     vad,
		cfg:     cfg,
		window:  int(vad.Model.WindowSize),
		pending: make([]float32, 0, int(vad.Model.WindowSize)),
		audio:   make([]float32, 0, 16000*20),
		probs:   make([]float32, 0, 1024),
	}, nil
}

// Reset starts a fresh stream.
func (s *StreamingChunker) Reset() {
	if s == nil {
		return
	}
	s.vad.Reset()
	s.pending = s.pending[:0]
	s.audio = s.audio[:0]
	s.probs = s.probs[:0]
	s.emittedTo = 0
}

// AddSamples appends audio and returns newly stable speech chunks.
func (s *StreamingChunker) AddSamples(samples []float32) []AudioChunk {
	if len(samples) == 0 {
		return nil
	}

	s.audio = append(s.audio, samples...)
	s.pending = append(s.pending, samples...)

	for len(s.pending) >= s.window {
		chunk := s.pending[:s.window]
		s.probs = append(s.probs, s.vad.ProcessChunkStateful(chunk))
		s.pending = s.pending[s.window:]
	}

	return s.collect(false)
}

// Flush finalizes the current stream and emits any remaining chunk.
func (s *StreamingChunker) Flush() []AudioChunk {
	if len(s.pending) > 0 {
		chunk := make([]float32, s.window)
		copy(chunk, s.pending)
		s.probs = append(s.probs, s.vad.ProcessChunkStateful(chunk))
		s.pending = s.pending[:0]
	}
	return s.collect(true)
}

func (s *StreamingChunker) collect(final bool) []AudioChunk {
	if len(s.probs) == 0 {
		return nil
	}

	segments := DetectSegments(s.probs, s.window, s.cfg.VAD)
	if len(segments) == 0 {
		return nil
	}

	processedSamples := len(s.probs) * s.window
	stableCutoff := processedSamples
	if !final {
		stableCutoff -= 16000 * s.cfg.StableSilenceMs / 1000
		if stableCutoff < 0 {
			stableCutoff = 0
		}
	}

	minEmitSamples := 16000 * s.cfg.MinEmitChunkMs / 1000
	if minEmitSamples < 0 {
		minEmitSamples = 0
	}

	var out []AudioChunk
	for _, seg := range segments {
		if !final && seg.EndSample > stableCutoff {
			continue
		}
		if seg.EndSample <= s.emittedTo {
			continue
		}

		start := seg.StartSample
		if start < s.emittedTo {
			start = s.emittedTo
		}
		end := seg.EndSample
		if end > len(s.audio) {
			end = len(s.audio)
		}
		if end <= start {
			continue
		}
		if end-start < minEmitSamples {
			s.emittedTo = end
			continue
		}

		chunkSamples := make([]float32, end-start)
		copy(chunkSamples, s.audio[start:end])
		out = append(out, AudioChunk{
			StartSample: start,
			EndSample:   end,
			Samples:     chunkSamples,
		})
		s.emittedTo = end
	}

	return out
}
