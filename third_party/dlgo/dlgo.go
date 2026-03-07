// Package dlgo provides a high-level API for deep learning inference in Go.
//
// Quick start for text generation:
//
//	model, _ := dlgo.LoadLLM("path/to/model.gguf")
//	text, _ := model.Chat("You are helpful.", "What is Go?")
//	fmt.Println(text)
//
// Quick start for speech-to-text:
//
//	whisper, _ := dlgo.LoadWhisper("path/to/whisper.gguf")
//	text, _ := whisper.TranscribeFile("audio.wav")
//	fmt.Println(text)
//
// Quick start for voice activity detection:
//
//	vad, _ := dlgo.LoadVAD("path/to/silero.onnx")
//	segments := vad.DetectFile("audio.wav")
package dlgo

import (
	"fmt"
	"strings"
	"time"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/models/whisper"
)

// LLM wraps a loaded language model with a simple API for text generation.
type LLM struct {
	pipeline *llm.Pipeline
}

// LoadLLM loads a GGUF language model and returns a ready-to-use LLM.
// maxSeqLen controls the maximum context window (0 = use model default).
func LoadLLM(modelPath string, maxSeqLen ...int) (*LLM, error) {
	seqLen := 0
	if len(maxSeqLen) > 0 {
		seqLen = maxSeqLen[0]
	}
	p, err := llm.NewPipeline(modelPath, seqLen)
	if err != nil {
		return nil, fmt.Errorf("load LLM: %w", err)
	}
	return &LLM{pipeline: p}, nil
}

// Generate produces text from a raw prompt string.
// Returns the generated text.
func (m *LLM) Generate(prompt string, opts ...Option) (string, error) {
	cfg := applyOpts(opts)
	text, _, err := m.pipeline.GenerateTextWithStopStrings(prompt, cfg)
	return text, err
}

// Chat formats a single-turn chat message and generates a response.
// System prompt is optional (pass "" to skip).
func (m *LLM) Chat(system, user string, opts ...Option) (string, error) {
	prompt := llm.FormatChat(m.pipeline.Model.Config, system, user)
	cfg := applyOpts(opts)
	text, _, err := m.pipeline.GenerateTextWithStopStrings(prompt, cfg)
	return text, err
}

// ChatMessages formats a multi-turn conversation and generates the next response.
func (m *LLM) ChatMessages(messages []Message, opts ...Option) (string, error) {
	llmMsgs := make([]llm.Message, len(messages))
	for i, msg := range messages {
		llmMsgs[i] = llm.Message{Role: msg.Role, Content: msg.Content}
	}
	prompt := llm.FormatMessages(m.pipeline.Model.Config, llmMsgs)
	cfg := applyOpts(opts)
	text, _, err := m.pipeline.GenerateTextWithStopStrings(prompt, cfg)
	return text, err
}

// GenerateStream produces text token-by-token, calling onToken for each.
func (m *LLM) GenerateStream(prompt string, onToken func(string), opts ...Option) error {
	cfg := applyOpts(opts)
	cfg.Stream = onToken
	_, _, err := m.pipeline.GenerateTextWithStopStrings(prompt, cfg)
	return err
}

// ChatStream generates a chat response, streaming tokens to onToken.
func (m *LLM) ChatStream(system, user string, onToken func(string), opts ...Option) error {
	prompt := llm.FormatChat(m.pipeline.Model.Config, system, user)
	return m.GenerateStream(prompt, onToken, opts...)
}

// Benchmark runs a generation benchmark and returns detailed results.
func (m *LLM) Benchmark(prompt string, opts ...Option) (*BenchmarkResult, error) {
	cfg := applyOpts(opts)

	start := time.Now()
	result, err := m.pipeline.GenerateDetailed(prompt, cfg)
	totalMs := float64(time.Since(start).Microseconds()) / 1000.0

	if err != nil {
		return nil, err
	}

	return &BenchmarkResult{
		Text:          result.Text,
		TokensPerSec:  result.TokensPerSec,
		PrefillMs:     result.PrefillTimeMs,
		GenerateMs:    result.GenerateTimeMs,
		TotalMs:       totalMs,
		PromptTokens:  result.PromptTokens,
		OutputTokens:  result.TotalTokens,
		Architecture:  m.pipeline.Model.Config.Architecture,
	}, nil
}

// ModelInfo returns metadata about the loaded model.
func (m *LLM) ModelInfo() ModelInfo {
	cfg := m.pipeline.Model.Config
	return ModelInfo{
		Architecture: cfg.Architecture,
		Layers:       cfg.NumLayers,
		Dimension:    cfg.EmbeddingDim,
		Heads:        cfg.NumHeads,
		KVHeads:      cfg.NumKVHeads,
		FFNDim:       cfg.FFNDim,
		VocabSize:    cfg.VocabSize,
		ContextLen:   cfg.ContextLength,
		ChatTemplate: cfg.ChatTemplate,
	}
}

// Whisper wraps a loaded Whisper model for speech-to-text.
type Whisper struct {
	model     *whisper.WhisperModel
	tokenizer *whisper.Tokenizer
}

// LoadWhisper loads a Whisper GGUF model for speech recognition.
// tokenizerPath is the path to a HuggingFace tokenizer.json file or directory.
func LoadWhisper(modelPath, tokenizerPath string) (*Whisper, error) {
	m, err := whisper.LoadWhisperModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load Whisper: %w", err)
	}
	tok, err := whisper.LoadTokenizer(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	return &Whisper{model: m, tokenizer: tok}, nil
}

// TranscribeFile transcribes a WAV audio file to text.
func (w *Whisper) TranscribeFile(wavPath string) (string, error) {
	return w.model.TranscribeFile(wavPath, w.tokenizer)
}

// TranscribeSamples transcribes raw 16kHz mono audio samples to text.
func (w *Whisper) TranscribeSamples(samples []float32) (string, error) {
	mel := whisper.ExtractMel(samples, w.model.Config.NMels)
	return w.model.Transcribe(mel, w.tokenizer)
}

// Message represents a chat message with a role and content.
type Message struct {
	Role    string // "system", "user", or "assistant"
	Content string
}

// ModelInfo holds metadata about a loaded model.
type ModelInfo struct {
	Architecture string
	Layers       int
	Dimension    int
	Heads        int
	KVHeads      int
	FFNDim       int
	VocabSize    int
	ContextLen   int
	ChatTemplate string
}

func (m ModelInfo) String() string {
	return fmt.Sprintf("%s (%d layers, %d dim, %d heads, vocab %d, ctx %d)",
		m.Architecture, m.Layers, m.Dimension, m.Heads, m.VocabSize, m.ContextLen)
}

// BenchmarkResult holds performance metrics from a generation run.
type BenchmarkResult struct {
	Text         string
	TokensPerSec float64
	PrefillMs    float64
	GenerateMs   float64
	TotalMs      float64
	PromptTokens int
	OutputTokens int
	Architecture string
}

func (r *BenchmarkResult) String() string {
	preview := r.Text
	if len(preview) > 80 {
		preview = preview[:80] + "..."
	}
	preview = strings.ReplaceAll(preview, "\n", " ")
	return fmt.Sprintf("%.1f tok/s | prefill: %.0fms | gen: %.0fms | %d tokens | %s",
		r.TokensPerSec, r.PrefillMs, r.GenerateMs, r.OutputTokens, preview)
}

// Option configures generation behavior.
type Option func(*llm.GenerateConfig)

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) Option {
	return func(c *llm.GenerateConfig) { c.MaxTokens = n }
}

// WithTemperature sets the sampling temperature (0 = greedy, 0.7 = creative).
func WithTemperature(t float32) Option {
	return func(c *llm.GenerateConfig) { c.Sampler.Temperature = t }
}

// WithTopK sets the top-K sampling parameter.
func WithTopK(k int) Option {
	return func(c *llm.GenerateConfig) { c.Sampler.TopK = k }
}

// WithTopP sets the nucleus sampling threshold.
func WithTopP(p float32) Option {
	return func(c *llm.GenerateConfig) { c.Sampler.TopP = p }
}

// WithSeed sets the random seed for reproducible generation.
func WithSeed(seed int64) Option {
	return func(c *llm.GenerateConfig) { c.Seed = seed }
}

// WithGreedy enables greedy (argmax) decoding.
func WithGreedy() Option {
	return func(c *llm.GenerateConfig) { c.Sampler.Temperature = 0 }
}

func applyOpts(opts []Option) llm.GenerateConfig {
	cfg := llm.DefaultGenerateConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}
