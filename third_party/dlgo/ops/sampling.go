package ops

import (
	"math"
	"math/rand"
	"sort"
)

// SamplerConfig holds parameters for token sampling.
type SamplerConfig struct {
	Temperature       float32 // 0 = greedy, 0.7 = creative, 1.0 = original distribution
	TopK              int     // keep top K candidates (0 = disabled)
	TopP              float32 // nucleus sampling threshold (1.0 = disabled)
	MinP              float32 // minimum probability threshold relative to top token (0 = disabled)
	RepetitionPenalty float32 // penalize repeated tokens (1.0 = off, 1.1 = typical)
}

// DefaultSamplerConfig returns sensible defaults for text generation.
func DefaultSamplerConfig() SamplerConfig {
	return SamplerConfig{
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0,
		RepetitionPenalty: 1.1,
	}
}

// SampleToken applies the full sampling pipeline and returns a token index.
// Pipeline: repetition penalty → temperature → top-k → top-p → min-p → sample.
// Pass rng=nil for greedy (argmax) decoding regardless of temperature.
func SampleToken(logits []float32, cfg SamplerConfig, recentTokens []int32, rng *rand.Rand) int {
	if cfg.Temperature <= 0 || rng == nil {
		return Argmax(logits)
	}

	n := len(logits)
	work := make([]float32, n)
	copy(work, logits)

	ApplyRepetitionPenalty(work, recentTokens, cfg.RepetitionPenalty)
	ApplyTemperature(work, cfg.Temperature)
	ApplyTopK(work, cfg.TopK)
	ApplyTopP(work, cfg.TopP)
	ApplyMinP(work, cfg.MinP)

	return sampleFromLogits(work, rng)
}

// ApplyRepetitionPenalty penalizes tokens in recentTokens.
// Positive logits are divided by penalty; negative logits are multiplied.
// Matches llama.cpp behavior.
func ApplyRepetitionPenalty(logits []float32, recentTokens []int32, penalty float32) {
	if penalty <= 1.0 || len(recentTokens) == 0 {
		return
	}
	seen := make(map[int32]bool, len(recentTokens))
	for _, tok := range recentTokens {
		if tok < 0 || int(tok) >= len(logits) || seen[tok] {
			continue
		}
		seen[tok] = true
		if logits[tok] > 0 {
			logits[tok] /= penalty
		} else {
			logits[tok] *= penalty
		}
	}
}

// ApplyTemperature divides all logits by temperature. Higher temperature = more random.
func ApplyTemperature(logits []float32, temp float32) {
	if temp <= 0 || temp == 1.0 {
		return
	}
	invTemp := 1.0 / temp
	for i := range logits {
		logits[i] *= invTemp
	}
}

// ApplyTopK zeros out all logits below the K-th largest value.
func ApplyTopK(logits []float32, k int) {
	if k <= 0 || k >= len(logits) {
		return
	}
	indices := topKIdxSort(logits, k)
	cutoff := logits[indices[k-1]]
	negInf := float32(math.Inf(-1))
	for i := range logits {
		if logits[i] < cutoff {
			logits[i] = negInf
		}
	}
}

// ApplyTopP applies nucleus sampling: keeps the smallest set of tokens whose
// cumulative probability exceeds p. All other logits are set to -inf.
func ApplyTopP(logits []float32, p float32) {
	if p >= 1.0 {
		return
	}
	n := len(logits)
	probs := make([]float32, n)
	copy(probs, logits)
	Softmax(probs)

	type iv struct {
		idx  int
		prob float32
	}
	items := make([]iv, n)
	for i, v := range probs {
		items[i] = iv{i, v}
	}
	sort.Slice(items, func(a, b int) bool { return items[a].prob > items[b].prob })

	var cumSum float32
	cutoffIdx := n - 1
	for i, item := range items {
		cumSum += item.prob
		if cumSum >= p {
			cutoffIdx = i
			break
		}
	}

	allowed := make(map[int]bool, cutoffIdx+1)
	for i := 0; i <= cutoffIdx; i++ {
		allowed[items[i].idx] = true
	}
	negInf := float32(math.Inf(-1))
	for i := range logits {
		if !allowed[i] {
			logits[i] = negInf
		}
	}
}

// ApplyMinP filters tokens whose probability is below minP * max_probability.
// More adaptive than top-k: keeps more tokens when the distribution is uncertain,
// fewer when it's confident.
func ApplyMinP(logits []float32, minP float32) {
	if minP <= 0 {
		return
	}
	probs := make([]float32, len(logits))
	copy(probs, logits)
	Softmax(probs)

	maxProb := probs[0]
	for _, p := range probs[1:] {
		if p > maxProb {
			maxProb = p
		}
	}

	threshold := minP * maxProb
	negInf := float32(math.Inf(-1))
	for i := range logits {
		if probs[i] < threshold {
			logits[i] = negInf
		}
	}
}

func sampleFromLogits(logits []float32, rng *rand.Rand) int {
	probs := make([]float32, len(logits))
	copy(probs, logits)
	Softmax(probs)

	r := rng.Float32()
	var cumSum float32
	for i, p := range probs {
		cumSum += p
		if cumSum >= r {
			return i
		}
	}
	return len(probs) - 1
}

func topKIdxSort(vals []float32, k int) []int {
	type iv struct {
		idx int
		val float32
	}
	items := make([]iv, len(vals))
	for i, v := range vals {
		items[i] = iv{i, v}
	}
	sort.Slice(items, func(a, b int) bool { return items[a].val > items[b].val })
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = items[i].idx
	}
	return out
}
