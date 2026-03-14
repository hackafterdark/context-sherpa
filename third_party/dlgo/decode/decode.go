// Package decode provides decoding algorithms for sequence-to-sequence and
// transducer models.
//
// Includes greedy decoding, beam search, and utility functions like top-k selection
// and length normalization used across different decoding strategies.
package decode

import (
	"container/heap"
	"math"

	"github.com/computerex/dlgo/ops"
)

// GreedyDecode performs simple greedy argmax decoding.
// logitsFunc is called for each step with the current token and position,
// returning logits over the vocabulary. Decoding stops when eotToken is emitted
// or maxSteps is reached.
func GreedyDecode(firstToken, eotToken, maxSteps int, logitsFunc func(tokenID, pos int) []float32) []int {
	var tokens []int
	currentToken := firstToken

	for step := 0; step < maxSteps; step++ {
		logits := logitsFunc(currentToken, step)
		nextToken := ops.Argmax(logits)
		if nextToken == eotToken {
			break
		}
		tokens = append(tokens, nextToken)
		currentToken = nextToken
	}
	return tokens
}

// BeamHypothesis represents one candidate sequence during beam search.
type BeamHypothesis struct {
	Tokens   []int
	Score    float32
	Finished bool
	State    interface{} // opaque decoder state for the hypothesis
}

// BeamConfig controls beam search behavior.
type BeamConfig struct {
	BeamSize    int
	MaxSteps    int
	EOTToken    int
	LengthAlpha float32 // length normalization exponent (0 = disabled)
}

// LengthNormalization computes Google NMT length penalty:
//   penalty = ((5 + seqLen) / 6) ^ alpha
func LengthNormalization(seqLen int, alpha float32) float32 {
	if alpha == 0 {
		return 1.0
	}
	return float32(math.Pow(float64(5+seqLen)/6.0, float64(alpha)))
}

// TopKIndicesHeap returns the indices of the k largest values using a min-heap.
// O(n log k) complexity.
func TopKIndicesHeap(vals []float32, k int) []int {
	if k >= len(vals) {
		k = len(vals)
	}
	h := &indexedValHeap{}
	heap.Init(h)
	for i, v := range vals {
		if h.Len() < k {
			heap.Push(h, indexedVal{idx: i, val: v})
		} else if v > (*h)[0].val {
			(*h)[0] = indexedVal{idx: i, val: v}
			heap.Fix(h, 0)
		}
	}
	result := make([]int, h.Len())
	for i := h.Len() - 1; i >= 0; i-- {
		result[i] = heap.Pop(h).(indexedVal).idx
	}
	return result
}

type indexedVal struct {
	idx int
	val float32
}

type indexedValHeap []indexedVal

func (h indexedValHeap) Len() int            { return len(h) }
func (h indexedValHeap) Less(i, j int) bool  { return h[i].val < h[j].val }
func (h indexedValHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *indexedValHeap) Push(x interface{}) { *h = append(*h, x.(indexedVal)) }
func (h *indexedValHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// BeamSearch performs generic beam search decoding.
//
//   initState:  initial decoder state
//   expandFunc: given a hypothesis, returns (logits, newState) for each possible next step
//   cfg:        beam search configuration
//
// The expandFunc receives the current hypothesis and should return the log-probability
// distribution over the next token and an updated state.
func BeamSearch(
	initState interface{},
	expandFunc func(hyp *BeamHypothesis) (logProbs []float32, newState interface{}),
	cfg BeamConfig,
) []int {
	beams := []BeamHypothesis{{
		Tokens: nil,
		Score:  0,
		State:  initState,
	}}

	var finished []BeamHypothesis

	for step := 0; step < cfg.MaxSteps; step++ {
		type candidate struct {
			parentIdx int
			tokenID   int
			score     float32
			state     interface{}
		}
		var candidates []candidate

		for bi, beam := range beams {
			if beam.Finished {
				finished = append(finished, beam)
				continue
			}

			logProbs, newState := expandFunc(&beam)
			topK := TopKIndicesHeap(logProbs, cfg.BeamSize)

			for _, tokID := range topK {
				candidates = append(candidates, candidate{
					parentIdx: bi,
					tokenID:   tokID,
					score:     beam.Score + logProbs[tokID],
					state:     newState,
				})
			}
		}

		if len(candidates) == 0 {
			break
		}

		// Select top candidates
		scores := make([]float32, len(candidates))
		for i, c := range candidates {
			scores[i] = c.score
		}
		topIdx := TopKIndicesHeap(scores, cfg.BeamSize)

		beams = beams[:0]
		for _, ci := range topIdx {
			c := candidates[ci]
			newTokens := make([]int, len(beams)+1)
			if ci < len(candidates) {
				parent := candidates[c.parentIdx]
				_ = parent
			}
			// Build token sequence from parent
			parentBeam := beams
			_ = parentBeam

			newTokens = append(append([]int(nil), beams[0].Tokens...), c.tokenID)
			if len(beams) > c.parentIdx {
				newTokens = append(append([]int(nil), beams[c.parentIdx].Tokens...), c.tokenID)
			}

			hyp := BeamHypothesis{
				Tokens:   newTokens,
				Score:    c.score,
				Finished: c.tokenID == cfg.EOTToken,
				State:    c.state,
			}

			if hyp.Finished {
				finished = append(finished, hyp)
			} else {
				beams = append(beams, hyp)
			}
		}

		if len(beams) == 0 {
			break
		}
	}

	for _, b := range beams {
		finished = append(finished, b)
	}

	if len(finished) == 0 {
		return nil
	}

	bestIdx := 0
	bestScore := float32(math.Inf(-1))
	for i, b := range finished {
		norm := b.Score / LengthNormalization(len(b.Tokens), cfg.LengthAlpha)
		if norm > bestScore {
			bestScore = norm
			bestIdx = i
		}
	}
	return finished[bestIdx].Tokens
}
