package llm

import (
	"container/heap"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Tokenizer encodes text to token IDs and decodes token IDs back to text.
// Supports both SentencePiece (LLaMA) and GPT-2 BPE (Qwen) tokenization.
type Tokenizer struct {
	// Backward-compatible fields
	Tokens    []string
	TokenToID map[string]int32
	BOS       int32
	EOS       int32
	AddBOS    bool

	// BPE/SPM-specific fields
	MergeRanks    map[[2]string]int // GPT-2: merge pair -> priority (lower = merge first)
	Scores        []float32         // SentencePiece: token scores for merge ordering
	ModelType     string            // "llama" or "gpt2"
	SpecialTokens map[string]int32  // special/control token text -> id
	unicodeToByte map[rune]byte     // GPT-2: unicode char -> byte (for decode)
	byteToUnicode [256]rune         // GPT-2: byte -> unicode char (for encode)

	// SPM: vocab entry for score lookup
	vocabMap map[string]spmVocabEntry
}

type spmVocabEntry struct {
	id    int32
	score float32
}

// NewTokenizerFromGGUF extracts vocabulary and tokenizer config from GGUF metadata.
// Auto-detects tokenizer type based on tokenizer.ggml.model: "llama" -> SentencePiece,
// "gpt2" -> GPT-2 BPE. Falls back to heuristic if model key is missing.
func NewTokenizerFromGGUF(md map[string]interface{}, cfg ModelConfig) (*Tokenizer, error) {
	model := metaString(md, "tokenizer.ggml.model")
	if model == "" {
		if _, hasMerges := md["tokenizer.ggml.merges"]; hasMerges {
			model = "gpt2"
		} else if _, hasScores := md["tokenizer.ggml.scores"]; hasScores {
			model = "llama"
		} else {
			return nil, fmt.Errorf("tokenizer.ggml.model not found and cannot infer from metadata")
		}
	}

	tokensArr, ok := md["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens not found in GGUF metadata")
	}
	arr, ok := tokensArr.([]interface{})
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens is not an array")
	}

	tokens := make([]string, len(arr))
	tokenToID := make(map[string]int32, len(arr))
	for i, v := range arr {
		s, ok := v.(string)
		if !ok {
			s = fmt.Sprintf("<token_%d>", i)
		}
		tokens[i] = s
		tokenToID[s] = int32(i)
	}

	t := &Tokenizer{
		Tokens:        tokens,
		TokenToID:     tokenToID,
		BOS:           cfg.BOS,
		EOS:           cfg.EOS,
		AddBOS:        cfg.AddBOS,
		ModelType:     model,
		SpecialTokens: make(map[string]int32),
	}

	switch model {
	case "llama":
		return initSPMTokenizer(t, md)
	case "gpt2":
		return initBPETokenizer(t, md)
	default:
		return nil, fmt.Errorf("unsupported tokenizer model: %s", model)
	}
}

func initSPMTokenizer(t *Tokenizer, md map[string]interface{}) (*Tokenizer, error) {
	scoresRaw, ok := md["tokenizer.ggml.scores"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.scores not found (required for SentencePiece)")
	}
	t.Scores = make([]float32, len(scoresRaw))
	for i, s := range scoresRaw {
		t.Scores[i] = toFloat32(s)
	}
	if len(t.Scores) != len(t.Tokens) {
		return nil, fmt.Errorf("tokenizer.ggml.scores length %d != tokens length %d", len(t.Scores), len(t.Tokens))
	}

	t.vocabMap = make(map[string]spmVocabEntry)
	for i, token := range t.Tokens {
		t.vocabMap[token] = spmVocabEntry{id: int32(i), score: t.Scores[i]}
	}

	if tokenTypes, ok := md["tokenizer.ggml.token_type"].([]interface{}); ok {
		for i, tt := range tokenTypes {
			tokenType := 0
			if n, ok := toInt(tt); ok {
				tokenType = n
			}
			if tokenType == 3 && i < len(t.Tokens) {
				text := t.Tokens[i]
				if len(text) > 1 && text != "<unk>" {
					t.SpecialTokens[text] = int32(i)
				}
			}
		}
	}

	return t, nil
}

func initBPETokenizer(t *Tokenizer, md map[string]interface{}) (*Tokenizer, error) {
	t.MergeRanks = make(map[[2]string]int)
	if mergesRaw, ok := md["tokenizer.ggml.merges"].([]interface{}); ok {
		for i, m := range mergesRaw {
			mergeStr, _ := m.(string)
			parts := strings.SplitN(mergeStr, " ", 2)
			if len(parts) == 2 {
				t.MergeRanks[[2]string{parts[0], parts[1]}] = i
			}
		}
	}

	if v, ok := md["tokenizer.ggml.bos_token_id"]; ok {
		if n, ok := toInt(v); ok {
			t.BOS = int32(n)
		}
	}
	if v, ok := md["tokenizer.ggml.eos_token_id"]; ok {
		if n, ok := toInt(v); ok {
			t.EOS = int32(n)
		}
	}
	if v, ok := md["tokenizer.ggml.add_bos_token"]; ok {
		if b, ok := v.(bool); ok {
			t.AddBOS = b
		}
	}

	var specialTokens []string
	for _, s := range t.Tokens {
		if len(s) > 2 && strings.HasPrefix(s, "<") && strings.HasSuffix(s, ">") {
			if strings.Contains(s, "|") || strings.HasPrefix(s, "</") ||
				strings.HasPrefix(s, "<tool") || s == "<think>" {
				specialTokens = append(specialTokens, s)
				t.SpecialTokens[s] = t.TokenToID[s]
			}
		}
	}
	sort.Slice(specialTokens, func(i, j int) bool {
		return len(specialTokens[i]) > len(specialTokens[j])
	})

	t.unicodeToByte, t.byteToUnicode = buildGPT2ByteMap()
	return t, nil
}

func toFloat32(v interface{}) float32 {
	switch x := v.(type) {
	case float32:
		return x
	case float64:
		return float32(x)
	default:
		return 0
	}
}

// buildGPT2ByteMap creates the GPT-2 bytes_to_unicode mapping.
func buildGPT2ByteMap() (map[rune]byte, [256]rune) {
	byteToUnicode := make(map[byte]rune)
	for b := byte(33); b <= 126; b++ {
		byteToUnicode[b] = rune(b)
	}
	for b := 161; b <= 172; b++ {
		byteToUnicode[byte(b)] = rune(b)
	}
	for b := 174; b <= 255; b++ {
		byteToUnicode[byte(b)] = rune(b)
	}
	n := 0
	for b := 0; b < 256; b++ {
		if _, exists := byteToUnicode[byte(b)]; !exists {
			byteToUnicode[byte(b)] = rune(256 + n)
			n++
		}
	}
	reverse := make(map[rune]byte, 256)
	var forward [256]rune
	for b, u := range byteToUnicode {
		reverse[u] = b
		forward[b] = u
	}
	return reverse, forward
}

// Encode converts text to a sequence of token IDs.
func (t *Tokenizer) Encode(text string) []int32 {
	var result []int32
	if t.AddBOS && t.BOS >= 0 {
		result = append(result, t.BOS)
	}

	switch t.ModelType {
	case "gpt2":
		return t.encodeBPE(text, result)
	case "llama":
		return t.encodeSPM(text, result)
	default:
		return t.encodeFallback(text, result)
	}
}

func (t *Tokenizer) encodeBPE(text string, result []int32) []int32 {
	if len(text) == 0 {
		return result
	}

	segments := t.splitOnSpecialTokensBPE(text)
	for _, seg := range segments {
		if id, ok := t.TokenToID[seg]; ok {
			result = append(result, id)
			continue
		}

		symbols := t.textToInitialTokensBPE(seg)
		symbols = t.applyBPE(symbols)

		for _, sym := range symbols {
			if id, ok := t.TokenToID[sym]; ok {
				result = append(result, id)
			} else {
				for _, b := range []byte(sym) {
					byteTok := fmt.Sprintf("<0x%02X>", b)
					if id, ok := t.TokenToID[byteTok]; ok {
						result = append(result, id)
					}
				}
			}
		}
	}
	return result
}

func (t *Tokenizer) splitOnSpecialTokensBPE(text string) []string {
	if len(t.SpecialTokens) == 0 {
		return []string{text}
	}

	var specialList []string
	for s := range t.SpecialTokens {
		specialList = append(specialList, s)
	}
	sort.Slice(specialList, func(i, j int) bool {
		return len(specialList[i]) > len(specialList[j])
	})

	var segments []string
	for len(text) > 0 {
		matched := false
		for _, st := range specialList {
			if strings.HasPrefix(text, st) {
				segments = append(segments, st)
				text = text[len(st):]
				matched = true
				break
			}
		}
		if !matched {
			nextIdx := len(text)
			for _, st := range specialList {
				if idx := strings.Index(text, st); idx >= 0 && idx < nextIdx {
					nextIdx = idx
				}
			}
			segments = append(segments, text[:nextIdx])
			text = text[nextIdx:]
		}
	}
	return segments
}

func (t *Tokenizer) textToInitialTokensBPE(text string) []string {
	var tokens []string
	for _, b := range []byte(text) {
		tokens = append(tokens, string(t.byteToUnicode[b]))
	}
	return tokens
}

func (t *Tokenizer) applyBPE(symbols []string) []string {
	for {
		if len(symbols) < 2 {
			break
		}
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			pair := [2]string{symbols[i], symbols[i+1]}
			if rank, ok := t.MergeRanks[pair]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}
		if bestIdx == -1 {
			break
		}
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		if bestIdx+2 < len(symbols) {
			newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		}
		symbols = newSymbols
	}
	return symbols
}

func (t *Tokenizer) encodeSPM(text string, result []int32) []int32 {
	segments := t.splitControlTokensSPM(text)
	for _, seg := range segments {
		if seg.isControl {
			result = append(result, seg.tokenID)
		} else if len(seg.text) > 0 {
			result = append(result, t.encodeSPMSegment(seg.text)...)
		}
	}
	return result
}

type textSegment struct {
	text      string
	isControl bool
	tokenID   int32
}

func (t *Tokenizer) splitControlTokensSPM(text string) []textSegment {
	if len(t.SpecialTokens) == 0 {
		return []textSegment{{text: text}}
	}

	var segments []textSegment
	remaining := text

	for len(remaining) > 0 {
		bestPos := len(remaining)
		bestToken := ""
		bestID := int32(-1)

		for ctrlText, ctrlID := range t.SpecialTokens {
			pos := strings.Index(remaining, ctrlText)
			if pos >= 0 && pos < bestPos {
				bestPos = pos
				bestToken = ctrlText
				bestID = ctrlID
			}
		}

		if bestID < 0 {
			segments = append(segments, textSegment{text: remaining})
			break
		}

		if bestPos > 0 {
			segments = append(segments, textSegment{text: remaining[:bestPos]})
		}
		segments = append(segments, textSegment{isControl: true, tokenID: bestID})
		remaining = remaining[bestPos+len(bestToken):]
	}

	return segments
}

type spmSymbol struct {
	text string
	prev int
	next int
	n    int
}

type spmBigram struct {
	left  int
	right int
	score float32
	size  int
}

type bigramHeap []spmBigram

func (h bigramHeap) Len() int { return len(h) }
func (h bigramHeap) Less(i, j int) bool {
	if h[i].score != h[j].score {
		return h[i].score > h[j].score
	}
	return h[i].left > h[j].left
}
func (h bigramHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *bigramHeap) Push(x interface{}) { *h = append(*h, x.(spmBigram)) }
func (h *bigramHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func (t *Tokenizer) encodeSPMSegment(text string) []int32 {
	processedText := strings.ReplaceAll(text, " ", "▁")
	if len(processedText) > 0 && !strings.HasPrefix(processedText, "▁") {
		processedText = "▁" + processedText
	}

	var symbols []spmSymbol
	for len(processedText) > 0 {
		r, size := utf8.DecodeRuneInString(processedText)
		if r == utf8.RuneError {
			break
		}
		prev := -1
		if len(symbols) > 0 {
			prev = len(symbols) - 1
		}
		next := len(symbols) + 1
		symbols = append(symbols, spmSymbol{
			text: processedText[:size],
			prev: prev,
			next: next,
			n:    size,
		})
		processedText = processedText[size:]
	}

	if len(symbols) == 0 {
		return []int32{}
	}

	symbols[0].prev = -1
	symbols[len(symbols)-1].next = -1

	h := &bigramHeap{}
	heap.Init(h)

	for i := 0; i < len(symbols)-1; i++ {
		if symbols[i].n == 0 {
			continue
		}
		combined := symbols[i].text + symbols[i+1].text
		if entry, ok := t.vocabMap[combined]; ok {
			heap.Push(h, spmBigram{
				left:  i,
				right: i + 1,
				score: entry.score,
				size:  len(combined),
			})
		}
	}

	for h.Len() > 0 {
		bigram := heap.Pop(h).(spmBigram)

		if symbols[bigram.left].n == 0 || symbols[bigram.right].n == 0 {
			continue
		}
		if symbols[bigram.left].n+symbols[bigram.right].n != bigram.size {
			continue
		}

		symbols[bigram.left].text = symbols[bigram.left].text + symbols[bigram.right].text
		symbols[bigram.left].n = len(symbols[bigram.left].text)

		symbols[bigram.right].n = 0
		symbols[bigram.left].next = symbols[bigram.right].next
		if symbols[bigram.right].next != -1 {
			symbols[symbols[bigram.right].next].prev = bigram.left
		}

		if symbols[bigram.left].prev != -1 && symbols[symbols[bigram.left].prev].n > 0 {
			combined := symbols[symbols[bigram.left].prev].text + symbols[bigram.left].text
			if entry, ok := t.vocabMap[combined]; ok {
				heap.Push(h, spmBigram{
					left:  symbols[bigram.left].prev,
					right: bigram.left,
					score: entry.score,
					size:  len(combined),
				})
			}
		}

		if symbols[bigram.left].next != -1 && symbols[symbols[bigram.left].next].n > 0 {
			combined := symbols[bigram.left].text + symbols[symbols[bigram.left].next].text
			if entry, ok := t.vocabMap[combined]; ok {
				heap.Push(h, spmBigram{
					left:  bigram.left,
					right: symbols[bigram.left].next,
					score: entry.score,
					size:  len(combined),
				})
			}
		}
	}

	unkID := int32(0)
	if v, ok := t.TokenToID["<unk>"]; ok {
		unkID = v
	}

	var out []int32
	for i := 0; i < len(symbols); {
		if symbols[i].n == 0 {
			i++
			continue
		}

		if entry, ok := t.vocabMap[symbols[i].text]; ok {
			out = append(out, entry.id)
		} else {
			for _, b := range []byte(symbols[i].text) {
				byteName := fmt.Sprintf("<0x%02X>", b)
				if entry, ok := t.vocabMap[byteName]; ok {
					out = append(out, entry.id)
				} else {
					out = append(out, unkID)
				}
			}
		}

		if symbols[i].next != -1 {
			i = symbols[i].next
		} else {
			break
		}
	}

	return out
}

func (t *Tokenizer) encodeFallback(text string, result []int32) []int32 {
	type vocabEntry struct {
		token string
		id    int32
	}
	var vocab []vocabEntry
	for tok, id := range t.TokenToID {
		if len(tok) > 0 && tok[0] != '<' {
			vocab = append(vocab, vocabEntry{tok, id})
		}
	}
	sort.Slice(vocab, func(i, j int) bool {
		return len(vocab[i].token) > len(vocab[j].token)
	})

	remaining := text
	for len(remaining) > 0 {
		matched := false
		for _, v := range vocab {
			if strings.HasPrefix(remaining, v.token) {
				result = append(result, v.id)
				remaining = remaining[len(v.token):]
				matched = true
				break
			}
		}
		if !matched {
			b := remaining[0]
			byteToken := fmt.Sprintf("<0x%02X>", b)
			if id, ok := t.TokenToID[byteToken]; ok {
				result = append(result, id)
			}
			remaining = remaining[1:]
		}
	}
	return result
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(tokens []int32) string {
	if t.ModelType == "gpt2" && t.unicodeToByte != nil {
		return t.decodeBPE(tokens)
	}
	return t.decodeSPM(tokens)
}

func (t *Tokenizer) decodeBPE(tokens []int32) string {
	var raw strings.Builder
	for _, token := range tokens {
		if token < 0 || int(token) >= len(t.Tokens) {
			continue
		}
		s := t.Tokens[token]
		if len(s) > 0 && s[0] == '<' && s[len(s)-1] == '>' {
			continue
		}
		raw.WriteString(s)
	}

	var result []byte
	for _, r := range raw.String() {
		if b, ok := t.unicodeToByte[r]; ok {
			result = append(result, b)
		} else {
			buf := make([]byte, 4)
			n := utf8.EncodeRune(buf, r)
			result = append(result, buf[:n]...)
		}
	}
	return string(result)
}

func (t *Tokenizer) decodeSPM(tokens []int32) string {
	var result strings.Builder
	for _, token := range tokens {
		if token < 0 || int(token) >= len(t.Tokens) {
			continue
		}
		text := t.Tokens[token]
		text = strings.ReplaceAll(text, "▁", " ")
		if strings.HasPrefix(text, "<0x") && strings.HasSuffix(text, ">") && len(text) == 6 {
			if b, err := strconv.ParseUint(text[3:5], 16, 8); err == nil {
				result.WriteByte(byte(b))
				continue
			}
		}
		result.WriteString(text)
	}
	return result.String()
}

// DecodeToken converts a single token ID to its string representation.
func (t *Tokenizer) DecodeToken(id int32) string {
	if id < 0 || int(id) >= len(t.Tokens) {
		return fmt.Sprintf("<unk_%d>", id)
	}
	s := t.Tokens[id]
	if len(s) > 0 && s[0] == '<' && s[len(s)-1] == '>' {
		return ""
	}
	if t.ModelType == "gpt2" && t.unicodeToByte != nil {
		var result []byte
		for _, r := range s {
			if b, ok := t.unicodeToByte[r]; ok {
				result = append(result, b)
			} else {
				buf := make([]byte, 4)
				n := utf8.EncodeRune(buf, r)
				result = append(result, buf[:n]...)
			}
		}
		return string(result)
	}
	if t.ModelType == "llama" {
		s = strings.ReplaceAll(s, "▁", " ")
		if strings.HasPrefix(s, "<0x") && strings.HasSuffix(s, ">") && len(s) == 6 {
			if b, err := strconv.ParseUint(s[3:5], 16, 8); err == nil {
				return string([]byte{byte(b)})
			}
		}
	}
	return s
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int {
	return len(t.Tokens)
}
