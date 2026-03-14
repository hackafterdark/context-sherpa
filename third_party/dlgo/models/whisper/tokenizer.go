package whisper

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// Tokenizer wraps a byte-level BPE tokenizer loaded from a HuggingFace tokenizer.json.
type Tokenizer struct {
	vocab      map[string]int
	vocabByID  map[int]string
	mergeRanks map[string]int

	addedTokens     map[string]int
	addedTokensByID map[int]string

	language *int
	task     *int
}

type tokenizerJSON struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

// LoadTokenizer loads a Whisper BPE tokenizer from a tokenizer.json file or
// a directory containing one. Initializes for English transcription by default.
func LoadTokenizer(path string) (*Tokenizer, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat tokenizer path: %w", err)
	}
	if info.IsDir() {
		path = filepath.Join(path, "tokenizer.json")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer: %w", err)
	}

	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("parse tokenizer JSON: %w", err)
	}

	t := &Tokenizer{
		vocab:           make(map[string]int, len(tj.Model.Vocab)),
		vocabByID:       make(map[int]string, len(tj.Model.Vocab)),
		mergeRanks:      make(map[string]int, len(tj.Model.Merges)),
		addedTokens:     make(map[string]int),
		addedTokensByID: make(map[int]string),
	}

	for token, id := range tj.Model.Vocab {
		t.vocab[token] = id
		t.vocabByID[id] = token
	}

	for i, merge := range tj.Model.Merges {
		t.mergeRanks[merge] = i
	}

	for _, at := range tj.AddedTokens {
		t.addedTokens[at.Content] = at.ID
		t.addedTokensByID[at.ID] = at.Content
		t.vocab[at.Content] = at.ID
		t.vocabByID[at.ID] = at.Content
	}

	// Default: English transcription
	if id, ok := t.addedTokens["<|en|>"]; ok {
		t.language = &id
	}
	if id, ok := t.addedTokens["<|transcribe|>"]; ok {
		t.task = &id
	}

	return t, nil
}

func (t *Tokenizer) tokenToID(s string) int {
	if id, ok := t.addedTokens[s]; ok {
		return id
	}
	if id, ok := t.vocab[s]; ok {
		return id
	}
	return -1
}

// Eot returns the end-of-text token ID.
func (t *Tokenizer) Eot() int { return t.tokenToID("<|endoftext|>") }

// Sot returns the start-of-transcript token ID.
func (t *Tokenizer) Sot() int { return t.tokenToID("<|startoftranscript|>") }

// NoTimestamps returns the no-timestamps token ID.
func (t *Tokenizer) NoTimestamps() int { return t.tokenToID("<|notimestamps|>") }

// SotSequence returns the prompt token sequence for decoding.
func (t *Tokenizer) SotSequence() []int {
	seq := []int{t.Sot()}
	if t.language != nil {
		seq = append(seq, *t.language)
	}
	if t.task != nil {
		seq = append(seq, *t.task)
	}
	return seq
}

// Decode converts token IDs to text, filtering out special tokens.
func (t *Tokenizer) Decode(tokens []int) string {
	eot := t.Eot()
	filtered := make([]int, 0, len(tokens))
	for _, id := range tokens {
		if id < eot {
			filtered = append(filtered, id)
		}
	}
	return t.decodeTokens(filtered)
}

func (t *Tokenizer) decodeTokens(tokens []int) string {
	var pieces []string
	for _, id := range tokens {
		if tok, ok := t.vocabByID[id]; ok {
			pieces = append(pieces, tok)
		}
	}
	return unicodeToBytes(strings.Join(pieces, ""))
}

// Encode tokenizes text into token IDs using byte-level BPE.
func (t *Tokenizer) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}
	if id, ok := t.addedTokens[text]; ok {
		return []int{id}
	}

	segments := t.splitByAddedTokens(text)
	var result []int
	for _, seg := range segments {
		if id, ok := t.addedTokens[seg]; ok {
			result = append(result, id)
		} else {
			result = append(result, t.bpeEncode(seg)...)
		}
	}
	return result
}

func (t *Tokenizer) splitByAddedTokens(text string) []string {
	if len(t.addedTokens) == 0 {
		return []string{text}
	}

	tokens := make([]string, 0, len(t.addedTokens))
	for tok := range t.addedTokens {
		tokens = append(tokens, tok)
	}
	sort.Slice(tokens, func(i, j int) bool { return len(tokens[i]) > len(tokens[j]) })

	var result []string
	remaining := text
	for len(remaining) > 0 {
		found := false
		for _, tok := range tokens {
			if strings.HasPrefix(remaining, tok) {
				result = append(result, tok)
				remaining = remaining[len(tok):]
				found = true
				break
			}
		}
		if !found {
			nextPos := len(remaining)
			for _, tok := range tokens {
				if idx := strings.Index(remaining, tok); idx >= 0 && idx < nextPos {
					nextPos = idx
				}
			}
			result = append(result, remaining[:nextPos])
			remaining = remaining[nextPos:]
		}
	}
	return result
}

func (t *Tokenizer) bpeEncode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	bytes := []byte(text)
	symbols := make([]string, len(bytes))
	for i, b := range bytes {
		symbols[i] = byteToUnicodeMap[b]
	}

	for len(symbols) >= 2 {
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i] + " " + symbols[i+1]
			if rank, ok := t.mergeRanks[pair]; ok {
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

	result := make([]int, 0, len(symbols))
	for _, sym := range symbols {
		if id, ok := t.vocab[sym]; ok {
			result = append(result, id)
		}
	}
	return result
}

// GPT-2 byte-level BPE unicode mapping

var byteToUnicodeMap [256]string
var unicodeToByteMap map[rune]byte

func init() {
	unicodeToByteMap = make(map[rune]byte, 256)

	bs := make([]int, 0, 256)
	cs := make([]int, 0, 256)

	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
		cs = append(cs, i)
	}
	for i := int('¡'); i <= int('¬'); i++ {
		bs = append(bs, i)
		cs = append(cs, i)
	}
	for i := int('®'); i <= int('ÿ'); i++ {
		bs = append(bs, i)
		cs = append(cs, i)
	}

	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, existing := range bs {
			if existing == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}

	for i := 0; i < len(bs); i++ {
		byteToUnicodeMap[byte(bs[i])] = string(rune(cs[i]))
		unicodeToByteMap[rune(cs[i])] = byte(bs[i])
	}
}

func unicodeToBytes(s string) string {
	var result []byte
	for _, r := range s {
		if b, ok := unicodeToByteMap[r]; ok {
			result = append(result, b)
		} else {
			buf := make([]byte, utf8.RuneLen(r))
			utf8.EncodeRune(buf, r)
			result = append(result, buf...)
		}
	}
	return string(result)
}
