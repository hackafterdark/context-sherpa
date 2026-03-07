package llm

import (
	"fmt"
	"math"
	"strings"
)

// ModelConfig holds all architecture parameters for a decoder-only transformer LLM.
// Auto-populated from GGUF metadata.
type ModelConfig struct {
	Architecture  string
	VocabSize     int
	ContextLength int
	EmbeddingDim  int
	NumLayers     int
	FFNDim        int
	NumHeads      int
	NumKVHeads    int
	HeadDim       int
	RMSNormEps    float32
	RopeFreqBase  float32
	RopeNeox      bool
	RopeDim       int      // partial RoPE: 0 = full headDim, else only first RopeDim dims
	BOS           int32
	EOS           int32
	StopTokens    []int32
	AddBOS        bool
	FFNGelu       bool     // true = GeGLU (Gemma), false = SwiGLU (LLaMA/Qwen)
	EmbedScale    float32  // non-zero = scale embeddings (Gemma: sqrt(dim))
	ChatTemplate  string   // chat format: "chatml", "llama2", "llama3", "gemma", "phi"

	// Qwen3.5 hybrid Mamba/Attention
	FullAttentionInterval int // 0 = all attention; N = every Nth layer is attention
	SSMConvKernel         int
	SSMInnerSize          int
	SSMStateSize          int
	SSMTimeStepRank       int
	SSMGroupCount         int
}

// parseConfig extracts a ModelConfig from GGUF metadata.
func parseConfig(md map[string]interface{}) (ModelConfig, error) {
	arch := metaString(md, "general.architecture")
	if arch == "" {
		return ModelConfig{}, fmt.Errorf("missing general.architecture")
	}

	// Vocab size: prefer metadata, fall back to token list length
	vocabSize := metaInt(md, arch+".vocab_size", 0)
	if vocabSize == 0 {
		if tokArr, ok := md["tokenizer.ggml.tokens"].([]interface{}); ok {
			vocabSize = len(tokArr)
		}
	}
	if vocabSize == 0 {
		vocabSize = 32000
	}

	// Norm epsilon: try RMSNorm key first, then LayerNorm key (Phi-2)
	normEps := metaFloat(md, arch+".attention.layer_norm_rms_epsilon", 0)
	if normEps == 0 {
		normEps = metaFloat(md, arch+".attention.layer_norm_epsilon", 1e-5)
	}

	c := ModelConfig{
		Architecture:  arch,
		VocabSize:     vocabSize,
		ContextLength: metaInt(md, arch+".context_length", 2048),
		EmbeddingDim:  metaInt(md, arch+".embedding_length", 0),
		NumLayers:     metaInt(md, arch+".block_count", 0),
		FFNDim:        metaInt(md, arch+".feed_forward_length", 0),
		NumHeads:      metaInt(md, arch+".attention.head_count", 0),
		NumKVHeads:    metaInt(md, arch+".attention.head_count_kv", 0),
		HeadDim:       metaInt(md, arch+".attention.key_length", 0),
		RMSNormEps:    normEps,
		RopeFreqBase:  metaFloat(md, arch+".rope.freq_base", 10000.0),
		BOS:           int32(metaInt(md, "tokenizer.ggml.bos_token_id", 1)),
		EOS:           int32(metaInt(md, "tokenizer.ggml.eos_token_id", 2)),
		AddBOS:        metaBool(md, "tokenizer.ggml.add_bos_token", true),

		FullAttentionInterval: metaInt(md, arch+".full_attention_interval", 0),
		SSMConvKernel:         metaInt(md, arch+".ssm.conv_kernel", 4),
		SSMInnerSize:          metaInt(md, arch+".ssm.inner_size", 0),
		SSMStateSize:          metaInt(md, arch+".ssm.state_size", 0),
		SSMTimeStepRank:       metaInt(md, arch+".ssm.time_step_rank", 0),
		SSMGroupCount:         metaInt(md, arch+".ssm.group_count", 0),
		ChatTemplate:          inferChatTemplate(md, arch),
	}

	if c.EmbeddingDim == 0 {
		return c, fmt.Errorf("missing %s.embedding_length", arch)
	}
	if c.NumLayers == 0 {
		return c, fmt.Errorf("missing %s.block_count", arch)
	}
	if c.NumHeads == 0 {
		return c, fmt.Errorf("missing %s.attention.head_count", arch)
	}
	if c.NumKVHeads == 0 {
		c.NumKVHeads = c.NumHeads
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.EmbeddingDim / c.NumHeads
	}
	c.RopeDim = metaInt(md, arch+".rope.dimension_count", 0)
	if c.RopeDim == 0 || c.RopeDim > c.HeadDim {
		c.RopeDim = c.HeadDim
	}

	applyArchDefaults(&c)

	// Parse stop tokens from GGUF metadata
	if eosArr, ok := md["tokenizer.ggml.eos_token_id"].([]interface{}); ok {
		for _, v := range eosArr {
			if id, ok := toInt(v); ok {
				c.StopTokens = append(c.StopTokens, int32(id))
			}
		}
	}

	return c, nil
}

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// metaString extracts a string from GGUF metadata.
func metaString(md map[string]interface{}, key string) string {
	if v, ok := md[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// metaInt extracts an integer from GGUF metadata (handles uint32, int32, uint64).
func metaInt(md map[string]interface{}, key string, def int) int {
	v, ok := md[key]
	if !ok {
		return def
	}
	if i, ok := toInt(v); ok {
		return i
	}
	return def
}

func toInt(v interface{}) (int, bool) {
	switch x := v.(type) {
	case uint32:
		return int(x), true
	case int32:
		return int(x), true
	case uint64:
		return int(x), true
	case int64:
		return int(x), true
	case uint8:
		return int(x), true
	case int8:
		return int(x), true
	case uint16:
		return int(x), true
	case int16:
		return int(x), true
	default:
		return 0, false
	}
}

func metaFloat(md map[string]interface{}, key string, def float32) float32 {
	v, ok := md[key]
	if !ok {
		return def
	}
	switch x := v.(type) {
	case float32:
		return x
	case float64:
		return float32(x)
	default:
		return def
	}
}

func metaBool(md map[string]interface{}, key string, def bool) bool {
	v, ok := md[key]
	if !ok {
		return def
	}
	if b, ok := v.(bool); ok {
		return b
	}
	return def
}

func inferChatTemplate(md map[string]interface{}, arch string) string {
	chatTemplate := metaString(md, "tokenizer.chat_template")
	if chatTemplate != "" {
		lower := strings.ToLower(chatTemplate)
		switch {
		case strings.Contains(lower, "<|start_header_id|>") || strings.Contains(lower, "<|eot_id|>"):
			return "llama3"
		case strings.Contains(lower, "<|im_start|>"):
			return "chatml"
		case strings.Contains(lower, "<start_of_turn>"):
			return "gemma"
		case strings.Contains(lower, "<|assistant|>"):
			return "llama2"
		}
	}

	// Llama-family models can be either Llama-2/3 chat templates.
	if arch == "llama" && hasTokenizerToken(md, "<|start_header_id|>") {
		return "llama3"
	}

	if strings.HasPrefix(arch, "gemma") {
		name := strings.ToLower(metaString(md, "general.name"))
		switch {
		case strings.Contains(name, " instruct"), strings.Contains(name, "-instruct"), strings.Contains(name, "_instruct"):
			return "gemma"
		case strings.Contains(name, " it"), strings.Contains(name, "-it"), strings.Contains(name, "_it"):
			return "gemma"
		default:
			return "plain"
		}
	}

	return ""
}

func hasTokenizerToken(md map[string]interface{}, token string) bool {
	raw, ok := md["tokenizer.ggml.tokens"]
	if !ok {
		return false
	}
	tokens, ok := raw.([]interface{})
	if !ok {
		return false
	}
	for _, t := range tokens {
		if s, ok := t.(string); ok && s == token {
			return true
		}
	}
	return false
}
