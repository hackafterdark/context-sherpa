//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	models := []struct {
		name, path string
	}{
		{"SmolLM2 360M Q8_0", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`},
		{"TinyLlama 1.1B Q4_0", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`},
		{"Qwen 2.5 0.5B Q4_K_M", `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`},
		{"Gemma 3 1B Q4_K_M", `C:\projects\evoke\models\gemma-3-1b-it-Q4_K_M.gguf`},
	}
	for _, m := range models {
		pipe, err := llm.NewPipeline(m.path, 512)
		if err != nil {
			fmt.Printf("%s: %v\n", m.name, err)
			continue
		}
		c := pipe.Model.Config
		fmt.Printf("%s: dim=%d heads=%d kv_heads=%d head_dim=%d ffn=%d layers=%d vocab=%d\n",
			m.name, c.EmbeddingDim, c.NumHeads, c.NumKVHeads, c.HeadDim, c.FFNDim, c.NumLayers, c.VocabSize)
		fmt.Printf("  RopeFreqBase=%.0f RopeNeox=%v RopeDim=%d EmbedScale=%.3f RMSNormEps=%e\n",
			c.RopeFreqBase, c.RopeNeox, c.RopeDim, c.EmbedScale, c.RMSNormEps)
		spec := pipe.Model.Layers[0].Spec
		fmt.Printf("  Spec: norm=%d core=%d ffn=%d residual=%d QKNorm=%v GatedQ=%v\n",
			spec.Norm, spec.Core, spec.FFN, spec.Residual, spec.QKNorm, spec.GatedQ)
		fmt.Println()
	}
	os.Exit(0)
}
