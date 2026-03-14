//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func typeName(t uint32) string {
	names := map[uint32]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
		8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
		13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
	}
	if n, ok := names[t]; ok {
		return n
	}
	return fmt.Sprintf("?%d", t)
}

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
		fmt.Printf("\n═══ %s ═══\n", m.name)
		pipe, err := llm.NewPipeline(m.path, 512)
		if err != nil {
			fmt.Printf("  SKIP: %v\n", err)
			continue
		}
		mdl := pipe.Model

		types := map[uint32]int{}
		printTensor := func(name string, t interface{ GetType() uint32 }) {
			if t == nil {
				return
			}
			tp := t.GetType()
			types[tp]++
		}
		_ = printTensor

		// Just check the types directly
		if mdl.TokenEmbed != nil {
			fmt.Printf("  token_embed: type=%s\n", typeName(mdl.TokenEmbed.Type))
			types[mdl.TokenEmbed.Type]++
		}
		if mdl.Output != nil {
			fmt.Printf("  output: type=%s\n", typeName(mdl.Output.Type))
			types[mdl.Output.Type]++
		}

		for l := 0; l < len(mdl.Layers); l++ {
			cl := &mdl.Layers[l]
			if cl.Wq != nil {
				types[cl.Wq.Type]++
			}
			if cl.Wk != nil {
				types[cl.Wk.Type]++
			}
			if cl.Wv != nil {
				types[cl.Wv.Type]++
			}
			if cl.Wo != nil {
				types[cl.Wo.Type]++
			}
			if cl.FFNGate != nil {
				types[cl.FFNGate.Type]++
			}
			if cl.FFNUp != nil {
				types[cl.FFNUp.Type]++
			}
			if cl.FFNDown != nil {
				types[cl.FFNDown.Type]++
			}
			if l == 0 {
				fmt.Printf("  layer0 wq=%s wk=%s wv=%s wo=%s ffn_up=%s ffn_down=%s",
					typeName(cl.Wq.Type), typeName(cl.Wk.Type), typeName(cl.Wv.Type),
					typeName(cl.Wo.Type), typeName(cl.FFNUp.Type), typeName(cl.FFNDown.Type))
				if cl.FFNGate != nil {
					fmt.Printf(" ffn_gate=%s", typeName(cl.FFNGate.Type))
				}
				fmt.Println()
			}
		}

		fmt.Printf("  Types used: ")
		for tp, count := range types {
			fmt.Printf("%s(%d) ", typeName(tp), count)
		}
		fmt.Println()
	}
	os.Exit(0)
}
