package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"

	"github.com/computerex/dlgo/models/llm"
)

var defaultModels = []struct {
	name string
	path string
}{
	{"TinyLlama 1.1B Q4_0", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`},
	{"Qwen 2.5 0.5B Q4_K_M", `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`},
	{"Gemma 3 270M Q8_0", `C:\projects\evoke\models\gemma-3-270m-it-Q8_0.gguf`},
	{"SmolLM2 360M Q8_0", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`},
}

func main() {
	if os.Getenv("DLGO_PROFILE") != "" {
		f, _ := os.Create("cpu.prof")
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	models := defaultModels
	if len(os.Args) > 1 {
		models = nil
		for _, arg := range os.Args[1:] {
			name := strings.TrimSuffix(filepath.Base(arg), filepath.Ext(arg))
			models = append(models, struct {
				name string
				path string
			}{name, arg})
		}
	}

	userMsg := "Write a short poem about the ocean."
	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 64
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               dlgo LLM Inference Benchmark                    ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Printf("\nUser: %q\nMax tokens: %d (greedy)\n\n", userMsg, cfg.MaxTokens)

	for _, m := range models {
		if _, err := os.Stat(m.path); err != nil {
			fmt.Printf("%-30s  SKIP (not found)\n", m.name)
			continue
		}

		fmt.Printf("%-30s  loading...", m.name)
		p, err := llm.NewPipeline(m.path, 512)
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			continue
		}

		prompt := llm.FormatChat(p.Model.Config, "You are a helpful assistant.", userMsg)
		result, err := p.GenerateDetailed(prompt, cfg)
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			continue
		}

		preview := result.Text
		if len(preview) > 60 {
			preview = preview[:60] + "..."
		}
		preview = strings.ReplaceAll(preview, "\n", " ")

		fmt.Printf("\r%-30s  %5.1f tok/s  prefill:%5.0fms  gen:%5.0fms  [%d tok]\n",
			m.name, result.TokensPerSec, result.PrefillTimeMs, result.GenerateTimeMs, result.TotalTokens)
		fmt.Printf("  → %s\n\n", preview)
	}
}
