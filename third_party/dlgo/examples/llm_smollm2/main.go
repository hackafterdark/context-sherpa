package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	fmt.Println("Loading SmolLM2 360M...")
	p, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 128
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0.7

	fmt.Println("\n--- Text completion with SmolLM2 ---")
	prompt := "The quick brown fox"
	fmt.Printf("Prompt: %s\n", prompt)
	fmt.Print("Output: ")

	cfg.Stream = func(token string) { fmt.Print(token) }

	text, tokPerSec, err := p.GenerateText(prompt, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}

	_ = text
	fmt.Printf("\n\n[%.1f tokens/sec]\n", tokPerSec)
}
