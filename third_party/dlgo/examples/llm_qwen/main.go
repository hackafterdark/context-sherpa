package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	fmt.Println("Loading Qwen 2.5 0.5B...")
	p, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 128
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0.7

	fmt.Println("\n--- Chat with Qwen ---")
	fmt.Println("Q: What is the capital of France?")
	fmt.Print("A: ")

	cfg.Stream = func(token string) { fmt.Print(token) }

	text, tokPerSec, err := p.Chat("You are a helpful assistant.", "What is the capital of France?", cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}

	_ = text
	fmt.Printf("\n\n[%.1f tokens/sec]\n", tokPerSec)
}
