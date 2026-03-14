package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := `C:\projects\evoke\models\gemma-3-270m-it-Q8_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	fmt.Println("Loading Gemma 3 270M...")
	p, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 128
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0.7

	fmt.Println("\n--- Chat with Gemma ---")
	fmt.Println("Q: Write a haiku about programming.")
	fmt.Print("A: ")

	cfg.Stream = func(token string) { fmt.Print(token) }

	text, tokPerSec, err := p.Chat("", "Write a haiku about programming.", cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}

	_ = text
	fmt.Printf("\n\n[%.1f tokens/sec]\n", tokPerSec)
}
